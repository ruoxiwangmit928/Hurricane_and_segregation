"""Harvey displacement-onset window and receiving-area sensitivity.

Reconstruct destinations from complete stored displacement spells and nighttime
stays, then shift each onset-window boundary by up to three days. This starts
from the stored daily-distance spell extraction; it does not detect new spells
from GPS pings. Original minimum duration/nighttime support rules are retained.
Missing required daily files raise an error rather than silently altering the
sample. Intermediate spell/device files remain private in the output directory.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(periods, homes, stays, cbg_shapes, output, rebuild_cache=False):
    import geopandas as gpd
    PERIOD_FILE=Path(periods)
    HOME_FILE=Path(homes)
    STAY_DIR=Path(stays)
    BG_SHP=Path(cbg_shapes)
    OUT_DIR=Path(output)
    TABLE_DIR=OUT_DIR/"tables"
    CACHE_DIR=OUT_DIR/"intermediate"
    for folder in (OUT_DIR,TABLE_DIR,CACHE_DIR):
        folder.mkdir(parents=True,exist_ok=True)
    SPELL_DEST_CACHE=CACHE_DIR/"spell_destination_support.csv.gz"
    MAIN_START=pd.Timestamp("2017-08-24")
    MAIN_END=pd.Timestamp("2017-09-02")
    START_SHIFTS=list(range(-3,4))
    END_SHIFTS=list(range(-3,4))
    MIN_DISPLACEMENT_DAYS=4
    MIN_DESTINATION_N=5
    MIN_DESTINATION_DATES=3
    NIGHT_END_HOUR=7
    NIGHT_START_HOUR=20
    # The old destination reconstruction read stay points only through 28 September.
    OBSERVATION_END=pd.Timestamp("2017-09-28")
    CHUNK_SIZE=500_000
    REBUILD_DESTINATION_CACHE=rebuild_cache

    required={"complete displacement periods":PERIOD_FILE,"pre-event home":HOME_FILE,
              "daily stay points":STAY_DIR,"Houston CBG shapefile":BG_SHP}
    audit=pd.DataFrame([{"input":k,"path":str(v),"exists":v.exists()} for k,v in required.items()])
    print(audit)
    if not audit.exists.all():raise FileNotFoundError("Missing inputs:\n"+"\n".join(audit.loc[~audit.exists,"path"]))


    def norm_id(s):return s.astype("string").str.strip()
    def norm_cbg(s):return s.astype("string").str.replace(r"\.0$","",regex=True).str.zfill(12)

    periods=pd.read_csv(PERIOD_FILE,dtype={"device_id":"string"},low_memory=False)
    periods["device_id"]=norm_id(periods.device_id)
    for c in ["start_date","end_date"]:periods[c]=pd.to_datetime(periods[c],errors="coerce").dt.normalize()
    periods["days_count"]=pd.to_numeric(periods.days_count,errors="coerce")
    periods=periods.dropna(subset=["device_id","start_date","end_date","days_count"]).copy()

    homes=pd.read_csv(HOME_FILE,usecols=["device_id","visitor_home_cbgs"],dtype="string")
    homes["device_id"]=norm_id(homes.device_id);homes["home_cbg"]=norm_cbg(homes.visitor_home_cbgs)
    valid_home_ids=set(homes.dropna(subset=["home_cbg"]).device_id)

    period_base=periods[periods.days_count.ge(MIN_DISPLACEMENT_DAYS)&periods.device_id.isin(valid_home_ids)].copy()
    period_base["spell_id"]=(period_base.device_id+"|"+period_base.start_date.dt.strftime("%Y-%m-%d")+"|"+
                             period_base.end_date.dt.strftime("%Y-%m-%d"))

    spec_rows=[]
    for shift in START_SHIFTS:
     spec_rows.append({"axis":"start","shift_days":shift,"window_name":"main" if shift==0 else f"start_{shift:+d}d",
                       "window_start":MAIN_START+pd.Timedelta(days=shift),"window_end":MAIN_END})
    for shift in END_SHIFTS:
     if shift==0:continue
     spec_rows.append({"axis":"end","shift_days":shift,"window_name":f"end_{shift:+d}d",
                       "window_start":MAIN_START,"window_end":MAIN_END+pd.Timedelta(days=shift)})
    specs=pd.DataFrame(spec_rows)

    parts=[]
    for s in specs.itertuples(index=False):
     q=period_base[period_base.start_date.between(s.window_start,s.window_end)].copy()
     q=(q.sort_values(["device_id","start_date","days_count"],ascending=[True,True,False]).drop_duplicates("device_id"))
     q["axis"]=s.axis;q["shift_days"]=s.shift_days;q["window_name"]=s.window_name
     q["window_start"]=s.window_start;q["window_end"]=s.window_end
     parts.append(q)
    candidate_spells=pd.concat(parts,ignore_index=True)
    candidate_spells.to_csv(TABLE_DIR/"candidate_spells_before_destination_validation.csv.gz",index=False,compression="gzip")
    print(specs)
    print(pd.DataFrame({"quantity":["stored spells","duration/home-eligible spells","union selected spell IDs"],
                          "n":[len(periods),len(period_base),candidate_spells.spell_id.nunique()]}))


    union_spells=(candidate_spells[["spell_id","device_id","start_date","end_date"]]
                  .drop_duplicates("spell_id").copy())
    union_spells["observed_end"]=union_spells.end_date.clip(upper=OBSERVATION_END)
    truncated=union_spells.end_date.gt(OBSERVATION_END)
    print("Spells extending beyond available old destination observation end:",int(truncated.sum()))

    if REBUILD_DESTINATION_CACHE or not SPELL_DEST_CACHE.exists():
     bg=gpd.read_file(BG_SHP)
     geoid=next((c for c in ["GEOID_2","GEOID","geoid"] if c in bg.columns),None)
     if geoid is None:raise ValueError(f"No GEOID field in shapefile: {bg.columns.tolist()}")
     bg=bg[[geoid,"geometry"]].rename(columns={geoid:"destination_cbg"}).to_crs("EPSG:4326")
     bg["destination_cbg"]=norm_cbg(bg.destination_cbg)
     support=[]
     scan_start=union_spells.start_date.min();scan_end=union_spells.observed_end.max()
     for date in pd.date_range(scan_start,scan_end):
      active=union_spells[union_spells.start_date.le(date)&union_spells.observed_end.ge(date)][["spell_id","device_id"]]
      if active.empty:continue
      active_ids=set(active.device_id)
      path=STAY_DIR/f"{date.date()}.txt"
      if not path.exists():
       raise FileNotFoundError(f"Missing required daily stay file: {path}")
      names=["device_id","start_time","date","lat","lon","n","end_time","stay_time"]
      for ch in pd.read_csv(path,names=names,dtype={"device_id":"string"},chunksize=CHUNK_SIZE,low_memory=False):
       z=ch[ch.device_id.isin(active_ids)].copy()
       if z.empty:continue
       z["start_time"]=pd.to_datetime(z.start_time,errors="coerce")
       z["hour"]=z.start_time.dt.hour;z["n"]=pd.to_numeric(z.n,errors="coerce")
       z["lat"]=pd.to_numeric(z.lat,errors="coerce");z["lon"]=pd.to_numeric(z.lon,errors="coerce")
       z=z[(z.hour.le(NIGHT_END_HOUR)|z.hour.ge(NIGHT_START_HOUR))].dropna(subset=["lat","lon","n"])
       if z.empty:continue
       z=z.merge(active,on="device_id",how="inner")
       pts=gpd.GeoDataFrame(z,geometry=gpd.points_from_xy(z.lon,z.lat),crs="EPSG:4326")
       joined=gpd.sjoin(pts,bg,predicate="within",how="inner")
       if len(joined):support.append(joined[["spell_id","device_id","destination_cbg","n"]].assign(obs_date=date))
     if not support:raise RuntimeError("No nighttime destination observations were recovered")
     support=pd.concat(support,ignore_index=True)
     spell_cbg=(support.groupby(["spell_id","device_id","destination_cbg"],as_index=False)
                .agg(Sum_of_n=("n","sum"),Unique_dates=("obs_date","nunique")))
     total_dates=(support.groupby("spell_id").obs_date.nunique().rename("total_observed_night_dates"))
     spell_cbg=spell_cbg.merge(total_dates,on="spell_id")
     spell_cbg.to_csv(SPELL_DEST_CACHE,index=False,compression="gzip")
    else:
     spell_cbg=pd.read_csv(SPELL_DEST_CACHE,dtype={"spell_id":"string","device_id":"string","destination_cbg":"string"})


    print("Cached spells with at least one mapped nighttime destination:",spell_cbg.spell_id.nunique())


    x=spell_cbg.copy()
    x["date_share"]=x.Unique_dates/x.total_observed_night_dates.replace(0,np.nan)
    x["support_share"]=x.Sum_of_n/x.groupby("spell_id").Sum_of_n.transform("sum").replace(0,np.nan)
    x=x[x.date_share.ge(1/3)].copy()
    primary=(x.sort_values(["spell_id","Sum_of_n","Unique_dates","destination_cbg"],ascending=[True,False,False,True])
             .drop_duplicates("spell_id"))
    primary=primary[primary.Sum_of_n.ge(MIN_DESTINATION_N)&primary.Unique_dates.ge(MIN_DESTINATION_DATES)].copy()
    primary=primary.rename(columns={"support_share":"primary_destination_share"})
    primary.to_csv(TABLE_DIR/"rebuilt_primary_destination_by_spell.csv.gz",index=False,compression="gzip")

    validated=candidate_spells.merge(primary[["spell_id","destination_cbg","Sum_of_n","Unique_dates",
                                               "total_observed_night_dates","primary_destination_share"]],on="spell_id",how="inner")
    validated.to_csv(TABLE_DIR/"validated_window_person_classifications.csv.gz",index=False,compression="gzip")
    validation_audit=(candidate_spells.groupby("window_name").device_id.nunique().rename("before_destination")
                      .to_frame().join(validated.groupby("window_name").device_id.nunique().rename("after_destination")))
    validation_audit["retained_share"]=validation_audit.after_destination/validation_audit.before_destination
    print(validation_audit.reset_index());validation_audit.to_csv(TABLE_DIR/"destination_validation_audit.csv")


    main_ids=set(validated.loc[validated.window_name.eq("main"),"device_id"])
    rows=[]
    for s in specs.itertuples(index=False):
     q=validated[validated.window_name.eq(s.window_name)]
     ids=set(q.device_id);inter=len(ids&main_ids);union=len(ids|main_ids)
     rows.append({"axis":s.axis,"shift_days":s.shift_days,"window_name":s.window_name,
                  "window_start":s.window_start,"window_end":s.window_end,"n_evacuees":len(ids),
                  "added_vs_main":len(ids-main_ids),"removed_vs_main":len(main_ids-ids),
                  "retained_main_share":inter/len(main_ids) if main_ids else np.nan,
                  "jaccard_with_main":inter/union if union else np.nan,
                  "median_duration":q.days_count.median(),"median_primary_destination_share":q.primary_destination_share.median()})
    window_summary=pd.DataFrame(rows).sort_values(["axis","shift_days"])
    print(window_summary);window_summary.to_csv(TABLE_DIR/"corrected_window_sensitivity_summary.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for axis,color in [("start","#4C78A8"),("end","#F28E2B")]:
     g=window_summary[window_summary.axis.eq(axis)]
     axes[0].plot(g.shift_days,g.n_evacuees,marker="o",color=color,label=axis)
     axes[1].plot(g.shift_days,g.jaccard_with_main,marker="o",color=color,label=axis)
    for ax in axes:
     ax.axvline(0,color="0.4",ls="--",lw=.9);ax.set_xlabel("Boundary shift from main (days)");ax.legend(frameon=False)
    axes[0].set_title("Classified evacuees");axes[0].set_ylabel("Unique destination-validated devices")
    axes[1].set_title("Cohort overlap with main");axes[1].set_ylabel("Jaccard similarity");axes[1].set_ylim(0,1.02)
    plt.tight_layout();plt.savefig(OUT_DIR / "window_classification_sensitivity.png", dpi=300, bbox_inches="tight");plt.close(fig)


    expanded=period_base[period_base.start_date.between(pd.Timestamp("2017-08-21"),pd.Timestamp("2017-09-05"))].copy()
    expanded=(expanded.sort_values(["device_id","start_date","days_count"],ascending=[True,True,False]).drop_duplicates("device_id"))
    expanded=expanded.merge(primary[["spell_id","destination_cbg","primary_destination_share"]],on="spell_id",how="inner")
    daily=(expanded.groupby("start_date").device_id.nunique().rename("n_evacuees").reset_index())
    calendar=pd.DataFrame({"start_date":pd.date_range("2017-08-21","2017-09-05")})
    daily=calendar.merge(daily,on="start_date",how="left").fillna({"n_evacuees":0})
    daily.to_csv(TABLE_DIR/"corrected_expanded_daily_departures.csv",index=False)
    fig,ax=plt.subplots(figsize=(12,4.5))
    inside=daily.start_date.between(MAIN_START,MAIN_END)
    ax.bar(daily.start_date,daily.n_evacuees,width=.82,color=np.where(inside,"#4C78A8","#AFC3D5"))
    ax.axvspan(MAIN_START-pd.Timedelta(hours=12),MAIN_END+pd.Timedelta(hours=12),color="#F3D36A",alpha=.18,label="Original window")
    for d,c,l in [("2017-08-23","#7A5195","Texas hurricane watch / Harvey re-forms"),
                  ("2017-08-24","#EF9B20","Warnings escalate"),("2017-08-25","#D62728","Texas landfall")]:
     ax.axvline(pd.Timestamp(d),color=c,ls="--",lw=1.3,label=l)
    ax.axvline(MAIN_END,color="0.2",ls=":",lw=1.1)
    ax.set_title("Daily destination-validated evacuation departures")
    ax.set_xlabel("Inferred departure date");ax.set_ylabel("Unique evacuees");ax.legend(frameon=False,ncol=2,fontsize=9)
    fig.autofmt_xdate();plt.tight_layout();plt.savefig(OUT_DIR / "daily_departures.png", dpi=300, bbox_inches="tight");plt.close(fig)


    top_sets={};dest_rows=[]
    for s in specs.itertuples(index=False):
     q=validated[validated.window_name.eq(s.window_name)]
     cnt=q.groupby("destination_cbg").device_id.nunique().sort_values(ascending=False)
     top_sets[s.window_name]=set(cnt.head(100).index)
     dest_rows.extend({"window_name":s.window_name,"destination_cbg":cbg,"n_evacuees":n,"rank":rank+1}
                      for rank,(cbg,n) in enumerate(cnt.items()))
    main_top=top_sets["main"]
    top_summary=[]
    for s in specs.itertuples(index=False):
     a=top_sets[s.window_name];u=len(a|main_top)
     top_summary.append({"axis":s.axis,"shift_days":s.shift_days,"window_name":s.window_name,
                         "top100_intersection":len(a&main_top),"top100_jaccard":len(a&main_top)/u if u else np.nan})
    top_summary=pd.DataFrame(top_summary).sort_values(["axis","shift_days"])
    pd.DataFrame(dest_rows).to_csv(TABLE_DIR/"receiving_cbg_counts_by_window.csv",index=False)
    top_summary.to_csv(TABLE_DIR/"corrected_top100_destination_overlap.csv",index=False)
    print(top_summary)
    fig,ax=plt.subplots(figsize=(6,4))
    for axis,color in [("start","#4C78A8"),("end","#F28E2B")]:
     g=top_summary[top_summary.axis.eq(axis)];ax.plot(g.shift_days,g.top100_jaccard,marker="o",color=color,label=axis)
    ax.axvline(0,color="0.4",ls="--",lw=.9);ax.set_ylim(0,1.02)
    ax.set_xlabel("Boundary shift from main (days)");ax.set_ylabel("Top-100 CBG Jaccard with main")
    ax.set_title("Receiving-neighborhood stability");ax.legend(frameon=False);plt.tight_layout();plt.savefig(OUT_DIR / "receiving_area_stability.png", dpi=300, bbox_inches="tight");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--periods',type=Path,required=True)
    parser.add_argument('--homes',type=Path,required=True)
    parser.add_argument('--stays',type=Path,required=True)
    parser.add_argument('--cbg-shapes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rebuild-cache',action='store_true')
    args=parser.parse_args()
    run(args.periods,args.homes,args.stays,args.cbg_shapes,args.output,args.rebuild_cache)


if __name__=='__main__':
    main()
