"""Harvey income-assignment robustness on fixed observed person-week support.

Input: saved individual estimates for seven socioeconomic specifications.
Pool 20 assignment draws within each person-week; compare M=1-S levels,
matched agreement and the lower-within-CBG-variation subset. Outputs are
aggregate summaries and plots, not device records. This file does not create
new ACS/B19001 income assignments.

Temporal changes additionally use 100*(M-M0)/(1-M0), with each method's
weeks 0--5 mean as M0. This normalization follows the existing pooling of
draws within person-week and aggregation across observed hosts; it does not
change the fixed sample or the absolute-M individual agreement analysis.

To add normalized temporal outputs from an existing weekly export without
rereading the individual file:
    python harvey_income_robustness.py --weekly weekly_social_mixing.csv \
        --output normalized_results
This --weekly mode does not rerun or replace individual agreement results.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def normalized_weekly_outputs(weekly, output):
    """Add normalized trajectories to the original saved method-week means."""
    output = Path(output)
    required = {'label', 'week', 'mean_social_mixing'}
    if not required.issubset(weekly.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(weekly.columns))}')
    out = weekly.copy()
    if out[list(required)].isna().any().any() or out.duplicated(['label', 'week']).any():
        raise ValueError('Expected one nonmissing mean for each method and week')
    if not np.isfinite(out.mean_social_mixing).all() or not out.mean_social_mixing.between(0,1).all():
        raise ValueError('Mixing levels must be finite and in [0,1]')
    style={
        'median_q4':('Original: median + quartiles','#222222','o'),
        'median_q3':('Median + tertiles','#2B79B2','s'),
        'median_q5':('Median + quintiles','#EE861B','^'),
        'official_mean_q4':('ACS mean + quartiles','#EE861B','s'),
        'uniform_bin_q4':('B19001 uniform-bin (20 draws)','#27875B','^'),
        'lognormal_q4':('B19001 log-normal (20 draws)','#2B79B2','D'),
        'median_q4_lowvar50':('Below-median within-CBG income variation','#2B79B2','s'),
    }
    if set(out.label) != set(style):
        raise ValueError('Expected all seven original socioeconomic specifications')
    rows=[]
    for label, g in out.groupby('label', sort=False):
        if set(g.week) != set(range(12)):
            raise ValueError(f'Expected all 12 weeks for {label}')
        baseline=g.loc[g.week.between(0,5), 'mean_social_mixing'].mean()
        if not 0 <= baseline < 1:
            raise ValueError(f'Undefined normalized mixing change for {label}: M0={baseline}')
        out.loc[g.index, 'baseline_mixing_weeks0_5']=baseline
        out.loc[g.index, 'normalized_change_percent']=100*(g.mean_social_mixing-baseline)/(1-baseline)
        rows.append(dict(label=label,baseline_mixing_weeks0_5=baseline,
            week7_normalized_change_percent=100*(g.loc[g.week.eq(7), 'mean_social_mixing'].iloc[0]-baseline)/(1-baseline),
            weeks8_11_mean_normalized_change_percent=100*(g.loc[g.week.between(8,11), 'mean_social_mixing'].mean()-baseline)/(1-baseline)))
    np.testing.assert_allclose(out.normalized_change_percent,
        -100*((1-out.mean_social_mixing)-(1-out.baseline_mixing_weeks0_5))/(1-out.baseline_mixing_weeks0_5),atol=1e-10)
    for directory in [output/'tables', output/'figures']:
        directory.mkdir(parents=True,exist_ok=True)
    out.to_csv(output/'tables/weekly_normalized_social_mixing.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'tables/period_normalized_change_summary.csv',index=False)
    panels=[('A  SES quantile cutoffs',['median_q4','median_q3','median_q5']),
            ('B  Income measure / assignment',['median_q4','official_mean_q4','uniform_bin_q4','lognormal_q4']),
            ('C  Low within-CBG income variation',['median_q4','median_q4_lowvar50'])]
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.1),layout='constrained')
    for ax,(title,methods) in zip(axes,panels):
        for label in methods:
            g=out.loc[out.label.eq(label)].sort_values('week')
            name,color,marker=style[label]
            ax.plot(g.week,g.normalized_change_percent,label=name,color=color,marker=marker,ms=4,
                    lw=2.2 if label=='median_q4' else 1.6)
        ax.axvspan(7,8,color='#F5D36C',alpha=.2,zorder=0)
        ax.axhline(0,color='0.6',linestyle='--',linewidth=.8)
        ax.set(title=title,xlabel='Study week',ylabel='Normalized mixing change (%)',xticks=range(12))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=8.8,loc='lower center',bbox_to_anchor=(.5,1.01),borderaxespad=1.1)
        ax.set_title(title,pad=80,fontsize=11)
    for ext in ['png','pdf']:
        fig.savefig(output/f'figures/income_robustness_normalized_change.{ext}',dpi=300)
    plt.close(fig)
    audit=dict(formula='100*(M-M0)/(1-M0)',baseline_weeks=list(range(6)),
        baseline_weighting='equal weekly means separately within each method',
        temporal_input='existing weekly M after original within-person draw pooling',
        sample_and_absolute_values_changed=False,individual_agreement_normalized=False,
        normalized_rows=len(out),methods=len(style))
    (output/'temporal_normalization_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return out


def run(data, output):
    PATH=Path(data); OUT=Path(output)
    labels=['median_q4','median_q3','median_q5','official_mean_q4','uniform_bin_q4','lognormal_q4','median_q4_lowvar50']
    mi={'uniform_bin_q4','lognormal_q4'}
    expected_rows={label:326782 if label.endswith('lowvar50') else 419154 for label in labels}
    expected_people={label:45416 if label.endswith('lowvar50') else 57590 for label in labels}
    source_stat=PATH.stat()
    pending={}; pending_n={}; complete=set(); supports={}; sums={}; draw_counts={}; draw_audit=[]
    raw_rows=0; all_rows=0

    def finish_block(label,draw,frames):
        nonlocal supports,sums,draw_counts
        block=pd.concat(frames,ignore_index=True)
        index=pd.MultiIndex.from_frame(block[['device_id','week']])
        assert index.is_unique, f'Duplicate person-week in {label}, draw {draw}'
        assert len(index)==expected_rows[label]
        assert index.get_level_values('device_id').nunique()==expected_people[label]
        if label not in supports:
            supports[label]=index
            sums[label]=np.zeros(len(index),dtype=np.float64)
            draw_counts[label]=np.zeros(len(index),dtype=np.int16)
        if index.equals(supports[label]):
            values=block.individual_seg.to_numpy()
        else:
            positions=index.get_indexer(supports[label])
            assert (positions>=0).all(), f'Unequal draw support: {label}/{draw}'
            values=block.individual_seg.to_numpy()[positions]
        sums[label]+=values
        draw_counts[label]+=1
        complete.add((label,draw))
        draw_audit.append(dict(label=label,imputation=int(draw),rows=len(block),unique_hosts=expected_people[label]))
        if len(complete)%5==0:
            print(f'Validated {len(complete)} method/draw blocks; {raw_rows:,} source rows read',flush=True)

    for chunk in pd.read_csv(PATH,chunksize=400000,dtype={'device_id':'string','label':'category','place_group':'category','imputation':'int16','week':'int8'}):
        raw_rows+=len(chunk)
        assert not chunk.isna().any().any(), 'Missing source fields; do not silently discard rows.'
        assert np.isfinite(chunk.individual_seg).all() and chunk.individual_seg.between(0,1).all()
        assert chunk.week.between(0,11).all()
        chunk=chunk.loc[chunk.place_group.eq('all')]
        all_rows+=len(chunk)
        assert set(chunk.label).issubset(labels), 'Unexpected method in downloaded file.'
        for (label,draw),g in chunk.groupby(['label','imputation'],sort=False,observed=True):
            label=str(label);draw=int(draw);key=(label,draw)
            assert key not in complete, f'Extra rows for already complete block: {label}/{draw}'
            assert (1<=draw<=20) if label in mi else draw==0
            pending.setdefault(key,[]).append(g[['device_id','week','individual_seg']].copy())
            pending_n[key]=pending_n.get(key,0)+len(g)
            assert pending_n[key]<=expected_rows[label], f'Too many rows: {label}/{draw}'
            if pending_n[key]==expected_rows[label]:
                finish_block(label,draw,pending.pop(key))
                del pending_n[key]

    assert not pending, f'Incomplete method/draw blocks: {pending_n}'
    assert raw_rows==all_rows==18769558
    expected_blocks={(label,d) for label in labels for d in (range(1,21) if label in mi else [0])}
    assert complete==expected_blocks
    assert PATH.stat().st_size==source_stat.st_size and PATH.stat().st_mtime_ns==source_stat.st_mtime_ns
    assert all(np.all(draw_counts[label]==(20 if label in mi else 1)) for label in labels)

    reference=supports['median_q4']
    pooled={}
    for label in labels:
        s=sums[label]/draw_counts[label]
        if label!='median_q4_lowvar50':
            positions=supports[label].get_indexer(reference)
            assert (positions>=0).all() and len(supports[label])==len(reference)
            s=s[positions]
            supports[label]=reference
        else:
            assert (reference.get_indexer(supports[label])>=0).all(), 'Restricted sample is not a subset.'
        pooled[label]=s
        del sums[label]

    for d in [OUT,OUT/'figures',OUT/'tables']:d.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(draw_audit).sort_values(['label','imputation']).to_csv(OUT/'tables/draw_coverage.csv',index=False)
    weekly_rows=[]; count_rows=[]
    for label in labels:
        s=pooled[label];m=1-s;idx=supports[label]
        weeks=idx.get_level_values('week').to_numpy()
        for week in range(12):
            keep=weeks==week; sw=s[keep];mw=m[keep]
            assert len(sw)>0
            np.testing.assert_allclose(mw.mean(),1-sw.mean(),atol=1e-14)
            weekly_rows.append(dict(label=label,week=week,n_hosts=int(keep.sum()),
                mean_social_mixing=mw.mean(),median_social_mixing=np.median(mw),
                sd_social_mixing=mw.std(ddof=1),mean_original_segregation=sw.mean()))
        count_rows.append(dict(label=label,person_weeks=len(s),unique_hosts=idx.get_level_values('device_id').nunique(),
                               observed_weeks=len(np.unique(weeks)),draws_per_person_week=20 if label in mi else 1))
    weekly=pd.DataFrame(weekly_rows)
    weekly.to_csv(OUT/'tables/weekly_social_mixing.csv',index=False)
    pd.DataFrame(count_rows).to_csv(OUT/'tables/sample_counts.csv',index=False)

    # Independently reproduce the six-digit values saved by the original notebook.
    expected_agreement={
        'median_q3':(.906130,.882756,.068354),
        'median_q5':(.940415,.918961,.050071),
        'official_mean_q4':(.930795,.905872,.049720),
        'uniform_bin_q4':(.812272,.823183,.132735),
        'lognormal_q4':(.790355,.803410,.139130),
    }
    comparison=list(expected_agreement)
    agreement=[];complement_stat_errors=[];ref_s=pd.Series(pooled['median_q4']);ref_m=1-ref_s
    for label in comparison:
        alt_s=pd.Series(pooled[label]);alt_m=1-alt_s
        s_stats=[alt_s.corr(ref_s),alt_s.corr(ref_s,method='spearman'),(alt_s-ref_s).abs().mean()]
        m_stats=[alt_m.corr(ref_m),alt_m.corr(ref_m,method='spearman'),(alt_m-ref_m).abs().mean()]
        np.testing.assert_allclose(s_stats,expected_agreement[label],atol=5.1e-7,rtol=0)
        np.testing.assert_allclose(np.array(m_stats)[[0,2]],np.array(s_stats)[[0,2]],atol=1e-12,rtol=0)
        # Subtraction can merge machine-precision ties, affecting average ranks at
        # ~1e-10. Keep actual M values and record this, rather than rounding data.
        np.testing.assert_allclose(m_stats[1],s_stats[1],atol=1e-8,rtol=0)
        complement_stat_errors.append(dict(label=label,
            pearson_abs_difference=abs(m_stats[0]-s_stats[0]),
            spearman_abs_difference=abs(m_stats[1]-s_stats[1]),
            MAD_abs_difference=abs(m_stats[2]-s_stats[2])))
        agreement.append(dict(label=label,matched_host_weeks=len(ref_s),matched_hosts=57590,
                              pearson_r=m_stats[0],spearman_rho=m_stats[1],mean_absolute_difference=m_stats[2],
                              original_mean_mixing=ref_m.mean(),alternative_mean_mixing=alt_m.mean()))
    agreement=pd.DataFrame(agreement)
    agreement.to_csv(OUT/'tables/individual_social_mixing_agreement.csv',index=False)

    period=[]
    for label,g in weekly.groupby('label',sort=False):
        pre=g.loc[g.week.between(0,5),'mean_social_mixing'].mean()
        peak=g.loc[g.week.eq(7),'mean_social_mixing'].iloc[0]
        post=g.loc[g.week.between(8,11),'mean_social_mixing'].mean()
        assert peak<pre and post<pre
        period.append(dict(label=label,pre_weeks0_5_mean=pre,week7_mean=peak,post_weeks8_11_mean=post,
                           week7_minus_pre_index=peak-pre,post_minus_pre_index=post-pre,
                           all_followup_weeks_below_pre=bool((g.loc[g.week.between(8,11),'mean_social_mixing']<pre).all())))
    pd.DataFrame(period).to_csv(OUT/'tables/period_level_summary.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':.8,
                        'pdf.fonttype':42,'savefig.facecolor':'white'})
    style={
        'median_q4':('Original: median + quartiles','#222222','o'),
        'median_q3':('Median + tertiles','#2B79B2','s'),
        'median_q5':('Median + quintiles','#EE861B','^'),
        'official_mean_q4':('ACS mean + quartiles','#EE861B','s'),
        'uniform_bin_q4':('B19001 uniform-bin (20 draws)','#27875B','^'),
        'lognormal_q4':('B19001 log-normal (20 draws)','#2B79B2','D'),
        'median_q4_lowvar50':('Below-median within-CBG income variation','#2B79B2','s'),
    }
    panels=[('A  SES quantile cutoffs',['median_q4','median_q3','median_q5']),
            ('B  Income measure / assignment',['median_q4','official_mean_q4','uniform_bin_q4','lognormal_q4']),
            ('C  Low within-CBG income variation',['median_q4','median_q4_lowvar50'])]
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.1),layout='constrained')
    for ax,(title,methods) in zip(axes,panels):
        for label in methods:
            g=weekly.loc[weekly.label.eq(label)].sort_values('week')
            name,color,marker=style[label]
            ax.plot(g.week,g.mean_social_mixing,label=name,color=color,marker=marker,ms=4,
                    lw=2.2 if label=='median_q4' else 1.6)
        ax.axvspan(7,8,color='#F5D36C',alpha=.2,zorder=0)
        ax.set(title=title,xlabel='Study week',ylabel='Mean experienced social mixing',xticks=range(12))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=8.8,loc='lower center',bbox_to_anchor=(.5,1.01),
                  borderaxespad=1.1)
        ax.set_title(title,pad=80,fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/income_robustness_weekly_mixing.{ext}',dpi=300)
    plt.close(fig)

    # Keep the absolute plot above; add the revised temporal measure separately.
    normalized_weekly_outputs(weekly, OUT)

    titles={
        'median_q3':'Median + tertiles','median_q5':'Median + quintiles',
        'official_mean_q4':'ACS mean + quartiles',
        'uniform_bin_q4':'B19001 uniform-bin',
        'lognormal_q4':'B19001 log-normal',
    }
    # Only presentation changes below. Use the same unrounded paired values,
    # full [0, 1] ranges, grid, and common log-count normalization as before.
    fig=plt.figure(figsize=(12.6,9.2))
    panel_w=3.25/12.6
    panel_h=3.25/9.2
    top_x=[.075,.382,.689]
    bottom_x=[.2285,.5355]
    axes=[fig.add_axes([x,.555,panel_w,panel_h]) for x in top_x]
    axes += [fig.add_axes([x,.105,panel_w,panel_h]) for x in bottom_x]
    hexes=[]
    hexbin_totals={}
    for panel,ax,label in zip('ABCDE',axes,comparison):
        m=1-pooled[label];r=agreement.set_index('label').loc[label]
        hb=ax.hexbin(ref_m.to_numpy(),m,gridsize=40,mincnt=1,extent=(0,1,0,1),cmap='magma',norm=LogNorm())
        hb.set_rasterized(True);hexes.append(hb)
        assert int(hb.get_array().sum())==len(ref_m), 'The hexbin must retain every matched observation.'
        hexbin_totals[label]=int(hb.get_array().sum())
        ax.plot([0,1],[0,1],'--',color='#2582B7',lw=1.0,alpha=.9)
        ax.set(xlim=(0,1),ylim=(0,1),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6))
        ax.set_aspect('equal',adjustable='box')
        ax.tick_params(labelsize=9.5,length=3,width=.7,pad=4)
        for spine in ax.spines.values():spine.set_color('#666666')
        ax.text(-.065,1.12,panel,transform=ax.transAxes,fontsize=12,fontweight='bold',ha='left',va='bottom')
        ax.text(.5,1.12,titles[label],transform=ax.transAxes,fontsize=11,ha='center',va='bottom')
        ax.text(.5,1.035,f'$r$ = {r.pearson_r:.3f}   $ρ$ = {r.spearman_rho:.3f}',
                transform=ax.transAxes,fontsize=10,color='#444444',ha='center',va='bottom')
    max_count=max(float(h.get_array().max()) for h in hexes)
    common_norm=LogNorm(vmin=1,vmax=max_count)
    for h in hexes:h.set_norm(common_norm)
    fig.text(.515,.045,'Original experienced social mixing',ha='center',va='center',fontsize=12)
    fig.text(.022,.505,'Alternative experienced social mixing',ha='center',va='center',rotation=90,fontsize=12)
    # Small external colour bar; sample counts and draw definitions belong in the caption.
    cax=fig.add_axes([.9,.17,.012,.215])
    cb=fig.colorbar(hexes[0],cax=cax,orientation='vertical',ticks=[1,10,100,1000])
    cb.set_label('Count (log scale)',fontsize=10,labelpad=9)
    cb.ax.tick_params(labelsize=9,width=.6,length=3)
    cb.outline.set_linewidth(.6)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/income_robustness_individual_agreement.{ext}',dpi=300)
    plt.close(fig)

    audit=dict(source_file=str(PATH),source_bytes=source_stat.st_size,raw_rows=raw_rows,
               place_group='all',validated_draw_blocks=len(complete),expected_draw_blocks=len(expected_blocks),
               unique_draw_keys=True,draws_complete=True,fixed_unrestricted_support_verified=True,
               restricted_sample_is_subset=True,
               transform='M_it = 1-S_it, absolute index; average 20 draws within each host-week before weekly aggregation or matching.',
               source_method_preserved='No IQR, trimming, sample exclusion, or new income assignments. Absolute agreement is unchanged.',
               temporal_normalization='Additional output: 100*(M-M0)/(1-M0), using each method\'s weeks 0--5 mean after original draw pooling.',
               agreement_plot_source='Freshly read source CSV; paired unrounded host-week values; not traced or copied from an image.',
               agreement_plot_hexbin_totals=hexbin_totals,
               agreement_plot_layout='Three panels above and two centered below; shared labels and external common colour bar. MAD retained in numeric table and response caption.',
               weekly_complement_max_error=float(np.abs(weekly.mean_social_mixing+weekly.mean_original_segregation-1).max()),
               saved_notebook_correlations_and_MAE_reproduced=True,
               paired_complement_statistics_invariant_to_reported_precision=True,
               complement_stat_numerical_errors=complement_stat_errors,
               weekly_n_note='Weekly available records are used; fixed support across SES methods does not imply every host is observed in every week.',
               uncertainty='Draws are averaged, not treated as independent observations. No inferential confidence intervals are shown.',
               restricted_hosts_percent=100*45416/57590)
    (OUT/'analysis_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    print(agreement.to_string(index=False),flush=True)
    print(pd.DataFrame(period).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path,
                        help='Saved fixed-support individual estimates; restricted input')
    source.add_argument('--weekly',type=Path,
                        help='Existing weekly_social_mixing.csv; add temporal normalization only')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.weekly:
        normalized_weekly_outputs(pd.read_csv(args.weekly),args.output)
        print(f'Saved normalized temporal outputs to {args.output}; individual agreement not rerun')
    else:
        run(args.data,args.output)


if __name__=='__main__':
    main()
