#!/usr/bin/env python3
"""Refit the SI receiving-count models from a prepared CBG-level feature table.

The outcome is distinct qualifying arrivals (n_received), not max_value or a
social-mixing outcome. This entry point does not reconstruct restricted device
records or download inputs. It preserves the original Top-100 membership.

Requires numpy, pandas, patsy, scipy and statsmodels. The input must provide
cbg_id, county, tract_id, original_top100, n_received, and the prepared numeric
predictors named in BASE, ENVIRONMENT, SOCIOECONOMIC and SPECS below. Shares use
the 0-1 scale. log_population, log_area, log_coverage and log_income are natural
logs; log_coast, log_road and log_hospital are log(1 + original measure).

The historical complete samples contain 2,916 metropolitan CBGs (2,910 for M4)
and 97 eligible members of the original Top-100 roster. Sizes are computed from
the supplied data, not forced to these historical values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm


BASE = ["log_population", "log_area", "log_coverage"]
ENVIRONMENT = ["log_coast", "flood100_share", "flood500_only_share"]
SOCIOECONOMIC = [
    "log_income", "homeowner_share", "college_share", "white_share",
    "health_insured_share",
]
SPECS = {
    "M1": BASE + ENVIRONMENT,
    "M2": BASE + SOCIOECONOMIC,
    "M3": BASE + ENVIRONMENT + SOCIOECONOMIC,
    "M4": BASE + ENVIRONMENT + SOCIOECONOMIC + ["log_road", "log_hospital"],
    "M5": BASE + ENVIRONMENT + SOCIOECONOMIC + ["fema_amount"],
}


def load_features(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"cbg_id": str, "county": str, "tract_id": str})
    required = {
        "cbg_id", "county", "tract_id", "original_top100", "n_received",
        *[column for variables in SPECS.values() for column in variables],
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    if data.cbg_id.isna().any() or data.cbg_id.duplicated().any():
        raise ValueError("cbg_id must be nonmissing and unique")
    if data[["county", "tract_id"]].isna().any().any():
        raise ValueError("county and tract_id must be nonmissing")
    # bool('False') is True, so never cast CSV membership strings with astype(bool).
    membership = data.original_top100.astype(str).str.strip().str.lower()
    if not membership.isin(["true", "false", "1", "0"]).all():
        raise ValueError("original_top100 must contain True/False or 1/0")
    data["original_top100"] = membership.isin(["true", "1"])
    numeric = set(required) - {"cbg_id", "county", "tract_id", "original_top100"}
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    count = data.n_received
    if not (np.isfinite(count).all() and count.ge(0).all()
            and np.equal(count, np.round(count)).all()):
        raise ValueError("n_received must be finite nonnegative integer counts")
    if not data.original_top100.any():
        raise ValueError("No eligible original Top-100 CBGs in the input")
    return data


def fit_ppml(frame: pd.DataFrame, variables: list[str], county: bool = True):
    # Match the canonical refit: model-specific complete cases, no reranking,
    # no influence-based exclusions and no offsets.
    sample = frame.dropna(subset=variables + ["n_received"]).copy()
    if sample.empty or not np.isfinite(sample[variables].to_numpy()).all():
        raise ValueError("Model sample is empty or includes infinite predictors")
    formula = "1 + " + " + ".join(variables)
    if county:
        formula += " + C(county)"
    design = patsy.dmatrix(formula, sample, return_type="dataframe")
    rank = int(np.linalg.matrix_rank(design))
    clusters = int(sample.tract_id.nunique())
    if rank != design.shape[1] or len(sample) <= design.shape[1] or clusters <= 1:
        raise ValueError("Insufficient observations/clusters or rank-deficient design")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model = sm.GLM(
            sample.n_received.astype(float), design, family=sm.families.Poisson()
        ).fit(
            maxiter=150,
            cov_type="cluster",
            cov_kwds={"groups": sample.tract_id, "use_correction": True},
            use_t=True,
        )
    if not model.converged or not np.isfinite(model.params).all() or not np.isfinite(model.bse).all():
        raise ValueError("PPML did not converge to finite coefficients and standard errors")
    diagnostic = {
        "n": len(sample), "tract_clusters": clusters,
        "counties": int(sample.county.nunique()), "parameters": design.shape[1],
        "rank": rank, "converged": bool(model.converged),
        "df_inference": float(model.df_resid_inference),
        "zero_outcomes": int(sample.n_received.eq(0).sum()),
        "warnings": "; ".join(str(item.message) for item in recorded),
    }
    return model, diagnostic


def check_reference(actual: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    required = {"sample", "model", "variable", "beta", "se", "ci_low", "ci_high", "p", "n", "tract_clusters"}
    if not required.issubset(reference.columns):
        raise ValueError("Reference must use the scale_comparison_all_coefficients.csv schema")
    # The historical comparison also contains labelled alternate-count fits.
    reference = reference[reference["sample"].isin(["MSA", "Top100"])].copy()
    reference["model"] = reference.model.str.split().str[0]
    keys = ["sample", "model", "variable"]
    expected = reference.set_index(keys).sort_index()
    observed = actual.set_index(keys).sort_index()
    if not expected.index.equals(observed.index):
        raise AssertionError("Reference and fitted coefficient keys differ")
    np.testing.assert_array_equal(observed[["n", "tract_clusters"]], expected[["n", "tract_clusters"]])
    np.testing.assert_allclose(
        observed[["beta", "se", "ci_low", "ci_high", "p"]],
        expected[["beta", "se", "ci_low", "ci_high", "p"]],
        rtol=1e-8, atol=1e-10,
    )


def run(features_path: Path, output: Path, reference: Path | None = None) -> None:
    data = load_features(features_path)
    coefficients, diagnostics, blocks, bridges = [], [], [], []
    for sample_name, frame in [("MSA", data), ("Top100", data[data.original_top100].copy())]:
        for model_name, variables in SPECS.items():
            model, diagnostic = fit_ppml(frame, variables)
            diagnostic.update(sample=sample_name, model=model_name)
            diagnostics.append(diagnostic)
            for variable in model.params.index:
                lo, hi = model.conf_int().loc[variable]
                coefficients.append({
                    "sample": sample_name, "model": model_name, "variable": variable,
                    "beta": float(model.params[variable]), "se": float(model.bse[variable]),
                    "ci_low": float(lo), "ci_high": float(hi), "p": float(model.pvalues[variable]),
                    "n": diagnostic["n"], "tract_clusters": diagnostic["tract_clusters"],
                })
            if model_name == "M3":
                for block, columns in [("Environmental", ENVIRONMENT), ("Socioeconomic", SOCIOECONOMIC)]:
                    restriction = np.zeros((len(columns), len(model.params)))
                    for row, column in enumerate(columns):
                        restriction[row, model.params.index.get_loc(column)] = 1
                    test = model.wald_test(restriction, use_f=True, scalar=True)
                    blocks.append({
                        "sample": sample_name, "block": block,
                        "F": float(test.statistic), "p": float(test.pvalue),
                        "df_num": float(test.df_num), "df_denom": float(test.df_denom),
                    })
        # SI matched-sample income adjustment: all four fits use the M3 sample.
        bridge_sample = frame.dropna(subset=SPECS["M3"] + ["n_received"])
        for label, variables, county in [
            ("Income only", ["log_income"], False),
            ("Income + basic controls", BASE + ["log_income"], True),
            ("Income + environmental controls", BASE + ["log_income"] + ENVIRONMENT, True),
            ("Joint", SPECS["M3"], True),
        ]:
            model, diagnostic = fit_ppml(bridge_sample, variables, county)
            lo, hi = model.conf_int().loc["log_income"]
            bridges.append({
                "sample": sample_name, "model": label, "beta": float(model.params.log_income),
                "ci_low": float(lo), "ci_high": float(hi), "p": float(model.pvalues.log_income),
                "n": diagnostic["n"], "tract_clusters": diagnostic["tract_clusters"],
            })
    result = pd.DataFrame(coefficients)
    if reference is not None:
        check_reference(result, reference)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "receiving_count_coefficients.csv", index=False)
    for label, filename in [("MSA", "msa_receiving_count_coefficients.csv"), ("Top100", "top100_receiving_count_coefficients.csv")]:
        result[result["sample"].eq(label)].to_csv(output / filename, index=False)
    pd.DataFrame(diagnostics).to_csv(output / "model_diagnostics.csv", index=False)
    pd.DataFrame(blocks).to_csv(output / "joint_block_tests.csv", index=False)
    pd.DataFrame(bridges).to_csv(output / "income_model_bridge.csv", index=False)
    audit = {
        "input_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "outcome": "n_received: distinct qualifying arrivals",
        "original_top100_preserved": True,
        "reference_checked": reference is not None,
        "covariance": "census-tract cluster; finite-sample correction; t reference",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "patsy": patsy.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "diagnostics": diagnostics,
    }
    (output / "refit_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(diagnostics)[["sample", "model", "n", "tract_clusters", "converged"]].to_string(index=False))
    if reference is not None:
        print("All canonical coefficients, standard errors, confidence intervals, p values and sample sizes match the reference.")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Prepared destination_model_sample.csv")
    parser.add_argument("--output", type=Path, required=True, help="Directory for derived aggregate results")
    parser.add_argument("--reference", type=Path, help="Optional scale_comparison_all_coefficients.csv for strict validation")
    arguments = parser.parse_args()
    run(arguments.features, arguments.output, arguments.reference)
