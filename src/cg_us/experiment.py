"""Agreement between computed PMF free energies and experimental affinities."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .config import Protocol


def compare(summaries: list[dict], proto: Protocol) -> dict:
    df = pd.DataFrame(summaries)
    usable = df[(df["dg_exp"].notna()) & (df["dG_mean"].notna())
                & (~df["role"].isin(proto.analysis.exclude_roles))]

    out: dict = {
        "n_systems_total": int(len(df)),
        "n_without_estimate": int(df["dG_mean"].isna().sum()),
        "systems_without_estimate": df.loc[df["dG_mean"].isna(), "system"].tolist(),
        "n_in_regression": int(len(usable)),
        "excluded_roles": list(proto.analysis.exclude_roles),
        "controls": _controls(df, proto),
    }
    if len(usable) < 2:
        out["note"] = "fewer than two systems with experimental data; regression not computed"
        return out

    calc = usable["dG_mean"].to_numpy(dtype=float)
    exp = usable["dg_exp"].to_numpy(dtype=float)
    resid = calc - exp

    slope, intercept, r, p, stderr = stats.linregress(exp, calc)
    spearman = stats.spearmanr(exp, calc)
    kendall = stats.kendalltau(exp, calc)

    out.update({
        "MAE": round(float(np.mean(np.abs(resid))), 3),
        "RMSE": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "ME_bias": round(float(np.mean(resid)), 3),
        "max_abs_error": round(float(np.max(np.abs(resid))), 3),
        "pearson_r": round(float(r), 3),
        "pearson_p": float(p),
        "R2": round(float(r ** 2), 3),
        "spearman_rho": round(float(spearman.statistic), 3),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": round(float(kendall.statistic), 3),
        "kendall_p": float(kendall.pvalue),
        "slope": round(float(slope), 3),
        "slope_stderr": round(float(stderr), 3),
        "intercept": round(float(intercept), 3),
        "pairwise": _pairwise(usable),
        "per_system": usable.assign(residual=resid.round(3))[
            ["system", "dG_mean", "dG_sd", "dg_exp", "error_vs_exp"]
        ].to_dict("records"),
    })
    return out


def _controls(df: pd.DataFrame, proto: Protocol) -> list[dict]:
    ctrl = df[df["role"].isin(proto.analysis.exclude_roles) & df["dG_mean"].notna()]
    thr = proto.analysis.control_threshold
    return [
        {
            "system": row["system"],
            "dG_mean": row["dG_mean"],
            "threshold": thr,
            "passes": bool(row["dG_mean"] > thr),
        }
        for _, row in ctrl.iterrows()
    ]


def _pairwise(usable: pd.DataFrame) -> dict:
    """Rank agreement over all system pairs, plus the ddG table."""
    rows = []
    for a, b in itertools.combinations(usable.itertuples(index=False), 2):
        ddg_calc = a.dG_mean - b.dG_mean
        ddg_exp = a.dg_exp - b.dg_exp
        rows.append({
            "pair": f"{a.system} vs {b.system}",
            "ddG_calc": round(float(ddg_calc), 3),
            "ddG_exp": round(float(ddg_exp), 3),
            "error": round(float(ddg_calc - ddg_exp), 3),
            "sign_agrees": bool(np.sign(ddg_calc) == np.sign(ddg_exp)),
        })
    if not rows:
        return {}
    errs = np.array([r["error"] for r in rows])
    return {
        "n_pairs": len(rows),
        "ddG_MAE": round(float(np.mean(np.abs(errs))), 3),
        "ddG_RMSE": round(float(np.sqrt(np.mean(errs ** 2))), 3),
        "sign_agreement_pct": round(100 * float(np.mean([r["sign_agrees"] for r in rows])), 1),
        "pairs": rows,
    }


def comparison_table(result: dict) -> pd.DataFrame:
    metrics = [
        ("N systems in regression", result.get("n_in_regression")),
        ("MAE (kcal/mol)", result.get("MAE")),
        ("RMSE (kcal/mol)", result.get("RMSE")),
        ("Bias ME (kcal/mol)", result.get("ME_bias")),
        ("Max |error| (kcal/mol)", result.get("max_abs_error")),
        ("Pearson r", result.get("pearson_r")),
        ("R2", result.get("R2")),
        ("Spearman rho", result.get("spearman_rho")),
        ("Kendall tau", result.get("kendall_tau")),
        ("Regression slope", result.get("slope")),
        ("Regression intercept", result.get("intercept")),
        ("ddG MAE (kcal/mol)", result.get("pairwise", {}).get("ddG_MAE")),
        ("ddG sign agreement (%)", result.get("pairwise", {}).get("sign_agreement_pct")),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])
