"""Calibration of the umbrella-sampling dG against experiment, and the check points around it.

`cg-us calplot --bench runs/big_bench --check runs/gdf8_smoketest runs/prodomain_arm_gdf8`

* The fit uses the benchmark systems that have an experimental dG and a calculated one,
  without the negative controls and the native-ligand references (those are shown, not fitted):
  dG_calib = a * dG_calc + b.
* Every system with an experimental dG in a `--check` root is a check point drawn like DF3 was:
  outside the fit, with the spread of its replicas as error bar.
* The negative controls of the benchmark give the horizontal band of what "no binding" looks
  like on the calibrated scale.

Everything is read from the `analysis/systems.csv` that `cg-us analyze` writes for each root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_ROLES = ("reference",)
NEGATIVE_ROLES = ("negative_control",)
MARKERS = ("s", "^", "P", "X", "v")


def read_systems(root: str | Path) -> pd.DataFrame:
    path = Path(root) / "analysis" / "systems.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run `cg-us analyze --root {root}` first")
    df = pd.read_csv(path)
    for col in ("dg_exp", "dG_mean", "dG_sd", "n_replicas"):
        if col not in df:
            df[col] = np.nan
    return df


def fit(bench: pd.DataFrame) -> dict:
    """Least-squares line, its errors and the leave-one-out error, on the benchmark points."""
    pts = bench[~bench.role.isin(REFERENCE_ROLES + NEGATIVE_ROLES) & bench.dg_exp.notna() & bench.dG_mean.notna()]
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} benchmark systems have both an experimental and a calculated dG")
    x, y = pts.dG_mean.to_numpy(float), pts.dg_exp.to_numpy(float)
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    pred = a * x + b
    loo = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        a2, b2 = np.linalg.lstsq(np.column_stack([x[keep], np.ones(keep.sum())]), y[keep], rcond=None)[0]
        loo.append(a2 * x[i] + b2 - y[i])
    return {"a": float(a), "b": float(b), "n": int(len(x)), "systems": list(pts.system),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "mae": float(np.mean(np.abs(pred - y))),
            "loo_rmse": float(np.sqrt(np.mean(np.square(loo)))), "r": float(np.corrcoef(x, y)[0, 1])}


def calibrate(dg_calc, f: dict):
    return f["a"] * np.asarray(dg_calc, float) + f["b"]


def check_table(bench: pd.DataFrame, check_roots: list[str | Path], f: dict) -> pd.DataFrame:
    """One row per check point: experiment, calibrated estimate (mean of the valid replicas), error."""
    rows = []
    for name, df in [("bench", bench)] + [(Path(r).name, read_systems(r)) for r in check_roots]:
        if name == "bench":
            df = df[df.role.isin(REFERENCE_ROLES)]                       # native-ligand references
            kind = "reference"
        else:
            df = df[~df.role.isin(NEGATIVE_ROLES)]
            kind = name
        for r in df[df.dg_exp.notna() & df.dG_mean.notna()].itertuples():
            cal = float(calibrate(r.dG_mean, f))
            rows.append({"set": kind, "system": r.system, "dg_exp": float(r.dg_exp), "dG_calc": float(r.dG_mean),
                         "dG_calib": cal, "error": cal - float(r.dg_exp),
                         "sd_calib": float(f["a"] * (0.0 if pd.isna(r.dG_sd) else r.dG_sd)),
                         "n_replicas": int(r.n_replicas) if pd.notna(r.n_replicas) else 0})
    return pd.DataFrame(rows)


def negative_band(bench: pd.DataFrame, f: dict):
    neg = bench[bench.role.isin(NEGATIVE_ROLES) & bench.dG_mean.notna()]
    if neg.empty:
        return None
    cal = calibrate(neg.dG_mean, f)
    return float(cal.min()), float(cal.max()), int(len(neg))


def short(name: str) -> str:
    for a, b in (("_D3", ""), ("bindcraft", "bc"), ("boltzgen", "bg"), ("graft", "gr"), ("dlk1_variant", "dlk1_v"),
                 ("prodarm_", ""), ("gdf8_", "")):
        name = name.replace(a, b)
    return name


def plot(bench_root, check_roots, out) -> tuple[dict, pd.DataFrame]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import theme

    bench = read_systems(bench_root)
    f = fit(bench)
    table = check_table(bench, list(check_roots), f)
    band = negative_band(bench, f)
    fitted = bench[bench.system.isin(f["systems"])]

    theme.apply()
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    lo = min(-13.0, table.dg_exp.min() - 1.5 if len(table) else -13.0)
    hi = -1.5
    if band:
        ax.axhspan(band[0], band[1], color=theme.STATUS["warning"], alpha=0.2, lw=0, zorder=0,
                   label=f"негативные контроли (n={band[2]}): {band[0]:.1f} … {band[1]:.1f}")
    ax.fill_between([lo, hi], [lo - f["loo_rmse"], hi - f["loo_rmse"]], [lo + f["loo_rmse"], hi + f["loo_rmse"]],
                    color=theme.SEQUENTIAL[1], alpha=0.55, lw=0,
                    label=f"±LOO RMSE ({f['loo_rmse']:.2f})")
    ax.plot([lo, hi], [lo, hi], color=theme.INK_SECONDARY, lw=1.4, ls="--", label="y = x")
    for r in fitted.itertuples():
        ax.errorbar(r.dg_exp, calibrate(r.dG_mean, f), yerr=f["a"] * (0 if pd.isna(r.dG_sd) else r.dG_sd), fmt="o",
                    color=theme.CATEGORICAL[0 if r.system.split("_")[0] in ("mdm2", "keap1") else 2],
                    ecolor=theme.MUTED, elinewidth=1, capsize=2, ms=7, mec="white", mew=0.6, zorder=3)
        ax.annotate(short(r.system), (r.dg_exp, calibrate(r.dG_mean, f)), textcoords="offset points", xytext=(6, -3),
                    fontsize=7, color=theme.INK_SECONDARY)
    sets = list(dict.fromkeys(table.set)) if len(table) else []
    colors = {s: theme.CATEGORICAL[(1 if s == "reference" else 3 + i) % 8] for i, s in enumerate(sets)}
    for i, s in enumerate(sets):
        marker = "D" if s == "reference" else MARKERS[i % len(MARKERS)]
        sub = table[table.set == s]
        for r in sub.itertuples():
            ax.errorbar(r.dg_exp, r.dG_calib, yerr=r.sd_calib, fmt=marker, color=colors[s], ecolor=theme.MUTED,
                        elinewidth=1, capsize=2, ms=8, mec="white", mew=0.7, zorder=5)
            ax.annotate(f"{short(r.system)}\nрасч. {r.dG_calib:.1f}, эксп. {r.dg_exp:.1f}", (r.dg_exp, r.dG_calib),
                        textcoords="offset points", xytext=(8, 4), fontsize=7, color=colors[s])
        ax.scatter([], [], marker=marker, color=colors[s],
                   label=("нативные лиганды-мономеры (вне подгонки)" if s == "reference"
                          else f"проверочные точки: {s} (вне подгонки)"))
    ax.scatter([], [], color=theme.CATEGORICAL[0], label="MDM2 / KEAP1")
    ax.scatter([], [], color=theme.CATEGORICAL[2], label="ActRIIB / DLK1 (подгонка)")
    ax.text(0.03, 0.97, f"n = {f['n']}\nΔG$_{{калибр}}$ = {f['a']:.3f}·ΔG$_{{расч}}$ {f['b']:+.2f}\n"
            f"RMSE = {f['rmse']:.2f}   MAE = {f['mae']:.2f}\nLOO RMSE = {f['loo_rmse']:.2f}   r = {f['r']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=theme.INK_SECONDARY)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\Delta G_{эксп}$ (ккал/моль)")
    ax.set_ylabel(r"$\Delta G_{калибр}$ (ккал/моль); усы — sd реплик")
    ax.set_title(f"{Path(bench_root).name}: калибр. vs эксп. ΔG ({f['n']} в подгонке + проверочные точки)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    theme.despine(ax)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    table.round(3).to_csv(out.with_suffix(".csv"), index=False)
    return f, table
