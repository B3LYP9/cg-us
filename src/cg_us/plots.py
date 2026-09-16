"""Figures: PMF, window histograms, overlap, convergence, replica spread, experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import theme
from .xvg import read_xvg

theme.apply()

XI_LABEL = "COM distance ξ (nm)"
G_LABEL = "PMF (kcal/mol)"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pmf_profiles(system: str, replicas: list[dict], out: Path,
                 dg_exp: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = None
    stack = []

    for i, rep in enumerate(replicas):
        d = np.load(rep["npz"])
        xi, g = d["xi"], d["energy"] - d["energy"].min()
        color = theme.series_color(i)
        dg = rep.get("dG")
        label = (f"replica {rep['replica']}  ΔG = {dg:+.1f}" if dg is not None
                 else f"replica {rep['replica']}  no ΔG")
        cut = rep.get("xi_connected_nm")
        if cut is not None:
            # past the first disconnected pair WHAM only knows the shape, not the
            # offset, so that part is drawn but never used for a number
            live = xi <= cut
            ax.plot(xi[live], g[live], color=color, label=label)
            ax.plot(xi[~live], g[~live], color=color, lw=1.0, alpha=0.35)
            ax.axvline(cut, color=color, lw=0.8, ls=(0, (2, 3)), alpha=0.6)
        else:
            ax.plot(xi, g, color=color, label=label)
            live = np.ones_like(xi, dtype=bool)
        if d["error"].size:
            ax.fill_between(xi, g - d["error"], g + d["error"], color=color, alpha=0.15, lw=0)
        if "ui_xi" in d and d["ui_xi"].size:
            u = d["ui_energy"] - d["ui_energy"].min()
            ax.plot(d["ui_xi"], u, color=color, lw=1.0, ls=(0, (1, 2)),
                    label=("umbrella integration" if i == 0 else None))
        if dg is not None:
            grid = xi[live] if grid is None else grid
            stack.append(np.interp(grid, xi[live], g[live]))

    if len(stack) > 1:
        mean = np.mean(stack, axis=0)
        ax.plot(grid, mean, color=theme.INK, lw=1.4, ls=(0, (4, 3)), label="replica mean")

    if dg_exp is not None:
        ax.axhline(-dg_exp, color=theme.MUTED, lw=1.0, ls=":")
        ax.annotate(f"experimental well depth {abs(dg_exp):.1f}",
                    xy=(ax.get_xlim()[0], -dg_exp), xytext=(6, 5),
                    textcoords="offset points", ha="left",
                    color=theme.INK_SECONDARY, fontsize=9)

    ax.set_title(f"{system} — potential of mean force")
    ax.set_xlabel(XI_LABEL)
    ax.set_ylabel(G_LABEL)
    ax.legend(loc="lower right")
    return _save(fig, out)


def window_histograms(system: str, replica: int, npz: Path, out: Path,
                      overlap: dict | None = None) -> Path:
    d = np.load(npz)
    centers, hist = d["hist_centers"], d["hist"]
    colors = theme.ramp(hist.shape[1])

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for j in range(hist.shape[1]):
        y = np.nan_to_num(hist[:, j])
        ax.plot(centers, y, color=colors[j], lw=1.0)

    ax.set_title(f"{system} rep{replica} — umbrella window histograms")
    ax.set_xlabel(XI_LABEL)
    ax.set_ylabel("counts")
    if overlap and overlap.get("min") is not None:
        ax.annotate(f"min neighbour overlap {overlap['min']:.3f}",
                    xy=(0.99, 0.95), xycoords="axes fraction", ha="right",
                    color=theme.INK_SECONDARY, fontsize=9)
    return _save(fig, out)


def overlap_profile(system: str, per_replica: list[dict], out: Path,
                    threshold: float = 0.03) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, rep in enumerate(per_replica):
        ov = rep["overlap"]["overlaps"]
        centers = rep["overlap"].get("window_centers_nm", list(range(len(ov) + 1)))
        x = [(centers[k] + centers[k + 1]) / 2 for k in range(len(ov))]
        ax.plot(x, ov, color=theme.series_color(i), marker="o", ms=3.5,
                label=f"replica {rep['replica']}")

    ax.axhline(threshold, color=theme.STATUS["critical"], lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f"minimum acceptable {threshold}", xy=(ax.get_xlim()[1], threshold),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                color=theme.STATUS["critical"], fontsize=9)
    ax.set_title(f"{system} — neighbouring-window histogram overlap")
    ax.set_xlabel("midpoint between window centres (nm)")
    ax.set_ylabel("overlap coefficient")
    if len(per_replica) > 1:
        ax.legend(loc="upper right")
    return _save(fig, out)


def convergence(system: str, replicas: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    plotted = False
    for i, rep in enumerate(replicas):
        blocks = [b for b in rep.get("convergence", {}).get("blocks", [])
                  if b.get("dG") is not None]
        if not blocks:
            continue
        ax.plot([b["used_ns"] for b in blocks], [b["dG"] for b in blocks],
                color=theme.series_color(i), marker="o", ms=4,
                label=f"replica {rep['replica']}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return out

    ax.set_title(f"{system} — ΔG vs umbrella sampling used")
    ax.set_xlabel("cumulative production time per window (ns)")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.legend(loc="best")
    return _save(fig, out)


def replica_spread(summaries: list[dict], out: Path) -> Path:
    scored = [s for s in summaries if s.get("dG_mean") is not None]
    if not scored:
        return out
    order = sorted(scored, key=lambda s: s["dG_mean"])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(order) + 2.0))

    for i, s in enumerate(order):
        lo, hi = s["dG_ci95"]
        ax.plot([lo, hi], [i, i], color=theme.AXIS, lw=2, solid_capstyle="round")
        ax.plot([s["dG_mean"]], [i], "o", color=theme.CATEGORICAL[0], ms=7,
                markeredgecolor=theme.SURFACE, markeredgewidth=1.5, zorder=3)
        if s.get("dg_exp") is not None:
            ax.plot([s["dg_exp"]], [i - 0.22], "D", color=theme.CATEGORICAL[1], ms=6,
                    markeredgecolor=theme.SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{s['dG_mean']:+.1f} ± {s['dG_sd']:.1f}",
                    xy=(hi, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=theme.INK_SECONDARY)

    ax.set_yticks(y, [s["system"] for s in order])
    ax.set_xlabel("ΔG (kcal/mol)")
    ax.set_title("Computed ΔG per system (mean, 95% CI over replicas)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=theme.CATEGORICAL[0], label="PMF (calc)"),
               plt.Line2D([], [], marker="D", ls="", color=theme.CATEGORICAL[1], label="experiment")]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, out)


def correlation(summaries: list[dict], result: dict, out: Path) -> Path:
    pts = [s for s in summaries if s.get("dg_exp") is not None
           and s.get("dG_mean") is not None
           and s["role"] not in result.get("excluded_roles", [])]
    if len(pts) < 2:
        return out

    exp = np.array([s["dg_exp"] for s in pts])
    calc = np.array([s["dG_mean"] for s in pts])
    err = np.array([s["dG_sd"] for s in pts])

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = [min(exp.min(), calc.min()) - 1.5, max(exp.max(), calc.max()) + 1.5]
    ax.plot(lim, lim, color=theme.AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(lim, [v - 1 for v in lim], [v + 1 for v in lim],
                    color=theme.CATEGORICAL[0], alpha=0.07, lw=0, zorder=0)

    ax.errorbar(exp, calc, yerr=err, fmt="o", color=theme.CATEGORICAL[0],
                ecolor=theme.AXIS, elinewidth=1.2, capsize=3, ms=7,
                markeredgecolor=theme.SURFACE, markeredgewidth=1.2, zorder=3)

    if result.get("slope") is not None:
        xs = np.array(lim)
        ax.plot(xs, result["slope"] * xs + result["intercept"],
                color=theme.CATEGORICAL[1], lw=1.6, zorder=2)

    for s, x, y in zip(pts, exp, calc):
        ax.annotate(s["system"], xy=(x, y), xytext=(6, -3), textcoords="offset points",
                    fontsize=8, color=theme.INK_SECONDARY)

    stats_txt = (f"R² = {result.get('R2')}   ρ = {result.get('spearman_rho')}\n"
                 f"MAE = {result.get('MAE')}   RMSE = {result.get('RMSE')} kcal/mol")
    ax.annotate(stats_txt, xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=9, color=theme.INK_SECONDARY)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("experimental ΔG (kcal/mol)")
    ax.set_ylabel("PMF ΔG (kcal/mol)")
    ax.set_title("Computed vs experimental binding free energy")
    handles = [plt.Line2D([], [], color=theme.AXIS, ls=(0, (4, 3)), label="y = x (±1 kcal/mol band)"),
               plt.Line2D([], [], color=theme.CATEGORICAL[1], label="least-squares fit")]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, out)


def window_sampling(system: str, replicas: list[dict], out: Path,
                    min_neff: float = 50.0) -> Path:
    """Where the sampling actually landed: effective samples per window."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = False

    for i, rep in enumerate(replicas):
        diag = [d for d in rep.get("window_diagnostics", []) if d.get("usable")]
        if not diag:
            continue
        x = [d["mean_nm"] for d in diag]
        y = [d["n_eff"] for d in diag]
        ax.plot(x, y, color=theme.series_color(i), marker="o", ms=4, lw=1.4,
                label=f"replica {rep['replica']}")
        short = [(d["mean_nm"], d["n_eff"]) for d in diag if d["n_eff"] < min_neff]
        if short:
            ax.plot(*zip(*short), ls="", marker="o", ms=9, mfc="none",
                    mec=theme.STATUS["critical"], mew=1.6, zorder=4)
        plotted = True

    if not plotted:
        plt.close(fig)
        return out

    ax.axhline(min_neff, color=theme.STATUS["critical"], lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f"N_eff floor {min_neff:.0f}", xy=(ax.get_xlim()[1], min_neff),
                xytext=(-4, 5), textcoords="offset points", ha="right",
                color=theme.STATUS["critical"], fontsize=9)
    ax.set_yscale("log")
    ax.set_title(f"{system} — effective samples per window")
    ax.set_xlabel("window centre ξ (nm)")
    ax.set_ylabel("N_eff (samples / statistical inefficiency)")
    ax.legend(loc="upper left")
    return _save(fig, out)


def smd_force(system: str, replicas: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    plotted = False
    for i, rep in enumerate(replicas):
        f = Path(rep["workdir"], "pullf.xvg")
        x = Path(rep["workdir"], "pullx.xvg")
        if not (f.exists() and x.exists()):
            continue
        force, _ = read_xvg(f)
        disp, _ = read_xvg(x)
        n = min(len(force), len(disp))
        ax.plot(disp[:n, 1], force[:n, 1], color=theme.series_color(i), lw=1.2,
                label=f"replica {rep['replica']}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return out

    ax.set_title(f"{system} — steered MD pull force")
    ax.set_xlabel("COM displacement (nm)")
    ax.set_ylabel("force (kJ mol⁻¹ nm⁻¹)")
    ax.legend(loc="upper right")
    return _save(fig, out)
