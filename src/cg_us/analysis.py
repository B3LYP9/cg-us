"""Per-replica metrics and per-system aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import convergence, integration as UI, wham as W
from .config import Protocol
from .manifest import Entry
from .windows import spacing_report
from .xvg import read_xvg


def _pick_estimator(proto: Protocol, wham_dg: dict, ui_dg: dict | None,
                    match: dict, xi_connected: float | None) -> tuple[dict, str]:
    """Which profile provides the reported dG.

    Umbrella integration is only believed once it has reproduced WHAM on the
    stretch where WHAM itself is sound; agreeing there is the only evidence
    that its gradients, signs and units are right. It is then used exactly
    where WHAM cannot be: across a break in the window ladder.
    """
    mode = proto.analysis.estimator
    if mode == "wham" or ui_dg is None or ui_dg["dG"] is None:
        return wham_dg, "wham"
    if mode == "umbrella_integration":
        return ui_dg, "umbrella_integration"

    rms = match.get("rms")
    validated = rms is not None and rms <= proto.analysis.estimator_tolerance
    if xi_connected is not None and validated:
        return ui_dg, "umbrella_integration"
    if xi_connected is not None and not validated:
        reason = ("umbrella integration disagrees with WHAM by "
                  f"{rms:.2f} kcal/mol rms on the connected stretch"
                  if rms is not None else
                  "umbrella integration could not be compared with WHAM")
        return {**wham_dg, "no_estimate_reason": wham_dg.get("no_estimate_reason") or reason}, "wham"
    return wham_dg, "wham"


def analyse_replica(workdir: str | Path, entry: Entry, proto: Protocol, replica: int,
                    do_convergence: bool = True) -> dict:
    workdir = Path(workdir)
    paths = W.run_wham(workdir, proto, prefix="prod",
                       begin_ps=proto.umbrella.discard_ns * 1000)
    pmf = W.load_pmf(paths)

    centers, hist = W.load_histograms(paths["hist"])
    overlap = W.histogram_overlap(centers, hist)
    xi_connected = W.connected_range(overlap, proto.analysis.overlap_min)

    wham_dg = W.binding_free_energy(pmf, proto.analysis.plateau_width,
                                    jacobian=proto.analysis.jacobian_correction,
                                    kT=proto.kT_kcal, xi_max=xi_connected)

    force = UI.profile_from_windows(workdir, proto.umbrella.discard_ns * 1000,
                                    k=proto.umbrella.k)
    ui_dg = None
    if force.xi.size:
        ui_dg = W.binding_free_energy(W.PMF(force.xi, force.energy, force.error),
                                      proto.analysis.plateau_width,
                                      jacobian=proto.analysis.jacobian_correction,
                                      kT=proto.kT_kcal)
    match = UI.agreement(pmf.xi, pmf.energy, force, xi_max=xi_connected)
    dg, source = _pick_estimator(proto, wham_dg, ui_dg, match, xi_connected)

    diag = convergence.survey(workdir, proto.umbrella.discard_ns * 1000,
                              proto.umbrella.min_neff, proto.umbrella.max_drift_sd)
    usable = [d for d in diag if d.get("usable")]
    pd.DataFrame(diag).to_csv(workdir / "analysis" / "window_diagnostics.csv", index=False)

    chosen_pmf = (W.PMF(force.xi, force.energy, force.error)
                  if source == "umbrella_integration" else pmf)
    span = W.depth_over_span(chosen_pmf, proto.analysis.common_span_nm,
                             jacobian=proto.analysis.jacobian_correction, kT=proto.kT_kcal)

    record = {
        "system": entry.name,
        "replica": replica,
        "workdir": str(workdir),
        **dg,
        "xi_connected_nm": round(float(xi_connected), 3) if xi_connected is not None else None,
        **span,
        "dG_source": source,
        "dG_wham": wham_dg["dG"],
        "dG_umbrella_integration": ui_dg["dG"] if ui_dg else None,
        "dG_ui_error": ui_dg["dG_bootstrap_error"] if ui_dg else None,
        "estimator_agreement": match,
        "umbrella_integration": UI.summarise(force),
        "sampled_ns": round(sum(d["sampled_ns"] for d in usable), 1) if usable else None,
        "n_eff_min": round(min(d["n_eff"] for d in usable), 1) if usable else None,
        "n_eff_median": round(float(np.median([d["n_eff"] for d in usable])), 1) if usable else None,
        "windows_under_sampled": sum(1 for d in diag if d.get("needs_more")),
        "windows_bimodal": sum(1 for d in diag if d.get("bimodal")),
        "n_windows": hist.shape[1],
        "overlap_min": overlap["min"],
        "overlap_median": overlap["median"],
        "overlap_below_threshold": int(sum(o < proto.analysis.overlap_min
                                           for o in overlap["overlaps"])),
        "windows_connected": (len(overlap["overlaps"]) + 1 if xi_connected is None
                              else next(i + 1 for i, o in enumerate(overlap["overlaps"])
                                        if o < proto.analysis.overlap_min)),
        "smd": smd_metrics(workdir),
        "window_spacing": _window_spacing(workdir),
    }
    if do_convergence:
        record["convergence"] = W.convergence(workdir, proto, xi_max=xi_connected)
        record["convergence_ui"] = _ui_convergence(workdir, proto)
        record["ui_block_drift"] = UI.block_drift(record["convergence_ui"])

    record["window_diagnostics"] = diag
    detail = {"pmf": {"xi": pmf.xi, "energy": pmf.energy, "error": pmf.error},
              "overlap": overlap, "paths": paths}
    W.save({**record, "detail_overlap": overlap}, workdir / "analysis" / "replica_result.json")
    np.savez(workdir / "analysis" / "pmf.npz", xi=pmf.xi, energy=pmf.energy,
             error=pmf.error if pmf.error is not None else np.array([]),
             hist_centers=centers, hist=hist,
             ui_xi=force.xi, ui_energy=force.energy, ui_error=force.error)
    return record | {"_detail": detail}


def _ui_convergence(workdir: Path, proto: Protocol) -> list[dict]:
    """Convergence from the mean force, which needs no ladder and no gmx call.

    Reported for every replica, including the ones whose WHAM blocks return
    nothing because the ladder is broken - those are exactly the replicas whose
    convergence matters most.
    """
    def score(prof):
        return W.binding_free_energy(W.PMF(prof.xi, prof.energy, prof.error),
                                     proto.analysis.plateau_width,
                                     jacobian=proto.analysis.jacobian_correction,
                                     kT=proto.kT_kcal)
    return UI.convergence_blocks(workdir, proto.umbrella.discard_ns * 1000, None,
                                 proto.analysis.convergence_blocks, score,
                                 k=proto.umbrella.k)


def smd_metrics(workdir: str | Path) -> dict:
    f = Path(workdir, "pullf.xvg")
    x = Path(workdir, "pullx.xvg")
    if not f.exists():
        return {}
    force, _ = read_xvg(f)
    out = {
        "max_force_kJ_mol_nm": round(float(np.nanmax(force[:, 1])), 1),
        "time_at_max_ps": round(float(force[np.nanargmax(force[:, 1]), 0]), 1),
    }
    if x.exists():
        disp, _ = read_xvg(x)
        out["pull_range_nm"] = [round(float(np.nanmin(disp[:, 1])), 3),
                                round(float(np.nanmax(disp[:, 1])), 3)]
        n = min(len(force), len(disp))
        integrate = getattr(np, "trapezoid", None) or np.trapz
        out["work_kJ_mol"] = round(float(integrate(force[:n, 1], disp[:n, 1])), 1)
    return out


def _window_spacing(workdir: Path) -> dict:
    f = workdir / "windows.json"
    if f.exists():
        return json.loads(f.read_text()).get("report", {})
    cfg = workdir / "configuratns_list.txt"
    if not cfg.exists():
        return {}
    from .windows import read_config_list
    return spacing_report(read_config_list(cfg))


def aggregate_system(records: list[dict], entry: Entry, proto: Protocol) -> dict:
    scored = [r for r in records if r.get("dG") is not None]
    if not scored:
        return _no_result(records, entry)
    dg = np.array([r["dG"] for r in scored], dtype=float)
    n = len(dg)
    sd = float(dg.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)

    boot_err = [r["dG_bootstrap_error"] for r in scored if r.get("dG_bootstrap_error")]
    drift = [r.get("convergence", {}).get("half_split_drift") for r in scored]
    drift = [d for d in drift if d is not None]

    return {
        "system": entry.name,
        "role": entry.role,
        "binder_seq": entry.binder_seq,
        "cyclic_type": entry.cyclic_type,
        "n_replicas": n,
        "n_replicas_run": len(records),
        "replicas_without_estimate": [
            {"replica": r["replica"], "reason": r.get("no_estimate_reason")}
            for r in records if r.get("dG") is None],
        "dG_mean": round(float(dg.mean()), 3),
        "dG_sd": round(sd, 3),
        "dG_sem": round(float(sem), 3),
        "dG_ci95": [round(float(dg.mean() - tcrit * sem), 3),
                    round(float(dg.mean() + tcrit * sem), 3)],
        "dG_min": round(float(dg.min()), 3),
        "dG_max": round(float(dg.max()), 3),
        "dG_spread": round(float(dg.max() - dg.min()), 3),
        "dG_cv_pct": round(float(abs(sd / dg.mean()) * 100), 1) if dg.mean() else None,
        "dG_bootstrap_error_mean": round(float(np.mean(boot_err)), 3) if boot_err else None,
        "half_split_drift_mean": round(float(np.mean(drift)), 3) if drift else None,
        "dG_sources": sorted({r.get("dG_source") for r in scored if r.get("dG_source")}),
        "overlap_min": round(float(np.min([r["overlap_min"] for r in records
                                           if r["overlap_min"] is not None])), 4)
        if any(r["overlap_min"] is not None for r in records) else None,
        "windows_mean": round(float(np.mean([r["n_windows"] for r in records])), 1),
        "dg_exp": entry.dg_exp,
        "dg_exp_source": entry.dg_exp_source,
        "error_vs_exp": round(float(dg.mean() - entry.dg_exp), 3) if entry.dg_exp is not None else None,
        "quality_flags": quality_flags(records, proto),
    }


def _no_result(records: list[dict], entry: Entry) -> dict:
    """Every replica hit a gap before the plateau: report that, do not invent a number."""
    return {
        "system": entry.name,
        "role": entry.role,
        "binder_seq": entry.binder_seq,
        "cyclic_type": entry.cyclic_type,
        "n_replicas": 0,
        "n_replicas_run": len(records),
        "replicas_without_estimate": [
            {"replica": r["replica"], "reason": r.get("no_estimate_reason")} for r in records],
        "dG_mean": None, "dG_sd": None, "dG_sem": None, "dG_ci95": None,
        "dG_min": None, "dG_max": None, "dG_spread": None, "dG_cv_pct": None,
        "dG_bootstrap_error_mean": None, "half_split_drift_mean": None,
        "overlap_min": min((r["overlap_min"] for r in records
                            if r["overlap_min"] is not None), default=None),
        "windows_mean": round(float(np.mean([r["n_windows"] for r in records])), 1),
        "dg_exp": entry.dg_exp,
        "dg_exp_source": entry.dg_exp_source,
        "error_vs_exp": None,
        "quality_flags": [f"rep{r['replica']}: {r.get('no_estimate_reason')}" for r in records],
    }


def quality_flags(records: list[dict], proto: Protocol) -> list[str]:
    flags: list[str] = []
    a = proto.analysis
    for r in records:
        tag = f"rep{r['replica']}"
        if r.get("no_estimate_reason"):
            flags.append(f"{tag}: no dG - {r['no_estimate_reason']}")
            continue
        if r.get("truncated_at_gap"):
            flags.append(f"{tag}: profile cut at {r['xi_connected_nm']} nm, the first "
                         f"neighbouring pair below overlap {a.overlap_min}; dG comes from the "
                         "connected stretch only")
        if r["overlap_min"] is not None and r["overlap_min"] < a.overlap_min:
            flags.append(f"{tag}: histogram overlap {r['overlap_min']:.3f} < {a.overlap_min}")
        if r.get("windows_under_sampled"):
            flags.append(f"{tag}: {r['windows_under_sampled']} window(s) below "
                         f"N_eff {proto.umbrella.min_neff:.0f} - run cg-us extend")
        if r.get("windows_bimodal"):
            flags.append(f"{tag}: {r['windows_bimodal']} window(s) with a bimodal "
                         "histogram; more time may not fix a slow orthogonal motion")
        if r["plateau_roughness"] > 0.5:
            flags.append(f"{tag}: PMF tail not flat (sd {r['plateau_roughness']:.2f} kcal/mol)")
        drift = r.get("convergence", {}).get("half_split_drift")
        if drift is not None and drift > 1.0:
            flags.append(f"{tag}: first/second-half dG differ by {drift:.2f} kcal/mol")
        slope = (r.get("ui_block_drift") or {}).get("slope_kcal_per_ns")
        if slope is not None and abs(slope) > 0.25:
            flags.append(f"{tag}: dG still moving at {slope:+.2f} kcal/mol per ns over the "
                         "second half of the sampling - the estimate is not converged")
        if r.get("span_complete") is False:
            flags.append(f"{tag}: the ladder does not reach {a.common_span_nm} nm past the "
                         "minimum, so dG_span is not comparable with the other systems")
        span = r["xi_range_nm"][1] - r["xi_range_nm"][0]
        if span < proto.umbrella.max_distance * 0.8:
            flags.append(f"{tag}: reaction coordinate spans only {span:.2f} nm")
    return flags


def replica_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "system": r["system"],
            "replica": r["replica"],
            "dG_kcal": r["dG"],
            "dG_span_kcal": r.get("dG_span"),
            "span_complete": r.get("span_complete"),
            "bootstrap_err": r["dG_bootstrap_error"],
            "bound_xi_nm": r["bound_xi_nm"],
            "barrier_kcal": r["unbinding_barrier"],
            "n_windows": r["n_windows"],
            "overlap_min": r["overlap_min"],
            "overlap_median": r["overlap_median"],
            "xi_connected_nm": r.get("xi_connected_nm"),
            "dG_source": r.get("dG_source"),
            "dG_wham": r.get("dG_wham"),
            "dG_ui": r.get("dG_umbrella_integration"),
            "ui_vs_wham_rms": (r.get("estimator_agreement") or {}).get("rms"),
            "ui_bimodal_windows": (r.get("umbrella_integration") or {}).get("bimodal_windows"),
            "windows_connected": r.get("windows_connected"),
            "no_estimate_reason": r.get("no_estimate_reason"),
            "plateau_roughness": r["plateau_roughness"],
            "half_split_drift": r.get("convergence", {}).get("half_split_drift"),
            "ui_drift_kcal": (r.get("ui_block_drift") or {}).get("drift_kcal"),
            "ui_drift_per_ns": (r.get("ui_block_drift") or {}).get("slope_kcal_per_ns"),
            "sampled_ns": r.get("sampled_ns"),
            "n_eff_min": r.get("n_eff_min"),
            "under_sampled": r.get("windows_under_sampled"),
            "smd_max_force": r.get("smd", {}).get("max_force_kJ_mol_nm"),
        })
    return pd.DataFrame(rows).sort_values(["system", "replica"]).reset_index(drop=True)


def system_table(summaries: list[dict]) -> pd.DataFrame:
    cols = ["system", "role", "n_replicas", "n_replicas_run", "dG_mean", "dG_sd", "dG_sem", "dG_ci95",
            "dG_spread", "dG_cv_pct", "dG_bootstrap_error_mean", "dG_sources", "overlap_min",
            "windows_mean", "dg_exp", "error_vs_exp"]
    df = pd.DataFrame(summaries)
    return df[[c for c in cols if c in df.columns]]
