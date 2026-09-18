"""Direct GROMACS driver.

Reproduces the CHAPERONg umbrella-sampling stage sequence with explicit gmx
calls, so windows can be run concurrently and the pipeline survives on machines
where CHAPERONg's interactive prompts (or PyMOL/ImageMagick) are unavailable.
"""

from __future__ import annotations

import copy
import json
import queue
import re
import zlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import windows as win
from ..xvg import read_xvg
from .base import (Gmx, GmxError, RunContext, available_cores, build_index,
                   check_minimisation, effective_max_distance, pull_pbcatoms,
                   setup_restraints, threads_per_worker, verify_cyclisation,
                   verify_pull_range, window_files, write_pull_pbcatoms,
                   write_window_lists)

STAGES = [
    "topology", "box", "solvate", "ions", "minimize", "index",
    "nvt", "npt", "smd", "frames", "distances", "select", "sample",
]


def run(ctx: RunContext, start: str = "topology", stop: str | None = None) -> dict:
    order = STAGES[STAGES.index(start):]
    if stop:
        order = order[:order.index(stop) + 1]

    state_path = ctx.workdir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}

    for stage in order:
        globals()[f"_{stage}"](ctx)
        state["done"] = sorted(set(state["done"]) | {stage}, key=STAGES.index)
        state_path.write_text(json.dumps(state, indent=2))
    return state


def _prep_info(ctx: RunContext) -> dict:
    return json.loads((ctx.workdir.parent / "prep.json").read_text())


UNKNOWN_OPTION = re.compile(r"(?i)(unknown|invalid) command-line option")


def _topology(ctx: RunContext) -> None:
    p = ctx.proto.prep
    args = ["pdb2gmx", "-f", f"{ctx.prefix}.pdb", "-o", "processed.gro",
            "-p", "topol.top", "-water", p.water, "-ff", p.force_field]
    if p.ignh:
        args.append("-ignh")
    if p.heavy_hydrogens:
        args.append("-heavyh")

    # pdb2gmx closes a backbone ring when the terminal N-C distance falls
    # between -sb and -lb. Both are hidden options, so fall back if the build
    # in use does not accept them.
    info = _prep_info(ctx)
    cyclic = ["-sb", str(info.get("cyclic_sb_nm", p.cyclic_sb)),
              "-lb", str(info.get("cyclic_lb_nm", p.cyclic_lb))]
    proc = ctx.gmx.run(args + cyclic, check=False)
    if proc.returncode != 0:
        if not UNKNOWN_OPTION.search(proc.stdout or ""):
            tail = "\n".join((proc.stdout or "").splitlines()[-40:])
            raise GmxError(f"pdb2gmx failed (rc={proc.returncode}):\n{tail}")
        print(f"[warn] gmx rejects {' '.join(cyclic)}; running pdb2gmx without "
              "explicit cyclisation thresholds (defaults: -sb 0.05 -lb 0.25 nm)")
        ctx.gmx.run(args)

    verify_cyclisation(ctx)
    setup_restraints(ctx)


def _box(ctx: RunContext) -> None:
    info = _prep_info(ctx)
    center = " ".join(str(v) for v in info["center_nm"]).split()
    box = " ".join(str(v) for v in info["box_nm"]).split()
    ctx.gmx.run(["editconf", "-f", "processed.gro", "-o", "newbox.gro",
                 "-center", *center, "-box", *box])


def _solvate(ctx: RunContext) -> None:
    ctx.gmx.run(["solvate", "-cp", "newbox.gro", "-cs", "spc216.gro",
                 "-o", "solv.gro", "-p", "topol.top"])


def _ions(ctx: RunContext) -> None:
    p = ctx.proto
    ctx.gmx.run(["grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "topol.top",
                 "-o", "ions.tpr", "-maxwarn", str(p.run.maxwarn)])
    ctx.gmx.run(["genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "topol.top",
                 "-neutral", "-conc", str(p.prep.ion_conc),
                 "-pname", p.prep.pname, "-nname", p.prep.nname], stdin="SOL\n")


def _minimize(ctx: RunContext) -> None:
    ctx.gmx.run(["grompp", "-f", "minim.mdp", "-c", "solv_ions.gro", "-p", "topol.top",
                 "-o", "em.tpr", "-maxwarn", str(ctx.proto.run.maxwarn)])
    ctx.gmx.mdrun("em", ctx.proto, dynamical=False)
    fmax = check_minimisation(ctx)
    print(f"      EM converged, Fmax = {fmax:.0f} kJ/mol/nm")


def _index(ctx: RunContext) -> None:
    build_index(ctx, "solv_ions.gro")
    if ctx.proto.prep.pull_pbcatom:
        atoms = pull_pbcatoms(ctx, "solv_ions.gro")
        write_pull_pbcatoms(ctx, atoms)
        print("      pull reference atoms: "
              + ", ".join(f"{k} {v}" for k, v in atoms.items()))


def _nvt(ctx: RunContext) -> None:
    ctx.gmx.run(["grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro", "-p", "topol.top",
                 "-n", "index.ndx", "-o", "nvt.tpr", "-maxwarn", str(ctx.proto.run.maxwarn)])
    ctx.gmx.mdrun("nvt", ctx.proto)


def _npt(ctx: RunContext) -> None:
    ctx.gmx.run(["grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro", "-t", "nvt.cpt",
                 "-p", "topol.top", "-n", "index.ndx", "-o", "npt.tpr",
                 "-maxwarn", str(ctx.proto.run.maxwarn)])
    ctx.gmx.mdrun("npt", ctx.proto)


def _smd(ctx: RunContext) -> None:
    verify_pull_range(ctx)
    ctx.gmx.run(["grompp", "-f", "md_pull.mdp", "-c", "npt.gro", "-r", "npt.gro", "-t", "npt.cpt",
                 "-p", "topol.top", "-n", "index.ndx", "-o", "pull.tpr",
                 "-maxwarn", str(ctx.proto.run.maxwarn)])
    ctx.gmx.mdrun("pull", ctx.proto, extra=["-pf", "pullf.xvg", "-px", "pullx.xvg"])


def _frames(ctx: RunContext) -> None:
    out = ctx.workdir / "coordinates_SMD"
    out.mkdir(exist_ok=True)
    ctx.gmx.run(["trjconv", "-s", "pull.tpr", "-f", "pull.xtc",
                 "-o", "coordinates_SMD/coordinate.gro", "-sep"], stdin="0\n")


def _distances(ctx: RunContext) -> None:
    """The SMD trace in the pull coordinate itself, not the 3-D COM distance.

    The umbrella acts on pull_coord1 with dim = N N Y, i.e. the z component of
    the COM separation, while `gmx distance` returns the full 3-D distance. The
    two differ by the lateral excursion of the binder, which grows during the
    pull. A ladder laid out uniformly in the 3-D distance is therefore NOT
    uniform in the coordinate the windows actually sample, and the neighbouring
    histograms drift apart wherever the binder swings sideways - which is what
    put isolated zero-overlap pairs into every replica of the first campaign.
    """
    wd = ctx.workdir
    dists = None
    if (wd / "pullx.xvg").exists():
        data, _ = read_xvg(wd / "pullx.xvg")
        if data.size:
            dists = data[:, 1]
    if dists is None:
        select = 'com of group "Target" plus com of group "Binder"'
        ctx.gmx.run(["distance", "-s", "pull.tpr", "-f", "pull.xtc", "-n", "index.ndx",
                     "-select", select, "-oall", "smd_distance.xvg"])
        data, _ = read_xvg(wd / "smd_distance.xvg")
        dists = data[:, 1]
        print("      [warn] pullx.xvg missing; falling back to the 3-D COM distance")

    n_frames = len(list((wd / "coordinates_SMD").glob("coordinate*.gro")))
    if n_frames and len(dists) != n_frames:
        dists = dists[:n_frames]
    frames = np.arange(len(dists))
    win.write_distance_summary(wd / "distances_summary.txt", frames, dists)


def _select(ctx: RunContext) -> None:
    frames, dists = win.read_distance_summary(ctx.workdir / "distances_summary.txt")
    u = ctx.proto.umbrella
    reach = effective_max_distance(ctx)
    picked = win.select_windows(frames, dists, u.window_spacing, reach,
                                dense_until=min(u.dense_until, reach),
                                dense_spacing=u.dense_spacing)
    win.write_config_list(ctx.workdir / "configuratns_list.txt", picked)
    (ctx.workdir / "windows.json").write_text(
        json.dumps({"windows": picked, "report": win.spacing_report(picked)}, indent=2)
    )


def _sample(ctx: RunContext) -> None:
    picked = win.read_config_list(ctx.workdir / "configuratns_list.txt")
    workers = max(1, ctx.proto.run.window_workers)
    per = threads_per_worker(ctx.proto, workers)
    print(f"      {len(picked)} windows, {workers} at a time, "
          f"{per} OpenMP threads each ({available_cores(ctx.proto)} cores available)")

    with _slots(workers) as take:
        def one(w: dict) -> dict:
            with take() as slot:
                return _guarded_window(ctx, w, slot, workers)

        if workers == 1:
            results = [one(w) for w in picked]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one, picked))

    failed = [r for r in results if "error" in r]
    records = [r for r in results if "error" not in r]
    if failed:
        _report_failures(ctx, failed)

    write_window_lists(ctx.workdir, records)
    (ctx.workdir / "window_records.json").write_text(json.dumps(results, indent=2))


def extend_windows(ctx: RunContext, plan: list[dict]) -> list[dict]:
    """Grow only the windows that asked for it, appending to the same outputs."""
    tprs, _ = window_files(ctx.workdir)
    workers = max(1, ctx.proto.run.window_workers)

    with _slots(workers) as take:
        def one(item: dict) -> dict:
            idx = item["window"]
            if idx >= len(tprs) or item["extend_ns"] <= 0:
                return {**item, "extended": False}
            deffnm = Path(tprs[idx]).stem
            gmx = Gmx(ctx.proto.run.gmx, ctx.workdir, ctx.workdir / f"gmx_{deffnm}.log")
            gmx.run(["convert-tpr", "-s", f"{deffnm}.tpr",
                     "-extend", str(item["extend_ns"] * 1000), "-o", f"{deffnm}.tpr"])
            with take() as slot:
                gmx.mdrun(deffnm, ctx.proto,
                          extra=["-cpi", f"{deffnm}.cpt", "-append",
                                 "-pf", f"{deffnm}_pullf.xvg", "-px", f"{deffnm}_pullx.xvg"],
                          slot=slot, workers=workers)
            return {**item, "extended": True, "deffnm": deffnm}

        if workers == 1:
            return [one(p) for p in plan]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, plan))


def fill_gaps(ctx: RunContext, threshold: float | None = None,
             max_new: int | None = None) -> list[dict]:
    """Add one window per near-zero-overlap adjacent pair.

    A broken pair (no shared samples between neighbouring histograms) leaves
    the offset WHAM would fix from them unconstrained, and no amount of
    additional sampling in either neighbour manufactures a configuration the
    other one would have visited - that needs a window physically between
    them. The steered-MD trajectory already has one: `_frames`/`_distances`
    wrote every frame cg-us did not pick for the original ladder, so filling
    a gap costs one more mdrun, not a new pull.

    Requires `cg-us analyze` to have already run for this replica (reads its
    histogram overlap from `analysis/replica_result.json`); run it again
    afterwards to see whether the gap is gone.
    """
    wd = ctx.workdir
    result_path = wd / "analysis" / "replica_result.json"
    if not result_path.exists():
        raise GmxError(
            f"{wd}: no analysis/replica_result.json - run `cg-us analyze` for this "
            "replica before filling gaps, so there is a histogram overlap to fill them from"
        )
    detail = json.loads(result_path.read_text())["detail_overlap"]
    overlaps, centers = detail.get("overlaps") or [], detail.get("window_centers_nm") or []
    threshold = ctx.proto.analysis.overlap_min if threshold is None else threshold
    gaps = win.find_gaps(overlaps, centers, threshold)
    if not gaps:
        return []
    max_new = ctx.proto.run.gap_fill_max_new if max_new is None else max_new
    if max_new is not None and max_new >= 0:
        gaps = sorted(gaps, key=lambda g: g["overlap"])[:max_new]

    frames, dists = win.read_distance_summary(wd / "distances_summary.txt")
    windows_json = json.loads((wd / "windows.json").read_text())
    existing = windows_json["windows"]
    used = {w["frame"] for w in existing}
    picked = win.select_gap_frames(frames, dists, gaps, used)
    if not picked:
        return []

    next_idx = max(w["window"] for w in existing) + 1
    new_windows = [{"window": next_idx + i, **p} for i, p in enumerate(picked)]

    # window_targets() (used to pin each window's reference) reads this file,
    # so the new entries must land here before _run_window is called.
    windows_json["windows"] = existing + new_windows
    (wd / "windows.json").write_text(json.dumps(windows_json, indent=2))

    workers = max(1, ctx.proto.run.window_workers)
    with _slots(workers) as take:
        def one(w: dict) -> dict:
            with take() as slot:
                r = _guarded_window(ctx, {"window": w["window"], "frame": w["frame"]}, slot, workers)
                return r if "error" in r else {**r, "gap_before_nm": w["gap_before_nm"],
                                               "gap_after_nm": w["gap_after_nm"],
                                               "gap_overlap": w["gap_overlap"]}

        if workers == 1:
            results = [one(w) for w in new_windows]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one, new_windows))

    failed = [r for r in results if "error" in r]
    records = [r for r in results if "error" not in r]
    if failed:
        _report_failures(ctx, failed)

    old_records = json.loads((wd / "window_records.json").read_text())
    write_window_lists(wd, old_records + records)
    (wd / "window_records.json").write_text(json.dumps(old_records + records, indent=2))

    for r in records:
        print(f"      +window {r['window']} (frame {r['frame']}) between "
              f"{r['gap_before_nm']:.3f} and {r['gap_after_nm']:.3f} nm "
              f"(overlap was {r['gap_overlap']:.4f})")
    return records


def _guarded_window(ctx: RunContext, w: dict, slot: int, workers: int) -> dict:
    """Run one window, and do not let its blow-up take the other 21 with it.

    A single window integrating into a nan used to abort the whole replica from
    inside the thread pool, throwing away every window that had already
    finished. Here the failure is caught, the window is retried on its own terms
    and only then reported.
    """
    window, frame = w["window"], w["frame"]
    try:
        return _run_window(ctx, window, frame, slot, workers)
    except GmxError as exc:
        first = str(exc)

    for attempt in range(1, ctx.proto.run.window_retries + 1):
        scale = ctx.proto.run.retry_dt_scale ** attempt
        print(f"      window {window} failed; retry {attempt} at dt x{scale:g} on the CPU update path")
        try:
            return _run_window(ctx, window, frame, slot, workers,
                               retry=attempt, dt_scale=scale)
        except GmxError as exc:
            first = str(exc)
    return {"window": window, "frame": frame, "error": first}


def _report_failures(ctx: RunContext, failed: list[dict]) -> None:
    names = ", ".join(str(f["window"]) for f in failed)
    detail = "\n\n".join(f"window {f['window']}:\n{f['error']}" for f in failed)
    (ctx.workdir / "failed_windows.json").write_text(json.dumps(failed, indent=2))
    print(f"      [warn] {len(failed)} window(s) failed after retries: {names}")
    if len(failed) > ctx.proto.run.allow_failed_windows:
        raise GmxError(
            f"{len(failed)} window(s) did not finish: {names}\n"
            "Every other window is on disk, so rerun this replica to pick up only these.\n"
            "A gap in the window ladder biases the PMF, so WHAM is not run until they are "
            "back; raise run.allow_failed_windows only if you have checked the histograms.\n\n"
            + detail
        )


NAN_RE = re.compile(r"(?i)potential energy is nan|is not finite")


PULL_INIT_RE = re.compile(r"^pull_coord1_(init|start)\b.*$", re.M)


def window_targets(ctx: RunContext) -> dict[int, float]:
    """Planned reference for each window, in the pull coordinate."""
    f = ctx.workdir / "windows.json"
    if not f.exists():
        return {}
    out = {}
    for w in json.loads(f.read_text()).get("windows", []):
        val = w.get("target_distance", w.get("distance"))
        if val is not None:
            out[int(w["window"])] = float(val)
    return out


def _pinned_mdp(ctx: RunContext, source: str, tag: str, init: float) -> str:
    """Copy of `source` whose umbrella sits at `init` instead of wherever the run drifted.

    With pull_coord1_start = yes the reference is read from the structure grompp
    is handed, so after 0.5 ns of NPT each window ends up a random ~sigma from
    its planned node. Neighbours then sit anywhere between 0.05 and 0.25 nm
    apart and the overlap collapses in the pairs that spread. Pinning the
    reference makes the ladder exactly what was planned.
    """
    text = PULL_INIT_RE.sub("", (ctx.workdir / source).read_text())
    text = text.rstrip() + (f"\npull_coord1_start        = no"
                            f"\npull_coord1_init         = {init:.4f}\n")
    name = f"{Path(source).stem}_{tag}.mdp"
    (ctx.workdir / name).write_text(text)
    return name


def _retry_inputs(ctx: RunContext, tag: str, dt_scale: float, seed: int) -> str:
    """A slower, freshly seeded copy of md_umbrella.mdp for one window.

    dt does not change the ensemble being sampled, only how accurately it is
    integrated, so a window rerun at a shorter step still belongs in the same
    WHAM set. New velocities are what break the trajectory out of the state that
    diverged.
    """
    base = f"md_umbrella_{tag}.mdp"
    if not (ctx.workdir / base).exists():
        base = "md_umbrella.mdp"
    src = (ctx.workdir / base).read_text().splitlines()
    dt = ctx.proto.umbrella.dt * dt_scale
    nsteps = int(ctx.proto.umbrella.time_ns * 1000 / dt)
    out = []
    for line in src:
        key = line.split("=")[0].strip()
        if key == "dt":
            out.append(f"dt                       = {dt}")
        elif key == "nsteps":
            out.append(f"nsteps                   = {nsteps}")
        elif key in ("gen_vel", "gen_seed", "gen_temp", "continuation"):
            continue
        else:
            out.append(line)
    out += [f"continuation             = no",
            f"gen_vel                  = yes",
            f"gen_temp                 = {ctx.proto.temperature}",
            f"gen_seed                 = {seed}"]
    name = f"md_umbrella_retry_{tag}.mdp"
    (ctx.workdir / name).write_text("\n".join(out) + "\n")
    return name


@contextmanager
def _slots(workers: int):
    """Hand each concurrent mdrun a distinct core range to pin to."""
    pool = queue.Queue()
    for i in range(workers):
        pool.put(i)

    @contextmanager
    def take():
        slot = pool.get()
        try:
            yield slot
        finally:
            pool.put(slot)

    yield take


def _run_window(ctx: RunContext, window: int, frame: int,
                slot: int = 0, workers: int = 1,
                retry: int = 0, dt_scale: float = 1.0) -> dict:
    tag = f"win{window}_conf{frame}"
    gro = f"./coordinates_SMD/coordinate{frame}.gro"
    gmx = Gmx(ctx.proto.run.gmx, ctx.workdir, ctx.workdir / f"gmx_{tag}.log")
    mw = str(ctx.proto.run.maxwarn)
    proto = ctx.proto
    if retry:
        # GPU-resident mode silences LINCS: a diverging window prints nothing
        # until the energy is already nan. The retry runs the update on the CPU
        # so the constraint warnings that explain the blow-up reach the log.
        proto = copy.deepcopy(ctx.proto)
        proto.run.gpu_update = False

    target = window_targets(ctx).get(window)
    npt_mdp = ("npt_umbrella.mdp" if target is None
               else _pinned_mdp(ctx, "npt_umbrella.mdp", tag, target))

    if not (ctx.workdir / f"npt_{tag}.gro").exists():
        gmx.run(["grompp", "-f", npt_mdp, "-c", gro, "-r", gro, "-p", "topol.top",
                 "-n", "index.ndx", "-o", f"npt_{tag}.tpr", "-maxwarn", mw])
        gmx.mdrun(f"npt_{tag}", proto, slot=slot, workers=workers)

    # mdrun writes the final .gro only when it reaches the last step, so that is
    # the completion marker. The pull xvg appears at the first output step and a
    # window that died at 2.6 ns of 5 leaves a plausible-looking partial file,
    # which used to make a rerun skip exactly the window that needed redoing.
    if not (ctx.workdir / f"umbrella_{tag}.gro").exists():
        if retry:
            seed = zlib.crc32(f"{window}:{frame}:{retry}".encode()) % 100000
            mdp_file = _retry_inputs(ctx, tag, dt_scale, seed)
            source = ["-c", f"npt_{tag}.gro", "-r", f"npt_{tag}.gro"]
        else:
            mdp_file = ("md_umbrella.mdp" if target is None
                        else _pinned_mdp(ctx, "md_umbrella.mdp", tag, target))
            source = ["-c", f"npt_{tag}.gro", "-t", f"npt_{tag}.cpt", "-r", f"npt_{tag}.gro"]
        gmx.run(["grompp", "-f", mdp_file, *source, "-p", "topol.top", "-n", "index.ndx",
                 "-o", f"umbrella_{tag}.tpr", "-maxwarn", mw])
        gmx.mdrun(f"umbrella_{tag}", proto,
                  extra=["-pf", f"umbrella_{tag}_pullf.xvg", "-px", f"umbrella_{tag}_pullx.xvg"],
                  slot=slot, workers=workers)

    return {
        "window": window,
        "frame": frame,
        "tpr": f"umbrella_{tag}.tpr",
        "pullf": f"umbrella_{tag}_pullf.xvg",
        "pullx": f"umbrella_{tag}_pullx.xvg",
    }
