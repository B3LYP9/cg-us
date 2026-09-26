"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import analysis, contacts, experiment, plots, prep, report, wham
from .backends import chaperong as chaperong_backend
from .backends import direct as direct_backend
from .backends.base import RunContext
from .config import Protocol
from .manifest import Entry, read_manifest


def _entries(root: Path) -> list[Entry]:
    state = json.loads((root / "run_state.json").read_text())
    return read_manifest(state["manifest"], state["manifest_root"])


def _protocol(root: Path) -> Protocol:
    return Protocol.load(root / "protocol.yaml")


def cmd_validate(args) -> int:
    entries = read_manifest(args.manifest, args.data_root)
    rows = []
    for e in entries:
        rows.append({
            "system": e.name,
            "pdb": e.pdb.name,
            "target": e.target,
            "binder": e.binder,
            "role": e.role,
            "dG_exp": None if e.dg_exp is None else round(e.dg_exp, 3),
            "source": e.dg_exp_source,
            "cyclic": e.cyclic_type or "",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_exp = df["dG_exp"].notna().sum()
    print(f"\n{len(entries)} systems, {n_exp} with experimental ΔG, "
          f"{len(entries) - n_exp} without (controls / unknowns)")
    return 0


def cmd_prep(args) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    proto = Protocol.load(args.protocol)
    if args.replicas:
        proto.replicas = args.replicas
    if args.backend:
        proto.run.backend = args.backend
    proto.validate()
    proto.dump(root / "protocol.yaml")

    u = proto.umbrella
    print(f"[prep] {u.time_ns} ns/window, dt {u.dt * 1000:.0f} fs, "
          f"constraints {proto.prep.constraints}, "
          f"spacing {u.dense_spacing}/{u.window_spacing} nm over {u.max_distance} nm, "
          f"{proto.replicas} replicas")

    entries = read_manifest(args.manifest, args.data_root)
    if getattr(args, "system", None):
        entries = [e for e in entries if e.name in args.system]
        if not entries:
            print("[prep] no manifest entry matched --system")
            return 1
    infos = []
    for e in entries:
        info = prep.prepare_entry(e, proto, root, ff_source=args.ff_dir)
        infos.append(info)
        flag = "  ".join(info["warnings"])
        print(f"[prep] {e.name:<24} box {info['box_nm']} ({info['box_z_driver']})  "
              f"d0 {info['com_distance_nm']:.2f} nm  pull {proto.umbrella.max_distance} "
              f"of {info['max_pull_distance_nm']} nm available"
              + (f"\n        ! {flag}" if flag else ""))

    (root / "run_state.json").write_text(json.dumps({
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_root": str(Path(args.data_root or Path(args.manifest).parent).resolve()),
        "systems": [i["name"] for i in infos],
    }, indent=2))
    pd.DataFrame([{k: v for k, v in i.items() if not isinstance(v, (dict, list))}
                  for i in infos]).to_csv(root / "systems.csv", index=False)
    print(f"\nprepared {len(infos)} systems x {proto.replicas} replicas in {root}")
    return 0


def cmd_run(args) -> int:
    root = Path(args.root)
    proto = _protocol(root)
    if args.backend:
        proto.run.backend = args.backend
    if getattr(args, "chaperong", None):
        proto.run.chaperong = args.chaperong
    entries = _entries(root)
    if args.system:
        entries = [e for e in entries if e.name in args.system]

    from .backends.base import available_cores, threads_per_worker
    workers = max(1, proto.run.window_workers)
    print(f"[run] {available_cores(proto)} cores, {workers} window(s) at a time, "
          f"{threads_per_worker(proto, workers)} OpenMP threads each"
          + ("" if proto.run.backend == "direct" else "  (chaperong runs windows one by one)"))

    if proto.run.backend == "chaperong" and not args.dry_run:
        launcher, cg_root = chaperong_backend.preflight(proto)
        print(f"[run] CHAPERONg: {launcher} (CHAPERONg_PATH={cg_root})")

    for e in entries:
        for rep in range(1, proto.replicas + 1):
            if args.replica and rep != args.replica:
                continue
            wd = prep.replica_dir(root, e, rep)
            ctx = RunContext(entry=e, proto=proto, workdir=wd, replica=rep)
            windows = wd / "tpr_files.dat"
            if windows.exists() and windows.read_text().strip() and not args.force:
                print(f"[run] {e.name} rep{rep}: already sampled, skipping")
                continue
            print(f"[run] {e.name} rep{rep} via {proto.run.backend} in {wd}")
            if args.dry_run:
                continue
            if proto.run.backend == "chaperong":
                chaperong_backend.run(ctx, from_stage=args.from_stage)
            else:
                direct_backend.run(ctx, start=args.start, stop=args.stop)
    return 0


def cmd_analyze(args) -> int:
    root = Path(args.root)
    proto = _protocol(root)
    if getattr(args, "estimator", None):
        proto.analysis.estimator = args.estimator
    if getattr(args, "bootstraps", None) is not None:
        proto.wham.bootstraps = args.bootstraps
    entries = _entries(root)
    wanted = set(args.system) if getattr(args, "system", None) else None

    replica_records: list[dict] = []
    summaries: list[dict] = []
    figures: dict = {}

    for e in entries:
        refresh = wanted is None or e.name in wanted
        records = []
        for rep in range(1, proto.replicas + 1):
            wd = prep.replica_dir(root, e, rep)
            if not (wd / "tpr_files.dat").exists():
                if refresh:
                    print(f"[analyze] {e.name} rep{rep}: no windows, skipped")
                continue
            cached = wd / "analysis" / "replica_result.json"
            if refresh:
                print(f"[analyze] {e.name} rep{rep}")
                rec = analysis.analyse_replica(wd, e, proto, rep,
                                               do_convergence=not args.no_convergence)
                rec["npz"] = str(wd / "analysis" / "pmf.npz")
                rec["overlap"] = rec.pop("_detail")["overlap"]
            elif cached.exists():
                # --system given and this one wasn't asked for: reuse the last
                # analyse_replica() result instead of paying for wham/UI again -
                # the whole point of scoping analyze is to skip untouched systems.
                rec = json.loads(cached.read_text())
                rec["overlap"] = rec.pop("detail_overlap")
                rec["npz"] = str(wd / "analysis" / "pmf.npz")
            else:
                print(f"[analyze] {e.name} rep{rep}: not selected and never "
                      "analyzed before, skipped (run without --system once first)")
                continue
            records.append(rec)
        if not records:
            continue
        replica_records.extend(records)
        summaries.append(analysis.aggregate_system(records, e, proto))
        figures[e.name] = _system_figures(root, e, records, proto)

    if not summaries:
        print("no completed systems to analyse", file=sys.stderr)
        return 1

    comparison = experiment.compare(summaries, proto)
    out = root / "analysis"
    out.mkdir(exist_ok=True)

    rep_df = analysis.replica_table(replica_records)
    sys_df = analysis.system_table(summaries)
    cmp_df = experiment.comparison_table(comparison)
    rep_df.to_csv(out / "replicas.csv", index=False)
    sys_df.to_csv(out / "systems.csv", index=False)
    cmp_df.to_csv(out / "experiment_comparison.csv", index=False)
    wham.save({"systems": summaries, "replicas":
               [{k: v for k, v in r.items() if k not in ("overlap",)} for r in replica_records],
               "comparison": comparison}, out / "results.json")

    figures["_global"] = {
        "correlation": str(plots.correlation(summaries, comparison, out / "correlation.png")),
        "spread": str(plots.replica_spread(summaries, out / "replica_spread.png")),
    }

    tables = {"systems": sys_df, "replicas": rep_df, "comparison": cmp_df}
    path = report.build(root, summaries, replica_records, comparison, figures, tables,
                        json.loads(json.dumps(_protocol_dict(proto))))
    used = sorted({r.get("dG_source") for r in replica_records if r.get("dG_source")})
    n_ui = sum(1 for r in replica_records if r.get("dG_source") == "umbrella_integration")
    print(f"\nestimators used: {', '.join(used) or 'none'}"
          + (f" ({n_ui} of {len(replica_records)} replicas rescued by umbrella integration)"
             if n_ui else ""))
    print(f"report: {path}")
    print(cmp_df.to_string(index=False))
    return 0


def _system_figures(root: Path, entry: Entry, records: list[dict], proto: Protocol) -> dict:
    out = root / "analysis" / entry.name
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "pmf": str(plots.pmf_profiles(entry.name, records, out / "pmf.png", entry.dg_exp)),
        "overlap": str(plots.overlap_profile(entry.name, records, out / "overlap.png",
                                             proto.analysis.overlap_min)),
        "sampling": str(plots.window_sampling(entry.name, records, out / "window_sampling.png",
                                              proto.umbrella.min_neff)),
        "smd": str(plots.smd_force(entry.name, records, out / "smd_force.png")),
        "convergence": str(plots.convergence(entry.name, records, out / "convergence.png")),
    }
    for r in records:
        figs[f"hist_rep{r['replica']}"] = str(plots.window_histograms(
            entry.name, r["replica"], Path(r["npz"]),
            out / f"histograms_rep{r['replica']}.png", r["overlap"]))
    return figs


def _protocol_dict(proto: Protocol) -> dict:
    from dataclasses import asdict
    return asdict(proto)


def cmd_extend(args) -> int:
    from . import convergence as C
    from . import windows as win

    root = Path(args.root)
    proto = _protocol(root)
    u = proto.umbrella
    entries = _entries(root)
    if args.system:
        entries = [e for e in entries if e.name in args.system]

    total_added = 0.0
    total_gaps_found = 0
    total_gaps_filled = 0
    for e in entries:
        for rep in range(1, proto.replicas + 1):
            if args.replica and rep != args.replica:
                continue
            wd = prep.replica_dir(root, e, rep)
            if not (wd / "tpr_files.dat").exists():
                continue

            if args.fill_gaps:
                if not (wd / "analysis" / "replica_result.json").exists():
                    print(f"[extend] {e.name} rep{rep}: no analysis yet - "
                          "run `cg-us analyze` first, skipping --fill-gaps")
                    continue
                ctx = RunContext(entry=e, proto=proto, workdir=wd, replica=rep)
                min_err = getattr(args, "min_error", None)
                well = bool(getattr(args, "well_only", False))
                gaps = direct_backend.plan_gaps(ctx, min_error_kcal=min_err, well_only=well)
                overlap_min = proto.analysis.overlap_min
                total_gaps_found += len(gaps)
                print(f"[extend] {e.name} rep{rep}: {len(gaps)} gap(s) below overlap {overlap_min}")
                for g in gaps:
                    err = g.get("est_error_kcal")
                    print(f"          {g['before_nm']:.3f}-{g['after_nm']:.3f} nm "
                          f"(overlap {g['overlap']:.4f}) -> window mean {g['target_distance']:.3f} nm, "
                          f"pin at {g.get('ref_distance', g['target_distance']):.3f} nm"
                          + (f", est. dG error {err:.3f} kcal/mol" if err is not None else ""))
                if gaps and not args.dry_run:
                    added = direct_backend.fill_gaps(ctx, min_error_kcal=min_err, well_only=well)
                    total_gaps_filled += len(added)
                    left = len(gaps) - len(added)
                    print(f"          added {len(added)} window(s)"
                          + (f"; {left} gap(s) left over run.gap_fill_max_new="
                             f"{proto.run.gap_fill_max_new}, run again after re-analyzing"
                             if left > 0 else "")
                          + "; re-run `cg-us analyze` to see the effect")
                continue

            for round_no in range(1, args.rounds + 1):
                diag = C.survey(wd, u.discard_ns * 1000, u.min_neff, u.max_drift_sd)
                plan = C.budget(diag, u.extend_ns, u.max_ns)
                actionable = [p for p in plan if p["extend_ns"] > 0]
                pd.DataFrame(diag).to_csv(wd / "window_diagnostics.csv", index=False)

                done = sum(d.get("sampled_ns", 0) for d in diag)
                print(f"[extend] {e.name} rep{rep} round {round_no}: "
                      f"{len(diag)} windows, {done:.0f} ns sampled, "
                      f"{len(actionable)} need more")
                for p in actionable[:12]:
                    print(f"          win{p['window']:>3} +{p['extend_ns']:.1f} ns  {p['reason']}")
                capped = [p for p in plan if p.get("capped")]
                for p in capped:
                    print(f"          win{p['window']:>3} at the {u.max_ns} ns ceiling, "
                          f"still: {p['reason']}")

                if not actionable:
                    print("          all windows converged")
                    break
                total_added += sum(p["extend_ns"] for p in actionable)
                if args.dry_run:
                    break
                ctx = RunContext(entry=e, proto=proto, workdir=wd, replica=rep)
                direct_backend.extend_windows(ctx, actionable)

    if args.fill_gaps:
        if args.dry_run:
            print(f"\n{total_gaps_found} gap(s) found in total (dry run, nothing filled)")
        else:
            print(f"\nfilled {total_gaps_filled} of {total_gaps_found} gap(s) found")
    else:
        print(f"\n{'would add' if args.dry_run else 'added'} {total_added:.0f} ns in total")
    return 0


def cmd_bench(args) -> int:
    from . import bench as B

    root = Path(args.root)
    proto = _protocol(root)
    if args.workdir:
        wd = Path(args.workdir)
    else:
        entries = _entries(root)
        entry = next((e for e in entries if e.name == args.system), entries[0]) \
            if args.system else entries[0]
        wd = prep.replica_dir(root, entry, args.replica or 1)

    print(f"[bench] {wd}  ({args.steps} steps per configuration)")
    from .backends.base import available_cores
    cores = available_cores(proto)
    ntomp = tuple(args.ntomp) if args.ntomp else B.core_sweep(cores)
    print(f"        {cores} cores visible, sweeping ntomp {list(ntomp)}")
    df = B.benchmark(wd, gmx=proto.run.gmx, steps=args.steps, ntomp_values=ntomp,
                     concurrency=tuple(args.concurrency), gpu_id=proto.run.gpu_id,
                     keep=args.keep, cores=cores)

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    print("\n" + df.to_string(index=False))

    windows = args.windows or _expected_windows(proto)
    rec = B.recommend(df, windows, proto.umbrella.time_ns, proto.replicas)
    if rec:
        print(f"\nfastest: {rec['best_config']} at {rec['best_aggregate_ns_per_day']} ns/day")
        if "speedup_vs_nb_gpu" in rec:
            print(f"  {rec['speedup_vs_nb_gpu']}x the '-nb gpu' baseline "
                  f"({rec['baseline_hours_per_system']} h -> {rec['hours_per_system']} h)")
        print(f"  {windows} windows x {proto.umbrella.time_ns} ns x {proto.replicas} replicas "
              f"= {rec['total_ns_per_system']:.0f} ns -> ~{rec['hours_per_system']} h per system")
    return 0


def _expected_windows(proto: Protocol) -> int:
    u = proto.umbrella
    dense = u.dense_until / u.dense_spacing if u.dense_spacing else 0
    coarse = max(0.0, u.max_distance - u.dense_until) / u.window_spacing
    return int(round(dense + coarse)) + 1


def cmd_all(args) -> int:
    for fn in (cmd_prep, cmd_run, cmd_analyze):
        rc = fn(args)
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("cg-us", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="parse the manifest and check structures")
    v.add_argument("--manifest", required=True)
    v.add_argument("--data-root", default=None)
    v.set_defaults(func=cmd_validate)

    pr = sub.add_parser("prep", help="build the run tree")
    pr.add_argument("--manifest", required=True)
    pr.add_argument("--root", required=True)
    pr.add_argument("--data-root", default=None)
    pr.add_argument("--protocol", default=None)
    pr.add_argument("--ff-dir", default=None, help="force-field directory to link into each workdir")
    pr.add_argument("--replicas", type=int, default=None)
    pr.add_argument("--backend", choices=["chaperong", "direct"], default=None)
    pr.add_argument("--system", nargs="*", default=None,
                    help="rebuild only these manifest entries, leaving finished trees alone")
    pr.set_defaults(func=cmd_prep)

    r = sub.add_parser("run", help="run the simulations")
    r.add_argument("--root", required=True)
    r.add_argument("--system", nargs="*", default=None)
    r.add_argument("--replica", type=int, default=None)
    r.add_argument("--backend", choices=["chaperong", "direct"], default=None)
    r.add_argument("--chaperong", default=None,
                   help="path to run_CHAPERONg.sh (default: PATH, then $CHAPERONg_PATH)")
    r.add_argument("--from-stage", type=int, choices=[0, 14], default=None,
                   help="chaperong backend: force the entry stage "
                        "(0 = from topology, 14 = resume after steered MD)")
    r.add_argument("--start", default="topology", help="direct backend: first stage")
    r.add_argument("--stop", default=None, help="direct backend: last stage")
    r.add_argument("--force", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    x = sub.add_parser("extend", help="lengthen only the windows that are under-sampled")
    x.add_argument("--root", required=True)
    x.add_argument("--system", nargs="*", default=None)
    x.add_argument("--replica", type=int, default=None)
    x.add_argument("--rounds", type=int, default=3)
    x.add_argument("--dry-run", action="store_true", help="report the plan, simulate nothing")
    x.add_argument("--fill-gaps", action="store_true",
                   help="instead of lengthening windows, add one window (from an existing "
                        "SMD frame) between each adjacent pair whose histogram overlap is "
                        "below analysis.overlap_min - needs `cg-us analyze` to have run first")
    x.add_argument("--min-error", type=float, default=None, metavar="KCAL",
                   help="with --fill-gaps: skip gaps where umbrella integration's trapezoid "
                        "error across the gap is below this (kcal/mol, raw units); "
                        "default run.gap_fill_min_error_kcal")
    x.add_argument("--well-only", action="store_true",
                   help="with --fill-gaps: only gaps between the bound-state minimum and "
                        "the start of the plateau - anywhere else they cannot move dG")
    x.set_defaults(func=cmd_extend)

    b = sub.add_parser("bench", help="measure mdrun throughput and window concurrency")
    b.add_argument("--root", required=True)
    b.add_argument("--system", default=None)
    b.add_argument("--replica", type=int, default=None)
    b.add_argument("--workdir", default=None, help="benchmark this directory directly")
    b.add_argument("--steps", type=int, default=10000)
    b.add_argument("--ntomp", type=int, nargs="*", default=None)
    b.add_argument("--concurrency", type=int, nargs="*", default=[1, 2, 4])
    b.add_argument("--windows", type=int, default=None, help="override the window count")
    b.add_argument("--keep", action="store_true", help="keep the benchmark scratch files")
    b.set_defaults(func=cmd_bench)

    a = sub.add_parser("analyze", help="WHAM, statistics, figures, report")
    a.add_argument("--root", required=True)
    a.add_argument("--system", nargs="*", default=None,
                   help="only re-run wham/umbrella-integration for these systems; "
                        "everything else is pulled from its last replica_result.json "
                        "(run once without --system first, so there is one to pull)")
    a.add_argument("--no-convergence", action="store_true")
    a.add_argument("--bootstraps", type=int, default=None, metavar="N",
                   help="gmx wham bootstrap count for this run (protocol default 200). The "
                        "bootstraps dominate analysis time (minutes to hours per replica on a "
                        "shared box) and only feed the WHAM error bar; umbrella integration "
                        "does not use them, so --bootstraps 0 is the fast path")
    a.add_argument("--estimator", choices=["wham", "umbrella_integration", "auto"], default=None,
                   help="which PMF estimator reports dG (default: the protocol's, normally auto)")
    a.set_defaults(func=cmd_analyze)

    al = sub.add_parser("all", help="prep + run + analyze")
    for parser in (al,):
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--data-root", default=None)
        parser.add_argument("--protocol", default=None)
        parser.add_argument("--ff-dir", default=None)
        parser.add_argument("--replicas", type=int, default=None)
        parser.add_argument("--backend", choices=["chaperong", "direct"], default=None)
        parser.add_argument("--chaperong", default=None)
        parser.add_argument("--system", nargs="*", default=None)
        parser.add_argument("--replica", type=int, default=None)
        parser.add_argument("--from-stage", type=int, choices=[0, 14], default=None)
        parser.add_argument("--start", default="topology")
        parser.add_argument("--stop", default=None)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--no-convergence", action="store_true")
    al.set_defaults(func=cmd_all)

    ct = sub.add_parser("contacts", help="interface contacts (GetContacts types) along the unbinding path")
    ct.add_argument("--root", required=True)
    ct.add_argument("--system", nargs="*", help="only these systems")
    ct.add_argument("--replica", type=int, nargs="*", help="only these replica numbers")
    ct.add_argument("--source", choices=["windows", "smd", "both"], default="both",
                    help="umbrella windows (equilibrium, one state per window), the steered-MD pull "
                         "(non-equilibrium, binned by distance) or both")
    ct.add_argument("--stride-windows", type=int, default=1, help="use every Nth window frame")
    ct.add_argument("--stride-smd", type=int, default=5, help="use every Nth pull frame (2 ps each)")
    ct.set_defaults(func=cmd_contacts)

    qq = sub.add_parser("queue", help="inspect the ClearML queue that --enqueue submits to")
    qq.add_argument("action", choices=["status"])
    qq.add_argument("--queue", default=None, help="queue name (default $CGUS_CLEARML_QUEUE or a100-1)")
    qq.set_defaults(func=cmd_queue)

    # documented on every command that can be queued; consumed in main() before parsing
    for parser in (pr, r, x, a, al, b, ct):
        parser.add_argument("--enqueue", action="store_true",
                            help="do not run now: submit the command to a ClearML queue, where it "
                                 "starts after the jobs already waiting (needs `pip install clearml`)")
        parser.add_argument("--queue", default=None, metavar="NAME",
                            help="ClearML queue for --enqueue (default $CGUS_CLEARML_QUEUE or a100-1); "
                                 "implies --enqueue")
        parser.add_argument("--project", default=None, metavar="NAME",
                            help="ClearML project for --enqueue (default $CGUS_CLEARML_PROJECT or GP20181); "
                                 "the task lands in <project>/Umbrella sampling/<run|analysis|extend>; "
                                 "implies --enqueue")
        parser.add_argument("--task-name", default=None, metavar="TEXT",
                            help="task title in the ClearML UI (default: command, root, systems)")
    return p


def cmd_contacts(args) -> int:
    root = Path(args.root)
    proto = _protocol(root)
    entries = _entries(root)
    wanted = set(args.system) if args.system else None
    sources = ("windows", "smd") if args.source == "both" else (args.source,)
    rows, hot, dirs = [], {}, {}
    failed = 0
    for e in entries:
        if wanted and e.name not in wanted:
            continue
        for rep in range(1, proto.replicas + 1):
            if args.replica and rep not in args.replica:
                continue
            wd = prep.replica_dir(root, e, rep)
            if not (wd / "index.ndx").exists() or not (wd / "em.gro").exists():
                print(f"[contacts] {e.name} rep{rep}: not prepared, skipped")
                continue
            print(f"[contacts] {e.name} rep{rep}")
            try:
                res = contacts.analyse_replica(wd, proto, sources, args.stride_windows, args.stride_smd)
            except RuntimeError as exc:                     # MDAnalysis missing: nothing will work
                print(f"[contacts] {exc}", file=sys.stderr)
                return 1
            except (OSError, ValueError) as exc:
                print(f"[contacts] {e.name} rep{rep}: failed: {exc}", file=sys.stderr)
                failed += 1
                continue
            dirs.setdefault(e.name, []).append(wd)
            for source, summary in res.items():
                rows.append(contacts.summary_row(e.name, rep, summary))
                hot.setdefault((e.name, source), []).append(summary["hotspots"])
    if not rows:
        print("[contacts] nothing analysed", file=sys.stderr)
        return 1
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "contacts_summary.csv", index=False)
    for (system, source), lists in hot.items():
        cons = contacts.consensus(lists)
        print(f"[contacts] {system} {source}: consensus hotspots {', '.join(cons) or 'none'}")
        contacts.plot_system(system, source, dirs[system], out / f"contacts_{system}_{source}.png")
    for r in rows:
        flag = "" if r["pull_complete"] else "  PULL INCOMPLETE: contacts remain over the last 0.3 nm"
        print(f"[contacts] {r['system']} rep{r['replica']} {r['source']}: "
              f"{r[contacts.ANY]} residue pairs bound, xi_half {r['xi_half_nm']}{flag}")
    return 1 if failed and len(rows) == 0 else 0


def cmd_queue(args) -> int:
    from . import clearml_queue as cq

    if args.action == "status":
        rows = cq.queue_status(args.queue)
        if not rows:
            print(f"queue '{args.queue or cq.default_queue()}' is empty")
            return 0
        for r in rows:
            pos = "running" if r["position"] == "running" else f"#{r['position']}"
            print(f"{pos:>8}  {r['id']}  {r['name']}" + (f"  [{r['worker']}]" if r["worker"] else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from . import clearml_queue as cq

    # `queue status --queue X` is a plain option of that command, not a request to enqueue
    child, own = cq.split_own_flags(argv) if argv and argv[0] in cq.QUEUEABLE else (argv, {"enqueue": False})
    if own["enqueue"]:
        try:
            info = cq.submit(child, queue=own["queue"], name=own["name"], project=own["project"])
        except cq.QueueError as exc:
            print(f"[queue] {exc}", file=sys.stderr)
            return 1
        cq.print_submitted(info)
        return 0
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
