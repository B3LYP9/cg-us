# Changelog

## 0.18.0 — ClearML queue

`cg-us prep|run|extend|analyze|all|bench ... --enqueue` submits the command to a
ClearML queue instead of running it; the shared agent takes tasks one at a
time, so queued commands run one after another instead of competing for the
GPU and the cores.

* Tasks are created in `<project>/Umbrella sampling/<run|analysis|extend>`
  (`prep`, `all`, `bench` go to `run`). The project is its own argument,
  `--project NAME` (default `$CGUS_CLEARML_PROJECT`, else `GP20181`); the
  queue is `--queue NAME` (default `$CGUS_CLEARML_QUEUE`, else `a100-1`);
  `--task-name` sets the title. `--queue`/`--project` imply `--enqueue`.
* The task is a self-contained launcher (`_clearml_launcher.py`) that runs the
  submitting interpreter's `cg_us.cli.main` in the recorded working directory
  with the recorded `PATH`/GROMACS variables, streams its output to the task
  console and propagates the exit code. The agent's environment and daemon
  are not touched.
* `cg-us queue status [--queue NAME]` lists the running and waiting tasks.
* `clearml` is an optional dependency: `pip install 'cg-us[queue]'`.

## 0.17.0 — gap-filling that lands where it is aimed, and only where it matters

First big_bench pass of `extend --fill-gaps` (163 windows, ~900 ns) closed 66%
of the gaps but 36% of the new windows (44% inside the well-to-plateau
region) did not sample where they were pinned: on a steep PMF a window settles
at `ref - grad/k`, so a window pinned at the gap midpoint slid back onto its
neighbours (keap1_nrf2 rep1: pinned 2.75 nm, sampled 2.505 nm) and the gap
stayed open. Separately, umbrella integration (the estimator on 64% of
replicas) does not need histogram overlap at all, and the trapezoid error
across a typical remaining gap is ~0.01 kcal/mol (raw), so most gaps could not
change dG.

### Added / changed

* `windows.annotate_gaps`: interpolates the neighbouring windows' mean-force
  gradient to the gap midpoint, returns the reference to pin at
  (`target + grad/k`, capped at 0.35 nm) so the window's *mean* lands on the
  midpoint, and the trapezoid error umbrella integration makes across the gap,
  `dx^2 |d grad| / 12` (kcal/mol).
* `direct.plan_gaps` / `fill_gaps` pin gap windows at that reference
  (`ref_distance` in `windows.json`, honoured by `window_targets`).
* `extend --fill-gaps --min-error KCAL` (protocol: `run.gap_fill_min_error_kcal`)
  skips gaps whose estimated integration error is below KCAL;
  `--well-only` keeps only gaps between the bound-state minimum and the start
  of the plateau. The dry run now prints the pin position and estimated error.
* `cg-us analyze --bootstraps N`: overrides `wham.bootstraps` for the run.
  The 200 bootstraps dominate analysis time (one replica ran >75 min on a
  shared box, against ~2 min without contention) and only feed the WHAM error
  bar; umbrella integration does not use them, so `--bootstraps 0` is the fast
  path when dG comes from UI.

### Measured on big_bench (4 replicas whose first-pass gap did not move)

keap1_nrf2 rep1, mdm2_pmi rep3, mdm2_p53 rep3, dlk1_variant2 rep3, one
compensated window each: every gap narrowed and its overlap rose (mdm2_p53
rep3 0.0016 -> 0.022; mdm2_pmi rep3 0.0 -> 0.0056, 0.46 -> 0.33 nm wide;
dlk1_variant2 rep3 0.0 -> 0.0038; keap1 rep1 0.0 -> 0.0004), where the
first-pass windows left them unchanged. New windows landed inside their gap
(mdm2_pmi rep3: mean 1.737 nm, pinned 1.795, target 1.641; before the fix
1.328 nm). One pass is not enough on the steepest cliffs (keap1 rep1: mean
2.603 for a 2.75 target - the slope inside the gap is steeper than the linear
interpolation of the neighbours), so re-analyze and run it again: each pass
adds a measured gradient point inside the gap.

## 0.16.0 — scoped analyze

`cg-us analyze` always re-ran wham + umbrella integration for every system in
the manifest, even when only one had changed (e.g. after `extend --fill-gaps`
on a handful of systems) - on a 30+ system campaign that means paying for a
full re-analysis to look at one.

### Added

* `cg-us analyze --system <names>`: only re-runs `analyse_replica` (wham,
  umbrella integration, convergence) for the listed systems. Every other
  system is pulled from its own `analysis/replica_result.json`, written by
  the last time it *was* analyzed - so `results.json`, `systems.csv`,
  `replicas.csv`, the report and the correlation plot still cover the whole
  campaign, not just the scoped subset. A system that was never analyzed
  before and isn't in `--system` is skipped with a message telling you to
  run once without `--system` first.

### Compatibility

`analyze` without `--system` is unchanged - full re-analysis of everything,
exactly as before.

## 0.15.0 — gap-filling windows

`cg-us extend` only ever bought a window more time; a neighbouring pair with
(near) zero histogram overlap stays broken no matter how long either one
runs, because a longer run of window i or i+1 does not manufacture a
configuration the other one would have visited. That is what put isolated
zero-overlap pairs, and the `windows_connected` << `n_windows` fallback to
umbrella_integration, into nearly every replica of both the ActRIIB benchmark
and the first GDF8 multi-chain runs.

### Added

* `windows.find_gaps` — adjacent window pairs (ordered by empirical mean
  position, as `wham.histogram_overlap` already computes them) whose overlap
  falls below a threshold, each reported with the midpoint distance a new
  window should target.
* `windows.select_gap_frames` — one not-yet-used SMD frame per gap midpoint,
  drawn from the same `distances_summary.txt` the original ladder was built
  from (`_frames`/`_distances` already wrote every frame cg-us did not pick;
  filling a gap costs one more mdrun, not a new pull).
* `backends.direct.fill_gaps` — runs those new windows (reusing `_run_window`,
  so retries/pinning/GPU-resident handling are identical to any other
  window), appends them to `windows.json`, `window_records.json` and the
  `tpr_files.dat`/`pullf_files.dat`/`pullx_files.dat` WHAM reads, without
  touching the existing windows. Requires `cg-us analyze` to have already run
  for that replica, since the gaps come from its histogram overlap.
* `cg-us extend --fill-gaps`: reports the gaps below `analysis.overlap_min`
  for each replica and, unless `--dry-run`, fills up to `run.gap_fill_max_new`
  of them (worst overlap first). Re-run `cg-us analyze` afterwards to see
  whether the gap is gone.

### Compatibility

`extend` without `--fill-gaps` is unchanged. New windows get fresh indices
appended after the existing ones; WHAM and the overlap/connectivity analysis
already sort by empirical position rather than input order, so nothing
downstream needs to know a window was inserted rather than planned upfront.

## 0.14.0 — multi-chain pull groups

Before this version `target` and `binder` in the manifest were single chains
(`target_chain: str`, `binder_chain: str`): prep took exactly two chains,
chain-to-moleculetype mapping returned one name per group, and chain order was
derived from the assumption that there were exactly two chains. A pull group
that lives at the interface of two chains (a ligand dimer, a receptor with a
co-receptor, a Fab) could not be described this way — the target had to be
reduced to one chain, and the reaction coordinate was measured to the COM of
half the target.

### Added

* Multi-chain group specs in the `target` and `binder` columns: `M+N`
  (separators `+`, `,`, `/`, whitespace). Parsing lives in
  `structure.parse_chains`; `Entry.target_chains` / `binder_chains` /
  `all_chains` / `multi_chain` expose the parsed groups. Single-letter specs
  keep working exactly as before.
* `structure.select_chains` — atom selection for a whole group;
  `structure.group_extent` — per-axis extent of a group. `structure.
  orient_for_pull` accepts group specs and computes the COM over the whole
  group; `prep.json` gained `target_chains`, `binder_chains`,
  `target_extent_nm`, `binder_extent_nm`.
* `topology.chain_moltype_groups` — one list of molecule types per group; both
  prior matching routes are kept (the `_chain_X` suffix, and, when chain
  letters are lost, positional matching against `[ molecules ]`).
  `topology.pull_group_indices` collects atom indices across every chain of a
  group.
* Position restraints are written per chain of the group, one
  `position_restraints` block each; a single-chain group keeps the previous
  filename (`posre_target.itp`), a multi-chain group gets one
  `posre_target_<moltype>.itp` per chain (`backends.base._restrain_group`).
* Pre-run check (`backends.base._check_group_fits_box`): no atom of a pull
  group may sit farther than half a box edge from the group's reference atom
  on any axis, otherwise gmx would min-image it to the wrong side and the
  group COM — and the reaction coordinate — would be silently wrong.
* The cyclic-backbone check and the binding-agent warnings now walk every
  chain of a group. If a group has more than one cyclic chain, prep aborts:
  `pdb2gmx` only closes the backbone ring of the first cyclic chain it
  processes (GROMACS issue 5091), so the rest would come out linear with
  charged termini a bond length apart.

### Compatibility

Manifests and `protocol.yaml` from earlier versions keep working unchanged.
`topology.chain_moltypes` (the flat, one-name-per-group variant) is kept, and
now explicitly raises if a group spans more than one chain, instead of
silently returning the first chain and being wrong.

### Tests

`tests/test_multichain.py` adds 16 tests for the multi-chain paths (chain-spec
parsing, group selection, orientation, topology mapping in both the by-chain
and positional-fallback routes, cyclic-order validation and its two-cyclic-
chains failure mode, per-chain restraints, and the box-fit check); all
previously-passing tests are unaffected.

## 0.13.1 — mean-force convergence and depth-over-span analysis

Synced from the wheel already installed in `/home/orrls/venv/cgus`
(`cg_us-0.13.1-py3-none-any.whl`), built from this source tree before the
multi-chain work above and not carried back until now. See the "Sync
mean-force convergence..." commit for the detail; summary:

* `wham.depth_over_span` / `analysis._ui_convergence` / `block_drift`: well
  depth measured over a fixed distance past the bound-state minimum instead of
  whatever the window ladder covered, and convergence tracked from the
  mean-force profile alone (no `gmx wham` call, works even when the ladder is
  broken).
* `integration.window_force` / `profile_from_windows` gain `end_ps` for block
  analysis over a fixed sampling slice; `window_span_ps` reports the shortest
  window's real sampled length.
* `config.Analysis` gains `common_span_nm` (default 1.5 nm) and
  `convergence_blocks` (default 4).
