# Changelog

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
