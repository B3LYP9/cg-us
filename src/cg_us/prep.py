"""Build the per-system / per-replica run tree."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import mdp, structure
from .config import Protocol
from .manifest import Entry


def effective_lb(entry: Entry, proto: Protocol) -> float:
    """pdb2gmx -lb for this system, in nm.

    Widening -lb to 0.4 nm is what makes pdb2gmx close genuine head-to-tail
    peptides whose terminal N-C distance sits above the 0.25 nm default. It also
    sweeps in loops that are merely close-ended: an excised CDR graft with free
    termini would be silently cyclised. When the manifest says the binder is
    cyclised some other way (a disulfide, a staple), the conservative default is
    used for that system instead.
    """
    if not proto.prep.cyclic_lb_from_manifest:
        return proto.prep.cyclic_lb
    if entry.head_to_tail is False:
        return min(proto.prep.cyclic_lb, proto.prep.cyclic_lb_conservative)
    return proto.prep.cyclic_lb


PULL_BOX_FRACTION = 0.49


def pull_box_length(extent_z: float, com_distance: float, proto: Protocol) -> tuple[float, str]:
    """Box length along the pull axis, in nm.

    Two independent constraints, and the binding one is not always the obvious:

    * geometry - the two molecules once separated, plus solvent padding;
    * the pull code - gmx aborts when the COM separation exceeds 0.49 of the box
      along the pulled dimension, and the final separation is the *initial* COM
      distance plus the pulling length, not just the pulling length. NPT then
      shrinks the box by a percent or two, so a margin is added on top.
    """
    pull = proto.umbrella.max_distance
    head = proto.prep.pull_headroom
    geometric = extent_z + pull + 2 * proto.prep.box_edge_z
    minimum_image = ((com_distance + pull + head) / PULL_BOX_FRACTION
                     * (1 + proto.prep.pull_box_margin))
    if minimum_image > geometric:
        return minimum_image, "pull minimum-image check"
    return geometric, "geometry"


def max_pull_distance(box_z: float, com_distance: float, proto: Protocol) -> float:
    """How far this box could pull before gmx refuses, keeping both margins."""
    usable = box_z / (1 + proto.prep.pull_box_margin) * PULL_BOX_FRACTION
    return max(0.0, usable - com_distance - proto.prep.pull_headroom)


def system_dir(root: str | Path, entry: Entry) -> Path:
    return Path(root, "systems", entry.name)


def replica_dir(root: str | Path, entry: Entry, replica: int) -> Path:
    return system_dir(root, entry) / f"rep{replica}"


def _chain_order(entry: Entry, reports: dict[str, structure.ChainReport]) -> list[str]:
    """Cyclic chain first.

    pdb2gmx only closes the backbone ring of a chain if no linear chain was
    processed before it (GROMACS issue 5091). With the target written first the
    macrocyclic binder came out linear *and* end-capped: NH3+ and COO- landed
    1.3 A apart, EM reported an infinite force and the run died in the first
    GPU step. Leading with the cyclic chain is the upstream workaround.
    """
    chains = entry.all_chains
    cyclic = [ch for ch in chains if reports[ch].cyclic]
    if len(cyclic) > 1:
        raise ValueError(
            f"{entry.name}: chains {', '.join(cyclic)} are all cyclic, and pdb2gmx closes "
            "the ring of the first chain only (GROMACS issue 5091), so every later one "
            "would come out linear with charged termini a bond length apart. Build the "
            "system from a pre-cyclised topology instead."
        )
    return cyclic + [ch for ch in chains if ch not in cyclic]


def prepare_entry(entry: Entry, proto: Protocol, root: str | Path,
                  ff_source: str | Path | None = None) -> dict:
    sysdir = system_dir(root, entry)
    sysdir.mkdir(parents=True, exist_ok=True)

    atoms = structure.read_pdb(entry.pdb)
    present = structure.chains(atoms)
    for chs, label in ((entry.target_chains, "target"), (entry.binder_chains, "binder")):
        for ch in chs:
            if ch not in present:
                raise ValueError(f"{entry.name}: {label} chain '{ch}' absent "
                                 f"(chains: {', '.join(present)})")

    group_chains = entry.all_chains
    kept = [a for a in atoms if a.chain in set(group_chains)]
    lb = effective_lb(entry, proto)
    reports = {
        ch: structure.describe_chain(kept, ch, cyclic_min=proto.prep.cyclic_sb * 10,
                                     cyclic_max=lb * 10)
        for ch in group_chains
    }
    oriented, geom = structure.orient_for_pull(kept, entry.target_chains, entry.binder_chains)

    chain_order = _chain_order(entry, reports)
    input_pdb = sysdir / f"{entry.name}.pdb"
    structure.write_pdb(oriented, input_pdb, chain_order=chain_order)

    coords = np.array([a.xyz for a in oriented]) / 10.0
    extent = coords.max(axis=0) - coords.min(axis=0)
    box_z, z_driver = pull_box_length(extent[2], geom["com_distance_nm"], proto)
    box = np.array([
        extent[0] + 2 * proto.prep.box_edge_xy,
        extent[1] + 2 * proto.prep.box_edge_xy,
        box_z,
    ])
    center = np.array([box[0] / 2, box[1] / 2, proto.prep.box_edge_z + extent[2] / 2])

    info = {
        "name": entry.name,
        "source_pdb": str(entry.pdb),
        "target_chain": entry.target,
        "binder_chain": entry.binder,
        "target_chains": entry.target_chains,
        "binder_chains": entry.binder_chains,
        "role": entry.role,
        "dg_exp": entry.dg_exp,
        "dg_exp_source": entry.dg_exp_source,
        "affinity_type": entry.affinity_type,
        "binder_seq": entry.binder_seq,
        "cyclic_type": entry.cyclic_type,
        "chain_order": chain_order,
        "n_atoms_kept": len(kept),
        "com_distance_nm": round(geom["com_distance_nm"], 3),
        "target_extent_nm": [round(v, 3) for v in geom["target_extent_nm"]],
        "binder_extent_nm": [round(v, 3) for v in geom["binder_extent_nm"]],
        "box_nm": [round(float(v), 3) for v in box],
        "box_z_driver": z_driver,
        "center_nm": [round(float(v), 3) for v in center],
        "max_pull_distance_nm": round(
            max_pull_distance(float(box[2]), geom["com_distance_nm"], proto), 3),
        "chain_reports": {ch: asdict(r) for ch, r in reports.items()},
        "cyclic_sb_nm": proto.prep.cyclic_sb,
        "cyclic_lb_nm": lb,
        "warnings": _warnings(entry, reports, proto, lb),
        "replicas": [],
    }

    ff_local = materialise_forcefield(ff_source, Path(root)) if ff_source else []

    for rep in range(1, proto.replicas + 1):
        rdir = replica_dir(root, entry, rep)
        rdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_pdb, rdir / f"{entry.name}.pdb")
        # zlib, not hash(): str hashing is salted per interpreter run, so the
        # same prep gave a different velocity seed every time it was re-run
        seed = proto.seed_base + 1000 * rep + zlib.crc32(entry.name.encode()) % 997
        mdp.write_all(rdir, proto, restrain_binder=entry.restrain_binder, seed=seed)
        shim = write_gmx_shim(rdir, proto, lb=lb)
        mdp.write_parafile(rdir, proto, gmx_exe=str(shim))
        for ff in ff_local:
            _link_forcefield(ff, rdir)
        info["replicas"].append({"replica": rep, "dir": str(rdir), "seed": seed})

    (sysdir / "prep.json").write_text(json.dumps(info, indent=2))
    return info


def _warnings(entry: Entry, reports: dict[str, structure.ChainReport],
              proto: Protocol, lb: float) -> list[str]:
    out: list[str] = []
    sb_a, lb_a = proto.prep.cyclic_sb * 10, lb * 10
    clamped = lb < proto.prep.cyclic_lb

    for bch in entry.binder_chains:
        binder = reports[bch]
        d = binder.head_tail_distance
        if clamped and d is not None and lb_a < d < proto.prep.cyclic_lb * 10:
            out.append(
                f"binder chain {bch}: N(first)-C(last) = {d:.2f} A would fall inside the "
                f"requested -lb {proto.prep.cyclic_lb} nm window, but the manifest calls the binder "
                f"'{entry.cyclic_type}' rather than head-to-tail, so -lb is clamped to {lb} nm here "
                "and the termini stay free (prep.cyclic_lb_from_manifest: false to override)"
            )
        if binder.cyclic:
            out.append(
                f"binder chain {bch}: N(first)-C(last) = {d:.2f} A, inside the pdb2gmx "
                f"ring-closure window (-sb {proto.prep.cyclic_sb} to -lb {lb} nm) -> backbone "
                "will be closed automatically; verify the bond in topol.top once"
            )
        elif d is not None and entry.head_to_tail:
            out.append(
                f"binder chain {bch}: manifest says '{entry.cyclic_type}' but "
                f"N(first)-C(last) = {d:.2f} A is outside the ring-closure window "
                f"({sb_a:.1f}-{lb_a:.1f} A); pdb2gmx will build charged termini instead"
            )
        if binder.disulfides:
            pairs = ", ".join(f"{a}-{b}" for a, b in binder.disulfides)
            out.append(f"binder chain {bch} disulfides at residues {pairs}; "
                       "verify specbond handling")
    for ch, rep in reports.items():
        if rep.nonstandard:
            out.append(f"chain {ch}: non-standard residues {', '.join(rep.nonstandard)}")
    if entry.affinity_type and entry.affinity_type.lower() not in {"kd", "kd_app"}:
        out.append(f"experimental value is a {entry.affinity_type}, not a Kd - dG_exp is approximate")
    return out


GMX_SHIM = """#!/bin/bash
# Generated by cg-us. CHAPERONg builds its own gmx command lines, so the options
# it has no flags for are injected here:
#   pdb2gmx  -sb/-lb  ring-closure window for cyclic peptides
#   mdrun    GPU offload; without it PME and the integrator stay on the CPU and
#            the GPU spends most of the step waiting
# Anything else is passed through untouched. Each injected set is dropped if
# this gmx build rejects it, so the run degrades instead of dying.
GMX={gmx}
PDB2GMX_EXTRA=({pdb2gmx_extra})
MDRUN_LADDER=({mdrun_ladder})
MDRUN_LADDER_EM=({mdrun_ladder_em})

# Energy minimisation uses a non-dynamical integrator, and gmx refuses PME,
# bonded and update offload for it. CHAPERONg always names that run "em".
is_minimisation() {{
    local prev=""
    local a
    for a in "$@"; do
        if [ "$prev" = "-deffnm" ]; then
            case "$(basename "$a")" in em|minim|steep|em_*) return 0 ;; esac
        fi
        prev="$a"
    done
    return 1
}}

rejected() {{
    grep -qiE "unknown command-line option|invalid command-line option|not supported|does not support|incompatible|cannot be used|cannot compute|inconsistency in user input" "$1"
}}

# CHAPERONg puts -nb gpu on the command line itself when launched with -g, and
# gmx refuses an option given twice, so only the missing halves are added.
has_opt() {{
    local want="$1"; shift
    local a
    for a in "$@"; do [ "$a" = "$want" ] && return 0; done
    return 1
}}

filter_level() {{
    local level="$1"; shift
    local -a toks=($level) out=()
    local i=0
    while [ $i -lt ${{#toks[@]}} ]; do
        if ! has_opt "${{toks[$i]}}" "$@"; then
            out+=("${{toks[$i]}}" "${{toks[$((i+1))]}}")
        fi
        i=$((i+2))
    done
    printf '%s' "${{out[*]}}"
}}

case "$1" in
pdb2gmx)
    if [ -t 0 ]; then
        exec "$GMX" "$@" "${{PDB2GMX_EXTRA[@]}}"
    fi
    stdin_buf=$(mktemp) || exit 1
    out_buf=$(mktemp) || exit 1
    trap 'rm -f "$stdin_buf" "$out_buf"' EXIT
    cat > "$stdin_buf"
    if "$GMX" "$@" "${{PDB2GMX_EXTRA[@]}}" < "$stdin_buf" > "$out_buf" 2>&1; then
        cat "$out_buf"; exit 0
    fi
    if rejected "$out_buf"; then
        echo "cg-us: gmx rejects ${{PDB2GMX_EXTRA[*]}}; retrying pdb2gmx without them" >&2
        "$GMX" "$@" < "$stdin_buf"
        exit $?
    fi
    cat "$out_buf"; exit 1
    ;;
mdrun)
    out_buf=$(mktemp) || exit 1
    trap 'rm -f "$out_buf"' EXIT
    previous=""
    tried=0
    if is_minimisation "$@"; then
        ladder=("${{MDRUN_LADDER_EM[@]}}")
    else
        ladder=("${{MDRUN_LADDER[@]}}")
    fi
    for level in "${{ladder[@]}}"; do
        extra=$(filter_level "$level" "$@")
        # a level that adds nothing new to what caller already passed
        if [ "$tried" -eq 1 ] && [ "$extra" = "$previous" ]; then continue; fi
        previous="$extra"
        tried=1
        "$GMX" "$@" $extra 2>&1 | tee "$out_buf"
        rc=${{PIPESTATUS[0]}}
        [ "$rc" -eq 0 ] && exit 0
        if ! rejected "$out_buf"; then exit "$rc"; fi
        echo "cg-us: mdrun rejected '$extra', stepping down the offload ladder" >&2
    done
    exit 1
    ;;
*)
    exec "$GMX" "$@"
    ;;
esac
"""


def _mdrun_ladder(proto: Protocol, dynamical: bool = True) -> str:
    from .backends.base import gpu_offload_ladder
    levels = [" ".join(flags) for flags in gpu_offload_ladder(proto, dynamical=dynamical)]
    return " ".join(shlex.quote(l) for l in levels)


def write_gmx_shim(rdir: Path, proto: Protocol, lb: float | None = None) -> Path:
    extra = ["-sb", str(proto.prep.cyclic_sb),
             "-lb", str(proto.prep.cyclic_lb if lb is None else lb)]
    if proto.prep.heavy_hydrogens:
        extra.append("-heavyh")

    shim = rdir / "gmx_cgus.sh"
    shim.write_text(GMX_SHIM.format(
        gmx=shlex.quote(proto.run.gmx),
        pdb2gmx_extra=" ".join(extra),
        mdrun_ladder=_mdrun_ladder(proto),
        mdrun_ladder_em=_mdrun_ladder(proto, dynamical=False),
    ))
    shim.chmod(0o755)
    return shim.resolve()


TDB_SECTIONS = {"delete", "replace", "add", "impropers", "dihedrals", "angles", "bonds"}


def forcefield_dirs(source: str | Path) -> list[Path]:
    """The .ff directories under `source`, or `source` itself if it is one.

    Passing the parent of charmm36-jul2022.ff used to link the parent into the
    work directory, where pdb2gmx cannot see it: it looks for *.ff in the
    current directory only. The run then silently fell back to $GMXLIB.
    """
    source = Path(source).resolve()
    if source.name.endswith(".ff"):
        return [source]
    return sorted(source.glob("*.ff")) or [source]


def prune_terminus_db(ffdir: Path, codes: set[str] | None = None) -> list[str]:
    """Drop terminus entries that shadow an amino acid by name.

    pdb2gmx merges every *.tdb in the force field and, without -ter, picks the
    entry whose name starts with the residue name. charmm36 ships ethers.n.tdb
    with an entry called MET1 (append a terminal methyl, adds atom C1), so any
    chain beginning with methionine died on "atom C1 not found in building
    block 1MET". We only ever build proteins, so the non-protein databases lose
    the entries that can collide.
    """
    from .structure import STANDARD_AA

    codes = codes or STANDARD_AA
    removed = []
    for tdb in sorted(ffdir.glob("*.tdb")):
        if tdb.name.startswith("aminoacids"):
            continue
        lines = tdb.read_text().splitlines()
        out, drop, changed = [], False, False
        for line in lines:
            head = line.strip()
            if head.startswith("[") and head.endswith("]"):
                name = head[1:-1].strip()
                if name.lower() not in TDB_SECTIONS:
                    drop = any(name.upper().startswith(c) for c in codes)
                    if drop:
                        removed.append(f"{tdb.name}:[{name}]")
                        changed = True
            if not drop:
                out.append(line)
        if changed:
            tdb.write_text("\n".join(out) + "\n")
    return removed


def materialise_forcefield(source: str | Path, root: Path) -> list[Path]:
    """One pruned copy per run tree; replicas link to it."""
    made = []
    for ff in forcefield_dirs(source):
        target = Path(root) / ff.name
        if not target.exists():
            shutil.copytree(ff, target, symlinks=True)
            removed = prune_terminus_db(target)
            if removed:
                print(f"[prep] {ff.name}: dropped terminus entries that shadow an amino "
                      f"acid ({', '.join(removed)})")
        made.append(target)
    return made


def _link_forcefield(source: str | Path, dest: Path) -> None:
    for ff in forcefield_dirs(source):
        link = dest / ff.name
        if link.exists() or link.is_symlink():
            continue
        try:
            os.symlink(Path(ff).resolve(), link, target_is_directory=True)
        except OSError:
            shutil.copytree(ff, link)


def write_run_script(rdir: Path, entry: Entry, proto: Protocol, replica: int) -> Path:
    script = rdir / "run.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"cd \"$(dirname \"$0\")\"\n"
        f"cg-us run --root {Path(rdir).parents[2]} --system {entry.name} --replica {replica}\n"
    )
    script.chmod(0o755)
    return script
