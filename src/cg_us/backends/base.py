"""Shared plumbing for the run backends."""

from __future__ import annotations

import json
import os
import re
import math
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Protocol
from ..manifest import Entry
from ..mdp import PULL_GROUPS
from .. import structure, topology


class GmxError(RuntimeError):
    pass


FATAL_RE = re.compile(r"^(Fatal error|Program:|ERROR \d+|Internal error|Assertion failed)", re.M)


def error_excerpt(output: str, tail: int = 40) -> str:
    """The block gmx actually complained about, not the last 40 lines.

    gmx writes diagnostics to stderr unbuffered while stdout is block-buffered,
    so in the merged stream the fatal error lands in the middle and a plain tail
    shows unrelated progress output. Three failures in this project were
    diagnosed from the log file rather than from the exception because of that.
    """
    lines = output.splitlines()
    hits = [i for i, l in enumerate(lines) if FATAL_RE.match(l)]
    if not hits:
        return "\n".join(lines[-tail:])
    start = max(0, hits[0] - 6)
    end = min(len(lines), hits[-1] + 12)
    block = lines[start:end]
    if end < len(lines):
        block.append(f"... ({len(lines) - end} more lines in gmx.log)")
    return "\n".join(block)


@dataclass
class Gmx:
    exe: str = "gmx"
    cwd: Path = Path(".")
    log: Path | None = None

    def run(self, args: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.exe, *args]
        self._log(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        proc = subprocess.run(
            cmd, cwd=self.cwd, input=stdin, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._log(proc.stdout or "")
        if check and proc.returncode != 0:
            raise GmxError(f"{' '.join(cmd[:2])} failed (rc={proc.returncode}):\n"
                           + error_excerpt(proc.stdout or ""))
        return proc

    def _fail(self, cmd: list[str], proc) -> GmxError:
        return GmxError(f"{' '.join(cmd[:2])} failed (rc={proc.returncode}):\n"
                        + error_excerpt(proc.stdout or ""))

    def mdrun(self, deffnm: str, proto: Protocol, extra: list[str] | None = None,
              slot: int = 0, workers: int = 1, dynamical: bool = True) -> None:
        args = ["mdrun", "-deffnm", deffnm, "-cpo", f"{deffnm}.cpt"]
        args += thread_flags(proto, slot=slot, workers=workers)
        if proto.run.gpu_id:
            args += ["-gpu_id", proto.run.gpu_id]
        args += extra or []

        for offload in gpu_offload_ladder(proto, dynamical=dynamical):
            proc = self.run(args + offload, check=False)
            if proc.returncode == 0:
                return
            if not _offload_rejected(proc.stdout or ""):
                raise GmxError(f"mdrun {deffnm} failed (rc={proc.returncode}):\n"
                               + error_excerpt(proc.stdout or ""))
            self._log(f"\n; offload {' '.join(offload) or '(cpu)'} rejected, stepping down\n")
        raise GmxError(f"mdrun {deffnm} failed with every offload combination")

    def _log(self, text: str) -> None:
        if self.log:
            with open(self.log, "a") as fh:
                fh.write(text)


def available_cores(proto: Protocol) -> int:
    return proto.run.cpu_cores or os.cpu_count() or 1


def threads_per_worker(proto: Protocol, workers: int = 1) -> int:
    """Split the core budget across concurrently running windows.

    Without this every worker would ask for run.ntomp threads and the node ends
    up oversubscribed, which is slower than running the windows one at a time.
    """
    cores = available_cores(proto)
    if workers <= 1:
        return proto.run.ntomp or cores
    share = max(1, cores // workers)
    return min(proto.run.ntomp, share) if proto.run.ntomp else share


def thread_flags(proto: Protocol, slot: int = 0, workers: int = 1) -> list[str]:
    ntomp = threads_per_worker(proto, workers)
    flags = ["-ntmpi", str(proto.run.ntmpi or 1), "-ntomp", str(ntomp)]
    if workers > 1 and proto.run.pin_threads:
        flags += ["-pin", "on", "-pinoffset", str(slot * ntomp), "-pinstride", "1"]
    return flags


def gpu_offload_ladder(proto: Protocol, dynamical: bool = True) -> list[list[str]]:
    """Offload combinations from fastest to most conservative.

    GPU-resident mode is the big win for small umbrella windows, but support
    depends on the build and on what the .mdp asks for, so each step down is
    tried only when mdrun explicitly rejects the one above it.

    Energy minimisation is not a dynamical integrator: PME, bonded and update
    offload are all refused for `steep`, so only the non-bonded kernels move.
    """
    if not proto.run.gpu:
        return [[]]
    if not dynamical:
        return [["-nb", "gpu"], []]
    full = ["-nb", "gpu"]
    if proto.run.gpu_pme:
        full += ["-pme", "gpu", "-pmefft", "gpu"]
    if proto.run.gpu_bonded:
        full += ["-bonded", "gpu"]

    ladder = []
    if proto.run.gpu_update:
        ladder.append(full + ["-update", "gpu"])
    ladder.append(full)
    if full != ["-nb", "gpu"]:
        ladder.append(["-nb", "gpu"])
    return ladder


def _offload_rejected(output: str) -> bool:
    return bool(re.search(
        r"(?i)(not supported|does not support|cannot be used|cannot compute|"
        r"incompatible|unsupported|inconsistency in user input|"
        r"unknown command-line option|invalid command-line option|"
        r"requires.*(GPU|rank))", output))


@dataclass
class RunContext:
    entry: Entry
    proto: Protocol
    workdir: Path
    replica: int
    started: float = field(default_factory=time.time)

    @property
    def prefix(self) -> str:
        return self.entry.name

    @property
    def gmx(self) -> Gmx:
        return Gmx(self.proto.run.gmx, self.workdir, self.workdir / "gmx.log")


def build_index(ctx: RunContext, gro: str, top: str = "topol.top", out: str = "index.ndx") -> Path:
    gmx = ctx.gmx
    gmx.run(["make_ndx", "-f", gro, "-o", out], stdin="q\n")
    groups = topology.pull_group_indices(ctx.workdir / top, ctx.entry.target,
                                        ctx.entry.binder, chain_order(ctx))
    topology.append_index_groups(ctx.workdir / out, groups)
    return ctx.workdir / out


def setup_restraints(ctx: RunContext, top: str = "topol.top") -> list[str]:
    top_path = ctx.workdir / top
    mapping = topology.chain_moltype_groups(top_path, ctx.entry.target, ctx.entry.binder,
                                            chain_order(ctx))
    added = []
    if ctx.proto.smd.restrain_target:
        added += _restrain_group(ctx, top_path, mapping["Target"], "target",
                                 "POSRES_TARGET", "backbone", None)
    if ctx.entry.restrain_binder:
        added += _restrain_group(ctx, top_path, mapping["Binder"], "binder",
                                 "POSRES_BINDER", "ca", 200.0)
    return added


def _restrain_group(ctx: RunContext, top_path: Path, moltypes: list[str], label: str,
                    define: str, selection: str, k: float | None) -> list[str]:
    """Restrain every molecule type of the group, not just the first.

    A group spanning several chains has to be pinned chain by chain: each chain
    is its own [ moleculetype ] and takes its own position_restraints block. Pin
    one and leave the others free and the unpinned chains drift, the group COM
    drifts with them, and the reaction coordinate stops measuring the
    separation it is named after.
    """
    added = []
    for mt in moltypes:
        itp = f"posre_{label}.itp" if len(moltypes) == 1 else f"posre_{label}_{mt}.itp"
        kw = {} if k is None else {"k": k}
        topology.restraint_itp(top_path, mt, ctx.workdir / itp, selection=selection, **kw)
        where = topology.wire_restraints(top_path, mt, itp, define)
        added.append(f"{define} -> {mt} in {Path(where).name}")
    return added


def verify_pull_range(ctx: RunContext, gro: str = "npt.gro", tpr: str = "npt.tpr") -> dict:
    """Re-check the pull budget against the equilibrated system, not the crystal.

    The box is fixed at editconf time from the crystal COM distance, but the
    complex drifts during solvation and equilibration, and the outermost window
    then rattles a few sigma around its centre. Both push the COM separation past
    what prep assumed, and gmx aborts mid-window. Here the real numbers are
    measured and the pull is shortened if the box cannot take the full length.
    """
    wd = ctx.workdir
    u = ctx.proto.umbrella
    if not (wd / gro).exists():
        return {}

    select = 'com of group "Target" plus com of group "Binder"'
    proc = ctx.gmx.run(["distance", "-s", tpr, "-f", gro, "-n", "index.ndx",
                        "-select", select, "-oall", "npt_distance.xvg"], check=False)
    if proc.returncode != 0:
        return {}

    from ..xvg import read_xvg
    data, _ = read_xvg(wd / "npt_distance.xvg")
    if data.size == 0:
        return {}
    d_npt = float(data[-1, 1])
    box_z = float(Path(wd, gro).read_text().splitlines()[-1].split()[2])

    kT = 8.314462618e-3 * ctx.proto.temperature
    sigma = math.sqrt(kT / u.k) if u.k > 0 else 0.0
    fluctuation = 4 * sigma
    allowed = 0.49 * box_z * (1 - ctx.proto.prep.pull_box_margin) - d_npt - fluctuation

    info = {
        "com_distance_after_npt_nm": round(d_npt, 3),
        "box_z_after_npt_nm": round(box_z, 3),
        "window_sigma_nm": round(sigma, 4),
        "max_distance_requested": u.max_distance,
        "max_distance_effective": round(min(u.max_distance, max(0.0, allowed)), 3),
    }

    if info["max_distance_effective"] < u.max_distance - 1e-3:
        _shorten_pull(ctx, info["max_distance_effective"])
        print(f"      [warn] box takes only {info['max_distance_effective']:.2f} nm of pulling "
              f"(d after NPT {d_npt:.2f} nm, box_z {box_z:.2f} nm); "
              "md_pull.mdp shortened to match")
    (wd / "pull_range.json").write_text(json.dumps(info, indent=2))
    return info


def _shorten_pull(ctx: RunContext, distance: float) -> None:
    mdp = ctx.workdir / "md_pull.mdp"
    if not mdp.exists():
        return
    steps = max(1, int(distance / ctx.proto.smd.rate / ctx.proto.smd.dt))
    lines = []
    for line in mdp.read_text().splitlines():
        if line.split("=")[0].strip() == "nsteps":
            line = f"nsteps                   = {steps}"
        lines.append(line)
    mdp.write_text("\n".join(lines) + "\n")


def effective_max_distance(ctx: RunContext) -> float:
    f = ctx.workdir / "pull_range.json"
    if f.exists():
        return float(json.loads(f.read_text()).get("max_distance_effective",
                                                   ctx.proto.umbrella.max_distance))
    return ctx.proto.umbrella.max_distance


def window_files(workdir: Path) -> tuple[list[str], list[str]]:
    tprs = _read_list(workdir / "tpr_files.dat")
    pullf = _read_list(workdir / "pullf_files.dat")
    return tprs, pullf


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def write_window_lists(workdir: Path, records: list[dict]) -> None:
    (workdir / "tpr_files.dat").write_text("\n".join(r["tpr"] for r in records) + "\n")
    (workdir / "pullf_files.dat").write_text("\n".join(r["pullf"] for r in records) + "\n")
    (workdir / "pullx_files.dat").write_text("\n".join(r["pullx"] for r in records) + "\n")


PBCATOM_RE = re.compile(r"^pull_group(\d)_pbcatom\b", re.M)


def chain_order(ctx: RunContext) -> list[str] | None:
    info = ctx.workdir.parent / "prep.json"
    if not info.exists():
        return None
    return json.loads(info.read_text()).get("chain_order")


def verify_cyclisation(ctx: RunContext, top: str = "topol.top") -> list[str]:
    """Abort if a chain prep called cyclic came out of pdb2gmx linear.

    The failure is silent: GROMACS drops the ring bond and caps both ends
    instead, leaving a charged N-terminus on top of a carboxylate. EM then
    reports an infinite force and mdrun dies inside CUDA several stages later,
    far from the cause.
    """
    info = json.loads((ctx.workdir.parent / "prep.json").read_text())
    reports = info.get("chain_reports", {})
    expected = {ch: r.get("cyclic") for ch, r in reports.items()}
    if not any(expected.values()):
        return []
    mapping = topology.chain_moltype_groups(ctx.workdir / top, ctx.entry.target,
                                            ctx.entry.binder, chain_order(ctx))
    labels = {"Target": ctx.entry.target_chains, "Binder": ctx.entry.binder_chains}
    checked, broken = [], []
    for label, mt_names in mapping.items():
        for ch, mt_name in zip(labels[label], mt_names):
            if not expected.get(ch):
                continue
            checked.append(f"{label} (chain {ch}, {mt_name})")
            mt = topology.moltype(ctx.workdir / top, mt_name)
            if mt is None or not topology.is_cyclic(mt):
                broken.append(f"{label} (chain {ch}, {mt_name})")
    if broken:
        raise GmxError(
            "pdb2gmx did not close the backbone ring of " + ", ".join(broken) + ".\n"
            "The chain was written with charged termini a bond length apart, which "
            "gives an infinite force in EM.\nCheck that the cyclic chain is first in "
            f"{ctx.prefix}.pdb (GROMACS issue 5091) and that the N-C distance is "
            "inside the -sb/-lb window."
        )
    return [f"ring closed: {b}" for b in checked]


FMAX_RE = re.compile(r"^Maximum force\s*=\s*(\S+)", re.M)


def check_minimisation(ctx: RunContext, log: str = "em.log") -> float:
    """Stop at EM instead of carrying a broken topology into the GPU."""
    path = ctx.workdir / log
    if not path.exists():
        return float("nan")
    found = FMAX_RE.findall(path.read_text())
    if not found:
        return float("nan")
    fmax = float(found[-1])
    limit = ctx.proto.prep.em_fmax_max
    if not math.isfinite(fmax) or fmax > limit:
        raise GmxError(
            f"energy minimisation ended at Fmax = {fmax} kJ/mol/nm (limit {limit}).\n"
            "That is a broken topology, not a tight contact - overlapping termini on a "
            "chain that failed to cyclise are the usual cause.\nInspect em.log and "
            "topol_*.itp before rerunning; NVT on these coordinates crashes the GPU."
        )
    return fmax


def pull_pbcatoms(ctx: RunContext, gro: str = "solv_ions.gro",
                  index: str = "index.ndx") -> dict[str, int]:
    """Pick a spatially central reference atom for each pull group.

    gmx defaults to the middle atom *by number*, which for an elongated binder
    sits nowhere near the middle in space; grompp then refuses the run unless a
    reference atom is named explicitly. Choosing the atom closest to the group
    centre also keeps the largest intra-group distance as small as the geometry
    allows.
    """
    groups = topology.read_index_groups(ctx.workdir / index)
    coords = structure.read_gro_coords(ctx.workdir / gro)
    box = structure.read_gro_box(ctx.workdir / gro)
    picked: dict[str, int] = {}
    for label in PULL_GROUPS:
        idx = np.array(groups[label]) - 1
        xyz = coords[idx]
        d = xyz - xyz.mean(axis=0)
        d -= box * np.round(d / box)
        centre = int(np.argmin(np.linalg.norm(d, axis=1)))
        picked[label] = int(idx[centre]) + 1
        _check_group_fits_box(label, xyz, xyz[centre], box)
    return picked


def _check_group_fits_box(label: str, xyz: np.ndarray, ref: np.ndarray,
                          box: np.ndarray) -> None:
    """A pull group must fit within half a box vector of its reference atom.

    gmx rebuilds the group COM by min-imaging every atom against the reference
    atom, so an atom further away than half a box vector is imaged to the wrong
    side and the COM lands somewhere between the two halves - silently, with a
    reaction coordinate that is simply wrong. Coordinates are read as editconf
    left them, i.e. the group is contiguous, so the raw spread is the real one.
    A single chain almost never trips this; a multi-chain group is a different
    size class, and the box was sized for the complex extent plus a margin.
    """
    reach = np.abs(xyz - ref).max(axis=0)
    if np.any(reach >= 0.5 * box):
        worst = int(np.argmax(reach / box))
        raise GmxError(
            f"pull group {label} reaches {reach[worst]:.2f} nm from its reference atom "
            f"along {'xyz'[worst]}, which is past half the box ({box[worst]:.2f} nm).\n"
            "gmx would min-image the far atoms to the wrong side, so the group COM - and "
            "the whole reaction coordinate - would be wrong without any warning.\n"
            "Enlarge the box (prep.box_edge_xy / prep.box_edge_z), or keep only the "
            "chains that actually form the interface in the pull group."
        )


def write_pull_pbcatoms(ctx: RunContext, atoms: dict[str, int],
                        mdps: tuple[str, ...] = ("md_pull.mdp", "npt_umbrella.mdp",
                                                 "md_umbrella.mdp")) -> None:
    lines = "\n".join(f"pull_group{i + 1}_pbcatom = {atoms[g]}"
                      for i, g in enumerate(PULL_GROUPS))
    for name in mdps:
        path = ctx.workdir / name
        if not path.exists():
            continue
        text = "\n".join(l for l in path.read_text().splitlines()
                         if not PBCATOM_RE.match(l))
        anchor = "pull_pbc_ref_prev_step_com"
        if anchor in text:
            text = text.replace(anchor, lines + "\n" + anchor)
        else:
            text = text.rstrip() + "\n" + lines + "\npull_pbc_ref_prev_step_com = yes\n"
        path.write_text(text)
