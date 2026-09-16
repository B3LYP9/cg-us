"""PDB handling: chain extraction, cyclisation detection, pull-axis alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WATER = {"HOH", "WAT", "DOD", "TIP3", "SOL"}
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "MSE", "SEC", "PYL",
}


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    element: str
    line: str


@dataclass
class ChainReport:
    chain: str
    n_atoms: int
    n_residues: int
    first_res: int
    last_res: int
    head_tail_distance: float | None
    cyclic: bool
    disulfides: list[tuple[int, int]]
    nonstandard: list[str]


def read_pdb(path: str | Path) -> list[Atom]:
    """Parse a PDB file, keeping the first model only.

    Predicted and ensemble structures ship several MODEL blocks; reading them
    all would stack every atom on top of itself and hand solvate a system with
    duplicated coordinates.
    """
    atoms: list[Atom] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        resname = line[17:20].strip()
        if resname in WATER:
            continue
        atoms.append(
            Atom(
                record=line[:6].strip(),
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                resname=resname,
                chain=line[21].strip() or "_",
                resseq=int(line[22:26]),
                icode=line[26].strip(),
                xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                element=(line[76:78].strip() or line[12:16].strip()[:1]),
                line=line,
            )
        )
    if not atoms:
        raise ValueError(f"no atoms parsed from {path}")
    return atoms


def chains(atoms: list[Atom]) -> list[str]:
    seen: list[str] = []
    for a in atoms:
        if a.chain not in seen:
            seen.append(a.chain)
    return seen


def select(atoms: list[Atom], chain: str) -> list[Atom]:
    picked = [a for a in atoms if a.chain == chain]
    if not picked:
        raise ValueError(f"chain '{chain}' not present (available: {', '.join(chains(atoms))})")
    return picked


def com(atoms: list[Atom]) -> np.ndarray:
    return np.mean([a.xyz for a in atoms], axis=0)


def residues(atoms: list[Atom]) -> list[tuple[int, str, list[Atom]]]:
    out: list[tuple[int, str, list[Atom]]] = []
    for a in atoms:
        key = (a.resseq, a.icode)
        if not out or (out[-1][0], out[-1][1]) != key:
            out.append((a.resseq, a.icode, [a]))
        else:
            out[-1][2].append(a)
    return out


def describe_chain(atoms: list[Atom], chain: str,
                   cyclic_min: float = 0.5, cyclic_max: float = 4.0) -> ChainReport:
    """Cyclisation is judged by the same rule pdb2gmx uses.

    GROMACS closes a backbone ring when the terminal N-C distance falls between
    ``-sb`` and ``-lb``; both bounds are passed here in angstrom so the report
    says exactly what pdb2gmx is going to do with the chain.
    """
    sel = select(atoms, chain)
    res = residues(sel)
    head_tail = _head_tail_distance(res)
    nonstd = sorted({a.resname for a in sel if a.resname not in STANDARD_AA})
    return ChainReport(
        chain=chain,
        n_atoms=len(sel),
        n_residues=len(res),
        first_res=res[0][0],
        last_res=res[-1][0],
        head_tail_distance=head_tail,
        cyclic=head_tail is not None and cyclic_min < head_tail < cyclic_max,
        disulfides=find_disulfides(sel),
        nonstandard=nonstd,
    )


def _head_tail_distance(res) -> float | None:
    n_term = next((a for a in res[0][2] if a.name == "N"), None)
    c_term = next((a for a in res[-1][2] if a.name == "C"), None)
    if n_term is None or c_term is None:
        return None
    return float(np.linalg.norm(n_term.xyz - c_term.xyz))


def find_disulfides(atoms: list[Atom], cutoff: float = 2.5) -> list[tuple[int, int]]:
    sg = [a for a in atoms if a.name == "SG"]
    pairs = []
    for i, a in enumerate(sg):
        for b in sg[i + 1:]:
            if np.linalg.norm(a.xyz - b.xyz) < cutoff:
                pairs.append((a.resseq, b.resseq))
    return pairs


def rotation_to_z(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix taking `vector` onto +z."""
    v = vector / np.linalg.norm(vector)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        return np.eye(3) if v[2] > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    c = float(np.dot(v, z))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def orient_for_pull(atoms: list[Atom], target: str, binder: str) -> tuple[list[Atom], dict]:
    """Rotate the complex so that target-COM -> binder-COM points along +z."""
    t, b = select(atoms, target), select(atoms, binder)
    axis = com(b) - com(t)
    d0 = float(np.linalg.norm(axis))
    rot = rotation_to_z(axis)
    origin = com(t)

    rotated = []
    for a in atoms:
        moved = rot @ (a.xyz - origin)
        rotated.append(Atom(a.record, a.serial, a.name, a.resname, a.chain, a.resseq,
                            a.icode, moved, a.element, a.line))

    coords = np.array([a.xyz for a in rotated])
    shift = -coords.min(axis=0) + 1.0
    for a in rotated:
        a.xyz = a.xyz + shift

    info = {
        "com_distance_nm": d0 / 10.0,
        "extent_nm": ((coords.max(axis=0) - coords.min(axis=0)) / 10.0).tolist(),
    }
    return rotated, info


def write_pdb(atoms: list[Atom], path: str | Path, chain_order: list[str] | None = None) -> None:
    order = chain_order or chains(atoms)
    out: list[str] = []
    serial = 1
    for ch in order:
        prev_res = None
        for a in (x for x in atoms if x.chain == ch):
            out.append(_format_atom(a, serial))
            serial += 1
            prev_res = a.resseq
        out.append(f"TER   {serial:5d}      {'':3s} {ch}{prev_res or 0:4d}")
        serial += 1
    out.append("END")
    Path(path).write_text("\n".join(out) + "\n")


def _format_atom(a: Atom, serial: int) -> str:
    name = a.name if len(a.name) >= 4 else f" {a.name:<3s}"
    return (
        f"ATOM  {serial:5d} {name:<4s} {a.resname:>3s} {a.chain:1s}{a.resseq:4d}{a.icode or ' ':1s}   "
        f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}  1.00  0.00          {a.element:>2s}"
    )


def read_gro_box(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    return np.array([float(x) for x in lines[-1].split()[:3]])


def read_gro_coords(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    n = int(lines[1])
    return np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in lines[2:2 + n]])
