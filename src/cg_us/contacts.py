"""Interface contacts along the unbinding path, classified as in GetContacts.

The criteria follow https://getcontacts.github.io/interactions.html:

  hydrogen bond   |D-A| < 3.5 A and 180 - angle(A,H,D) < 70 deg, D and A any N/O
                  (split into backbone-backbone, sidechain-backbone, sidechain-sidechain)
  salt bridge     |A-C| < 4.0 A, Asp/Glu carboxylate O to Lys NZ / Arg NE,NH1,NH2
  pi-cation       ring centroid to Lys NZ / Arg CZ < 6.0 A, angle(normal, centroid->cation) < 60
  pi-stacking     centroids < 7.0 A, angle(n1,n2) < 30, angle(n,axis) < 45 for both rings
  t-stacking      centroids < 5.0 A, 60 < angle(n1,n2) < 90, angle(n,axis) < 45 for one ring
                  (the page lists both; with perpendicular normals that cannot hold for both,
                  so the face ring's normal must point at the other ring's centroid)
  hydrophobic     C/S atoms of A C F G I L M P V W closer than Rvdw+Rvdw+0.5 A
  vdw             any two heavy atoms closer than Rvdw+Rvdw+0.5 A

Contacts are counted per residue pair and type (one per pair per frame), between the
target group and the binder group of the pull. Water bridges are not detected.

Two trajectories are analysed: the umbrella windows (equilibrium sampling at a fixed
distance, one state per window) and the steered-MD pull (frames binned by distance,
non-equilibrium). The geometry code needs only numpy and scipy; reading the
trajectories needs MDAnalysis (`pip install 'cg-us[contacts]'`).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .xvg import read_xvg

TYPES = ("hbond_bb_bb", "hbond_sc_bb", "hbond_sc_sc", "salt_bridge", "pi_cation",
         "pi_stacking", "t_stacking", "hydrophobic", "vdw")
ANY = "any"                       # union of all types: "these two residues touch"

HBOND_DIST = 3.5
HBOND_MIN_ANGLE = 110.0           # angle(A,H,D); GetContacts: 180 - angle < 70
SALT_DIST = 4.0
PI_CATION_DIST, PI_CATION_ANGLE = 6.0, 60.0
PI_STACK_DIST, PI_STACK_ANGLE, PI_AXIS_ANGLE = 7.0, 30.0, 45.0
T_STACK_DIST, T_STACK_ANGLES = 5.0, (60.0, 90.0)
VDW_MARGIN = 0.5
END_SPAN_NM = 0.3                 # the last stretch of the path that must be contact-free
PAIR_CUTOFF = 4.6                 # covers every atom-pair criterion above (S-S vdw is 4.1)

RVDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
HYDROPHOBIC_RES = {"ALA", "CYS", "PHE", "GLY", "ILE", "LEU", "MET", "PRO", "VAL", "TRP"}
ANION = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
CATION_SALT = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
CATION_PI = {("LYS", "NZ"), ("ARG", "CZ")}
BACKBONE_HETERO = {"N", "O", "OT1", "OT2", "OXT"}
RINGS = {"PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
         "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
         "TRP": [["CG", "CD1", "NE1", "CE2", "CD2"],
                 ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]]}
# CHARMM names for the ring atoms above; the same names are used by amber/gromos protein
# topologies for these residues, so no renaming is attempted


@dataclass
class System:
    """Static part of the analysis: which atoms are what. Indices are local (0..n)."""
    n: int
    side: np.ndarray                 # 0 target, 1 binder
    res: np.ndarray                  # residue index per atom
    elem: np.ndarray                 # 'C','N','O','S','H'
    heavy: np.ndarray                # bool
    backbone: np.ndarray             # bool: backbone N/O
    donor_h: np.ndarray              # (n,3) hydrogen indices attached to a donor, -1 padded
    acceptor: np.ndarray             # bool
    anion: np.ndarray
    cation_salt: np.ndarray
    cation_pi: np.ndarray
    hydrophobic: np.ndarray
    rvdw: np.ndarray
    rings: list = field(default_factory=list)   # list of (atom_idx array, side, res)


def element(name: str) -> str:
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "X"


def build_system(names, resnames, resindices, side, ref_pos, box=None) -> System:
    """names/resnames/resindices/side: arrays over the local atoms (target then binder);
    ref_pos: (n,3) reference coordinates used only to attach hydrogens to their donors."""
    names = np.asarray(names)
    resnames = np.asarray(resnames)
    resindices = np.asarray(resindices)
    n = len(names)
    elem = np.array([element(x) for x in names])
    heavy = elem != "H"
    backbone = heavy & np.isin(names, list(BACKBONE_HETERO))
    idx = {}
    for i in range(n):
        idx[(resindices[i], names[i])] = i

    donor_h = np.full((n, 3), -1, dtype=int)
    is_donor = np.zeros(n, bool)
    by_res = defaultdict(list)
    for i in range(n):
        by_res[resindices[i]].append(i)
    for members in by_res.values():
        heavies = [i for i in members if heavy[i] and elem[i] in ("N", "O")]
        if not heavies:
            continue
        hp = ref_pos[heavies]
        for h in (i for i in members if not heavy[i]):
            d = np.linalg.norm(hp - ref_pos[h], axis=1)
            k = int(np.argmin(d))
            if d[k] < 1.3:
                parent = heavies[k]
                slot = int(np.sum(donor_h[parent] >= 0))
                if slot < 3:
                    donor_h[parent, slot] = h
                    is_donor[parent] = True

    def flag(pairs):
        return np.array([(resnames[i], names[i]) in pairs for i in range(n)])

    rings = []
    for r, members in by_res.items():
        rn = resnames[members[0]]
        for ring_names in RINGS.get(rn, []):
            atoms = [idx.get((r, a)) for a in ring_names]
            if all(a is not None for a in atoms):
                rings.append((np.array(atoms), int(side[members[0]]), int(r)))

    rvdw = np.array([RVDW.get(e, 1.7) for e in elem])
    hydrophobic = np.array([heavy[i] and elem[i] in ("C", "S") and resnames[i] in HYDROPHOBIC_RES
                            for i in range(n)])
    return System(n=n, side=np.asarray(side), res=resindices, elem=elem, heavy=heavy,
                  backbone=backbone, donor_h=donor_h,
                  acceptor=heavy & np.isin(elem, ["N", "O"]),
                  anion=flag(ANION), cation_salt=flag(CATION_SALT), cation_pi=flag(CATION_PI),
                  hydrophobic=hydrophobic, rvdw=rvdw, rings=rings)


def _mi(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    return v - box * np.round(v / box)


def _angle_folded(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between two axes in degrees, 0..90 (the sign of a ring normal is arbitrary)."""
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    c = np.abs(np.sum(u * v, axis=-1)) / np.maximum(nu * nv, 1e-12)
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def _ring_geometry(sys: System, pos: np.ndarray, box: np.ndarray):
    cent, norm, side, res = [], [], [], []
    for size in sorted({len(r[0]) for r in sys.rings}):
        group = [r for r in sys.rings if len(r[0]) == size]
        p = pos[np.stack([r[0] for r in group])]                 # (m, size, 3)
        p = p[:, :1] + _mi(p - p[:, :1], box)
        c = p.mean(axis=1)
        q = p - c[:, None]
        n = np.linalg.svd(q)[2][:, -1, :]
        cent.append(c)
        norm.append(n)
        side += [r[1] for r in group]
        res += [r[2] for r in group]
    if not cent:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, int), np.empty(0, int)
    return np.vstack(cent), np.vstack(norm), np.array(side), np.array(res)


def frame_contacts(sys: System, pos: np.ndarray, box: np.ndarray) -> dict[str, set]:
    """{type: {(target_residue, binder_residue), ...}} for one frame.

    pos: (n,3) Angstrom, box: (3,) orthorhombic edges in Angstrom."""
    out: dict[str, set] = {t: set() for t in TYPES}
    w = np.mod(pos, box)
    w = np.where(w >= box, 0.0, w)
    ht = np.where(sys.heavy & (sys.side == 0))[0]
    hb = np.where(sys.heavy & (sys.side == 1))[0]
    if len(ht) and len(hb):
        tt = cKDTree(w[ht], boxsize=box)
        tb = cKDTree(w[hb], boxsize=box)
        m = tt.sparse_distance_matrix(tb, PAIR_CUTOFF, output_type="ndarray")
        ia, ib, d = ht[m["i"]], hb[m["j"]], m["v"]
    else:
        ia = ib = np.empty(0, int)
        d = np.empty(0)

    def add(kind, a, b, mask):
        for x, y in zip(sys.res[a[mask]], sys.res[b[mask]]):
            out[kind].add((int(x), int(y)))

    if len(d):
        near = d < sys.rvdw[ia] + sys.rvdw[ib] + VDW_MARGIN
        add("vdw", ia, ib, near)
        add("hydrophobic", ia, ib, near & sys.hydrophobic[ia] & sys.hydrophobic[ib])
        salt = (d < SALT_DIST) & ((sys.anion[ia] & sys.cation_salt[ib])
                                  | (sys.cation_salt[ia] & sys.anion[ib]))
        add("salt_bridge", ia, ib, salt)

        for don, acc in ((ia, ib), (ib, ia)):
            ok = (d < HBOND_DIST) & (sys.donor_h[don][:, 0] >= 0) & sys.acceptor[acc]
            for k in range(3):
                h = sys.donor_h[don, k]
                sel = ok & (h >= 0)
                if not sel.any():
                    continue
                hh = np.where(sel, h, 0)
                v1 = _mi(pos[don] - pos[hh], box)
                v2 = _mi(pos[acc] - pos[hh], box)
                cos = np.sum(v1 * v2, axis=1) / np.maximum(
                    np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
                good = sel & (np.degrees(np.arccos(np.clip(cos, -1, 1))) > HBOND_MIN_ANGLE)
                bb_d, bb_a = sys.backbone[don], sys.backbone[acc]
                for kind, mask in (("hbond_bb_bb", bb_d & bb_a),
                                   ("hbond_sc_bb", bb_d ^ bb_a),
                                   ("hbond_sc_sc", ~bb_d & ~bb_a)):
                    sel2 = good & mask
                    for x, y in zip(sys.res[ia[sel2]], sys.res[ib[sel2]]):
                        out[kind].add((int(x), int(y)))

    cent, norm, rside, rres = _ring_geometry(sys, pos, box)
    if len(cent):
        t = np.where(rside == 0)[0]
        b = np.where(rside == 1)[0]
        if len(t) and len(b):
            v = _mi(cent[b][None, :, :] - cent[t][:, None, :], box)          # (t,b,3)
            dist = np.linalg.norm(v, axis=2)
            n1 = norm[t][:, None, :]
            n2 = norm[b][None, :, :]
            a12 = _angle_folded(n1, n2)
            a1 = _angle_folded(n1, v)
            a2 = _angle_folded(n2, v)
            stack = (dist < PI_STACK_DIST) & (a12 < PI_STACK_ANGLE) & (a1 < PI_AXIS_ANGLE) & (a2 < PI_AXIS_ANGLE)
            tst = ((dist < T_STACK_DIST) & (a12 > T_STACK_ANGLES[0]) & (a12 <= T_STACK_ANGLES[1] + 1e-6)
                   & (np.minimum(a1, a2) < PI_AXIS_ANGLE))
            for kind, mask in (("pi_stacking", stack), ("t_stacking", tst)):
                for i, j in zip(*np.where(mask)):
                    out[kind].add((int(rres[t[i]]), int(rres[b[j]])))

        cat = np.where(sys.cation_pi)[0]
        for ring_side in (0, 1):
            r = np.where(rside == ring_side)[0]
            c = cat[sys.side[cat] != ring_side]
            if not len(r) or not len(c):
                continue
            v = _mi(pos[c][None, :, :] - cent[r][:, None, :], box)          # (r,c,3)
            dist = np.linalg.norm(v, axis=2)
            ang = _angle_folded(norm[r][:, None, :], v)
            for i, j in zip(*np.where((dist < PI_CATION_DIST) & (ang < PI_CATION_ANGLE))):
                a, bb = int(rres[r[i]]), int(sys.res[c[j]])
                out["pi_cation"].add((a, bb) if ring_side == 0 else (bb, a))
    return out


class Accumulator:
    """Frames of one state (a window, or a distance bin of the pull)."""

    def __init__(self):
        self.n = 0
        self.xi: list[float] = []
        self.type_count: Counter = Counter()
        self.pair_count: Counter = Counter()      # (target_res, binder_res, type)
        self.res_count: Counter = Counter()       # residue -> frames with any cross contact

    def add(self, contacts: dict[str, set], xi: float) -> None:
        self.n += 1
        self.xi.append(xi)
        union: set = set()
        for kind, pairs in contacts.items():
            self.type_count[kind] += len(pairs)
            for a, b in pairs:
                self.pair_count[(a, b, kind)] += 1
            union |= pairs
        self.type_count[ANY] += len(union)
        for a, b in union:
            self.pair_count[(a, b, ANY)] += 1
        for r in {x for p in union for x in p}:
            self.res_count[r] += 1

    def state(self) -> dict:
        n = max(self.n, 1)
        return {"xi": float(np.mean(self.xi)) if self.xi else float("nan"), "n_frames": self.n,
                "counts": {k: v / n for k, v in self.type_count.items()},
                "pairs": {k: v / n for k, v in self.pair_count.items()},
                "residues": {k: v / n for k, v in self.res_count.items()}}


def merge_states(states: list[dict]) -> dict:
    """Frame-weighted mean of several states (used for the bound state)."""
    total = sum(s["n_frames"] for s in states) or 1
    out = {"xi": sum(s["xi"] * s["n_frames"] for s in states) / total, "n_frames": total,
           "counts": defaultdict(float), "pairs": defaultdict(float), "residues": defaultdict(float)}
    for s in states:
        w = s["n_frames"] / total
        for field_ in ("counts", "pairs", "residues"):
            for k, v in s[field_].items():
                out[field_][k] += v * w
    return {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in out.items()}


def summarize(states: list[dict], bound_span_nm: float = 0.2, top: int = 10) -> dict:
    """Bound-state contact inventory, where contacts are lost, and what is left at the end."""
    states = sorted((s for s in states if s["n_frames"] and np.isfinite(s["xi"])),
                    key=lambda s: s["xi"])
    if not states:
        return {}
    xi0 = states[0]["xi"]
    bound = merge_states([s for s in states if s["xi"] <= xi0 + bound_span_nm])
    total0 = bound["counts"].get(ANY, 0.0)
    xi_half = None
    for s in states:
        if s["counts"].get(ANY, 0.0) < 0.5 * total0:
            xi_half = s["xi"]
            break
    xi_end = states[-1]["xi"]
    end = [s for s in states if s["xi"] >= xi_end - END_SPAN_NM]
    end_contacts = float(np.mean([s["counts"].get(ANY, 0.0) for s in end]))

    residues = {}
    for r, occ in bound["residues"].items():
        if occ < 0.5:
            continue
        last = max((s["xi"] for s in states if s["residues"].get(r, 0.0) >= 0.5), default=xi0)
        residues[r] = {"bound_occupancy": occ, "xi_last": last}
    hot = sorted(residues, key=lambda r: (-residues[r]["xi_last"], -residues[r]["bound_occupancy"]))[:top]
    return {"xi_bound": xi0, "bound_counts": {k: bound["counts"].get(k, 0.0) for k in TYPES + (ANY,)},
            "xi_half": xi_half, "contacts_at_end": end_contacts,
            "pull_complete": end_contacts < 1.0, "xi_end": xi_end,
            "interface_residues": len(residues), "residue_persistence": residues, "hotspots": hot}


# ---------------------------------------------------------------- trajectories


def read_index_groups(path: Path, names=("Target", "Binder")) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            current = line.strip("[] ").strip()
            groups[current] = []
        elif line and current:
            groups[current].extend(int(x) - 1 for x in line.split())
    missing = [n for n in names if n not in groups]
    if missing:
        raise ValueError(f"{path}: index groups {missing} not found")
    return {n: np.array(groups[n], dtype=int) for n in names}


def residue_labels(resnames, resids, resindices, side) -> dict[int, str]:
    """T:ARG25 / B:LYS3; with several chains in a group a segment number follows the prefix."""
    labels: dict[int, str] = {}
    for s, prefix in ((0, "T"), (1, "B")):
        seen, order = set(), []
        for i in np.where(side == s)[0]:
            r = int(resindices[i])
            if r not in seen:
                seen.add(r)
                order.append(i)
        segs, prev, seg = {}, None, 1
        for i in order:
            if prev is not None and resids[i] <= prev:
                seg += 1
            prev = resids[i]
            segs[int(resindices[i])] = seg
        multi = seg > 1
        for i in order:
            r = int(resindices[i])
            labels[r] = f"{prefix}{segs[r] if multi else ''}:{resnames[i]}{resids[i]}"
    return labels


def _import_mda():
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError("contact analysis reads trajectories with MDAnalysis; "
                           "run `pip install 'cg-us[contacts]'` (or `pip install MDAnalysis`)") from exc
    return mda


def load_system(workdir: Path):
    """(Universe, System, labels, atom index array) from em.gro + index.ndx."""
    mda = _import_mda()
    wd = Path(workdir)
    u = mda.Universe(str(wd / "em.gro"))
    groups = read_index_groups(wd / "index.ndx")
    atoms = np.concatenate([groups["Target"], groups["Binder"]])
    side = np.concatenate([np.zeros(len(groups["Target"]), int), np.ones(len(groups["Binder"]), int)])
    a = u.atoms[atoms]
    ref = a.positions.astype(float)
    system = build_system(a.names, a.resnames, a.resindices, side, ref)
    labels = residue_labels(a.resnames, a.resids, a.resindices, side)
    return u, system, labels, atoms


def _xi_of_frames(pullx: Path):
    data, _ = read_xvg(pullx)
    if data.size == 0 or data.shape[1] < 2:
        raise ValueError(f"{pullx}: no pull coordinate")
    return data[:, 0], data[:, 1]


def _iterate(u, atoms, system, xtc: Path, pullx: Path, stride: int, skip_ps: float):
    """Yield (xi_nm, contacts) for the kept frames of one trajectory."""
    t_x, xi = _xi_of_frames(pullx)
    u.load_new(str(xtc))
    if u.atoms.n_atoms < atoms.max() + 1:
        raise ValueError(f"{xtc}: {u.atoms.n_atoms} atoms, topology needs {atoms.max() + 1}")
    t0 = None
    for ts in u.trajectory[::stride]:
        t0 = ts.time if t0 is None else t0
        if ts.time - t0 < skip_ps or ts.time < t_x[0] - 1 or ts.time > t_x[-1] + 1:
            continue
        box = np.asarray(ts.dimensions[:3], dtype=float)
        if not np.allclose(ts.dimensions[3:], 90.0, atol=0.5):
            raise ValueError("contact analysis supports rectangular boxes only")
        yield float(np.interp(ts.time, t_x, xi)), frame_contacts(system, ts.positions[atoms].astype(float), box)


def analyse_windows(workdir: Path, proto, stride: int = 1, ctx=None) -> list[dict]:
    wd = Path(workdir)
    u, system, labels, atoms = ctx or load_system(wd)
    listing = wd / "pullx_files.dat"
    files = [wd / l.strip() for l in listing.read_text().splitlines() if l.strip()]
    states = []
    for f in files:
        xtc = f.with_name(f.name.replace("_pullx.xvg", ".xtc"))
        if not (f.exists() and xtc.exists()):
            continue
        acc = Accumulator()
        for xi, c in _iterate(u, atoms, system, xtc, f, stride, proto.umbrella.discard_ns * 1000.0):
            acc.add(c, xi)
        if acc.n:
            st = acc.state()
            st["name"] = f.name.replace("_pullx.xvg", "")
            states.append(st)
    return states


def analyse_smd(workdir: Path, proto, stride: int = 5, bin_nm: float = 0.1, ctx=None) -> list[dict]:
    wd = Path(workdir)
    u, system, labels, atoms = ctx or load_system(wd)
    xtc, pullx = wd / "pull.xtc", wd / "pullx.xvg"
    if not (xtc.exists() and pullx.exists()):
        return []
    bins: dict[int, Accumulator] = {}
    for xi, c in _iterate(u, atoms, system, xtc, pullx, stride, 0.0):
        bins.setdefault(int(np.floor(xi / bin_nm)), Accumulator()).add(c, xi)
    states = []
    for k in sorted(bins):
        st = bins[k].state()
        st["name"] = f"smd_bin{k}"
        states.append(st)
    return states


def _key(d: dict) -> dict:
    return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in d.items()}


def states_table(states: list[dict], labels: dict[int, str]):
    """(pairs, counts) DataFrames for one trajectory set."""
    import pandas as pd
    pairs, counts = [], []
    for i, s in enumerate(sorted(states, key=lambda s: s["xi"])):
        row = {"state": s.get("name", i), "xi_nm": round(s["xi"], 4), "n_frames": s["n_frames"]}
        counts.append({**row, **{k: round(s["counts"].get(k, 0.0), 3) for k in TYPES + (ANY,)}})
        for (a, b, kind), occ in s["pairs"].items():
            pairs.append({**row, "target_res": labels.get(a, a), "binder_res": labels.get(b, b),
                          "type": kind, "occupancy": round(occ, 3)})
    return pd.DataFrame(pairs), pd.DataFrame(counts)


def analyse_replica(workdir: Path, proto, sources=("windows", "smd"), stride_windows: int = 1,
                    stride_smd: int = 5) -> dict:
    """Write analysis/contacts_<source>_{pairs,counts}.csv and contacts_<source>.json."""
    wd = Path(workdir)
    out = wd / "analysis"
    out.mkdir(exist_ok=True)
    ctx = load_system(wd)
    labels = ctx[2]
    result = {}
    for source in sources:
        states = (analyse_windows(wd, proto, stride_windows, ctx=ctx) if source == "windows"
                  else analyse_smd(wd, proto, stride_smd, ctx=ctx))
        if not states:
            continue
        pairs, counts = states_table(states, labels)
        pairs.to_csv(out / f"contacts_{source}_pairs.csv", index=False)
        counts.to_csv(out / f"contacts_{source}_counts.csv", index=False)
        summary = summarize(states)
        summary["hotspots"] = [labels.get(r, str(r)) for r in summary["hotspots"]]
        summary["residue_persistence"] = {labels.get(r, str(r)): v
                                          for r, v in summary["residue_persistence"].items()}
        summary["source"] = source
        (out / f"contacts_{source}.json").write_text(json.dumps(summary, indent=2))
        result[source] = summary
    return result


# ---------------------------------------------------------------- across replicas


def summary_row(system: str, replica: int, summary: dict) -> dict:
    row = {"system": system, "replica": replica, "source": summary["source"],
           "xi_bound_nm": round(summary["xi_bound"], 3)}
    for k in TYPES + (ANY,):
        row[k] = round(summary["bound_counts"][k], 2)
    row.update(interface_residues=summary["interface_residues"],
               xi_half_nm=None if summary["xi_half"] is None else round(summary["xi_half"], 3),
               xi_end_nm=round(summary["xi_end"], 3),
               contacts_at_end=round(summary["contacts_at_end"], 2),
               pull_complete=summary["pull_complete"],
               hotspots=";".join(summary["hotspots"]))
    return row


def consensus(hotspot_lists: list[list[str]], min_replicas: int = 2) -> list[str]:
    """Residues that are among the last to let go in at least `min_replicas` replicas."""
    n = Counter(r for lst in hotspot_lists for r in set(lst))
    need = min(min_replicas, len(hotspot_lists))
    return [r for r, c in sorted(n.items(), key=lambda kv: (-kv[1], kv[0])) if c >= need]


def plot_system(system: str, source: str, replica_dirs: list[Path], path: Path, grid_nm: float = 0.1):
    """Contacts per type against distance, and the residue x distance occupancy map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from . import theme

    counts, pairs = [], []
    for wd in replica_dirs:
        c = wd / "analysis" / f"contacts_{source}_counts.csv"
        p = wd / "analysis" / f"contacts_{source}_pairs.csv"
        if c.exists() and p.exists():
            counts.append(pd.read_csv(c))
            pairs.append(pd.read_csv(p))
    if not counts:
        return None
    theme.apply()
    g = lambda x: np.round(x / grid_nm) * grid_nm
    cnt = pd.concat(counts)
    cnt["xi"] = g(cnt["xi_nm"])
    cnt["hbond"] = cnt[["hbond_bb_bb", "hbond_sc_bb", "hbond_sc_sc"]].sum(axis=1)
    cnt["pi"] = cnt[["pi_cation", "pi_stacking", "t_stacking"]].sum(axis=1)
    mean = cnt.groupby("xi").mean(numeric_only=True)

    pr = pd.concat([p.assign(rep=i) for i, p in enumerate(pairs)])
    pr = pr[pr["type"] == ANY].copy()
    pr["xi"] = g(pr["xi_nm"])
    pr["res_t"], pr["res_b"] = pr["target_res"], pr["binder_res"]
    long = pd.concat([pr[["rep", "xi", "res_t", "occupancy"]].rename(columns={"res_t": "res"}),
                      pr[["rep", "xi", "res_b", "occupancy"]].rename(columns={"res_b": "res"})])
    occ = long.groupby(["rep", "xi", "res"]).occupancy.max().groupby(["xi", "res"]).mean().unstack("xi").fillna(0.0)
    first = occ.columns.min()
    occ = occ[occ[first] >= 0.5] if (occ[first] >= 0.5).any() else occ
    key = lambda r: (r.split(":")[0][0] != "T", r.split(":")[0], int("".join(ch for ch in r.split(":")[1] if ch.isdigit()) or 0))
    occ = occ.loc[sorted(occ.index, key=key)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, max(4.5, 0.22 * len(occ) + 1.5)),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    series = [("hbond", "H-bonds", 0), ("salt_bridge", "salt bridges", 1), ("pi", "pi (cation, stack, T)", 4),
              ("hydrophobic", "hydrophobic", 2)]
    for col, label, i in series:
        a.plot(mean.index, mean[col], color=theme.CATEGORICAL[i], label=label)
    a.plot(mean.index, mean[ANY], color=theme.INK_SECONDARY, ls="--", label="any (residue pairs)")
    a.set_xlabel("distance xi (nm)")
    a.set_ylabel("residue-pair contacts per frame")
    a.set_title(f"{system}: contacts vs distance ({source}, n={len(counts)} rep)")
    a.legend()
    im = b.imshow(occ.values, aspect="auto", origin="upper", cmap="Blues", vmin=0, vmax=1,
                  extent=[occ.columns.min() - grid_nm / 2, occ.columns.max() + grid_nm / 2, len(occ), 0])
    b.set_yticks(np.arange(len(occ)) + 0.5)
    b.set_yticklabels(occ.index, fontsize=7)
    b.set_xlabel("distance xi (nm)")
    b.set_title("interface residues: fraction of frames in contact")
    b.grid(False)
    fig.colorbar(im, ax=b, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
