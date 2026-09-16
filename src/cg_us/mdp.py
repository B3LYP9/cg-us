"""Generation of the .mdp set consumed by CHAPERONg / GROMACS.

Key naming constraint: CHAPERONg greps ``pull_group1_name``/``pull_group2_name``
out of md_pull.mdp with awk field 3, so those two lines must keep the underscore
spelling and a bare ``key = value`` layout.
"""

from __future__ import annotations

from pathlib import Path

from .config import Protocol

PULL_GROUPS = ("Target", "Binder")

_CHARMM_NB = """cutoff-scheme            = Verlet
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.12
rcoulomb                 = 1.2
vdwtype                  = cutoff
vdw-modifier             = force-switch
rvdw_switch              = 1.0
rvdw                     = 1.2
rlist                    = 1.2
DispCorr                 = no"""

_AMBER_NB = """cutoff-scheme            = Verlet
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.12
rcoulomb                 = 1.0
rvdw                     = 1.0
rlist                    = 1.0
DispCorr                 = EnerPres"""


def nonbonded_block(force_field: str) -> str:
    return _CHARMM_NB if force_field.lower().startswith("charmm") else _AMBER_NB


def _mass_block(p: Protocol) -> str:
    """Hydrogen mass repartitioning, the validated way.

    pdb2gmx -heavyh scales hydrogens to 4x and subtracts the difference from the
    bonded heavy atom with no lower bound, which leaves a methyl carbon at ~2.9
    amu - lighter than each of its own hydrogens. The mdp option scales to 3x and
    errors out if any bound atom would fall below that mass.
    """
    f = p.prep.mass_repartition_factor
    return f"mass-repartition-factor  = {f}" if f and f > 1 else ""


def _tcoupl(temp: float) -> str:
    return f"""tcoupl                   = V-rescale
tc-grps                  = Protein Water_and_ions
tau_t                    = 0.1     0.1
ref_t                    = {temp}   {temp}"""


def _pull_block(rate: float, k: float, nst: int, geometry: str = "distance") -> str:
    return f"""pull                     = yes
pull_ncoords             = 1
pull_ngroups             = 2
pull_group1_name = {PULL_GROUPS[0]}
pull_group2_name = {PULL_GROUPS[1]}
pull_coord1_type         = umbrella
pull_coord1_geometry     = {geometry}
pull_coord1_groups       = 1 2
pull_coord1_dim          = N N Y
pull_coord1_start        = yes
pull_coord1_init         = 0
pull_coord1_rate         = {rate}
pull_coord1_k            = {k}
pull_nstxout             = {nst}
pull_nstfout             = {nst}
pull_pbc_ref_prev_step_com = yes"""


def _defines(restrain_target: bool) -> str:
    """Biased stages restrain only the target.

    A position restraint on the binder would work against the pull force and
    against the umbrella potential, so ``restrain_binder`` from the manifest is
    deliberately confined to the pre-SMD equilibration.
    """
    return "-DPOSRES_TARGET" if restrain_target else ""


def write_all(workdir: str | Path, proto: Protocol, *, restrain_binder: bool, seed: int) -> None:
    workdir = Path(workdir)
    nb = nonbonded_block(proto.prep.force_field)
    T = proto.temperature
    equil = "-DPOSRES -DPOSRES_BINDER" if restrain_binder else "-DPOSRES"
    files = {
        "ions.mdp": _ions(nb),
        "minim.mdp": _minim(nb),
        "nvt.mdp": _nvt(nb, T, proto, seed, equil),
        "npt.mdp": _npt(nb, T, proto, equil),
        "md_pull.mdp": _pull(nb, T, proto),
        "npt_umbrella.mdp": _npt_umbrella(nb, T, proto),
        "md_umbrella.mdp": _md_umbrella(nb, T, proto),
    }
    for name, text in files.items():
        (workdir / name).write_text(text.rstrip() + "\n")


def _ions(nb: str) -> str:
    return f"""integrator               = steep
emtol                    = 1000.0
emstep                   = 0.01
nsteps                   = 50000
nstlist                  = 10
{nb}
pbc                      = xyz
"""


def _minim(nb: str) -> str:
    return f"""integrator               = steep
emtol                    = 500.0
emstep                   = 0.01
nsteps                   = 50000
nstlist                  = 10
{nb}
pbc                      = xyz
"""


def _nvt(nb: str, T: float, p: Protocol, seed: int, define: str) -> str:
    nstlist = p.run.nstlist
    mass = _mass_block(p)
    return f"""title                    = NVT equilibration
define                   = {define}
integrator               = md
dt                       = {p.smd.dt}
nsteps                   = {int(0.1 * 1000 / p.smd.dt)}
nstxout-compressed       = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = no
constraint_algorithm     = lincs
constraints              = {p.prep.constraints}
lincs_iter               = 1
lincs_order              = 4
{mass}
nstlist                  = {nstlist}
{nb}
{_tcoupl(T)}
pcoupl                   = no
pbc                      = xyz
gen_vel                  = yes
gen_temp                 = {T}
gen_seed                 = {seed}
refcoord_scaling         = com
"""


def _npt(nb: str, T: float, p: Protocol, define: str) -> str:
    nstlist = p.run.nstlist
    mass = _mass_block(p)
    return f"""title                    = NPT equilibration
define                   = {define}
integrator               = md
dt                       = {p.smd.dt}
nsteps                   = {int(0.5 * 1000 / p.smd.dt)}
nstxout-compressed       = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = yes
constraint_algorithm     = lincs
constraints              = {p.prep.constraints}
lincs_iter               = 1
lincs_order              = 4
{mass}
nstlist                  = {nstlist}
{nb}
{_tcoupl(T)}
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = 1.0
compressibility          = 4.5e-5
refcoord_scaling         = com
pbc                      = xyz
gen_vel                  = no
"""


def _pull(nb: str, T: float, p: Protocol) -> str:
    nstlist = p.run.nstlist
    mass = _mass_block(p)
    nsteps = int(p.smd.time_ns * 1000 / p.smd.dt)
    return f"""title                    = Steered MD along the target-binder COM axis
define                   = {_defines(p.smd.restrain_target)}
integrator               = md
dt                       = {p.smd.dt}
nsteps                   = {nsteps}
nstxout-compressed       = 500
nstxout                  = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = yes
constraint_algorithm     = lincs
constraints              = {p.prep.constraints}
lincs_iter               = 1
lincs_order              = 4
{mass}
nstlist                  = {nstlist}
{nb}
{_tcoupl(T)}
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = 1.0
compressibility          = 4.5e-5
refcoord_scaling         = com
pbc                      = xyz
gen_vel                  = no
{_pull_block(p.smd.rate, p.smd.k, 500)}
"""


def _npt_umbrella(nb: str, T: float, p: Protocol) -> str:
    nstlist = p.run.nstlist
    mass = _mass_block(p)
    return f"""title                    = Umbrella window equilibration
define                   = {_defines(p.smd.restrain_target)}
integrator               = md
dt                       = {p.umbrella.dt}
nsteps                   = {int(p.umbrella.equil_ns * 1000 / p.umbrella.dt)}
nstxout-compressed       = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = yes
constraint_algorithm     = lincs
constraints              = {p.prep.constraints}
lincs_iter               = 1
lincs_order              = 4
{mass}
nstlist                  = {nstlist}
{nb}
{_tcoupl(T)}
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = 1.0
compressibility          = 4.5e-5
refcoord_scaling         = com
pbc                      = xyz
gen_vel                  = no
{_pull_block(0.0, p.umbrella.k, 1000)}
"""


def _md_umbrella(nb: str, T: float, p: Protocol) -> str:
    nstlist = p.run.nstlist
    mass = _mass_block(p)
    nst = max(1, int(1.0 / p.umbrella.dt))
    return f"""title                    = Umbrella sampling production
define                   = {_defines(p.smd.restrain_target)}
integrator               = md
dt                       = {p.umbrella.dt}
nsteps                   = {int(p.umbrella.time_ns * 1000 / p.umbrella.dt)}
nstxout-compressed       = 25000
nstenergy                = 25000
nstlog                   = 25000
continuation             = yes
constraint_algorithm     = lincs
constraints              = {p.prep.constraints}
lincs_iter               = 1
lincs_order              = 4
{mass}
nstlist                  = {nstlist}
{nb}
{_tcoupl(T)}
pcoupl                   = {p.umbrella.barostat}
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = 1.0
compressibility          = 4.5e-5
refcoord_scaling         = com
pbc                      = xyz
gen_vel                  = no
{_pull_block(0.0, p.umbrella.k, nst)}
"""


def write_parafile(workdir: str | Path, proto: Protocol, gmx_exe: str | None = None) -> None:
    """paraFile.par for CHAPERONg.

    CHAP_set_US_starting_configs.py substring-matches ``us_window_spacing`` and
    takes whitespace field 3, so the key must appear exactly once, uncommented.
    """
    lines = [
        "; generated by cg-us",
        f"bt                =      triclinic",
        f"water             =      {proto.prep.water}",
        f"ff                =      {'wd' if proto.prep.ff_in_workdir else proto.prep.force_field}",
        f"temp              =      {int(proto.temperature)}",
        f"conc              =      {proto.prep.ion_conc}",
        f"posname           =      {proto.prep.pname}",
        f"negname           =      {proto.prep.nname}",
        f"maxwarn           =      {proto.run.maxwarn}",
        f"dist              =      {proto.prep.box_edge_xy}",
        f"auto_mode         =      full",
        f"us_window_spacing =      {proto.umbrella.window_spacing}",
    ]
    # CHAPERONg only reaches a sane -ntmpi/-ntomp branch when both are non-zero;
    # leaving either unset makes it emit "-ntmpi 1 -ntomp 0 -nt 0".
    from .backends.base import threads_per_worker
    lines.append(f"ntmpi             =      {proto.run.ntmpi or 1}")
    lines.append(f"ntomp             =      {threads_per_worker(proto, 1)}")
    if proto.run.gpu_id:
        lines.append(f"gpu_id            =      {proto.run.gpu_id}")
    if gmx_exe:
        lines.append(f"gmx_exe           =      {gmx_exe}")
    if proto.run.movie_frames and not proto.run.skip_movie:
        # setting movieFrame also removes CHAPERONg's "1 or 2" movie-length prompt
        lines.append(f"movieFrame        =      {proto.run.movie_frames}")
    Path(workdir, "paraFile.par").write_text("\n".join(lines) + "\n")
