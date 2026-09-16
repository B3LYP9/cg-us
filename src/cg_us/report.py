"""Self-contained HTML report."""

from __future__ import annotations

import base64
import datetime as dt
import html
import json
from pathlib import Path

import pandas as pd

CSS = """
:root { color-scheme: light; }
body { margin:0; padding:32px; background:#f9f9f7; color:#0b0b0b;
       font-family: system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }
main { max-width: 1080px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:17px; margin:36px 0 10px; padding-bottom:6px; border-bottom:1px solid #e1e0d9; }
h3 { font-size:14px; margin:22px 0 8px; color:#52514e; }
.sub { color:#52514e; font-size:13px; margin:0 0 24px; }
.cards { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0 8px; }
.card { background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:10px;
        padding:12px 16px; min-width:150px; }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#898781; }
.card .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
table { border-collapse:collapse; width:100%; font-size:13px; background:#fcfcfb;
        border:1px solid rgba(11,11,11,.10); border-radius:8px; overflow:hidden; }
th { text-align:left; font-weight:600; color:#52514e; background:#f4f3ef; padding:8px 10px;
     border-bottom:1px solid #e1e0d9; white-space:nowrap; }
td { padding:7px 10px; border-bottom:1px solid #f0efec; font-variant-numeric:tabular-nums; }
tr:last-child td { border-bottom:none; }
img { max-width:100%; border:1px solid rgba(11,11,11,.10); border-radius:8px; background:#fcfcfb; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }
.flag { background:#fff6e5; border-left:3px solid #fab219; padding:8px 12px; border-radius:0 6px 6px 0;
        margin:6px 0; font-size:13px; }
.bad { background:#fdeceb; border-left-color:#d03b3b; }
.ok { background:#eaf7ea; border-left-color:#0ca30c; }
details { margin-top:10px; }
summary { cursor:pointer; color:#52514e; font-size:13px; }
pre { background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:8px;
      padding:12px; overflow:auto; font-size:12px; }
"""


def _img(path: Path) -> str:
    if not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{b64}">'


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='sub'>no data</p>"
    return df.to_html(index=False, border=0, na_rep="—", justify="left")


def _cards(items: list[tuple[str, object]]) -> str:
    cells = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{"—" if v is None else html.escape(str(v))}</div></div>'
        for k, v in items
    )
    return f'<div class="cards">{cells}</div>'


def build(root: Path, summaries: list[dict], replicas: list[dict], comparison: dict,
          figures: dict, tables: dict, protocol: dict) -> Path:
    root = Path(root)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    head = _cards([
        ("systems", len(summaries)),
        ("replicas / system", protocol.get("replicas")),
        ("MAE kcal/mol", comparison.get("MAE")),
        ("RMSE kcal/mol", comparison.get("RMSE")),
        ("Pearson r", comparison.get("pearson_r")),
        ("Spearman ρ", comparison.get("spearman_rho")),
    ])

    controls = ""
    for c in comparison.get("controls", []):
        cls = "ok" if c["passes"] else "bad"
        verdict = "passes" if c["passes"] else "FAILS"
        controls += (f'<div class="flag {cls}">specificity control <b>{html.escape(c["system"])}</b>: '
                     f'ΔG = {c["dG_mean"]:+.2f} kcal/mol, threshold &gt; {c["threshold"]} — {verdict}</div>')

    flags = ""
    for s in summaries:
        for f in s.get("quality_flags", []):
            flags += f'<div class="flag">{html.escape(s["system"])} — {html.escape(f)}</div>'
    if not flags:
        flags = '<div class="flag ok">no sampling-quality flags raised</div>'

    per_system = ""
    for s in summaries:
        figs = figures.get(s["system"], {})
        blocks = "".join(f"<div>{_img(Path(p))}</div>" for p in figs.values() if p and Path(p).exists())
        reps = [r for r in replicas if r["system"] == s["system"]]
        df = pd.DataFrame([{
            "replica": r["replica"], "ΔG": r["dG"], "bootstrap ±": r["dG_bootstrap_error"],
            "windows": r["n_windows"], "min overlap": r["overlap_min"],
            "ξ bound (nm)": r["bound_xi_nm"], "barrier": r["unbinding_barrier"],
            "half-split drift": r.get("convergence", {}).get("half_split_drift"),
        } for r in reps])
        head_dg = ("no ΔG (every replica hit a gap before the plateau)"
                   if s.get("dG_mean") is None
                   else f"ΔG = {s['dG_mean']:+.2f} ± {s['dG_sd']:.2f} kcal/mol")
        per_system += (
            f"<h3>{html.escape(s['system'])} — {head_dg}"
            + (f" (exp {s['dg_exp']:+.2f})" if s.get("dg_exp") is not None else "")
            + f"</h3>{_table(df)}<div class='grid'>{blocks}</div>"
        )

    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Umbrella sampling — binding free energies</title><style>{CSS}</style></head><body><main>
<h1>Umbrella sampling of protein–peptide complexes</h1>
<p class="sub">CHAPERONg / GROMACS pipeline · generated {ts} · root <code>{html.escape(str(root))}</code></p>
{head}

<h2>Agreement with experiment</h2>
<div class="grid"><div>{_img(Path(figures.get("_global", {}).get("correlation", "")))}</div>
<div>{_img(Path(figures.get("_global", {}).get("spread", "")))}</div></div>
{_table(tables.get("comparison"))}
{controls}

<h2>Per-system summary</h2>
{_table(tables.get("systems"))}

<h2>Replica statistics</h2>
{_table(tables.get("replicas"))}

<h2>Sampling quality</h2>
{flags}

<h2>Systems in detail</h2>
{per_system}

<h2>Pairwise ΔΔG</h2>
{_table(pd.DataFrame(comparison.get("pairwise", {}).get("pairs", [])))}

<h2>Protocol</h2>
<details><summary>full protocol used for this run</summary>
<pre>{html.escape(json.dumps(protocol, indent=2))}</pre></details>
</main></body></html>"""

    out = root / "report.html"
    out.write_text(body, encoding="utf-8")
    return out
