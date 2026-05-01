"""
Generate single-color (YlOrRd) per-cell heatmaps as a complement to the
existing diverging heatmaps, and write combined_comparison.html pairing
them side by side.

Scope: every <img src> referenced by bf_comparison.html and
photo_comparison.html. For each, loads the sibling causal_scores.pt,
clamps negatives to 0, and renders YlOrRd with vmin=0 and
vmax = 99th percentile of that cell's positive values. Saves as
<stem>_singlecolor.png next to the original heatmap.

Run from repo root:
    python scripts/make_singlecolor_heatmaps.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]

HTMLS = {
    "belief": ROOT / "bf_comparison.html",
    "photo": ROOT / "photo_comparison.html",
}

SECTION_TITLES = {
    "belief": "Abstract belief · belief-formation patch vs end patch (3 columns)",
    "photo": "Photo · photo-event patch vs movement patch vs end patch (3 columns)",
}


def parse_sections(html_path: Path):
    """Return list of (heading, columns, image_paths) per <h2> block.

    columns: list of column header strings from the <th>s in that section.
    image_paths: list of <img src> values, in order.
    """
    text = html_path.read_text()
    parts = re.split(r"<h2>", text)[1:]
    out = []
    for p in parts:
        m = re.match(r"([^<]+)</h2>", p)
        heading = m.group(1).replace("&middot;", "·").strip() if m else "?"
        ths = re.findall(r"<th>([^<]+)</th>", p)
        imgs = re.findall(r'<img src="([^"]+)"', p)
        out.append({"heading": heading, "columns": ths, "imgs": imgs})
    return out


def render_singlecolor(scores_pt: Path, out_png: Path, title: str):
    x = torch.load(scores_pt, map_location="cpu", weights_only=False).numpy()
    x = np.maximum(x, 0.0)
    pos = x[x > 0]
    vmax = float(pos.max()) if pos.size else 1.0
    if vmax <= 0:
        vmax = 1e-6
    n_clipped = 0  # no clipping: vmax = pos.max()

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        x,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xlabel("Head index (0–39)", fontsize=11)
    ax.set_ylabel("Layer index (0–47)", fontsize=11)
    ax.set_title(
        f"{title}\n(neg→0, vmax={vmax:.3f} · no thresholding)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return vmax, n_clipped


def process_img(rel_src: str) -> tuple[str | None, str]:
    """Render singlecolor next to original heatmap. Returns (relative_png, info)."""
    img_path = ROOT / rel_src
    leaf = img_path.parent
    scores = leaf / "causal_scores.pt"
    if not scores.exists():
        return None, f"NO causal_scores.pt in {leaf}"
    out_png = leaf / f"{img_path.stem}_singlecolor.png"
    try:
        vmax, n_clipped = render_singlecolor(scores, out_png, img_path.stem)
    except Exception as e:
        return None, f"ERROR: {e}"
    return str(out_png.relative_to(ROOT)), f"vmax={vmax:.4f} clipped={n_clipped}"


def main():
    all_sections = {}
    total, ok, skipped = 0, 0, 0
    for label, html in HTMLS.items():
        secs = parse_sections(html)
        for s in secs:
            rendered = []
            for img in s["imgs"]:
                total += 1
                single_rel, info = process_img(img)
                tag = img.split("/")[-1][:55].replace("\n", "")
                print(f"  [{label}] {s['heading'][:34]:34s} | {tag:55s} | {info}")
                rendered.append({"orig": img, "single": single_rel})
                if single_rel:
                    ok += 1
                else:
                    skipped += 1
            s["rendered"] = rendered
        all_sections[label] = secs
    print(f"\nrendered {ok}/{total} cells ({skipped} skipped)")

    write_combined_html(all_sections)


def write_combined_html(all_sections):
    out_path = ROOT / "combined_comparison.html"
    chunks = []
    chunks.append(
        """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CMA combined comparison</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }
h1 { font-size: 20px; margin-bottom: 6px; }
h2 { font-size: 15px; margin-top: 36px; color: #222; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 13px; margin-top: 18px; color: #555; }
.note { font-size: 13px; color: #555; max-width: 960px; line-height: 1.55; }
table { border-collapse: collapse; margin-top: 6px; }
th { font-size: 12px; padding: 6px 10px; background: #eee; text-align: center; }
td { padding: 4px; vertical-align: top; text-align: center; border: 1px solid #e5e5e5; }
.pair { display: flex; gap: 6px; align-items: flex-start; }
.pair img { max-width: 250px; height: auto; border: 1px solid #ccc; display: block; }
.pair > div { display: flex; flex-direction: column; align-items: center; }
.lbl { font-size: 10.5px; color: #666; margin-top: 2px; }
</style></head>
<body>
<h1>CMA comparison: diverging + single-color (per-cell scaled)</h1>
<p class="note">
For each cell, two heatmaps are shown side by side:
<b>left</b> is the canonical diverging heatmap (signed, auto-scaled by the plotting code);
<b>right</b> is a single-color <code>YlOrRd</code> render from <code>causal_scores.pt</code> with
negatives clamped to zero and <b>vmax = max of that cell's positive values</b> (no thresholding).
The single-color scale is tuned per cell, so magnitudes are not comparable across cells —
use the diverging map for that. The single-color map surfaces the within-cell peak
structure that a shared-scale render would otherwise flatten.
No floor threshold is applied; near-zero heads read as near-white.
</p>
"""
    )

    def render_section(label):
        chunks.append(f"<h2>{SECTION_TITLES[label]}</h2>")
        for s in all_sections[label]:
            chunks.append(f"<h3>{s['heading']}</h3>")
            chunks.append("<table><tr>")
            for col in s["columns"]:
                chunks.append(f"<th>{col}</th>")
            chunks.append("</tr><tr>")
            for entry in s["rendered"]:
                chunks.append("<td><div class='pair'>")
                chunks.append(
                    f'<div><img src="{entry["orig"]}"><div class="lbl">diverging</div></div>'
                )
                if entry["single"]:
                    chunks.append(
                        f'<div><img src="{entry["single"]}"><div class="lbl">single-color (per-cell)</div></div>'
                    )
                else:
                    chunks.append(
                        '<div class="lbl">(no single-color: causal_scores.pt missing)</div>'
                    )
                chunks.append("</div></td>")
            chunks.append("</tr></table>")

    render_section("belief")
    render_section("photo")
    chunks.append("</body></html>\n")
    out_path.write_text("\n".join(chunks))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
