"""
Assemble the clean, canonical post-fix CMA results for PI-facing viewing.

Produces:
  - results_for_pi/                  (symlinks to canonical leaf dirs, organized by condition)
  - results_for_pi.html              (one-page HTML: diverging + single-color per cell)

Canonical post-fix mapping (see slurm_scripts/rerun_faulty_bf.sh and the belief-patch
bug notes for why these are the right sources):

  C1 abstract_belief        bf:  abstract_belief/.../patch_after_movement/
                            end: abstract_context/.../patch_before_movement/   (end never buggy)
  C2 abstract_photo         bf:  abstract_photo_new/.../patch_after_movement/
                            end: photo_context/.../patch_before_movement/
  C3 answer_changes_belief  bf:  answer_changes_belief/.../patch_after_movement/
                            end: answer_changes_belief/.../patch_before_movement/
  C4 answer_changes_photo   bf:  answer_changes_photo_new/.../patch_after_movement/
                            end: answer_changes_photo/.../patch_before_movement/
  C5 control                bf:  OMITTED (detector collapsed — bf==end for 5/6 cells)
                            end: control_context/.../patch_before_movement/

Run from repo root:
    python scripts/build_pi_results.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
PI_DIR = ROOT / "results_for_pi"
PI_HTML = ROOT / "results_for_pi.html"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]


@dataclass
class Cell:
    cond_key: str        # C1, C2, ... (short tag for folders/anchors)
    cond_label: str      # human-readable condition name
    template: str        # food_truck, hair_styling, library_book
    direction: str       # ABA_to_ABB or ABB_to_ABA (primary label used in PI page)
    position: str        # bf or end
    source: Path         # canonical leaf dir containing causal_scores.pt
    caveat: str = ""     # short note if something is worth flagging about this cell


def build_condition_cells() -> list[Cell]:
    cells: list[Cell] = []

    # Helper to append if source exists
    def add(cond_key, cond_label, template, direction, position, source: Path, caveat: str = ""):
        if not source.exists():
            return False
        pt = source / "causal_scores.pt"
        if not pt.exists():
            return False
        cells.append(Cell(cond_key, cond_label, template, direction, position, source, caveat))
        return True

    # C1: abstract_belief.  Rule pairs are ABA↔ABB (belief type change).
    for tmpl in TEMPLATES:
        for (base, exp, direction) in [("ABA", "ABB", "ABA_to_ABB"), ("ABB", "ABA", "ABB_to_ABA")]:
            bf_root = RESULTS_ROOT / f"abstract_belief/template_{tmpl}/base_rule_{base}_exp_rule_{exp}/patch_after_movement/z_seed_0_shuffle_False/logit"
            end_root = RESULTS_ROOT / f"abstract_context/template_{tmpl}/base_rule_{base}_exp_rule_{exp}/patch_before_movement/z_seed_0_shuffle_False/logit"
            for leaf in sorted(bf_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C1", "Abstract belief (same answer, belief type varies)", tmpl, direction, "bf", leaf)
                break
            for leaf in sorted(end_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C1", "Abstract belief (same answer, belief type varies)", tmpl, direction, "end", leaf)
                break

    # C2: abstract_photo.  Rule pairs are ABA↔ABB as config-level flag for photo timing.
    for tmpl in TEMPLATES:
        for (base, exp, direction) in [("ABA", "ABB", "ABA_to_ABB"), ("ABB", "ABA", "ABB_to_ABA")]:
            bf_root = RESULTS_ROOT / f"abstract_photo_new/template_{tmpl}/base_rule_{base}_exp_rule_{exp}/patch_after_movement/z_seed_0_shuffle_False/logit"
            # End patch: legacy photo_context uses same-answer convention (ABA↔ABA etc.)
            legacy_rule = "base_rule_ABA_exp_rule_ABA" if base == "ABA" else "base_rule_ABB_exp_rule_ABB"
            end_root = RESULTS_ROOT / f"photo_context/template_{tmpl}/{legacy_rule}/patch_before_movement/z_seed_0_shuffle_False/logit"
            for leaf in sorted(bf_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C2", "Abstract photo (same answer, photo timing varies)", tmpl, direction, "bf", leaf)
                break
            for leaf in sorted(end_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C2", "Abstract photo (same answer, photo timing varies)", tmpl, direction, "end", leaf)
                break

    # C3: answer_changes_belief.  Rule pairs ABA↔ABB.
    for tmpl in TEMPLATES:
        for (base, exp, direction) in [("ABA", "ABB", "ABA_to_ABB"), ("ABB", "ABA", "ABB_to_ABA")]:
            bf_root = RESULTS_ROOT / f"answer_changes_belief/template_{tmpl}/base_rule_{base}_exp_rule_{exp}/patch_after_movement/z_seed_0_shuffle_False/logit"
            end_root = RESULTS_ROOT / f"answer_changes_belief/template_{tmpl}/base_rule_{base}_exp_rule_{exp}/patch_before_movement/z_seed_0_shuffle_False/logit"
            # Flag the hair_styling ABB→ABA bf which has anomalous large-magnitude negatives
            bf_caveat = "check magnitude: bf range dips to -3.3 (possible partial fallback)" if (tmpl == "hair_styling" and direction == "ABB_to_ABA") else ""
            for leaf in sorted(bf_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C3", "Answer-changes belief (answer varies with belief type)", tmpl, direction, "bf", leaf, bf_caveat)
                break
            for leaf in sorted(end_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C3", "Answer-changes belief (answer varies with belief type)", tmpl, direction, "end", leaf)
                break

    # C4: answer_changes_photo.  Rule pairs are same-answer (ABA↔ABA, ABB↔ABB).
    for tmpl in TEMPLATES:
        for (rule, direction) in [("ABA", "ABA_timing_varies"), ("ABB", "ABB_timing_varies")]:
            bf_root = RESULTS_ROOT / f"answer_changes_photo_new/template_{tmpl}/base_rule_{rule}_exp_rule_{rule}/patch_after_movement/z_seed_0_shuffle_False/logit"
            end_root = RESULTS_ROOT / f"answer_changes_photo/template_{tmpl}/base_rule_{rule}_exp_rule_{rule}/patch_before_movement/z_seed_0_shuffle_False/logit"
            for leaf in sorted(bf_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C4", "Answer-changes photo (answer varies with photo timing)", tmpl, direction, "bf", leaf)
                break
            for leaf in sorted(end_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C4", "Answer-changes photo (answer varies with photo timing)", tmpl, direction, "end", leaf)
                break

    # C5: control.  end only — bf is invalid per detector-collapse bug.
    for tmpl in TEMPLATES:
        for (rule, direction) in [("ABA", "ABA_location_only"), ("ABB", "ABB_location_only")]:
            end_root = RESULTS_ROOT / f"control_context/template_{tmpl}/base_rule_{rule}_exp_rule_{rule}/patch_before_movement/z_seed_0_shuffle_False/logit"
            for leaf in sorted(end_root.glob("sample_num_*_gen_acc_0.9"), reverse=True):
                add("C5", "Control (location retrieval; no belief)", tmpl, direction, "end", leaf,
                    "control bf omitted — detector collapsed (bf==end for 5/6 cells)")
                break

    return cells


def render_singlecolor(scores_pt: Path, out_png: Path, title: str,
                       vmax: float, scale_label: str = "per-cell") -> tuple[float, int]:
    x = torch.load(scores_pt, map_location="cpu", weights_only=False).numpy()
    x = np.maximum(x, 0.0)
    if vmax <= 0:
        vmax = 1e-6
    n_clipped = int((x > vmax).sum())

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(x, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_xlabel("Head index (0–39)", fontsize=11)
    ax.set_ylabel("Layer index (0–47)", fontsize=11)
    ax.set_title(
        f"{title}\n(neg→0, {scale_label} vmax={vmax:.3f} · {n_clipped}/1920 clipped)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return vmax, n_clipped


def compute_group_vmax(cells: list["Cell"]) -> dict[str, float]:
    """Pool positive values from every cell's causal_scores.pt per patch position,
    and return {"bf": vmax_bf, "end": vmax_end} where vmax = 99th percentile of the pool."""
    pools: dict[str, list[np.ndarray]] = {"bf": [], "end": []}
    for c in cells:
        scores = c.source / "causal_scores.pt"
        if not scores.exists():
            continue
        x = torch.load(scores, map_location="cpu", weights_only=False).numpy()
        pos = x[x > 0]
        if pos.size:
            pools[c.position].append(pos.ravel())
    out = {}
    for pos, arrs in pools.items():
        if arrs:
            flat = np.concatenate(arrs)
            out[pos] = float(np.percentile(flat, 99))
        else:
            out[pos] = 1e-6
    return out


def find_diverging_png(leaf: Path) -> Path | None:
    """Pick the canonical diverging heatmap PNG in a leaf dir."""
    # prefer the one without 'singlecolor' in name, without 'true belief' relabel
    candidates = [p for p in leaf.glob("*_heatmap.png") if "singlecolor" not in p.name]
    if not candidates:
        return None
    # shorter filename = the primary heatmap (not relabeled variants)
    return sorted(candidates, key=lambda p: len(p.name))[0]


def ensure_singlecolor_png(leaf: Path, vmax: float, scale_label: str,
                           force: bool = False) -> Path | None:
    diverging = find_diverging_png(leaf)
    if diverging is None:
        return None
    # Use a new suffix so the per-cell version is not clobbered
    out = leaf / f"{diverging.stem}_singlecolor_global.png"
    if force or not out.exists():
        scores = leaf / "causal_scores.pt"
        if not scores.exists():
            return None
        render_singlecolor(scores, out, diverging.stem, vmax=vmax, scale_label=scale_label)
    return out


def build_symlink_tree(cells: list[Cell]):
    PI_DIR.mkdir(exist_ok=True)
    # Write a README at top
    readme = PI_DIR / "README.md"
    readme.write_text(
        "# results_for_pi\n\n"
        "Symlinks to the canonical post-fix CMA leaf directories. Source of truth\n"
        "is still `results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct/` — this\n"
        "folder just re-organizes for convenience. See scripts/build_pi_results.py\n"
        "for the mapping rationale.\n\n"
        "Layout: `{condition}/template_{name}/{direction}/{bf,end}/` → canonical leaf.\n"
    )
    made = 0
    for c in cells:
        link_dir = PI_DIR / c.cond_key / f"template_{c.template}" / c.direction
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / c.position
        if link.exists() or link.is_symlink():
            link.unlink()
        # Use relative path so the symlink survives a whole-repo move
        rel = Path("../" * len(link.parent.relative_to(ROOT).parts)) / c.source.relative_to(ROOT)
        link.symlink_to(rel, target_is_directory=True)
        made += 1
    return made


def write_html(cells: list[Cell], group_vmax: dict[str, float]):
    # group by condition
    by_cond: dict[str, list[Cell]] = {}
    for c in cells:
        by_cond.setdefault(c.cond_key, []).append(c)

    chunks = [
        """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CMA results (post-fix)</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; max-width: 1600px; }
h1 { font-size: 22px; margin-bottom: 6px; }
h2 { font-size: 16px; margin-top: 40px; color: #222; border-bottom: 2px solid #bbb; padding-bottom: 4px; }
h3 { font-size: 13px; margin-top: 18px; color: #555; }
.note { font-size: 13px; color: #444; line-height: 1.55; max-width: 1100px; }
.toc { font-size: 13px; margin: 10px 0 20px 0; }
.toc a { margin-right: 14px; color: #0055aa; text-decoration: none; }
table { border-collapse: collapse; margin-top: 6px; }
th { font-size: 12px; padding: 6px 10px; background: #eee; text-align: center; }
td { padding: 4px; vertical-align: top; text-align: center; border: 1px solid #e5e5e5; }
.pair { display: flex; gap: 6px; align-items: flex-start; }
.pair img { max-width: 260px; height: auto; border: 1px solid #ccc; display: block; }
.pair > div { display: flex; flex-direction: column; align-items: center; }
.lbl { font-size: 10.5px; color: #666; margin-top: 2px; }
.caveat { font-size: 11px; color: #a83232; background: #fff4f4; padding: 2px 6px; border-radius: 3px; margin-top: 2px; }
.meta { font-size: 12px; color: #555; margin: 4px 0 10px 0; }
</style></head>
<body>
<h1>CMA results — post-fix canonical</h1>
<p class="note">
Every heatmap on this page comes from a <b>post-fix, canonical</b> leaf directory — i.e., the
bug-fixed reruns where available, and legacy end-patch folders where those were verified unaffected
by the belief-formation / photo-event detector bugs. Each cell shows two renderings of the same
<code>causal_scores.pt</code>: the left panel is the canonical diverging heatmap (signed, auto-scaled);
the right panel is a single-color <code>YlOrRd</code> render with negatives clamped to zero and
a <b>global vmax shared across all cells at the same patch position</b>, so single-color
heatmaps are directly comparable across conditions and templates within a patch position.
</p>
<p class="note">
<b>Scale (99th percentile of positive scores pooled across all cells at that patch position):</b>
bf vmax = <b>{BF_VMAX}</b> &nbsp;·&nbsp; end vmax = <b>{END_VMAX}</b>.
bf and end use different scales because end-patch magnitudes are substantially larger
(see caveat 3); within each patch position the color mapping is identical everywhere on the page.
</p>
<p class="note">
Patch positions: <b>bf</b> = belief-formation (for belief) / photo-event (for photo), patched at the
token where the relevant epistemic event occurs. <b>end</b> = final token (−1). Rule directions for
belief conditions are ABA↔ABB (false↔true belief); for photo conditions the folder rule is a
config-level flag indicating which photo-timing is base vs experiment.
</p>
<p class="note">
<b>Known caveats (flagged inline):</b>
(1) Control bf is <b>omitted entirely</b> — the belief-formation detector collapsed to −1 for 5 of 6 control cells, so bf and end are bit-identical. The control row shows end only.
(2) In C3 <code>answer_changes_belief</code>, one cell (hair_styling, ABB→ABA) has anomalously large negative bf values — possible residual partial-fallback, flagged in its caveat box.
(3) End-patch magnitudes are generally much larger than bf-patch magnitudes. Part of this is a genuine effect of patching at the output-committed position; treat absolute magnitudes across patch positions with care.
</p>
""".replace("{BF_VMAX}", f"{group_vmax.get('bf', 0.0):.4f}")
        .replace("{END_VMAX}", f"{group_vmax.get('end', 0.0):.4f}")
    ]

    COND_ORDER = ["C1", "C2", "C3", "C4", "C5"]
    chunks.append('<p class="toc">Jump to: ')
    for k in COND_ORDER:
        if k in by_cond:
            chunks.append(f'<a href="#{k}">{k} {by_cond[k][0].cond_label.split(" (")[0]}</a>')
    chunks.append("</p>")

    for cond_key in COND_ORDER:
        if cond_key not in by_cond:
            continue
        cond_cells = by_cond[cond_key]
        label = cond_cells[0].cond_label
        chunks.append(f'<h2 id="{cond_key}">{cond_key}: {label}</h2>')

        # group by (template, direction) — each such group is one row
        row_groups: dict[tuple[str, str], list[Cell]] = {}
        for c in cond_cells:
            row_groups.setdefault((c.template, c.direction), []).append(c)

        for (tmpl, direction), rowcells in sorted(row_groups.items()):
            chunks.append(f"<h3>template_{tmpl} · {direction.replace('_', ' ')}</h3>")
            chunks.append("<table><tr>")
            for pos in ["bf", "end"]:
                match = [c for c in rowcells if c.position == pos]
                if match:
                    chunks.append(f"<th>{pos} patch</th>")
                else:
                    chunks.append(f"<th>{pos} patch (n/a)</th>")
            chunks.append("</tr><tr>")
            for pos in ["bf", "end"]:
                match = [c for c in rowcells if c.position == pos]
                if not match:
                    chunks.append("<td><span class='lbl'>(omitted)</span></td>")
                    continue
                c = match[0]
                diverging = find_diverging_png(c.source)
                vmax = group_vmax.get(c.position, 1e-6)
                single = ensure_singlecolor_png(
                    c.source, vmax=vmax, scale_label=f"global-{c.position}", force=True,
                )
                chunks.append("<td>")
                if c.caveat:
                    chunks.append(f'<div class="caveat">⚠ {c.caveat}</div>')
                sample_dir = c.source.name  # e.g. sample_num_50_gen_acc_0.9
                chunks.append(f'<div class="meta">{sample_dir}</div>')
                chunks.append("<div class='pair'>")
                if diverging:
                    rel = diverging.relative_to(ROOT)
                    chunks.append(f'<div><img src="{rel}"><div class="lbl">diverging</div></div>')
                if single:
                    rel = single.relative_to(ROOT)
                    chunks.append(f'<div><img src="{rel}"><div class="lbl">single-color (global vmax, per patch position)</div></div>')
                chunks.append("</div></td>")
            chunks.append("</tr></table>")

    chunks.append("</body></html>\n")
    PI_HTML.write_text("\n".join(chunks))


def main():
    cells = build_condition_cells()
    print(f"assembled {len(cells)} canonical cells")
    by_cond = {}
    for c in cells:
        by_cond.setdefault(c.cond_key, 0)
        by_cond[c.cond_key] += 1
    for k in sorted(by_cond):
        print(f"  {k}: {by_cond[k]} cells")

    # compute global vmax per patch position (pool positives across all cells)
    group_vmax = compute_group_vmax(cells)
    print(f"\nglobal vmax (99th percentile of positives):")
    for pos, v in group_vmax.items():
        n_cells = sum(1 for c in cells if c.position == pos)
        print(f"  {pos}: vmax={v:.4f}  ({n_cells} cells pooled)")

    # regenerate single-color PNGs with group-global vmax
    print("\nrendering single-color PNGs (global-scaled)...")
    missing = 0
    for c in cells:
        vmax = group_vmax.get(c.position, 1e-6)
        out = ensure_singlecolor_png(c.source, vmax=vmax,
                                     scale_label=f"global-{c.position}", force=True)
        if out is None:
            print(f"  MISSING diverging png or scores for {c}")
            missing += 1
    print(f"  missing: {missing}")

    made = build_symlink_tree(cells)
    print(f"\nbuilt symlink tree: {made} links under {PI_DIR}")

    write_html(cells, group_vmax)
    print(f"wrote {PI_HTML}")


if __name__ == "__main__":
    main()
