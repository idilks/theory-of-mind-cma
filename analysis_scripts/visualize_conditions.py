#!/usr/bin/env python3
"""
5-condition CMA visualization for PI meeting.

produces:
  pass 1 (stitched existing PNGs):
    - end_patch_5x2_stitched.png
    - bf_patch_5x2_stitched.png
  pass 2 (re-rendered from causal_scores.pt, shared color scale):
    - end_patch_5x2_shared_scale.png
    - bf_patch_5x2_shared_scale.png
  additional:
    - patch_position_comparison.png  (H3/H4: 2×2 grid)
    - end_patch_5x2_{template}.png   (per-template supplementary)
    - condition_correlation_matrix.png (5×5 pearson r)

no CLI args — paths are fixed. output to results/visualization/.
"""

import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from PIL import Image, ImageDraw, ImageFont

# ── global font defaults ──
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.titlesize": 28,
})


# ── constants ──

ROOT = Path("results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct")
OUT = Path("results/visualization")
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
N_LAYERS, N_HEADS = 48, 40

# condition definitions: (folder_name, dir1_base, dir1_exp, dir2_base, dir2_exp)
# dir1 = primary, dir2 = reverse
CONDITION_DEFS = {
    "C1: Abstract, Belief": ("abstract_belief", "ABA", "ABB", "ABB", "ABA"),
    "C2: Abstract, Photo": ("abstract_photo", "ABA", "ABA", "ABB", "ABB"),
    "C3: Answer Changes, Belief": ("answer_changes_belief", "ABA", "ABB", "ABB", "ABA"),
    "C4: Answer Changes, Photo": ("answer_changes_photo", "ABA", "ABA", "ABB", "ABB"),
    "C5: Control": ("control", "ABA", "ABA", "ABB", "ABB"),
}

CONDITION_DESCRIPTIONS = {
    "C1: Abstract, Belief": "belief type varies, same answers — isolates belief-coding heads",
    "C2: Abstract, Photo": "photo timing varies, same answers — isolates state-tracking heads",
    "C3: Answer Changes, Belief": "belief type varies, different answers — belief + answer heads (confounded)",
    "C4: Answer Changes, Photo": "photo timing varies, different answers — state-tracking + answer heads",
    "C5: Control": "only location tokens vary, different answers — location retrieval baseline",
}

COND_NAMES = list(CONDITION_DEFS.keys())

PATCH_MAP = {
    "bf": "patch_after_movement",
    "end": "patch_before_movement",
}

DIR_LABELS = {1: "Primary (ABA→ABB)", 2: "Reverse (ABB→ABA)"}


# ── result discovery ──

def find_best_leaf(condition_folder, template, base_rule, exp_rule, patch_pos):
    """find the leaf folder with highest sample_num (skip _ohwell)."""
    base_dir = (
        ROOT / condition_folder / f"template_{template}"
        / f"base_rule_{base_rule}_exp_rule_{exp_rule}"
        / patch_pos / "z_seed_0_shuffle_False" / "logit"
    )
    if not base_dir.exists():
        return None
    candidates = []
    for d in base_dir.iterdir():
        if "_ohwell" in d.name:
            continue
        m = re.match(r"sample_num_(\d+)", d.name)
        if m and d.is_dir():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_heatmap(leaf_dir):
    """find *_heatmap.png in a leaf directory."""
    if leaf_dir is None:
        return None
    pngs = list(leaf_dir.glob("*_heatmap.png"))
    return pngs[0] if pngs else None


def load_scores(leaf_dir):
    """load causal_scores.pt and reshape to (48, 40)."""
    if leaf_dir is None:
        return None
    pt_file = leaf_dir / "causal_scores.pt"
    if not pt_file.exists():
        return None
    scores = torch.load(pt_file, map_location="cpu")
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()
    scores = np.array(scores).flatten()
    if scores.shape[0] != N_LAYERS * N_HEADS:
        print(f"  WARNING: unexpected shape {scores.shape} in {pt_file}")
        return None
    return scores.reshape(N_LAYERS, N_HEADS)


def get_leaf(cond_name, direction, patch_key, template):
    """get leaf dir for a given condition × direction × patch × template."""
    folder, d1_base, d1_exp, d2_base, d2_exp = CONDITION_DEFS[cond_name]
    if direction == 1:
        base_rule, exp_rule = d1_base, d1_exp
    else:
        base_rule, exp_rule = d2_base, d2_exp
    return find_best_leaf(folder, template, base_rule, exp_rule, PATCH_MAP[patch_key])


def load_template_averaged_scores(cond_name, direction, patch_key):
    """load and average causal scores across 3 templates. returns (48, 40) or None."""
    mats = []
    for tmpl in TEMPLATES:
        leaf = get_leaf(cond_name, direction, patch_key, tmpl)
        scores = load_scores(leaf)
        if scores is not None:
            mats.append(scores)
    if len(mats) == 0:
        return None
    if len(mats) < 3:
        print(f"  WARNING: only {len(mats)}/3 templates for {cond_name} dir{direction} {patch_key}")
    return np.mean(mats, axis=0)


# ── pass 1: stitch existing heatmap PNGs ──

def make_stitched_grid(patch_key, output_name):
    """5 rows (conditions) × 2 columns (directions), stitched from existing PNGs."""
    print(f"\n--- stitching {patch_key} grid ---")

    # first pass: collect images and determine panel size
    panels = {}  # (row, col) -> PIL Image
    for row, cond in enumerate(COND_NAMES):
        for col, direction in enumerate([1, 2]):
            # use first available template's heatmap
            for tmpl in TEMPLATES:
                leaf = get_leaf(cond, direction, patch_key, tmpl)
                hm = find_heatmap(leaf)
                if hm is not None:
                    panels[(row, col)] = Image.open(hm)
                    print(f"  [{row},{col}] {cond} dir{direction}: {hm.parent.name} ({tmpl})")
                    break
            else:
                print(f"  [{row},{col}] {cond} dir{direction}: MISSING")

    if not panels:
        print("  no panels found, skipping")
        return

    # determine panel dimensions from first image
    sample_img = next(iter(panels.values()))
    pw, ph = sample_img.size

    # layout parameters — single banner with description only
    banner_h = 80
    col_header_h = 70
    margin = 20
    n_rows, n_cols = 5, 2

    canvas_w = margin + n_cols * (pw + margin)
    canvas_h = margin + col_header_h + n_rows * (banner_h + ph + margin)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # try to get a reasonable font
    try:
        font_banner = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 26)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_col_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font_banner = ImageFont.load_default()
        font_med = font_banner
        font_col_header = font_banner

    # column headers
    for col, label in enumerate([DIR_LABELS[1], DIR_LABELS[2]]):
        x = margin + col * (pw + margin) + pw // 2
        draw.text((x, margin + col_header_h // 2), label, fill="black", font=font_col_header, anchor="mm")

    # paste panels
    for row, cond in enumerate(COND_NAMES):
        y_banner = margin + col_header_h + row * (banner_h + ph + margin)

        # single banner — description text in blue box
        desc = CONDITION_DESCRIPTIONS[cond]
        # wrap at em-dash for two-line display
        if " — " in desc:
            line1, line2 = desc.split(" — ", 1)
            draw.rectangle(
                [margin, y_banner, canvas_w - margin, y_banner + banner_h],
                fill="#2c3e50"
            )
            draw.text(
                (canvas_w // 2, y_banner + banner_h // 3),
                line1, fill="white", font=font_banner, anchor="mm"
            )
            draw.text(
                (canvas_w // 2, y_banner + banner_h * 2 // 3),
                line2, fill="#bdc3c7", font=font_banner, anchor="mm"
            )
        else:
            draw.rectangle(
                [margin, y_banner, canvas_w - margin, y_banner + banner_h],
                fill="#2c3e50"
            )
            draw.text(
                (canvas_w // 2, y_banner + banner_h // 2),
                desc, fill="white", font=font_banner, anchor="mm"
            )

        y_img = y_banner + banner_h
        for col in range(n_cols):
            if (row, col) in panels:
                x_img = margin + col * (pw + margin)
                canvas.paste(panels[(row, col)], (x_img, y_img))
            else:
                # gray placeholder
                x_img = margin + col * (pw + margin)
                draw.rectangle(
                    [x_img, y_img, x_img + pw, y_img + ph],
                    fill="#ecf0f1", outline="#bdc3c7"
                )
                draw.text(
                    (x_img + pw // 2, y_img + ph // 2),
                    "NO DATA", fill="#95a5a6", font=font_med, anchor="mm"
                )

    canvas.save(OUT / output_name, dpi=(150, 150))
    print(f"  saved: {OUT / output_name}")


# ── pass 2: re-rendered with shared color scale ──

def make_shared_scale_grid(patch_key, output_name):
    """5×2 grid rendered from causal_scores.pt with shared vmin/vmax."""
    print(f"\n--- rendering {patch_key} shared-scale grid ---")

    # load all template-averaged scores
    scores = {}  # (row, col) -> (48, 40) array
    for row, cond in enumerate(COND_NAMES):
        for col, direction in enumerate([1, 2]):
            avg = load_template_averaged_scores(cond, direction, patch_key)
            if avg is not None:
                scores[(row, col)] = avg
                print(f"  [{row},{col}] {cond} dir{direction}: loaded")
            else:
                print(f"  [{row},{col}] {cond} dir{direction}: MISSING")

    if not scores:
        print("  no data, skipping")
        return

    # shared color scale — clamp negatives to 0, clip at 99th percentile
    all_vals = np.concatenate([s.flatten() for s in scores.values()])
    pos_vals = all_vals[all_vals > 0]
    vmax = np.percentile(pos_vals, 99) if len(pos_vals) > 0 else 1.0
    n_clipped = int(np.sum(all_vals > vmax))
    print(f"  color scale: 0 to {vmax:.4f} (99th pct of positive; {n_clipped} values clipped)")

    fig, axes = plt.subplots(5, 2, figsize=(24, 42))

    for row, cond in enumerate(COND_NAMES):
        for col, direction in enumerate([1, 2]):
            ax = axes[row, col]
            if (row, col) in scores:
                sns.heatmap(
                    np.clip(scores[(row, col)], 0, None),
                    ax=ax, cmap="YlOrRd",
                    vmin=0, vmax=vmax,
                    xticklabels=10, yticklabels=10,
                    cbar=(col == 1),
                    cbar_kws={"shrink": 0.8, "label": "CMA score"} if col == 1 else {},
                )
                ax.set_xlabel("Head" if row == 4 else "")
                ax.set_ylabel("Layer" if col == 0 else "")
            else:
                ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                        ha="center", va="center", fontsize=18, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])

            if row == 0:
                ax.set_title(DIR_LABELS[col + 1], fontweight="bold", pad=12)

        # row label — description only, in blue box, italic
        desc = CONDITION_DESCRIPTIONS[cond]
        desc_wrapped = desc.replace(" — ", "\n") if " — " in desc else desc
        axes[row, 0].annotate(
            desc_wrapped,
            xy=(-0.42, 0.5), xycoords="axes fraction",
            fontsize=16, ha="center", va="center",
            rotation=90, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.9),
            color="white",
        )

    plt.suptitle(
        f"CMA Scores — {'Belief Formation' if patch_key == 'bf' else 'End'} Patch Position\n"
        f"(template-averaged, negatives clamped to 0, 99th pct cap: {vmax:.4f})",
        fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0.13, 0, 1, 0.98])
    plt.savefig(OUT / output_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {OUT / output_name}")


# ── figure 3: patch position comparison (H3/H4) ──

def make_patch_position_comparison():
    """2×2 grid: {control, abstract_belief} × {bf, end}, direction 1, template-averaged."""
    print("\n--- patch position comparison (H3/H4) ---")

    conds = ["C5: Control", "C1: Abstract, Belief"]
    patches = ["bf", "end"]
    patch_labels = ["Belief Formation", "End (-1)"]
    hypothesis_notes = [
        "H4: control insensitive to patch position",
        "H3: bf more specific than end",
    ]

    scores = {}
    for row, cond in enumerate(conds):
        for col, pk in enumerate(patches):
            avg = load_template_averaged_scores(cond, 1, pk)
            if avg is not None:
                scores[(row, col)] = avg

    if not scores:
        print("  no data, skipping")
        return

    all_vals = np.concatenate([s.flatten() for s in scores.values()])
    pos_vals = all_vals[all_vals > 0]
    vmax = np.percentile(pos_vals, 99) if len(pos_vals) > 0 else 1.0
    n_clipped = int(np.sum(all_vals > vmax))
    print(f"  color scale: 0 to {vmax:.4f} (99th pct of positive; {n_clipped} values clipped)")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for row, cond in enumerate(conds):
        for col, pk in enumerate(patches):
            ax = axes[row, col]
            if (row, col) in scores:
                sns.heatmap(
                    np.clip(scores[(row, col)], 0, None),
                    ax=ax, cmap="YlOrRd",
                    vmin=0, vmax=vmax,
                    xticklabels=10, yticklabels=10,
                    cbar=(col == 1),
                    cbar_kws={"shrink": 0.8, "label": "CMA score"} if col == 1 else {},
                )
            else:
                ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                        ha="center", va="center", fontsize=18, color="gray")

            if row == 0:
                ax.set_title(patch_labels[col], fontweight="bold", pad=12)
            ax.set_xlabel("Head" if row == 1 else "")
            ax.set_ylabel("Layer" if col == 0 else "")

        # row annotation — description + hypothesis note
        desc = CONDITION_DESCRIPTIONS[cond]
        desc_short = desc.split(" — ")[0] if " — " in desc else desc
        label_text = f"{desc_short}\n({hypothesis_notes[row]})"
        axes[row, 0].annotate(
            label_text,
            xy=(-0.38, 0.5), xycoords="axes fraction",
            fontsize=16, ha="center", va="center", rotation=90,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.9),
            color="white",
        )

    plt.suptitle(
        "Patch Position Comparison — Primary Direction, Template-Averaged",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    plt.savefig(OUT / "patch_position_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {OUT / 'patch_position_comparison.png'}")


# ── figure 4: per-template end patch ──

def make_per_template_grids():
    """one 5×2 grid per template (end patch only)."""
    print("\n--- per-template end patch grids ---")

    for tmpl in TEMPLATES:
        print(f"\n  template: {tmpl}")
        scores = {}
        for row, cond in enumerate(COND_NAMES):
            for col, direction in enumerate([1, 2]):
                leaf = get_leaf(cond, direction, "end", tmpl)
                s = load_scores(leaf)
                if s is not None:
                    scores[(row, col)] = s

        if not scores:
            print(f"    no data for {tmpl}, skipping")
            continue

        all_vals = np.concatenate([s.flatten() for s in scores.values()])
        pos_vals = all_vals[all_vals > 0]
        vmax = np.percentile(pos_vals, 99) if len(pos_vals) > 0 else 1.0
        n_clipped = int(np.sum(all_vals > vmax))
        print(f"    color scale: 0 to {vmax:.4f} (99th pct of positive; {n_clipped} values clipped)")

        fig, axes = plt.subplots(5, 2, figsize=(22, 38))

        for row, cond in enumerate(COND_NAMES):
            for col, direction in enumerate([1, 2]):
                ax = axes[row, col]
                if (row, col) in scores:
                    sns.heatmap(
                        np.clip(scores[(row, col)], 0, None),
                        ax=ax, cmap="YlOrRd",
                        vmin=0, vmax=vmax,
                        xticklabels=10, yticklabels=10,
                        cbar=(col == 1),
                        cbar_kws={"shrink": 0.8, "label": "CMA score"} if col == 1 else {},
                    )
                else:
                    ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                            ha="center", va="center", fontsize=18, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])

                if row == 0:
                    ax.set_title(DIR_LABELS[col + 1], fontweight="bold", pad=12)
                ax.set_xlabel("Head" if row == 4 else "")
                ax.set_ylabel("Layer" if col == 0 else "")

            desc = CONDITION_DESCRIPTIONS[cond]
            desc_wrapped = desc.replace(" — ", "\n") if " — " in desc else desc
            axes[row, 0].annotate(
                desc_wrapped, xy=(-0.38, 0.5), xycoords="axes fraction",
                fontsize=14, ha="center", va="center", rotation=90,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.9),
                color="white",
            )

        plt.suptitle(
            f"CMA Scores — End Patch — {tmpl}\n(negatives clamped to 0, 99th pct cap: {vmax:.4f})",
            fontweight="bold",
        )
        plt.tight_layout(rect=[0.1, 0, 1, 0.97])
        fname = f"end_patch_5x2_{tmpl}.png"
        plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    saved: {OUT / fname}")


# ── figure 5: pairwise condition correlations ──

def make_correlation_matrix():
    """5×5 pearson r matrix of raw CMA scores (end patch, template-averaged, dir1)."""
    print("\n--- condition correlation matrix ---")

    cond_scores = {}
    for cond in COND_NAMES:
        avg = load_template_averaged_scores(cond, 1, "end")
        if avg is not None:
            cond_scores[cond] = avg.flatten()

    n = len(COND_NAMES)
    r_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    for i, ci in enumerate(COND_NAMES):
        for j, cj in enumerate(COND_NAMES):
            if ci in cond_scores and cj in cond_scores:
                r, p = pearsonr(cond_scores[ci], cond_scores[cj])
                r_matrix[i, j] = r
                p_matrix[i, j] = p

    # short labels for axis
    short_labels = [c.split(": ")[1] for c in COND_NAMES]

    fig, ax = plt.subplots(figsize=(14, 12))

    # annotations: r value only (no stars — with n=1920, all p≈0 trivially)
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if np.isnan(r_matrix[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{r_matrix[i, j]:.2f}"

    sns.heatmap(
        r_matrix, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=short_labels, yticklabels=short_labels,
        annot=annot, fmt="", annot_kws={"fontsize": 22, "fontweight": "bold"},
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        linewidths=1.5,
    )
    ax.set_title(
        "Pairwise Condition Correlations\n"
        "Pearson r of raw CMA scores\n"
        "(1920 heads, end patch, template-averaged, primary direction)",
        pad=20,
    )
    ax.tick_params(axis="x", rotation=35, labelsize=18)
    ax.tick_params(axis="y", rotation=0, labelsize=18)

    plt.tight_layout()
    plt.savefig(OUT / "condition_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {OUT / 'condition_correlation_matrix.png'}")

    # print matrix
    print("\n  Pearson r matrix (end patch, template-averaged, dir1):")
    print(f"  {'':>25s}", end="")
    for s in short_labels:
        print(f"  {s:>15s}", end="")
    print()
    for i, ci in enumerate(short_labels):
        print(f"  {ci:>25s}", end="")
        for j in range(n):
            if np.isnan(r_matrix[i, j]):
                print(f"  {'—':>15s}", end="")
            else:
                print(f"  {r_matrix[i, j]:>15.3f}", end="")
        print()


# ── main ──

def main():
    print("=" * 60)
    print("5-condition CMA visualization")
    print("=" * 60)

    # verify data availability
    print("\ndata availability check:")
    for cond in COND_NAMES:
        for pk in ["bf", "end"]:
            for d in [1, 2]:
                found = []
                for tmpl in TEMPLATES:
                    leaf = get_leaf(cond, d, pk, tmpl)
                    if leaf is not None:
                        found.append(tmpl)
                status = f"{len(found)}/3" if found else "NONE"
                if len(found) < 3:
                    missing = set(TEMPLATES) - set(found)
                    status += f" (missing: {', '.join(missing)})"
                print(f"  {cond:>35s} | {pk:>3s} | dir{d} | {status}")

    # pass 1: stitched
    print("\n" + "=" * 60)
    print("PASS 1: stitched heatmap PNGs")
    print("=" * 60)
    make_stitched_grid("end", "end_patch_5x2_stitched.png")
    make_stitched_grid("bf", "bf_patch_5x2_stitched.png")

    # pass 2: shared scale
    print("\n" + "=" * 60)
    print("PASS 2: shared-scale re-rendered")
    print("=" * 60)
    make_shared_scale_grid("end", "end_patch_5x2_shared_scale.png")
    make_shared_scale_grid("bf", "bf_patch_5x2_shared_scale.png")

    # figure 3: patch position comparison
    print("\n" + "=" * 60)
    print("FIGURE 3: patch position comparison")
    print("=" * 60)
    make_patch_position_comparison()

    # figure 4: per-template
    print("\n" + "=" * 60)
    print("FIGURE 4: per-template end patch")
    print("=" * 60)
    make_per_template_grids()

    # figure 5: correlation matrix
    print("\n" + "=" * 60)
    print("FIGURE 5: condition correlation matrix")
    print("=" * 60)
    make_correlation_matrix()

    print("\n" + "=" * 60)
    print(f"all outputs in: {OUT}/")
    print("done.")


if __name__ == "__main__":
    main()
