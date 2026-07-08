"""
Per-class recall (overall + by GSP) and confusion matrices (count + fraction).
Saves:
  images/class_accuracy_overall.png
  images/class_accuracy_by_gsp.png
  images/binary_accuracy_overall.png
  images/binary_accuracy_by_gsp.png
  images/confusion_matrices.png
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]

TRIAL_GSPS = [
    ("BigValley",           "Big Valley",      "1_BigValley_DraftGSP_ScoringRubric.csv"),
    ("EastContraCosta",     "E. Contra Costa", "14_EastContraCosta_DraftGSP_ScoringRubric.csv"),
    ("Fillmore",            "Fillmore",         "15_Fillmore_DraftGSP_ScoringRubric.csv"),
    ("SonomaValley",        "Sonoma",           "30_SonomaValley_DraftGSP_ScoringRubric.csv"),
    ("SanLuisObispoValley", "San Luis Obispo",  "50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv"),
]

CLASSES = ["Yes", "Somewhat", "No"]
GSP_CNAMES = [g[0] for g in TRIAL_GSPS]
GSP_LABELS = [g[1] for g in TRIAL_GSPS]

os.makedirs("images", exist_ok=True)

# helpers
def HumanRubric(path):
    df = pd.read_csv(path)
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]
    return df[1:]

def load_rubric_answers(rubric_filename):
    rubric  = HumanRubric(os.path.join(RUBRIC_DIR, rubric_filename))
    answers = rubric["Answer"].drop(no_test, errors="ignore")
    return [str(v).strip() for v in answers]

def parse_answer(response):
    for line in str(response).split("\n"):
        if line.strip().upper().startswith("ANSWER:"):
            return line.strip()[7:].strip().split(",")[0].strip()
    return "Unknown"

def per_class_recall(true_list, pred_list):
    """Returns {cls: recall} for Yes/Somewhat/No, ignoring NotApplicable rows."""
    out = {}
    for cls in CLASSES:
        idx = [i for i, t in enumerate(true_list) if t == cls]
        out[cls] = sum(pred_list[i] == cls for i in idx) / len(idx) if idx else np.nan
    return out

def confusion_matrix_3class(true_list, pred_list):
    """Returns 3x3 count matrix, rows=true, cols=predicted (Yes/Somewhat/No)."""
    mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_list, pred_list):
        if t in CLASSES and p in CLASSES:
            mat[CLASSES.index(t), CLASSES.index(p)] += 1
    return mat

# load rubric answers per GSP
rubric_by_gsp = {cname: load_rubric_answers(rf) for cname, _, rf in TRIAL_GSPS}

# load predictions per model per GSP
def load_ckpt_dict(ckpt_file, key_map):
    with open(ckpt_file) as f:
        ckpt = json.load(f)
    return {cname: [parse_answer(r) for r in ckpt[ckpt_key]]
            for cname, ckpt_key in key_map.items()}

def load_opus_preds():
    gsp_files = {
        "BigValley":           "bigvalley",
        "EastContraCosta":     "eastcontracosta",
        "Fillmore":            "fillmore",
        "SonomaValley":        "sonoma",
        "SanLuisObispoValley": "sanluisobispovalley",
    }
    out = {}
    for cname, slug in gsp_files.items():
        fpath = sorted(glob.glob(f"results/checkpoint_opus47_vision_{slug}_*.json"))[-1]
        with open(fpath) as f:
            out[cname] = [parse_answer(r) for r in json.load(f)]
    return out

std_keys = {c: c for c in GSP_CNAMES}
o3_keys  = {
    "BigValley": "BigValley", "EastContraCosta": "EastContraCosta",
    "Fillmore": "Fillmore", "SonomaValley": "Sonoma", "SanLuisObispoValley": "SLO",
}

MODELS = [
    ("GPT-3.5 FT",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt35ftv4_trial5_*.json"))[-1], std_keys),
     "#E69F00"),
    ("GPT-4o",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt4o_trial5_*.json"))[-1], std_keys),
     "#56B4E9"),
    ("GPT-4o FT",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt4oftv4_trial5_*.json"))[-1], std_keys),
     "#0072B2"),
    ("GPT-4.1",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt41_trial5_*.json"))[-1], std_keys),
     "#009E73"),
    ("GPT-4.1 FT",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt41ftv4_trial5_*.json"))[-1], std_keys),
     "#D55E00"),
    ("GPT-5.5",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_gpt55_trial5_*.json"))[-1], std_keys),
     "#CC79A7"),
    ("o3",
     load_ckpt_dict("results/checkpoint_o3_finetuned_20260312_174405.json", o3_keys),
     "#000000"),
    ("Sonnet 4.6",
     load_ckpt_dict(sorted(glob.glob("results/checkpoint_sonnet46_trial5_*.json"))[-1], std_keys),
     "#C9A800"),
    ("Opus 4.7\n(vision)",
     load_opus_preds(),
     "#666666"),
]

# CSV-based binary predictions (score >= 0.5 -> Yes, else No)
# Matches the Rocs_ scoring used in Table 4 and ROC/PRC figures for consistency.
_CSV_GSP = {
    "BigValley":           "BigValley",
    "EastContraCosta":     "East Contra Costa",
    "Fillmore":            "Fillmore",
    "SonomaValley":        "Sonoma",
    "SanLuisObispoValley": "San Luis Obispo",
}

def _load_csv_bin(csv_path):
    df   = pd.read_csv(csv_path)
    sc   = [c for c in df.columns if c.startswith("Rocs_")][0]
    gcol = "GSP" if "GSP" in df.columns else df.columns[1]
    return {cname: ["Yes" if s >= 0.5 else "No"
                    for s in df[df[gcol] == _CSV_GSP[cname]][sc]]
            for cname in GSP_CNAMES}

def _load_opus_csv_bin():
    slugs = {"BigValley": "bigvalley", "EastContraCosta": "eastcontracosta",
             "Fillmore": "fillmore", "SonomaValley": "sonoma",
             "SanLuisObispoValley": "sanluisobispovalley"}
    out = {}
    for cname, slug in slugs.items():
        df = pd.read_csv(sorted(glob.glob(
            f"results/results_opus47_vision_{slug}_*.csv"))[-1])
        sc = [c for c in df.columns if c.startswith("Rocs_")][0]
        out[cname] = ["Yes" if s >= 0.5 else "No" for s in df[sc]]
    return out

MODELS_BIN = [
    ("GPT-3.5 FT",
     _load_csv_bin(sorted(glob.glob("results/results_gpt35ftv4_trial5_*.csv"))[-1]),
     "#E69F00"),
    ("GPT-4o",
     _load_csv_bin(sorted(glob.glob("results/results_gpt4o_trial5_*.csv"))[-1]),
     "#56B4E9"),
    ("GPT-4o FT",
     _load_csv_bin(sorted(glob.glob("results/results_gpt4oftv4_trial5_*.csv"))[-1]),
     "#0072B2"),
    ("GPT-4.1",
     _load_csv_bin(sorted(glob.glob("results/results_gpt41_trial5_*.csv"))[-1]),
     "#009E73"),
    ("GPT-4.1 FT",
     _load_csv_bin(sorted(glob.glob("results/results_gpt41ftv4_trial5_*.csv"))[-1]),
     "#D55E00"),
    ("GPT-5.5",
     _load_csv_bin(sorted(glob.glob("results/results_gpt55_trial5_*.csv"))[-1]),
     "#CC79A7"),
    ("o3",
     _load_csv_bin("results/results_o3_finetuned_20260312_174405.csv"),
     "#000000"),
    ("Sonnet 4.6",
     _load_csv_bin(sorted(glob.glob("results/results_sonnet46_trial5_*.csv"))[-1]),
     "#C9A800"),
    ("Opus 4.7\n(vision)",
     _load_opus_csv_bin(),
     "#666666"),
]

n_models = len(MODELS)
x        = np.arange(len(CLASSES))
width    = 0.08
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

def draw_class_accuracy_panel(ax, true_list, pred_by_model, title,
                              ylabel=True, include_overall=False, show_labels=False,
                              tick_fs=14, label_fs=16):
    n_per_cls = {cls: sum(t == cls for t in true_list) for cls in CLASSES}
    n_total   = sum(n_per_cls.values())

    x_pos = np.arange(len(CLASSES) + (1 if include_overall else 0))

    for mi, (name, _, color) in enumerate(MODELS):
        rec  = per_class_recall(true_list, pred_by_model[name])
        vals = [rec[cls] for cls in CLASSES]

        if include_overall:
            preds = pred_by_model[name]
            n_correct = sum(t == p for t, p in zip(true_list, preds) if t in CLASSES)
            vals_plot = vals + [n_correct / n_total]
        else:
            vals_plot = vals

        bars = ax.bar(x_pos + offsets[mi], vals_plot, width, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.4, label=name)
        if show_labels:
            for bar, val in zip(bars, vals_plot):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                            f"{val:.0%}", ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color=color)

    tick_labels = [f"{cls}\n(n={n_per_cls[cls]})" for cls in CLASSES]
    if include_overall:
        tick_labels += [f"Overall\n(n={n_total})"]
        ax.axvline(len(CLASSES) - 0.5, color="#aaaaaa", lw=1.2, linestyle="--", zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel("Accuracy", fontsize=label_fs, fontweight="bold")

# overall: all 5 GSPs pooled
true_overall     = sum((rubric_by_gsp[c] for c in GSP_CNAMES), [])
pred_overall     = {name: sum((preds[c] for c in GSP_CNAMES), []) for name, preds, _ in MODELS}
bin_pred_overall = {name: sum((preds[c] for c in GSP_CNAMES), []) for name, preds, _ in MODELS_BIN}

fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
draw_class_accuracy_panel(ax, true_overall, pred_overall, "Overall (all 5 Trial GSPs)",
                          include_overall=True, show_labels=True)
ax.set_title("Model Performance by Response Category", fontsize=19, fontweight="bold", pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=13, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), framealpha=0.92, ncol=5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig("images/class_accuracy_overall.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/class_accuracy_overall.png")

def draw_overall_panel(ax, true_list, pred_by_model, binary=False, tick_fs=15, label_fs=17):
    if binary:
        def _bin(lbl):
            return "Yes" if lbl == "Yes" else ("No+Somewhat" if lbl in ("No", "Somewhat") else None)
        true_eff  = [_bin(t) for t in true_list]
        pred_eff  = {nm: [_bin(p) for p in preds] for nm, preds in pred_by_model.items()}
        valid_cls = {"Yes", "No+Somewhat"}
    else:
        true_eff  = true_list
        pred_eff  = pred_by_model
        valid_cls = set(CLASSES)

    n_total = sum(1 for t in true_eff if t in valid_cls)
    names   = [name.replace("\n", " ") for name, _, _ in MODELS]
    colors  = [c for _, _, c in MODELS]
    accs    = [
        sum(t == p for t, p in zip(true_eff, pred_eff[name]) if t in valid_cls) / n_total
        for name, _, _ in MODELS
    ]

    xpos = np.arange(len(MODELS))
    for mi, (name, _, color) in enumerate(MODELS):
        bar = ax.bar(xpos[mi], accs[mi], color=color, alpha=0.88,
                     edgecolor="white", linewidth=0.4, label=name)
        ax.text(xpos[mi], accs[mi] + 0.012, f"{accs[mi]:.0%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=tick_fs, rotation=35, ha="right")
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_ylabel("Accuracy", fontsize=label_fs, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

# per-GSP: 2x3 grid (5 panels + 1 legend slot)
fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=150)
axes = axes.flatten()

for panel_idx, (cname, label) in enumerate(zip(GSP_CNAMES, GSP_LABELS)):
    pred_gsp = {name: preds[cname] for name, preds, _ in MODELS}
    draw_class_accuracy_panel(axes[panel_idx], rubric_by_gsp[cname], pred_gsp,
                              label, ylabel=(panel_idx % 3 == 0), show_labels=False,
                              tick_fs=18, label_fs=20)
    axes[panel_idx].set_title(label, fontsize=19, fontweight="bold")
    axes[panel_idx].text(0.02, 1.0, f'{chr(ord("a") + panel_idx)}.',
                         transform=axes[panel_idx].transAxes,
                         fontsize=18, fontweight="bold", va="bottom")

draw_overall_panel(axes[5], true_overall, pred_overall, binary=False)
axes[5].set_title("Overall", fontsize=19, fontweight="bold")
axes[5].text(0.02, 1.0, "f.", transform=axes[5].transAxes,
             fontsize=18, fontweight="bold", va="bottom")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", fontsize=14, framealpha=0.92,
           ncol=5, title="Model", title_fontsize=14)
fig.suptitle("Model Performance by Response Category", fontsize=22, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.savefig("images/class_accuracy_by_gsp.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/class_accuracy_by_gsp.png")

# Figures 3 and 4: binary (Yes vs. No+Somewhat) accuracy
BIN_CLASSES = ["Yes", "No+Somewhat"]

def binarize(label):
    return "Yes" if label == "Yes" else ("No+Somewhat" if label in ("No", "Somewhat") else None)

def draw_binary_panel(ax, true_list, pred_by_model, title, ylabel=True,
                      include_overall=False, show_labels=False,
                      tick_fs=14, label_fs=16):
    true_bin  = [binarize(t) for t in true_list]
    n_per_cls = {cls: sum(t == cls for t in true_bin if t) for cls in BIN_CLASSES}
    n_total   = sum(n_per_cls.values())

    x_pos = np.arange(len(BIN_CLASSES) + (1 if include_overall else 0))
    for mi, (name, _, color) in enumerate(MODELS):
        pred_bin = [binarize(p) for p in pred_by_model[name]]
        vals = [
            sum(p == cls for t, p in zip(true_bin, pred_bin) if t == cls) /
            n_per_cls[cls] if n_per_cls[cls] else np.nan
            for cls in BIN_CLASSES
        ]
        if include_overall:
            n_correct = sum(t == p for t, p in zip(true_bin, pred_bin) if t)
            vals = vals + [n_correct / n_total]
        bars = ax.bar(x_pos + offsets[mi], vals, width, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.4, label=name)
        if show_labels:
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                            f"{val:.0%}", ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color=color)

    tick_labels = [f"{cls}\n(n={n_per_cls.get(cls, 0)})" for cls in BIN_CLASSES]
    if include_overall:
        tick_labels += [f"Overall\n(n={n_total})"]
        ax.axvline(len(BIN_CLASSES) - 0.5, color="#aaaaaa", lw=1.2, linestyle="--", zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel("Accuracy", fontsize=label_fs, fontweight="bold")

# overall binary
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
draw_binary_panel(ax, true_overall, bin_pred_overall, "Overall - Binary (Yes vs. No+Somewhat)",
                  include_overall=True, show_labels=True)
ax.set_title("Model Performance by Response Category", fontsize=19, fontweight="bold", pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=13, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), framealpha=0.92, ncol=5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig("images/binary_accuracy_overall.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/binary_accuracy_overall.png")

# per-GSP binary
fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=150)
axes = axes.flatten()
for panel_idx, (cname, label) in enumerate(zip(GSP_CNAMES, GSP_LABELS)):
    pred_gsp = {name: preds[cname] for name, preds, _ in MODELS_BIN}
    draw_binary_panel(axes[panel_idx], rubric_by_gsp[cname], pred_gsp,
                      label, ylabel=(panel_idx % 3 == 0), show_labels=False,
                      tick_fs=18, label_fs=20)
    axes[panel_idx].set_title(label, fontsize=19, fontweight="bold")
    axes[panel_idx].text(0.02, 1.0, f'{chr(ord("a") + panel_idx)}.',
                         transform=axes[panel_idx].transAxes,
                         fontsize=18, fontweight="bold", va="bottom")
draw_overall_panel(axes[5], true_overall, bin_pred_overall, binary=True)
axes[5].set_title("Overall", fontsize=19, fontweight="bold")
axes[5].text(0.02, 1.0, "f.", transform=axes[5].transAxes,
             fontsize=18, fontweight="bold", va="bottom")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", fontsize=14, framealpha=0.92,
           ncol=5, title="Model", title_fontsize=14)
fig.suptitle("Model Performance by GSP", fontsize=22, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.savefig("images/binary_accuracy_by_gsp.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/binary_accuracy_by_gsp.png")

# Figures: overall accuracy by GSP (no class breakdown), binary + 3-class
for binary, ylabel_str, suptitle_str, outfile in [
    (True,  "Binary Accuracy",  "Model Performance by GSP",               "images/binary_accuracy_by_gsp_overall.png"),
    (False, "3-Class Accuracy", "Model Performance by GSP",                "images/class_accuracy_by_gsp_overall.png"),
]:
    _pred_src  = MODELS_BIN     if binary else MODELS
    _pred_pool = bin_pred_overall if binary else pred_overall
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=150)
    axes = axes.flatten()
    for panel_idx, (cname, label) in enumerate(zip(GSP_CNAMES, GSP_LABELS)):
        pred_gsp = {name: preds[cname] for name, preds, _ in _pred_src}
        draw_overall_panel(axes[panel_idx], rubric_by_gsp[cname], pred_gsp, binary=binary)
        axes[panel_idx].set_title(label, fontsize=19, fontweight="bold")
        axes[panel_idx].text(0.02, 1.0, f'{chr(ord("a") + panel_idx)}.',
                             transform=axes[panel_idx].transAxes,
                             fontsize=18, fontweight="bold", va="bottom")
        if panel_idx % 3 == 0:
            axes[panel_idx].set_ylabel(ylabel_str, fontsize=20, fontweight="bold")
    draw_overall_panel(axes[5], true_overall, _pred_pool, binary=binary)
    axes[5].set_title("Overall", fontsize=19, fontweight="bold")
    axes[5].text(0.02, 1.0, "f.", transform=axes[5].transAxes,
                 fontsize=18, fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=14, framealpha=0.92,
               ncol=5, title="Model", title_fontsize=14)
    fig.suptitle(suptitle_str, fontsize=22, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")

# Figure 5: confusion matrices, 3x3 grid (one panel per model)
true_all   = sum((rubric_by_gsp[c] for c in GSP_CNAMES), [])
active_idx = [i for i, t in enumerate(true_all) if t in CLASSES]
true_active = [true_all[i] for i in active_idx]

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 13), dpi=150,
                         constrained_layout=True)

for mi, (name, preds_by_gsp, color) in enumerate(MODELS):
    row, col    = divmod(mi, ncols)
    ax          = axes[row, col]
    pred_all    = sum((preds_by_gsp[c] for c in GSP_CNAMES), [])
    pred_active = [pred_all[i] for i in active_idx]
    mat_count   = confusion_matrix_3class(true_active, pred_active)
    row_sums    = mat_count.sum(axis=1, keepdims=True).astype(float)
    mat_frac    = np.where(row_sums > 0, mat_count / row_sums, 0.0)

    ax.imshow(mat_frac, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for r in range(3):
        for c in range(3):
            frac    = mat_frac[r, c]
            count   = mat_count[r, c]
            bg_dark = frac > 0.55
            txt_col = "white" if bg_dark else "#222222"
            ax.text(c, r, f"{frac:.0%}\n(n={count})",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=txt_col, linespacing=1.4)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_title(name.replace("\n", " "), fontsize=15, fontweight="bold",
                 color=color, pad=8)
    if row == nrows - 1:
        ax.set_xticklabels(CLASSES, fontsize=13, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_yticklabels(CLASSES, fontsize=13)
    else:
        ax.set_yticklabels([])

    for d in range(3):
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                                   fill=False, edgecolor="#FF6B35", lw=2.0))

fig.suptitle("Confusion Matrices (Row-Normalized)", fontsize=19, fontweight="bold")
fig.supxlabel("Predicted Response", fontsize=15, fontweight="bold")
fig.supylabel("True Response", fontsize=15, fontweight="bold")

plt.savefig("images/confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/confusion_matrices.png")
