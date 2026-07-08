"""
ROC + PRC comparison figure and Yes/No/Somewhat per-class recall figure.
Saves images/roc_prc_comparison.png and images/class_recall_comparison.png.
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]

TRIAL_GSPS = [
    (1,  "BigValley",           "1_BigValley_DraftGSP_ScoringRubric.csv"),
    (14, "EastContraCosta",     "14_EastContraCosta_DraftGSP_ScoringRubric.csv"),
    (15, "Fillmore",            "15_Fillmore_DraftGSP_ScoringRubric.csv"),
    (30, "SonomaValley",        "30_SonomaValley_DraftGSP_ScoringRubric.csv"),
    (50, "SanLuisObispoValley", "50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv"),
]

os.makedirs("images", exist_ok=True)

# helpers
def load_score_df(source):
    if isinstance(source, list):
        parts = []
        for p in source:
            d = pd.read_csv(p)
            sc = [c for c in d.columns if c.startswith("Rocs_")][0]
            parts.append(d[["GSP", "Human Answers"]].assign(score=d[sc]))
        return pd.concat(parts, ignore_index=True)
    d = pd.read_csv(source)
    sc = [c for c in d.columns if c.startswith("Rocs_")][0]
    return d[["GSP", "Human Answers"]].assign(score=d[sc])

def compute_curves(df):
    df = df[df["Human Answers"].isin(["Yes", "No", "Somewhat"])]
    y = (df["Human Answers"] == "Yes").astype(int)
    s = df["score"]
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return fpr, tpr, auc(fpr, tpr), rec, prec, auc(rec, prec)

def HumanRubric(path):
    df = pd.read_csv(path)
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]
    return df[1:]

def load_rubric_answers(rubric_filename):
    rubric = HumanRubric(os.path.join(RUBRIC_DIR, rubric_filename))
    answers = rubric["Answer"].drop(no_test, errors="ignore")
    return [str(v).strip() for v in answers]

def parse_answer(response):
    for line in str(response).split("\n"):
        if line.strip().upper().startswith("ANSWER:"):
            return line.strip()[7:].strip().split(",")[0].strip()
    return "Unknown"

# load score CSVs
opus_csvs = [
    sorted(glob.glob(f"results/results_opus47_vision_{g}_*.csv"))[-1]
    for g in ["bigvalley", "eastcontracosta", "fillmore", "sonoma", "sanluisobispovalley"]
]

df_o3      = load_score_df("results/results_o3_finetuned_20260312_174405.csv")
df_sonnet  = load_score_df(sorted(glob.glob("results/results_sonnet46_trial5_*.csv"))[-1])
df_gpt41   = load_score_df(sorted(glob.glob("results/results_gpt41_trial5_*.csv"))[-1])
df_gpt41ft = load_score_df(sorted(glob.glob("results/results_gpt41ftv4_trial5_*.csv"))[-1])
df_gpt4o   = load_score_df(sorted(glob.glob("results/results_gpt4o_trial5_*.csv"))[-1])
df_gpt4oft = load_score_df(sorted(glob.glob("results/results_gpt4oftv4_trial5_*.csv"))[-1])
df_gpt35ft = load_score_df(sorted(glob.glob("results/results_gpt35ftv4_trial5_*.csv"))[-1])
df_opus    = load_score_df(opus_csvs)
df_gpt55   = load_score_df(sorted(glob.glob("results/results_gpt55_trial5_*.csv"))[-1])

PAPER_MODELS = []

COMPUTED_MODELS = [
    ("GPT-3.5 FT",        df_gpt35ft, "#E69F00"),
    ("GPT-4o",            df_gpt4o,   "#56B4E9"),
    ("GPT-4o FT",         df_gpt4oft, "#0072B2"),
    ("GPT-4.1",           df_gpt41,   "#009E73"),
    ("GPT-4.1 FT",        df_gpt41ft, "#D55E00"),
    ("GPT-5.5",           df_gpt55,   "#CC79A7"),
    ("o3",                df_o3,      "#000000"),
    ("Sonnet 4.6",        df_sonnet,  "#C9A800"),
    ("Opus 4.7 (vision)", df_opus,    "#666666"),
]

# figure 1: ROC + PRC (all solid, color only)
fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(13, 9), dpi=150)

for ax in (ax_prc, ax_roc):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=17)
    ax.set_box_aspect(1)

# PRC
for label, _, _, prc_pts, prc_auc, color in PAPER_MODELS:
    xs, ys = zip(*prc_pts)
    ax_prc.plot(xs, ys, color=color, lw=2.0, label=f"{label} (AUCPR={prc_auc:.3f})")

for label, df_m, color in COMPUTED_MODELS:
    _, _, _, rec, prec, prc_auc = compute_curves(df_m)
    ax_prc.plot(rec, prec, color=color, lw=2.0, label=f"{label} (AUCPR={prc_auc:.3f})")

baseline = (df_o3[df_o3["Human Answers"].isin(["Yes", "No", "Somewhat"])]["Human Answers"] == "Yes").mean()
ax_prc.axhline(baseline, color="#888888", linestyle=":", lw=1.3,
               label=f"Random (AUCPR≈{baseline:.3f})")
ax_prc.set_xlabel("Recall", fontsize=19, fontweight="bold")
ax_prc.set_ylabel("Precision", fontsize=19, fontweight="bold")
ax_prc.text(0.5, 1.0, "Precision-Recall Curve", transform=ax_prc.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="center")
ax_prc.text(0.0, 1.0, "b.", transform=ax_prc.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="left")
ax_prc.set_xlim(-0.02, 1.02)
ax_prc.set_ylim(0, 1.05)

# ROC
for label, roc_pts, roc_auc, _, _, color in PAPER_MODELS:
    xs, ys = zip(*roc_pts)
    ax_roc.plot(xs, ys, color=color, lw=2.0, label=f"{label} (AUCROC={roc_auc:.3f})")

for label, df_m, color in COMPUTED_MODELS:
    fpr, tpr, roc_auc, _, _, _ = compute_curves(df_m)
    ax_roc.plot(fpr, tpr, color=color, lw=2.0, label=f"{label} (AUCROC={roc_auc:.3f})")

ax_roc.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1.3, label="Random (AUCROC=0.500)")
ax_roc.set_xlabel("False Positive Rate", fontsize=19, fontweight="bold")
ax_roc.set_ylabel("True Positive Rate", fontsize=19, fontweight="bold")
ax_roc.text(0.5, 1.0, "ROC Curve", transform=ax_roc.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="center")
ax_roc.text(0.0, 1.0, "a.", transform=ax_roc.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="left")
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(0, 1.05)

leg_kw = dict(fontsize=11, ncol=1, framealpha=0.92, edgecolor="#cccccc", handlelength=1.5)
ax_roc.legend(loc="lower right", **leg_kw)
ax_prc.legend(loc="lower left", **leg_kw)
plt.tight_layout()
plt.savefig("images/roc_prc_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/roc_prc_comparison.png")

# figure 1b: same but with titled subplots instead of a./b. labels
fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(13, 9), dpi=150)

for ax in (ax_prc, ax_roc):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=17)
    ax.set_box_aspect(1)

# PRC
for label, _, _, prc_pts, prc_auc, color in PAPER_MODELS:
    xs, ys = zip(*prc_pts)
    ax_prc.plot(xs, ys, color=color, lw=2.0, label=f"{label} (AUCPR={prc_auc:.3f})")

for label, df_m, color in COMPUTED_MODELS:
    _, _, _, rec, prec, prc_auc = compute_curves(df_m)
    ax_prc.plot(rec, prec, color=color, lw=2.0, label=f"{label} (AUCPR={prc_auc:.3f})")

ax_prc.axhline(baseline, color="#888888", linestyle=":", lw=1.3,
               label=f"Random (AUCPR≈{baseline:.3f})")
ax_prc.set_title("Precision-Recall Curve", fontsize=20, fontweight="bold", pad=10)
ax_prc.set_xlabel("Recall", fontsize=19, fontweight="bold")
ax_prc.set_ylabel("Precision", fontsize=19, fontweight="bold")
ax_prc.set_xlim(-0.02, 1.02)
ax_prc.set_ylim(0, 1.05)

# ROC
for label, roc_pts, roc_auc, _, _, color in PAPER_MODELS:
    xs, ys = zip(*roc_pts)
    ax_roc.plot(xs, ys, color=color, lw=2.0, label=f"{label} (AUCROC={roc_auc:.3f})")

for label, df_m, color in COMPUTED_MODELS:
    fpr, tpr, roc_auc, _, _, _ = compute_curves(df_m)
    ax_roc.plot(fpr, tpr, color=color, lw=2.0, label=f"{label} (AUCROC={roc_auc:.3f})")

ax_roc.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1.3, label="Random (AUCROC=0.500)")
ax_roc.set_title("ROC Curve", fontsize=20, fontweight="bold", pad=10)
ax_roc.set_xlabel("False Positive Rate", fontsize=19, fontweight="bold")
ax_roc.set_ylabel("True Positive Rate", fontsize=19, fontweight="bold")
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(0, 1.05)

leg_kw = dict(fontsize=11, ncol=1, framealpha=0.92, edgecolor="#cccccc", handlelength=1.5)
ax_roc.legend(loc="lower right", **leg_kw)
ax_prc.legend(loc="lower left", **leg_kw)
plt.tight_layout()
plt.savefig("images/roc_prc_comparison_titled.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/roc_prc_comparison_titled.png")

# print AUC table
print(f"\n{'Model':<24}  {'AUCROC':>8}  {'AUCPR':>8}")
print("-" * 44)
for label, _, roc_auc, _, prc_auc, _ in PAPER_MODELS:
    print(f"  {label:<22}  {roc_auc:>8.3f}  {prc_auc:>8.3f}  (digitized)")
for label, df_m, _ in COMPUTED_MODELS:
    _, _, roc_auc, _, _, prc_auc = compute_curves(df_m)
    print(f"  {label:<22}  {roc_auc:>8.3f}  {prc_auc:>8.3f}")

# figure 2: Yes / No / Somewhat per-class recall (computed models only)

# load all rubric answers (true labels) across 5 trial GSPs
all_true = []
for _, cname, rubric_file in TRIAL_GSPS:
    all_true.extend(load_rubric_answers(rubric_file))

# load predicted answers from each model's checkpoint
def load_preds_from_checkpoint(ckpt_file, gsp_keys):
    with open(ckpt_file) as f:
        ckpt = json.load(f)
    preds = []
    for key in gsp_keys:
        preds.extend([parse_answer(r) for r in ckpt[key]])
    return preds

def load_preds_opus():
    preds = []
    for fpath in [
        sorted(glob.glob("results/checkpoint_opus47_vision_bigvalley_*.json"))[-1],
        sorted(glob.glob("results/checkpoint_opus47_vision_eastcontracosta_*.json"))[-1],
        sorted(glob.glob("results/checkpoint_opus47_vision_fillmore_*.json"))[-1],
        sorted(glob.glob("results/checkpoint_opus47_vision_sonoma_*.json"))[-1],
        sorted(glob.glob("results/checkpoint_opus47_vision_sanluisobispovalley_*.json"))[-1],
    ]:
        with open(fpath) as f:
            responses = json.load(f)
        preds.extend([parse_answer(r) for r in responses])
    return preds

std_keys = ["BigValley", "EastContraCosta", "Fillmore", "SonomaValley", "SanLuisObispoValley"]
CKPT_MODELS = [
    ("GPT-3.5 FT",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt35ftv4_trial5_*.json"))[-1], std_keys),
     "#E69F00"),
    ("GPT-4o",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt4o_trial5_*.json"))[-1], std_keys),
     "#56B4E9"),
    ("GPT-4o FT",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt4oftv4_trial5_*.json"))[-1], std_keys),
     "#0072B2"),
    ("GPT-4.1",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt41_trial5_*.json"))[-1], std_keys),
     "#009E73"),
    ("GPT-4.1 FT",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt41ftv4_trial5_*.json"))[-1], std_keys),
     "#D55E00"),
    ("GPT-5.5",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_gpt55_trial5_*.json"))[-1], std_keys),
     "#CC79A7"),
    ("o3",
     load_preds_from_checkpoint("results/checkpoint_o3_finetuned_20260312_174405.json",
                                ["BigValley", "EastContraCosta", "Fillmore", "Sonoma", "SLO"]),
     "#000000"),
    ("Sonnet 4.6",
     load_preds_from_checkpoint(
         sorted(glob.glob("results/checkpoint_sonnet46_trial5_*.json"))[-1], std_keys),
     "#C9A800"),
    ("Opus 4.7 (vision)", load_preds_opus(), "#666666"),
]

# compute per-class recall for each model
CLASSES = ["Yes", "Somewhat", "No"]
n_models = len(CKPT_MODELS)
recall = {cls: [] for cls in CLASSES}
n_per_cls = {}

for cls in CLASSES:
    idx = [i for i, t in enumerate(all_true) if t == cls]
    n_per_cls[cls] = len(idx)
    for name, preds, _ in CKPT_MODELS:
        correct = sum(preds[i] == cls for i in idx)
        recall[cls].append(correct / len(idx) if idx else 0.0)

print(f"\nPer-class counts in rubric: {n_per_cls}")
print(f"\n{'Model':<24}  {'Yes recall':>10}  {'Somewhat recall':>15}  {'No recall':>10}")
print("-" * 65)
for mi, (name, _, _) in enumerate(CKPT_MODELS):
    print(f"  {name:<22}  {recall['Yes'][mi]:>10.1%}  {recall['Somewhat'][mi]:>15.1%}  {recall['No'][mi]:>10.1%}")

# plot
x = np.arange(len(CLASSES))
n = n_models
width = 0.08
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)

for mi, (name, _, color) in enumerate(CKPT_MODELS):
    vals = [recall[cls][mi] for cls in CLASSES]
    bars = ax.bar(x + offsets[mi], vals, width, label=name,
                  color=color, alpha=0.88, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=color)

xlabel = [f"{cls}\n(n={n_per_cls[cls]})" for cls in CLASSES]
ax.set_xticks(x)
ax.set_xticklabels(xlabel, fontsize=16)
ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
ax.set_ylabel("Recall (fraction correct)", fontsize=16, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax.tick_params(labelsize=15)
ax.legend(fontsize=13, loc="upper right", framealpha=0.92, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("images/class_recall_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/class_recall_comparison.png")
