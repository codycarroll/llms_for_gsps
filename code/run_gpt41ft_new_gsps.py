"""Evaluate GPT-4.1 FT on 6 new GSPs not in the original all-GSP run.

Pages are extracted inline from PDFs in GSP_Drafts/ using pdfplumber.
Embeddings are cached in results/gsp_embeddings/.
"""
# run from the repo root: python3 "code/run_gpt41ft_new_gsps.py"
import re
import os
import json
import time
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pdfplumber
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score

from prompts_2 import prompts

# config
openai_key = os.environ["OPENAI_API_KEY"]

MODEL = os.environ.get("GSP_FT_MODEL", "ft:gpt-4.1-2025-04-14:personal:gspv4:DbO4oSN8")
MODEL_NAME = "gpt41ft"

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
DRAFTS_DIR = os.path.join(os.path.dirname(__file__), "..", "GSP_Drafts")
EMB_DIR = "results/gsp_embeddings"
METRICS_CSV = "results/gpt41ft_allgsps_per_gsp_metrics.csv"
TOP_N = 10

# hardcoded: (gsp_id, gsp_name_key, pdf_filename, rubric_filename)
NEW_GSPS = [
    (4,  "4_SouthAmerican_DraftGSP",
     "4_SouthAmerican_DraftGSP.pdf",
     "4_SouthAmerican_DraftGSP_ScoringRubric.csv"),
    (8,  "8_LosMolinos_DraftGSP",
     "8_LosMolinos_DraftGSP.pdf",
     "8_LosMolinos_DraftGSP_ScoringRubric.csv"),
    (48, "48_PleasantValley_DraftGSP",
     "48_PleasantValley_DraftGSP.pdf",
     "48_PleasantValley_DraftGSP_ScoringRubric.csv"),
    (57, "57_SantaYnezRiverValleyWestern_DraftGSP",
     "57_SantaYnezRiverValleyWestern_DraftGSP.pdf",
     "57_SantaYnezRiverValley_Western_DraftGSP_ScoringRubric.csv"),
    (58, "58_SantaYnezRiverValleyCentral_DraftGSP",
     "58_SantaYnezRiverValleyCentral_DraftGSP.pdf",
     "58_SantaYnezRiverValley_Central_DraftGSP_ScoringRubric.csv"),
    (59, "59_SantaYnezRiverValleyEastern_DraftGSP",
     "59_SantaYnezRiverValleyEastern_DraftGSP.pdf",
     "59_SantaYnezRiverValley_Eastern_DraftGSP_ScoringRubric.csv"),
]

no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
N_QUESTIONS = sum(1 for i in range(1, 71) if i not in no_test)  # 51
active_qs = [i for i in range(1, 71) if i not in no_test]

SYSTEM_PROMPT = (
    "You are a skeptical environmental scientist reviewing a section of a "
    "Groundwater Sustainability Plan (GSP).\n\n"
    "For each question, follow these steps:\n"
    "1. Quote the most relevant passage(s) from the provided text "
    "(or state 'No relevant text found').\n"
    "2. Briefly explain your reasoning.\n"
    "3. On the final line, give your answer in exactly this format:\n"
    "ANSWER: X, Z\n"
    "where X is Yes, No, or Somewhat, and Z is one of: "
    "Extremely Confident, 100% | Very Confident, 85% | "
    "Fairly Confident, 75% | Modest Confidence, 60% | Random Guess, 50%\n\n"
    "Only use 'Extremely Confident, 100%' if the answer is irrefutably "
    "supported by the text. Use Somewhat when the GSP partially addresses "
    "the criterion but not fully."
)

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true",
                    help="Run on first GSP only (cost estimation)")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

client = OpenAI(api_key=openai_key)

# helpers
def extract_pdf_pages(pdf_path):
    """Extract pages as list of strings using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
    return pages

def get_embedding(text):
    text = text.strip().replace("\n", " ")
    if not text:
        return np.zeros(3072).tolist()
    try:
        resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
        return resp.data[0].embedding
    except Exception as e:
        print(f"    Embedding error: {e}")
        return np.zeros(3072).tolist()

def load_or_compute_embeddings(gsp_name, pages):
    emb_path = os.path.join(EMB_DIR, f"{gsp_name}_embeddings.json")
    if os.path.exists(emb_path):
        with open(emb_path) as f:
            embeddings = json.load(f)
        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]
        print(f"  Loaded cached embeddings ({len(embeddings)} pages)")
        return embeddings
    print(f"  Computing embeddings ({len(pages)} pages)...")
    embeddings = [get_embedding(p) for p in pages]
    with open(emb_path, "w") as f:
        json.dump(embeddings, f)
    print(f"  Cached: {emb_path}")
    return embeddings

def find_most_relevant_pages(pages, embeddings, question, top_n=TOP_N):
    q_emb = get_embedding(question)
    scores = [cosine_similarity([q_emb], [e])[0][0] for e in embeddings]
    top_idx = np.argsort(scores)[::-1][:top_n]
    return "\n".join(pages[i] for i in top_idx)

def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).replace("\r", " ")

def parse_answer(response):
    for line in str(response).split("\n"):
        if line.strip().upper().startswith("ANSWER:"):
            return line.strip()[7:].strip().split(",")[0].strip()
    return "Unknown"

def extract_yes_probabilities(responses):
    conf = {"100%": 1.0, "85%": 0.85, "75%": 0.75, "60%": 0.60, "50%": 0.50}
    probs = []
    for r in responses:
        line = next((l.strip()[7:].strip() for l in r.split("\n")
                     if l.strip().upper().startswith("ANSWER:")), r.strip())
        parts = line.split(", ")
        prob = conf.get(parts[-1].strip(), 0.5)
        probs.append(prob if parts[0].strip() == "Yes" else 1 - prob)
    return probs

def with_retry(fn, max_retries=5, base_delay=15):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    retry {attempt+1}/{max_retries} in {delay}s: {e}")
            time.sleep(delay)

def HumanRubric(path):
    df = pd.read_csv(path)
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]
    return df[1:]

def load_rubric_answers(rubric_path):
    rubric = HumanRubric(rubric_path)
    answers = rubric["Answer"].drop(no_test, errors="ignore")
    return [str(v).strip() for v in answers]

# checkpoint
existing = sorted(glob.glob("results/checkpoint_gpt41ft_newgsps_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"Resuming: {checkpoint_file}")
else:
    run_id = f"gpt41ft_newgsps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"Starting: {run_id}")

checkpoint = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else {}

# evaluation loop
print(f'\n{"="*60}')
print(f"GPT-4.1 FT: {len(NEW_GSPS)} new GSPs x {N_QUESTIONS} questions")
print(f'{"="*60}')

for gid, gsp_name, pdf_file, rubric_file in NEW_GSPS:
    pdf_path = os.path.join(DRAFTS_DIR, pdf_file)
    rubric_path = os.path.join(RUBRIC_DIR, rubric_file)

    if not os.path.exists(pdf_path):
        print(f"[{gid:2d}] {gsp_name}: PDF not found: {pdf_path}")
        continue
    if not os.path.exists(rubric_path):
        print(f"[{gid:2d}] {gsp_name}: Rubric not found: {rubric_path}")
        continue

    responses = checkpoint.get(gsp_name, [])
    start_idx = len(responses)

    if start_idx == N_QUESTIONS:
        print(f"[{gid:2d}] {gsp_name}: already complete")
        continue

    print(f"\n[{gid:2d}] {gsp_name}: {start_idx}/{N_QUESTIONS} done")

    # extract pages from PDF
    print(f"  Extracting pages from PDF...")
    pages = extract_pdf_pages(pdf_path)
    print(f"  {len(pages)} pages extracted")

    embeddings = load_or_compute_embeddings(gsp_name, pages)

    for idx, qi in enumerate(active_qs[start_idx:], start=start_idx):
        section = find_most_relevant_pages(pages, embeddings, prompts[qi - 1])

        def _call(s=section, q=prompts[qi - 1]):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": clean_text(s + q)},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content

        resp = with_retry(_call)
        responses.append(resp)
        checkpoint[gsp_name] = responses
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)

        answered = idx + 1
        if answered % 10 == 0 or answered == N_QUESTIONS:
            print(f"  Q{qi:2d} ({answered}/{N_QUESTIONS}): {parse_answer(resp)}")

    print(f"  [{gsp_name}] done.")

    if args.test:
        print("\nTest mode: stopping after first GSP.")
        break

# build results CSV
print(f"\nBuilding results CSV...")

all_gsp, all_gsp_id, all_human, all_probs = [], [], [], []

for gid, gsp_name, _, rubric_file in NEW_GSPS:
    if gsp_name not in checkpoint:
        continue
    rubric_path = os.path.join(RUBRIC_DIR, rubric_file)
    human = load_rubric_answers(rubric_path)
    probs = extract_yes_probabilities(checkpoint[gsp_name])
    if len(human) != len(probs):
        print(f"  Warning: {gsp_name}: rubric {len(human)} rows vs {len(probs)} responses, skipping")
        continue
    all_gsp += [gsp_name] * len(probs)
    all_gsp_id += [gid] * len(probs)
    all_human += human
    all_probs += probs

score_col = f"Rocs_{run_id}"
df = pd.DataFrame({"GSP_ID": all_gsp_id, "GSP": all_gsp,
                   "Human Answers": all_human, score_col: all_probs})
csv_path = f"results/results_{run_id}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# per-GSP metrics
print(f"\nComputing per-GSP metrics...")

new_rows = []
for gsp_name in df["GSP"].unique():
    sub = df[df["GSP"] == gsp_name].copy()
    valid = sub[sub["Human Answers"].isin(["Yes", "No", "Somewhat"])].copy()
    n = len(valid)

    valid["true_bin"] = valid["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
    valid["pred_bin"] = valid[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
    valid["correct"] = valid["true_bin"] == valid["pred_bin"]
    acc_bin = valid["correct"].mean()

    all_labels = sub["Human Answers"].tolist()
    all_preds = [parse_answer(r) for r in checkpoint.get(gsp_name, [])]
    n_3 = len(all_labels)
    acc_3 = (sum(t == p for t, p in zip(all_labels, all_preds) if t in ["Yes", "No", "Somewhat"])
             / n) if n else 0

    y_true = (valid["Human Answers"] == "Yes").astype(int)
    if y_true.nunique() > 1:
        auc = roc_auc_score(y_true, valid[score_col])
    else:
        auc = float("nan")

    new_rows.append({
        "GSP": gsp_name,
        "N_Questions": n,
        "Accuracy_Binary": round(acc_bin, 4),
        "Accuracy_3Class": round(acc_3, 4),
        "AUCROC": round(auc, 4) if not np.isnan(auc) else float("nan"),
    })
    print(f"  {gsp_name:<45} Binary={acc_bin:.1%}  3-class={acc_3:.1%}  "
          f"ROC={auc:.3f}")

# update per-GSP metrics CSV
df_metrics = pd.read_csv(METRICS_CSV)
new_df = pd.DataFrame(new_rows)

# remove any existing rows for these GSPs (handles 59 update)
existing_names = set(new_df["GSP"])
df_metrics = df_metrics[~df_metrics["GSP"].isin(existing_names)]

df_metrics = pd.concat([df_metrics, new_df], ignore_index=True)
df_metrics = df_metrics.sort_values("GSP").reset_index(drop=True)
df_metrics.to_csv(METRICS_CSV, index=False)
print(f"\nUpdated: {METRICS_CSV}  ({len(df_metrics)} GSPs total)")

# summary
df_valid = df[df["Human Answers"].isin(["Yes", "No", "Somewhat"])].copy()
df_valid["true_bin"] = df_valid["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
df_valid["pred_bin"] = df_valid[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
df_valid["correct"] = df_valid["true_bin"] == df_valid["pred_bin"]
overall = df_valid["correct"].mean()

print(f'\n{"="*60}')
print(f'New GSPs - Binary accuracy: {overall:.1%}  '
      f'({df_valid["correct"].sum()}/{len(df_valid)})')
print(f"Checkpoint: {checkpoint_file}")
print(f"CSV:        {csv_path}")
print(f"Metrics:    {METRICS_CSV}")
print(f'{"="*60}')
