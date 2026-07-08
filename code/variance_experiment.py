"""
Repeated-inference variance experiment for GPT-4.1 FT.

Runs the full 51-question inference N_RUNS times on the 5 trial GSPs, holding
retrieval fixed (page embeddings cached; each question's top-10 context is
computed once and reused across all runs). The chat completion call is left
identical to production: it passes no temperature and no seed, so it uses
OpenAI's default temperature of 1.0, and all run-to-run variation reflects the
LLM sampling stochasticity a real deployer would experience.

Retrieval matches run_gpt41ft_all_gsps.py: page-level cosine, TOP_N=10.
Writes a checkpoint JSON, a tidy long CSV, and a per-GSP summary CSV to results/.
"""
import re
import os
import json
import time
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from prompts_2 import prompts

# config
openai_key = os.environ["OPENAI_API_KEY"]

MODEL = os.environ.get("GSP_FT_MODEL", "ft:gpt-4.1-2025-04-14:personal:gspv4:DbO4oSN8")
MODEL_NAME = "gpt41ft"

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
PAGES_DIR = os.path.expanduser("~/Desktop/gsps_all/GSP_Pages")
EMB_DIR = "results/gsp_embeddings"   # trial GSP page embeddings cached here on first run
TOP_N = 10                           # pages retrieved per question (matches deployment)

# Trial GSPs (evaluation set): gid, gsp_name (page/emb stem), rubric filename
TRIAL_GSPS = [
    (1,  "1_BigValley_DraftGSP",            "1_BigValley_DraftGSP_ScoringRubric.csv"),
    (14, "14_EastContraCosta_DraftGSP",     "14_EastContraCosta_DraftGSP_ScoringRubric.csv"),
    (15, "15_Fillmore_DraftGSP",            "15_Fillmore_DraftGSP_ScoringRubric.csv"),
    (30, "30_SonomaValley_DraftGSP",        "30_SonomaValley_DraftGSP_ScoringRubric.csv"),
    (50, "50_SanLuisObispoValley_DraftGSP", "50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv"),
]

no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
active_qs = [i for i in range(1, 71) if i not in no_test]  # 51 questions
N_QUESTIONS = len(active_qs)

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
parser.add_argument("--runs", type=int, default=5, help="Number of repeated inference runs (default 5)")
parser.add_argument("--metrics-only", action="store_true",
                    help="Skip inference; recompute tables from newest checkpoint")
args = parser.parse_args()
N_RUNS = args.runs

os.makedirs("results", exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)
client = OpenAI(api_key=openai_key)

# helpers (identical to run_gpt41ft_all_gsps.py)
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

def load_or_compute_embeddings(gsp_name, pages_path):
    emb_path = os.path.join(EMB_DIR, f"{gsp_name}_embeddings.json")
    with open(pages_path, "r", encoding="utf-8") as f:
        pages = f.read().split("\n\n")
    if os.path.exists(emb_path):
        with open(emb_path, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]
        return pages, embeddings
    print(f"  Computing embeddings for {gsp_name} ({len(pages)} pages)...")
    embeddings = [get_embedding(p) for p in pages]
    with open(emb_path, "w") as f:
        json.dump(embeddings, f)
    print(f"  Cached embeddings to {emb_path}")
    return pages, embeddings

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

# checkpoint discovery
existing = sorted(glob.glob("results/checkpoint_variance_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"Resuming checkpoint: {checkpoint_file}")
else:
    run_id = f"variance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"Starting: {run_id}")

# checkpoint[gsp_name] = list of runs; each run = list of <=51 response strings
checkpoint = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else {}

# inference loop
if not args.metrics_only:
    print(f'\n{"="*64}')
    print(f"Variance experiment - {N_RUNS} runs x {len(TRIAL_GSPS)} GSPs x {N_QUESTIONS} questions")
    print(f"Model: {MODEL}  (temperature=default 1.0, no seed)")
    print(f'{"="*64}')

    for gid, gsp_name, _ in TRIAL_GSPS:
        pages_path = os.path.join(PAGES_DIR, f"{gsp_name}_pages.txt")
        runs = checkpoint.get(gsp_name, [])
        # ensure N_RUNS slots exist
        while len(runs) < N_RUNS:
            runs.append([])
        if all(len(r) == N_QUESTIONS for r in runs[:N_RUNS]):
            print(f"\n[{gid:2d}] {gsp_name} - all {N_RUNS} runs complete")
            checkpoint[gsp_name] = runs
            continue

        print(f"\n[{gid:2d}] {gsp_name}")
        pages, embeddings = load_or_compute_embeddings(gsp_name, pages_path)

        # Retrieval computed ONCE per question, reused across all runs.
        print(f"  Retrieving fixed top-{TOP_N} context for {N_QUESTIONS} questions...")
        sections = [find_most_relevant_pages(pages, embeddings, prompts[qi - 1])
                    for qi in active_qs]

        for r in range(N_RUNS):
            start_idx = len(runs[r])
            if start_idx == N_QUESTIONS:
                continue
            print(f"  Run {r+1}/{N_RUNS} - resuming at Q{start_idx+1}")
            for idx in range(start_idx, N_QUESTIONS):
                qi = active_qs[idx]

                def _call(s=sections[idx], q=prompts[qi - 1]):
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": clean_text(s + q)},
                        ],
                        max_tokens=1024,
                    )
                    return resp.choices[0].message.content

                runs[r].append(with_retry(_call))
                checkpoint[gsp_name] = runs
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint, f)
                answered = idx + 1
                if answered % 10 == 0 or answered == N_QUESTIONS:
                    print(f"    Q{qi:2d} ({answered}/{N_QUESTIONS}): {parse_answer(runs[r][idx])}")
        print(f"  [{gsp_name}] done.")

# metrics
print(f'\n{"="*64}\nComputing variance metrics\n{"="*64}')

def binarize_pred(prob):
    return "Yes" if prob >= 0.5 else "No"

long_rows = []       # tidy per-run/per-question
summary_rows = []    # per-GSP supplementary table
pool_single_accs = None   # accumulate per-run accuracy across pooled questions

# For the pooled row we track, per run, correct/total over all GSPs
pool_correct = [0] * N_RUNS
pool_total = 0
pool_mv_correct = 0
pool_unan_agree = 0
pool_unan_total = 0

for gid, gsp_name, rubric_file in TRIAL_GSPS:
    runs = checkpoint.get(gsp_name, [])
    runs = [r for r in runs if len(r) == N_QUESTIONS][:N_RUNS]
    if len(runs) < N_RUNS:
        print(f"  Warning: {gsp_name}: only {len(runs)}/{N_RUNS} complete runs - skipping")
        continue

    human = load_rubric_answers(os.path.join(RUBRIC_DIR, rubric_file))
    if len(human) != N_QUESTIONS:
        print(f"  Warning: {gsp_name}: rubric {len(human)} rows vs {N_QUESTIONS} - skipping")
        continue

    # per-run continuous scores and binary predictions: shape [run][question]
    probs = [extract_yes_probabilities(r) for r in runs]          # N_RUNS x 51
    preds = [[binarize_pred(p) for p in run_probs] for run_probs in probs]

    # valid = questions with a human Yes/No/Somewhat label
    valid = [i for i in range(N_QUESTIONS) if human[i] in ("Yes", "No", "Somewhat")]
    true_bin = {i: ("Yes" if human[i] == "Yes" else "No") for i in valid}

    # per-run single-pass accuracy over valid questions
    run_accs = []
    for r in range(N_RUNS):
        correct = sum(1 for i in valid if preds[r][i] == true_bin[i])
        run_accs.append(correct / len(valid))
        pool_correct[r] += correct
    pool_total += len(valid)

    mean_acc = float(np.mean(run_accs))
    sd_acc = float(np.std(run_accs, ddof=1))   # sample SD across runs

    # majority-vote (5-run) binary prediction per question
    mv_correct = 0
    for i in valid:
        yes_votes = sum(1 for r in range(N_RUNS) if preds[r][i] == "Yes")
        mv_pred = "Yes" if yes_votes > N_RUNS / 2 else "No"
        mv_correct += (mv_pred == true_bin[i])
    mv_acc = mv_correct / len(valid)
    pool_mv_correct += mv_correct

    # unanimous agreement rate: fraction of ALL 51 questions where the binary
    # prediction is identical across all N_RUNS runs (decision stability)
    unanimous = sum(1 for i in range(N_QUESTIONS)
                    if len({preds[r][i] for r in range(N_RUNS)}) == 1)
    unan_rate = unanimous / N_QUESTIONS
    pool_unan_agree += unanimous
    pool_unan_total += N_QUESTIONS

    summary_rows.append({
        "GSP": gsp_name.replace("_DraftGSP", "").split("_", 1)[1],
        "GSP_ID": gid,
        "N_runs": N_RUNS,
        "N_questions": len(valid),
        "Mean single-run acc": round(mean_acc, 4),
        "SD single-run acc": round(sd_acc, 4),
        "Min single-run acc": round(min(run_accs), 4),
        "Max single-run acc": round(max(run_accs), 4),
        "Majority-vote acc": round(mv_acc, 4),
        "Unanimous agreement rate": round(unan_rate, 4),
    })

    # tidy long rows
    for r in range(N_RUNS):
        for i in range(N_QUESTIONS):
            long_rows.append({
                "GSP": gsp_name, "GSP_ID": gid, "run": r + 1,
                "question": active_qs[i],
                "answer_3class": parse_answer(runs[r][i]),
                "yes_prob": probs[r][i], "pred_bin": preds[r][i],
                "human": human[i],
            })

    print(f"  [{gid:2d}] {gsp_name}: mean={mean_acc:.1%}  SD={sd_acc:.1%}  "
          f"MV={mv_acc:.1%}  unanimous={unan_rate:.1%}")

# pooled row across all trial GSPs
if pool_total > 0:
    pool_run_accs = [pool_correct[r] / pool_total for r in range(N_RUNS)]
    summary_rows.append({
        "GSP": "All trial GSPs (pooled)",
        "GSP_ID": -1,
        "N_runs": N_RUNS,
        "N_questions": pool_total,
        "Mean single-run acc": round(float(np.mean(pool_run_accs)), 4),
        "SD single-run acc": round(float(np.std(pool_run_accs, ddof=1)), 4),
        "Min single-run acc": round(min(pool_run_accs), 4),
        "Max single-run acc": round(max(pool_run_accs), 4),
        "Majority-vote acc": round(pool_mv_correct / pool_total, 4),
        "Unanimous agreement rate": round(pool_unan_agree / pool_unan_total, 4),
    })

# save
summary_df = pd.DataFrame(summary_rows)
summary_path = "results/variance_metrics_by_gsp.csv"
summary_df.to_csv(summary_path, index=False)

long_df = pd.DataFrame(long_rows)
long_path = f"results/variance_runs_long_{run_id}.csv"
long_df.to_csv(long_path, index=False)

print(f'\n{"="*64}')
print(summary_df.to_string(index=False))
print(f"\nSaved supplementary table: {summary_path}")
print(f"Saved tidy per-run data:   {long_path}")
print(f'{"="*64}')
