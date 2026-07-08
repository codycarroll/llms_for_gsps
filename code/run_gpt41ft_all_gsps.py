"""
Evaluate GPT-4.1 FT (ft:gpt-4.1-2025-04-14:personal:gspv4:DbO4oSN8) on all GSPs
except those in SKIP_IDS / SKIP_NAMES.

Retrieval is page-level cosine similarity. Pages are pre-extracted text files in
PAGES_DIR. Embeddings are computed on first run and cached in EMB_DIR.
Writes a checkpoint JSON and a results CSV to results/.
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
EMB_DIR = "results/gsp_embeddings"
TOP_N = 10  # pages retrieved per question

# GSP IDs to skip entirely
SKIP_IDS = {
    # Already evaluated (trial GSPs - results in checkpoint_gpt41ftv4_trial5_*.json)
    1,    # BigValley
    14,   # EastContraCosta
    15,   # Fillmore
    30,   # SonomaValley
    50,   # SanLuisObispoValley
    # Excluded by design
    24,   # BedfordColdwater - rubric length mismatch
    57,   # SantaYnezRiverValley - corrupt/duplicate
    58,   # SantaYnezRiverValley - corrupt/duplicate
}

no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
N_QUESTIONS = sum(1 for i in range(1, 71) if i not in no_test)  # 51

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
                    help="Run on one GSP only (for cost estimation)")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

client = OpenAI(api_key=openai_key)

# helpers
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

# discover GSPs
print("Discovering GSPs...")

# pages_map: gid -> (gsp_name, pages_path)
# gsp_name is the filename stem before "_pages.txt"
pages_map = {}
for fname in os.listdir(PAGES_DIR):
    if fname.endswith("_pages.txt"):
        m = re.match(r"(\d+)", fname)
        if m:
            gid = int(m.group(1))
            gsp_name = fname.replace("_pages.txt", "")
            pages_map[gid] = (gsp_name, os.path.join(PAGES_DIR, fname))

# rubric_map: gid -> rubric_path
rubric_map = {}
for fname in os.listdir(RUBRIC_DIR):
    if fname.endswith(".csv"):
        m = re.match(r"(\d+)", fname)
        if m:
            rubric_map[int(m.group(1))] = os.path.join(RUBRIC_DIR, fname)

gsp_ids = sorted(set(pages_map) & set(rubric_map) - SKIP_IDS)
print(f"Found {len(gsp_ids)} GSPs to evaluate (after {len(SKIP_IDS)} exclusions)")

# checkpoint
existing = sorted(glob.glob(f"results/checkpoint_{MODEL_NAME}_allgsps_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"Resuming: {checkpoint_file}")
else:
    run_id = f"{MODEL_NAME}_allgsps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"Starting: {run_id}")

checkpoint = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else {}
active_qs = [i for i in range(1, 71) if i not in no_test]

# evaluation loop
if args.test:
    print("*** TEST MODE - stopping after first GSP ***")

print(f'\n{"="*60}')
print(f"GPT-4.1 FT - {len(gsp_ids)} GSPs x {N_QUESTIONS} questions")
print(f'{"="*60}')

gsps_run = 0
for gid in gsp_ids:
    gsp_name, pages_path = pages_map[gid]
    rubric_path = rubric_map[gid]

    responses = checkpoint.get(gsp_name, [])
    start_idx = len(responses)

    if start_idx == N_QUESTIONS:
        print(f"[{gid:2d}] {gsp_name} - already complete")
        continue

    print(f"\n[{gid:2d}] {gsp_name} - {start_idx}/{N_QUESTIONS} done")

    try:
        pages, embeddings = load_or_compute_embeddings(gsp_name, pages_path)
    except Exception as e:
        print(f"  Failed to load embeddings: {e} - skipping")
        continue

    for idx, qi in enumerate(active_qs[start_idx:], start=start_idx):
        section = find_most_relevant_pages(pages, embeddings, prompts[qi - 1])

        def _call(s=section, q=prompts[qi - 1]):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": clean_text(s + q)},
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
    gsps_run += 1
    if args.test:
        print(f"\nTest complete - check platform.openai.com/usage for cost.")
        break

# build and save results CSV
print(f"\nBuilding results CSV...")

all_gsp, all_gsp_id, all_human, all_probs = [], [], [], []

for gid in gsp_ids:
    gsp_name, _ = pages_map[gid]
    rubric_path = rubric_map[gid]
    if gsp_name not in checkpoint:
        continue

    human = load_rubric_answers(rubric_path)
    probs = extract_yes_probabilities(checkpoint[gsp_name])

    if len(human) != len(probs):
        print(f"  Warning: {gsp_name}: rubric {len(human)} rows vs {len(probs)} responses "
              f"- skipping from CSV")
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

df_valid = df[df["Human Answers"].isin(["Yes", "No", "Somewhat"])].copy()
df_valid["true_bin"] = df_valid["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
df_valid["pred_bin"] = df_valid[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
df_valid["correct"] = df_valid["true_bin"] == df_valid["pred_bin"]
overall = df_valid["correct"].mean()

print(f'\n{"="*60}')
print(f"GPT-4.1 FT All-GSPs - Binary accuracy: {overall:.1%}  "
      f'({df_valid["correct"].sum()}/{len(df_valid)})')
print(f'GSPs scored: {df["GSP_ID"].nunique()}')
print(f"Saved: {csv_path}")
for g, a in df_valid.groupby("GSP")["correct"].mean().items():
    print(f"  {g:<40s} {a:.1%}")
print(f'{"="*60}')
