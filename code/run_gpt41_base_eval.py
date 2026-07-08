"""
Evaluate GPT-4.1 base (gpt-4.1-2025-04-14) on 5 trial GSPs.
Uses same retrieval pipeline and system prompt as all other models.
"""
import re
import os
import json
import pickle
import time
import glob
from datetime import datetime

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from prompts_2 import prompts

openai_key = os.environ["OPENAI_API_KEY"]

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
EMB_CACHE_DIR = "results/embeddings_fullset"
MODEL = "gpt-4.1-2025-04-14"
MODEL_NAME = "gpt41"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
TOP_N = 15
no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
N_QUESTIONS = sum(1 for i in range(1, 71) if i not in no_test)

TRIAL_GSPS = [
    (1,  "BigValley",           "BigValley",           "1_BigValley_DraftGSP_ScoringRubric.csv"),
    (14, "EastContraCosta",     "East Contra Costa",   "14_EastContraCosta_DraftGSP_ScoringRubric.csv"),
    (15, "Fillmore",            "Fillmore",             "15_Fillmore_DraftGSP_ScoringRubric.csv"),
    (30, "SonomaValley",        "Sonoma",               "30_SonomaValley_DraftGSP_ScoringRubric.csv"),
    (50, "SanLuisObispoValley", "San Luis Obispo",      "50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv"),
]

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

os.makedirs("results", exist_ok=True)

def build_bm25(chunks):
    return BM25Okapi([c.lower().split() for c in chunks])

def get_embedding(text):
    return OpenAI(api_key=openai_key).embeddings.create(
        input=[text.replace("\n", " ")], model="text-embedding-3-large"
    ).data[0].embedding

_ce_path = "models/cross_encoder_gsp" if os.path.exists("models/cross_encoder_gsp") \
           else "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(_ce_path)

def find_most_relevant_pages(chunks, embeddings, question, bm25_index, top_n=TOP_N, n_cands=25):
    q_emb = get_embedding(question)
    cos_scores = np.array([cosine_similarity([q_emb], [e])[0][0] for e in embeddings])
    bm25_scores = np.array(bm25_index.get_scores(question.lower().split()))
    cos_n = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min() + 1e-8)
    bm25_n = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    combined = 0.5 * cos_n + 0.5 * bm25_n
    cand_idx = np.argsort(combined)[::-1][:min(n_cands, len(chunks))]
    scores = cross_encoder.predict([[question, chunks[i]] for i in cand_idx])
    top_idx = cand_idx[np.argsort(scores)[::-1][:top_n]]
    return "\n".join(chunks[i] for i in top_idx)

def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).replace("\r", " ")

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

def load_rubric_answers(rubric_filename):
    rubric = HumanRubric(os.path.join(RUBRIC_DIR, rubric_filename))
    answers = rubric["Answer"].drop(no_test, errors="ignore")
    return ["NotApplicable" if str(v) == "Not Applicable" else v for v in answers]

# load cached embeddings
print("Loading cached embeddings...")
gsp_data = {}
for gid, cname, dname, _ in TRIAL_GSPS:
    with open(os.path.join(EMB_CACHE_DIR, f"{gid}_{cname}.pkl"), "rb") as f:
        # only load embedding caches you generated yourself; pickle can execute arbitrary code
        chunks, embeddings = pickle.load(f)
    gsp_data[cname] = {"chunks": list(chunks), "embeddings": embeddings,
                       "bm25": build_bm25(chunks)}
    print(f"  [{gid:2d}] {dname}: {len(chunks)} chunks")

# checkpoint
existing = sorted(glob.glob(f"results/checkpoint_{MODEL_NAME}_trial5_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"\nResuming: {checkpoint_file}")
else:
    run_id = f"{MODEL_NAME}_trial5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"\nStarting: {run_id}")

checkpoint = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else {}
active_qs = [i for i in range(1, 71) if i not in no_test]

print(f"\n{'='*60}")
print(f"GPT-4.1 Base - 51 questions x 5 GSPs = 255 total")
print(f"{'='*60}")

for gid, cname, dname, rubric_file in TRIAL_GSPS:
    d = gsp_data[cname]
    responses = checkpoint.get(cname, [])
    start_idx = len(responses)
    if start_idx == N_QUESTIONS:
        print(f"[{dname}] already complete")
        continue
    print(f"\n[{dname}] {start_idx}/{N_QUESTIONS} done - continuing...")
    for idx, qi in enumerate(active_qs[start_idx:], start=start_idx):
        section = find_most_relevant_pages(
            d["chunks"], d["embeddings"], prompts[qi - 1], d["bm25"])
        def _call(s=section, q=prompts[qi - 1]):
            c = OpenAI(api_key=openai_key)
            resp = c.chat.completions.create(
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
        checkpoint[cname] = responses
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
        answered = idx + 1
        if answered % 10 == 0 or answered == N_QUESTIONS:
            line = next((l.strip()[7:].strip() for l in resp.split("\n")
                         if l.strip().upper().startswith("ANSWER:")), "?")
            print(f"  [{dname}] {answered}/{N_QUESTIONS} (Q{qi}): {line}")
    print(f"  [{dname}] done.")

# save CSV
all_human, all_probs, all_gsp, all_gsp_id = [], [], [], []
for gid, cname, dname, rubric_file in TRIAL_GSPS:
    human = load_rubric_answers(rubric_file)
    probs = extract_yes_probabilities(checkpoint[cname])
    all_human += human
    all_probs += probs
    all_gsp += [dname] * len(probs)
    all_gsp_id += [gid] * len(probs)

score_col = f"Rocs_{run_id}"
df = pd.DataFrame({"GSP_ID": all_gsp_id, "GSP": all_gsp,
                   "Human Answers": all_human, score_col: all_probs})
csv_path = f"results/results_{run_id}.csv"
df.to_csv(csv_path, index=False)

df["true_bin"] = df["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
df["pred_bin"] = df[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
df["correct"] = df["true_bin"] == df["pred_bin"]
overall = df["correct"].mean()

print(f"\n{'='*55}")
print(f"GPT-4.1 Base - Overall: {overall:.1%}  ({df['correct'].sum()}/{len(df)})")
print(f"Saved: {csv_path}")
for g, a in df.groupby("GSP")["correct"].mean().items():
    print(f"  {g:<22s} {a:.1%}")
print(f"{'='*55}")
