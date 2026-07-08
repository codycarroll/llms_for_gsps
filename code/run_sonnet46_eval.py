import re
import os
import json
import pickle
import glob as _glob
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import pandas as pd
from prompts_2 import prompts

# keys
openai_key = os.environ["OPENAI_API_KEY"]
anthropic_key = os.environ["ANTHROPIC_API_KEY"]

# constants
PAGES_DIR = os.path.expanduser("~/Desktop/gsps_all/GSP_Pages")
RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
EMB_CACHE_DIR = "results/embeddings_fullset"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
TOP_N = 15
no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
N_QUESTIONS = sum(1 for i in range(1, 71) if i not in no_test)  # 51

TRIAL_GSPS = [
    (1,  "BigValley",           "BigValley",         "1_BigValley_DraftGSP_pages.txt",            "1_BigValley_DraftGSP_ScoringRubric.csv"),
    (14, "EastContraCosta",     "East Contra Costa", "14_EastContraCosta_DraftGSP_pages.txt",     "14_EastContraCosta_DraftGSP_ScoringRubric.csv"),
    (15, "Fillmore",            "Fillmore",           "15_Fillmore_DraftGSP_pages.txt",            "15_Fillmore_DraftGSP_ScoringRubric.csv"),
    (30, "SonomaValley",        "Sonoma",             "30_SonomaValley_DraftGSP_pages.txt",        "30_SonomaValley_DraftGSP_ScoringRubric.csv"),
    (50, "SanLuisObispoValley", "San Luis Obispo",    "50_SanLuisObispoValley_DraftGSP_pages.txt", "50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv"),
]

# model
_ce_path = "models/cross_encoder_gsp" if os.path.exists("models/cross_encoder_gsp") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(_ce_path)
print(f"Cross-encoder: {_ce_path}")

# functions
def split_into_chunks(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS])
            for i in range(0, len(words), step)
            if words[i:i + CHUNK_WORDS]]

def get_embeddings(full_text, batch_size=500):
    chunks = split_into_chunks(full_text)
    client = OpenAI(api_key=openai_key)
    all_emb = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.replace("\n", " ") for c in chunks[i:i + batch_size]]
        response = client.embeddings.create(input=batch, model="text-embedding-3-large")
        all_emb += [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
    return chunks, all_emb

def get_embedding(text):
    client = OpenAI(api_key=openai_key)
    return client.embeddings.create(
        input=[text.replace("\n", " ")], model="text-embedding-3-large"
    ).data[0].embedding

def build_bm25_index(chunks):
    return BM25Okapi([c.lower().split() for c in chunks])

def find_most_relevant_pages(chunks, embeddings, question, bm25_index=None):
    q_emb = get_embedding(question)
    cos_scores = np.array([cosine_similarity([q_emb], [e])[0][0] for e in embeddings])
    if bm25_index is not None:
        bm25_scores = np.array(bm25_index.get_scores(question.lower().split()))
        cos_n = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min() + 1e-8)
        bm25_n = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        combined = 0.5 * cos_n + 0.5 * bm25_n
    else:
        combined = cos_scores
    n_cands = min(25, len(chunks))
    cand_idx = np.argsort(combined)[::-1][:n_cands]
    scores = cross_encoder.predict([[question, chunks[i]] for i in cand_idx])
    top_idx = cand_idx[np.argsort(scores)[::-1][:TOP_N]]
    return "\n".join(chunks[i] for i in top_idx)

def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).replace("\r", " ")

def gde_answer(section_and_question: str) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
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
        ),
        messages=[{"role": "user", "content": clean_text(section_and_question)}],
    )
    return response.content[0].text

def HumanRubric(rubric_file):
    df = pd.read_csv(rubric_file)
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]
    return df[1:]

def load_rubric_answers(rubric_filename):
    rubric = HumanRubric(os.path.join(RUBRIC_DIR, rubric_filename))
    answers = rubric["Answer"]
    answers = answers.drop(no_test, errors="ignore")
    answers = ["NotApplicable" if str(item) == "Not Applicable" else item for item in answers]
    return list(answers)

def extract_yes_probabilities(responses):
    confidence_mapping = {"100%": 1.0, "85%": 0.85, "75%": 0.75, "60%": 0.60, "50%": 0.50}
    probs = []
    for response in responses:
        answer_line = next((l.strip()[7:].strip() for l in response.split("\n")
                            if l.strip().upper().startswith("ANSWER:")), response.strip())
        parts = answer_line.split(", ")
        answer = parts[0].strip()
        confidence = parts[-1].strip()
        prob = confidence_mapping.get(confidence, 0.5)
        probs.append(prob if answer == "Yes" else 1 - prob)
    return probs

# load or compute embeddings
print("\nEmbeddings")
gsp_data = {}
for gid, cache_name, display_name, pages_file, rubric_file in TRIAL_GSPS:
    cache_path = os.path.join(EMB_CACHE_DIR, f"{gid}_{cache_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            # only load embedding caches you generated yourself; pickle can execute arbitrary code
            chunks, embeddings = pickle.load(f)
        print(f"  [{gid:2d}] {display_name}: loaded {len(chunks)} chunks from cache")
    else:
        print(f"  [{gid:2d}] {display_name}: computing embeddings...", end=" ")
        with open(os.path.join(PAGES_DIR, pages_file), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        chunks, embeddings = get_embeddings(text)
        with open(cache_path, "wb") as f:
            pickle.dump((chunks, embeddings), f)
        print(f"{len(chunks)} chunks cached.")
    gsp_data[cache_name] = {
        "chunks": list(chunks), "embeddings": embeddings,
        "bm25": build_bm25_index(chunks),
        "display_name": display_name, "rubric_file": rubric_file,
    }

# checkpoint / resume
run_prefix = "sonnet46_trial5"
existing = sorted(_glob.glob(f"results/checkpoint_{run_prefix}_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"\nResuming: {checkpoint_file}")
else:
    run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"\nStarting new run: {run_id}")

checkpoint = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else {}
print(f"Already complete: {list(checkpoint.keys())}")

# run evaluation
PROGRESS_EVERY = 10
for gid, cache_name, display_name, pages_file, rubric_file in TRIAL_GSPS:
    if cache_name in checkpoint:
        print(f"[{display_name}] skipping, already done ({len(checkpoint[cache_name])} responses)")
        continue

    d = gsp_data[cache_name]
    print(f"\n{'='*60}")
    print(f"[{display_name}] starting, {N_QUESTIONS} questions")
    print(f"{'='*60}")

    responses = []
    answered = 0
    for i in range(1, 71):
        if i not in no_test:
            section = find_most_relevant_pages(d["chunks"], d["embeddings"], prompts[i - 1], bm25_index=d["bm25"])
            responses.append(gde_answer(section + prompts[i - 1]))
            answered += 1
            if answered % PROGRESS_EVERY == 0 or answered == N_QUESTIONS:
                print(f"  [{display_name}] {answered}/{N_QUESTIONS} (Q{i})")

    checkpoint[cache_name] = responses
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)
    print(f"  [{display_name}] done. Checkpoint saved.")

# assemble results
print(f"\n{'='*60}\nAssembling results...\n{'='*60}")
all_human, all_probs, all_gsp, all_gsp_id = [], [], [], []

for gid, cache_name, display_name, pages_file, rubric_file in TRIAL_GSPS:
    human = load_rubric_answers(rubric_file)
    probs = extract_yes_probabilities(checkpoint[cache_name])
    if len(human) != len(probs):
        print(f"WARNING: {display_name} length mismatch, human={len(human)}, probs={len(probs)}")
        continue
    all_human += human
    all_probs += probs
    all_gsp += [display_name] * len(probs)
    all_gsp_id += [gid] * len(probs)

score_col = f"Rocs_{run_id}"
df = pd.DataFrame({"GSP_ID": all_gsp_id, "GSP": all_gsp, "Human Answers": all_human, score_col: all_probs})

csv_path = f"results/results_{run_id}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}  ({len(df)} rows)")

raw_path = f"results/raw_{run_id}.json"
with open(raw_path, "w") as f:
    json.dump(dict(checkpoint), f, indent=2)
print(f"Saved {raw_path}")

# accuracy summary
df["true_bin"] = df["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
df["pred_bin"] = df[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
df["correct"] = df["true_bin"] == df["pred_bin"]

overall_acc = df["correct"].mean()
print(f'\nOverall accuracy (Yes vs Somewhat+No): {overall_acc:.1%}  ({df["correct"].sum()}/{len(df)})')
print("(o3_finetuned baseline on same 5 GSPs: 76.5%)")
print("\nPer-GSP accuracy:")
print(df.groupby("GSP")["correct"].mean().sort_values().apply(lambda x: f"{x:.1%}").to_string())

model_answers = []
for responses in checkpoint.values():
    for resp in responses:
        line = next((l.strip()[7:].strip() for l in resp.split("\n")
                     if l.strip().upper().startswith("ANSWER:")), "")
        model_answers.append(line.split(",")[0].strip())
print("\nModel answer distribution:")
print(pd.Series(model_answers).value_counts().to_string())
