"""Run the Opus 4.7 vision evaluation on a single GSP PDF."""
import sys
import re
import os
import pickle
import base64
import json
import glob as _glob
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import pandas as pd
from prompts_2 import prompts

gsp_id, cache_name, display_name, pdf_path, rubric_file, sonnet_display = sys.argv[1:]
gsp_id = int(gsp_id)

openai_key = os.environ["OPENAI_API_KEY"]
anthropic_key = os.environ["ANTHROPIC_API_KEY"]

CHUNK_WORDS = 300
OVERLAP_WORDS = 50
TOP_N_CHUNKS = 15
TOP_N_PAGES = 5
RENDER_DPI = 100
no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]
N_QUESTIONS = sum(1 for i in range(1, 71) if i not in no_test)

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
CACHE_PATH = f"results/embeddings_vision/{gsp_id}_{cache_name}_vision.pkl"
os.makedirs("results/embeddings_vision", exist_ok=True)
os.makedirs("results", exist_ok=True)

_ce_path = "models/cross_encoder_gsp" if os.path.exists("models/cross_encoder_gsp") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(_ce_path)
print(f"[{display_name}] Cross-encoder: {_ce_path}")


def extract_chunks_with_pages(pdf_path):
    doc = fitz.open(pdf_path)
    word_page = []
    for pn in range(len(doc)):
        for word in doc[pn].get_text("text").split():
            word_page.append((word, pn))
    doc.close()
    step, chunks = CHUNK_WORDS - OVERLAP_WORDS, []
    for i in range(0, len(word_page), step):
        w = word_page[i:i + CHUNK_WORDS]
        if w:
            chunks.append((" ".join(x for x, _ in w),
                           Counter(p for _, p in w).most_common(1)[0][0]))
    return chunks

def get_embeddings_batched(cwp, batch_size=500):
    texts, client, all_emb = [t.replace("\n", " ") for t, _ in cwp], OpenAI(api_key=openai_key), []
    for i in range(0, len(texts), batch_size):
        r = client.embeddings.create(input=texts[i:i+batch_size], model="text-embedding-3-large")
        all_emb += [e.embedding for e in sorted(r.data, key=lambda x: x.index)]
    return all_emb

def get_query_embedding(text):
    return OpenAI(api_key=openai_key).embeddings.create(
        input=[text.replace("\n", " ")], model="text-embedding-3-large").data[0].embedding

def build_bm25(cwp):
    return BM25Okapi([t.lower().split() for t, _ in cwp])

def find_relevant_pages(cwp, embeddings, question, bm25):
    texts = [t for t, _ in cwp]
    q = get_query_embedding(question)
    cos = np.array([cosine_similarity([q], [e])[0][0] for e in embeddings])
    bm = np.array(bm25.get_scores(question.lower().split()))
    cn = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)
    bn = (bm - bm.min()) / (bm.max() - bm.min() + 1e-8)
    comb = 0.5*cn + 0.5*bn
    cand_idx = np.argsort(comb)[::-1][:min(25, len(texts))]
    ce_sc = cross_encoder.predict([[question, texts[i]] for i in cand_idx])
    top_idx = cand_idx[np.argsort(ce_sc)[::-1][:TOP_N_CHUNKS]]
    seen, pages = set(), []
    for i in top_idx:
        p = cwp[i][1]
        if p not in seen:
            seen.add(p); pages.append(p)
        if len(pages) == TOP_N_PAGES:
            break
    return pages

def render_pages(path, page_nums):
    doc = fitz.open(path)
    mat = fitz.Matrix(RENDER_DPI/72, RENDER_DPI/72)
    imgs = [doc[pn].get_pixmap(matrix=mat).tobytes("png") for pn in page_nums if 0 <= pn < len(doc)]
    doc.close()
    return imgs

def clean_text(t):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t).replace("\r", " ")

def gde_answer_vision(pdf_path, page_nums, question):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                "data": base64.b64encode(img).decode()}} for img in render_pages(pdf_path, page_nums)]
    content.append({"type": "text", "text": clean_text(question)})
    r = anthropic.Anthropic(api_key=anthropic_key).messages.create(
        model="claude-opus-4-7", max_tokens=1024,
        system=(
            "You are a skeptical environmental scientist reviewing pages from a "
            "Groundwater Sustainability Plan (GSP). The pages are provided as images "
            "and may contain text, figures, maps, and tables.\n\n"
            "For each question, follow these steps:\n"
            "1. Reference the most relevant content from the provided pages "
            "(text, figures, tables, or maps - or state 'No relevant content found').\n"
            "2. Briefly explain your reasoning.\n"
            "3. On the final line, give your answer in exactly this format:\n"
            "ANSWER: X, Z\n"
            "where X is Yes, No, or Somewhat, and Z is one of: "
            "Extremely Confident, 100% | Very Confident, 85% | "
            "Fairly Confident, 75% | Modest Confidence, 60% | Random Guess, 50%\n\n"
            "Only use 'Extremely Confident, 100%' if the answer is irrefutably "
            "supported by the content. Use Somewhat when the GSP partially addresses "
            "the criterion but not fully."
        ),
        messages=[{"role": "user", "content": content}])
    return r.content[0].text

def load_rubric_answers(fname):
    df = pd.read_csv(os.path.join(RUBRIC_DIR, fname))
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]; df = df[1:]
    ans = df["Answer"].drop(no_test, errors="ignore")
    return ["NotApplicable" if str(v) == "Not Applicable" else v for v in ans]

def extract_yes_probs(responses):
    conf = {"100%": 1.0, "85%": 0.85, "75%": 0.75, "60%": 0.60, "50%": 0.50}
    probs = []
    for r in responses:
        line = next((l.strip()[7:].strip() for l in r.split("\n")
                     if l.strip().upper().startswith("ANSWER:")), r.strip())
        parts = line.split(", ")
        p = conf.get(parts[-1].strip(), 0.5)
        probs.append(p if parts[0].strip() == "Yes" else 1-p)
    return probs


# load or compute embeddings
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        # only load embedding caches you generated yourself; pickle can execute arbitrary code
        cwp, embeddings = pickle.load(f)
    print(f"[{display_name}] Loaded {len(cwp)} chunks from cache.")
else:
    print(f"[{display_name}] Extracting chunks...", end=" ")
    cwp = extract_chunks_with_pages(pdf_path)
    print(f"{len(cwp)} chunks. Computing embeddings...", end=" ")
    embeddings = get_embeddings_batched(cwp)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((cwp, embeddings), f)
    print("cached.")

bm25 = build_bm25(cwp)

# checkpoint / resume
run_prefix = f"opus47_vision_{cache_name.lower()}"
existing = sorted(_glob.glob(f"results/checkpoint_{run_prefix}_*.json"))
if existing:
    checkpoint_file = existing[-1]
    run_id = checkpoint_file.replace("results/checkpoint_", "").replace(".json", "")
    print(f"[{display_name}] Resuming: {checkpoint_file}")
else:
    run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_file = f"results/checkpoint_{run_id}.json"
    print(f"[{display_name}] Starting: {run_id}")

responses = json.load(open(checkpoint_file)) if os.path.exists(checkpoint_file) else []
start_idx = len(responses)
active_qs = [i for i in range(1, 71) if i not in no_test]
print(f"[{display_name}] Already answered: {start_idx}/{N_QUESTIONS}")

for idx, qi in enumerate(active_qs[start_idx:], start=start_idx):
    page_nums = find_relevant_pages(cwp, embeddings, prompts[qi-1], bm25)
    responses.append(gde_answer_vision(pdf_path, page_nums, prompts[qi-1]))
    answered = idx + 1
    if answered % 10 == 0 or answered == N_QUESTIONS:
        with open(checkpoint_file, "w") as f:
            json.dump(responses, f)
        line = next((l.strip()[7:].strip() for l in responses[-1].split("\n")
                     if l.strip().upper().startswith("ANSWER:")), "?")
        print(f"  [{display_name}] {answered}/{N_QUESTIONS} (Q{qi}): {line}")

# save results
human = load_rubric_answers(rubric_file)
probs = extract_yes_probs(responses)
score_col = f"Rocs_{run_id}"
df = pd.DataFrame({"GSP_ID": gsp_id, "GSP": display_name, "Human Answers": human, score_col: probs})
csv_path = f"results/results_{run_id}.csv"
df.to_csv(csv_path, index=False)
with open(f"results/raw_{run_id}.json", "w") as f:
    json.dump(responses, f, indent=2)

df["true_bin"] = df["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No")
df["pred_bin"] = df[score_col].apply(lambda p: "Yes" if p >= 0.5 else "No")
df["correct"] = df["true_bin"] == df["pred_bin"]
vision_acc = df["correct"].mean()

df_s = pd.read_csv(sorted(_glob.glob("results/results_sonnet46_trial5_*.csv"))[-1])
df_s = df_s[df_s["GSP"] == sonnet_display]
sc = [c for c in df_s.columns if c.startswith("Rocs_")][0]
df_s["correct"] = (df_s["Human Answers"].apply(lambda x: "Yes" if x == "Yes" else "No") ==
                   df_s[sc].apply(lambda p: "Yes" if p >= 0.5 else "No"))
sonnet_acc = df_s["correct"].mean()

print(f'\n{"="*50}')
print(f'  {display_name}: vision {vision_acc:.1%} ({df["correct"].sum()}/51)  |  sonnet text {sonnet_acc:.1%} ({df_s["correct"].sum()}/51)')
print(f"  Saved: {csv_path}")
print(f'{"="*50}')
