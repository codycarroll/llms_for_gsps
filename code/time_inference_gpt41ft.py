"""
Time pure inference (no embedding computation) for GPT-4.1 FT on 3 GSPs with
pre-cached embeddings.
"""
import re
import os
import json
import time

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from prompts_2 import prompts

# config
openai_key = os.environ["OPENAI_API_KEY"]

MODEL = os.environ.get("GSP_FT_MODEL", "ft:gpt-4.1-2025-04-14:personal:gspv4:DbO4oSN8")
PAGES_DIR = os.path.expanduser("~/Desktop/gsps_all/GSP_Pages")
EMB_DIR = "results/gsp_embeddings"
TOP_N = 10

# Small (~282 pages), medium (~468 pages), large (~676 pages)
TIMING_GSPS = [
    (3,  "3_Yolo_DraftGSP"),
    (9,  "9_Solano_DraftGSP"),
    (10, "10_Sutter_DraftGSP"),
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

client = OpenAI(api_key=openai_key)

# helpers
def get_embedding(text):
    text = text.strip().replace("\n", " ")
    if not text:
        return np.zeros(3072).tolist()
    resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return resp.data[0].embedding

def load_cached_embeddings(gsp_name, pages_path):
    emb_path = os.path.join(EMB_DIR, f"{gsp_name}_embeddings.json")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"No cached embeddings for {gsp_name}")
    with open(pages_path, "r", encoding="utf-8") as f:
        pages = f.read().split("\n\n")
    with open(emb_path, "r") as f:
        embeddings = json.load(f)
    if embeddings and not isinstance(embeddings[0], list):
        embeddings = [embeddings]
    return pages, embeddings

def find_most_relevant_pages(pages, embeddings, question):
    q_emb = get_embedding(question)
    scores = [cosine_similarity([q_emb], [e])[0][0] for e in embeddings]
    top_idx = np.argsort(scores)[::-1][:TOP_N]
    return "\n".join(pages[i] for i in top_idx)

def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).replace("\r", " ")

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

# timing run
print(f"Timing pure inference on {len(TIMING_GSPS)} GSPs (embeddings pre-cached)")
print(f"Model: {MODEL}")
print(f"Questions per GSP: {N_QUESTIONS}")
print("=" * 60)

gsp_times = []

for gid, gsp_name in TIMING_GSPS:
    pages_path = os.path.join(PAGES_DIR, f"{gsp_name}_pages.txt")
    pages, embeddings = load_cached_embeddings(gsp_name, pages_path)
    n_pages = len(pages)
    print(f"\n[{gid}] {gsp_name} ({n_pages} pages)")

    t_gsp_start = time.perf_counter()

    for idx, qi in enumerate(active_qs):
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

        with_retry(_call)

        answered = idx + 1
        if answered % 10 == 0 or answered == N_QUESTIONS:
            elapsed = (time.perf_counter() - t_gsp_start) / 60
            print(f"  Q{qi:2d} ({answered}/{N_QUESTIONS}) - {elapsed:.1f} min elapsed")

    gsp_elapsed = (time.perf_counter() - t_gsp_start) / 60
    gsp_times.append(gsp_elapsed)
    print(f"  [{gsp_name}] done - {gsp_elapsed:.2f} min")

# summary
print(f'\n{"=" * 60}')
print(f'{"GSP":<40}  {"Pages":>5}  {"Time (min)":>10}')
print("-" * 60)
for (gid, gsp_name), t in zip(TIMING_GSPS, gsp_times):
    pages_path = os.path.join(PAGES_DIR, f"{gsp_name}_pages.txt")
    with open(pages_path) as f:
        n_pages = len(f.read().split("\n\n"))
    print(f"{gsp_name:<40}  {n_pages:>5}  {t:>10.2f}")

avg = sum(gsp_times) / len(gsp_times)
print("-" * 60)
print(f'{"Average":<40}  {"":>5}  {avg:>10.2f}')
print(f"\nEstimated pure inference time per GSP: {avg:.2f} min")
print(f"Estimated embedding/overhead per GSP:  {8.0 - avg:.2f} min "
      f"(based on 8.0 min all-in average)")
