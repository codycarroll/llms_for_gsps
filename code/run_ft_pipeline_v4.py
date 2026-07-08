"""
Fine-tune v4 for GPT-4.1, GPT-4o, and GPT-3.5 on 5 new training GSPs.

Training GSPs: Yolo, San Jacinto, Modesto, Scott River Valley, Santa Rosa Plain
Eval GSPs: Big Valley, East Contra Costa, Fillmore, Sonoma, San Luis Obispo

Confidence mapping (same as v3):
  Yes      Extremely Confident, 100%
  Somewhat Fairly Confident, 75%
  No       Very Confident, 85%
"""
import re
import os
import json
import time
from collections import Counter
from datetime import datetime

import pandas as pd
from openai import OpenAI

from prompts_2 import prompts

# config
openai_key = os.environ["OPENAI_API_KEY"]

RUBRIC_DIR = os.path.expanduser("~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV")
FT_JSONL_PATH = "results/ft_v4_training.jsonl"
FILE_ID_PATH = "results/ft_v4_file_id.txt"

no_test = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]

TRAIN_GSPS = [
    (3,  "Yolo",             "3_Yolo_DraftGSP_ScoringRubric.csv"),
    (45, "SanJacinto",       "45_SanJacinto_DraftGSP_ScoringRubric.csv"),
    (46, "Modesto",          "46_Modesto_DraftGSP_ScoringRubric.csv"),
    (60, "ScottRiverValley", "60_ScottRiverValley_DraftGSP_Scoringrubric.csv"),
    (56, "SantaRosaPlain",   "56_SantaRosaPlain_DraftGSP_ScoringRubric.csv"),
]

BASE_MODELS = [
    ("gpt41",  "gpt-4.1-2025-04-14",      "gspv4",   "results/ft_v4_gpt41_job_id.txt",  "results/ft_v4_gpt41_model_id.txt"),
    ("gpt4o",  "gpt-4o-2024-08-06",        "gsp4ov4", "results/ft_v4_gpt4o_job_id.txt",  "results/ft_v4_gpt4o_model_id.txt"),
    ("gpt35",  "gpt-3.5-turbo-0125",       "gsp35v4", "results/ft_v4_gpt35_job_id.txt",  "results/ft_v4_gpt35_model_id.txt"),
]

ANSWER_CONF = {
    "Yes":      "Extremely Confident, 100%",
    "Somewhat": "Fairly Confident, 75%",
    "No":       "Very Confident, 85%",
}

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


def clean_text(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).replace("\r", " ")


def HumanRubric(path):
    df = pd.read_csv(path)
    df = df.iloc[10:, 3:].reset_index().drop("index", axis=1)
    df.columns = df.iloc[0]
    return df[1:]


# build the training JSONL
print("\n" + "="*60)
print("Building v4 training JSONL")
print("="*60)

all_examples = []
for gid, cname, rubric_file in TRAIN_GSPS:
    rubric = HumanRubric(os.path.join(RUBRIC_DIR, rubric_file))
    active = rubric.drop(no_test, errors="ignore")
    gsp_examples, skipped = [], 0
    for idx, row in active.iterrows():
        answer = str(row.get("Answer", "")).strip()
        relevant_txt = str(row.get("Relevant Text from GSP", "")).strip()
        if answer not in ANSWER_CONF or relevant_txt in ("nan", ""):
            skipped += 1
            continue
        qi = int(idx)
        if qi < 1 or qi > len(prompts):
            skipped += 1
            continue
        question = prompts[qi - 1]
        section = relevant_txt.replace("&&", "\n\n").strip()
        conf_label = ANSWER_CONF[answer]
        gsp_examples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": clean_text(section + "\n\n" + question)},
                {"role": "assistant", "content": f"ANSWER: {answer}, {conf_label}"},
            ]
        })
    print(f"  [{gid:2d}] {cname:<18s} {len(gsp_examples):3d} examples ({skipped} skipped)")
    all_examples.extend(gsp_examples)

print(f"\nTotal: {len(all_examples)} training examples")
conf_dist = Counter(e["messages"][2]["content"].split(", ", 1)[1] for e in all_examples)
print("Confidence distribution:")
for k, v in sorted(conf_dist.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

with open(FT_JSONL_PATH, "w") as f:
    for ex in all_examples:
        f.write(json.dumps(ex) + "\n")
print(f"\nSaved: {FT_JSONL_PATH}")

# upload the training file once, shared across all 3 base models
print("\n" + "="*60)
print("Upload training file (shared across all 3 base models)")
print("="*60)

client = OpenAI(api_key=openai_key)

if os.path.exists(FILE_ID_PATH):
    with open(FILE_ID_PATH) as f:
        file_id = f.read().strip()
    print(f"Reusing existing upload: {file_id}")
else:
    print("Uploading...")
    with open(FT_JSONL_PATH, "rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")
    file_id = upload.id
    with open(FILE_ID_PATH, "w") as f:
        f.write(file_id)
    print(f"Uploaded: {file_id}  (saved to {FILE_ID_PATH})")

# submit fine-tuning jobs for all 3 base models
print("\n" + "="*60)
print("Submit fine-tuning jobs for all 3 base models")
print("="*60)

job_ids = {}
for short, base_model, suffix, job_id_file, model_id_file in BASE_MODELS:
    print(f"\n  {base_model}:")
    if os.path.exists(job_id_file):
        with open(job_id_file) as f:
            job_id = f.read().strip()
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"    Existing job: {job_id}  status: {job.status}")
    else:
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
            suffix=suffix,
        )
        job_id = job.id
        with open(job_id_file, "w") as f:
            f.write(job_id)
        print(f"    Job ID: {job_id}  (saved to {job_id_file})")
        print(f"    Status: {job.status}")
    job_ids[short] = job_id

# poll until all 3 jobs complete
print("\n" + "="*60)
print("Polling until all 3 jobs complete")
print("="*60)

pending = {short: (job_id, model_id_file)
           for (short, _, _, job_id_file, model_id_file), job_id
           in zip(BASE_MODELS, job_ids.values())
           if not os.path.exists(model_id_file)}

# re-build pending from files properly
pending = {}
for short, base_model, suffix, job_id_file, model_id_file in BASE_MODELS:
    if os.path.exists(model_id_file):
        with open(model_id_file) as f:
            print(f"  {short} already complete: {f.read().strip()}")
    else:
        with open(job_id_file) as f:
            pending[short] = (f.read().strip(), model_id_file)

while pending:
    time.sleep(60)
    ts = datetime.now().strftime("%H:%M:%S")
    done = []
    for short, (job_id, model_id_file) in pending.items():
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{ts}] {short:<8s} {job.status:<12s} {job.fine_tuned_model or '(pending)'}")
        if job.status == "succeeded":
            with open(model_id_file, "w") as f:
                f.write(job.fine_tuned_model)
            print(f"    Saved model ID to {model_id_file}")
            done.append(short)
        elif job.status in ("failed", "cancelled"):
            print(f"    Job {job.status}!")
            done.append(short)
    for s in done:
        del pending[s]

print("\nAll jobs finished.")
print("\nSaved model IDs:")
for short, base_model, suffix, job_id_file, model_id_file in BASE_MODELS:
    if os.path.exists(model_id_file):
        with open(model_id_file) as f:
            print(f"  {short}: {f.read().strip()}")
