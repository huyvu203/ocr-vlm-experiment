# Joint OCR & DocQA Plan

Audience: Long (Textract lead) and Huy (GPT/VLM lead). Follow the shared setup, then execute the system-specific tasks. Each of you must deliver an individual report plus artifacts; afterward, combine your findings into a single comparative summary.

---

## 1. Objectives & Scope
- Keep things simple: compare two DocQA pipelines on the same questions.
  1. **Long**: run AWS Textract to OCR each page, then send the resulting text (plus lightweight structure) to a text-only LLM that answers the questions.
  2. **Huy**: send the page image + question directly to a vision-capable LLM (VLM) and capture its answer.
- Use DocQA datasets that already include question/answer labels so you can report exact-match / F1 without building new annotations.
- Measure per-question accuracy, latency, and cost for each pipeline and produce short write-ups plus a combined summary.
- Deliverables:
  - Individual reports (Long + Huy, 2–3 pages each) detailing pipeline design, metrics, and notable errors.
  - A joint memo (1–2 pages) after both runs, highlighting where each system excels/fails, recommended production choices, and next experiments.

---

## 2. Shared Environment Setup
1. **Python env**: Python 3.10+, `pip install -r requirements.txt` (create one if missing). Core deps: `boto3`, `datasets`, `Pillow`, `numpy`, `pandas`, `typer`, `rapidfuzz`, `sacrebleu`, `python-Levenshtein`, `openai` (or equivalent client), `huggingface_hub`.
2. **CLI auth**:
   - `aws configure` with Textract-enabled IAM credentials.
   - `huggingface-cli login`.
   - Configure OpenAI (or other VLM) key via `OPENAI_API_KEY`.
3. **Repo layout** (create if absent):
   ```
   data/
     raw/
     processed/
   outputs/
     textract/
     vlm/
   results/
   scripts/
   reports/
   ```

---

## 3. Dataset Acquisition (shared)
Use ready-made DocQA datasets so Long and Huy can focus on pipeline differences instead of dataset hunting. Download at least one of the following (all hosted on Hugging Face). Commands assume you already ran `huggingface-cli login`.

| Dataset | Hugging Face repo | What you get | Download command |
|---------|-------------------|--------------|------------------|
| **InfographicVQA** | `docvqa_infographics` | 5k+ infographic images with bounding-box metadata and QA pairs (`question`, `answer`). Great mix of text + charts. | ```bash\nmkdir -p data/infographicvqa && \\\nhuggingface-cli download docvqa_infographics --repo-type dataset \\\n  --local-dir data/infographicvqa \\\n  --include \"train/*\" \"val/*\"\n``` |
| **ChartQA** | `chartqa` | Chart images (bar, line, pie tables) with natural-language questions + numeric answers. Tests table reasoning. | ```bash\nmkdir -p data/chartqa && \\\nhuggingface-cli download chartqa --repo-type dataset \\\n  --local-dir data/chartqa \\\n  --include \"train/*\" \"val/*\"\n``` |
| **TextVQA** | `textvqa` | Real-world photos that require reading embedded text. Comes with `train_v1.0.json`, `val_v1.0.json` containing `image_id`, `question`, `answers`. | ```bash\nmkdir -p data/textvqa && \\\nhuggingface-cli download textvqa --repo-type dataset \\\n  --local-dir data/textvqa \\\n  --include \"train_images/*\" \"val_images/*\" \"train_v1.0.json\" \"val_v1.0.json\"\n``` |
| **DocVQA Task 1 (Document QA)** | `naver-clova-ix/cord-v2` (CORD) or `donut-data/docvqa_task1` | Scanned documents (receipts, forms) with key-value QA style questions. | ```bash\nmkdir -p data/docvqa_task1 && \\\nhuggingface-cli download donut-data/docvqa_task1 --repo-type dataset \\\n  --local-dir data/docvqa_task1\n``` |

How to prep them:
1. Use `datasets` to subsample ~500 QA pairs per dataset (e.g., `load_dataset(...).select(range(500))`) so runs stay quick.
2. Store images under `data/raw/<dataset>/...`; keep original filenames so question JSON references stay valid.
3. Build manifests:
   - `data/processed/manifests/dev.jsonl` / `test.jsonl` with `{ "page_id": "...", "image_path": "..." }`.
   - `data/processed/questions/{dev,test}.jsonl` copying `{page_id, question_id, question, answer}` exactly from the dataset fields (for TextVQA, pick the majority answer string).
4. Share these manifests with both students; they must not diverge.

---

## 4. Evaluation Workflow (both students)
1. **Shared schema**: keep things lightweight—each record needs `page_id`, `image_path`, plus any OCR text Long exports. Questions live in `data/processed/questions/{split}.jsonl`.
2. **Scripts to maintain**:
   - `02_run_textract.py` — Long’s OCR runner that writes `outputs/textract/{split}.jsonl`.
   - `03_run_vlm.py` — optional helper for Huy if he wants to cache raw OCR-style transcripts from the VLM (not required).
   - `05_answer_questions.py` — shared QA harness with modes:
     - `--mode textract_llm` (question + OCR text to a text LLM).
     - `--mode vlm_image` (question + image to a VLM).
   - `06_evaluate_qa.py` — simple metrics script that compares predictions vs ground truth (EM/F1) and aggregates latency/cost.
3. **Execution order**:
   1. Prepare manifests/questions (or reuse from dataset split).
   2. Long runs `02_run_textract.py` on dev/test to store OCR text snippets per page.
   3. Long calls `05_answer_questions.py --mode textract_llm ...` using those snippets and records answers.
   4. Huy directly calls `05_answer_questions.py --mode vlm_image ...` (no OCR needed) to produce answers.
   5. Both pass their prediction files to `06_evaluate_qa.py` to compute EM/F1 plus latency and token counts.
4. **Metric suite** (kept minimal):
   - QA accuracy: per-question Exact Match (EM) + token-level F1, aggregated per dataset and per question type (text vs numeric). Accept numeric answers within ±0.5%.
   - Operational metrics: average/median latency per question, tokens or images billed, and estimated USD cost per 100 questions.

---

## 5A. Long — Textract + Text LLM
1. **OCR run**
   ```bash
   python scripts/02_run_textract.py \
     --manifest data/processed/manifests/dev.jsonl \
     --output outputs/textract/dev_text.jsonl \
     --region us-east-1
   ```
   - For each page, save a single plain-text block (concatenate Textract LINEs) plus optional lightweight tables (Markdown). Keep it simple so prompts stay short.
2. **QA prompts**
   ```bash
   python scripts/05_answer_questions.py \
     --mode textract_llm \
     --context outputs/textract/dev_text.jsonl \
     --questions data/processed/questions/dev.jsonl \
     --model gpt-5-turbo \
     --output outputs/answers/textract_llm_dev.jsonl
   ```
   - Prompt template: question + extracted text, ask for JSON `{ "answer": "...", "confidence": 0-1 }`.
   - Repeat for the test split. Track tokens/latency (store in the JSON file).
3. **Deliverables**
   - `reports/long_report.md` summarizing EM/F1, average latency, token spend, and sample successes/failures.
   - Prediction files for dev/test plus a quick README describing how to rerun the two commands above.

---

## 5B. Huy — Direct VLM
1. **QA run (no OCR step)**
   ```bash
   python scripts/05_answer_questions.py \
     --mode vlm_image \
     --images-root data/raw \
     --questions data/processed/questions/dev.jsonl \
     --model gpt-5-vision \
     --output outputs/answers/vlm_dev.jsonl \
     --prompt-template prompts/qa_vlm.txt
   ```
   - Prompt template: question + image, respond with JSON `{ "answer": "...", "confidence": 0-1 }`.
   - Include 2–3 few-shot examples in the system prompt if the API allows.
2. **Retry policy**: if the VLM fails JSON validation, auto-retry once with a corrective instruction (“Return valid JSON only”).
3. **Deliverables**
   - `reports/huy_report.md` covering EM/F1, latency, cost per 100 questions, and representative examples where the VLM beats/falls behind the Textract+LLM pipeline.
   - Prediction files for dev/test (JSON lines) with timing + token metadata.

---

## 6. Joint Analysis & Combined Report
1. Collect the two prediction files for the **test** split (Textract+LLM and VLM).
2. Run `python scripts/06_evaluate_qa.py --pred textract=outputs/answers/textract_llm_test.jsonl --pred vlm=outputs/answers/vlm_test.jsonl --questions data/processed/questions/test.jsonl --output results/qa_comparison.json`.
3. Schedule a review session:
   - Walk through EM/F1 deltas per dataset and highlight cases where one pipeline consistently wins.
   - Build an error taxonomy (OCR omissions, VLM hallucinations, numeric slips, etc.).
4. Joint memo content (store as `reports/combined_summary.md`):
   - Executive summary with table of key metrics (EM, F1, latency, cost).
   - Recommendation matrix: when to use Textract+LLM vs direct VLM (e.g., cost-sensitive vs zero-setup scenarios).
   - Future work backlog (prompt tweaks, better context chunking, hybrid voting ideas).

---

## 7. Runbook (quick reference)
1. Prepare dataset manifests/questions (sample 500 Qs): `python scripts/01_prepare_dataset.py ...` (or manual export).
2. Long runs Textract: `python scripts/02_run_textract.py --manifest ... --output outputs/textract/{split}.jsonl`.
3. Long answers questions via LLM: `python scripts/05_answer_questions.py --mode textract_llm ...`.
4. Huy answers directly via VLM: `python scripts/05_answer_questions.py --mode vlm_image ...`.
5. Compare accuracy: `python scripts/06_evaluate_qa.py --pred textract=... --pred vlm=... --questions ...`.
6. Write individual reports (`reports/long_report.md`, `reports/huy_report.md`) and then a shared `reports/combined_summary.md`.

Stick to these steps so both individual and combined analyses stay comparable. Reach out to each other immediately if schema changes or dataset issues arise; no unilateral tweaks after Day 2.
