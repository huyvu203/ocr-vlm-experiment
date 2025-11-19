# Instructions for Huy — GPT/VLM \+ QA Evaluation

## 1\. Objective

- Evaluate how a vision-capable LLM (GPT-5 Vision Preview or GPT-4o as fallback) performs on the same document types the Textract team is testing, with special attention to reasoning over tables.  
- Demonstrate that the VLM output can directly answer document-grounded questions (DocVQA-style) and compare that accuracy to Long’s Textract pipeline.  
- Explore prompt/formatting strategies that make LLM outputs easy to score against ground truth.

## 2\. Dataset Acquisition

- Use the exact DocQA datasets Long pulled. If you need to mirror them yourself, run one or both commands:  
    
  huggingface-cli download docvqa\_infographics \--repo-type dataset \\  
    
    \--local-dir data/infographicvqa \--include "train/\*" "val/\*"  
    
  huggingface-cli download chartqa \--repo-type dataset \\  
    
    \--local-dir data/chartqa \--include "train/\*" "val/\*"  
    
- Dev/test split: reserve \~100 QA pairs for quick prompt iteration (`dev`) and the remaining \~400 for evaluation (`test`). Long’s manifest should already specify these; copy them verbatim so `page_id` stays consistent.  
- Keep `data/processed/questions/{split}.jsonl` exactly the same as Long’s (no relabeling or shuffling).

## 3\. Prompting & Inference Workflow

1. **Images**: reuse the PNG/JPG paths from Long’s manifests; no OCR needed.  
2. **QA prompt**:  
   - System: “You answer questions about a document image. Respond with JSON `{ "answer": "...", "confidence": 0-1 }` using only visible text/numbers.”  
   - User content: include the image plus the question text; optionally add a short reminder like “If unsure, answer `unknown`.”  
3. **Model**: default to `gpt-5-vision`. If unavailable, use `gpt-4o-mini` and note it in your report.  
4. **Command**:  
     
   python scripts/05\_answer\_questions.py \\  
     
     \--mode vlm\_image \\  
     
     \--model gpt-5-vision \\  
     
     \--questions data/processed/questions/dev.jsonl \\  
     
     \--images-root data/raw \\  
     
     \--output outputs/answers/vlm\_dev.jsonl  
     
   (repeat for `test`).  
     
5. **Retries/logging**: if the API returns invalid JSON, send one corrective prompt and log the retry count. Record tokens/latency per question (the script should append them to each JSON line).

## 4\. Evaluation Metrics

- **QA accuracy**: EM \+ token-level F1 per dataset, with numeric tolerance ±0.5% for currency/percent answers. Mark questions answered as `unknown` as incorrect.  
- **Latency & cost**: average latency/question, average tokens (input+output), and estimated USD per 100 questions (based on current OpenAI pricing).  
- **Stability**: retries per 100 questions and % of responses that violated the JSON schema on the first attempt.

## 5\. Analysis & Deliverables

1. `reports/huy_report.md` (2 pages max) with EM/F1, latency/cost, retry stats, and at least five qualitative examples (wins and losses vs Textract+LLM).  
2. Prediction files: `outputs/answers/vlm_{dev,test}.jsonl` including `{question_id, answer, confidence, latency_ms, tokens}`.  
3. Note any prompt variants you tried (bullet list in the report) and which one you kept for the final run.

## 6\. Timeline & Checkpoints

- **Day 1**: Mirror Long’s dataset splits locally and confirm you can load a handful of questions/images.  
- **Day 2**: Try two prompt templates on the dev subset (\~100 Qs) and pick the more stable one.  
- **Day 3**: Run the chosen prompt on the full dev split, inspect metrics/logs.  
- **Day 4**: Run the same setup on the test split, capture latency/cost stats.  
- **Day 5**: Finish the short report and push the prediction files.

