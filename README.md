# VLM-based DocQA Pipeline

Implementation of a multimodal DocQA pipeline for evaluating direct vision-language models on document question answering tasks.

## Overview

This project implements a **single-step vision-language pipeline**:

```
Image + Question → VLM (GPT-4o/GPT-5) → Answer
```

The pipeline eliminates traditional OCR stages by sending images and questions directly to multimodal language models, simplifying architecture while maintaining high accuracy on document understanding tasks.

## Project Structure

```
ocr-vlm-exp/
├── data/
│   ├── raw/                          # Dataset images
│   └── processed/
│       ├── manifests/                # Page manifests (dev/test)
│       └── questions/                # Question files (dev/test)
├── outputs/
│   └── answers/                      # VLM predictions
├── results/                          # Evaluation metrics
├── reports/                          # Analysis reports
├── scripts/
│   ├── 01_prepare_dataset.py         # Dataset download & splitting
│   ├── 05_answer_questions.py        # VLM inference engine
│   ├── 06_evaluate_qa.py            # Metrics computation
│   └── utils/                        # Core utilities
│       ├── vlm_client.py            # GPT-5 Vision API wrapper
│       ├── prompt_templates.py       # Prompt engineering
│       ├── metrics.py               # EM, F1, numeric tolerance
│       └── retry_handler.py         # JSON validation & retry
├── prompts/
│   └── qa_vlm.txt                   # Prompt template from article
└── requirements.txt
```

## Complete Step-by-Step Guide

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd ocr-vlm-exp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor (vim, code, etc.)
```

Add this line to `.env`:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

Load the environment variable:
```bash
# Load API key into current session
export $(grep -v '^#' .env | xargs)

# Or manually export it
export OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Download and Prepare Dataset

Download InfographicVQA dataset and create dev/test splits:

```bash
# Option A: Automatic download (requires Hugging Face login)
huggingface-cli login  # One-time setup
python scripts/01_prepare_dataset.py --dev-size 120 --test-size 380

# Option B: If automatic download fails, script creates placeholder structure
# Then manually add images to data/raw/infographicvqa/
python scripts/01_prepare_dataset.py --dev-size 120 --test-size 380
```

**This creates:**
- `data/processed/manifests/{dev,test}.jsonl` - Page metadata (120 + 380 entries)
- `data/processed/questions/{dev,test}.jsonl` - Questions with ground truth
- `data/raw/infographicvqa/*.png` - Dataset images (500 images)

**Verify dataset:**
```bash
# Check question files
wc -l data/processed/questions/dev.jsonl   # Should show 120
wc -l data/processed/questions/test.jsonl  # Should show 380

# Check images
ls data/raw/infographicvqa/*.png | wc -l   # Should show 500
```

### Step 4: Run VLM Inference

**Quick test (2 questions, ~10 seconds):**
```bash
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-4o \
    --questions data/processed/questions/dev.jsonl \
    --images-root data/raw \
    --output outputs/answers/vlm_quick_test.jsonl \
    --max-questions 2
```

**Dev set (120 questions, ~6 minutes, ~$1.07):**
```bash
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-4o \
    --questions data/processed/questions/dev.jsonl \
    --images-root data/raw \
    --output outputs/answers/vlm_gpt4o_dev.jsonl
```

**Test set (380 questions, ~20 minutes, ~$3.38):**
```bash
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-4o \
    --questions data/processed/questions/test.jsonl \
    --images-root data/raw \
    --output outputs/answers/vlm_gpt4o_test.jsonl
```

**Model options:**
- `gpt-4o` - Recommended, works perfectly with vision
- `gpt-5` - May have vision support issues (use gpt-4o instead)
- `gpt-4-turbo` - Alternative option

**Optional flags:**
- `--prompt-template prompts/qa_vlm.txt` - Use custom prompt template
- `--max-questions 10` - Limit number of questions for testing

### Step 5: Evaluate Results

Compute accuracy and operational metrics:

```bash
# Evaluate dev set
python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_gpt4o_dev.jsonl \
    --questions data/processed/questions/dev.jsonl \
    --output results/vlm_gpt4o_dev_metrics.json

# Evaluate test set
python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_gpt4o_test.jsonl \
    --questions data/processed/questions/test.jsonl \
    --output results/vlm_gpt4o_test_metrics.json
```

**Add `--verbose` flag to see individual errors:**
```bash
python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_gpt4o_test.jsonl \
    --questions data/processed/questions/test.jsonl \
    --output results/vlm_gpt4o_test_metrics.json \
    --verbose
```

### Step 6: Review Results

**View metrics in terminal:**
```bash
# The evaluation script prints summary to terminal
# Or view the JSON file:
cat results/vlm_gpt4o_test_metrics.json | jq '.'

# View specific metrics:
cat results/vlm_gpt4o_test_metrics.json | jq '.vlm.accuracy'
cat results/vlm_gpt4o_test_metrics.json | jq '.vlm.operational'
cat results/vlm_gpt4o_test_metrics.json | jq '.vlm.by_question_type'
```

**Output includes:**
- **Accuracy Metrics:** Exact Match (EM), Token F1, Numeric Tolerance
- **Operational Metrics:** Latency (median, p95), token counts, cost per 100Q
- **Per-question-type breakdown:** Factual lookup, numeric extraction, spatial reasoning, multi-hop logic
- **Error analysis:** Retry statistics, sample failures

### Step 7: Generate Report (Optional)

Use the template to document your findings:

```bash
# Copy template
cp reports/huy_report_template.md reports/huy_report.md

# Edit with your results
nano reports/huy_report.md
```

Fill in the metrics from `results/vlm_gpt4o_test_metrics.json`.

## Implementation Details

### Model Configuration

- **Models**: GPT-4o (recommended), GPT-5 (experimental)
- **Temperature**: 0.0 (deterministic) for GPT-4o, default (1.0) for GPT-5
- **Max tokens**: 150
- **Image mode**: High detail
- **Retry policy**: Single retry with corrective prompt on JSON validation failure

### Prompt Engineering

System message enforces:
- JSON schema: `{"answer": "text", "confidence": 0.0-1.0}`
- Strict extraction (no inference beyond visible content)
- Few-shot examples (2 minimal examples)

User message includes:
- Question text
- Explicit guidelines (percentage symbols, date formats)
- Negative instructions ("Do not infer or calculate")

See `prompts/qa_vlm.txt` for the exact template.

### Evaluation Metrics

The evaluation script computes:

**Accuracy:**
- **Exact Match (EM)**: Case-insensitive exact string match
- **Token F1**: Word-level F1 score
- **Numeric Tolerance**: ±0.5% for percentages/numbers

**Operational:**
- **Latency**: Median, mean, p95 (ms)
- **Cost**: Per question and per 100 questions (USD)
- **Tokens**: Input/output token counts
- **Retry rate**: Percentage requiring JSON correction

**Question Types:**
- Factual lookup
- Numeric extraction
- Spatial reasoning
- Multi-hop logic

## Architecture Design

### Single-Step Pipeline

```python
# Simplified workflow
image = load_image(image_path)
response = vlm_client.answer_question(
    image_path=image_path,
    question=question_text,
    system_message=SYSTEM_PROMPT,
    user_template=USER_TEMPLATE
)
answer = parse_and_validate_json(response)
```

### Error Handling

1. **JSON Validation**: Regex extraction + schema validation
2. **Retry Logic**: Single retry with corrective prompt (article: 5.2% → 1.3% failure rate)
3. **Image Handling**: Automatic base64 encoding with high-detail mode

### Cost Optimization

- No image resizing (article: 6% accuracy drop when resized to 768px)
- Native resolution processing for best accuracy
- Pricing: $0.0075/1K vision tokens, $0.02/1K output tokens

## Full Workflow Example

Here's the complete end-to-end workflow:

```bash
# 1. Setup (one-time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)

# 2. Prepare dataset (one-time)
python scripts/01_prepare_dataset.py --dev-size 120 --test-size 380

# 3. Run inference
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-4o \
    --questions data/processed/questions/dev.jsonl \
    --images-root data/raw \
    --output outputs/answers/vlm_gpt4o_dev.jsonl

python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-4o \
    --questions data/processed/questions/test.jsonl \
    --images-root data/raw \
    --output outputs/answers/vlm_gpt4o_test.jsonl

# 4. Evaluate
python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_gpt4o_dev.jsonl \
    --questions data/processed/questions/dev.jsonl \
    --output results/vlm_gpt4o_dev_metrics.json

python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_gpt4o_test.jsonl \
    --questions data/processed/questions/test.jsonl \
    --output results/vlm_gpt4o_test_metrics.json

# 5. View results
cat results/vlm_gpt4o_test_metrics.json | jq '.vlm'

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI for GPT-4o and GPT-5 Vision APIs
- Hugging Face for dataset hosting
- InfographicVQA dataset creators
