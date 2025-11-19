# Direct Vision-Language Models for DocQA: Evaluation Report

**Author:** Huy  
**Date:** [Date]  
**Model:** GPT-5 Vision  
**Dataset:** InfographicVQA

---

## Executive Summary

This report evaluates the GPT-5 multimodal model on document question answering tasks using a direct vision-language approach. The pipeline eliminates traditional OCR stages by sending image and question directly to the VLM.

**Key Findings:**
- Exact Match Accuracy: [X.X]%
- Median Latency: [X,XXX]ms
- Cost per 100 questions: $[X.XX]
- [Other key metric]

---

## 1. Experimental Setup

### Dataset
- **Name:** InfographicVQA
- **Dev set:** 120 QA pairs
- **Test set:** 380 QA pairs
- **Image characteristics:** [Average size, complexity]

### Model Configuration
- **Model:** gpt-5
- **Temperature:** 0.0 (deterministic)
- **Max tokens:** 150
- **Image mode:** High detail
- **Prompt template:** See `prompts/qa_vlm.txt`

### Evaluation Metrics
- Exact Match (EM)
- Token-level F1
- Numeric Tolerance (±0.5%)
- Latency (median, p95)
- Cost analysis
- Retry statistics

---

## 2. Results

### 2.1 Accuracy Metrics

#### Overall Performance (Test Set, n=380)

| Metric | Dev Set | Test Set |
|--------|---------|----------|
| Exact Match (EM) | [XX.X]% | [XX.X]% |
| Token F1 | [XX.X]% | [XX.X]% |
| Numeric Tolerance | [XX.X]% | [XX.X]% |

#### Performance by Question Type

| Question Type | Count | EM | F1 |
|---------------|-------|-----|-----|
| Factual lookup | [XX] | [XX]% | [XX]% |
| Numeric extraction | [XX] | [XX]% | [XX]% |
| Spatial reasoning | [XX] | [XX]% | [XX]% |
| Multi-hop logic | [XX] | [XX]% | [XX]% |

**Key observations:**
- [Observation 1]
- [Observation 2]
- [Observation 3]

### 2.2 Operational Performance

#### Latency

| Metric | Dev Set | Test Set |
|--------|---------|----------|
| Median | [X,XXX]ms | [X,XXX]ms |
| Mean | [X,XXX]ms | [X,XXX]ms |
| P95 | [X,XXX]ms | [X,XXX]ms |

#### Cost Breakdown

| Item | Value |
|------|-------|
| Avg input tokens | [X,XXX] |
| Avg output tokens | [XX] |
| Cost per question | $[X.XXXX] |
| **Cost per 100 questions** | **$[X.XX]** |

#### Stability Metrics

| Metric | Dev Set | Test Set |
|--------|---------|----------|
| Total questions | [XXX] | [XXX] |
| Retries needed | [XX] ([X.X]%) | [XX] ([X.X]%) |
| Validation failures | [XX] ([X.X]%) | [XX] ([X.X]%) |
| API errors | [XX] | [XX] |

---

## 3. Qualitative Analysis

### 3.1 Success Cases

**Example 1: Spatial Reasoning**
```
Question: "Which category has the tallest bar?"
Ground Truth: "Healthcare"
Prediction: "Healthcare"
Confidence: 0.92
Analysis: Model correctly interpreted bar chart layout
```

**Example 2: Numeric Extraction**
```
Question: "What is the total revenue in 2023?"
Ground Truth: "$2.4 million"
Prediction: "$2.4 million"
Confidence: 0.95
Analysis: Accurate extraction with proper formatting
```

### 3.2 Failure Cases

**Example 1: Hallucination**
```
Question: "What was the growth rate in 2022?"
Ground Truth: "12.4%"
Prediction: "11.8%"
Confidence: 0.87
Analysis: Model calculated from visible 2021/2023 data instead of extracting
```

**Example 2: Small Text**
```
Question: "What is stated in the footer?"
Ground Truth: "[Specific footer text]"
Prediction: "unknown"
Confidence: 0.3
Analysis: 8pt text too small for reliable extraction
```

### 3.3 Error Taxonomy

| Error Type | Count | % of Errors | Description |
|------------|-------|-------------|-------------|
| Hallucination | [XX] | [XX]% | Inferred answers not in image |
| Small text | [XX] | [XX]% | Failed on fine-print/footnotes |
| Confidence miscalibration | [XX] | [XX]% | High confidence on wrong answers |
| Table structure | [XX] | [XX]% | Misaligned multi-column tables |
| Other | [XX] | [XX]% | Misc errors |

---

## 4. Prompt Engineering Insights

### Tested Variants

1. **Variant A** (baseline): [Description]
   - Dev EM: [XX.X]%
   - Retry rate: [X.X]%

2. **Variant B** (with format instructions): [Description]
   - Dev EM: [XX.X]%
   - Retry rate: [X.X]%

3. **Variant C** (final - negative instructions): [Description]
   - Dev EM: [XX.X]%
   - Retry rate: [X.X]%

### Final Prompt Selection

Selected **Variant C** based on:
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

---

## 5. Comparison with Baseline

### vs. Textract + GPT-5 (if available)

| Metric | Textract+GPT-5 | GPT-5 Vision | Δ |
|--------|----------------|--------------|---|
| Exact Match | [XX.X]% | [XX.X]% | [±X.X]% |
| Token F1 | [XX.X]% | [XX.X]% | [±X.X]% |
| Median Latency | [X,XXX]ms | [X,XXX]ms | [±XX]% |
| Cost/100Q | $[X.XX] | $[X.XX] | [±XX]% |

**Key advantages of VLM approach:**
- [Advantage 1]
- [Advantage 2]

**Key limitations:**
- [Limitation 1]
- [Limitation 2]

---

## 6. Recommendations

### Production Deployment

**Use VLM approach when:**
- ✅ [Use case 1]
- ✅ [Use case 2]
- ✅ [Use case 3]

**Use OCR+LLM when:**
- ✅ [Use case 1]
- ✅ [Use case 2]

### Optimization Opportunities

1. **Prompt caching:** [Description and expected impact]
2. **Batch processing:** [Description and expected impact]
3. **Image preprocessing:** [Description and expected impact]

---

## 7. Reproducibility

### Commands Run

```bash
# Dataset preparation
python scripts/01_prepare_dataset.py --dev-size 120 --test-size 380

# Dev set inference
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-5 \
    --questions data/processed/questions/dev.jsonl \
    --output outputs/answers/vlm_dev.jsonl

# Test set inference
python scripts/05_answer_questions.py \
    --mode vlm_image \
    --model gpt-5 \
    --questions data/processed/questions/test.jsonl \
    --output outputs/answers/vlm_test.jsonl

# Evaluation
python scripts/06_evaluate_qa.py \
    --pred vlm=outputs/answers/vlm_test.jsonl \
    --questions data/processed/questions/test.jsonl \
    --output results/vlm_metrics.json
```

### Artifacts

- Predictions: `outputs/answers/vlm_{dev,test}.jsonl`
- Metrics: `results/vlm_metrics.json`
- Prompts: `prompts/qa_vlm.txt`

---

## 8. Conclusion

[Summary of findings, key takeaways, and next steps]

---

## Appendix

### A. Full Metrics Output

```json
[Paste results/vlm_metrics.json content]
```

### B. Sample Predictions

[Include 5-10 representative examples with images if possible]

### C. Environment Details

- Python version: [X.X.X]
- OpenAI library version: [X.X.X]
- Run date: [Date]
- Total runtime: [X hours]
