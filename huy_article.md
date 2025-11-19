# Direct Vision-Language Models for DocQA: GPT-5 Multimodal in Action

**Evaluating single-step image-to-answer pipelines with GPT-5's native multimodal capabilities**

---

## The Vision Promise

Traditional document understanding pipelines follow a rigid two-stage pattern: OCR extraction → text processing. But modern multimodal models like GPT-5 promise to collapse this workflow into a single API call—feed the image and question directly, get the answer back.

This article evaluates whether **direct multimodal inference** can match or exceed the accuracy of OCR-based pipelines while simplifying architecture and potentially improving spatial reasoning. We test GPT-5 (released August 2025 with native vision capabilities) on the same InfographicVQA benchmark used for Textract evaluation, measuring accuracy, latency, and cost in head-to-head comparison.

---

## System Architecture

### Single-Step Pipeline

The multimodal approach eliminates the OCR stage entirely:

```
Image + Question → GPT-5 → Answer
```

**Key advantages:**
- **Spatial awareness**: Model processes visual layout directly
- **Simplified architecture**: One API call vs. orchestrating OCR + LLM
- **Handling edge cases**: Rotated text, overlapping elements, stylized fonts

**Potential challenges:**
- **Cost**: Vision tokens typically 3–5× more expensive than text
- **Hallucination risk**: Model may infer answers not present in image
- **Prompt engineering**: Limited few-shot examples due to image size constraints

---

## Experimental Setup

### Dataset: InfographicVQA

We used identical data splits as the Textract evaluation for direct comparison:

- **Dev set**: 120 QA pairs (prompt tuning and validation)
- **Test set**: 380 QA pairs (final evaluation)

Infographics averaged 1200×900 pixels, containing dense text, charts, and visual elements—an ideal stress test for VLM capabilities.

### Model Configuration

- **Primary model**: GPT-5 (gpt-5, released August 2025 with native multimodal capabilities)
- **Image resolution**: High detail mode (vision tokens based on image dimensions)
- **Temperature**: 0.0 (deterministic responses)
- **Max tokens**: 150 (most answers <20 tokens)

### Prompt Design

After testing three prompt variants on the dev set, we settled on this structure:

**System message:**
```
You are a document analysis assistant. Answer questions about infographics using ONLY information visible in the image. Return JSON format: {"answer": "exact text or number", "confidence": 0.0-1.0}. If unsure, respond with {"answer": "unknown", "confidence": 0.0}.
```

**User message:**
```
[Image attachment]

Question: {question_text}

Guidelines:
- Extract exact text/numbers from the image
- Do not infer or calculate beyond what's shown
- For percentages, include the % symbol
- For dates, match the format shown

Respond with valid JSON only.
```

**Few-shot examples**: Due to token cost, we embedded two minimal examples in the system message rather than full image demonstrations.

---

## Implementation & Challenges

### Retry Mechanism

Initial testing revealed 5.2% of responses failed JSON schema validation—GPT-5 sometimes added explanatory text or returned malformed quotes.

**Solution**: Implement single-retry logic with corrective prompt:
```
"Your previous response was not valid JSON. Respond with exactly this format: {\"answer\": \"text\", \"confidence\": 0.8} and nothing else."
```

This reduced validation failures to **1.3%** on the test set.

### Token Optimization

Vision tokens for InfographicVQA images:
- Average image: 1,420 vision tokens (high detail mode)
- Prompt text: ~180 tokens
- Response: ~25 tokens

**Cost management**: We experimented with auto-resizing images to 768px width (from 1200px) but saw 6% accuracy drop on small-text questions. Final implementation used native resolution with high detail mode.

---

## Results & Analysis

### Accuracy Metrics

| Metric | Dev Set (n=120) | Test Set (n=380) |
|--------|-----------------|------------------|
| **Exact Match (EM)** | 74.2% | 72.1% |
| **Token F1** | 84.5% | 82.8% |
| **Numeric Tolerance (±0.5%)** | 78.3% | 76.2% |

**Performance by question type:**

| Question Type | GPT-5 EM | Textract+GPT-5 EM | Delta |
|---------------|----------|-------------------|---------|
| Factual lookup | 84% | 78% | +6% |
| Numeric extraction | 77% | 69% | +8% |
| Spatial reasoning | 71% | 52% | +19% ⭐ |
| Multi-hop logic | 62% | 48% | +14% |

**Key finding**: GPT-5's multimodal capabilities significantly outperform the OCR-based pipeline on spatial reasoning tasks (chart interpretation, visual comparisons), achieving **+19% EM** advantage with overall improvements across all question types.

### Operational Performance

**Latency (per question, test set):**
- Median: 2,450ms
- 95th percentile: 3,820ms
- **31% faster** than Textract+LLM (3,530ms median)

Single API call eliminates orchestration overhead and network round-trips between OCR and LLM stages.

**Cost breakdown (OpenAI pricing, August 2025):**
- Vision input: 1,420 tokens × $0.0075/1K = $0.0107
- Text input: 180 tokens × $0.0075/1K = $0.0014
- Output: 25 tokens × $0.02/1K = $0.0005
- **Total per question**: ~$0.0126
- **Total per 100 questions**: ~$1.26

**14% cheaper** than Textract+GPT-5 ($1.47/100 questions), making the multimodal approach both faster and more cost-effective while delivering superior accuracy.

### Error Analysis

**Where GPT-5 excels (38% error reduction vs. Textract):**

1. **Rotated text**: Model correctly read axis labels at 90° angles that Textract missed
   
   *Example*: Question "What is the Y-axis unit?" → GPT-5: "Million USD" ✓, Textract: [empty extraction]

2. **Visual context**: Questions requiring spatial relationships
   
   *Example*: Question "Which category has the tallest bar?" → GPT-5: "Healthcare" ✓, Textract+GPT-5: "Education" ✗ (OCR lost layout)

3. **Stylized fonts**: Decorative headers and callout boxes
   
   *Example*: Textract confidence <0.6 on stylized title text; GPT-5 extracted correctly

**Where GPT-5 struggles (28% of total errors):**

1. **Hallucination (35% of errors)**: Model inferred answers not present in image
   
   *Example*: Question "What was the growth rate in 2022?" → Ground truth: "12.4%", GPT-5: "11.8%" [calculated from visible 2021/2023 data—not shown in image]

2. **Small text (26% of errors)**: Dense footnotes or fine-print annotations
   
   *Example*: 8pt footer text often returned "unknown" or incorrect extraction

3. **Confidence miscalibration (22% of errors)**: Model returned high confidence (>0.8) on incorrect answers
   
   *Example*: Wrong numeric answer with confidence 0.89—creates false certainty for downstream systems

4. **Table structure (17% of errors)**: Complex multi-column tables occasionally misaligned
   
   *Example*: Extracted row data from wrong column due to visual ambiguity

---

## Head-to-Head Comparison

### Accuracy vs. Cost Trade-off

```
              Cost/100Q    Test EM    F1 Score
Textract+GPT-5   $1.47     64.2%      76.4%
GPT-5            $1.26     72.1%      82.8%
----------------------------------------
Difference       -14%      +7.9%      +6.4%
```

**Interpretation**: GPT-5 Vision achieves substantial accuracy gains while being **cheaper** than the OCR-based pipeline—a game-changer. For production systems processing 100K+ questions monthly, this translates to:
- $210 cost savings
- ~7,900 more correct answers

### Latency Comparison

| Pipeline | Median (ms) | p95 (ms) |
|----------|-------------|----------|
| Textract+GPT-5 | 3,530 | 5,680 |
| GPT-5 | 2,450 | 3,820 |
| **Improvement** | **-31%** | **-33%** |

Single-call architecture reduces tail latency variability—critical for user-facing applications.

### Engineering Complexity

**Textract+GPT-5:**
- ❌ Two API integrations (AWS + OpenAI)
- ❌ State management (cache OCR outputs)
- ❌ Error handling for two failure modes
- ✅ Debuggable OCR intermediate outputs

**GPT-5:**
- ✅ Single API integration
- ✅ Stateless architecture
- ✅ Simpler error handling
- ✅ Now cheaper and faster than two-stage pipeline
- ❌ Black-box inference (harder to debug)

---

## Production Recommendations

### When to Use Direct VLM

✅ **Spatial reasoning tasks**: Charts, diagrams, visual comparisons

✅ **Low-to-medium volume**: <50K questions/month where cost difference is negligible

✅ **Rapid prototyping**: Simplest path from idea to production

✅ **Handling diverse layouts**: No assumptions about text orientation or structure

### When to Prefer OCR+LLM

✅ **High-volume batch processing**: Cost savings compound at scale

✅ **Audit requirements**: Need to inspect and verify extracted text

✅ **Multi-language support**: Textract handles 55+ languages with specialized models

✅ **Strict hallucination constraints**: Two-stage pipeline limits model's ability to fabricate answers

### Hybrid Strategy

Our analysis suggests an **intelligent routing approach**:

```python
def route_question(question, image):
    if requires_spatial_reasoning(question):
        return call_vlm(image, question)
    elif is_simple_lookup(question):
        return call_textract_llm(image, question)
    else:
        # For ambiguous cases, parallel call with voting
        result_vlm = call_vlm(image, question)
        result_ocr = call_textract_llm(image, question)
        return ensemble(result_vlm, result_ocr)
```

**Heuristics for routing:**
- Keywords: "which is larger", "compare", "trend" → VLM
- Keywords: "what is the title", "list all", "total" → OCR+LLM
- Confidence voting for uncertain classifications

---

## Lessons Learned

### Prompt Engineering Insights

1. **Explicit JSON formatting**: Including example structure in every prompt reduced retry rate significantly
2. **Negative instructions**: "Do not infer or calculate" helped reduce hallucination rate
3. **Format matching**: "Include % symbol", "match date format shown" improved EM on numeric questions by 4%

### Operational Considerations

1. **Confidence scores are unreliable**: GPT-5 often returned 0.85–0.95 confidence on incorrect answers. Don't use raw confidence for filtering.
2. **Image quality matters**: Compressing images below 768px width significantly degraded small-text extraction
3. **Rate limiting**: OpenAI's multimodal endpoint has stricter rate limits than text-only API—implement exponential backoff

### Future Optimizations

1. **Prompt caching**: For repeated questions on same image, cache vision tokens
2. **Batch processing**: Current implementation is sequential; parallel processing could improve throughput
3. **Model comparison**: Benchmark against Claude 4 and Gemini 2.0 on same datasets

---

## Reproducibility

Complete pipeline implementation:

```bash
# Setup
pip install openai pillow datasets pandas numpy

export OPENAI_API_KEY=sk-...

# Run multimodal pipeline
python scripts/05_answer_questions.py \
  --mode vlm_image \
  --images-root data/raw \
  --questions data/processed/questions/test.jsonl \
  --model gpt-5 \
  --output outputs/answers/vlm_test.jsonl \
  --prompt-template prompts/qa_vlm.txt

# Evaluate
python scripts/06_evaluate_qa.py \
  --pred vlm=outputs/answers/vlm_test.jsonl \
  --questions data/processed/questions/test.jsonl \
  --output results/vlm_metrics.json
```

Dataset and evaluation scripts available in the project repository.

---

## Conclusion

GPT-5 achieves **72.1% exact match accuracy** on InfographicVQA—a **7.9 percentage point improvement** over the Textract+GPT-5 baseline—while being **31% faster and 14% cheaper** at $1.26 per 100 questions.

This represents a paradigm shift: the multimodal approach now delivers superior accuracy, lower latency, **and** lower cost compared to traditional OCR pipelines. GPT-5 particularly excels on **spatial reasoning tasks** (+19% EM), making it ideal for chart-heavy documents and visual comparisons.

**Recommendation**: For most production systems, **GPT-5 should be the default choice**:
- Superior accuracy across all question types
- Single API call simplifies architecture
- Cost-effective at scale
- Significantly faster inference

OCR+LLM pipelines remain valuable for:
- Strict audit requirements (need to inspect extracted text)
- Specialized OCR needs (non-Latin scripts, historical documents)
- Air-gapped environments without API access

The August 2025 release of GPT-5 with native multimodal capabilities has made direct inference the clear winner for document AI applications, eliminating the traditional cost/accuracy trade-off.

**Next steps:**
- Test GPT-5 on additional DocQA datasets (ChartQA, TextVQA)
- Evaluate Claude 4 and Gemini 2.0 on same benchmark
- Optimize prompt caching to reduce costs further
- Explore fine-tuning for domain-specific documents

---

*This research was conducted as part of a comparative study on OCR vs. VLM approaches to document understanding. See the companion article on Textract+GPT-5 evaluation for the complete baseline analysis.*

