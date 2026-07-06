# Hiver AI Suggested-Response System

An end-to-end system that takes an incoming customer support email, retrieves relevant
past examples, generates a policy-grounded suggested reply using an LLM, and then scores
that reply for quality using a validated LLM-judge evaluator.

## 1. Overview / Architecture

```
data/dataset.json          -> 15 synthetic (email, policy, golden_reply) examples
scripts/generate_dataset.py -> builds dataset.json via LLM
src/retriever.py           -> embeds dataset locally, does semantic top-k retrieval
src/generator.py           -> RAG: retrieve -> prompt -> generate suggested reply
src/evaluator.py           -> LLM-as-judge: scores a generated reply on 3 axes
scripts/validate_metric.py -> sanity check that the judge actually penalizes bad replies
data/generated_replies.json   -> output of generator.py on test emails
data/evaluation_results.json  -> output of evaluator.py on those replies
```

Pipeline: `generate_dataset.py` → `generator.py` (uses `retriever.py`) → `evaluator.py`,
validated independently by `validate_metric.py`.

## 2. The Dataset

**Source: synthetic, LLM-generated** (`scripts/generate_dataset.py`, Llama 3.3 70B via Groq).
I did not use a real Hiver dataset — I don't have access to one — so I generated 15
customer support emails for a fictional shared-inbox product, each paired with a strict
internal policy and a golden reply that follows it.

**Why this is representative (and where it's weak):**
- Covers 4 realistic support categories (billing, login, feature requests, bugs) and varies
  tone (angry, confused, polite), which mirrors the spread of tone/intent a real support
  queue sees.
- Each example ties a policy to a reply, which is the actual thing I need to evaluate
  (policy adherence), not just "is this a nice email."
- **Honest limitation**: synthetic data is cleaner and more internally consistent than real
  tickets. Real emails are messier (typos, multiple issues in one email, ambiguous intent),
  and real internal policies are longer and sometimes contradictory. This dataset is good
  enough to demonstrate the pipeline and the evaluation methodology, but I would not trust
  it as a benchmark for a production system without swapping in real, anonymized tickets.

## 3. The Generator (Gen AI, not classical ML)

**Approach: RAG (retrieval-augmented few-shot prompting)**, not fine-tuning, not a
classifier.

- `retriever.py` embeds every dataset item locally with `all-MiniLM-L6-v2`
  (sentence-transformers) and does cosine-similarity search to pull the top-2 most similar
  past emails for a new incoming email.
- `generator.py` injects those 2 retrieved (email, policy, golden_reply) triples into the
  prompt as few-shot examples, along with a system prompt instructing the model to follow
  the demonstrated policies strictly and not hallucinate features. It then calls Llama 3.3
  70B (Groq) at low temperature (0.3) to draft the reply.

**Why RAG over the alternatives:**
- **vs. fine-tuning**: fine-tuning needs far more than 15 examples to be reliable, costs
  more, and is opaque — if the model gets a policy wrong, you can't see why. RAG keeps the
  exact policy text visible in the prompt every time, which matters a lot when
  "policy adherence" is the thing being graded.
- **vs. plain few-shot with a fixed example set**: retrieval means the examples shown to
  the model are actually relevant to the new email (e.g., a billing complaint is grounded
  in the billing policy example, not the GDPR one). This directly reduces the chance of the
  model applying the wrong policy.
- **Trade-off I accepted**: retrieval quality depends entirely on embedding similarity
  between the new email and the dataset. With only 15 examples, if a genuinely novel
  category of email comes in, there's no good match to retrieve, and the model has to
  fall back on general instruction-following rather than a demonstrated policy. This is
  the main scaling risk — it would need a larger, denser dataset to stay reliable.

## 4. The Evaluation System (the core of this challenge)

### What "accurate" means here

Exact-match against a golden reply is the wrong metric: a good reply doesn't need to match
the golden reply's wording, it needs to (a) follow the same internal policy, (b) be
empathetic and professional, and (c) actually resolve the customer's question concisely.
Those three things can vary independently — a reply can be polite but policy-violating, or
policy-correct but robotic — so I evaluate them as **three separate axes**, not one blended
score, and then combine them with explicit weights.

### The metric: LLM-as-judge, 3 axes, weighted score

`evaluator.py` prompts Llama 3.3 70B (temperature 0.1, for consistency) to score each
generated reply against the *retrieved* policy context (not the whole dataset — only what
the generator actually had access to, so the judge is grading the same information the
generator saw):

| Axis | Weight | What it checks |
|---|---|---|
| Policy adherence | 0.4 | Did the reply follow the specific internal policy, without hallucinating features or rules that weren't stated? |
| Tone & empathy | 0.3 | Is it polite, human, and appropriately empathetic to the customer's stated emotion? |
| Completeness & clarity | 0.3 | Does it directly and concisely answer what was asked? |

**Why this weighting**: policy adherence is weighted highest because in a real support
tool, a warm but policy-violating reply (e.g. promising a refund the company doesn't
actually offer) is a compliance and trust problem — a much worse failure mode than a
correct-but-slightly-terse reply. Tone and completeness matter but are more forgiving.

I compute the weighted score myself (`calculated_weighted_score`) from the judge's raw
per-axis integers, rather than trusting the model's own self-reported `overall_score`.
**Known inconsistency I'm flagging honestly**: the raw JSON still contains both values, and
they don't always agree (e.g. one response has `overall_score: 4.67` from the model but
`calculated_weighted_score: 4.7` from my own math). I trust my own calculated version
because it's deterministic and auditable — the LLM's self-reported overall score is not.
In a next iteration I'd drop the LLM's self-reported field entirely to avoid the confusion.

### How I validated the metric reflects real quality, not just a number

This is what `scripts/validate_metric.py` does: I fed the judge a **deliberately terrible
reply** (rude, ignores the customer's actual question, hallucinates an unrequested $1000
credit) against a simple, unambiguous policy. The test asserts the weighted score comes
back below 2.0/5.

This matters because an LLM judge that always scores in the 4-5 range regardless of input
quality is useless — it's not measuring anything. Running a known-bad input through the
judge and checking it gets penalized is the minimum bar for trusting the metric at all. It
passed: the judge correctly scored `policy_adherence` low for ignoring the reset-link
policy and inventing a refund, and the weighted score landed under the threshold.

**What I'd add with more time**: a second known-good "perfect reply" case asserting a score
above 4.5 (to catch a judge that's just uniformly harsh), and a small human-labeled holdout
set to check the judge's scores correlate with human judgment, not just with my own
hand-picked adversarial example.

### Reporting

- **Per-response**: `data/evaluation_results.json` → `per_response_evaluations`, each with
  the 3 axis scores + reasons, the LLM's self-reported overall score, and my calculated
  weighted score.
- **System-level**: same file → `overall_system_scores`, averaging each axis and the
  weighted score across all evaluated replies.

## 5. How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 3. Generate the dataset (optional — dataset.json is already committed)
python scripts/generate_dataset.py

# 4. Generate suggested replies for test emails (RAG)
python scripts/generator.py

# 5. Evaluate the generated replies
python scripts/evaluator.py

# 6. Sanity-check that the judge actually penalizes bad replies
python scripts/validate_metric.py
```

Outputs land in `data/generated_replies.json` and `data/evaluation_results.json`.

## 6. Known Limitations (honest list)

- Dataset is synthetic and small (15 examples) — good for demonstrating the pipeline, not
  production-representative.
- Single LLM judge, no second model or human cross-check for inter-rater reliability.
- Retrieval quality degrades for novel email categories not represented in the 15-item
  dataset.
- `overall_score` (model self-reported) and `calculated_weighted_score` (mine) coexist in
  the output; I trust the latter, but this should be cleaned up so only one is reported.

## 7. How I Used AI Tools

I used an LLM (Groq/Llama 3.3 70B) as a first-class part of the system itself — both to
generate the synthetic dataset and as the reply generator and the judge. I also used an AI
assistant (Claude) to help debug and review structure while building this, and to help
draft this README from the actual code and output files in this repo.