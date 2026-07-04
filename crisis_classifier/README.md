# Crisis-Risk Classifier — Safety Layer for Mental-Health-Assistant

## What this is
A lightweight PyTorch classifier that scores incoming chat messages for
crisis/suicidal-intent risk *before* they reach the RAG/LLM pipeline. If a
message is flagged, the app can short-circuit to a safe, resource-pointing
response instead of relying purely on the LLM's judgement.

## Why it matters (Responsible AI angle)
LLMs can miss or mishandle high-risk messages, especially under prompt
variation. A dedicated, auditable classifier gives:
- a deterministic, testable safety gate independent of the LLM
- a component you can evaluate with real metrics (precision/recall) instead
  of "trusting the prompt"
- an explicit tradeoff you can reason about: false negatives (missed crisis
  message) are far more costly than false positives (extra caution shown to
  a non-crisis message), so recall on the crisis class is the metric to
  optimize.

## Pipeline
```
user message
    -> sentence-transformers embedding (all-MiniLM-L6-v2, 384-dim)
    -> small feedforward NN (384 -> 128 -> 64 -> 1)
    -> sigmoid -> risk score [0, 1]
```

## Dataset
"Suicide and Depression Detection" (Kaggle, Nikhileswar Komati) — 232K
Reddit posts labeled suicide / non-suicide. A balanced subsample (~5,000-6,000
rows) is used for fast training on CPU.

## Files
- `train_crisis_classifier.py` — loads data, embeds text, trains the
  classifier, evaluates on a held-out test set, saves weights + metrics.
- `crisis_check.py` — drop-in inference wrapper for the FastAPI backend.
- `metrics.txt` — results from the most recent training run.

## How to run
```bash
pip install torch sentence-transformers scikit-learn pandas
python train_crisis_classifier.py --csv Suicide_Detection.csv --sample 5000 --embed sentence-transformers
```

This produces:
- `crisis_classifier.pt` — trained weights
- `metrics.txt` — accuracy / precision / recall / F1 on the test split

## Integrating into the existing backend
```python
from crisis_check import CrisisChecker

checker = CrisisChecker("crisis_classifier.pt")

@app.post("/chat")
def chat(message: str):
    if checker.is_crisis(message):
        return {"crisis_flag": True, "response": SAFE_RESOURCE_MESSAGE}
    # else continue to normal RAG pipeline
    ...
```

## Talking points for interviews
- **Architecture**: embedding + small feedforward head, not a huge model —
  deliberate choice for low latency in a safety-critical, synchronous check.
- **Evaluation**: report precision/recall/F1, and specifically call out
  recall on the crisis class and the false-negative rate — that's the number
  that matters in production.
- **What I'd improve**: fine-tune the embedding model itself (not just a
  frozen encoder + head), try a transformer-based classifier (e.g. DistilBERT)
  for comparison, add calibration (Platt scaling) so risk scores are more
  interpretable as probabilities, and validate against a clinician-reviewed
  or C-SSRS-labeled dataset instead of a purely Reddit-scraped one for better
  real-world reliability.
