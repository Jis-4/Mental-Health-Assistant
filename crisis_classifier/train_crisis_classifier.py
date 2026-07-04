"""
Crisis-Risk Classifier for Mental-Health-Assistant
----------------------------------------------------
A lightweight safety layer: given a user message, predict whether it shows
signs of suicidal/crisis intent BEFORE the message is passed to the LLM.

Pipeline:
  raw text -> sentence-transformers embedding (same model family already
  used in Mental-Health-Assistant) -> small PyTorch feedforward classifier
  -> risk score (0-1)

Dataset: Suicide and Depression Detection (Kaggle, Nikhileswar Komati)
  https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
  Columns expected: "text", "class" (values: "suicide" / "non-suicide")

Usage:
  python train_crisis_classifier.py --csv Suicide_Detection.csv --sample 5000
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str, sample_size: int, seed: int = 42) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # The Kaggle file usually has an unnamed index column - drop it if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    assert "text" in df.columns and "class" in df.columns, (
        f"Expected 'text' and 'class' columns, got: {df.columns.tolist()}"
    )

    df["label"] = (df["class"].str.strip().str.lower() == "suicide").astype(int)

    # Balanced sample so training is fast and classes aren't skewed
    n_per_class = sample_size // 2
    df_pos = df[df["label"] == 1].sample(n=n_per_class, random_state=seed)
    df_neg = df[df["label"] == 0].sample(n=n_per_class, random_state=seed)
    df_sample = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=seed)

    print(f"Using {len(df_sample)} rows "
          f"({df_sample['label'].sum()} crisis / {len(df_sample) - df_sample['label'].sum()} non-crisis)")
    return df_sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Encoding texts (this is the slow step, be patient)...")
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def embed_texts_tfidf(texts, max_features: int = 384):
    """
    Fallback embedding method requiring no internet/model download.
    Use --embed tfidf when sentence-transformers can't reach Hugging Face
    (e.g. restricted network). On your own machine, prefer the default
    sentence-transformers path for the real repo submission.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"Using TF-IDF fallback embeddings (max_features={max_features})")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return embeddings


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
class CrisisClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Train / eval loop
# ---------------------------------------------------------------------------
def train(model, X_train, y_train, X_val, y_val, epochs=30, lr=1e-3, device="cpu"):
    model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float().cpu().numpy()
            _, _, val_f1, _ = precision_recall_fscore_support(
                y_val, val_preds, average="binary", zero_division=0
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={loss.item():.4f} | val_f1={val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"\nBest validation F1: {best_val_f1:.4f}")
    return model


def evaluate(model, X_test, y_test, device="cpu"):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}   <-- most important for a crisis flag")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print("              Pred: Non-Crisis  Pred: Crisis")
    print(f"True: Non-Crisis   {cm[0][0]:>10d}      {cm[0][1]:>10d}")
    print(f"True: Crisis       {cm[1][0]:>10d}      {cm[1][1]:>10d}")
    print("\nFull classification report:")
    print(classification_report(y_test, preds, target_names=["non-crisis", "crisis"]))

    fn_rate = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    print(f"False-negative rate on crisis class: {fn_rate:.4f}")
    print("(This is the number that matters most in production - missed crisis messages)")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to Suicide_Detection.csv")
    parser.add_argument("--sample", type=int, default=5000, help="Total rows to use (balanced)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed", type=str, default="sentence-transformers",
                         choices=["sentence-transformers", "tfidf"],
                         help="tfidf works offline; sentence-transformers is what should ship in the repo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_data(args.csv, args.sample, seed=args.seed)

    t0 = time.time()
    if args.embed == "tfidf":
        X = embed_texts_tfidf(df["text"].astype(str).tolist())
    else:
        X = embed_texts(df["text"].astype(str).tolist())
    y = df["label"].values
    print(f"Embedding took {time.time() - t0:.1f}s | shape={X.shape}")

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )
    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    model = CrisisClassifier(input_dim=X.shape[1])
    model = train(model, X_train, y_train, X_val, y_val, epochs=args.epochs, device=device)
    metrics = evaluate(model, X_test, y_test, device=device)

    torch.save(model.state_dict(), "crisis_classifier.pt")
    print("\nSaved model weights to crisis_classifier.pt")

    with open("metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print("Saved metrics to metrics.txt")


if __name__ == "__main__":
    main()
