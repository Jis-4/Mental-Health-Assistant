"""
Drop-in inference module for Mental-Health-Assistant.

Usage in your FastAPI backend:

    from crisis_check import CrisisChecker

    checker = CrisisChecker("crisis_classifier.pt")

    @app.post("/chat")
    def chat(message: str):
        risk_score = checker.score(message)
        if risk_score > 0.5:
            # short-circuit: return a crisis-resource response,
            # optionally still log/alert, skip normal LLM call
            return {"crisis_flag": True, "risk_score": risk_score, ...}
        # otherwise continue to normal RAG/LLM pipeline
        ...
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


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


class CrisisChecker:
    def __init__(self, weights_path: str, embed_model: str = "all-MiniLM-L6-v2", embed_dim: int = 384):
        self.embedder = SentenceTransformer(embed_model)
        self.model = CrisisClassifier(input_dim=embed_dim)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def score(self, text: str) -> float:
        """Returns a risk score between 0 and 1. Higher = more crisis-like."""
        with torch.no_grad():
            emb = self.embedder.encode([text], convert_to_numpy=True)
            emb_t = torch.tensor(emb, dtype=torch.float32)
            logit = self.model(emb_t)
            prob = torch.sigmoid(logit).item()
        return prob

    def is_crisis(self, text: str, threshold: float = 0.5) -> bool:
        return self.score(text) > threshold


if __name__ == "__main__":
    # quick manual smoke test
    checker = CrisisChecker("crisis_classifier.pt")
    tests = [
        "I had a great day today, feeling good about my exams",
        "I don't want to be here anymore, I've been thinking about ending it",
    ]
    for t in tests:
        print(f"[{checker.score(t):.3f}] {t}")
