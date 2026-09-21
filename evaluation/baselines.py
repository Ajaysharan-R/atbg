"""
A subset of the spec's 8 required baselines — the ones cheap enough to run
on a laptop without their own training pipeline (B1, B2). B3-B8 (XLM-R,
BiLSTM, Transformer, HMM, Static GNN, LLM zero/5-shot) need more setup and
are documented as follow-ups rather than half-implemented here — see
README.md "Extending evaluation" section.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SCAM_KEYWORDS = re.compile(
    r"\b(otp|password|pin|urgent|blocked|suspended|verify|click here|"
    r"gift card|wire transfer|act now|limited time|congratulations|winner)\b",
    re.IGNORECASE,
)


class KeywordRegexBaseline:
    """B1 — floor baseline. Flags a conversation as an attack if any turn
    matches a hand-picked keyword list."""

    def predict(self, conversation_texts: list[str]) -> float:
        full_text = " ".join(conversation_texts)
        matches = len(SCAM_KEYWORDS.findall(full_text))
        return min(matches / 3.0, 1.0)  # crude score, not a calibrated probability


class TfidfLogisticBaseline:
    """B2 — classical ML floor. Concatenates each conversation into one
    document, TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, conversations: list[list[str]], labels: list[int]):
        docs = [" ".join(turns) for turns in conversations]
        X = self.vectorizer.fit_transform(docs)
        self.classifier.fit(X, labels)
        self._fitted = True

    def predict_proba(self, conversations: list[list[str]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        docs = [" ".join(turns) for turns in conversations]
        X = self.vectorizer.transform(docs)
        return self.classifier.predict_proba(X)[:, 1]
