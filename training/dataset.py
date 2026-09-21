"""
PyTorch Dataset over a split's JSONL file. Each item is ONE conversation
(the model processes a whole conversation per forward call — batch size in
the usual sense doesn't apply the same way here, since conversations have
different turn counts; see train_stage_a/b/c.py for how this is looped over).
"""

import json

import torch
from torch.utils.data import Dataset

from api.services.redaction import redact_conversation
from atbg_model.embeddings import embed_turns
from atbg_model.avq_bs import BEHAVIOR_STATES
from ingestion.pipeline import _compute_elapsed_seconds

STATE_TO_IDX = {state: i for i, state in enumerate(BEHAVIOR_STATES)}


class ConversationDataset(Dataset):
    def __init__(self, jsonl_path: str, cache_embeddings: bool = True):
        # utf-8-sig transparently strips the UTF-8 BOM if present (e.g. from
        # PowerShell's default Set-Content encoding) and behaves exactly like
        # plain utf-8 otherwise, so this is safe for any UTF-8 file.
        with open(jsonl_path, encoding="utf-8-sig") as f:
            self.conversations = [json.loads(line) for line in f]
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, torch.Tensor] = {}

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        conv = self.conversations[idx]
        conv_id = conv["conversation_id"]
        texts = [t["text"] for t in conv["turns"]]

        if self.cache_embeddings and conv_id in self._embedding_cache:
            embeddings = self._embedding_cache[conv_id]
        else:
            redacted = redact_conversation(texts)
            embeddings = embed_turns(redacted)
            if self.cache_embeddings:
                self._embedding_cache[conv_id] = embeddings

        elapsed = _compute_elapsed_seconds(
            [{"timestamp": t.get("timestamp")} for t in conv["turns"]]
        )

        anchor_labels = torch.tensor(
            [STATE_TO_IDX.get(t.get("behavior_state"), -100) for t in conv["turns"]],
            dtype=torch.long,
        )

        is_attack = torch.tensor(1.0 if conv.get("is_attack") else 0.0)

        return {
            "conversation_id": conv_id,
            "embeddings": embeddings,
            "elapsed_seconds": elapsed,
            "anchor_labels": anchor_labels,
            "is_attack": is_attack,
            "speakers": [t.get("speaker", "unknown") for t in conv["turns"]],
        }


def collate_single(batch: list[dict]) -> dict:
    """No padding/batching across conversations — the model consumes one
    conversation's full turn sequence per forward call, so batch_size=1 at
    the DataLoader level is intentional here, not a bug. Gradient
    accumulation over several conversations happens in the training loop."""
    assert len(batch) == 1
    return batch[0]
