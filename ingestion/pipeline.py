"""
End-to-end ingestion for one conversation, matching the pipeline in
Appendix E.7:

  turns -> redaction -> artifact extraction -> embeddings -> elapsed-time
  tensor -> ready for atbg_model.model.ATBGModel.forward()

OCR (ingestion/ocr.py) and ASR (ingestion/asr.py) run BEFORE this, to turn
a screenshot or voice recording into the `turns` list this function expects.
"""

from datetime import datetime

import torch

from api.services.redaction import redact_conversation
from api.services.artifact_extraction import extract_artifacts
from atbg_model.embeddings import embed_turns


def process_conversation(turns: list[dict]) -> dict:
    """
    turns: list of {"turn_index": int, "speaker": str, "text": str,
                     "timestamp": datetime | None}, already in order.

    Returns a dict with redacted turns, extracted artifacts, embeddings,
    and an elapsed-seconds tensor ready for the model.
    """
    raw_texts = [t["text"] for t in turns]

    # Artifacts extracted from ORIGINAL text — URLs/wallets are the useful
    # signal here and redaction would otherwise never touch them anyway,
    # but doing this first keeps the two concerns clearly separate.
    artifacts_per_turn = [extract_artifacts(text) for text in raw_texts]

    redacted_texts = redact_conversation(raw_texts)

    embeddings = embed_turns(redacted_texts)  # (T, embed_dim)

    elapsed = _compute_elapsed_seconds(turns)

    return {
        "turns_redacted": [
            {**t, "text": redacted_texts[i]} for i, t in enumerate(turns)
        ],
        "artifacts": artifacts_per_turn,
        "embeddings": embeddings,
        "elapsed_seconds": elapsed,
    }


def _compute_elapsed_seconds(turns: list[dict]) -> torch.Tensor:
    """Elapsed time before each turn. First turn = 0. If a timestamp is
    missing anywhere, falls back to a neutral constant (median-ish chat gap)
    for THAT turn only — never invents a specific value, just avoids feeding
    the model a nonsensical null."""
    FALLBACK_GAP_SECONDS = 60.0  # documented assumption, not a measurement

    values = [0.0]
    for i in range(1, len(turns)):
        t_prev = turns[i - 1].get("timestamp")
        t_curr = turns[i].get("timestamp")
        if isinstance(t_prev, datetime) and isinstance(t_curr, datetime):
            gap = max((t_curr - t_prev).total_seconds(), 0.0)
        else:
            gap = FALLBACK_GAP_SECONDS
        values.append(gap)

    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)  # (T, 1)
