"""
Converts each raw dataset into ONE unified JSONL schema:

{
  "conversation_id": "...", "source": "...", "channel": "chat",
  "language": "en", "is_attack": true, "scam_type": "..." | null,
  "is_synthetic": true,
  "turns": [{"turn_id": 1, "speaker": "attacker", "timestamp": null, "text": "..."}]
}

IMPORTANT — verified against the actual downloaded data (not guessed):
all HF dialogue sources store the ENTIRE conversation as a single string with
inline speaker prefixes, so every converter must split it back into turns.

  scam_dialogue        {"dialogue": "caller: .. receiver: ..", "type", "label"}
  multi_agent_scam     {"dialogue": "Innocent: .. Suspect: ..", "personality", "type", "labels"}
  single_agent_scam    {"dialogue": "Suspect: .. Innocent: ..", "type", "labels"}
  scammer_conversation {"conversation": "Person A: .. Person B: ..", "label"}
  all_scam_spam        {"text": "..", "is_spam"}   <- single-turn EMAIL, not a dialogue

Rules: never invent a timestamp, never guess scam_type / channel / language —
use null / "unknown" instead.
"""

import json
import os
import re
import uuid

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "conversations.jsonl")

# Which inline prefix means the manipulator vs. the target — taken from the
# datasets' own vocabulary, no inference beyond this mapping.
ATTACKER_PREFIXES = {"caller", "suspect", "scammer", "person a"}
VICTIM_PREFIXES = {"receiver", "innocent", "victim", "person b"}

SPEAKER_RE = re.compile(
    r"\b(caller|receiver|suspect|innocent|scammer|victim|person\s+a|person\s+b)\s*:\s*",
    re.IGNORECASE,
)

# The datasets' own `type` values -> the project's 9 scam_type categories.
# Anything not listed maps to None (unknown) rather than being force-fitted.
TYPE_MAP = {
    "ssn": "phishing",
    "refund": "phishing",
    "bank": "phishing",
    "phishing": "phishing",
    "delivery": "phishing",
    "tech_support": "tech_support",
    "techsupport": "tech_support",
    "support": "tech_support",
    "reward": "lottery_419",
    "lottery": "lottery_419",
    "prize": "lottery_419",
    "investment": "investment",
    "crypto": "investment",
    "romance": "romance",
    "job": "job_task",
    "task": "job_task",
    "insurance": "other",
    "charity": "other",
    # Benign-leaning categories in these sets — don't force a scam label.
    "appointment": None,
    "utility": None,
    "restaurant": None,
    "retail": None,
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _split_dialogue(text: str) -> list[dict]:
    """Splits a single-string dialogue into ordered turns using the inline
    speaker prefixes. Returns [] if no prefixes are found."""
    if not text:
        return []

    matches = list(SPEAKER_RE.finditer(text))
    if not matches:
        return []

    turns = []
    for i, match in enumerate(matches):
        raw_speaker = re.sub(r"\s+", " ", match.group(1).strip().lower())
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if not content:
            continue

        if raw_speaker in ATTACKER_PREFIXES:
            speaker = "attacker"
        elif raw_speaker in VICTIM_PREFIXES:
            speaker = "victim"
        else:
            speaker = "unknown"

        turns.append({
            "turn_id": len(turns) + 1,
            "speaker": speaker,
            "timestamp": None,
            "text": content,
        })

    return turns


def _iter_jsonl(raw_dir: str):
    if not os.path.isdir(raw_dir):
        return
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _convert_dialogue_source(raw_dir: str, source_name: str, id_prefix: str,
                              text_field: str, label_field: str) -> list[dict]:
    """Shared converter for the four multi-turn dialogue datasets — they differ
    only in which field holds the text and which holds the label."""
    conversations = []
    for row in _iter_jsonl(raw_dir):
        text = row.get(text_field)
        if not text:
            continue

        turns = _split_dialogue(text)
        if len(turns) < 2:      # need a real multi-turn exchange
            continue

        label = row.get(label_field)
        is_attack = bool(label) if label is not None else None

        raw_type = (row.get("type") or "").strip().lower()
        scam_type = TYPE_MAP.get(raw_type) if is_attack else None

        conversations.append({
            "conversation_id": _new_id(id_prefix),
            "source": source_name,
            "channel": "unknown",     # transcript-style; platform not stated
            "language": "en",
            "is_attack": is_attack,
            "scam_type": scam_type,
            "is_synthetic": True,      # all four are LLM-generated sets
            "turns": turns,
        })
    return conversations


def convert_scam_dialogue(raw_dir):
    return _convert_dialogue_source(raw_dir, "scam-dialogue", "scamdlg", "dialogue", "label")


def convert_multi_agent_scam(raw_dir):
    return _convert_dialogue_source(raw_dir, "multi-agent-scam", "multiagent", "dialogue", "labels")


def convert_single_agent_scam(raw_dir):
    return _convert_dialogue_source(raw_dir, "single-agent-scam", "singleagent", "dialogue", "labels")


def convert_scammer_conversation(raw_dir):
    return _convert_dialogue_source(raw_dir, "scammer-conversation", "scammerconv", "conversation", "label")


def convert_all_scam_spam(raw_dir: str) -> list[dict]:
    """all_scam_spam is single-message email/spam, NOT a dialogue. Each row
    becomes a 1-turn conversation: contributes scam *language* signal but no
    temporal/behavioural progression, so it gives no next-state supervision."""
    conversations = []
    for row in _iter_jsonl(raw_dir):
        text = (row.get("text") or "").strip()
        if not text:
            continue

        is_spam = row.get("is_spam")
        conversations.append({
            "conversation_id": _new_id("scamspam"),
            "source": "all-scam-spam",
            "channel": "email",
            "language": "unknown",     # multilingual set; per-row language not stated
            "is_attack": bool(is_spam) if is_spam is not None else None,
            "scam_type": None,          # not labelled by type in this set
            "is_synthetic": False,
            "turns": [{"turn_id": 1, "speaker": "unknown", "timestamp": None, "text": text}],
        })
    return conversations


def convert_daily_dialog(raw_dir: str) -> list[dict]:
    """DailyDialog — benign multi-turn negatives. Stored as a list under 'dialog'."""
    conversations = []
    for row in _iter_jsonl(raw_dir):
        dialog = row.get("dialog") or row.get("dialogue") or []
        if not isinstance(dialog, list) or len(dialog) < 2:
            continue
        turns = [
            {"turn_id": i + 1,
             "speaker": "speaker_a" if i % 2 == 0 else "speaker_b",
             "timestamp": None,
             "text": str(t).strip()}
            for i, t in enumerate(dialog) if str(t).strip()
        ]
        if len(turns) < 2:
            continue
        conversations.append({
            "conversation_id": _new_id("dailydialog"),
            "source": "DailyDialog",
            "channel": "chat",
            "language": "en",
            "is_attack": False,
            "scam_type": None,
            "is_synthetic": False,
            "turns": turns,
        })
    return conversations


SOURCE_CONVERTERS = {
    "scam_dialogue": convert_scam_dialogue,
    "multi_agent_scam": convert_multi_agent_scam,
    "single_agent_scam": convert_single_agent_scam,
    "scammer_conversation": convert_scammer_conversation,
    "all_scam_spam": convert_all_scam_spam,
    "daily_dialog": convert_daily_dialog,
}


def main():
    all_conversations = []
    for name, converter in SOURCE_CONVERTERS.items():
        raw_dir = os.path.join(RAW_DIR, name)
        if not os.path.isdir(raw_dir):
            print(f"[{name}] skipped — {raw_dir} not found")
            continue
        convs = converter(raw_dir)
        n_attack = sum(1 for c in convs if c["is_attack"] is True)
        n_benign = sum(1 for c in convs if c["is_attack"] is False)
        print(f"[{name}] {len(convs)} conversations  (attack={n_attack}, benign={n_benign})")
        all_conversations.extend(convs)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    total_attack = sum(1 for c in all_conversations if c["is_attack"] is True)
    total_benign = sum(1 for c in all_conversations if c["is_attack"] is False)
    multi_turn = sum(1 for c in all_conversations if len(c["turns"]) > 1)

    print(f"\nTOTAL: {len(all_conversations)} conversations -> {OUT_PATH}")
    print(f"  attack: {total_attack}   benign: {total_benign}")
    print(f"  multi-turn: {multi_turn}   single-turn: {len(all_conversations) - multi_turn}")

    if total_benign == 0:
        print("\nWARNING: no benign conversations — the risk head can only learn")
        print("'everything is an attack'. Get DailyDialog/MultiWOZ in before training.")
    elif total_benign < total_attack * 0.3:
        print("\nNOTE: benign class is small relative to attack — watch for imbalance.")


if __name__ == "__main__":
    main()
