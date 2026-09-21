"""
Weak/heuristic behaviour-state labeling — NOT gold labels. Applies simple
keyword rules to tag each turn with one of the 12 states, so the pipeline
has *something* to train the AVQ-BS anchoring loss against before real
human annotation happens.

Every conversation this touches should be marked label_source="weak" in
the DB / dataset, and excluded from your final reported validation/test
accuracy numbers — the spec is explicit that gold labels come from real
double-annotation with a measured Krippendorff's alpha.
"""

import json
import os
import re

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "conversations.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "weak_labels.jsonl")

# Order matters: more specific/urgent categories are checked first so a
# turn containing both "bank" and "OTP" lands on credential_request, not authority.
RULES: list[tuple[str, re.Pattern]] = [
    ("credential_request", re.compile(r"\b(otp|password|pin|cvv|verification code|one[- ]time code)\b", re.I)),
    ("payment_request", re.compile(r"\b(transfer|pay now|deposit|gift card|wire|send money|processing fee)\b", re.I)),
    ("verification_bypass", re.compile(r"\b(disable|turn off|bypass|don'?t worry about|skip verification)\b", re.I)),
    ("isolation", re.compile(r"\b(don'?t tell|keep this secret|between us|don'?t call|do not contact)\b", re.I)),
    ("urgency", re.compile(r"\b(within \d+|urgent|immediately|right now|expires? (today|soon)|last chance)\b", re.I)),
    ("fear", re.compile(r"\b(blocked|suspended|arrest|legal action|account.*(closed|frozen)|police)\b", re.I)),
    ("authority", re.compile(r"\b(bank|government|officer|department|calling from|on behalf of|irs|income tax)\b", re.I)),
    ("reciprocity_bait", re.compile(r"\b(free gift|you'?ve won|reward|bonus|lucky|congratulations)\b", re.I)),
    ("rapport_trust", re.compile(r"\b(how are you|nice to meet|friend|trust me|i care about you)\b", re.I)),
    ("greeting", re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)),
    ("exit_closure", re.compile(r"\b(bye|goodbye|talk (later|soon)|thank you for your time)\b", re.I)),
]


def label_turn(text: str, is_attack: bool) -> str:
    for state, pattern in RULES:
        if pattern.search(text):
            return state
    return "benign" if not is_attack else "benign"  # unmatched turns default to benign either way


def main():
    if not os.path.exists(IN_PATH):
        print(f"{IN_PATH} not found — run unify_schema.py first.")
        return

    labeled = []
    with open(IN_PATH, encoding="utf-8-sig") as f:
        for line in f:
            conv = json.loads(line)
            for turn in conv["turns"]:
                turn["behavior_state"] = label_turn(turn["text"], conv.get("is_attack", False))
                turn["label_source"] = "weak"
            labeled.append(conv)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for conv in labeled:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Quick sanity summary
    state_counts: dict[str, int] = {}
    for conv in labeled:
        for turn in conv["turns"]:
            state_counts[turn["behavior_state"]] = state_counts.get(turn["behavior_state"], 0) + 1
    print(f"Labeled {len(labeled)} conversations -> {OUT_PATH}")
    print("Weak-label distribution:", state_counts)
    print("\nNOTE: these are heuristic/weak labels, not gold. Real annotation")
    print("(2 annotators + Krippendorff's alpha >= 0.70 pilot) still required.")


if __name__ == "__main__":
    main()
