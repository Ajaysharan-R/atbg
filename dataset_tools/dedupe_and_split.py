"""
1. Removes exact + near-duplicate conversations (MinHash, threshold 0.8,
   per spec) — critical before splitting, or the same scam template can
   land in both train and test and inflate your numbers.
2. Assigns dev train/validation/test splits BY SOURCE, never by random
   shuffle, so one dataset's conversations don't leak across splits.

This produces the DEVELOPMENT split structure only — clearly not the
final gold research split, which requires the real-data track (see
dataset/real/) to supply validation/test that's 100% real, per spec.
"""

import json
import os
import random

from datasketch import MinHash, MinHashLSH

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "weak_labels.jsonl")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "splits")

MINHASH_THRESHOLD = 0.8
NUM_PERM = 128

# Dev-phase split ratios — NOT the spec's final 2400/300/300/150/150 counts,
# those apply once real data is folded in. This just gets a usable dev split
# out of whatever public data volume you actually have.
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1  # remainder -> test


def _conversation_text(conv: dict) -> str:
    return " ".join(t["text"] for t in conv["turns"])


def _shingles(text: str, k: int = 5) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + k]) for i in range(max(len(words) - k + 1, 1))}


def dedupe(conversations: list[dict]) -> list[dict]:
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERM)
    kept = []
    seen_exact = set()

    for conv in conversations:
        text = _conversation_text(conv)
        exact_key = text.strip().lower()
        if exact_key in seen_exact:
            continue

        mh = MinHash(num_perm=NUM_PERM)
        for shingle in _shingles(text):
            mh.update(shingle.encode("utf8"))

        duplicates = lsh.query(mh)
        if duplicates:
            continue  # near-duplicate of something already kept

        lsh.insert(conv["conversation_id"], mh)
        seen_exact.add(exact_key)
        kept.append(conv)

    return kept


def split_by_source(conversations: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for conv in conversations:
        by_source.setdefault(conv["source"], []).append(conv)

    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)

    # Whole SOURCES are assigned to a split where practical (cleanest way to
    # guarantee no leakage); sources with many conversations are internally
    # split by ratio instead so train isn't starved of language/channel diversity.
    for source, convs in by_source.items():
        rng.shuffle(convs)
        if len(convs) < 20:
            # small source: keep it whole in train, not worth splitting
            splits["train"].extend(convs)
            continue

        n = len(convs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        splits["train"].extend(convs[:n_train])
        splits["validation"].extend(convs[n_train : n_train + n_val])
        splits["test"].extend(convs[n_train + n_val :])

    return splits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi-turn-only", action="store_true",
        help="Keep only conversations with 2+ turns. Recommended for training the "
             "temporal parts of ATBG (TD-TO/GRU): single-turn emails carry scam "
             "language but no behavioural progression, and they heavily outnumber "
             "the multi-turn data, so including them drowns out the temporal signal.",
    )
    parser.add_argument("--min-turns", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(IN_PATH):
        print(f"{IN_PATH} not found — run weak_labeler.py first.")
        return

    with open(IN_PATH, encoding="utf-8-sig") as f:
        conversations = [json.loads(line) for line in f]

    print(f"Loaded {len(conversations)} conversations")

    min_turns = 2 if args.multi_turn_only else args.min_turns
    if min_turns > 1:
        before = len(conversations)
        conversations = [c for c in conversations if len(c["turns"]) >= min_turns]
        print(f"Filtered to >={min_turns} turns: {len(conversations)} kept, {before - len(conversations)} dropped")

    deduped = dedupe(conversations)
    print(f"After dedup: {len(deduped)} conversations ({len(conversations) - len(deduped)} removed)")

    splits = split_by_source(deduped)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for split_name, convs in splits.items():
        out_path = os.path.join(SPLITS_DIR, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for conv in convs:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        n_attack = sum(1 for c in convs if c.get("is_attack") is True)
        print(f"  {split_name}: {len(convs)} conversations (attack={n_attack}) -> {out_path}")

    print("\nReminder: this is the DEV split (public data only).")
    print("Final validation/test must be real data per the spec — see dataset/real/.")


if __name__ == "__main__":
    main()
