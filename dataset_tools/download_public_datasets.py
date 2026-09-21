"""
Downloads the agreed public datasets into dataset/raw/<source>/.

RUN THIS ON A MACHINE WITH INTERNET ACCESS (not inside this sandbox).
Requires: pip install datasets

Usage:
    python -m dataset_tools.download_public_datasets --all
    python -m dataset_tools.download_public_datasets --only scam_dialogue,covax
"""

import argparse
import json
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")

# HuggingFace dataset ids. Verify each one still resolves and check its
# license/card before using — some of these are newer/less-established
# than the original spec's source list, per the caveat you already agreed on.
HF_SOURCES = {
    "scam_dialogue": "BothBosu/scam-dialogue",
    "multi_agent_scam": "BothBosu/multi-agent-scam-conversation",
    "single_agent_scam": "BothBosu/single-agent-scam-conversations",
    "scammer_conversation": "BothBosu/Scammer-Conversation",
    "all_scam_spam": "FredZhang7/all-scam-spam",
    "daily_dialog": "daily_dialog",
    "multiwoz": "multi_woz_v22",
}

# Community mirrors without custom loading scripts, tried if the primary id
# fails even with trust_remote_code=True.
FALLBACK_IDS = {
    "daily_dialog": "li2017dailydialog/daily_dialog",
    "multiwoz": "pfb30/multi_woz_v22",
}

# Not on HuggingFace in a clean form / need manual download — documented
# here so nothing is silently skipped.
MANUAL_SOURCES = {
    "nazario": "https://monkey.org/~jose/phishing/ — download the mbox files by hand",
    "spamassassin": "https://spamassassin.apache.org/old/publiccorpus/ — download and extract",
    "uci_sms_spam": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection — download the zip",
    "fraud_419": "Search Kaggle for '419 fraud email dataset' — check the specific dataset's license before use",
    "enron": "https://www.cs.cmu.edu/~enron/ — or the 'enron_emails' Kaggle mirror",
    "cova": "Search for the COVA / COVA-X paper's release repo — not consistently on HuggingFace as of writing, verify current location",
}


def download_hf(name: str, hf_id: str):
    from datasets import load_dataset

    out_dir = os.path.join(RAW_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}] downloading {hf_id} ...")
    try:
        ds = load_dataset(hf_id)
    except Exception as e:
        # Some older datasets (DailyDialog, MultiWOZ) ship a custom loading
        # script and refuse to load without explicit opt-in. Retry with it,
        # then fall back to a community mirror that needs no custom code.
        ds = None
        try:
            print(f"[{name}] retrying with trust_remote_code=True ...")
            ds = load_dataset(hf_id, trust_remote_code=True)
        except Exception as e2:
            fallback = FALLBACK_IDS.get(name)
            if fallback:
                try:
                    print(f"[{name}] retrying with mirror {fallback} ...")
                    ds = load_dataset(fallback)
                except Exception as e3:
                    print(f"[{name}] FAILED: {e3}")
            else:
                print(f"[{name}] FAILED: {e2}")

        if ds is None:
            print(f"[{name}] Check https://huggingface.co/datasets/{hf_id} manually.")
            return

    for split_name, split_data in ds.items():
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        split_data.to_json(out_path)
        print(f"[{name}] wrote {out_path} ({len(split_data)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="comma-separated source names")
    args = parser.parse_args()

    if args.only:
        selected = args.only.split(",")
    elif args.all:
        selected = list(HF_SOURCES.keys())
    else:
        print("Pass --all or --only name1,name2. Available:", list(HF_SOURCES.keys()))
        return

    for name in selected:
        if name in HF_SOURCES:
            download_hf(name, HF_SOURCES[name])
        else:
            print(f"'{name}' is not a HuggingFace source. If it's a manual source, see MANUAL_SOURCES below.")

    print("\n── Manual-download sources (need to be fetched by hand) ──")
    for name, note in MANUAL_SOURCES.items():
        print(f"  {name}: {note}")


if __name__ == "__main__":
    main()
