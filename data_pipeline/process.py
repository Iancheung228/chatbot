"""
Pipeline orchestration: screenshots → CSV + JSON + JSONL fine-tuning data.

Usage:
    python -m data_pipeline.process \
        --root_folder data_pipeline/convos_folder \
        --out_csv output/all_messages.csv \
        --out_json output/all_conversations.json \
        --out_jsonl output/unsloth_chatml.jsonl

On Colab (after !git pull and unzip):
    !python data_pipeline/process.py \
        --root_folder /content/convos_folder \
        --out_csv /content/all_messages.csv \
        --out_json /content/all_conversations.json \
        --out_jsonl /content/unsloth_chatml.jsonl
"""
import os
import re
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data_pipeline.ocr import get_ocr_engine, run_ocr_with_cache, make_row_id

TIMESTAMP_PATTERN = r'^\s*(昨天|今天|早上|晚上)\s*\d{1,2}:\d{2}\s*$'

_SYSTEM_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "training", "prompts", "system_v2.txt"
)

def _load_system_prompt() -> str:
    path = os.path.normpath(_SYSTEM_PROMPT_FILE)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = _load_system_prompt()


# ---------- CSV helpers ----------

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


# ---------- reconciliation ----------

def reconcile_rows(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows into existing, preserving locked/edited rows."""
    if existing is None or existing.empty:
        final = new.copy()
    else:
        existing = existing.set_index("row_id")
        new = new.set_index("row_id")

        locked = existing[existing["locked"] == 1]
        updatable = existing[existing["locked"] == 0]

        updated = new.loc[new.index.isin(updatable.index)]
        new_only = new.loc[~new.index.isin(existing.index)]

        final = pd.concat([locked, updated, new_only]).reset_index()

    # derive stable image order from numeric part of filename
    final["image_order"] = (
        final["source_image"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    final = (
        final
        .sort_values(
            by=["conversation_id", "image_order", "y_center"],
            ascending=[True, True, True],
            kind="mergesort",  # stable sort protects locked row order
        )
        .drop(columns="image_order")
        .reset_index(drop=True)
    )

    return final


# ---------- conversation processing ----------

def process_conversation_folder(
    subfolder_path: str,
    subfolder_name: str,
    ocr,
) -> pd.DataFrame:
    """Process all images in one conversation subfolder into a DataFrame."""
    all_entries = []

    cache_dir = os.path.join(subfolder_path, "_ocr_cache")
    os.makedirs(cache_dir, exist_ok=True)

    image_filenames = sorted(
        f for f in os.listdir(subfolder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for img_name in image_filenames:
        img_path = os.path.join(subfolder_path, img_name)
        cache_path = os.path.join(cache_dir, f"{img_name}.json")

        cache = run_ocr_with_cache(img_path, cache_path, ocr)

        for e in sorted(cache["entries"], key=lambda x: x["y_center"]):
            t = str(e["text"]).strip()
            # skip pure time stamps (e.g. "14:32" or "昨天22:40")
            if re.match(r'^\s*\d{1,2}:\d{2}\s*$', t):
                continue
            if re.match(TIMESTAMP_PATTERN, t):
                continue

            all_entries.append({
                "conversation_id": subfolder_name,
                "speaker": None,  # assigned after clustering
                "text": t,
                "confidence": e["conf"],
                "y_center": e["y_center"],
                "mean_h": e["mean_h"],
                "mean_s": e["mean_s"],
                "mean_v": e["mean_v"],
                "source_image": img_name,
                "ocr_status": cache["status"],
            })

    if not all_entries:
        return pd.DataFrame()

    # ---------- KMeans speaker detection ----------
    color_data = np.array([[e["mean_h"], e["mean_s"], e["mean_v"]] for e in all_entries])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(color_data)
    labels = kmeans.labels_
    # higher saturation cluster = "user" (green WeChat bubble)
    user_cluster = int(np.argmax(kmeans.cluster_centers_[:, 1]))

    for i, e in enumerate(all_entries):
        e["speaker"] = "user" if labels[i] == user_cluster else "assistant"
        e["row_id"] = make_row_id(subfolder_name, e["source_image"], e["y_center"], e["speaker"])
        e["edited"] = 0
        e["locked"] = 0

    return pd.DataFrame(all_entries)


# ---------- runner ----------

def process_all_root(
    root_folder: str,
    out_csv: str,
    lang: str = "ch",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Walk all conversation subfolders and return (merged_df, conversations)."""
    ocr = get_ocr_engine(lang)
    all_new = []

    for sub in sorted(os.listdir(root_folder)):
        sub_path = os.path.join(root_folder, sub)
        if os.path.isdir(sub_path):
            print(f"Processing: {sub}")
            df = process_conversation_folder(sub_path, sub, ocr)
            if not df.empty:
                all_new.append(df)

    new_df = pd.concat(all_new, ignore_index=True) if all_new else pd.DataFrame()
    existing_df = read_csv_safe(out_csv) if os.path.exists(out_csv) else None
    final_df = reconcile_rows(existing_df, new_df)

    # ---------- build conversation JSON ----------
    conversations = []
    for cid, g in final_df.groupby("conversation_id"):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for _, r in g.iterrows():
            messages.append({
                "role": "user" if r["speaker"] == "user" else "assistant",
                "content": r["text"],
            })
        conversations.append({"messages": messages})

    return final_df, conversations


# ---------- post-processing ----------

def clean_conversations(conversations: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove messages with empty/NaN content; drop degenerate conversations."""
    cleaned = []
    removed = 0

    for convo in conversations:
        msgs = convo.get("messages", [])
        kept = []
        for m in msgs:
            c = m.get("content", None)
            if c is None or (isinstance(c, float) and math.isnan(c)) or (isinstance(c, str) and not c.strip()):
                removed += 1
            else:
                kept.append(m)
        if kept:
            convo["messages"] = kept
            cleaned.append(convo)

    return cleaned, removed


def to_jsonl(conversations: List[Dict], out_jsonl: str) -> int:
    """Write conversations to Unsloth ChatML JSONL format."""
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for convo in conversations:
            if "messages" not in convo or len(convo["messages"]) < 2:
                continue
            f.write(json.dumps({"messages": convo["messages"]}, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WeChat screenshots into fine-tuning data.")
    parser.add_argument("--root_folder", required=True, help="Folder containing conversation subfolders")
    parser.add_argument("--out_csv", default="output/all_messages.csv", help="Output CSV path")
    parser.add_argument("--out_json", default="output/all_conversations.json", help="Output JSON path")
    parser.add_argument("--out_jsonl", default="output/unsloth_chatml.jsonl", help="Output JSONL path")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language (default: ch)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    print("Running pipeline...")
    df, conversations = process_all_root(args.root_folder, args.out_csv, args.lang)

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {args.out_csv} ({len(df)} rows)")

    conversations, removed = clean_conversations(conversations)
    print(f"Removed {removed} empty/NaN messages")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {args.out_json} ({len(conversations)} conversations)")

    count = to_jsonl(conversations, args.out_jsonl)
    print(f"Saved JSONL: {args.out_jsonl} ({count} conversations)")
