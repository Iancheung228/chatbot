"""
OCR helpers for the data pipeline.
Uses PaddleOCR with lazy singleton initialization.
"""
import os
import json
import hashlib
from typing import List, Dict

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Lazy singleton: { lang: PaddleOCR instance }
_ocr_engines: Dict[str, PaddleOCR] = {}


def get_ocr_engine(lang: str = "ch") -> PaddleOCR:
    """Return a cached PaddleOCR instance for the given language."""
    if lang not in _ocr_engines:
        _ocr_engines[lang] = PaddleOCR(use_textline_orientation=True, lang=lang)
    return _ocr_engines[lang]


# ---------- helpers ----------

def crop_bubble_color(img: np.ndarray, poly: np.ndarray, pad: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    x_min, x_max = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y_min, y_max = max(0, min(ys) - pad), min(h, max(ys) + pad)
    return img[y_min:y_max, x_min:x_max]


def mean_hsv(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.mean(axis=(0, 1))


def make_row_id(conversation_id, source_image, y_center, speaker) -> str:
    raw = f"{conversation_id}|{source_image}|{round(y_center, 1)}|{speaker}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------- OCR parsing ----------

def parse_paddle_result_dict(result_dict: dict) -> List[Dict]:
    """Extract text entries from a single PaddleOCR result dict."""
    entries = []

    rec_texts = result_dict.get("rec_texts", [])
    rec_scores = result_dict.get("rec_scores", [])
    rec_polys = result_dict.get("rec_polys", result_dict.get("dt_polys", []))
    output_img = result_dict.get("doc_preprocessor_res", {}).get("output_img", None)

    for text, conf, poly in zip(rec_texts, rec_scores, rec_polys):
        poly_arr = np.array(poly)
        crop = crop_bubble_color(output_img, poly_arr) if output_img is not None else None
        mean_h, mean_s, mean_v = mean_hsv(crop)
        y_center = float(np.mean(poly_arr[:, 1]))

        entries.append({
            "text": text,
            "conf": float(conf),
            "y_center": y_center,
            "mean_h": float(mean_h),
            "mean_s": float(mean_s),
            "mean_v": float(mean_v),
        })

    return entries


# ---------- OCR with caching ----------

def run_ocr_with_cache(img_path: str, cache_path: str, ocr: PaddleOCR) -> dict:
    """Run OCR on an image, loading from cache if available."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        results = ocr.predict(img_path)
        if isinstance(results, dict):
            results = [results]

        entries = []
        for r in results:
            entries.extend(parse_paddle_result_dict(r))

        cache = {
            "source_image": os.path.basename(img_path),
            "status": "ok" if entries else "empty",
            "error": None,
            "entries": entries,
        }

    except Exception as e:
        cache = {
            "source_image": os.path.basename(img_path),
            "status": "error",
            "error": str(e),
            "entries": [],
        }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return cache
