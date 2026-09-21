"""
OCR service for ATBG screenshot analysis.

Crops away phone status bar and chat header before OCR.
Uses EasyOCR bounding box X-positions to determine bubble side.
- Right side → victim (user messages, usually colored/right-aligned)
- Left side  → attacker (other person, usually grey/left-aligned)
"""

import re
import tempfile
from pathlib import Path

MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# ── Aggressive noise filters ─────────────────────────────────────────
SKIP_RE = re.compile(
    r"^(\d{1,2}[:.]\d{2}(\s*[aApP][mM])?$"           # timestamps 18:52 / 6:30 PM
    r"|\d{1,2}\s+\w+\s*,?\s*\d{0,4}$"                  # dates: 30 Dec, Dec 30 2023
    r"|today|yesterday|just now"                         # date separators
    r"|active\s+\d+\w*\s+ago"                           # Active 18m ago
    r"|online|last seen"                                 # status
    r"|\d{2,3}\s*%"                                     # battery 75%
    r"|read|delivered|sent|seen"                         # message status
    r"|double tap"                                       # Instagram UI
    r"|^\W{1,4}$"                                       # only symbols/icons
    r"|^[\d\s\.\$jJ]{1,8}$"                            # garbage like "18.52 $jj"
    r")",
    re.IGNORECASE,
)

def _is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) < 3:
        return True
    return bool(SKIP_RE.match(t))


def _crop_header(img):
    """Crop top ~15% (status bar + profile header) and bottom ~5% (input bar)."""
    w, h = img.size
    top = int(h * 0.15)
    bottom = int(h * 0.93)
    return img.crop((0, top, w, bottom))


def extract_with_easyocr_positional(image_path: str) -> dict:
    import easyocr
    from PIL import Image

    # Crop header/footer noise
    img = Image.open(image_path)
    img_w, img_h = img.size
    cropped = _crop_header(img)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cropped.save(tmp.name)
        tmp_path = tmp.name

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not results:
        raise ValueError("No text found in the image.")

    # Image width for midpoint calculation
    crop_w = cropped.width
    midpoint = crop_w * 0.52  # slightly right of center

    # Filter noise, keep coords
    items = []
    for bbox, text, conf in results:
        if _is_noise(text):
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        items.append({"text": text.strip(), "cx": cx, "cy": cy, "conf": conf})

    if not items:
        raise ValueError("No readable conversation text found after filtering.")

    # Sort by Y (top to bottom)
    items.sort(key=lambda x: x["cy"])

    # Merge items that are vertically close (same bubble, multi-line)
    groups = []
    MERGE_THRESHOLD = crop_w * 0.04  # ~4% of image height
    for item in items:
        if groups and abs(item["cy"] - groups[-1]["cy"]) < MERGE_THRESHOLD:
            # Same bubble — merge if on same side
            same_side = (item["cx"] > midpoint) == (groups[-1]["cx"] > midpoint)
            if same_side:
                groups[-1]["text"] += " " + item["text"]
                groups[-1]["cy"] = (groups[-1]["cy"] + item["cy"]) / 2
                groups[-1]["conf"] = (groups[-1]["conf"] + item["conf"]) / 2
                continue
        groups.append(dict(item))

    # Assign speaker from X position
    turns = []
    for g in groups:
        speaker = "victim" if g["cx"] > midpoint else "attacker"
        # Merge consecutive same-speaker turns (continuation)
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + g["text"]
        else:
            turns.append({
                "turn_index": len(turns) + 1,
                "speaker": speaker,
                "text": g["text"],
                "timestamp": None,
            })

    for i, t in enumerate(turns):
        t["turn_index"] = i + 1

    avg_conf = sum(g["conf"] for g in groups) / len(groups) if groups else 0
    raw_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)

    warnings = []
    if avg_conf < 0.6:
        warnings.append("OCR confidence is low. Review extracted text if needed.")
    if len(turns) < 2:
        warnings.append("Only one speaker detected. Try a clearer screenshot.")

    return {
        "text": raw_text,
        "messages": turns,
        "confidence": round(avg_conf, 3),
        "warnings": warnings,
    }


def process_image(file_bytes: bytes, content_type: str) -> dict:
    if content_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported file type: {content_type}. Use JPG, PNG, or WebP.")
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.")

    suffix = ".png" if "png" in content_type else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = extract_with_easyocr_positional(tmp_path)
    except ImportError:
        raise RuntimeError("EasyOCR not installed. Run: pip install easyocr pillow")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result
