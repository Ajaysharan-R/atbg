"""
OCR for chat screenshots. Uses EasyOCR, then a simple left/right bubble
heuristic to reconstruct speaker + turn order (WhatsApp/iMessage-style
screenshots: one speaker's bubbles are right-aligned, the other's left).
"""

import easyocr

_reader = None


def _get_reader(languages: list[str] = ["en", "hi"]):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(languages, gpu=False)  # CPU is fine for OCR volume here
    return _reader


def ocr_screenshot_to_turns(image_path: str, image_width: int | None = None) -> list[dict]:
    """Returns turns in reading order (top to bottom), with a best-effort
    speaker guess based on horizontal bubble position. `speaker` values are
    'speaker_a' / 'speaker_b' — the caller maps these to attacker/victim
    based on conversation context, since OCR alone can't know which is which."""
    reader = _get_reader()
    results = reader.readtext(image_path)  # [(bbox, text, confidence), ...]

    if not results:
        return []

    if image_width is None:
        image_width = max(max(p[0] for p in bbox) for bbox, _, _ in results)
    midpoint = image_width / 2

    # Sort top-to-bottom by the bubble's vertical position
    results.sort(key=lambda r: min(p[1] for p in r[0]))

    turns = []
    for i, (bbox, text, confidence) in enumerate(results):
        left_x = min(p[0] for p in bbox)
        right_x = max(p[0] for p in bbox)
        bubble_center = (left_x + right_x) / 2
        speaker = "speaker_b" if bubble_center > midpoint else "speaker_a"

        turns.append({
            "turn_index": i + 1,
            "speaker": speaker,
            "text": text,
            "timestamp": None,  # screenshots essentially never carry reliable timestamps
            "ocr_confidence": confidence,
            "needs_review": confidence < 0.6,
        })

    return turns
