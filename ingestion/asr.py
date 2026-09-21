"""
ASR for voice-call recordings, via faster-whisper, with speaker turns
approximated from segment boundaries + a naive silence-gap heuristic.
(Real speaker diarization, e.g. WhisperX + pyannote, is the documented
upgrade path in README.md — that adds a heavier dependency chain and is
worth doing once the rest of the pipeline is proven out.)
"""

from faster_whisper import WhisperModel

_model = None


def _get_model(model_size: str = "small"):
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_call_to_turns(audio_path: str, new_turn_silence_gap: float = 1.5) -> list[dict]:
    """Splits the transcript into turns wherever there's a silence gap longer
    than new_turn_silence_gap seconds between Whisper segments — a rough
    proxy for a speaker change, since faster-whisper alone doesn't diarize."""
    model = _get_model()
    segments, _info = model.transcribe(audio_path)

    turns = []
    current_text_parts = []
    current_start = None
    last_end = None
    turn_index = 1
    speaker_toggle = "speaker_a"

    for seg in segments:
        if last_end is not None and (seg.start - last_end) > new_turn_silence_gap:
            if current_text_parts:
                turns.append({
                    "turn_index": turn_index,
                    "speaker": speaker_toggle,
                    "text": " ".join(current_text_parts).strip(),
                    "timestamp": None,  # relative offsets only, not wall-clock time
                    "start_offset_seconds": current_start,
                })
                turn_index += 1
                speaker_toggle = "speaker_b" if speaker_toggle == "speaker_a" else "speaker_a"
                current_text_parts = []
                current_start = None

        if current_start is None:
            current_start = seg.start
        current_text_parts.append(seg.text.strip())
        last_end = seg.end

    if current_text_parts:
        turns.append({
            "turn_index": turn_index,
            "speaker": speaker_toggle,
            "text": " ".join(current_text_parts).strip(),
            "timestamp": None,
            "start_offset_seconds": current_start,
        })

    return turns
