"""Transcribes each utterance with per-word confidence.

Route: Khoi and Khoa (Khoi leads)
Publishes: dictation_preview, recognized text to commands
Subscribes: utterances from audio_in
APIs: faster-whisper WhisperModel("large-v3-turbo", device="cuda", compute_type="float16"), transcribe(word_timestamps=True, hotwords=...)
"""

class Dictation:
    def transcribe(self, audio) -> list[dict]:
        """Return [{"text": str, "conf": float}, ...] for the preview bubble."""
        raise NotImplementedError
