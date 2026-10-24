"""16 kHz mono mic stream, automatic gain for quiet voices, speech start and end.

Route: Khoi and Khoa (Khoa leads)
Publishes: listening, finished utterances (float32 audio) to dictation
Subscribes: settings (pause_seconds)
APIs: sounddevice InputStream (512-sample blocks), silero-vad VADIterator
"""

class AudioIn:
    def start(self) -> None:
        raise NotImplementedError
