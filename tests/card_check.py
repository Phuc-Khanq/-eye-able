"""See the dictation card without the camera, mic or models: it plays each state for a few seconds.
Run:  .\\.venv\\Scripts\\python.exe -m tests.card_check
Open Notepad first and click into it: while the card shows, keep typing there. If your typing ever stops
going into Notepad, the card took the focus: tell Khang. The card should sit at the bottom center, on top of
everything, and clicks should go through it.
"""
import time

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.eyes.dictation_card import DictationCard
from engine.eyes.dwell_meter import DwellMeter


def main() -> None:
    bus = EventBus()
    meter = DwellMeter(bus)
    card = DictationCard(bus, hide_from_capture=False)
    meter.add_window(card.attach)
    meter.start()

    def ui(kind, **f):
        bus.publish(topics.TO_UI, messages.make(kind, **f))

    steps = [
        ("heard", lambda: bus.publish(topics.TRANSCRIPT, {"text": "Dictate see you at nine tomorrow", "words": [], "ms": 120})),
        ("preview (\"nine\" unsure)", lambda: ui(messages.DICTATION_PREVIEW, words=[
            {"text": w, "conf": 0.3 if w == "nine" else 0.95} for w in "see you at nine tomorrow".split()])),
        ("cleaned preview", lambda: ui(messages.DICTATION_PREVIEW, words=[
            {"text": w, "conf": 0.95} for w in "See you at nine tomorrow.".split()], cleaned=True,
            original="um see you at nine tomorrow")),
        ("typed", lambda: (bus.publish(topics.OS_TYPE, {"text": "See you at nine tomorrow. "}),
                           ui(messages.DICTATION_PREVIEW, words=[]))),
        ("dictation mode", lambda: ui(messages.MODE, mode="dictation")),
        ("back to pointer (hidden)", lambda: ui(messages.MODE, mode="pointer")),
    ]
    try:
        time.sleep(1.0)
        print("now: listening")
        for _ in range(15):                       # the mic hears speech for 3 s
            ui(messages.LISTENING, level=0.6, speaking=True)
            time.sleep(0.2)
        for name, step in steps:
            print(f"now: {name}")
            step()
            time.sleep(3.0)
        print("done")
    except KeyboardInterrupt:
        pass
    finally:
        meter.stop()


if __name__ == "__main__":
    main()
