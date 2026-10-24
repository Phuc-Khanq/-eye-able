"""Head scrolling, live: the real eyes service on your camera, with F-keys standing in for the voice commands.
Nothing is saved: camera frames stay in memory.

Run:  .\\.venv\\Scripts\\python.exe -m tests.scroll_feel
      .\\.venv\\Scripts\\python.exe -m tests.scroll_feel --camera 1      (if the picture is black)

1. Look at the center of the screen for a moment: the one-look calibration runs.
2. Open a long page (a news article, a PDF, a Reddit thread) and move the cursor onto it with your head.
3. F8 = "scroll" (the same as saying it): the cursor stays put and the page scrolls with your head.
   Chin down scrolls down, chin up scrolls up. The first 0.4 s learns "straight ahead", so hold still then.
   Nothing happens inside the dead zone; a small tilt gives a slow crawl for reading, a bigger one is fast.
   F8 again = "stop": the page coasts to a stop and the cursor is yours again, right where it stayed.
4. F6 / F7 = "scroll up" / "scroll down" by voice: one smooth glide of 5 notches.
The terminal shows a live bar: the head tilt (| is straight ahead, the dots are the dead zone) and the speed.

Tune it while you scroll (keys work from any window):
  F1 / F2 = gentler / faster (GAIN, notches per second per degree)     F3 / F4 = smaller / bigger dead zone
  F5 / F10 = lower / higher top speed (MAX_RATE)                       F11 / F12 = slower / quicker ease (smoothing)
  F9 (or Ctrl+C in the terminal) = quit; prints the values to keep.
Send Khang the values at the end and how it felt: too touchy, too slow, too jumpy, or hard to stop.
"""
import argparse
import time


def main() -> None:
    p = argparse.ArgumentParser(description="Head scrolling feel test")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from pynput import keyboard

    from engine.config import settings
    from engine.core import messages, topics
    from engine.core.events import EventBus
    import engine.eyes.scroll as S
    from engine.eyes.os_output import OsOutput
    from engine.eyes.service import EyesService

    bus = EventBus()
    eyes = EyesService(bus, settings, pointer="head")
    if args.camera is not None:
        from engine.eyes.camera import Camera
        eyes._camera_factory = lambda: Camera(index=args.camera)
    OsOutput(bus, driver=eyes.driver, scroller=eyes.scroller)
    state = {"scroll": False, "quit": False, "last": None, "line_at": 0.0}

    def on_ui(msg: dict) -> None:
        kind = msg.get("type")
        if kind == messages.STATUS:
            print(f"\n[{msg['level']}] {msg['text']}")
        elif kind == messages.SCROLL and state["scroll"]:
            now = time.monotonic()
            if now - state["line_at"] >= 0.1:
                state["line_at"] = now
                print("\r" + bar(msg["offset"], msg["rate"], msg["ready"]), end="", flush=True)

    bus.subscribe(topics.TO_UI, on_ui)

    def bar(offset: float, rate: float, ready: bool) -> str:
        span, width = 14.0, 43     # degrees each side of straight ahead, characters
        cells = [" "] * width
        mid = width // 2
        for i in range(width):
            deg = (i - mid) / mid * span
            cells[i] = "." if abs(deg) <= S.DEAD_DEG else " "
        cells[mid] = "|"
        pos = min(max(int(round(mid + offset / span * mid)), 0), width - 1)
        cells[pos] = "O"
        word = "down" if rate < -0.05 else "up" if rate > 0.05 else "still"
        return (f"  up [{''.join(cells)}] down   tilt {offset:+5.1f} deg   speed {abs(rate):4.1f} notches/s {word:<5}"
                + ("" if ready else "  (learning straight ahead, hold still)"))

    def set_scroll(on: bool) -> None:
        state["scroll"] = on
        bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="scroll" if on else "pointer"))
        print("\nSCROLL: tilt your head. F8 stops." if on else "\nStopped: the cursor is yours again.")

    def show(name: str, value) -> None:
        print(f"\n{name} = {value:g}")

    def on_press(key) -> None:
        try:
            if key == keyboard.Key.f8:
                set_scroll(not state["scroll"])
            elif key == keyboard.Key.f6:
                bus.publish(topics.OS_SCROLL, {"direction": "up"})
            elif key == keyboard.Key.f7:
                bus.publish(topics.OS_SCROLL, {"direction": "down"})
            elif key == keyboard.Key.f1:
                S.GAIN = max(S.GAIN - 0.25, 0.25)
                show("GAIN", S.GAIN)
            elif key == keyboard.Key.f2:
                S.GAIN = min(S.GAIN + 0.25, 6.0)
                show("GAIN", S.GAIN)
            elif key == keyboard.Key.f3:
                S.DEAD_DEG = max(S.DEAD_DEG - 0.5, 0.5)
                show("DEAD_DEG", S.DEAD_DEG)
            elif key == keyboard.Key.f4:
                S.DEAD_DEG = min(S.DEAD_DEG + 0.5, 6.0)
                show("DEAD_DEG", S.DEAD_DEG)
            elif key == keyboard.Key.f5:
                S.MAX_RATE = max(S.MAX_RATE - 5.0, 5.0)
                show("MAX_RATE", S.MAX_RATE)
            elif key == keyboard.Key.f10:
                S.MAX_RATE = min(S.MAX_RATE + 5.0, 60.0)
                show("MAX_RATE", S.MAX_RATE)
            elif key == keyboard.Key.f11:       # quicker ease = shorter time constants
                S.TAU_UP_S, S.TAU_DOWN_S = max(S.TAU_UP_S - 0.03, 0.03), max(S.TAU_DOWN_S - 0.05, 0.05)
                show("TAU_UP_S", S.TAU_UP_S)
                show("TAU_DOWN_S", S.TAU_DOWN_S)
            elif key == keyboard.Key.f12:       # slower ease = longer time constants
                S.TAU_UP_S, S.TAU_DOWN_S = min(S.TAU_UP_S + 0.03, 0.6), min(S.TAU_DOWN_S + 0.05, 1.0)
                show("TAU_UP_S", S.TAU_UP_S)
                show("TAU_DOWN_S", S.TAU_DOWN_S)
            elif key == keyboard.Key.f9:
                state["quit"] = True
        except Exception as exc:   # a key handler must never kill the listener
            print(f"\nkey error: {type(exc).__name__}: {exc}")

    print(__doc__)
    eyes.start()
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while not state["quit"]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        eyes.stop()
        print("\n=== Head scrolling ===")
        print(f"  values at the end: GAIN {S.GAIN:g}, DEAD_DEG {S.DEAD_DEG:g}, MAX_RATE {S.MAX_RATE:g}, "
              f"TAU_UP_S {S.TAU_UP_S:g}, TAU_DOWN_S {S.TAU_DOWN_S:g}")
        print("  (send these to Khang, with how it felt)")


if __name__ == "__main__":
    main()
