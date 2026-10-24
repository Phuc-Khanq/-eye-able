"""Retrain the gaze model on every saved session of a profile and report how well it does.
gaze_dot already retrains after each run; use this to see the numbers or after deleting a bad session.

Run:  .\\.venv\\Scripts\\python.exe -m tests.gaze_train --profile khang
      .\\.venv\\Scripts\\python.exe -m tests.gaze_train --list
Sessions are in calibration\\sessions\\<profile>\\ (numbers only). Delete a file there to drop a bad run.
"""
import argparse
from collections import Counter

from engine.eyes.calibration import DEFAULT_PATH, GazeModel
from engine.eyes.gaze_data import DATA_DIR, list_profiles, load_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gaze model on all saved sessions")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--list", action="store_true", help="list profiles and their sessions")
    parser.add_argument("--y-inputs", default="auto", choices=["auto", "eye", "eye+mediapipe", "mediapipe", "full"])
    parser.add_argument("--no-save", action="store_true", help="report only, don't replace the model file")
    args = parser.parse_args()

    if args.list:
        profiles = list_profiles()
        if not profiles:
            print(f"No saved sessions yet in {DATA_DIR}")
        for p in profiles:
            sessions, _ = load_sessions(p)
            print(f"{p}: {len(sessions)} session(s)")
            for s in sessions:
                roles = Counter(d.role for d in s.dots)
                print(f"   {s.path.name}  {s.screen[0]}x{s.screen[1]}  " + ", ".join(f"{n} {r}" for r, n in roles.items()))
        return

    everything, notes = load_sessions(args.profile)
    for n in notes:
        print(n)
    if not everything:
        print(f"No sessions for profile '{args.profile}'. Run: python -m tests.gaze_dot --profile {args.profile}")
        return
    screen = Counter(tuple(s.screen) for s in everything).most_common(1)[0][0]
    sessions = [s for s in everything if tuple(s.screen) == screen]
    if len(sessions) < len(everything):
        print(f"Using the {len(sessions)} session(s) on a {screen[0]}x{screen[1]} screen; skipping {len(everything) - len(sessions)} from other screens.")

    model = GazeModel(*screen)
    report = model.fit_sessions(sessions, y_inputs=args.y_inputs)
    print(f"\nProfile '{args.profile}': {report.summary()}")
    print(report.compare())
    if report.mode == "dot":
        print("Only one session so far: this is leave-one-dot-out. Run gaze_dot again (full or --quick) to add sessions.")
    else:
        print(f"Up/down by input (new-session test): " + ", ".join(f"{n} {e:.0f} px" for n, e in report.y_compare.items()))
        print("New-session test = train on the other sessions, re-center on one or a few dots, measure the rest.")
    if not args.no_save:
        model.bias = model.default_bias
        print(f"Saved to {model.save(DEFAULT_PATH)}. Next time: python -m tests.gaze_dot --profile {args.profile} --quick")


if __name__ == "__main__":
    main()
