"""Run with: python -m unittest tests.test_command_bench  (the scoring only; no audio, no Whisper)"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.voice import command_bench
from engine.voice.command_bench import clips, headline, label, load_phrases, run, score, table, wrong


class TestScore(unittest.TestCase):
    def test_right_text_and_command(self):
        self.assertEqual(score("send it", "pointer", ["Send it."]),
                         {"text": True, "command": True, "empty": False, "became": "send_it"})

    def test_right_command_wrong_text(self):
        s = score("yes", "pointer", ["Yeah."])
        self.assertEqual((s["text"], s["command"]), (False, True))

    def test_wrong_command(self):
        s = score("no", "pointer", ["Know."])
        self.assertEqual((s["text"], s["command"], s["became"]), (False, False, "none"))
        s = score("scroll", "pointer", ["Scroll up."])
        self.assertEqual((s["command"], s["became"]), (False, "scroll_up"))
        s = score("send it", "pointer", ["Sent it."])
        self.assertEqual((s["text"], s["command"]), (False, True))   # wrong words, but the app still sends

    def test_empty(self):
        self.assertEqual(score("stop", "scroll", []),
                         {"text": False, "command": False, "empty": True, "became": "empty"})

    def test_stop_and_done_leave_scroll_mode(self):
        self.assertTrue(score("stop", "scroll", ["Stop."])["command"])
        self.assertTrue(score("done", "scroll", ["Done!"])["command"])
        self.assertFalse(score("stop", "scroll", ["Top."])["command"])

    def test_the_new_clicks(self):
        self.assertTrue(score("click", "pointer", ["Click."])["command"])
        self.assertTrue(score("right click", "pointer", ["Right-click."])["command"])
        self.assertTrue(score("right click", "pointer", ["Write click."])["command"])   # fuzzy: still a right click
        self.assertFalse(score("right click", "pointer", ["Click."])["command"])        # a left click instead
        self.assertTrue(score("go back", "pointer", ["Go back."])["command"])

    def test_two_utterances_for_one_clip_is_wrong(self):
        self.assertFalse(score("send it", "pointer", ["Send.", "It."])["command"])

    def test_exact_ignores_case_punctuation_and_fillers(self):
        for heard in ["Yes.", "YES!", "Um, yes please.", "okay yes"]:
            with self.subTest(heard=heard):
                self.assertTrue(score("yes", "pointer", [heard])["text"])
        self.assertTrue(score("right click", "pointer", ["Right-click."])["text"])
        self.assertFalse(score("yes", "pointer", ["Yes, yes."])["text"])

    def test_no_app_command_falls_back_to_exact(self):
        # "new tab" and "copy" aren't app commands: parse() says none for any text, so the words decide
        self.assertTrue(score("new tab", "pointer", ["New tab."])["command"])
        self.assertTrue(score("copy", "pointer", ["Copy."])["command"])
        for heard in ["Coffee.", "Cop."]:
            with self.subTest(heard=heard):
                self.assertEqual(score("copy", "pointer", [heard])["command"], False)
        # Whisper's "New tap." (Khang's live test, Oct 24) is now a new-tab command, so the app does the right thing
        for heard in ["Knew tab.", "New tap."]:
            with self.subTest(heard=heard):
                self.assertEqual(score("new tab", "pointer", [heard])["command"], True)


class TestPhrasesAndClips(unittest.TestCase):
    def test_the_phrases(self):
        phrases = load_phrases()
        self.assertEqual([p["text"] for p in phrases], ["yes", "no", "nope", "write it", "don't write it", "pause",
                                                         "take a break", "click", "right click", "go back", "escape",
                                                         "scroll", "stop", "send it", "new tab", "copy"])
        self.assertEqual(len({p["id"] for p in phrases}), 16)
        self.assertEqual({p["text"]: p["mode"] for p in phrases if p["mode"] != "pointer"},
                         {"stop": "scroll", "yes": "answer", "no": "answer", "nope": "answer", "write it": "answer",
                          "don't write it": "answer"})

    def test_an_answer_is_scored_the_way_the_app_hears_it(self):
        from engine.voice.command_bench import heard_command
        self.assertEqual(heard_command("Know.", "answer"), "no")       # what Khang got for "no"
        self.assertEqual(heard_command("Yeah.", "answer"), "yes")
        self.assertEqual(heard_command("Don't write it.", "answer"), "no")
        self.assertEqual(heard_command("Know.", "pointer"), "none")    # with nothing waiting it is just a word
        self.assertEqual(heard_command("Pot.", "pointer"), "pause")

    def test_not_in_khois_voice_check_set(self):
        import csv

        with (Path(__file__).resolve().parent / "speech_phrases.csv").open(encoding="utf-8", newline="") as f:
            ids = {r["id"] for r in csv.DictReader(f)}
        self.assertFalse(ids & {p["id"] for p in load_phrases()})

    def test_clips_order_and_missing_files(self):
        phrases = [{"id": "a", "text": "yes", "mode": "pointer"}, {"id": "b", "text": "no", "mode": "pointer"}]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [("b", "zira", "slow"), ("a", "david", "slow"), ("b", "david", "normal"), ("a", "david", "normal"),
                    ("a", "zira", "quiet")]
            lines = ["id,text,mode,voice,variant,rate,volume,file"]
            for pid, voice, variant in rows:
                lines.append(f"{pid},x,pointer,{voice},{variant},0,100,{pid}__{voice}__{variant}.wav")
                if (pid, voice, variant) != ("a", "zira", "quiet"):        # this WAV is missing
                    (d / f"{pid}__{voice}__{variant}.wav").write_bytes(b"")
            (d / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with mock.patch.object(command_bench, "CLIPS", d):
                got = clips(phrases, d / "manifest.csv")
        self.assertEqual([(c["phrase"], c["voice"], c["variant"]) for c in got],
                         [("yes", "david", "normal"), ("no", "david", "normal"), ("yes", "david", "slow"),
                          ("no", "zira", "slow")])


class TestRunAndReport(unittest.TestCase):
    ITEMS = [{"phrase": p, "mode": "scroll" if p == "stop" else "pointer", "voice": v, "variant": var}
             for var in ("normal", "slow") for v in ("david",) for p in ("yes", "stop", "copy")]

    def results(self):
        # the "stock" model hears everything right except slow "copy" ("Coffee."); "tuned" gets it all
        stock = run(self.ITEMS, lambda it: ["Coffee."] if (it["phrase"], it["variant"]) == ("copy", "slow")
                    else [it["phrase"].capitalize() + "."], log=lambda s: None)
        tuned = run(self.ITEMS, lambda it: [it["phrase"] + "."], log=lambda s: None)
        return {"stock tiny": stock, "fine-tuned elder": tuned}

    def test_headline_same_audio(self):
        self.assertEqual(headline(self.ITEMS, self.results()),
                         "Short commands, 3 phrases x 2 clips: stock tiny 83% (5/6) vs fine-tuned elder 100% (6/6), "
                         "same audio")

    def test_table_per_phrase_variant_and_total(self):
        rows = table(self.ITEMS, self.results())
        self.assertEqual(rows[0], "| phrase | variant | clips | stock tiny exact | stock tiny command | "
                                  "fine-tuned elder exact | fine-tuned elder command |")
        self.assertIn("| copy | slow | 1 | 0/1 | 0/1 | 1/1 | 1/1 |", rows)
        self.assertIn("| all | slow | 3 | 2/3 | 2/3 | 3/3 | 3/3 |", rows)
        self.assertEqual(rows[-1], "| **TOTAL** | all | 6 | 5/6 | 5/6 | 6/6 | 6/6 |")

    def test_wrong_lists_what_each_model_heard(self):
        self.assertEqual(wrong(self.ITEMS, self.results()),
                         ["- 'copy' (david/slow): stock tiny: 'Coffee.' WRONG -> none; fine-tuned elder: 'copy.' ok"])
        right = {"a": [{"command": True, "heard": "yes", "became": "yes"}]}
        self.assertEqual(wrong(self.ITEMS[:1], right), ["- none"])

    def test_label(self):
        self.assertEqual(label("large-v3-turbo"), "stock large-v3-turbo")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(label(d), f"fine-tuned {Path(d).name}")


if __name__ == "__main__":
    unittest.main()
