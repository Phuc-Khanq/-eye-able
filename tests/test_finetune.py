"""Run with: python -m unittest tests.test_finetune

The fine-tuning pipeline's logic, with no model and no real dataset: made-up TORGO-style
folders of synthetic tones (written to a temp folder and deleted), prompt cleaning, splits,
seen/unseen groups, archive extraction, batch mixing and the results page.
"""
import io
import json
import tarfile
import tempfile
import unittest
from unittest import mock
import wave
from pathlib import Path

import numpy as np

from engine.voice.finetune import common, get_data, prepare, run_all
from engine.voice.finetune.common import Clip


def write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def has_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401
        return True
    except Exception:
        return False


class TestText(unittest.TestCase):
    def test_clean_torgo_prompt(self):
        c = prepare.clean_torgo_prompt
        self.assertIsNone(c("[say Ah for 5 seconds]"))
        self.assertIsNone(c("[relax your mouth in its normal position]\n"))
        self.assertIsNone(c("input/images/kitchen.jpg"))
        self.assertIsNone(c("xxx"))
        self.assertIsNone(c("   \n"))
        self.assertIsNone(c(""))
        self.assertEqual(c("lead [as in I will lead you]"), "Lead.")
        self.assertEqual(c("slip\n"), "Slip.")
        self.assertEqual(c("Twice each day he plays skillfully and with zest upon our small organ."),
                         "Twice each day he plays skillfully and with zest upon our small organ.")
        self.assertEqual(c("but he always answers, Banana oil!"), "But he always answers, Banana oil!")

    def test_whisper_style(self):
        s = common.whisper_style
        self.assertEqual(s("HE HOPED THERE WOULD BE STEW"), "He hoped there would be stew.")
        self.assertEqual(s("I THINK I'M LATE AND I'LL GO"), "I think I'm late and I'll go.")
        self.assertEqual(s("yes"), "Yes.")
        self.assertEqual(s("Is it you?"), "Is it you?")
        self.assertEqual(s(""), "")

    def test_text_key_ignores_case_and_punctuation(self):
        self.assertEqual(prepare.text_key("The quick, brown fox."), prepare.text_key("the quick brown fox"))

    def test_wer_uses_whisper_normalizer(self):
        try:
            import jiwer  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            self.skipTest("jiwer/transformers not installed (requirements-train.txt)")
        r = common.wer_table([("His belly counselled him.", "his belly counseled him"),
                              ("Open my email.", "open my mail")])
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["words"], 7)
        self.assertAlmostEqual(r["wer"], 100 * 1 / 7, places=1)

    def test_spelling_map_is_complete(self):
        spelling = json.loads(common.SPELLING_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(spelling), common.SPELLING_ENTRIES)
        self.assertEqual(spelling["counselled"], "counseled")


class TestSplitFile(unittest.TestCase):
    def test_torgo_speakers_disjoint_and_balanced(self):
        split = common.load_split()["torgo"]
        groups = [set(split[k]) for k in ("train", "dev", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(set().union(*groups), get_data.TORGO_SPEAKERS)
        self.assertEqual(sorted(s[0] for s in split["test"]), ["F", "M"])   # one female, one male
        self.assertEqual(len(split["dev"]), 1)

    def test_split_file_has_no_people_data(self):
        text = common.SPLIT_FILE.read_text(encoding="utf-8")
        self.assertNotIn(".wav", text)
        self.assertNotIn(".flac", text)


@unittest.skipUnless(has_soundfile(), "soundfile not installed (requirements-train.txt)")
class TestPrepareTorgo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        prompts = {
            # speaker: [(id, prompt, mic)]
            "F01": [("0001", "The quick brown fox jumps over the lazy dog.", "array"),
                    ("0002", "yes", "array"),
                    ("0003", "[say Ah for 5 seconds]", "array"),
                    ("0004", "Open the door please.", "head")],      # no array mic: head mic
            "M02": [("0001", "the quick brown fox jumps over the lazy dog", "array"),
                    ("0002", "xxx", "array")],
            "M05": [("0001", "The quick brown fox jumps over the lazy dog.", "array"),
                    ("0002", "A brand new sentence for dev.", "array")],
            "F04": [("0001", "The quick brown fox jumps over the lazy dog.", "array"),
                    ("0002", "Nobody in training said this one.", "array"),
                    ("0003", "slip", "array"),
                    ("0004", "go home", "array"),
                    ("0005", "input/images/kitchen.jpg", "array"),
                    ("0006", "No audio for this prompt at all.", None)],
            "M01": [("0001", "Another unseen sentence here.", "array")],
        }
        for spk, items in prompts.items():
            session = self.data / "torgo" / spk / "Session1"
            for uid, text, mic in items:
                p = session / "prompts" / f"{uid}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text + "\n", encoding="utf-8")
                if mic:
                    write_wav(session / f"wav_{mic}Mic" / f"{uid}.wav")

    def tearDown(self):
        self.tmp.cleanup()

    def test_groups_splits_and_cleaning(self):
        from collections import Counter

        stats = Counter()
        clips = prepare.scan_torgo(self.data, common.load_split(), stats)
        by = {(c.speaker, Path(c.path).stem): c for c in clips}
        # splits by speaker
        self.assertEqual({c.split for c in clips if c.speaker == "F01"}, {"train"})
        self.assertEqual({c.split for c in clips if c.speaker == "M05"}, {"dev"})
        self.assertEqual({c.split for c in clips if c.speaker in ("F04", "M01")}, {"test"})
        # seen / unseen / words, judged against TRAINING text only
        self.assertEqual(by[("F04", "0001")].group, "dys_seen")
        self.assertEqual(by[("F04", "0002")].group, "dys_unseen")
        self.assertEqual(by[("F04", "0003")].group, "dys_words")
        self.assertEqual(by[("F04", "0004")].group, "dys_words")       # 2 words
        self.assertEqual(by[("M05", "0002")].group, "dys_unseen")
        self.assertEqual(by[("M01", "0001")].group, "dys_unseen")
        # dropped: instructions, xxx, pictures, missing audio
        self.assertNotIn(("F01", "0003"), by)
        self.assertNotIn(("M02", "0002"), by)
        self.assertNotIn(("F04", "0005"), by)
        self.assertNotIn(("F04", "0006"), by)
        self.assertEqual(stats["torgo dropped: no audio"], 1)
        # head-mic fallback
        self.assertIn("wav_headMic", by[("F01", "0004")].path)
        self.assertEqual(stats["torgo head-mic fallback"], 1)
        # paths are relative, with forward slashes
        self.assertTrue(all(not Path(c.path).is_absolute() and "\\" not in c.path for c in clips))
        prepare.check_disjoint(clips)

    def test_manifest_round_trip(self):
        from collections import Counter

        clips = prepare.scan_torgo(self.data, common.load_split(), Counter())
        path = common.manifest_path(self.data)
        common.write_manifest(path, clips)
        self.assertEqual(common.read_manifest(path), clips)
        self.assertTrue(str(path).startswith(str(self.data)))     # never inside the repo code


class TestLeakCheck(unittest.TestCase):
    def test_speaker_in_two_splits_is_refused(self):
        a = Clip("a.wav", "Hi.", "F01", "dys_words", "train", "torgo", 1.0)
        b = Clip("b.wav", "Hi.", "F01", "dys_words", "test", "torgo", 1.0)
        with self.assertRaises(SystemExit):
            prepare.check_disjoint([a, b])

    def test_personal_sentence_in_two_splits_is_refused(self):
        a = Clip("a.wav", "Open my email.", "personal", "personal", "train", "personal", 1.0)
        b = Clip("b.wav", "open my email", "personal", "personal", "test", "personal", 1.0)
        with self.assertRaises(SystemExit):
            prepare.check_disjoint([a, b])


@unittest.skipUnless(has_soundfile(), "soundfile not installed (requirements-train.txt)")
class TestPersonalPack(unittest.TestCase):
    def test_split_by_sentence(self):
        import csv
        from collections import Counter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal"
            rows = []
            for i in range(20):
                for take in range(2):                       # each sentence read twice
                    name = f"{i:03d}-{take}.wav"
                    write_wav(root / name, 0.5)
                    rows.append({"file": name, "text": f"sentence number {i}"})
            with open(root / "transcripts.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["file", "text"])
                w.writeheader()
                w.writerows(rows)
            clips = prepare.scan_personal(Path(tmp), common.load_split(), Counter())
        self.assertEqual(len(clips), 40)
        self.assertEqual(sum(c.split == "test" for c in clips), 8)       # 20% of 20 sentences x 2
        prepare.check_disjoint(clips)


class TestExtract(unittest.TestCase):
    def _tar(self, path: Path, files: dict[str, bytes], mode: str) -> None:
        with tarfile.open(path, mode) as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    def test_torgo_keeps_only_array_mic_and_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._tar(tmp / "F.tar.bz2", {
                "F01/Session1/wav_arrayMic/0001.wav": b"a",
                "F01/Session1/wav_headMic/0001.wav": b"h",      # array exists: removed
                "F01/Session1/wav_headMic/0002.wav": b"h",      # only head: kept
                "F01/Session1/prompts/0001.txt": b"yes",
                "F01/Session1/prompts/0002.txt": b"no",
                "F01/Session1/phn_arrayMic/0001.PHN": b"p",     # not needed
                "F01/Session1/video/0001.avi": b"v",
                "FC01/Session1/wav_arrayMic/0001.wav": b"c",    # control speaker: skipped
                "../evil.txt": b"x",                             # path traversal: refused
            }, "w:bz2")
            out = tmp / "torgo"
            get_data.extract_torgo(tmp / "F.tar.bz2", out, lambda *a: None)
            kept = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
            self.assertEqual(kept, ["F01/Session1/prompts/0001.txt", "F01/Session1/prompts/0002.txt",
                                    "F01/Session1/wav_arrayMic/0001.wav",
                                    "F01/Session1/wav_headMic/0002.wav"])
            self.assertFalse((tmp / "evil.txt").exists())

    def test_librispeech_keeps_flac_and_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._tar(tmp / "t.tar.gz", {
                "LibriSpeech/SPEAKERS.TXT": b"; x",
                "LibriSpeech/test-clean/1/2/1-2-0000.flac": b"f",
                "LibriSpeech/test-clean/1/2/1-2.trans.txt": b"1-2-0000 HI",
                "LibriSpeech/README.TXT": b"r",
            }, "w:gz")
            out = tmp / "librispeech"
            get_data.extract_librispeech(tmp / "t.tar.gz", out, lambda *a: None)
            kept = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
            self.assertEqual(kept, ["SPEAKERS.TXT", "test-clean/1/2/1-2-0000.flac",
                                    "test-clean/1/2/1-2.trans.txt"])


class TestMixedBatches(unittest.TestCase):
    def test_typical_share(self):
        from engine.voice.finetune.train import MixedBatches

        it = iter(MixedBatches(n_main=100, n_typical=50, batch_size=16, mix=0.25, seed=1))
        for _ in range(20):
            batch = next(it)
            self.assertEqual(len(batch), 16)
            self.assertEqual(sum(i >= 100 for i in batch), 4)

    def test_no_typical(self):
        from engine.voice.finetune.train import MixedBatches

        batch = next(iter(MixedBatches(n_main=10, n_typical=0, batch_size=4, mix=0.25, seed=1)))
        self.assertTrue(all(i < 10 for i in batch))


class TestResultsPage(unittest.TestCase):
    def test_markdown_targets(self):
        def res(unseen, libri, ms):
            return {"backend": "faster-whisper", "latency_5s_ms": ms, "speakers": {},
                    "groups": {"dys_unseen": {"n": 50, "wer": unseen}, "librispeech": {"n": 100, "wer": libri}}}

        md = run_all.results_markdown({"baseline": res(50.0, 3.0, 400), "best": res(35.0, 3.2, 410),
                                       "runs": [], "errors": [], "best_run": "r32"})
        self.assertIn("PRIMARY", md)
        line = next(l for l in md.splitlines() if l.startswith("| Dysarthric, unseen"))
        self.assertIn("met", line)                    # 35 <= 50 * 0.8
        self.assertNotIn("not met", line)
        self.assertIn("Never publish", md)

    def test_out_dir_must_not_hold_data(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(out=Path(tmp), tiny=True, platform="local", work_dir=None,
                                      data_dir=Path(tmp) / "data", hours=1.0, resume_from=None)
            with self.assertRaises(SystemExit):
                run_all.Pipeline(args)


class TestPersonalRun(unittest.TestCase):
    def test_search_personal_is_one_valid_config_that_repeats_the_personal_pack(self):
        from engine.voice.finetune import common
        from engine.voice.finetune.train import TrainConfig

        configs = common.load_search(common.HERE / "search_personal.json")
        self.assertEqual(len(configs), 1)
        cfg = TrainConfig.from_dict(configs[0])
        self.assertEqual((cfg.personal_repeat, cfg.num_workers), (4, 0))      # 0 workers: Windows
        self.assertEqual(TrainConfig().personal_repeat, 1)                    # other runs are unchanged
        self.assertGreater(len(common.load_search()), 1)                      # search.json still the default

    def test_train_clips_repeat_the_personal_ones(self):
        from types import SimpleNamespace

        from engine.voice.finetune import common
        from engine.voice.finetune.train import Trainer, TrainConfig

        def clip(source, i):
            return common.Clip(f"{source}/{i}.wav", f"text {i}", "spk", "dys_unseen", "train", source, 1.0)

        clips = [clip("torgo", i) for i in range(3)] + [clip("personal", i) for i in range(2)] + [clip("librispeech", 0)]
        with mock.patch.object(common, "read_manifest", return_value=clips):
            fake = SimpleNamespace(cfg=TrainConfig(personal_repeat=4, tiny=True), data_dir=Path("."))
            main, typical, _, _ = Trainer._clips(fake)
        self.assertEqual(sum(c.source == "personal" for c in main), 8)
        self.assertEqual(sum(c.source == "torgo" for c in main), 3)
        self.assertEqual(len(typical), 1)


if __name__ == "__main__":
    unittest.main()
