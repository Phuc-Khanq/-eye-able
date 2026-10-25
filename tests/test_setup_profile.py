"""Run with: python -m unittest tests.test_setup_profile

The installer's profile pick (engine/setup_profile.py): what a PC with this much GPU memory gets.
"""
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from engine import setup_profile as sp
from engine.config import Settings


class TestPick(unittest.TestCase):
    def test_cards(self):
        cases = {None: "cpu", 0: "cpu", 4096: "cpu", 6144: "cpu", 8188: "gpu8", 12282: "gpu12",
                 16303: "full", 16376: "full", 24564: "full", 32607: "full"}
        for vram, want in cases.items():
            with self.subTest(vram=vram):
                self.assertEqual(sp.pick(vram), want)

    def test_the_demo_pc_gets_everything(self):
        self.assertEqual(sp.pick(16303), "full")      # RTX 5080, as nvidia-smi reports it
        self.assertEqual(sp.PROFILES["full"], {})     # the defaults in engine/config.py


class TestProfilesUseRealSettings(unittest.TestCase):
    def test_every_key_is_a_real_setting(self):
        known = {f.name for f in fields(Settings)}
        for name, overrides in sp.PROFILES.items():
            with self.subTest(profile=name):
                self.assertLessEqual(set(overrides), known)    # config.py would ignore a misspelled one

    def test_cpu_runs_nothing_on_the_gpu(self):
        cpu = sp.PROFILES["cpu"]
        self.assertEqual(cpu["whisper_device"], "cpu")
        self.assertFalse(cpu["use_ui_venus"])
        self.assertFalse(cpu["rewrite_dictation"])
        self.assertFalse(cpu["live_preview"])

    def test_every_profile_says_what_you_get(self):
        self.assertEqual(set(sp.WHAT_YOU_GET), set(sp.PROFILES))


class TestWriteConfig(unittest.TestCase):
    def test_writes_plain_utf8_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.local.json"
            self.assertTrue(sp.write_config("cpu", path))
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))           # no BOM
            self.assertEqual(json.loads(raw.decode("utf-8")), sp.PROFILES["cpu"])

    def test_never_overwrites_the_persons_settings(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.local.json"
            path.write_text('{"whisper_model": "models/mine"}', encoding="utf-8")
            self.assertFalse(sp.write_config("cpu", path))
            self.assertEqual(path.read_text(encoding="utf-8"), '{"whisper_model": "models/mine"}')


class TestWantedModels(unittest.TestCase):
    def settings(self, **kw):
        base = dict(whisper_model="large-v3-turbo", use_ui_venus=True, ui_venus_model="inclusionAI/UI-Venus-1.5-2B",
                    rewrite_dictation=True, rewrite_model="Qwen/Qwen2.5-3B-Instruct")
        return SimpleNamespace(**{**base, **kw})

    def test_full(self):
        self.assertEqual(sp.wanted_models(self.settings()),
                         [("whisper", "large-v3-turbo"), ("hf", "inclusionAI/UI-Venus-1.5-2B"),
                          ("hf", "Qwen/Qwen2.5-3B-Instruct")])

    def test_cpu(self):
        self.assertEqual(sp.wanted_models(self.settings(whisper_model="small", use_ui_venus=False,
                                                        rewrite_dictation=False)), [("whisper", "small")])

    def test_a_local_fine_tune_folder_needs_no_download(self):
        with tempfile.TemporaryDirectory() as d:
            got = sp.wanted_models(self.settings(whisper_model=d, use_ui_venus=False, rewrite_dictation=False))
            self.assertEqual(got, [])


class TestNvidiaSmi(unittest.TestCase):
    def test_reads_the_first_gpu(self):
        run = lambda *a, **k: SimpleNamespace(stdout="16303\n8192\n")
        orig = sp.shutil.which
        sp.shutil.which = lambda name: "nvidia-smi"
        try:
            self.assertEqual(sp.nvidia_vram_mb(run), 16303)
            self.assertIsNone(sp.nvidia_vram_mb(lambda *a, **k: SimpleNamespace(stdout="")))
        finally:
            sp.shutil.which = orig

    def test_no_driver(self):
        orig = sp.shutil.which
        sp.shutil.which = lambda name: None
        try:
            self.assertIsNone(sp.nvidia_vram_mb())
        finally:
            sp.shutil.which = orig


if __name__ == "__main__":
    unittest.main()
