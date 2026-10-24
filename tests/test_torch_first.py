"""Run with: python -m unittest tests.test_torch_first

The cuBLAS load race on Windows: Dictation.load() imports torch before faster-whisper, so torch's CUDA
libraries load first. No GPU, torch or faster-whisper needed: the imports are faked.
"""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from engine.voice import dictation
from engine.voice.dictation import Dictation, load_torch_first


class TestLoadTorchFirst(unittest.TestCase):
    def test_windows_cuda_imports_torch(self):
        imported = []
        self.assertTrue(load_torch_first("cuda", "win32", import_module=imported.append))
        self.assertEqual(imported, ["torch"])

    def test_cpu_or_not_windows_skips_it(self):
        imported = []
        self.assertFalse(load_torch_first("cpu", "win32", import_module=imported.append))
        self.assertFalse(load_torch_first("cuda", "linux", import_module=imported.append))
        self.assertEqual(imported, [])

    def test_no_torch_installed_is_fine(self):
        def no_torch(name):
            raise ModuleNotFoundError("No module named 'torch'")

        self.assertFalse(load_torch_first("cuda", "win32", import_module=no_torch))


class TestDictationLoad(unittest.TestCase):
    def load(self, device, platform, torch_installed=True):
        order = []

        class WhisperModel:
            def __init__(self, name, **kwargs):
                order.append(("whisper", name, kwargs["device"]))

        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = WhisperModel

        def import_module(name):
            if not torch_installed:
                raise ModuleNotFoundError(name)
            order.append(("import", name))

        settings = SimpleNamespace(whisper_model="large-v3-turbo", whisper_device=device,
                                   whisper_compute_type="float16")
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_fw}), \
                mock.patch.object(dictation.sys, "platform", platform), \
                mock.patch.object(dictation.importlib, "import_module", import_module):
            # the default argument was bound at definition: pass the patched one through
            with mock.patch.object(dictation, "load_torch_first",
                                   lambda dev: load_torch_first(dev, platform, import_module)):
                Dictation(settings).load()
        return order

    def test_torch_is_imported_before_whisper_loads(self):
        self.assertEqual(self.load("cuda", "win32"),
                         [("import", "torch"), ("whisper", "large-v3-turbo", "cuda")])

    def test_cpu_install_without_torch_still_loads(self):
        self.assertEqual(self.load("cuda", "win32", torch_installed=False),
                         [("whisper", "large-v3-turbo", "cuda")])
        self.assertEqual(self.load("cpu", "win32"), [("whisper", "large-v3-turbo", "cpu")])


if __name__ == "__main__":
    unittest.main()
