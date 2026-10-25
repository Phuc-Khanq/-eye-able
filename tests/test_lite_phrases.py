"""The website's in-browser voice vocabulary (js/lite-phrases.js) is generated from the engine's own command tables:
this fails when someone changes engine/voice/commands.py or engine/config.py and forgets to run
python scripts/make_lite_phrases.py. Also runs the lite voice maths tests in node (skipped without node)."""
import importlib.util
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("make_lite_phrases", ROOT / "scripts" / "make_lite_phrases.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LitePhrases(unittest.TestCase):
    def test_generated_file_is_up_to_date(self):
        mod = _gen()
        self.assertEqual(mod.OUT.read_text(encoding="utf-8"), mod.build(),
                         "run: python scripts/make_lite_phrases.py")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_voice_logic_in_node(self):
        r = subprocess.run(["node", str(ROOT / "tests" / "lite_voice.test.js")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
