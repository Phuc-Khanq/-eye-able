"""Runs the lite-mode (in-browser tracking) maths tests in node. Skipped when node isn't installed.

The page check (tests/lite_page_check.js, needs playwright) is run by hand: node tests/lite_page_check.js
"""
import shutil
import subprocess
import unittest
from pathlib import Path

JS = Path(__file__).with_name("lite_core.test.js")


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class LiteCore(unittest.TestCase):
    def test_core(self):
        r = subprocess.run(["node", str(JS)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
