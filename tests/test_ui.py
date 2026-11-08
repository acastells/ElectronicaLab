"""Isolate GUI multimedia runtimes from numerical integration tests."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class GuiProcessTest(unittest.TestCase):
    def test_ui_in_separate_process(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "ui_smoke.py",
                "-v",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
