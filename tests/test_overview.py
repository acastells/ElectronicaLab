import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

from overview import decode_overview


class OverviewTests(unittest.TestCase):
    def test_overview_includes_end_of_track(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "long.wav"
            samples = np.zeros(48000 * 16, dtype=np.float32)
            samples[48000 * 14 :] = np.sin(
                np.arange(48000 * 2) * 2 * np.pi * 1000 / 48000
            )
            write(path, 48000, samples)
            result = decode_overview(path, 16, threading.Event())
            self.assertAlmostEqual(result["duration"], 16, places=3)
            self.assertEqual(float(result["peaks"][0]), 0)
            self.assertGreater(result["peaks"][-1], 0.9)
            self.assertGreater(
                result["spectrum_db"][:, -1].max(),
                result["spectrum_db"][:, 0].max() + 50,
            )

    def test_cancel_does_not_publish_partial_audio(self):
        stop = threading.Event()
        stop.set()
        self.assertIsNone(decode_overview("unused.wav", 1, stop))
