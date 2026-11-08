import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

from core import (
    BASE_VERSION,
    Library,
    analyze,
    predict_personal,
    spectral_profile,
    train_personal,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lib = Library(self.root / "data")

    def tearDown(self):
        self.lib.close()
        self.tmp.cleanup()

    def test_scan_analyze_and_preserve_labels(self):
        sr = 44100
        t = np.arange(sr * 3) / sr
        path = self.root / "tema ñ.wav"
        write(path, sr, (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16))
        self.assertEqual(self.lib.scan(self.root), 1)
        row = self.lib.rows()[0]
        self.assertAlmostEqual(row["duration"], 3, places=1)
        self.assertEqual(row["sr"], 44100)
        self.lib.set_labels([row["id"]], ["Dubstep"])
        self.assertEqual(self.lib.scan(self.root), 0)
        self.assertEqual(self.lib.rows()[0]["reviewed"], 1)
        a, x = analyze(path, 3, sr)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(x), 60)
        self.assertIsNone(a["cutoff_hz"])
        backup = self.root / "backup.db"
        self.lib.backup(backup)
        self.assertTrue(backup.exists())
        write(path, sr, np.zeros(sr * 4, dtype=np.int16))
        self.lib.scan(self.root)
        self.assertEqual(self.lib.rows()[0]["reviewed"], 0)

    def test_cutoff_not_confused_with_broadband(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=48000 * 3)
        self.assertIsNone(spectral_profile(y, 48000)[2])
        for cut in (16000, 18000, 20000):
            spectrum = np.fft.rfft(y)
            freq = np.fft.rfftfreq(len(y), 1 / 48000)
            spectrum[freq > cut] = 0
            filtered = np.fft.irfft(spectrum, n=len(y))
            self.assertEqual(spectral_profile(filtered, 48000)[2], cut)
        self.assertIsNone(spectral_profile(np.zeros(48000), 48000)[2])

    def test_train_only_reviewed_and_rename(self):
        rng = np.random.default_rng(3)
        for i in range(41):
            with self.lib.db:
                self.lib.db.execute(
                    "INSERT INTO tracks(path,title,artist,status,features,feature_version,labels,reviewed,content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        str(i),
                        str(i),
                        f"artist-{i // 2}",
                        "done",
                        json.dumps(
                            (rng.normal(size=6) + (4 if i % 2 else -4)).tolist()
                        ),
                        BASE_VERSION,
                        json.dumps(["Dubstep" if i % 2 else "Hardstyle"]),
                        int(i < 40),
                        str(i),
                    ),
                )
        report = train_personal(self.lib, BASE_VERSION)
        self.assertEqual(report["samples"], 40)
        predict_personal(self.lib)
        self.assertEqual(self.lib.track(41)["reviewed"], 0)
        row = self.lib.track(2)
        self.assertEqual(json.loads(row["predictions"])[0][0], "Dubstep")
        self.lib.rename_style("Dubstep", "Deep Dubstep")
        self.assertIn("Deep Dubstep", self.lib.styles())
        self.assertIn("Deep Dubstep", self.lib.track(2)["labels"])
        self.assertEqual(self.lib.track(2)["predictions"], "[]")

    def test_duplicate_conflict_and_csv(self):
        for i in range(12):
            with self.lib.db:
                self.lib.db.execute(
                    "INSERT INTO tracks(path,title,artist,status,features,feature_version,labels,reviewed,content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        str(i),
                        "=FORMULA",
                        "artist",
                        "done",
                        "[1,2]",
                        BASE_VERSION,
                        json.dumps(["Dubstep" if i % 2 else "Hardstyle"]),
                        1,
                        "same" if i < 2 else str(i),
                    ),
                )
        with self.assertRaisesRegex(ValueError, "copias idénticas"):
            train_personal(self.lib, BASE_VERSION)
        target = self.root / "export.csv"
        self.lib.export(target)
        self.assertIn("'=FORMULA", target.read_text(encoding="utf-8-sig"))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            train_personal(self.lib, BASE_VERSION)


if __name__ == "__main__":
    unittest.main()
