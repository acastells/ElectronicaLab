import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import Library, predict_personal


class DeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lib = Library(self.root / "data")
        self.audio = self.root / "song.wav"
        self.audio.write_bytes(b"test")
        with self.lib.db:
            self.lib.db.execute(
                "INSERT INTO tracks(path, labels, reviewed) VALUES (?, ?, 1)",
                (str(self.audio), json.dumps(["Dubstep", "House"])),
            )

    def tearDown(self):
        self.lib.close()
        self.temp.cleanup()

    def test_deleted_style_stays_deleted_and_requires_review(self):
        self.lib.delete_style("Dubstep")
        row = self.lib.track(1)
        self.assertEqual(json.loads(row["labels"]), ["House"])
        self.assertEqual(row["reviewed"], 0)
        self.lib.close()
        self.lib = Library(self.root / "data")
        self.assertNotIn("Dubstep", self.lib.styles())
        with self.assertRaisesRegex(ValueError, "entrenar"):
            predict_personal(self.lib)

    def test_delete_file_and_record(self):
        self.lib.delete_track_file(1)
        self.assertFalse(self.audio.exists())
        self.assertIsNone(self.lib.track(1))

    def test_failed_file_delete_preserves_record(self):
        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.lib.delete_track_file(1)
        self.assertIsNotNone(self.lib.track(1))
        self.assertTrue(self.audio.exists())
