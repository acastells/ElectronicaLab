"""Persistent library, bounded audio decoding, spectral inspection and supervised learning."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

EXTENSIONS = {
    ".mp3",
    ".flac",
    ".wav",
    ".aiff",
    ".aif",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}
DEFAULT_STYLES = [
    "Dubstep",
    "Drum and Bass",
    "Hardstyle",
    "Rawstyle",
    "Xtra Raw",
    "House",
    "Techno",
    "Trance",
    "EDM",
]
BASE_VERSION = "acoustic-v1"
CLAP_VERSION = "clap-htsat-unfused-v1"


def data_directory():
    root = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
        / "ElectronicaLab"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "library.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS tracks (
          id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, title TEXT, artist TEXT,
          size INTEGER, mtime INTEGER, duration REAL, codec TEXT, bitrate REAL, sr INTEGER,
          bits INTEGER, status TEXT DEFAULT 'pending', error TEXT, analysis TEXT,
          features TEXT, feature_version TEXT, labels TEXT DEFAULT '[]', reviewed INTEGER DEFAULT 0,
          predictions TEXT DEFAULT '[]', content_hash TEXT);
        CREATE TABLE IF NOT EXISTS styles (name TEXT PRIMARY KEY COLLATE NOCASE);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
        """)
        initialized = self.db.execute(
            "SELECT value FROM settings WHERE key='styles_initialized'"
        ).fetchone()
        if initialized is None:
            if not self.styles():
                self.db.executemany(
                    "INSERT OR IGNORE INTO styles VALUES (?)",
                    [(s,) for s in DEFAULT_STYLES],
                )
            self.db.execute("INSERT INTO settings VALUES ('styles_initialized', '1')")
        self.db.commit()

    def close(self):
        self.db.close()

    def rows(self):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM tracks ORDER BY title COLLATE NOCASE"
            )
        ]

    def track(self, ident):
        r = self.db.execute("SELECT * FROM tracks WHERE id=?", (ident,)).fetchone()
        return dict(r) if r else None

    def styles(self):
        return [r[0] for r in self.db.execute("SELECT name FROM styles ORDER BY name")]

    def add_style(self, name):
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Usa un nombre de 1 a 80 caracteres.")
        self.db.execute("INSERT OR IGNORE INTO styles VALUES (?)", (name,))
        self.db.commit()

    def update(self, ident, **fields):
        allowed = {
            "title",
            "artist",
            "size",
            "mtime",
            "duration",
            "codec",
            "bitrate",
            "sr",
            "bits",
            "status",
            "error",
            "analysis",
            "features",
            "feature_version",
            "labels",
            "reviewed",
            "predictions",
            "content_hash",
        }
        if not fields.keys() <= allowed:
            raise ValueError("Campo no permitido")
        self.db.execute(
            "UPDATE tracks SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?",
            (*fields.values(), ident),
        )
        self.db.commit()

    def set_labels(self, ids, labels):
        valid = self.styles()
        if any(x not in valid for x in labels):
            raise ValueError("Estilo desconocido")
        with self.db:
            self.db.executemany(
                "UPDATE tracks SET labels=?, reviewed=?, predictions=? WHERE id=?",
                [
                    (json.dumps(sorted(set(labels))), int(bool(labels)), "[]", i)
                    for i in ids
                ],
            )
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('model_dirty','1')"
            )

    def toggle_genre(self, ids, name):
        if name not in self.styles():
            raise ValueError("Estilo desconocido")
        rows = [self.track(ident) for ident in ids]
        labels_by_id = {
            row["id"]: set(json.loads(row["labels"])) for row in rows if row
        }
        remove = all(name in labels for labels in labels_by_id.values())
        with self.db:
            for ident, labels in labels_by_id.items():
                if remove:
                    labels.discard(name)
                else:
                    labels.add(name)
                self.db.execute(
                    "UPDATE tracks SET labels=?, reviewed=?, predictions=? WHERE id=?",
                    (json.dumps(sorted(labels)), int(bool(labels)), "[]", ident),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('model_dirty','1')"
            )

    def rename_style(self, old, new):
        new = new.strip()
        if not new or len(new) > 80:
            raise ValueError("Nombre no válido")
        with self.db:
            self.db.execute("UPDATE styles SET name=? WHERE name=?", (new, old))
            for r in self.rows():
                labels = [new if s == old else s for s in json.loads(r["labels"])]
                self.db.execute(
                    "UPDATE tracks SET labels=?, predictions=? WHERE id=?",
                    (json.dumps(labels), "[]", r["id"]),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('model_dirty','1')"
            )

    def invalidate_model(self):
        self.db.execute("INSERT OR REPLACE INTO settings VALUES ('model_dirty','1')")
        self.db.execute("UPDATE tracks SET predictions='[]'")

    def delete_style(self, name):
        with self.db:
            self.db.execute("DELETE FROM styles WHERE name=?", (name,))
            for row in self.rows():
                labels = json.loads(row["labels"])
                if name in labels:
                    labels.remove(name)
                    self.db.execute(
                        "UPDATE tracks SET labels=?, reviewed=0 WHERE id=?",
                        (json.dumps(labels), row["id"]),
                    )
            self.invalidate_model()

    def delete_track_file(self, ident):
        row = self.track(ident)
        if row is None:
            raise ValueError("La canción ya no existe en la biblioteca.")
        # Keep the database entry when Windows refuses to remove an open file.
        Path(row["path"]).unlink(missing_ok=True)
        with self.db:
            self.db.execute("DELETE FROM tracks WHERE id=?", (ident,))
            self.invalidate_model()

    def scan(self, folder, notify=lambda s: None, stop=None):
        import mutagen

        count = 0
        for parent, _, names in os.walk(folder):
            for name in names:
                if stop and stop.is_set():
                    return count
                p = Path(parent) / name
                if p.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    p = p.resolve()
                    st = p.stat()
                    existing = self.db.execute(
                        "SELECT * FROM tracks WHERE path=?", (str(p),)
                    ).fetchone()
                    if (
                        existing
                        and existing["size"] == st.st_size
                        and existing["mtime"] == st.st_mtime_ns
                    ):
                        continue
                    audio = mutagen.File(p)
                    info = audio.info if audio is not None else None
                    easy = mutagen.File(p, easy=True)
                    tags = easy.tags if easy is not None and easy.tags else {}
                    vals = dict(
                        title=str(tags.get("title", [p.stem])[0]),
                        artist=str(tags.get("artist", [""])[0]),
                        size=st.st_size,
                        mtime=st.st_mtime_ns,
                        duration=float(getattr(info, "length", 0)),
                        codec=p.suffix[1:].upper(),
                        bitrate=float(getattr(info, "bitrate", 0)) / 1000,
                        sr=int(getattr(info, "sample_rate", 0)),
                        bits=int(getattr(info, "bits_per_sample", 0)),
                        status="pending",
                        error=None,
                        analysis=None,
                        features=None,
                        feature_version=None,
                        predictions="[]",
                        content_hash=None,
                    )
                    if existing:
                        self.update(existing["id"], **vals, labels="[]", reviewed=0)
                        with self.db:
                            self.db.execute(
                                "INSERT OR REPLACE INTO settings VALUES ('model_dirty','1')"
                            )
                    else:
                        self.db.execute(
                            "INSERT INTO tracks (path) VALUES (?)", (str(p),)
                        )
                        self.db.commit()
                        ident = self.db.execute(
                            "SELECT id FROM tracks WHERE path=?", (str(p),)
                        ).fetchone()[0]
                        self.update(ident, **vals)
                    count += 1
                    if count % 20 == 0:
                        notify(f"Importados/actualizados: {count}")
                except Exception as e:
                    notify(f"No se pudo importar {name}: {e}")
        return count

    def backup(self, destination):
        dest = sqlite3.connect(destination)
        try:
            self.db.backup(dest)
        finally:
            dest.close()

    def export(self, destination):
        fields = [
            "path",
            "title",
            "artist",
            "codec",
            "bitrate",
            "sr",
            "bits",
            "labels",
            "reviewed",
            "analysis",
            "predictions",
        ]
        with open(destination, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in self.rows():
                # Prevent spreadsheet formula execution when opening a CSV.
                w.writerow(
                    {
                        k: (
                            "'" + v
                            if isinstance(v, str)
                            and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                            else v
                        )
                        for k, v in row.items()
                    }
                )


def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(path, offset=0, seconds=12, sr=48000):
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(offset),
            "-i",
            str(path),
            "-t",
            str(seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        timeout=90,
        creationflags=flags,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1200:])
    y = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(y) < 2048:
        raise ValueError("Audio vacío o demasiado corto.")
    return y


def spectral_profile(y, sr):
    from scipy.signal import welch

    f, p = welch(y, fs=sr, nperseg=min(8192, len(y)))
    p = np.maximum(p, 1e-20)
    ref = np.percentile(p[(f > 1000) & (f < 10000)], 75)
    db = 10 * np.log10(p / max(ref, 1e-20))
    # Require a large sustained drop, not merely a single spectral notch.
    candidates = []
    for cut in (16000, 18000, 20000):
        if sr / 2 < cut + 1500:
            continue
        lo = db[(f > cut - 1800) & (f < cut - 350)]
        hi = db[(f > cut + 350) & (f < min(cut + 1800, sr / 2 - 100))]
        tail = db[(f > cut + 350) & (f < sr / 2 - 100)]
        drop = float(np.median(lo) - np.median(hi))
        if (
            drop > 18
            and np.median(hi) < -35
            and np.percentile(tail, 90) < -28
            and np.median(lo) > -60
        ):
            candidates.append((drop, cut))
    return f, db, (max(candidates)[1] if candidates else None)


def analyze(path, duration, source_sr, engine="acoustic", clap=None):
    import librosa

    offsets = sorted(
        set(
            round(max(0, duration - 12) * fraction, 2)
            for fraction in (0.18, 0.50, 0.78)
        )
    )
    acoustic = []
    embeddings = []
    bpms = []
    profiles = []
    cuts = []
    rms = []
    for offset in offsets:
        y = decode(path, offset, 12, 48000)
        # Do not infer high-frequency content above the original Nyquist frequency.
        f, db, cut = spectral_profile(y, 48000)
        if source_sr and source_sr / 2 < 21500:
            cut = None
        profiles.append(db)
        cuts.append(cut)
        rms.append(float(np.sqrt(np.mean(y * y))))
        low = librosa.resample(y, orig_sr=48000, target_sr=22050)
        mel = librosa.feature.melspectrogram(y=low, sr=22050, n_mels=64)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        onset = librosa.onset.onset_strength(y=low, sr=22050)
        tempo = librosa.feature.tempo(onset_envelope=onset, sr=22050)
        bpm = float(np.asarray(tempo).ravel()[0])
        bpms.append(bpm)
        contrast = librosa.feature.spectral_contrast(y=low, sr=22050)
        chroma = librosa.feature.chroma_stft(y=low, sr=22050, tuning=0.0)
        acoustic.append(
            np.concatenate(
                [mfcc.mean(1), mfcc.std(1), contrast.mean(1), chroma.mean(1), [bpm]]
            )
        )
        if engine == "clap":
            embeddings.append(clap.embed(y))
    x = np.mean(embeddings if embeddings else acoustic, axis=0)
    supported = [c for c in cuts if c is not None]
    cut = max(set(supported), key=supported.count) if supported else None
    if cut and supported.count(cut) < 2:
        cut = None
    keep = f <= (source_sr / 2 if source_sr else 24000)
    result = dict(
        bpm=round(float(np.median(bpms)), 1),
        bpm_fragments=[round(v, 1) for v in bpms],
        cutoff_hz=cut,
        cutoff_note="Indicio espectral; no prueba recompresión."
        if cut
        else "Sin corte claro en 16/18/20 kHz.",
        rms_dbfs=round(float(20 * np.log10(max(np.mean(rms), 1e-10))), 1),
        offsets=offsets,
        frequency_hz=f[keep][::4].tolist(),
        spectrum_db=np.mean(profiles, axis=0)[keep][::4].tolist(),
    )
    return result, np.nan_to_num(x).tolist()


class ClapEncoder:
    """Optional frozen pretrained audio encoder. Never downloads implicitly."""

    def __init__(self, root):
        import torch
        from transformers import ClapFeatureExtractor, ClapModel

        self.torch = torch
        torch.set_num_threads(2)
        model_path = Path(root) / "clap"
        if not (model_path / "config.json").exists():
            raise RuntimeError(
                "Descarga primero CLAP ejecutando download_model.py; consulta el README."
            )
        self.processor = ClapFeatureExtractor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = ClapModel.from_pretrained(model_path, local_files_only=True).eval()

    def embed(self, y):
        inputs = self.processor(y[:480000], sampling_rate=48000, return_tensors="pt")
        with self.torch.inference_mode():
            v = self.model.get_audio_features(**inputs)[0].numpy()
        return v / max(np.linalg.norm(v), 1e-8)


def content_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def train_personal(lib, version):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [
        r
        for r in lib.rows()
        if r["reviewed"]
        and r["features"]
        and r["feature_version"] == version
        and r["status"] == "done"
    ]
    unique = {}
    for r in rows:
        key = r["content_hash"] or r["path"]
        if key in unique and unique[key]["labels"] != r["labels"]:
            raise ValueError(
                "Hay copias idénticas con etiquetas diferentes. Corrígelas antes de entrenar."
            )
        unique[key] = r
    rows = list(unique.values())
    if len(rows) < 10:
        raise ValueError(
            "Necesitas al menos 10 canciones analizadas y revisadas, con estilos variados."
        )
    X = np.array([json.loads(r["features"]) for r in rows])
    labels = [set(json.loads(r["labels"])) for r in rows]
    groups = [
        r["artist"].strip().casefold() or (r["content_hash"] or r["path"]) for r in rows
    ]
    models = {}
    report = {
        "samples": len(rows),
        "version": version,
        "styles": {},
        "evaluation": "Separación por artista (o archivo si falta artista). No controla remixes con archivos distintos.",
    }
    split = None
    if len(set(groups)) >= 4 and len(rows) >= 20:
        split = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
                X, groups=groups
            )
        )
    for name in lib.styles():
        y = np.array([name in s for s in labels], dtype=int)
        pos = int(y.sum())
        neg = len(y) - pos
        entry = {"positive": pos, "negative": neg, "f1": None}
        report["styles"][name] = entry
        if min(pos, neg) < 3:
            entry["state"] = "Faltan 3 positivos o 3 negativos"
            continue

        def new_model():
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5, class_weight="balanced", max_iter=1000, random_state=42
                ),
            )

        if split is not None:
            tr, te = split
            if len(set(y[tr])) == 2 and len(set(y[te])) == 2:
                evaluation = new_model().fit(X[tr], y[tr])
                entry["f1"] = round(
                    float(f1_score(y[te], evaluation.predict(X[te]), zero_division=0)),
                    3,
                )
                entry["test_samples"] = len(te)
        models[name] = new_model().fit(X, y)
        entry["state"] = "Entrenado"
    if not models:
        raise ValueError(
            "Cada estilo entrenable necesita al menos 3 positivos y 3 negativos revisados."
        )
    # Persist plain numeric coefficients; no unsafe pickle loading.
    payload = {"version": version, "report": report, "models": {}}
    for name, m in models.items():
        scaler = m.steps[0][1]
        estimator = m.steps[1][1]
        payload["models"][name] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
        }
    target = lib.root / "personal-model.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)
    with lib.db:
        lib.db.execute("INSERT OR REPLACE INTO settings VALUES ('model_dirty','0')")
    return report


def predict_personal(lib):
    dirty = lib.db.execute(
        "SELECT value FROM settings WHERE key='model_dirty'"
    ).fetchone()
    if dirty and dirty[0] == "1":
        raise ValueError(
            "Las etiquetas han cambiado. Vuelve a entrenar el clasificador."
        )
    payload = json.loads((lib.root / "personal-model.json").read_text(encoding="utf-8"))
    total = 0
    for row in lib.rows():
        if (
            row["status"] != "done"
            or not row["features"]
            or row["feature_version"] != payload["version"]
        ):
            continue
        x = np.array(json.loads(row["features"]))
        scores = []
        for name, m in payload["models"].items():
            z = float(
                np.dot((x - np.array(m["mean"])) / np.array(m["scale"]), m["coef"])
                + m["intercept"]
            )
            score = float(1 / (1 + np.exp(-np.clip(z, -60, 60))))
            scores.append([name, round(score, 4)])
        lib.update(
            row["id"],
            predictions=json.dumps(
                sorted(scores, key=lambda t: t[1], reverse=True)[:3]
            ),
        )
        total += 1
    return total
