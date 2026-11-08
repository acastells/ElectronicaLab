"""Bounded-memory whole-track audio overview decoding."""

import math
import subprocess
import sys
import tempfile

import numpy as np

from core import ffmpeg

SAMPLE_RATE = 48000
FFT_SIZE = 2048
MAX_COLUMNS = 1800


def read_block(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return chunks


def summarize_block(samples):
    peak = float(np.max(np.abs(samples)))
    padding = (-len(samples)) % FFT_SIZE
    padded = np.pad(samples, (0, padding))
    frames = padded.reshape(-1, FFT_SIZE)
    window = np.hanning(FFT_SIZE)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0) / window.sum() ** 2
    return peak, power.astype(np.float32)


def decode_overview(path, duration, stop):
    """Read every sample once; retain only a bounded set of visual columns."""
    block_samples = max(
        FFT_SIZE, math.ceil(max(duration, 1) * SAMPLE_RATE / MAX_COLUMNS)
    )
    block_samples = min(block_samples, SAMPLE_RATE * 30)
    peaks = []
    powers = []
    sample_count = 0
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    command = [
        ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, creationflags=flags
        )
        try:
            while not stop.is_set():
                block = read_block(process.stdout, block_samples * 4)
                if not block:
                    break
                samples = np.frombuffer(block, dtype="<f4")
                peak, power = summarize_block(samples)
                peaks.append(peak)
                powers.append(power)
                sample_count += len(samples)
                # Also bound memory when source duration metadata is inaccurate.
                if len(peaks) >= MAX_COLUMNS * 2:
                    peaks = np.asarray(peaks).reshape(-1, 2).max(axis=1).tolist()
                    powers = list(
                        np.asarray(powers)
                        .reshape(-1, 2, FFT_SIZE // 2 + 1)
                        .mean(axis=1)
                    )
                    block_samples *= 2
            if stop.is_set():
                return None
            if process.wait(timeout=10):
                errors.seek(0)
                raise RuntimeError(
                    errors.read().decode("utf-8", errors="replace")[-1200:]
                )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    if not peaks:
        raise ValueError("El archivo no contiene audio decodificable.")
    spectrum_db = 10 * np.log10(np.maximum(np.asarray(powers).T, 1e-12))
    return {
        "peaks": np.asarray(peaks),
        "spectrum_db": spectrum_db,
        "duration": sample_count / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
    }
