"""Regression test for the audio-file glob (VIVOS vs THCHS-30 layout).

THCHS-30 stores wav files directly under its data directory. VIVOS stores
them one level deeper, under ``train/waves/<speaker>/``. The glob was built
by string concatenation, ``dataset_path + "**/*.wav"``, which drops the
separator and collapses the pattern to ``waves*/*.wav``. That still matched
THCHS-30 by accident, but matched zero VIVOS files and produced empty output.

Run with pytest::

    pytest tests/test_wav_glob.py

Run without pytest::

    python tests/test_wav_glob.py
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

from tone_encoding.generate_audiofeature import find_wav_files

# A minimal 16-bit mono PCM wav file. The glob only inspects the path, so the
# content is irrelevant; writing a real header keeps the fixture honest.
_DUMMY_FRAMES = b"\x00\x00" * 16


def _write_dummy_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(_DUMMY_FRAMES)


def test_find_wav_files_matches_nested_speaker_layout(tmp_path) -> None:
    """A VIVOS-style tree must yield every wav, not zero."""
    waves = tmp_path / "train" / "waves"
    for speaker, stems in (("SPK001", ("a", "b")), ("SPK002", ("c",))):
        speaker_dir = waves / speaker
        speaker_dir.mkdir(parents=True)
        for stem in stems:
            _write_dummy_wav(speaker_dir / f"{stem}.wav")

    found = find_wav_files(str(waves))

    assert len(found) == 3
    assert {os.path.basename(f) for f in found} == {"a.wav", "b.wav", "c.wav"}


def test_find_wav_files_matches_flat_layout(tmp_path) -> None:
    """A THCHS-30-style flat tree must keep matching."""
    data = tmp_path / "data"
    data.mkdir()
    for stem in ("A1", "A2"):
        _write_dummy_wav(data / f"{stem}.wav")

    found = find_wav_files(str(data))

    assert {os.path.basename(f) for f in found} == {"A1.wav", "A2.wav"}


def test_find_wav_files_skips_collapsed_copies(tmp_path) -> None:
    """Paths that contain 'flat' stay excluded."""
    data = tmp_path / "data"
    data.mkdir()
    _write_dummy_wav(data / "keep.wav")
    flat_dir = data / "flat"
    flat_dir.mkdir()
    _write_dummy_wav(flat_dir / "drop.wav")

    found = find_wav_files(str(data))

    assert [os.path.basename(f) for f in found] == ["keep.wav"]


if __name__ == "__main__":
    import tempfile

    for test in (
        test_find_wav_files_matches_nested_speaker_layout,
        test_find_wav_files_matches_flat_layout,
        test_find_wav_files_skips_collapsed_copies,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
        print(f"{test.__name__} OK")
