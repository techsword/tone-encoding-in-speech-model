"""Round-trip check for the patched ``torch.save`` protocol.

The embeddings and audio-feature pipelines save large per-utterance
collections.  The default ``torch.save`` uses pickle protocol 2 and the new
zipfile container, which costs about 5.5x peak RAM for tens of thousands of
small arrays and raises ``OverflowError`` above 4 GiB.  The pipelines now pass
``pickle_protocol=5, _use_new_zipfile_serialization=False``.

Run with pytest::

    pytest tests/test_save_protocol.py

Run without pytest::

    python tests/test_save_protocol.py
"""

from __future__ import annotations

import numpy as np
import torch

# The patched call used by every large torch.save site.
PATCHED_SAVE_KWARGS = {"pickle_protocol": 5, "_use_new_zipfile_serialization": False}


def _representative_payload() -> list[dict]:
    """Build a many-small-arrays object like the per-utterance embedding data."""
    rng = np.random.default_rng(0)
    return [
        {
            "file_ID": f"utt-{i:05d}",
            "model_ID": "facebook/wav2vec2-base",
            "revision": None,
            "datasetname": "thchs30",
            "embs": rng.standard_normal((13, 64)).astype(np.float32),
        }
        for i in range(2000)
    ]


def _assert_same(a, b) -> None:
    """Assert an identical nested structure of dicts, lists, arrays, scalars."""
    assert type(a) is type(b), f"type mismatch: {type(a)} != {type(b)}"
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_same(left, right)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_legacy_protocol5_round_trip(tmp_path) -> None:
    """Save a representative payload with the patched call and load it back."""
    payload = _representative_payload()
    path = tmp_path / "embeddings.pt"

    torch.save(payload, path, **PATCHED_SAVE_KWARGS)
    loaded = torch.load(path, weights_only=False)

    _assert_same(payload, loaded)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_legacy_protocol5_round_trip(Path(tmp))
    print("round-trip OK")
