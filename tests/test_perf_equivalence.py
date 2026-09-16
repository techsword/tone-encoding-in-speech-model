"""Equivalence tests for the perf pass (P2, P3, P4, P5, P6).

Every test keeps a copy of the pre-perf implementation and asserts that the
patched code returns identical values.
"""

import itertools
import os
import pickle
import types

import numpy as np
import pytest
import torch
import torch.utils.checkpoint  # the legacy code used torch.utils.checkpoint

from numpy.random import MT19937, RandomState, SeedSequence  # noqa: E402

from tone_encoding.experiment_classification import (  # noqa: E402
    concat_raw_input_dataset,
    process_raw_input_dataset,
    run_subclass,
)
from tone_encoding.generate_audiofeature import default_n_jobs  # noqa: E402
from tone_encoding.generate_classifier_input import (  # noqa: E402
    _build_metadata_records,
    get_hidden_cnn,
)

CONTRAST_DICT = {
    'tone': {'column_filter': 'phonetic_wo_tone',
             'filter_consonant': False,
             'label': 'tone_labels'},
    'consonant': {'column_filter': 'ending',
                  'filter_consonant': True,
                  'label': 'onset'},
}


# --------------------------------------------------------------------------- #
# helpers: reference (pre-perf) implementations
# --------------------------------------------------------------------------- #
def reference_get_hidden_cnn(input_values, model):
    """Original get_hidden_cnn: checkpoint + per-window Python list stack."""
    cnn_feature_encoder = model.feature_extractor
    hidden_states = {}
    hidden_state = input_values[:, None]
    hidden_state.requires_grad = False
    layer = 0
    for conv_layer in cnn_feature_encoder.conv_layers:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_state = torch.utils.checkpoint.checkpoint(
            create_custom_forward(conv_layer),
            hidden_state,
        )
        hidden_states[layer] = hidden_state
        layer += 1

    num_frames = hidden_state.shape[-1]
    window_sizes = {0: 64, 1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
    averaged_cnn_layers = []
    for layer_idx in range(7):
        window_size = window_sizes[layer_idx]
        windows = torch.stack(
            [
                hidden_states[layer_idx][0][
                    :, window_size * y : window_size * (y + 1)
                ]
                for y in range(num_frames)
            ]
        )
        windows = windows.movedim(-1, 0)
        averaged_windows = torch.mean(windows, dim=0).cpu()
        averaged_cnn_layers.append(averaged_windows)
    return torch.stack(averaged_cnn_layers)


def reference_process_raw_input_dataset(raw_input_dataset, rs,
                                        contrast='tone',
                                        mode='heldout',
                                        group=None):
    """Original process_raw_input_dataset: concatenate on every call."""
    inputs, labels, column_filter = zip(*[
        (x['embs'],
         x[CONTRAST_DICT[contrast]['label']],
         x[CONTRAST_DICT[contrast]['column_filter']])
        for x in raw_input_dataset
    ])
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    column_filter = np.concatenate(column_filter, axis=0)

    if contrast == 'consonant':
        consonants = ['r', 'sh', 'ch', 's', 'z', 'j', 'zh', 'q', 'c', 'x']
        idx = np.isin(labels, consonants)
        inputs, labels, column_filter = inputs[idx], labels[idx], column_filter[idx]

    if mode == 'alldata':
        mask = np.zeros(1)
    elif mode == 'heldout':
        if group:
            group = [int(x) if str(x).isdigit() else x for x in group]
            gf = np.isin(labels, group)
            inputs, labels, column_filter = inputs[gf], labels[gf], column_filter[gf]
        unique_entries, counts = np.unique(column_filter, return_counts=True)
        num_test = round(0.2 * len(unique_entries))
        test_entries = rs.choice(unique_entries, size=num_test)
        mask = np.isin(column_filter, test_entries)

    while len(inputs.shape) < 3:
        inputs = np.expand_dims(inputs, axis=1)
    return inputs, labels, mask


def reference_run_subclass(emb_file, mode, seed, tgt_layers, contrast, results_path):
    """Original run_subclass: concat the raw records inside every group call."""
    rs = RandomState(MT19937(SeedSequence(seed)))
    raw_input_dataset = torch.load(emb_file)
    experiment_results = []
    from tone_encoding.experiment_classification import get_subclass_groups

    groups = get_subclass_groups(contrast)
    from tone_encoding.experiment_classification import classification_pipeline, process_emb_filename

    abs_save_path = process_emb_filename(emb_file, mode=mode, seed=seed,
                                         contrast=contrast, results_path=results_path)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    for group in groups:
        X, y, mask_array = reference_process_raw_input_dataset(
            raw_input_dataset, rs, contrast=contrast, mode=mode, group=group)
        results = classification_pipeline(X, y, seed=seed,
                                          mask_array=mask_array, tgt_layers=tgt_layers)
        experiment_results.extend([x | {'group': '-'.join(group)} for x in results])
    with open(abs_save_path, 'wb') as file:
        pickle.dump(experiment_results, file)
    return abs_save_path


def _synthetic_cnn_model():
    """A feature extractor with the wav2vec2-base conv layout (padding=0)."""
    configs = [(512, 10, 5)] + [(512, 3, 2)] * 6
    layers = []
    in_channels = 1
    for out_channels, kernel, stride in configs:
        layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel, stride=stride))
        in_channels = out_channels
    return types.SimpleNamespace(
        feature_extractor=types.SimpleNamespace(conv_layers=layers))


def _synthetic_raw_dataset(n_files=6, n_per_file=60, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    consonants = ['r', 's', 'sh', 'z', 'j', 'x', 'ch', 'c', 'zh', 'q']
    raw = []
    for _ in range(n_files):
        tone_labels = np.array([(i % 4) + 1 for i in range(n_per_file)])
        embs = rng.standard_normal((n_per_file, n_features)).astype(np.float32)
        embs += tone_labels.astype(np.float32)[:, None]
        onsets = np.array([consonants[i] for i in rng.integers(0, len(consonants), n_per_file)])
        phonetic_wo_tone = np.array([f'{o}a' for o in onsets])
        raw.append({
            'embs': embs,
            'tone_labels': tone_labels,
            'phonetic_wo_tone': phonetic_wo_tone,
            'onset': onsets,
            'ending': np.array([f'{o}a' for o in onsets]),
        })
    return raw


def _assert_same(a, b):
    assert type(a) is type(b), f'type mismatch: {type(a)} != {type(b)}'
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), 'dict key order/content differs'
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_same(left, right)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float):
        assert a == b
    else:
        assert a == b


# --------------------------------------------------------------------------- #
# P3 + P4: get_hidden_cnn
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('T', [16000, 12345, 8003])
def test_hidden_cnn_matches_reference(T):
    torch.manual_seed(0)
    model = _synthetic_cnn_model()
    wave = torch.randn(1, T)
    with torch.inference_mode():
        reference = reference_get_hidden_cnn(wave, model)
        patched = get_hidden_cnn(wave, model)
    assert reference.shape == patched.shape
    assert torch.equal(reference, patched)


# --------------------------------------------------------------------------- #
# P2: process_raw_input_dataset / run_subclass
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('contrast,mode,group', [
    ('tone', 'alldata', None),
    ('tone', 'heldout', None),
    ('tone', 'heldout', ('1', '2')),
    ('consonant', 'heldout', None),
])
def test_process_raw_input_dataset_matches_reference(contrast, mode, group):
    raw = _synthetic_raw_dataset()
    rs_ref = RandomState(MT19937(SeedSequence(42)))
    rs_new = RandomState(MT19937(SeedSequence(42)))
    ref_X, ref_y, ref_mask = reference_process_raw_input_dataset(
        raw, rs_ref, contrast=contrast, mode=mode, group=group)
    base = concat_raw_input_dataset(raw, contrast=contrast)
    new_X, new_y, new_mask = process_raw_input_dataset(
        raw, rs_new, contrast=contrast, mode=mode, group=group, base=base)
    np.testing.assert_array_equal(ref_X, new_X)
    np.testing.assert_array_equal(ref_y, new_y)
    np.testing.assert_array_equal(ref_mask, new_mask)


def test_group_loop_shares_rs_sequence():
    """The rs choice sequence must stay the same when base is cached once."""
    raw = _synthetic_raw_dataset()
    groups = list(itertools.combinations(['1', '2', '3', '4'], r=2))

    rs_ref = RandomState(MT19937(SeedSequence(42)))
    reference = [reference_process_raw_input_dataset(raw, rs_ref, 'tone', 'heldout', g)
                 for g in groups]

    rs_new = RandomState(MT19937(SeedSequence(42)))
    base = concat_raw_input_dataset(raw, contrast='tone')
    patched = [process_raw_input_dataset(raw, rs_new, 'tone', 'heldout', g, base=base)
               for g in groups]

    for (ref_X, ref_y, ref_mask), (new_X, new_y, new_mask) in zip(reference, patched):
        np.testing.assert_array_equal(ref_X, new_X)
        np.testing.assert_array_equal(ref_y, new_y)
        np.testing.assert_array_equal(ref_mask, new_mask)


def test_run_subclass_matches_reference(tmp_path):
    emb_dir = tmp_path / 'classifier_input'
    emb_dir.mkdir()
    emb_file = emb_dir / 'facebook-wav2vec2-base_thchs30_extracted-data.pt'
    torch.save(_synthetic_raw_dataset(), emb_file)

    reference_path = reference_run_subclass(
        str(emb_file), 'heldout', 42, None, 'tone', str(tmp_path / 'ref_results'))

    run_subclass(emb_file=str(emb_file), mode='heldout', seed=42, tgt_layers=None,
                 contrast='tone', results_path=str(tmp_path / 'new_results'))
    new_files = list((tmp_path / 'new_results').glob('*.pkl'))
    assert len(new_files) == 1

    with open(reference_path, 'rb') as file:
        reference_results = pickle.load(file)
    with open(new_files[0], 'rb') as file:
        new_results = pickle.load(file)
    _assert_same(reference_results, new_results)


# --------------------------------------------------------------------------- #
# P6: metadata record builder
# --------------------------------------------------------------------------- #
class _FakeDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {'embs': np.array(item['embs']),
                'file_ID': item['file_ID'],
                'tone_labels': np.array(item['tone_labels'])}


def test_build_metadata_records_matches_reference():
    metadata = {'model_ID': 'facebook/wav2vec2-base',
                'revision': None,
                'datasetname': 'thchs30'}
    dataset = _FakeDataset([
        {'embs': [i, i + 1], 'file_ID': f'f{i}', 'tone_labels': [i]}
        for i in range(7)
    ])
    reference = [metadata | dataset[i] for i in range(len(dataset))]
    for chunk_size in (1, 2, 3, 1024):
        patched = _build_metadata_records(metadata, dataset, chunk_size=chunk_size)
        _assert_same(reference, patched)


def test_build_metadata_records_save_load_identical(tmp_path):
    metadata = {'model_ID': 'facebook/wav2vec2-base',
                'revision': None,
                'datasetname': 'thchs30'}
    dataset = _FakeDataset([
        {'embs': [i, i + 1], 'file_ID': f'f{i}', 'tone_labels': [i]}
        for i in range(7)
    ])
    reference = [metadata | dataset[i] for i in range(len(dataset))]
    patched = _build_metadata_records(metadata, dataset, chunk_size=3)

    reference_path = tmp_path / 'reference.pt'
    patched_path = tmp_path / 'patched.pt'
    torch.save(reference, reference_path)
    torch.save(patched, patched_path)
    _assert_same(torch.load(reference_path, weights_only=False),
                 torch.load(patched_path, weights_only=False))


def test_read_dataset_insight_is_cached(tmp_path):
    import tone_encoding.classification as legacy_module
    import tone_encoding.experiment_classification as active_module
    import pandas as pd

    csv_path = tmp_path / 'sample_dataset.csv'
    pd.DataFrame({
        'transcription': ['word', 'word', 'word', 'word'],
        'pinyin': ['sha1', 'ra2', 'sa3', 'ta4'],
    }).to_csv(csv_path)

    for module in (active_module, legacy_module):
        module._DATASET_INSIGHT_CACHE.clear()
        first = module.read_dataset_insight(str(csv_path), filter_consonant=False)
        second = module.read_dataset_insight(str(csv_path), filter_consonant=False)
        assert first is second
        assert first.equals(second)

        filtered = module.read_dataset_insight(str(csv_path), filter_consonant=True)
        assert filtered is not first
        assert len(filtered) < len(first)


# --------------------------------------------------------------------------- #
# P5: worker count
# --------------------------------------------------------------------------- #
def test_default_n_jobs_uses_slurm_env(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '7')
    assert default_n_jobs() == 7


def test_default_n_jobs_falls_back(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', 'not-a-number')
    assert isinstance(default_n_jobs(), int)
    assert default_n_jobs() >= 1
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    assert default_n_jobs() >= 1
