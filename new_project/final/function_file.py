"""Backward-compatible data helpers with disjoint chronological partitions."""
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from stiction.data import data_path, split_indices
from stiction.methods import paired


def split_data(name, samples_per_set):
    if samples_per_set < 1:
        raise ValueError("samples_per_set must be positive.")
    pv = np.loadtxt(data_path(f"new_project/csv/final/{name}_pv.csv"), delimiter=",")
    op = np.loadtxt(data_path(f"new_project/csv/final/{name}_op.csv"), delimiter=",")
    pv, op = paired(pv, op)
    count = len(pv) // samples_per_set
    return ([pv[i*samples_per_set:(i+1)*samples_per_set] for i in range(count)],
            [op[i*samples_per_set:(i+1)*samples_per_set] for i in range(count)])


def train_test_split(pv, op, valid_size=0.1, test_size=0.2):
    pv, op = paired(pv, op)
    partitions = split_indices(len(pv), valid_size, test_size)
    return tuple(array[a:b] for a, b in partitions.values() for array in (pv, op))


def sample_data(split_pv, split_op, num_samples, rng=None):
    pv, op = paired(split_pv, split_op)
    if not 1 <= num_samples <= len(pv):
        raise ValueError("num_samples must fit inside the data.")
    rng = np.random.default_rng() if rng is None else rng
    start = int(rng.integers(0, len(pv)-num_samples+1))
    return pv[start:start+num_samples], op[start:start+num_samples]


def aggregate_points(input_array, group_size=3):
    if group_size < 1:
        raise ValueError("group_size must be positive.")
    return np.asarray(input_array)[::group_size]


def partial_shuffle(arr1, arr2, window_size, rng=None):
    if len(arr1) != len(arr2) or window_size < 1:
        raise ValueError("Arrays need equal lengths and a positive window_size.")
    rng = np.random.default_rng(30) if rng is None else rng
    indices = np.arange(len(arr1))
    for start in range(0, len(indices), window_size):
        rng.shuffle(indices[start:start+window_size])
    return np.asarray(arr1)[indices], np.asarray(arr2)[indices]


def count_samples_in_ranges(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Counts require finite values.")
    return np.array([
        np.sum((arr > -1) & (arr <= 1)),
        np.sum(((arr > -2) & (arr <= -1)) | ((arr > 1) & (arr <= 2))),
        np.sum(((arr > -3) & (arr <= -2)) | ((arr > 2) & (arr <= 3))),
        np.sum(((arr > -4) & (arr <= -3)) | ((arr > 3) & (arr <= 4))),
        np.sum((arr <= -4) | (arr > 4)),
    ], dtype=int)


if __name__ == "__main__":
    from stiction.cli import main
    raise SystemExit(main(["inspect", "--source", "mat"] + sys.argv[1:]))
