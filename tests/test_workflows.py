"""Regression tests for data leakage, degenerate signals, and model round trips."""
import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import joblib
import numpy as np

from stiction.data import ROOT, Loop, load_loops, split_indices
from stiction.methods import (Method1, Method2, Method3, Method4, Sigmoid,
                              LinearRegression, Quantification, Quantificatioin,
                              prepare_window, stiction_bands)
from stiction.modeling import build_features, evaluate, train
from new_project.final.function_file import count_samples_in_ranges, train_test_split


class SignalTests(unittest.TestCase):
    def test_disjoint_chronological_split(self):
        a = np.arange(1000)
        train_pv, _, valid_pv, _, test_pv, _ = train_test_split(a, a, 0.1, 0.2)
        self.assertEqual((len(train_pv), len(valid_pv), len(test_pv)), (700, 100, 200))
        self.assertFalse(set(train_pv) & set(valid_pv))
        self.assertFalse(set(train_pv) & set(test_pv))
        self.assertFalse(set(valid_pv) & set(test_pv))
        np.testing.assert_array_equal(np.r_[train_pv, valid_pv, test_pv], a)

    def test_bad_split_rejected(self):
        with self.assertRaises(ValueError):
            split_indices(100, 0.5, 0.5)

    def test_sigmoid_known_curve_and_column_arrays(self):
        x = np.linspace(-3, 3, 60)
        y = 1/(1+np.exp(-2*x+0.5))
        detector = Sigmoid(y[:, None], x[:, None])
        flag, score = detector.detect_stiction()
        self.assertTrue(flag)
        self.assertAlmostEqual(score, 1.0)
        np.testing.assert_allclose(detector.params, [2, -0.5], atol=1e-4)

    def test_constant_and_nonfinite_sigmoid(self):
        self.assertEqual(Sigmoid(np.ones(60), np.zeros(60)).detect_stiction(), (False, 0.0))
        with self.assertRaises(ValueError):
            Sigmoid([1, np.nan, 2], [0, 1, 2]).detect_stiction()
        with self.assertRaises(ValueError):
            prepare_window(np.ones(60), np.arange(60))

    def test_quantification_boundary_segments(self):
        np.testing.assert_array_equal(stiction_bands([2, 5, 9, 10, 14], [1, 1, 0, 1, 1]), [3, 4])
        self.assertEqual(Quantification([0, 0], [1, 2]).perform(), 0)
        self.assertEqual(Quantificatioin(co=np.arange(12), s=[1, 1]).perform(), 11)
        self.assertEqual(Quantification(np.ones(4), np.arange(4)).perform(), 3)

    def test_all_detectors_handle_constant_short_and_column_signals(self):
        for cls in (Method1, Method2, Method3, Method4):
            detector = cls(np.zeros((24, 1)), np.zeros((24, 1)))
            flags = detector.perform()
            self.assertFalse(flags.any())
            self.assertTrue(np.isfinite(detector.omega))
            self.assertEqual(len(cls([0, 1], [0, 1]).perform()), 0)

    def test_pair_detection_is_not_overwritten(self):
        # First pair flags two windows; a negative second pair must not erase it.
        with patch("stiction.methods._statistic", side_effect=[(0, 1), (2, 1), (2, 1), (0, 1)]):
            flags = Method1(np.arange(24), np.arange(24)).perform()
        np.testing.assert_array_equal(flags, [1, 1, 0])

    def test_linear_zero_slopes_and_alias(self):
        self.assertFalse(LinearRegression(np.ones(12), op=np.ones(12)).detect_stiction().any())
        self.assertTrue(LinearRegression(np.ones(12), op=np.arange(12)).detect_stiction().all())

    def test_missing_window_preserves_raw_sample_indices(self):
        rng = np.random.default_rng(31)
        pv, op = rng.normal(size=1200), rng.random(1200)
        pv[65] = np.nan
        x, y, rows, coverage = build_features([Loop("gap", pv, op)], "train")
        self.assertTrue(np.isfinite(x).all())
        self.assertNotIn(60, [r["start"] for r in rows])
        self.assertIn(120, [r["start"] for r in rows])
        self.assertEqual(coverage[0]["skipped"]["nonfinite_or_constant"], 1)

    def test_partitions_only_use_their_own_windows(self):
        rng = np.random.default_rng(9)
        loop = Loop("example", rng.normal(size=1200), rng.random(1200))
        first = {}
        for part, (a, b) in split_indices(1200).items():
            x, y, rows, _ = build_features([loop], part)
            self.assertTrue(all(a <= r["start"] < r["stop_exclusive"] <= b for r in rows))
            self.assertEqual(x.shape[1], 120)
            first[part] = (x, y)
        changed = Loop("example", loop.pv.copy(), loop.op.copy())
        changed.pv[960:] *= 1000
        changed.op[960:] += 50
        x, y, _, _ = build_features([changed], "train")
        np.testing.assert_array_equal(x, first["train"][0])
        np.testing.assert_array_equal(y, first["train"][1])

    def test_range_count_covers_endpoints(self):
        a = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.assertEqual(count_samples_in_ranges(a).sum(), len(a))


class ModelTests(unittest.TestCase):
    def test_best_model_saved_once_and_heldout_reload_is_identical(self):
        rng = np.random.default_rng(4)
        loops = [Loop("example", rng.normal(size=2400), rng.random(2400))]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.csv"
            source.write_text("synthetic test source", encoding="utf-8")
            config = {"source": "mat", "file": str(source), "loop_numbers": [2], "pairs": None}
            with patch("stiction.modeling.load_loops", return_value=loops), patch("stiction.modeling.ROOT", root):
                path, report = train(config, root/"run", c_values=[0.1, 1.0], epsilon_values=[0.01, 0.1])
                loaded = joblib.load(path)
                best = min(report["search"], key=lambda row: row["validation_mse"])
                self.assertEqual(loaded["best_parameters"]["C"], best["C"])
                self.assertEqual(loaded["best_parameters"]["epsilon"], best["epsilon"])
                scores = evaluate(path)
                self.assertEqual(scores, report["metrics"]["test"])
                saved = json.loads((root/"run"/"report.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["metrics"]["test"]["windows"], len(loops[0].pv[1920:])//60)
                source.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    evaluate(path)

    def test_old_models_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"old.joblib"
            joblib.dump({"feature_version": "old"}, path)
            with self.assertRaisesRegex(ValueError, "legacy/incompatible"):
                evaluate(path)


class RepositoryTests(unittest.TestCase):
    def test_nested_scripts_find_root_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("LC_FC_training/FC.py", "new_project/thesis_1/test.py"):
                result = subprocess.run([sys.executable, str(ROOT/name), "--help"],
                                        cwd=temp, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_sources_compile_and_legacy_imports_have_no_data_reads(self):
        paths = [p for p in ROOT.rglob("*.py") if not any(part in {".venv", ".git", "outputs", "build"} for part in p.relative_to(ROOT).parts)]
        self.assertGreater(len(paths), 29)
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        with patch("pandas.read_excel", side_effect=AssertionError("import read Excel")), patch("scipy.io.loadmat", side_effect=AssertionError("import read MAT")):
            for name in ["new_project.thesis_1.plot", "new_project.thesis_2.sigmoid_class", "new_project.thesis_3.test", "svr_use", "svr_test_new", "LC_FC_training.FC"]:
                importlib.import_module(name)

    def test_real_mat_and_csv_sources(self):
        for source in ("mat", "flow", "level"):
            loops = load_loops(source, loop_numbers=[2] if source == "mat" else None)
            self.assertTrue(all(len(loop.pv) > 60 for loop in loops))
            for cls in (Method1, Method2, Method3, Method4):
                detector = cls(loops[0].pv, loops[0].op)
                detector.perform()
                self.assertTrue(np.isfinite(detector.omega))
        flow = load_loops("flow")[0]
        self.assertAlmostEqual(flow.pv[-1], 102.890357971191)


if __name__ == "__main__":
    unittest.main()
