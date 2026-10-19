"""One feature/label path shared by training, validation, and held-out testing."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import DEFAULT_EXCEL, ROOT, data_path, load_loops, split_indices
from .methods import Sigmoid, prepare_window
from .plots import prediction_plot

FEATURE_VERSION = "window-delta-zscore-op-minmax-v2"


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_config(source="excel", file=None, loop_numbers=None, pairs=None):
    default = {"excel": DEFAULT_EXCEL, "mat": "isdb10.mat", "flow": "FIC-211_20241121~27.csv", "level": "LIC-106_20241121~27.csv"}
    path = data_path(file or default[source])
    return {"source": source, "file": str(path), "loop_numbers": loop_numbers, "pairs": pairs}


def build_features(loops, partition, window_size=60, sample_step=1, max_windows=0,
                   valid_size=0.1, test_size=0.2):
    if window_size < 3 or sample_step < 1 or len(range(0, window_size, sample_step)) < 3 or max_windows < 0:
        raise ValueError("Invalid window size, sampling step, or max_windows.")
    features, labels, rows, coverage = [], [], [], []
    for loop in loops:
        first, last = split_indices(len(loop.pv), valid_size, test_size)[partition]
        starts = np.arange(first, last-window_size+1, window_size, dtype=int)
        available = len(starts)
        if max_windows and available > max_windows:
            starts = starts[np.linspace(0, available-1, max_windows, dtype=int)]
        skipped = Counter()
        retained = 0
        for start in starts:
            stop = int(start + window_size)
            try:
                pv, op = prepare_window(loop.pv[start:stop], loop.op[start:stop])
            except ValueError:
                skipped["nonfinite_or_constant"] += 1
                continue
            pv, op = pv[::sample_step], op[::sample_step]
            if np.std(pv) <= 1e-12 or np.std(op) <= 1e-12:
                skipped["constant_after_sampling"] += 1
                continue
            try:
                _, label = Sigmoid(op=op, pv=pv).detect_stiction()
            except RuntimeError:
                skipped["fit_failed"] += 1
                continue
            features.append(np.r_[pv, op])
            labels.append(label)
            rows.append({"loop": loop.name, "partition": partition, "start": int(start),
                         "stop_exclusive": stop, "reference_r": label})
            retained += 1
        coverage.append({"loop": loop.name, "partition": partition, "raw_start": first,
                         "raw_stop_exclusive": last, "available_windows": available,
                         "selected_windows": len(starts), "retained_windows": retained,
                         "discarded_tail_samples": (last-first) % window_size, "skipped": dict(skipped)})
        print(f"{partition:5s} {loop.name:14s}: {retained}/{len(starts)} windows retained", flush=True)
    n_features = 2 * len(range(0, window_size, sample_step))
    return np.asarray(features, dtype=float).reshape(-1, n_features), np.asarray(labels), rows, coverage


def metrics(reference, prediction):
    if len(reference) == 0:
        raise ValueError("No usable windows in this partition. Increase data or reduce window size.")
    mse = float(mean_squared_error(reference, prediction))
    return {"windows": len(reference), "mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(reference, prediction)),
            "fraction_absolute_error_le_0_2": float(np.mean(np.abs(reference-prediction) <= 0.2))}


def prediction_rows(rows, prediction):
    return [{**row, "predicted_r": float(p), "error": float(p-row["reference_r"])}
            for row, p in zip(rows, prediction)]


def train(config, output=None, window_size=60, sample_step=1, max_windows=0,
          c_values=(1.0,), epsilon_values=(0.1,), valid_size=0.1, test_size=0.2):
    output = Path(output) if output else ROOT / "outputs" / f"{config['source']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=True)
    loops = load_loops(**config)
    settings = {"window_size": window_size, "sample_step": sample_step, "max_windows": max_windows,
                "valid_size": valid_size, "test_size": test_size}
    sets = {part: build_features(loops, part, **settings) for part in ("train", "valid", "test")}
    if any(not len(value[1]) for value in sets.values()):
        raise ValueError("Train, validation and test partitions must all contain usable windows.")
    best, best_mse, search = None, float("inf"), []
    for c in c_values:
        for epsilon in epsilon_values:
            if not np.isfinite(c) or not np.isfinite(epsilon) or c <= 0 or epsilon < 0:
                raise ValueError("C must be positive and epsilon nonnegative; both must be finite.")
            estimator = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=c, epsilon=epsilon, cache_size=512))
            estimator.fit(sets["train"][0], sets["train"][1])
            score = metrics(sets["valid"][1], estimator.predict(sets["valid"][0]))["mse"]
            search.append({"C": float(c), "epsilon": float(epsilon), "validation_mse": score})
            print(f"C={c:g}, epsilon={epsilon:g}, validation MSE={score:.6f}", flush=True)
            if score < best_mse:
                best, best_mse, best_parameters = estimator, score, {"C": float(c), "epsilon": float(epsilon)}
    if best is None:
        raise ValueError("Provide at least one C and epsilon value.")
    bundle = {"schema_version": 2, "feature_version": FEATURE_VERSION, "estimator": best,
              "dataset": config, "source_sha256": file_digest(config["file"]),
              "settings": settings, "best_parameters": best_parameters,
              "sklearn_version": sklearn.__version__, "training_label_mean": float(sets["train"][1].mean())}
    model_path = output / "model.joblib"
    joblib.dump(bundle, model_path)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": str(model_path.resolve()),
              "dataset": config, "source_sha256": bundle["source_sha256"], "feature_version": FEATURE_VERSION,
              "settings": settings, "best_parameters": best_parameters,
              "label_definition": "Rounded Pearson correlation of fitted Sigmoid OP and window-normalized OP; not physical valve ground truth.",
              "metrics": {}, "constant_mean_baseline": {}, "coverage": [], "search": search}
    for part, (x, y, rows, coverage) in sets.items():
        predictions = best.predict(x)  # Keep raw SVR predictions; no clipping or error-based filtering.
        report["metrics"][part] = metrics(y, predictions)
        report["constant_mean_baseline"][part] = metrics(y, np.full(len(y), bundle["training_label_mean"]))
        report["coverage"].extend(coverage)
        result_rows = prediction_rows(rows, predictions)
        pd.DataFrame(result_rows).to_csv(output / f"{part}_predictions.csv", index=False)
        prediction_plot(result_rows, output / f"{part}_predictions.png", f"{part.title()}: SVR versus Sigmoid reference")
    pd.DataFrame(search).to_csv(output / "parameter_search.csv", index=False)
    json_write(output / "report.json", report)
    json_write(ROOT / "outputs" / f"latest_{config['source']}.json", {"model": str(model_path.resolve())})
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"Saved model and reports: {output.resolve()}", flush=True)
    return model_path, report


def latest_model(source="excel"):
    pointer = ROOT / "outputs" / f"latest_{source}.json"
    if not pointer.exists():
        raise FileNotFoundError(f"No repaired {source} model yet. Run: python run.py train --source {source}")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["model"])


def evaluate(model_path, output=None):
    model_path = Path(model_path).resolve()
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("This is a legacy/incompatible model. Retrain using run.py train first.")
    if file_digest(bundle["dataset"]["file"]) != bundle["source_sha256"]:
        raise ValueError("Source file changed since training; retrain before evaluating held-out data.")
    loops = load_loops(**bundle["dataset"])
    x, y, rows, coverage = build_features(loops, "test", **bundle["settings"])
    prediction = bundle["estimator"].predict(x) if len(y) else np.array([])
    scores = metrics(y, prediction)
    output = Path(output) if output else model_path.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    results = prediction_rows(rows, prediction)
    pd.DataFrame(results).to_csv(output / "test_predictions.csv", index=False)
    prediction_plot(results, output / "test_predictions.png", "Held-out test: all retained windows")
    json_write(output / "report.json", {"model": str(model_path), "partition": "test", "metrics": scores, "coverage": coverage})
    print(json.dumps(scores, indent=2), flush=True)
    print(f"Saved held-out evaluation: {output.resolve()}", flush=True)
    return scores
