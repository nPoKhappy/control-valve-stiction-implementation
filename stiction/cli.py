"""Command-line entry points; every action is explicit and import-safe."""
import argparse
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import ROOT, data_path, load_loops
from .methods import (LinearRegression, Method1, Method2, Method3, Method4,
                      Sigmoid, prepare_window, stiction_bands)
from .modeling import dataset_config, evaluate, json_write, latest_model, train
from .plots import prediction_plot, signal_plot


def output_dir(value, label):
    result = Path(value) if value else ROOT / "outputs" / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def detect(loops, method="all", output=None):
    output = output_dir(output, "detection")
    summaries = []
    choices = ["method1", "method2", "method3", "method4", "linear", "sigmoid"] if method == "all" else [method]
    for loop in loops:
        for choice in choices:
            rows = []
            if choice.startswith("method"):
                cls = {"method1": Method1, "method2": Method2, "method3": Method3, "method4": Method4}[choice]
                detector = cls(loop.pv, loop.op)
                flags = detector.perform()
                width = detector.ws
                valid = detector.valid_windows
            elif choice == "linear":
                detector = LinearRegression(loop.pv, loop.op)
                flags = detector.detect_stiction()
                width, valid = detector.w, detector.valid_windows
            else:
                width = 60
                flags = np.zeros(len(loop.pv)//width, dtype=int)
                valid = np.zeros(len(flags), dtype=bool)
                for i in range(len(flags)):
                    start, stop = i*width, (i+1)*width
                    try:
                        pv, op = prepare_window(loop.pv[start:stop], loop.op[start:stop])
                        flag, r = Sigmoid(op, pv).detect_stiction()
                    except (ValueError, RuntimeError):
                        rows.append({"start": start, "stop_exclusive": stop, "valid": False, "flag": 0, "reference_r": None})
                        continue
                    valid[i], flags[i] = True, int(flag)
                    rows.append({"start": start, "stop_exclusive": stop, "valid": True, "flag": int(flag), "reference_r": r})
            if not rows:
                rows = [{"start": i*width, "stop_exclusive": (i+1)*width, "valid": bool(v), "flag": int(s)}
                        for i, (s, v) in enumerate(zip(flags, valid))]
            expanded = np.repeat(flags, width)
            bands = stiction_bands(loop.op, expanded)
            summary = {"loop": loop.name, "method": choice, "samples": len(loop.pv),
                       "windows": len(flags), "valid_windows": int(valid.sum()),
                       "invalid_windows": int((~valid).sum()), "tail_samples": len(loop.pv) % width,
                       "flagged_fraction_of_valid": float(flags.sum()/valid.sum()) if valid.any() else 0.0,
                       "op_band_p95": float(np.percentile(bands, 95)) if len(bands) else 0.0,
                       "op_band_max": float(bands.max()) if len(bands) else 0.0}
            summaries.append(summary)
            stem = f"{loop.name}_{choice}"
            pd.DataFrame(rows).to_csv(output / f"{stem}.csv", index=False)
            signal_plot(loop.pv, expanded, stem, output / f"{stem}.png")
            print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    json_write(output / "summary.json", summaries)
    print(f"Saved detection outputs: {output.resolve()}", flush=True)
    return summaries


def create_gif(loops, filename):
    from PIL import Image
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = filename.parent / f"{filename.stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    frames = []
    try:
        for loop in loops:
            pv, op = prepare_window(loop.pv, loop.op)
            sigmoid = Sigmoid(op, pv)
            sigmoid.detect_stiction()
            path = frames_dir / f"{loop.name}.png"
            sigmoid.delta_pv_op_plot(loop.name, path)
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        if not frames:
            raise ValueError("No loops were selected for the GIF.")
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1500, loop=0)
    finally:
        for frame in frames:
            frame.close()
    print(f"Saved GIF: {filename.resolve()}")
    return filename


def main(argv=None):
    parser = argparse.ArgumentParser(description="Control-valve stiction detection and SVR experiments")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="Run a small complete experiment on bundled MAT data")
    for command in ("train", "detect", "inspect"):
        p = sub.add_parser(command)
        p.add_argument("--source", choices=["excel", "mat", "flow", "level"], default="excel" if command == "train" else "mat")
        p.add_argument("--file", help="Data path or filename under new_project")
        p.add_argument("--loops", nargs="+", type=int, help="MAT loop numbers")
        p.add_argument("--out", help="Output directory")
        if command == "train":
            p.add_argument("--window-size", type=int, default=60)
            p.add_argument("--sample-step", type=int, default=1)
            p.add_argument("--max-windows", type=int, default=0, help="Cap per loop per partition; 0 uses all windows")
            p.add_argument("--c", nargs="+", type=float, default=[1.0])
            p.add_argument("--epsilon", nargs="+", type=float, default=[0.1])
        if command == "detect":
            p.add_argument("--method", choices=["all", "method1", "method2", "method3", "method4", "linear", "sigmoid"], default="all")
    p = sub.add_parser("test", help="Reload a repaired model and evaluate its held-out test partition")
    p.add_argument("--source", choices=["excel", "mat", "flow", "level"], default="excel")
    p.add_argument("--model")
    p.add_argument("--out")
    p = sub.add_parser("combine", help="Concatenate selected MAT loops (a constructed sequence, not measured degradation)")
    p.add_argument("--loops", nargs="+", type=int, default=[13, 11, 1, 10, 24])
    p.add_argument("--out")
    p = sub.add_parser("combined-r")
    p.add_argument("--name", default="13_11_1_10_24")
    p.add_argument("--out")
    p = sub.add_parser("gif")
    p.add_argument("--loops", nargs="+", type=int, default=[2, 3, 13])
    p.add_argument("--out")
    p = sub.add_parser("plot", help="Plot a repaired prediction CSV without removing high-error rows")
    p.add_argument("--file")
    p.add_argument("--source", default="excel", choices=["excel", "mat", "flow", "level"])
    p.add_argument("--out")
    args = parser.parse_args(argv)
    command = args.command or "demo"
    try:
        if command == "demo":
            config = dataset_config("mat", loop_numbers=[2, 3, 13])
            train(config, max_windows=40)
            detect(load_loops("mat", loop_numbers=[2]), method="all")
        elif command in {"train", "detect", "inspect"}:
            config = dataset_config(args.source, args.file, args.loops)
            if command == "train":
                train(config, args.out, args.window_size, args.sample_step, args.max_windows, args.c, args.epsilon)
            else:
                loops = load_loops(**config)
                if command == "detect":
                    detect(loops, args.method, args.out)
                else:
                    for loop in loops:
                        print(f"{loop.name}: {len(loop.pv)} samples; nonfinite PV={np.sum(~np.isfinite(loop.pv))}; OP={np.sum(~np.isfinite(loop.op))}")
        elif command == "test":
            evaluate(args.model or latest_model(args.source), args.out)
        elif command == "combine":
            loops = load_loops("mat", loop_numbers=args.loops)
            output = output_dir(args.out, "combined")
            name = "_".join(map(str, args.loops))
            for key in ("pv", "op"):
                np.savetxt(output / f"{name}_{key}.csv", np.concatenate([getattr(loop, key) for loop in loops]), delimiter=",")
            json_write(output / "segments.json", [{"loop": loop.name, "samples": len(loop.pv)} for loop in loops])
            print(f"Saved constructed sequence: {output.resolve()}")
        elif command == "combined-r":
            from .data import Loop
            pv = np.loadtxt(data_path(f"new_project/csv/final/{args.name}_pv.csv"), delimiter=",")
            op = np.loadtxt(data_path(f"new_project/csv/final/{args.name}_op.csv"), delimiter=",")
            detect([Loop(args.name, pv, op)], "sigmoid", args.out)
        elif command == "gif":
            create_gif(load_loops("mat", loop_numbers=args.loops), args.out or ROOT / "outputs" / "chemical_loops.gif")
        elif command == "plot":
            path = Path(args.file) if args.file else latest_model(args.source).parent / "test_predictions.csv"
            rows = pd.read_csv(path).to_dict("records")
            filename = Path(args.out) if args.out else path.with_suffix(".png")
            prediction_plot(rows, filename)
            print(f"Saved plot: {filename.resolve()}")
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0
