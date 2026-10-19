"""Read the bundled data without depending on the current working directory."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = "CVOP_20241001~1209.xlsx"
DEFAULT_LOOPS = (1, 2, 3, 6, 10, 11, 12, 13, 23, 24, 29, 32)
EXCEL_PAIRS = [(f"FT_{n}", f"FV_{n}.OUT") for n in (107, 117, 174, 181, 2004, 306, 308)]


@dataclass
class Loop:
    name: str
    pv: np.ndarray
    op: np.ndarray

    def __post_init__(self):
        self.pv = np.asarray(self.pv, dtype=float).reshape(-1)
        self.op = np.asarray(self.op, dtype=float).reshape(-1)
        if len(self.pv) != len(self.op) or len(self.pv) < 2:
            raise ValueError(f"{self.name}: PV and OP need equal lengths and at least two samples.")


def data_path(value):
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [ROOT / path, ROOT / "new_project" / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Data file not found: {value}")


def load_loops(source="excel", file=None, loop_numbers=None, pairs=None):
    if source == "mat":
        path = data_path(file or "isdb10.mat")
        chemicals = loadmat(path, simplify_cells=True)["cdata"]["chemicals"]
        return [Loop(f"chemical_{n}", chemicals[f"loop{n}"]["PV"], chemicals[f"loop{n}"]["OP"])
                for n in (DEFAULT_LOOPS if loop_numbers is None else loop_numbers)]
    if source == "excel":
        path = data_path(file or DEFAULT_EXCEL)
        pairs = EXCEL_PAIRS if pairs is None else list(pairs)
        columns = list(dict.fromkeys(c for pair in pairs for c in pair))
        df = pd.read_excel(path, header=1, usecols=columns, engine="openpyxl")
    elif source in {"flow", "level"}:
        filename, pair = ("FIC-211_20241121~27.csv", ("FT_211", "FV_211")) if source == "flow" else ("LIC-106_20241121~27.csv", ("FT_115", "LV_106"))
        path = data_path(file or filename)
        pairs = [pair] if pairs is None else list(pairs)
        df = pd.read_csv(path)
        # Keep missing samples in place: never concatenate across a gap.
        timestamps = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="raise")
        df = df.assign(_timestamp=timestamps).sort_values("_timestamp", kind="stable")
    else:
        raise ValueError(f"Unknown source: {source}")
    return [Loop(pv, pd.to_numeric(df[pv], errors="coerce").to_numpy(),
                 pd.to_numeric(df[op], errors="coerce").to_numpy()) for pv, op in pairs]


def split_indices(n, valid_size=0.1, test_size=0.2):
    if not (0 < valid_size < 1 and 0 < test_size < 1 and valid_size + test_size < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    train_end = int(round((1 - valid_size - test_size) * n))
    valid_end = int(round((1 - test_size) * n))
    return {"train": (0, train_end), "valid": (train_end, valid_end), "test": (valid_end, n)}
