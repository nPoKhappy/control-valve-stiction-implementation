"""Adapters for the project's original Svr classes."""
import numpy as np
import pandas as pd

from .data import ROOT
from .modeling import dataset_config, train
from .plots import finish, prediction_plot


class BaseSvr:
    def _ranges(self, c_start, c_end, epsilon_start, epsilon_end):
        self.c_values = np.arange(c_start, c_end, 0.2)
        self.epsilon_values = np.arange(epsilon_start, epsilon_end, 0.05)
        if not len(self.c_values) or not len(self.epsilon_values):
            raise ValueError("Parameter range end must be greater than start.")

    def train_valid(self, store=True, store_path=None, *, max_windows=0):
        # Results always have their own output directory. store_path is optional
        # and only used when the caller explicitly requests an additional copy.
        from pathlib import Path
        import shutil
        self.model_path, self.report = train(self.config, sample_step=self.sample_step,
                                            c_values=self.c_values, epsilon_values=self.epsilon_values,
                                            max_windows=max_windows)
        if store and store_path:
            target = Path(store_path)
            if not target.is_absolute():
                target = ROOT / target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.model_path, target)
        return self.report

    def draw_validation_mse(self, show=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        values = self.report["search"]
        ax.plot([v["validation_mse"] for v in values], marker="o")
        ax.set(xlabel="Parameter candidate", ylabel="Validation MSE")
        return finish(fig, self.model_path.parent / "validation_mse.png", show)

    def draw_r_val(self, file_name=None):
        rows = []
        for part in ("train", "valid", "test"):
            rows.extend(pd.read_csv(self.model_path.parent / f"{part}_predictions.csv").to_dict("records"))
        return prediction_plot(rows, self.model_path.parent / "all_predictions.png")


class MatSvr(BaseSvr):
    def __init__(self, loop_list, c_start=1, c_end=1.05, epsilon_start=0.1, epsilon_end=0.11):
        self.config = dataset_config("mat", loop_numbers=list(loop_list))
        self.sample_step = 3
        self._ranges(c_start, c_end, epsilon_start, epsilon_end)


class ExcelSvr(BaseSvr):
    def __init__(self, file_name, data_col, c_start=1, c_end=1.05, epsilon_start=0.1, epsilon_end=0.11):
        self.config = dataset_config("excel", file_name, pairs=list(data_col))
        self.sample_step = 1
        self._ranges(c_start, c_end, epsilon_start, epsilon_end)
