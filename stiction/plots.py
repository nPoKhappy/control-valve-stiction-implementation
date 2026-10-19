"""Save figures by default; interactive display is always opt-in."""
from pathlib import Path
import os

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, filename=None, show=False):
    if filename:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        if "agg" == matplotlib.get_backend().lower():
            print("Non-interactive backend: use the saved PNG to view the result.")
        else:
            plt.show()
    plt.close(fig)
    return filename


def phase_plot(sigmoid, name="", filename=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sigmoid.pv, sigmoid.op, s=7, alpha=0.6)
    if len(sigmoid.pv):
        order = np.argsort(sigmoid.pv)
        ax.plot(sigmoid.pv[order], sigmoid._sigmoid(sigmoid.pv[order], *sigmoid.params), color="black")
    ax.set(xlabel="Standardized delta PV", ylabel="Normalized OP", title=f"Sigmoid: {name}")
    return finish(fig, filename, show)


def signal_plot(pv, signal, name="", filename=None, show=False):
    pv, signal = np.asarray(pv).reshape(-1), np.asarray(signal).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pv, lw=0.7, label="PV")
    ax.set(xlabel="Sample", ylabel="PV", title=name)
    other = ax.twinx()
    other.step(np.arange(len(signal)), signal, where="post", color="tab:red", alpha=0.5)
    other.set(ylabel="Stiction flag", ylim=(-0.05, 1.1))
    return finish(fig, filename, show)


def plot_pv_sp_op(pv, sp=None, co=None, loop_name="", *, op=None, filename=None, show=False):
    co = op if co is None else co
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if sp is not None:
        axes[0].plot(np.asarray(sp).reshape(-1), label="SP")
    axes[0].plot(np.asarray(pv).reshape(-1), label="PV")
    axes[1].plot(np.asarray(co).reshape(-1), label="OP")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title(str(loop_name))
    axes[1].set_xlabel("Sample")
    return finish(fig, filename, show)


def band_plot(bands, value, filename=None, show=False):
    fig, ax = plt.subplots()
    if len(bands):
        ax.hist(bands, bins="auto", edgecolor="black")
    ax.axvline(value, color="red", label=f"95th percentile: {value:.3g}")
    ax.set(xlabel="OP band", ylabel="Count")
    ax.legend()
    return finish(fig, filename, show)


def prediction_plot(rows, filename, title="SVR versus Sigmoid reference"):
    fig, ax = plt.subplots(figsize=(12, 4))
    for key, label in (("reference_r", "Sigmoid reference"), ("predicted_r", "SVR prediction")):
        ax.plot([r[key] for r in rows], label=label, lw=0.8)
    ax.set(xlabel="Window (loop and time order; all retained windows)", ylabel="R", title=title)
    ax.legend()
    return finish(fig, filename)
