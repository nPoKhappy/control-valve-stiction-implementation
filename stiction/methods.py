"""Numerically guarded versions of the original detection algorithms.

These implementations preserve the project's method families; this repair is
not a new validation against the three papers or against physical valve labels.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import f, norm, t

TINY = 1e-12


def paired(pv, op):
    pv, op = np.asarray(pv, dtype=float).reshape(-1), np.asarray(op, dtype=float).reshape(-1)
    if len(pv) != len(op) or not len(pv):
        raise ValueError("PV and OP must be nonempty arrays of equal length.")
    return pv, op


def prepare_window(pv, op):
    """Use only this window: no fitted statistics from future/other partitions."""
    pv, op = paired(pv, op)
    if len(pv) < 3 or not (np.isfinite(pv).all() and np.isfinite(op).all()):
        raise ValueError("Window is too short or contains missing/nonfinite values.")
    delta = np.diff(pv, prepend=pv[0])
    spread, scale = np.std(delta), np.ptp(op)
    if spread <= TINY or scale <= TINY:
        raise ValueError("Window has constant PV differences or constant OP.")
    return (delta - delta.mean()) / spread, (op - op.min()) / scale


def step_function(x):
    x = np.asarray(x, dtype=float)
    return np.clip(0.15 * x + np.where(x < 0, 0.3, 0.7), 0, 1)


class Sigmoid:
    """Fit OP = sigmoid(a * x + b); callers explicitly prepare x and OP."""

    def __init__(self, op, pv):
        self.pv, self.op = paired(pv, op)
        self.pv_scale, self.op_scale = self.pv, self.op  # Historical read-only aliases.
        self.params = np.array([0.0, 0.0])
        self.r_value = 0.0

    @staticmethod
    def _sigmoid(x, a, b):
        return expit(a * np.asarray(x) + b)

    @staticmethod
    def _calculate_correlation_coefficient(y, y_fit):
        if np.std(y) <= TINY or np.std(y_fit) <= TINY:
            return 0.0
        return float(np.clip(np.corrcoef(y, y_fit)[0, 1], 0, 1))

    def detect_stiction(self, r_threshold=0.5):
        if not 0 <= r_threshold <= 1:
            raise ValueError("R threshold must lie between 0 and 1.")
        if not (np.isfinite(self.pv).all() and np.isfinite(self.op).all()):
            raise ValueError("Sigmoid inputs must be finite.")
        if len(self.pv) < 3 or np.std(self.pv) <= TINY or np.std(self.op) <= TINY:
            self.r_value = 0.0
            return False, 0.0

        def residual(params):
            return self._sigmoid(self.pv, *params) - self.op

        def jacobian(params):
            fitted = self._sigmoid(self.pv, *params)
            derivative = fitted * (1 - fitted)
            return np.column_stack((derivative * self.pv, derivative))

        result = least_squares(residual, [1.0, 0.0], jac=jacobian,
                               bounds=(-100, 100), max_nfev=1000)
        if not result.success:
            raise RuntimeError(f"Sigmoid fit did not converge: {result.message}")
        self.params = result.x
        self.r_value = round(self._calculate_correlation_coefficient(
            self.op, self._sigmoid(self.pv, *self.params)), 2)
        return self.r_value >= r_threshold, self.r_value

    def delta_pv_op_plot(self, name="", filename=None, show=False):
        from .plots import phase_plot
        return phase_plot(self, name, filename, show)


def _ratio(numerator, denominator, both_zero=0.0):
    if denominator <= TINY:
        return both_zero if numerator <= TINY else float("inf")
    return numerator / denominator


def _statistic(x, y, method, alpha):
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if method == 1:
        return _ratio(vx, vy, both_zero=1.0), float(f.ppf(1 - alpha, len(x)-1, len(y)-1))
    d = float(x.mean() - y.mean())
    if method == 2:
        pooled = ((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2)
        statistic = _ratio(abs(d), np.sqrt(pooled * (1/len(x)+1/len(y))))
        return statistic, float(t.ppf(1 - alpha/2, len(x)+len(y)-2))
    # Scalar form of the original modified T2 formula (equal-width windows).
    pooled = (vx + vy) / 2
    statistic = _ratio(0.5 * len(x) * d*d, pooled)
    weight = _ratio(vx*vx + vy*vy, (vx+vy)**2, both_zero=0.5)
    df = (len(x)-1) / weight
    return statistic, float(f.ppf(1 - alpha, 1, df))


class StatisticalMethod:
    method = 1

    def __init__(self, pv, op, ws=None, alpha=0.05):
        self.pv, self.op = paired(pv, op)
        self.ws = (3 if self.method == 4 else 8) if ws is None else int(ws)
        if self.ws < 3 or not 0 < alpha < 1:
            raise ValueError("Window size must be >=3; alpha must be between zero and one.")
        self.alpha, self.s, self.omega = alpha, None, 0.0

    def perform(self, show=False):
        count = len(self.pv) // self.ws
        self.s = np.zeros(count, dtype=int)
        self.valid_windows = np.zeros(count, dtype=bool)
        self.statistics = []
        p = self.pv[:count*self.ws].reshape(count, self.ws)
        o = self.op[:count*self.ws].reshape(count, self.ws)
        if count:
            self.valid_windows = np.isfinite(p).all(axis=1) & np.isfinite(o).all(axis=1)
        if self.method == 4:
            m = self.ws
            mean = m*(m-1)/4
            spread = np.sqrt((2*m**3 + 3*m*m - 5*m)/72)
            critical = norm.ppf(1 - self.alpha/2) * spread
            lower, upper = mean-critical+0.5, mean+critical-0.5
            for i in range(count):
                if not self.valid_windows[i]:
                    continue
                reverse = lambda a: sum(np.count_nonzero(a[j] > a[j+1:]) for j in range(m-1))
                a, b = reverse(p[i]), reverse(o[i])
                self.s[i] = int(lower < a < upper and not lower < b < upper)
                self.statistics.append((i, a, b))
        else:
            for i in range(count-1):
                if not self.valid_windows[i:i+2].all():
                    continue
                a, ac = _statistic(p[i], p[i+1], self.method, self.alpha)
                b, bc = _statistic(o[i], o[i+1], self.method, self.alpha)
                self.statistics.append((i, a, b))
                if a < ac and b > bc:
                    self.s[i:i+2] = 1  # Union: later pairs cannot erase a detection.
        self.omega = float(self.s.sum() / self.valid_windows.sum()) if self.valid_windows.any() else 0.0
        if show:
            from .plots import signal_plot
            signal_plot(self.pv, np.repeat(self.s, self.ws), f"Method {self.method}", show=True)
        return self.s

    def store_csv(self, name):
        from .data import ROOT
        if self.s is None:
            raise RuntimeError("Call perform() before store_csv().")
        path = ROOT / "outputs" / "statistical" / f"method{self.method}_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.s, delimiter=",", fmt="%d")
        return path


class Method1(StatisticalMethod):
    method = 1


class Method2(StatisticalMethod):
    method = 2


class Method3(StatisticalMethod):
    method = 3


class Method4(StatisticalMethod):
    method = 4


def stiction_bands(op, signal):
    op, signal = np.asarray(op, dtype=float).reshape(-1), np.asarray(signal).reshape(-1)
    count = min(len(op), len(signal))
    active = np.r_[False, signal[:count] > 0, False]
    starts = np.flatnonzero(np.diff(active.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(active.astype(int)) == -1)
    bands = [abs(op[a]-op[b-1]) for a, b in zip(starts, stops) if np.isfinite(op[a:b]).all()]
    return np.asarray(bands, dtype=float)


class Quantification:
    def __init__(self, S_expand, op):
        self.s_expand, self.op = S_expand, op

    def perform(self, show=False):
        self.bands = stiction_bands(self.op, self.s_expand)
        self.value = float(np.percentile(self.bands, 95)) if self.bands.size else 0.0
        if show:
            from .plots import band_plot
            band_plot(self.bands, self.value, show=True)
        return self.value

    perfrom = perform  # Original misspelling remains usable.


class LinearRegression:
    def __init__(self, pv, co=None, t=None, draw_log_plot=False, w=6, n=15, *, op=None):
        if co is None:
            co = op
        self.pv, self.op = paired(pv, co)
        self.t = np.arange(len(self.pv), dtype=float) if t is None else np.asarray(t, dtype=float).reshape(-1)
        self.w, self.n, self.draw = int(w), float(n), draw_log_plot
        if self.w < 2 or self.n <= 0 or len(self.t) != len(self.pv):
            raise ValueError("Invalid linear-regression window, threshold, or time array.")

    @staticmethod
    def _fitlinear(tdata, ydata):
        tdata = np.asarray(tdata, dtype=float)
        centered = tdata - tdata.mean()
        slope = _ratio(float(centered @ (ydata - np.mean(ydata))), float(centered @ centered))
        return slope, float(np.mean(ydata) - slope*tdata.mean())

    def detect_stiction(self):
        count = len(self.pv) // self.w
        self.s, self.ratios = np.zeros(count, dtype=int), np.zeros(count)
        self.valid_windows = np.zeros(count, dtype=bool)
        for i in range(count):
            sl = slice(i*self.w, (i+1)*self.w)
            p, o, times = self.pv[sl], self.op[sl], self.t[sl]
            if not (np.isfinite(p).all() and np.isfinite(o).all() and np.isfinite(times).all()) or np.ptp(times) <= TINY:
                continue
            self.valid_windows[i] = True
            mp, _ = self._fitlinear(times, p)
            mo, _ = self._fitlinear(times, o)
            self.ratios[i] = _ratio(abs(mo), abs(mp))
            self.s[i] = int(self.ratios[i] >= self.n)
        self.stiction_index = float(self.s.sum() / self.valid_windows.sum() * 100) if self.valid_windows.any() else 0.0
        if self.draw:
            from .plots import signal_plot
            signal_plot(self.pv, np.repeat(self.s, self.w), "Linear regression", show=True)
        return self.s

    def store_csv(self, name):
        from .data import ROOT
        path = ROOT / "outputs" / "linear" / f"stiction_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.s, delimiter=",", fmt="%d")
        return path


Linear_Rregression = LinearRegression


class Quantificatioin:
    def __init__(self, co=None, s=None, w=6, *, op=None):
        self.op, self.s, self.w = co if co is not None else op, s, int(w)

    def perform(self):
        self.bands = stiction_bands(self.op, np.repeat(self.s, self.w))
        self.value = float(self.bands.max()) if self.bands.size else 0.0
        return self.value
