"""
Univariate exponential Hawkes process estimation.

Model: λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ))   for all tᵢ < t

Parameters
----------
μ (mu)    : baseline intensity
α (alpha) : excitement magnitude (jump per event)
β (beta)  : decay rate; half-life = ln(2) / β
α / β     : branching ratio — expected offspring per event; must be < 1 for stationarity

Estimation via MLE using the recursive intensity formula (Ozaki 1979), which
reduces the O(N²) naive computation to O(N) per likelihood evaluation.

References: Hawkes (1971), Ozaki (1979).
"""

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore[import-untyped]

    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from lobpy._cext import ffi as _ffi
    from lobpy._cext import lib as _lib

    _HAWKES_C = hasattr(_lib, "hawkes_mle_fit")
except Exception:
    _HAWKES_C = False


# ---------------------------------------------------------------------------
# Internal: log-likelihood (Python reference implementation)
# ---------------------------------------------------------------------------


def _negloglik(params, ts, dt):
    mu, alpha, beta = params
    if mu <= 0.0 or alpha <= 0.0 or beta <= 0.0 or alpha >= beta:
        return np.inf

    n = len(ts)
    t_end = ts[-1]

    r = np.empty(n, dtype=np.float64)
    r[0] = 0.0
    exp_bdt = np.exp(-beta * dt)
    for i in range(1, n):
        r[i] = exp_bdt[i] * (1.0 + r[i - 1])

    lam = mu + alpha * r
    if np.any(lam <= 0.0):
        return np.inf

    log_lik = np.sum(np.log(lam))
    integral = mu * t_end + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (t_end - ts)))
    return -(log_lik - integral)


# ---------------------------------------------------------------------------
# Internal: Python-only fit (kept for convergence tests and fallback)
# ---------------------------------------------------------------------------


def _fit_py(ts):
    """
    Fit Hawkes MLE using scipy (Python reference implementation).

    ts: sorted float64 array already shifted to start at 0.
    Returns (mu, alpha, beta), or (nan, nan, nan) on failure.
    """
    if not _SCIPY:
        raise ImportError("scipy is required for Hawkes fitting: pip install scipy")

    n = len(ts)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    t_end = ts[-1]
    if t_end <= 0.0:
        return float("nan"), float("nan"), float("nan")

    dt = np.empty(n, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = np.diff(ts)

    rate = n / t_end
    best_nll = np.inf
    best_x = None

    for mu_frac, br0, beta_scale in [(0.5, 0.3, 1.0), (0.2, 0.6, 2.0), (0.8, 0.1, 0.5)]:
        beta0 = rate * beta_scale
        mu0 = rate * mu_frac
        alpha0 = br0 * beta0
        try:
            res = _scipy_minimize(
                _negloglik,
                [mu0, alpha0, beta0],
                args=(ts, dt),
                method="L-BFGS-B",
                bounds=[(1e-10, None), (1e-10, None), (1e-10, None)],
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception:  # nosec B112
            continue
        if np.isfinite(res.fun) and res.fun < best_nll:
            best_nll = res.fun
            best_x = res.x

    if best_x is None:
        return float("nan"), float("nan"), float("nan")

    return float(best_x[0]), float(best_x[1]), float(best_x[2])


# ---------------------------------------------------------------------------
# Internal: C-accelerated fit
# ---------------------------------------------------------------------------


def _fit_c(ts):
    """
    Fit Hawkes MLE using the C BFGS implementation.

    ts: sorted float64 array already shifted to start at 0.
    Returns (mu, alpha, beta), or (nan, nan, nan) on failure.
    """
    n = len(ts)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    ts_c = np.ascontiguousarray(ts, dtype=np.float64)
    dt_c = np.empty(n, dtype=np.float64)
    dt_c[0] = 0.0
    dt_c[1:] = np.diff(ts_c)

    ts_ptr = _ffi.cast("double *", ts_c.ctypes.data)
    dt_ptr = _ffi.cast("double *", dt_c.ctypes.data)
    mu_p = _ffi.new("double *")
    alpha_p = _ffi.new("double *")
    beta_p = _ffi.new("double *")

    ret = _lib.hawkes_mle_fit(ts_ptr, dt_ptr, n, mu_p, alpha_p, beta_p)
    if ret != 0:
        return float("nan"), float("nan"), float("nan")
    return float(mu_p[0]), float(alpha_p[0]), float(beta_p[0])


# ---------------------------------------------------------------------------
# Internal: dispatcher
# ---------------------------------------------------------------------------


def _fit(ts):
    """Fit Hawkes MLE: C implementation when available, Python fallback otherwise."""
    if _HAWKES_C:
        return _fit_c(ts)
    return _fit_py(ts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_hawkes(timestamps):
    """
    Fit a univariate exponential Hawkes process via MLE.

    λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ))

    Timestamps are shifted to start at 0 internally for numerical stability.
    Returned parameters are in the same time units as the input.

    Args:
        timestamps: array-like of sorted event timestamps.

    Returns:
        dict with keys:
            'mu'              – baseline intensity (events / time_unit)
            'alpha'           – excitement magnitude (events / time_unit)
            'beta'            – decay rate (1 / time_unit); half-life = ln(2) / beta
            'branching_ratio' – alpha / beta; < 1 required for stationarity

        All values are nan if fewer than 3 events are provided or fitting fails.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) >= 1:
        ts = ts - ts[0]
    mu, alpha, beta = _fit(ts)
    nan = float("nan")
    br = alpha / beta if (np.isfinite(alpha) and np.isfinite(beta) and beta > 0.0) else nan
    return {"mu": mu, "alpha": alpha, "beta": beta, "branching_ratio": br}
