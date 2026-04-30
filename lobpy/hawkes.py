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
    from scipy.optimize import minimize as _scipy_minimize

    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Internal: log-likelihood
# ---------------------------------------------------------------------------


def _negloglik(params, ts, dt):
    """
    Negative log-likelihood for univariate exponential Hawkes.

    Parameters
    ----------
    params : (mu, alpha, beta)
    ts     : sorted float64 array, shifted to start at 0, shape (n,)
    dt     : inter-arrival times, shape (n,); dt[0]=0, dt[i]=ts[i]-ts[i-1]
    """
    mu, alpha, beta = params
    if mu <= 0.0 or alpha <= 0.0 or beta <= 0.0 or alpha >= beta:
        return np.inf

    n = len(ts)
    T = ts[-1]

    # R[i] = Σⱼ<ᵢ exp(−β·(tᵢ−tⱼ)) = exp(−β·dt[i]) · (1 + R[i−1])
    R = np.empty(n, dtype=np.float64)
    R[0] = 0.0
    exp_bdt = np.exp(-beta * dt)
    for i in range(1, n):
        R[i] = exp_bdt[i] * (1.0 + R[i - 1])

    lam = mu + alpha * R
    if np.any(lam <= 0.0):
        return np.inf

    log_lik = np.sum(np.log(lam))
    integral = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - ts)))
    return -(log_lik - integral)


# ---------------------------------------------------------------------------
# Internal: single fit
# ---------------------------------------------------------------------------


def _fit(ts):
    """
    Fit Hawkes MLE on a sorted float64 array already shifted to start at 0.

    Uses multiple starting points to reduce local-minima sensitivity.

    Returns (mu, alpha, beta), or (nan, nan, nan) on failure.
    """
    if not _SCIPY:
        raise ImportError("scipy is required for Hawkes fitting: pip install scipy")

    n = len(ts)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    T = ts[-1]
    if T <= 0.0:
        return float("nan"), float("nan"), float("nan")

    dt = np.empty(n, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = np.diff(ts)

    rate = n / T  # mean event rate — used to set initial guesses

    best_nll = np.inf
    best_x = None

    # (mu_fraction, branching_ratio_init, beta_scale)
    starting_points = [
        (0.5, 0.3, 1.0),
        (0.2, 0.6, 2.0),
        (0.8, 0.1, 0.5),
    ]

    for mu_frac, br0, beta_scale in starting_points:
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
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_nll:
            best_nll = res.fun
            best_x = res.x

    if best_x is None:
        return float("nan"), float("nan"), float("nan")

    return float(best_x[0]), float(best_x[1]), float(best_x[2])


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
