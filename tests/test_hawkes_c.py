"""
Convergence tests: C Hawkes MLE vs Python (scipy) reference.

Checks that the C BFGS implementation reaches the same optimum as scipy's
L-BFGS-B on simulated and real-like inputs.  Parameter agreement tolerance
is loose (1%) because the two optimizers may converge to slightly different
local optima; the NLL check is tighter (0.01%) and is the real quality gate.
"""

import numpy as np
import pytest

from lobpy.hawkes import _HAWKES_C, _fit_c, _fit_py, _negloglik


def _make_dt(ts):
    n = len(ts)
    dt = np.empty(n)
    dt[0] = 0.0
    dt[1:] = np.diff(ts)
    return dt


def _nll(mu, alpha, beta, ts):
    dt = _make_dt(ts)
    return _negloglik([mu, alpha, beta], ts, dt)


def simulate_hawkes(mu, alpha, beta, T, seed=0):
    """Ogata thinning simulation of a univariate exponential Hawkes process."""
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    lam_bar = mu  # upper bound on intensity

    while t < T:
        lam_bar = max(lam_bar, mu)
        dt = rng.exponential(1.0 / lam_bar)
        t += dt
        if t > T:
            break
        # actual intensity at t
        lam_t = mu + alpha * sum(np.exp(-beta * (t - s)) for s in events)
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)
        lam_bar = lam_t + alpha  # intensity just after event

    ts = np.array(events, dtype=np.float64)
    return ts - ts[0] if len(ts) > 0 else ts


# ---------------------------------------------------------------------------
# Fixtures: simulated datasets with known ground truth
# ---------------------------------------------------------------------------

CASES = [
    # (mu, alpha, beta, T, min_events, label)
    (0.5,  0.3,  1.0,  200.0,  50,  "low_branching"),
    (0.2,  0.5,  0.8,  500.0, 100,  "moderate_branching"),
    (1.0,  0.1,  0.5, 1000.0, 200,  "high_rate_low_branching"),
    (0.1,  0.6,  1.0, 2000.0, 200,  "high_branching"),
    (2.0,  0.8,  3.0,  100.0,  50,  "fast_decay"),
]


@pytest.mark.skipif(not _HAWKES_C, reason="C extension not available")
@pytest.mark.parametrize("mu,alpha,beta,T,min_n,label", CASES)
def test_nll_matches_python(mu, alpha, beta, T, min_n, label):
    """C and Python fits reach NLL within 0.01% of each other."""
    ts = simulate_hawkes(mu, alpha, beta, T, seed=42)
    if len(ts) < min_n:
        pytest.skip(f"simulation too short ({len(ts)} events)")

    mu_c, alpha_c, beta_c = _fit_c(ts)
    mu_py, alpha_py, beta_py = _fit_py(ts)

    # Both must succeed
    assert np.isfinite(mu_c) and np.isfinite(alpha_c) and np.isfinite(beta_c), \
        f"C fit failed on {label}"
    assert np.isfinite(mu_py) and np.isfinite(alpha_py) and np.isfinite(beta_py), \
        f"Python fit failed on {label}"

    nll_c  = _nll(mu_c,  alpha_c,  beta_c,  ts)
    nll_py = _nll(mu_py, alpha_py, beta_py, ts)

    # NLL relative difference < 0.01%
    rel = abs(nll_c - nll_py) / (abs(nll_py) + 1e-12)
    assert rel < 1e-4, (
        f"{label}: NLL divergence {rel:.2e} "
        f"(C={nll_c:.6f}, py={nll_py:.6f})"
    )


@pytest.mark.skipif(not _HAWKES_C, reason="C extension not available")
@pytest.mark.parametrize("mu,alpha,beta,T,min_n,label", CASES)
def test_params_close_to_python(mu, alpha, beta, T, min_n, label):
    """C parameters agree with Python within 1% (relative)."""
    ts = simulate_hawkes(mu, alpha, beta, T, seed=42)
    if len(ts) < min_n:
        pytest.skip(f"simulation too short ({len(ts)} events)")

    mu_c, alpha_c, beta_c = _fit_c(ts)
    mu_py, alpha_py, beta_py = _fit_py(ts)

    if not (np.isfinite(mu_c) and np.isfinite(mu_py)):
        pytest.skip("one or both fits returned nan")

    for name, vc, vpy in [("mu", mu_c, mu_py), ("alpha", alpha_c, alpha_py), ("beta", beta_c, beta_py)]:
        rel = abs(vc - vpy) / (abs(vpy) + 1e-12)
        assert rel < 0.01, f"{label}/{name}: C={vc:.6f} py={vpy:.6f} rel={rel:.2e}"


@pytest.mark.skipif(not _HAWKES_C, reason="C extension not available")
def test_branching_ratio_below_one():
    """C fit must always return branching ratio < 1 (stationarity)."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        mu    = rng.uniform(0.1, 2.0)
        beta  = rng.uniform(0.5, 5.0)
        alpha = rng.uniform(0.05, 0.95) * beta
        ts = simulate_hawkes(mu, alpha, beta, T=500.0, seed=int(rng.integers(1000)))
        if len(ts) < 10:
            continue
        mu_c, alpha_c, beta_c = _fit_c(ts)
        if np.isfinite(mu_c):
            assert alpha_c < beta_c, (
                f"branching ratio >= 1: alpha={alpha_c:.4f} beta={beta_c:.4f}"
            )


@pytest.mark.skipif(not _HAWKES_C, reason="C extension not available")
def test_edge_fewer_than_3_events():
    """C fit returns nan for < 3 events."""
    for n in [0, 1, 2]:
        ts = np.linspace(0.0, 1.0, n)
        mu, alpha, beta = _fit_c(ts)
        assert np.isnan(mu) and np.isnan(alpha) and np.isnan(beta)


@pytest.mark.skipif(not _HAWKES_C, reason="C extension not available")
def test_c_faster_than_python(benchmark_n=2000):
    """Smoke-test: C fit completes on a large window without hanging."""
    import time
    ts = simulate_hawkes(0.5, 0.3, 1.0, T=5000.0, seed=99)
    ts = ts[:benchmark_n] if len(ts) > benchmark_n else ts
    if len(ts) < 10:
        pytest.skip("not enough events")

    t0 = time.perf_counter()
    mu, alpha, beta = _fit_c(ts)
    elapsed_c = time.perf_counter() - t0

    assert np.isfinite(mu), "C fit returned nan on large window"
    assert elapsed_c < 5.0, f"C fit too slow: {elapsed_c:.2f}s on {len(ts)} events"
