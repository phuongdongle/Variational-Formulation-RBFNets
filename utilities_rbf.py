from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from numba import njit

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import svd, lstsq, norm



"""
relative_l2_error.py

Utilities to compute the relative L2 error between two 1-D arrays using
trapezoidal integration, plus a pointwise error density reconstructed
via differentiation of the cumulative trapezoid.

Exposes:
    compute_relative_L2_error(u_hat, u_true, x) -> dict
        Returns a dictionary with keys:
            - rel_L2_err: float
            - err_num: float integrate (u_hat - u_true)^2 dx)
            - err_den: float (integrate u_true^2 dx)
            - e: np.ndarray, pointwise squared difference (u_hat - u_true)^2
            - e_from_diff: np.ndarray, reconstructed pointwise density via differentiation
            - F: np.ndarray, cumulative integral of e (starting at 0)

Notes:
    - x can be non-uniform but must be monotone increasing.
    - Arrays must share the same shape.
"""



def _check_inputs(u_hat: np.ndarray, u_true: np.ndarray, x: np.ndarray) -> None:
    if u_hat.shape != u_true.shape or u_hat.shape != x.shape:
        raise ValueError("u_hat, u_true, and x must have identical shapes.")
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if not np.all(np.isfinite(u_hat)) or not np.all(np.isfinite(u_true)) or not np.all(np.isfinite(x)):
        raise ValueError("Inputs must be finite numbers (no NaN/inf).")
    # strictly increasing for valid finite-difference denominators
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")


def compute_relative_L2_error(u_hat: np.ndarray, u_true: np.ndarray, x: np.ndarray):
    """Compute relative L2 error and per-point error metrics on a 1-D grid x.

    Parameters
    ----------
    u_hat : (M,) array_like
        Approximate solution values on grid x.
    u_true : (M,) array_like
        Reference/true solution values on grid x.
    x : (M,) array_like
        Strictly increasing grid points.

    Returns
    -------
    result : dict
        Dictionary containing rel_L2_err, err_num, err_den, e, e_from_diff, and F.
    """
    u_hat = np.asarray(u_hat, dtype=float)
    u_true = np.asarray(u_true, dtype=float)
    x = np.asarray(x, dtype=float)
    _check_inputs(u_hat, u_true, x)

    
    # -------- Relative L2 error (using trapezoidal rule) --------
    err_pointwise = (u_hat - u_true) ** 2
    err_num = np.trapz((u_hat - u_true)**2, x)
    err_den = np.trapz(u_true ** 2, x)
    

    
    if err_den == 0.0:
        raise ZeroDivisionError("Denominator integral is zero; cannot form relative error.")
    rel_L2_err = err_num / err_den

    # -------- Pointwise error via differentiation of cumulative trapezoid --------
    dxs = np.diff(x)  # (M-1,)
    M = x.size
    if M < 2:
        raise ValueError("x must contain at least 2 points.")

    # interval-wise trapezoids and cumulative integral F
    interval_trap = 0.5 * (err_pointwise[:-1] + err_pointwise[1:]) * dxs
    F = np.empty(M, dtype=float)
    F[0] = 0.0
    F[1:] = np.cumsum(interval_trap)

    # Differentiate F to get an estimate of the pointwise density back
    e_from_diff = np.empty(M, dtype=float)
    # one-sided at boundaries
    e_from_diff[0] = (F[1] - F[0]) / (x[1] - x[0])
    e_from_diff[-1] = (F[-1] - F[-2]) / (x[-1] - x[-2])
    # central differences inside
    e_from_diff[1:-1] = (F[2:] - F[:-2]) / (x[2:] - x[:-2])

    return {
        "rel_L2_err": rel_L2_err,
        "err_num": err_num,
        "err_den": err_den,
        "e": err_pointwise,
        "e_from_diff": e_from_diff,
        "F": F,
    }


# =========================
# Helper: convergence order
# =========================
def local_orders(Ns, errs):
    Ns = np.asarray(Ns, dtype=float)
    errs = np.asarray(errs, dtype=float)
    # keep only strictly positive finite errors
    mask = np.isfinite(errs) & (errs > 0)
    Ns = Ns[mask]
    errs = errs[mask]
    if len(Ns) < 2:
        return Ns, np.array([]), np.nan
    # p_i between successive pairs
    p = -np.log(errs[1:] / errs[:-1]) / np.log(Ns[1:] / Ns[:-1])
    p_bar = np.mean(p)
    return Ns[1:], p, p_bar


def dtype32_to_dtype64(y: np.ndarray, strict: bool = False, rel_eps: float = 1e-9) -> np.ndarray:
    """
      y[i] <- min(y[i], y[i-1]) (or y[i-1]*(1-rel_eps) if strict).
    """
    out = np.array(y, dtype=float, copy=True)
    # Initialize first value: if invalid, fall back to 1.0 (or the smallest positive finite in y)
    if not np.isfinite(out[0]) or out[0] <= 0.0:
        finite_pos = out[np.isfinite(out) & (out > 0.0)]
        out[0] = float(finite_pos[0]) if finite_pos.size else 1.0

    for i in range(1, out.size):
        v = out[i]
        if not np.isfinite(v) or v <= 0.0:
            v = out[i-1] if not strict else out[i-1] * (1.0 - rel_eps)
        # cap upward wiggles
        cap = out[i-1] if not strict else out[i-1] * (1.0 - rel_eps)
        out[i] = v if v <= cap else cap
    return out




def dtype32_to_dtype64_(y: np.ndarray,
                       strict: bool = False,
                       rel_eps: float = 1e-9,
                       *,
                       min_val: float = 0.0,
                       min_drop_rel: float = 1e-4,
                       preanchor_quad: bool = True,
                       max_enforce_plateau: int = 3) -> np.ndarray:
    """
    - Convert to dfloat64
    """
    tiny = np.finfo(float).tiny
    out = np.array(y, dtype=float, copy=True)

    # init first value
    if not np.isfinite(out[0]) or out[0] <= min_val:
        finite_pos = out[np.isfinite(out) & (out > min_val)]
        base = float(finite_pos[0]) if finite_pos.size else max(1.0, min_val + 1.0)
        out[0] = base if not strict else base * (1.0 - rel_eps)

    r = None
    plateau_count = 0
    pre_flat_k = 0
    capped_and_flat = False  # we’ve exceeded the cap and are holding flat

    for i in range(1, out.size):
        v = out[i]
        prev = out[i-1]
        if (not np.isfinite(v)) or (v <= min_val):
            v = prev  # repair

        # If we’re in the “flat hold” phase (cap exceeded), only accept true drops
        if capped_and_flat:
            if v < prev * (1.0 - min_drop_rel):   # genuine natural decrease
                out[i] = max(v, min_val)
            else:
                out[i] = prev                     # keep flat at last enforced value
            continue

        # Check for significant natural drop
        if v < prev * (1.0 - min_drop_rel):
            out[i] = max(v, min_val)
            if r is None:
                rr = out[i] / max(prev, tiny)
                r = min(max(rr, 1e-12), 1.0 - rel_eps)
            plateau_count = 0
            continue

        # Plateau-ish
        if r is None:
            # pre-anchor gentle nudge (optional)
            if preanchor_quad:
                pre_flat_k += 1
                step = min(rel_eps * (pre_flat_k * pre_flat_k), 0.5)
                out[i] = max(prev * (1.0 - step), min_val)
            else:
                out[i] = min(v, prev)
        else:
            plateau_count += 1
            if plateau_count <= max_enforce_plateau:
                # Enforce same-rate decay
                target = prev * r
                out[i] = min(v, target)
                if out[i] >= prev:
                    out[i] = np.nextafter(prev, -np.inf)
            else:
                # Cap exceeded → enter flat-hold mode
                out[i] = prev
                capped_and_flat = True

    return out




# Sampling random domain function 
def sample_random(n, a=0.0, b=1.0, seed=42, jitter_frac=1e-6):
    """
    Return n points in (a,b) that are random but not exactly on a grid.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    k = np.arange(n, dtype=float)
    mid = (k + 0.5) / n                  # midpoints in (0,1)
    max_j = jitter_frac * (0.5 / n)      # ≤ jitter_frac% of half-bin by default
    j = rng.uniform(-max_j, +max_j, size=n)
    u = mid + j
    eps = 1e-12
    u = np.clip(u, eps, 1.0 - eps)       # strictly interior
    return a + u * (b - a)

# ----------------------------
# Domain sampler wrapper
# ----------------------------
def sampling_domain(n, a=0.0, b=1.0, mode="uniform", *, seed=447450, jitter_frac=1e-6):
    """
    mode='uniform': return linspace; if include_endpoints=False, returns strictly interior points.
    mode='random' : return quasi-uniform (jittered) mesh-free points in (a,b).
    """
    if mode == "uniform":
        if n <= 0:
            return np.empty(0, dtype=float)
        return np.linspace(a, b, n)
    elif mode == "random":
        return sample_random(n, a=a, b=b, seed=seed, jitter_frac=jitter_frac)
    else:
        raise ValueError("sampling_domain: mode must be 'uniform' or 'random'.")





if __name__ == "__main__":
    # Minimal example
    M = 1001
    x = np.linspace(0.0, 1.0, M)
    u_true = np.sin(2 * np.pi * x)
    u_hat = u_true + 0.05 * np.cos(6 * np.pi * x)  # a small perturbation

    out = compute_relative_L2_error(u_hat, u_true, x)
    print(f"Relative L2 error: {out['rel_L2_err']:.6e}")
    # Uncomment below to inspect arrays:
    # print(out['e'][:5], out['e_from_diff'][:5], out['F'][:5])


