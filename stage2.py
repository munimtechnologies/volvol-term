# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Stage 2 of volvol-term (PREREG2.md): two controls and a Markovian rival for the one-day end. Stdlib only.

Three additions to stage 1 (PREREG.md), all fixed before any stage-1 or stage-2 result was seen:

1. Weekday seasonality. Albers and Kestner ("The daily rise and fall of the VIX1D", Finance Research Letters 2024)
   show that VIX1D's close-to-close log change has a strong day-of-week pattern (in 2022-2023 about +0.07 into
   Tuesday and Thursday, -0.02 into Wednesday, -0.06 into Friday), caused by how the index is computed. A fixed weekly
   pattern adds to the spread of daily changes and so inflates measured one-day vol-of-vol. Here each index's changes
   are centred on the median of their own weekday before the robust scale is taken.
2. Volatility regime. Pairs are split into terciles of the VIX close on the first day; the term structure and the
   long-end rough extrapolation are measured in each.
3. A Markovian rival. The two-factor Bergomi model has the kernel
       k(x) = (1 - theta) exp(-k1 x)  +  theta exp(-k2 x),   k1 > k2,
   with the two factors driven by Brownian motions of correlation rho. Abi Jaber and Li (Mathematical Finance 2025)
   find such a two-factor Markovian model fits SPX smiles better than its rough counterpart. Its moments are exact
   closed forms (below).
   Each model family is fitted with the one-day tenor held out and judged on how well it predicts it.

Exponential-kernel moments. With w = (1 - e^(-k Delta)) / (k Delta) and d = 1 - e^(-k delta), one factor seen through
a Delta-window has h_0(s) = -w d e^(k s) for s <= 0 and h_0(s) = w e^(-k (delta - s)) for 0 < s <= delta, so

    integral h_0^(1) h_0^(2) ds = w1 w2 (d1 d2 + 1 - e^(-(k1 + k2) delta)) / (k1 + k2),

which for k1 = k2 = k is w^2 d / k, the stage-1 closed form (volvol.exp_moments_closed).
"""

from __future__ import annotations

import math

import volvol as vv

# ---- fixed in PREREG2.md before any result was seen (rates per calendar day)
K1_GRID = tuple(10 ** (-2.5 + 0.125 * i) for i in range(29))  # fast factor: 0.0032 to 10 per day
K2_GRID = tuple(10 ** (-4.5 + 0.125 * j) for j in range(29))  # slow factor: 0.00003 to 0.1 per day
THETA_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))  # weight of the slow factor, 0.1 to 0.9
RHO_GRID = (0.0, 0.5, 1.0)


def exp_cross(k1: float, k2: float, Delta: float, delta: float) -> float:
    """Integral of h_0^(1) h_0^(2) for two exponential kernels (exact; see the module docstring)."""
    w1 = -math.expm1(-k1 * Delta) / (k1 * Delta)
    w2 = -math.expm1(-k2 * Delta) / (k2 * Delta)
    d1 = -math.expm1(-k1 * delta)
    d2 = -math.expm1(-k2 * delta)
    s = k1 + k2
    return w1 * w2 * (d1 * d2 - math.expm1(-s * delta)) / s


def two_factor_I(k1: float, k2: float, theta: float, rho: float, Delta: float, delta: float) -> float:
    """Variance (up to the vol-of-vol constant) of the change in log index^2 under the two-factor kernel."""
    a, b = 1.0 - theta, theta
    return (a * a * exp_cross(k1, k1, Delta, delta) + b * b * exp_cross(k2, k2, Delta, delta)
            + 2.0 * rho * a * b * exp_cross(k1, k2, Delta, delta))


def two_factor_curves(tenors, delta: float) -> dict:
    """{(k1, k2, theta, rho): [0.5 * ln I per tenor]} over the fixed grid, keeping k2 < k1."""
    out = {}
    for k1 in K1_GRID:
        for k2 in K2_GRID:
            if not k2 < k1 * (1.0 - 1e-9):
                continue
            for theta in THETA_GRID:
                for rho in RHO_GRID:
                    out[(k1, k2, theta, rho)] = [0.5 * math.log(two_factor_I(k1, k2, theta, rho, t, delta))
                                                 for t in tenors]
    return out


# ----------------------------------------------------------------------------------------------- statistics


def group_medians(values, groups) -> dict:
    by = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(v)
    return {g: vv.median(vs) for g, vs in by.items()}


def weekday_centred_scale(values, groups) -> float:
    """1.4826 x median |v - median of v's own group|: the robust scale after removing a fixed weekday pattern.

    With a single group this equals volvol.robust_scale.
    """
    centre = group_medians(values, groups)
    return 1.4826 * vv.median([abs(v - centre[g]) for v, g in zip(values, groups)])


def tercile_cutoffs(values):
    return vv.percentile(values, 1.0 / 3.0), vv.percentile(values, 2.0 / 3.0)


def tercile_label(value: float, cuts) -> int:
    return 0 if value <= cuts[0] else (1 if value <= cuts[1] else 2)


# ----------------------------------------------------------------------------------------------- fitting


def subset_curves(curves: dict, idx) -> dict:
    return {k: [m[i] for i in idx] for k, m in curves.items()}


def holdout_error(y, keep, hold: int, curves: dict, sub: dict):
    """Fit the family on the tenors `keep` (sub = subset_curves(curves, keep)) and predict tenor `hold`.

    Returns (observed - predicted, best key).
    """
    key, level, _ = vv.fit_curves([y[i] for i in keep], sub)
    return y[hold] - (level + curves[key][hold]), key


def long_end(y, tenors, rough: dict, rough_long: dict, long_idx):
    """Long-end rough fit (rough_long = subset_curves(rough, long_idx)) and ln(observed / extrapolated) elsewhere."""
    key, level, _ = vv.fit_curves([y[i] for i in long_idx], rough_long)
    return key, {tenors[i]: y[i] - level - rough[key][i] for i in range(len(tenors)) if i not in long_idx}
