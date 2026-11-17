# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""The term structure of vol-of-vol from 1 day to 1 year, measured with Cboe's VIX-tenor indices (stdlib only).

Cboe publishes model-free implied volatility indices for the S&P 500 at six tenors: VIX1D (1 day), VIX9D (9 days),
VIX (30 days), VIX3M (93 days), VIX6M (184 days) and VIX1Y (365 days). Each is (approximately) the square root of a
variance-swap rate. This module measures how volatile each index's daily log change is, as a function of its tenor,
and compares that term structure with what forward-variance models predict.

Model (first order, Gaussian; the level of vol-of-vol is a free constant): the log forward variance for delivery at
time u, seen at time t, moves as

    Z_t(u) = integral_{-inf}^{t} k(u - s) dW_s

with kernel k. The log of a tenor-Delta index squared is the window average (1/Delta) * integral_t^{t+Delta} Z_t(u) du,
so its change over a step delta is a Gaussian with variance proportional to

    I(Delta, delta) = integral h_0(s)^2 ds,      h_j(s) = K_{(j+1)delta}(s) - K_{j delta}(s),
    K_t(s) = (G(t + Delta - s) - G(t - s)) / Delta  for s <= t (else 0),     G(x) = integral_0^x k.

C1 = integral h_0(s) h_1(s) ds is the covariance of two consecutive changes. Kernels:

    rough (power law, Gatheral-Jaisson-Rosenbaum / rough Bergomi):  k(x) = x^(H - 1/2)
    time-shifted power law (TSPL, Guyon-El Amrani):                 k(x) = (x + eps)^(H - 1/2)
    one-factor exponential (Markov, one-factor Bergomi):           k(x) = exp(-kappa x)

For a step delta much shorter than the tenor, the rough kernel gives sd ~ Delta^(H - 1/2), a power law in the tenor.
The finite one-day step matters at the 1-day tenor (the window rolls completely), which is why I is computed exactly.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime, timedelta

TENORS = (("VIX1D", 1.0), ("VIX9D", 9.0), ("VIX", 30.0), ("VIX3M", 93.0), ("VIX6M", 184.0), ("VIX1Y", 365.0))

# FOMC statement days (last day of each scheduled meeting), from federalreserve.gov/monetarypolicy/fomccalendars.htm.
FOMC_STATEMENT_DAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    )
)

# ----------------------------------------------------------------------------------------------- data


def parse_date(text) -> date | None:
    text = (text or "").strip() if isinstance(text, str) else ""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_closes(text: str) -> dict:
    """{date: close} from a Cboe DATE,OPEN,HIGH,LOW,CLOSE history. Rows without a positive close are skipped."""
    out = {}
    for record in csv.DictReader(io.StringIO(text.lstrip(chr(0xFEFF)))):
        clean = {}
        for key, value in record.items():
            if isinstance(key, str):
                clean[key.replace(chr(0xFEFF), "").strip().upper()] = value
        day = parse_date(clean.get("DATE"))
        try:
            value = float(clean.get("CLOSE"))
        except (TypeError, ValueError):
            continue
        if day is not None and math.isfinite(value) and value > 0:
            out[day] = value
    return out


def event_window_days(days) -> set:
    """Each event day plus the weekday before it (the day whose 1-day window contains the event)."""
    out = set()
    for day in days:
        out.add(day)
        previous = day - timedelta(days=1)
        while previous.weekday() >= 5:
            previous -= timedelta(days=1)
        out.add(previous)
    return out


def make_pairs(series: dict, names, gap_days: int, start=None, end=None, exclude=frozenset()) -> list:
    """[(d, d + gap, [log change of each named index])] over dates where every index has a close on both days.

    gap_days = 1 keeps only consecutive calendar days (Mon-Tue ... Thu-Fri), so weekends and holidays never enter.
    Pairs touching a day in `exclude` are dropped; `end` bounds the second day.
    """
    common = set(series[names[0]])
    for name in names[1:]:
        common &= set(series[name])
    out = []
    for day in sorted(common):
        if start is not None and day < start:
            continue
        later = day + timedelta(days=gap_days)
        if end is not None and later > end:
            continue
        if later not in common or day in exclude or later in exclude:
            continue
        out.append((day, later, [math.log(series[n][later] / series[n][day]) for n in names]))
    return out


# ----------------------------------------------------------------------------------------------- statistics


def median(values) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def robust_scale(values) -> float:
    """1.4826 x median absolute deviation: the standard deviation for Gaussian data, robust to rare spikes."""
    centre = median(values)
    return 1.4826 * median([abs(v - centre) for v in values])


def stdev(values) -> float:
    n = len(values)
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def lag1_autocorr(pairs, col: int) -> float:
    """Pearson correlation of consecutive changes: pair (a, b) with the pair that starts at b."""
    by_start = {p[0]: p for p in pairs}
    xs, ys = [], []
    for p in pairs:
        q = by_start.get(p[1])
        if q is not None:
            xs.append(p[2][col])
            ys.append(q[2][col])
    if len(xs) < 3:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def percentile(values, q: float) -> float:
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def block_bootstrap_indices(n: int, block: int, rng) -> list:
    """Moving-block bootstrap: n indices made of runs of `block` consecutive indices with random starts."""
    if n < block:
        raise ValueError("series shorter than one block")
    out = []
    while len(out) < n:
        first = rng.randrange(0, n - block + 1)
        out.extend(range(first, first + block))
    return out[:n]


# ----------------------------------------------------------------------------------------------- kernels


def power_G(H: float, eps: float = 0.0):
    """Primitive of k(x) = (x + eps)^(H - 1/2): G(x) = ((x + eps)^a - eps^a) / a, a = H + 1/2."""
    a = H + 0.5
    base = eps ** a if eps > 0 else 0.0

    def G(x):
        return ((max(x, 0.0) + eps) ** a - base) / a

    return G


def power_tail(H: float, eps: float = 0.0):
    """Integral of h0*h0 and h0*h1 beyond T for the power kernel (h_j ~ delta * G'' far in the past)."""
    a = H + 0.5

    def tail(T, Delta, delta):
        c = (a - 1.0) ** 2 * delta ** 2 / (3.0 - 2.0 * a)
        m0 = eps + (delta + Delta) / 2.0
        m1 = m0 + delta / 2.0
        return c * (T + m0) ** (2.0 * a - 3.0), c * (T + m1) ** (2.0 * a - 3.0)

    return tail


def exp_G(kappa: float):
    """Primitive of k(x) = exp(-kappa x)."""

    def G(x):
        return -math.expm1(-kappa * max(x, 0.0)) / kappa

    return G


def exp_moments_closed(kappa: float, Delta: float, delta: float):
    """Exact (I, C1) for the exponential kernel: an Ornstein-Uhlenbeck factor seen through a Delta-window."""
    w = -math.expm1(-kappa * Delta) / (kappa * Delta)
    d = -math.expm1(-kappa * delta)
    return w * w * d / kappa, -w * w * d * d / (2.0 * kappa)


# ----------------------------------------------------------------------------------------------- integrals

_SUB = 6  # Simpson sub-intervals per geometric segment (even)


def _segments(length: float, first: float, ratio: float = 1.2) -> list:
    bounds = [0.0]
    x = first
    while x < length:
        bounds.append(x)
        x *= ratio
    bounds.append(length)
    return bounds


def _integrate_pair(f, length: float, first: float):
    """Integral over [0, length] of a function returning two values; Simpson on geometrically growing segments."""
    total0 = total1 = 0.0
    bounds = _segments(length, first)
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        step = (hi - lo) / _SUB
        acc0 = acc1 = 0.0
        for i in range(_SUB + 1):
            w = 1.0 if i in (0, _SUB) else (4.0 if i % 2 else 2.0)
            v0, v1 = f(lo + i * step)
            acc0 += w * v0
            acc1 += w * v1
        total0 += acc0 * step / 3.0
        total1 += acc1 * step / 3.0
    return total0, total1


def moments(G, Delta: float, delta: float, tail=None, horizon: float = 2000.0):
    """(I, C1): variance and lag-1 covariance (up to the vol-of-vol constant) of the change in log index^2.

    Shocks before the first window start (s = -tau <= 0) and inside the first step (s = delta - v, 0 <= v <= delta)
    are integrated separately, each on a grid that is dense where the kernel is singular.
    """

    def before(tau):
        k0 = (G(Delta + tau) - G(tau)) / Delta
        k1 = (G(delta + Delta + tau) - G(delta + tau)) / Delta
        k2 = (G(2.0 * delta + Delta + tau) - G(2.0 * delta + tau)) / Delta
        h0, h1 = k1 - k0, k2 - k1
        return h0 * h0, h0 * h1

    def inside(v):
        k1 = (G(Delta + v) - G(v)) / Delta
        k2 = (G(delta + Delta + v) - G(delta + v)) / Delta
        return k1 * k1, k1 * (k2 - k1)

    T = horizon * (Delta + 2.0 * delta)
    a0, a1 = _integrate_pair(before, T, 1e-6 * min(delta, Delta))
    b0, b1 = _integrate_pair(inside, delta, 1e-6 * delta)
    t0 = t1 = 0.0
    if tail is not None:
        t0, t1 = tail(T, Delta, delta)
    return a0 + b0 + t0, a1 + b1 + t1


# ----------------------------------------------------------------------------------------------- fitting


def fit_curves(y, curves: dict):
    """Best (key, level, sse) for y_i = level + m_i, least squares over a grid of model curves {key: [m_i]}.

    The level (log vol-of-vol constant) is solved exactly for each curve. Ties keep the first key in sorted order.
    """
    best = None
    n = len(y)
    for key in sorted(curves):
        m = curves[key]
        level = sum(yi - mi for yi, mi in zip(y, m)) / n
        sse = sum((yi - level - mi) ** 2 for yi, mi in zip(y, m))
        if best is None or sse < best[2] - 1e-15:
            best = (key, level, sse)
    return best


def fit_all(y, tenors, rough: dict, tspl: dict, expo: dict, long_min: float = 30.0) -> dict:
    """Fits of the log robust vol-of-vol term structure y (one value per tenor).

    rough: {H: [m per tenor]}, tspl: {(H, eps): [...]}, expo: {kappa: [...]}, where m = 0.5 * ln I.
    The rough model is also fitted to the long end only (tenors >= long_min) and extrapolated to the short tenors:
    log_ratio[tenor] = ln(observed / extrapolated).
    """
    r_key, r_level, r_sse = fit_curves(y, rough)
    t_key, t_level, t_sse = fit_curves(y, tspl)
    e_key, e_level, e_sse = fit_curves(y, expo)
    long_idx = [i for i, t in enumerate(tenors) if t >= long_min]
    short_idx = [i for i, t in enumerate(tenors) if t < long_min]
    sub = {k: [m[i] for i in long_idx] for k, m in rough.items()}
    l_key, l_level, l_sse = fit_curves([y[i] for i in long_idx], sub)
    log_ratio = {tenors[i]: y[i] - l_level - rough[l_key][i] for i in short_idx}
    return {
        "rough_H": r_key, "rough_level": r_level, "rough_sse": r_sse,
        "tspl_H": t_key[0], "tspl_eps": t_key[1], "tspl_level": t_level, "tspl_sse": t_sse,
        "exp_kappa": e_key, "exp_level": e_level, "exp_sse": e_sse,
        "long_H": l_key, "long_level": l_level, "long_sse": l_sse,
        "log_ratio": log_ratio,
    }


def verdict(r1_lo: float, r1_hi: float, r9_lo: float, r9_hi: float) -> str:
    """Pre-registered reading of the 90% bootstrap intervals of ln(observed / rough extrapolation)."""
    if r1_hi < 0:
        return "short-end flattening: 1-day vol-of-vol is below the rough-volatility extrapolation"
    if r1_lo > 0:
        return "short end steeper than the rough-volatility extrapolation"
    if r9_lo <= 0 <= r9_hi:
        return "power law holds down to one day"
    return "mixed: 1-day consistent with the extrapolation, 9-day not"
