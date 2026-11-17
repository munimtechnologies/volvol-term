# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Stage 3 of volvol-term (PREREG3.md): scheduled macro releases at the one-day end. Stdlib only.

VIX1D at the close of day d prices the variance of the next trading session. When that session holds a scheduled
release (CPI and the jobs report at 8:30 ET, before the open; the FOMC statement at 14:00 ET), the index carries a
known "event variance" on the eve of the release and drops it once the release is out. That roll-in and roll-out is a
deterministic move of the one-day window across a jump in the forward-variance curve, not diffusive volatility of
volatility, so it can inflate the measured one-day vol-of-vol whatever the kernel is.

This module holds the release calendars and classifies daily pairs (d1, d2) of consecutive calendar days:

    "in"    d2 is the eve of an event: the window of d2 contains it, the window of d1 does not;
    "out"   d1 is the eve of an event: the event leaves the window between d1 and d2;
    "both"  both at once (one event leaves as another enters);
    "clean" neither.

The eve of an event day E is the last trading day in the data strictly before E (the trading-day set comes from the
data, so market holidays are handled without a holiday table).
"""

from __future__ import annotations

import bisect
from datetime import date

import volvol as vv

# Consumer Price Index releases (8:30 ET), from the BLS archive www.bls.gov/bls/news-release/cpi.htm
# (file names cpi_MMDDYYYY). The October 2025 CPI was never published (2025 lapse in appropriations).
CPI_RELEASE_DAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13",
        "2022-10-13", "2022-11-10", "2022-12-13",
        "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
        "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
        "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
        "2025-09-11", "2025-10-24", "2025-12-18",
        "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
        "2026-09-11", "2026-10-14", "2026-11-10",
    )
)

# Employment Situation releases (8:30 ET), from the BLS archive www.bls.gov/bls/news-release/empsit.htm
# (file names empsit_MMDDYYYY). The October 2025 report was never published.
EMPSIT_RELEASE_DAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2022-02-04", "2022-03-04", "2022-04-01", "2022-05-06", "2022-06-03", "2022-07-08", "2022-08-05", "2022-09-02",
        "2022-10-07", "2022-11-04", "2022-12-02",
        "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07", "2023-05-05", "2023-06-02", "2023-07-07", "2023-08-04",
        "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
        "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
        "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
        "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
        "2025-09-05", "2025-11-20", "2025-12-16",
        "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
        "2026-09-04", "2026-10-02", "2026-11-06",
    )
)

LABELS = ("in", "out", "both", "clean")


def event_days() -> dict:
    """{event day: sorted tuple of event names} for CPI, NFP (the jobs report) and FOMC statement days."""
    out = {}
    for name, days in (("CPI", CPI_RELEASE_DAYS), ("NFP", EMPSIT_RELEASE_DAYS), ("FOMC", vv.FOMC_STATEMENT_DAYS)):
        for day in days:
            out.setdefault(day, set()).add(name)
    return {day: tuple(sorted(names)) for day, names in out.items()}


def eves(trading_days, events: dict) -> dict:
    """{eve: sorted tuple of event names}: the eve of E is the last trading day strictly before E.

    Events on or before the first trading day, or after the last one (whose next session is not in the data), are
    skipped.
    """
    days = sorted(trading_days)
    out = {}
    for day, names in events.items():
        i = bisect.bisect_left(days, day)
        if i == 0 or i == len(days):
            continue
        out.setdefault(days[i - 1], set()).update(names)
    return {eve: tuple(sorted(names)) for eve, names in out.items()}


def classify(pairs, eve_set) -> list:
    """One label per pair (d1, d2, ...): "in", "out", "both" or "clean" (see the module docstring)."""
    labels = []
    for p in pairs:
        enters, leaves = p[1] in eve_set, p[0] in eve_set
        labels.append("both" if enters and leaves else ("in" if enters else ("out" if leaves else "clean")))
    return labels


def medians_by_label(values, labels) -> dict:
    """{label: (median, count)} over the labels present."""
    by = {}
    for v, g in zip(values, labels):
        by.setdefault(g, []).append(v)
    return {g: (vv.median(vs), len(vs)) for g, vs in by.items()}


def event_share(values, labels) -> float:
    """Share of the sum of squared deviations from the overall mean that falls on pairs not labelled "clean"."""
    m = sum(values) / len(values)
    total = sum((v - m) ** 2 for v in values)
    touched = sum((v - m) ** 2 for v, g in zip(values, labels) if g != "clean")
    return touched / total if total > 0 else float("nan")


def comparison_reading(p05: float, p95: float) -> str:
    """Pre-registered reading of the 90% interval of |error rough| - |error two-factor| at the held-out 1-day tenor."""
    if p05 > 0:
        return "the two-factor Markov kernel predicts the one-day end better than the rough kernel"
    if p95 < 0:
        return "the rough kernel predicts the one-day end better than the two-factor Markov kernel"
    return "no clear winner at the one-day end"
