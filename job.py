# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Run the pre-registered vol-of-vol term-structure test (PREREG.md) end to end. Stdlib only.

Downloads Cboe's daily histories of VIX1D, VIX9D, VIX, VIX3M, VIX6M and VIX1Y, measures the robust volatility of each
index's log changes, fits rough, time-shifted power-law and one-factor exponential kernels, bootstraps, and writes
results.json and results.md to --out-dir. Only derived statistics are written, never Cboe's raw data.

    python job.py --data-dir /data --out-dir /out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from datetime import date, datetime, timezone

import volvol as vv

USER_AGENT = "volvol-term/0.1 (MV2 of Munim, Inc.; mv2@munimtech.com)"
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"

# ---- fixed in PREREG.md before any result was seen
H_GRID = tuple(round(0.01 * i, 2) for i in range(1, 100))
EPS_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
KAPPA_GRID = tuple(10 ** (-4 + 0.025 * i) for i in range(201))
BOOT_REPS = 1000
BLOCK = 20
SEED = 20261117
VIX1D_LIVE = date(2023, 4, 24)
ALL6 = [n for n, _ in vv.TENORS]
SPECS = (
    {"id": "A_daily", "names": ALL6, "gap": 1, "start": None, "fomc": False},  # primary
    {"id": "A_daily_noFOMC", "names": ALL6, "gap": 1, "start": None, "fomc": True},
    {"id": "A_weekly", "names": ALL6, "gap": 7, "start": None, "fomc": False},
    {"id": "A_daily_live", "names": ALL6, "gap": 1, "start": VIX1D_LIVE, "fomc": False},
    {"id": "B_daily_2011", "names": ALL6[1:], "gap": 1, "start": date(2011, 1, 1), "fomc": False},
)


def fetch(url: str, path: str, tries: int = 3) -> str:
    last_error = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8-sig", errors="replace")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return text
        except Exception as error:  # network hiccup: retry, then fall back to the cached copy
            last_error = error
            time.sleep(2 + 3 * attempt)
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as handle:
            return handle.read()
    raise RuntimeError(f"download failed: {url}: {last_error}")


_CACHE: dict = {}


def model_m(kind: str, param, Delta: float, delta: float) -> float:
    """0.5 * ln I for one kernel, tenor and step (cached)."""
    key = (kind, param, Delta, delta)
    if key not in _CACHE:
        if kind == "power":
            H, eps = param
            I, C1 = vv.moments(vv.power_G(H, eps), Delta, delta, vv.power_tail(H, eps))
        else:
            I, C1 = vv.exp_moments_closed(param, Delta, delta)
        _CACHE[key] = (0.5 * math.log(I) if I > 0 else float("nan"), C1 / I if I > 0 else float("nan"))
    return _CACHE[key][0]


def model_rho(kind: str, param, Delta: float, delta: float) -> float:
    model_m(kind, param, Delta, delta)
    return _CACHE[(kind, param, Delta, delta)][1]


def curves_for(tenors, delta):
    rough = {H: [model_m("power", (H, 0.0), t, delta) for t in tenors] for H in H_GRID}
    tspl = {(H, e): [model_m("power", (H, e), t, delta) for t in tenors] for H in H_GRID for e in EPS_GRID}
    expo = {k: [model_m("exp", k, t, delta) for t in tenors] for k in KAPPA_GRID}
    clean = lambda d: {k: v for k, v in d.items() if all(math.isfinite(x) for x in v)}
    return clean(rough), clean(tspl), clean(expo)


def interval(values):
    return {"p05": vv.percentile(values, 0.05), "p50": vv.percentile(values, 0.5), "p95": vv.percentile(values, 0.95)}


def analyse(spec, series, run_day, fomc_days):
    names = spec["names"]
    tenors = [dict(vv.TENORS)[n] for n in names]
    exclude = fomc_days if spec["fomc"] else frozenset()
    # Pairs ending on the run day are dropped (that day's closes may not be final).
    pairs = vv.make_pairs(series, names, spec["gap"], start=spec["start"],
                          end=date.fromordinal(run_day.toordinal() - 1), exclude=exclude)
    out = {"id": spec["id"], "names": names, "tenors_days": tenors, "step_days": spec["gap"], "pairs": len(pairs)}
    if len(pairs) < 100:
        out["error"] = "fewer than 100 pairs"
        return out
    out["first_day"], out["last_day"] = pairs[0][0].isoformat(), pairs[-1][1].isoformat()
    cols = [[p[2][j] for p in pairs] for j in range(len(names))]
    scale = [vv.robust_scale(c) for c in cols]
    if min(scale) <= 0:
        out["error"] = "an index has zero robust scale (stale closes)"
        return out
    y = [math.log(s) for s in scale]
    rough, tspl, expo = curves_for(tenors, float(spec["gap"]))
    fit = vv.fit_all(y, tenors, rough, tspl, expo)
    per = []
    for j, (name, tenor) in enumerate(zip(names, tenors)):
        row = {
            "index": name, "tenor_days": tenor, "robust_sd": scale[j], "sd": vv.stdev(cols[j]),
            "share_zero": sum(1 for v in cols[j] if v == 0.0) / len(cols[j]),
            "lag1_autocorr": vv.lag1_autocorr(pairs, j),
            "rough_fit_sd": math.exp(fit["rough_level"] + rough[fit["rough_H"]][j]),
            "long_fit_sd": math.exp(fit["long_level"] + rough[fit["long_H"]][j]),
            "tspl_fit_sd": math.exp(fit["tspl_level"] + tspl[(fit["tspl_H"], fit["tspl_eps"])][j]),
            "exp_fit_sd": math.exp(fit["exp_level"] + expo[fit["exp_kappa"]][j]),
            "rough_lag1_model": model_rho("power", (fit["rough_H"], 0.0), tenor, float(spec["gap"])),
            "long_lag1_model": model_rho("power", (fit["long_H"], 0.0), tenor, float(spec["gap"])),
        }
        per.append(row)
    slopes = []
    for j in range(len(names) - 1):
        slopes.append({
            "from": names[j], "to": names[j + 1],
            "empirical": (y[j + 1] - y[j]) / math.log(tenors[j + 1] / tenors[j]),
            "rough_long_fit": (rough[fit["long_H"]][j + 1] - rough[fit["long_H"]][j]) / math.log(tenors[j + 1] / tenors[j]),
        })
    out.update({"per_tenor": per, "local_slopes": slopes,
                "fit": {k: v for k, v in fit.items() if k != "log_ratio"},
                "log_ratio_observed_over_rough_extrapolation": {f"{k:g}": v for k, v in fit["log_ratio"].items()}})

    rng = random.Random(SEED + SPECS.index(spec))
    boot = {"long_H": [], "rough_H": [], "tspl_eps": [], "sse_exp_minus_rough": []}
    for t in fit["log_ratio"]:
        boot[f"log_ratio_{t:g}d"] = []
    for _ in range(BOOT_REPS):
        idx = vv.block_bootstrap_indices(len(pairs), BLOCK, rng)
        yb = [math.log(vv.robust_scale([c[i] for i in idx])) for c in cols]
        fb = vv.fit_all(yb, tenors, rough, tspl, expo)
        boot["long_H"].append(fb["long_H"])
        boot["rough_H"].append(fb["rough_H"])
        boot["tspl_eps"].append(fb["tspl_eps"])
        boot["sse_exp_minus_rough"].append(fb["exp_sse"] - fb["rough_sse"])
        for t, v in fb["log_ratio"].items():
            boot[f"log_ratio_{t:g}d"].append(v)
    out["bootstrap"] = {k: interval(v) for k, v in boot.items()}
    out["bootstrap"]["share_tspl_eps_positive"] = sum(1 for e in boot["tspl_eps"] if e > 0) / BOOT_REPS
    out["bootstrap"]["reps"], out["bootstrap"]["block"] = BOOT_REPS, BLOCK
    return out


def predictions(res):
    a, nf, wk, b = res["A_daily"], res["A_daily_noFOMC"], res["A_weekly"], res["B_daily_2011"]
    if any("error" in r for r in (a, nf, wk, b)):
        return {"error": "a required sample is missing"}
    ys = [r["robust_sd"] for r in a["per_tenor"]]
    lr = lambda r: r["log_ratio_observed_over_rough_extrapolation"]["1"]
    rho = {r["index"]: r["lag1_autocorr"] for r in a["per_tenor"]}
    return {
        "P1_robust_sd_strictly_falls_with_tenor": all(p > q for p, q in zip(ys, ys[1:])),
        "P2_long_H_in_0.02_0.30": 0.02 <= a["fit"]["long_H"] <= 0.30,
        "P3_1d_below_rough_extrapolation": lr(a) < 0,
        "P4_tspl_eps_positive": a["fit"]["tspl_eps"] > 0,
        "P5_rough_beats_exponential": a["fit"]["rough_sse"] < a["fit"]["exp_sse"],
        "P6_negative_lag1_VIX1D_VIX9D": rho["VIX1D"] < 0 and rho["VIX9D"] < 0,
        "P7_B_rough_H_within_0.10_of_A_long_H": abs(b["fit"]["rough_H"] - a["fit"]["long_H"]) <= 0.10 + 1e-9,
        "P8_same_side_noFOMC_and_weekly": (lr(a) < 0) == (lr(nf) < 0) == (lr(wk) < 0),
    }


def fmt(v, d=3):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args(argv)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    run_day = datetime.now(timezone.utc).date()
    started = time.time()

    series, sources = {}, {}
    for name in ALL6:
        url = URL.format(name=name)
        text = fetch(url, os.path.join(args.data_dir, f"{name}_History.csv"))
        series[name] = vv.parse_closes(text)
        sources[name] = {"url": url, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "first": min(series[name]).isoformat(), "last": max(series[name]).isoformat(),
                         "days": len(series[name])}
        time.sleep(0.5)
    print("downloaded", {k: (v["first"], v["last"]) for k, v in sources.items()}, flush=True)

    fomc = frozenset(vv.event_window_days(vv.FOMC_STATEMENT_DAYS))
    results = {"prereg": "PREREG.md", "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sources": sources, "samples": {}}
    for spec in SPECS:
        t0 = time.time()
        results["samples"][spec["id"]] = analyse(spec, series, run_day, fomc)
        print(spec["id"], "done in %.0f s" % (time.time() - t0), flush=True)

    res = results["samples"]
    results["predictions"] = predictions(res)
    a = res["A_daily"]
    if "error" not in a:
        bs = a["bootstrap"]
        results["verdict"] = vv.verdict(bs["log_ratio_1d"]["p05"], bs["log_ratio_1d"]["p95"],
                                        bs["log_ratio_9d"]["p05"], bs["log_ratio_9d"]["p95"])
    results["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)

    lines = ["# Vol-of-vol term structure from 1 day to 1 year: results", "",
             f"Run {results['run_utc']}. Pre-registration: PREREG.md. Verdict (primary sample A_daily): "
             f"**{results.get('verdict', 'n/a')}**", ""]
    for sid, r in res.items():
        lines += [f"## {sid}", ""]
        if "error" in r:
            lines += [f"Error: {r['error']} ({r['pairs']} pairs)", ""]
            continue
        f = r["fit"]
        lines += [f"{r['pairs']} pairs, {r['first_day']} to {r['last_day']}, step {r['step_days']} day(s).", "",
                  "| index | tenor (d) | robust sd | sd | lag-1 corr | rough fit | long-end rough | TSPL fit | exp fit |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in r["per_tenor"]:
            lines.append(f"| {p['index']} | {p['tenor_days']:g} | {fmt(p['robust_sd'], 4)} | {fmt(p['sd'], 4)} | "
                         f"{fmt(p['lag1_autocorr'], 2)} | {fmt(p['rough_fit_sd'], 4)} | {fmt(p['long_fit_sd'], 4)} | "
                         f"{fmt(p['tspl_fit_sd'], 4)} | {fmt(p['exp_fit_sd'], 4)} |")
        lines += ["", f"Rough H (all tenors) {fmt(f['rough_H'], 2)}, SSE {fmt(f['rough_sse'], 4)}; "
                      f"long-end H {fmt(f['long_H'], 2)}; TSPL H {fmt(f['tspl_H'], 2)}, eps {fmt(f['tspl_eps'], 2)} d, "
                      f"SSE {fmt(f['tspl_sse'], 4)}; exponential kappa {fmt(f['exp_kappa'], 4)}/d, SSE {fmt(f['exp_sse'], 4)}.",
                  "", "ln(observed / rough extrapolation from the long end): " +
                  ", ".join(f"{k} d: {fmt(v, 3)}" for k, v in r["log_ratio_observed_over_rough_extrapolation"].items()),
                  "", "| bootstrap | p05 | p50 | p95 |", "|---|---|---|---|"]
        for k, v in r["bootstrap"].items():
            if isinstance(v, dict):
                lines.append(f"| {k} | {fmt(v['p05'])} | {fmt(v['p50'])} | {fmt(v['p95'])} |")
        lines += ["", f"Share of bootstrap draws with TSPL eps > 0: {fmt(r['bootstrap']['share_tspl_eps_positive'], 3)}", ""]
    lines += ["## Pre-registered predictions", ""]
    for k, v in results["predictions"].items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
