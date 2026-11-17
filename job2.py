# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Run stage 2 of the vol-of-vol term-structure test (PREREG2.md) end to end. Stdlib only.

Uses the same six Cboe histories and the same primary sample as stage 1 (A_daily: consecutive calendar days, all six
indices). Writes results2.json and results2.md to --out-dir (derived statistics only, never Cboe's raw data).

    python job2.py --data-dir /data --out-dir /out
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
from datetime import date, datetime, timezone

import job
import stage2 as s2
import volvol as vv

SEED2 = 20261118
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
FAMILIES = ("rough", "tspl", "exp1", "exp2")


def log_scales(cols, idx, groups=None):
    """ln robust scale of each column over the pair indices idx (weekday-centred when groups is given)."""
    out = []
    for c in cols:
        values = [c[i] for i in idx]
        if len(values) < 20:
            return None
        if groups is None:
            s = vv.robust_scale(values)
        else:
            s = s2.weekday_centred_scale(values, [groups[i] for i in idx])
        if not s > 0:
            return None
        out.append(math.log(s))
    return out


def measure(idx, ctx) -> dict:
    """Every stage-2 statistic for one set of pair indices (the full sample or one bootstrap draw)."""
    cols, tenors = ctx["cols"], ctx["tenors"]
    out = {}
    for label, groups in (("plain", None), ("dow", ctx["weekday"])):
        y = log_scales(cols, idx, groups)
        if y is None:
            continue
        H, lr = s2.long_end(y, tenors, ctx["rough"], ctx["rough_long"], ctx["long_idx"])
        out[f"{label}_long_H"] = H
        out[f"{label}_ln_R1"] = lr[1.0]
        out[f"{label}_ln_R9"] = lr[9.0]
        for fam in FAMILIES:
            e1, _ = s2.holdout_error(y, ctx["keep1"], 0, ctx["families"][fam], ctx["subs1"][fam])
            e9, _ = s2.holdout_error(y, ctx["keep9"], 1, ctx["families"][fam], ctx["subs9"][fam])
            out[f"{label}_err1_{fam}"] = e1
            out[f"{label}_err9_{fam}"] = e9
        for fam in ("tspl", "exp1", "exp2"):
            out[f"{label}_abs_err1_rough_minus_{fam}"] = abs(out[f"{label}_err1_rough"]) - abs(out[f"{label}_err1_{fam}"])
    for g in range(3):
        sub = [i for i in idx if ctx["terc"][i] == g]
        y = log_scales(cols, sub)
        if y is None:
            continue
        H, lr = s2.long_end(y, tenors, ctx["rough"], ctx["rough_long"], ctx["long_idx"])
        out[f"t{g}_long_H"] = H
        out[f"t{g}_ln_R1"] = lr[1.0]
        out[f"t{g}_ln_R9"] = lr[9.0]
        for j, name in enumerate(ctx["names"]):
            out[f"t{g}_sd_{name}"] = math.exp(y[j])
    return out


def predictions(point, verdicts, wk_medians) -> dict:
    try:
        m = list(wk_medians["VIX1D"].values())
        signs = {point[f"t{g}_ln_R1"] < 0 for g in range(3)}
        hs = [point[f"t{g}_long_H"] for g in range(3)]
        return {
            "Q1_weekday_centred_verdict_equals_plain_verdict": verdicts["dow"] == verdicts["plain"],
            "Q2_VIX1D_weekday_median_range_above_0.05": max(m) - min(m) > 0.05,
            "Q3_weekday_centring_lowers_ln_R1": point["dow_ln_R1"] < point["plain_ln_R1"],
            "Q4_ln_R1_same_sign_in_all_VIX_terciles": len(signs) == 1,
            "Q5_long_H_spread_across_terciles_at_most_0.10": max(hs) - min(hs) <= 0.10 + 1e-9,
            "Q6_VIX_vol_of_vol_higher_in_top_tercile": point["t2_sd_VIX"] > point["t0_sd_VIX"],
            "Q7_two_factor_predicts_1d_better_than_rough": abs(point["dow_err1_exp2"]) < abs(point["dow_err1_rough"]),
            "Q8_tspl_predicts_1d_better_than_rough": abs(point["dow_err1_tspl"]) < abs(point["dow_err1_rough"]),
        }
    except (KeyError, ValueError) as error:
        return {"error": f"a required statistic is missing: {error}"}


def comparison_reading(iv) -> str:
    """Pre-registered reading of the 90% interval of |error rough| - |error two-factor| at the held-out 1-day tenor."""
    if iv["p05"] > 0:
        return "the two-factor Markov kernel predicts the one-day end better than the rough kernel"
    if iv["p95"] < 0:
        return "the rough kernel predicts the one-day end better than the two-factor Markov kernel"
    return "no clear winner at the one-day end"


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

    names = job.ALL6
    series, sources = {}, {}
    for name in names:
        url = job.URL.format(name=name)
        text = job.fetch(url, os.path.join(args.data_dir, f"{name}_History.csv"))
        series[name] = vv.parse_closes(text)
        sources[name] = {"url": url, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "first": min(series[name]).isoformat(), "last": max(series[name]).isoformat(),
                         "days": len(series[name])}
        time.sleep(0.5)
    print("downloaded", {k: (v["first"], v["last"]) for k, v in sources.items()}, flush=True)

    # The stage-1 primary sample A_daily.
    pairs = vv.make_pairs(series, names, 1, end=date.fromordinal(run_day.toordinal() - 1))
    n = len(pairs)
    tenors = [dict(vv.TENORS)[name] for name in names]
    cols = [[p[2][j] for p in pairs] for j in range(len(names))]
    weekday = [p[0].weekday() for p in pairs]
    vix = [series["VIX"][p[0]] for p in pairs]
    cuts = s2.tercile_cutoffs(vix)
    terc = [s2.tercile_label(v, cuts) for v in vix]

    t0 = time.time()
    rough, tspl, expo = job.curves_for(tenors, 1.0)
    two = s2.two_factor_curves(tenors, 1.0)
    print(f"model curves ready in {time.time() - t0:.0f} s ({len(two)} two-factor shapes)", flush=True)
    long_idx = [i for i, t in enumerate(tenors) if t >= 30.0]
    keep1 = [i for i in range(len(tenors)) if i != 0]
    keep9 = list(long_idx)
    families = {"rough": rough, "tspl": tspl, "exp1": expo, "exp2": two}
    ctx = {
        "names": names, "tenors": tenors, "cols": cols, "weekday": weekday, "terc": terc,
        "rough": rough, "rough_long": s2.subset_curves(rough, long_idx), "long_idx": long_idx,
        "families": families, "keep1": keep1, "keep9": keep9,
        "subs1": {f: s2.subset_curves(c, keep1) for f, c in families.items()},
        "subs9": {f: s2.subset_curves(c, keep9) for f, c in families.items()},
    }

    point = measure(list(range(n)), ctx)
    print("point estimates done", flush=True)

    wk_medians = {name: {} for name in names}
    wk_means = {name: {} for name in names}
    for j, name in enumerate(names):
        for w in sorted(set(weekday)):
            values = [cols[j][i] for i in range(n) if weekday[i] == w]
            wk_medians[name][f"{WEEKDAY_NAMES[w]}->{WEEKDAY_NAMES[w + 1]}"] = vv.median(values)
            wk_means[name][f"{WEEKDAY_NAMES[w]}->{WEEKDAY_NAMES[w + 1]}"] = sum(values) / len(values)

    full_fits = {}
    for label, groups in (("plain", None), ("dow", weekday)):
        y = log_scales(cols, list(range(n)), groups)
        row = {"observed_sd": {name: math.exp(v) for name, v in zip(names, y)}}
        for fam in FAMILIES:
            key, level, sse = vv.fit_curves(y, families[fam])
            row[fam] = {"params": list(key) if isinstance(key, tuple) else key, "sse": sse,
                        "fit_sd": {name: math.exp(level + m) for name, m in zip(names, families[fam][key])}}
        full_fits[label] = row

    rng = random.Random(SEED2)
    draws = {}
    t0 = time.time()
    for rep in range(job.BOOT_REPS):
        idx = vv.block_bootstrap_indices(n, job.BLOCK, rng)
        for k, v in measure(idx, ctx).items():
            draws.setdefault(k, []).append(v)
        if rep == 9:
            print(f"10 bootstrap draws in {time.time() - t0:.0f} s", flush=True)
    boot = {}
    for k, v in draws.items():
        boot[k] = dict(job.interval(v), n=len(v))

    verdicts = {}
    for label in ("plain", "dow"):
        r1, r9 = boot[f"{label}_ln_R1"], boot[f"{label}_ln_R9"]
        verdicts[label] = vv.verdict(r1["p05"], r1["p95"], r9["p05"], r9["p95"])
    readings = {label: comparison_reading(boot[f"{label}_abs_err1_rough_minus_exp2"]) for label in ("plain", "dow")}

    results = {
        "prereg": "PREREG2.md", "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources, "pairs": n, "first_day": pairs[0][0].isoformat(), "last_day": pairs[-1][1].isoformat(),
        "vix_tercile_cutoffs": list(cuts), "tercile_pairs": [terc.count(g) for g in range(3)],
        "weekday_median_change": wk_medians, "weekday_mean_change": wk_means,
        "point": point, "full_fits": full_fits, "bootstrap": boot,
        "bootstrap_reps": job.BOOT_REPS, "block": job.BLOCK, "seed": SEED2,
        "verdict_plain": verdicts["plain"], "verdict_weekday_centred": verdicts["dow"],
        "one_day_holdout_reading": readings,
        "predictions": predictions(point, verdicts, wk_medians),
    }
    results["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.out_dir, "results2.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)

    lines = ["# volvol-term stage 2: weekday seasonality, volatility regime and a two-factor rival", "",
             f"Run {results['run_utc']}. Pre-registration: PREREG2.md. {n} daily pairs, {results['first_day']} to "
             f"{results['last_day']}.", "",
             f"- Verdict, plain robust scale (stage-1 rule): **{verdicts['plain']}**",
             f"- Verdict, weekday-centred robust scale: **{verdicts['dow']}**",
             f"- One-day hold-out, plain: **{readings['plain']}**",
             f"- One-day hold-out, weekday-centred: **{readings['dow']}**", "",
             "## Median daily log change by weekday", "",
             "| index | " + " | ".join(wk_medians[names[0]]) + " |",
             "|---|" + "---|" * len(wk_medians[names[0]])]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(fmt(v, 4) for v in wk_medians[name].values()) + " |")
    lines += ["", "## Full-sample fits (all six tenors)", ""]
    for label, row in full_fits.items():
        lines += [f"### {label}", "", "| index | observed | rough | TSPL | 1-factor exp | 2-factor exp |",
                  "|---|---|---|---|---|---|"]
        for name in names:
            lines.append(f"| {name} | {fmt(row['observed_sd'][name], 4)} | " +
                         " | ".join(fmt(row[f]['fit_sd'][name], 4) for f in FAMILIES) + " |")
        lines += ["", "; ".join(f"{f}: params {row[f]['params']}, SSE {fmt(row[f]['sse'], 5)}" for f in FAMILIES), ""]
    lines += ["## VIX terciles", "", f"Cutoffs (VIX close on the first day): {fmt(cuts[0], 2)}, {fmt(cuts[1], 2)}; "
              f"pairs per tercile: {results['tercile_pairs']}.", ""]
    lines += ["## Point estimates and bootstrap intervals", "", "| statistic | point | p05 | p50 | p95 |",
              "|---|---|---|---|---|"]
    for k in sorted(boot):
        b = boot[k]
        lines.append(f"| {k} | {fmt(point.get(k), 4)} | {fmt(b['p05'], 4)} | {fmt(b['p50'], 4)} | {fmt(b['p95'], 4)} |")
    lines += ["", "## Pre-registered predictions", ""]
    for k, v in results["predictions"].items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(args.out_dir, "results2.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
