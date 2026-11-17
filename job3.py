# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Run stage 3 of the vol-of-vol term-structure test (PREREG3.md) end to end. Stdlib only.

Same six Cboe histories and the same daily pairs as stage 1's primary sample A_daily. Each pair is labelled by whether
a scheduled release (CPI, the jobs report, an FOMC statement) enters or leaves VIX1D's one-day window between its two
days (stage3.classify). The term-structure tests of stages 1 and 2 are repeated on the clean pairs. Writes
results3.json and results3.md to --out-dir (derived statistics only, never Cboe's raw data).

    python job3.py --data-dir /data --out-dir /out
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
import stage3 as s3
import volvol as vv

SEED3 = 20261119
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
    """Every stage-3 statistic for one set of pair indices."""
    out = {}
    for label, groups in (("plain", None), ("dow", ctx["weekday"])):
        y = log_scales(ctx["cols"], idx, groups)
        if y is None:
            continue
        H, lr = s2.long_end(y, ctx["tenors"], ctx["rough"], ctx["rough_long"], ctx["long_idx"])
        out[f"{label}_long_H"] = H
        out[f"{label}_ln_R1"] = lr[1.0]
        out[f"{label}_ln_R9"] = lr[9.0]
        for j, name in enumerate(ctx["names"]):
            out[f"{label}_sd_{name}"] = math.exp(y[j])
        for fam in FAMILIES:
            e1, _ = s2.holdout_error(y, ctx["keep1"], 0, ctx["families"][fam], ctx["subs1"][fam])
            e9, _ = s2.holdout_error(y, ctx["keep9"], 1, ctx["families"][fam], ctx["subs9"][fam])
            out[f"{label}_err1_{fam}"] = e1
            out[f"{label}_err9_{fam}"] = e9
        for fam in ("tspl", "exp1", "exp2"):
            out[f"{label}_abs_err1_rough_minus_{fam}"] = abs(out[f"{label}_err1_rough"]) - abs(out[f"{label}_err1_{fam}"])
    return out


def predictions(point_all, point_clean, verdicts, by_label, share_pairs, share_var) -> dict:
    try:
        out1 = by_label["VIX1D"]["out"][0]
        in1 = by_label["VIX1D"]["in"][0]
        out9 = by_label["VIX9D"]["out"][0]
        return {
            "R1_VIX1D_median_out_change_below_minus_0.10": out1 < -0.10,
            "R2_VIX1D_median_in_change_positive": in1 > 0,
            "R3_event_pairs_carry_over_1.5x_their_share_of_VIX1D_variance": share_var > 1.5 * share_pairs,
            "R4_clean_pairs_lower_VIX1D_robust_scale": point_clean["plain_sd_VIX1D"] < point_all["plain_sd_VIX1D"],
            "R5_clean_pairs_lower_weekday_centred_ln_R1": point_clean["dow_ln_R1"] < point_all["dow_ln_R1"],
            "R6_clean_weekday_centred_verdict_is_short_end_flattening":
                verdicts["dow"].startswith("short-end flattening"),
            "R7_two_factor_predicts_1d_better_than_rough_clean_weekday_centred":
                abs(point_clean["dow_err1_exp2"]) < abs(point_clean["dow_err1_rough"]),
            "R8_VIX9D_median_out_change_negative_and_smaller_than_VIX1D": out9 < 0 and abs(out9) < abs(out1),
        }
    except (KeyError, ValueError, TypeError, IndexError) as error:
        return {"error": f"a required statistic is missing: {error!r}"}


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

    # Stage 1's primary sample A_daily, and the trading days on which all six indices have a close.
    pairs = vv.make_pairs(series, names, 1, end=date.fromordinal(run_day.toordinal() - 1))
    n = len(pairs)
    common = set(series[names[0]])
    for name in names[1:]:
        common &= set(series[name])
    eve_map = s3.eves(common, s3.event_days())
    labels = s3.classify(pairs, eve_map)
    clean_idx = [i for i, g in enumerate(labels) if g == "clean"]
    counts = {g: labels.count(g) for g in s3.LABELS}
    print("pair labels", counts, flush=True)

    tenors = [dict(vv.TENORS)[name] for name in names]
    cols = [[p[2][j] for p in pairs] for j in range(len(names))]
    weekday = [p[0].weekday() for p in pairs]

    t0 = time.time()
    rough, tspl, expo = job.curves_for(tenors, 1.0)
    two = s2.two_factor_curves(tenors, 1.0)
    print(f"model curves ready in {time.time() - t0:.0f} s", flush=True)
    long_idx = [i for i, t in enumerate(tenors) if t >= 30.0]
    keep1 = [i for i in range(len(tenors)) if i != 0]
    keep9 = list(long_idx)
    families = {"rough": rough, "tspl": tspl, "exp1": expo, "exp2": two}
    ctx = {
        "names": names, "tenors": tenors, "cols": cols, "weekday": weekday,
        "rough": rough, "rough_long": s2.subset_curves(rough, long_idx), "long_idx": long_idx,
        "families": families, "keep1": keep1, "keep9": keep9,
        "subs1": {f: s2.subset_curves(c, keep1) for f, c in families.items()},
        "subs9": {f: s2.subset_curves(c, keep9) for f, c in families.items()},
    }

    point_all = measure(list(range(n)), ctx)
    point_clean = measure(clean_idx, ctx)
    print("point estimates done", flush=True)

    # Descriptive: the event moves themselves.
    by_label = {name: s3.medians_by_label(cols[j], labels) for j, name in enumerate(names)}
    share_pairs = 1.0 - len(clean_idx) / n
    share_var = s3.event_share(cols[0], labels)
    out_by_event = {}
    for ev in ("CPI", "NFP", "FOMC"):
        values = [cols[0][i] for i in range(n) if labels[i] == "out" and ev in eve_map.get(pairs[i][0], ())]
        out_by_event[ev] = {"median": vv.median(values) if values else None, "pairs": len(values)}
    sd_vix1d = {"all": vv.stdev(cols[0]), "clean": vv.stdev([cols[0][i] for i in clean_idx])}

    rng = random.Random(SEED3)
    draws = {}
    t0 = time.time()
    for rep in range(job.BOOT_REPS):
        pos = vv.block_bootstrap_indices(len(clean_idx), job.BLOCK, rng)
        for k, v in measure([clean_idx[j] for j in pos], ctx).items():
            draws.setdefault(k, []).append(v)
        if rep == 9:
            print(f"10 bootstrap draws in {time.time() - t0:.0f} s", flush=True)
    boot = {k: dict(job.interval(v), n=len(v)) for k, v in draws.items()}

    verdicts, readings = {}, {}
    for label in ("plain", "dow"):
        r1, r9 = boot[f"{label}_ln_R1"], boot[f"{label}_ln_R9"]
        verdicts[label] = vv.verdict(r1["p05"], r1["p95"], r9["p05"], r9["p95"])
        c = boot[f"{label}_abs_err1_rough_minus_exp2"]
        readings[label] = s3.comparison_reading(c["p05"], c["p95"])

    results = {
        "prereg": "PREREG3.md", "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources, "pairs": n, "first_day": pairs[0][0].isoformat(), "last_day": pairs[-1][1].isoformat(),
        "label_counts": counts, "event_pair_share": share_pairs, "VIX1D_event_variance_share": share_var,
        "median_change_by_label": {name: {g: {"median": m, "pairs": c} for g, (m, c) in d.items()}
                                   for name, d in by_label.items()},
        "VIX1D_out_median_by_event": out_by_event, "VIX1D_sd_all_vs_clean": sd_vix1d,
        "eves_used": len(eve_map),
        "point_all_pairs": point_all, "point_clean": point_clean, "bootstrap_clean": boot,
        "bootstrap_reps": job.BOOT_REPS, "block": job.BLOCK, "seed": SEED3,
        "verdict_clean_plain": verdicts["plain"], "verdict_clean_weekday_centred": verdicts["dow"],
        "one_day_holdout_reading_clean": readings,
        "predictions": predictions(point_all, point_clean, verdicts, by_label, share_pairs, share_var),
    }
    results["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.out_dir, "results3.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)

    lines = ["# volvol-term stage 3: scheduled macro releases at the one-day end", "",
             f"Run {results['run_utc']}. Pre-registration: PREREG3.md. {n} daily pairs, {results['first_day']} to "
             f"{results['last_day']}; {len(clean_idx)} clean.", "",
             f"- Verdict, clean pairs, weekday-centred (primary for stage 3): **{verdicts['dow']}**",
             f"- Verdict, clean pairs, plain: **{verdicts['plain']}**",
             f"- One-day hold-out, clean, weekday-centred: **{readings['dow']}**",
             f"- One-day hold-out, clean, plain: **{readings['plain']}**", "",
             f"Pair labels: {counts}. Event pairs are {fmt(share_pairs)} of all pairs and carry {fmt(share_var)} of "
             f"VIX1D's squared daily log changes. VIX1D standard deviation: all {fmt(sd_vix1d['all'], 4)}, clean "
             f"{fmt(sd_vix1d['clean'], 4)}.", "",
             "## Median daily log change by label", "", "| index | in | out | both | clean |", "|---|---|---|---|---|"]
    for name in names:
        d = by_label[name]
        lines.append(f"| {name} | " + " | ".join(
            f"{fmt(d[g][0], 4)} (n={d[g][1]})" if g in d else "-" for g in s3.LABELS) + " |")
    lines += ["", "VIX1D median change on out pairs by event: " +
              "; ".join(f"{ev} {fmt(v['median'], 4)} (n={v['pairs']})" for ev, v in out_by_event.items()), "",
              "## Point estimates (all pairs vs clean) and bootstrap intervals (clean)", "",
              "| statistic | all pairs | clean | p05 | p50 | p95 |", "|---|---|---|---|---|---|"]
    for k in sorted(boot):
        b = boot[k]
        lines.append(f"| {k} | {fmt(point_all.get(k), 4)} | {fmt(point_clean.get(k), 4)} | {fmt(b['p05'], 4)} | "
                     f"{fmt(b['p50'], 4)} | {fmt(b['p95'], 4)} |")
    lines += ["", "## Pre-registered predictions", ""]
    for k, v in results["predictions"].items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(args.out_dir, "results3.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
