# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Offline tests for volvol.py (no network). Expected values are worked out in the comments or in closed form."""

import math
import random
import unittest
from datetime import date

import volvol as vv


class DataTest(unittest.TestCase):
    def test_parse_closes(self):
        text = chr(0xFEFF) + "DATE,OPEN,HIGH,LOW,CLOSE\n05/13/2022,1,1,1,33.63\n05/16/2022,1,1,1,\nbad,1,1,1,2\n05/17/2022,1,1,1,0\n"
        self.assertEqual(vv.parse_closes(text), {date(2022, 5, 13): 33.63})

    def test_event_window(self):
        # 2026-01-28 is a Wednesday; 2026-11-16 is a Monday, so its previous weekday is Friday 11-13.
        self.assertEqual(vv.event_window_days([date(2026, 1, 28)]), {date(2026, 1, 28), date(2026, 1, 27)})
        self.assertEqual(vv.event_window_days([date(2026, 11, 16)]), {date(2026, 11, 16), date(2026, 11, 13)})

    def test_fomc_list(self):
        self.assertEqual(len(vv.FOMC_STATEMENT_DAYS), 40)
        self.assertTrue(all(d.weekday() < 5 for d in vv.FOMC_STATEMENT_DAYS))

    def test_make_pairs(self):
        mon, tue, wed, fri, mon2 = date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11), date(2026, 11, 13), date(2026, 11, 16)
        series = {
            "a": {mon: 10.0, tue: 20.0, wed: 20.0, fri: 10.0, mon2: 10.0},
            "b": {mon: 5.0, tue: 5.0, fri: 5.0, mon2: 5.0},  # no Wednesday
        }
        pairs = vv.make_pairs(series, ["a", "b"], 1)
        self.assertEqual(len(pairs), 1)  # Tue-Wed lacks b; Fri-Sat and Mon-Tue (17th) don't exist
        self.assertEqual(pairs[0][:2], (mon, tue))
        self.assertAlmostEqual(pairs[0][2][0], math.log(2.0))
        self.assertEqual(pairs[0][2][1], 0.0)
        self.assertEqual(vv.make_pairs(series, ["a", "b"], 1, exclude={tue}), [])
        weekly = vv.make_pairs(series, ["a", "b"], 7)
        self.assertEqual([p[:2] for p in weekly], [(mon, mon2)])
        self.assertEqual(vv.make_pairs(series, ["a", "b"], 1, start=tue), [])


class StatsTest(unittest.TestCase):
    def test_median_and_scale(self):
        self.assertEqual(vv.median([4.0, 1.0, 3.0, 2.0]), 2.5)
        # median 3, absolute deviations 2,1,0,1,97 -> median 1
        self.assertAlmostEqual(vv.robust_scale([1.0, 2.0, 3.0, 4.0, 100.0]), 1.4826)
        self.assertAlmostEqual(vv.stdev([1.0, 2.0, 3.0, 4.0]), math.sqrt(5.0 / 3.0))

    def test_lag1_autocorr(self):
        # Changes 1, 2, 1, 2 along a chain: consecutive couples (1,2), (2,1), (1,2) are perfectly anti-correlated.
        pairs = [(0, 1, [1.0]), (1, 2, [2.0]), (2, 3, [1.0]), (3, 4, [2.0])]
        self.assertAlmostEqual(vv.lag1_autocorr(pairs, 0), -1.0)
        self.assertTrue(math.isnan(vv.lag1_autocorr(pairs[:2], 0)))

    def test_percentile(self):
        self.assertEqual(vv.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5), 3.0)
        self.assertAlmostEqual(vv.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.05), 1.2)

    def test_block_bootstrap(self):
        idx = vv.block_bootstrap_indices(10, 3, random.Random(1))
        self.assertEqual(len(idx), 10)
        self.assertTrue(all(0 <= i < 10 for i in idx))
        with self.assertRaises(ValueError):
            vv.block_bootstrap_indices(2, 3, random.Random(1))


class MomentsTest(unittest.TestCase):
    def test_brownian_kernel(self):
        # H = 1/2: constant kernel, so the log index is a Brownian motion: variance delta, independent steps.
        I, C1 = vv.moments(vv.power_G(0.5), 30.0, 1.0, vv.power_tail(0.5))
        self.assertAlmostEqual(I, 1.0, places=8)
        self.assertAlmostEqual(C1, 0.0, places=8)

    def test_exponential_matches_closed_form(self):
        for kappa, Delta, delta in ((0.1, 10.0, 1.0), (0.02, 1.0, 1.0), (0.5, 93.0, 7.0)):
            I, C1 = vv.moments(vv.exp_G(kappa), Delta, delta)
            I_exact, C1_exact = vv.exp_moments_closed(kappa, Delta, delta)
            self.assertAlmostEqual(I / I_exact, 1.0, places=4)
            self.assertAlmostEqual(C1 / C1_exact, 1.0, places=4)
            self.assertLess(C1_exact, 0.0)

    def test_rough_power_law_for_short_steps(self):
        # For delta << Delta the rough kernel gives I ~ delta * Delta^(2H - 1) / (H + 1/2)^2.
        H, delta, Delta = 0.1, 1e-5, 100.0
        I, _ = vv.moments(vv.power_G(H), Delta, delta, vv.power_tail(H))
        expected = delta * Delta ** (2 * H - 1) / (H + 0.5) ** 2
        self.assertAlmostEqual(I / expected, 1.0, places=3)

    def test_rough_vol_of_vol_falls_with_tenor(self):
        values = [vv.moments(vv.power_G(0.1), d, 1.0, vv.power_tail(0.1))[0] for d in (1.0, 9.0, 30.0, 365.0)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))


class FitTest(unittest.TestCase):
    def test_fit_curves(self):
        key, level, sse = vv.fit_curves([1.0, 2.0, 3.0], {"a": [0.0, 1.0, 2.0], "b": [0.0, 0.0, 0.0]})
        self.assertEqual(key, "a")
        self.assertAlmostEqual(level, 1.0)
        self.assertAlmostEqual(sse, 0.0)

    def test_fit_all_extrapolation(self):
        tenors = [1.0, 30.0, 93.0]
        rough = {0.1: [3.0, 2.0, 1.0], 0.2: [2.0, 2.0, 1.0]}
        # Long end (30, 93) is y = 5, 4 -> both curves fit it exactly with level 3; the first sorted key (0.1) wins.
        # Extrapolated 1-day value: 3 + 3 = 6; observed 5.5 -> log ratio -0.5.
        y = [5.5, 5.0, 4.0]
        out = vv.fit_all(y, tenors, rough, {(0.1, 0.0): rough[0.1]}, {1.0: [0.0, 0.0, 0.0]})
        self.assertEqual(out["long_H"], 0.1)
        self.assertAlmostEqual(out["log_ratio"][1.0], -0.5)
        self.assertEqual(set(out["log_ratio"]), {1.0})

    def test_verdict(self):
        self.assertTrue(vv.verdict(-0.4, -0.1, -0.2, 0.1).startswith("short-end flattening"))
        self.assertTrue(vv.verdict(0.1, 0.4, -0.2, 0.1).startswith("short end steeper"))
        self.assertEqual(vv.verdict(-0.1, 0.1, -0.2, 0.1), "power law holds down to one day")
        self.assertTrue(vv.verdict(-0.1, 0.1, 0.1, 0.2).startswith("mixed"))


if __name__ == "__main__":
    unittest.main()
