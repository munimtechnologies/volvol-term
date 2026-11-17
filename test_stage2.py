# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Offline tests for stage2.py (no network). Expected values are worked out in the comments or in closed form."""

import math
import unittest

import stage2 as s2
import volvol as vv


class TwoFactorTest(unittest.TestCase):
    def test_cross_equals_one_factor_closed_form(self):
        for k, Delta, delta in ((0.1, 10.0, 1.0), (0.02, 1.0, 1.0), (0.5, 93.0, 7.0), (3.0, 365.0, 1.0)):
            self.assertAlmostEqual(s2.exp_cross(k, k, Delta, delta) / vv.exp_moments_closed(k, Delta, delta)[0],
                                   1.0, places=12)

    def test_cross_symmetric_and_positive(self):
        a = s2.exp_cross(0.3, 0.01, 30.0, 1.0)
        self.assertAlmostEqual(a, s2.exp_cross(0.01, 0.3, 30.0, 1.0), places=15)
        self.assertGreater(a, 0.0)

    def test_two_factor_limits(self):
        Delta, delta = 30.0, 1.0
        one = vv.exp_moments_closed(0.2, Delta, delta)[0]
        slow = vv.exp_moments_closed(0.01, Delta, delta)[0]
        # theta = 0: only the fast factor; theta = 1: only the slow factor.
        self.assertAlmostEqual(s2.two_factor_I(0.2, 0.01, 0.0, 0.5, Delta, delta) / one, 1.0, places=12)
        self.assertAlmostEqual(s2.two_factor_I(0.2, 0.01, 1.0, 0.5, Delta, delta) / slow, 1.0, places=12)
        # Equal rates and perfectly correlated factors: weights (1 - theta) + theta = 1, so the one-factor value.
        self.assertAlmostEqual(s2.two_factor_I(0.2, 0.2, 0.3, 1.0, Delta, delta) / one, 1.0, places=12)
        # Equal rates, independent factors: ((1 - theta)^2 + theta^2) times the one-factor value; theta 0.3 -> 0.58.
        self.assertAlmostEqual(s2.two_factor_I(0.2, 0.2, 0.3, 0.0, Delta, delta) / one, 0.58, places=12)

    def test_two_factor_matches_numerical_integral(self):
        # rho = 1 is a single shock through the kernel (1 - theta) e^(-k1 x) + theta e^(-k2 x): integrate it with
        # the stage-1 numerical integrator and compare with the closed form.
        k1, k2, theta = 0.5, 0.01, 0.4
        G1, G2 = vv.exp_G(k1), vv.exp_G(k2)

        def G(x):
            return (1.0 - theta) * G1(x) + theta * G2(x)

        for Delta, delta in ((1.0, 1.0), (30.0, 1.0), (93.0, 7.0)):
            numeric, _ = vv.moments(G, Delta, delta)
            exact = s2.two_factor_I(k1, k2, theta, 1.0, Delta, delta)
            self.assertAlmostEqual(numeric / exact, 1.0, places=4)

    def test_grid(self):
        self.assertEqual(len(s2.K1_GRID), 29)
        self.assertAlmostEqual(s2.K1_GRID[-1], 10.0)
        self.assertAlmostEqual(s2.K2_GRID[0], 10 ** -4.5)
        self.assertEqual(s2.THETA_GRID, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        curves = s2.two_factor_curves([1.0, 365.0], 1.0)
        # k2 < k1 keeps min(29, i + 16) slow rates for the i-th fast rate: 286 + 464 = 750 pairs, times 9 x 3.
        self.assertEqual(len(curves), 750 * 27)
        self.assertTrue(all(k[1] < k[0] for k in curves))
        self.assertTrue(all(math.isfinite(v) for m in curves.values() for v in m))
        # Vol-of-vol falls with tenor for every positive-weight mix of decaying factors.
        self.assertTrue(all(m[0] > m[1] for m in curves.values()))


class StatsTest(unittest.TestCase):
    def test_weekday_centred_scale(self):
        values = [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
        groups = [0, 0, 0, 1, 1, 1]
        # Group medians 2 and 12; deviations 1, 0, 1, 1, 0, 1 -> median of (0, 0, 1, 1, 1, 1) is 1.
        self.assertAlmostEqual(s2.weekday_centred_scale(values, groups), 1.4826)
        # Ignoring the groups: median 7, deviations 6, 5, 4, 4, 5, 6 -> median 5.
        self.assertAlmostEqual(vv.robust_scale(values), 5 * 1.4826)
        self.assertEqual(s2.group_medians(values, groups), {0: 2.0, 1: 12.0})

    def test_single_group_equals_robust_scale(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        self.assertAlmostEqual(s2.weekday_centred_scale(values, [0] * 5), vv.robust_scale(values))

    def test_terciles(self):
        values = [float(v) for v in range(1, 10)]
        cuts = s2.tercile_cutoffs(values)
        # Positions 8/3 and 16/3 in the sorted list: 3 + 2/3 and 6 + 1/3.
        self.assertAlmostEqual(cuts[0], 3.0 + 2.0 / 3.0)
        self.assertAlmostEqual(cuts[1], 6.0 + 1.0 / 3.0)
        labels = [s2.tercile_label(v, cuts) for v in values]
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1, 2, 2, 2])


class FitTest(unittest.TestCase):
    def test_holdout_error(self):
        curves = {"a": [0.0, 1.0, 2.0], "b": [0.0, 0.0, 2.0]}
        y = [9.5, 11.0, 12.0]
        keep = [1, 2]
        # a fits (11, 12) exactly with level 10 -> predicts 10 at index 0; b: level 10.5, SSE 0.5.
        err, key = s2.holdout_error(y, keep, 0, curves, s2.subset_curves(curves, keep))
        self.assertEqual(key, "a")
        self.assertAlmostEqual(err, -0.5)

    def test_long_end(self):
        tenors = [1.0, 9.0, 30.0, 93.0]
        rough = {0.1: [3.0, 2.5, 2.0, 1.0], 0.2: [2.0, 2.0, 2.0, 1.0]}
        long_idx = [2, 3]
        y = [5.5, 5.4, 5.0, 4.0]
        # Both keys fit the long end (5, 4) exactly with level 3; the first sorted key (0.1) is kept.
        # Extrapolated: 1 d -> 6, 9 d -> 5.5; log ratios -0.5 and -0.1.
        H, lr = s2.long_end(y, tenors, rough, s2.subset_curves(rough, long_idx), long_idx)
        self.assertEqual(H, 0.1)
        self.assertEqual(set(lr), {1.0, 9.0})
        self.assertAlmostEqual(lr[1.0], -0.5)
        self.assertAlmostEqual(lr[9.0], -0.1)


if __name__ == "__main__":
    unittest.main()
