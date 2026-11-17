# Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
"""Offline tests for stage3.py and job3.py helpers (no network). Expected values are worked out in the comments."""

import unittest
from datetime import date

import job3
import stage3 as s3
import volvol as vv


class CalendarTest(unittest.TestCase):
    def test_counts_and_order(self):
        # February 2022 to November 2026 is 58 releases each, minus the unpublished October 2025 one.
        self.assertEqual(len(s3.CPI_RELEASE_DAYS), 57)
        self.assertEqual(len(s3.EMPSIT_RELEASE_DAYS), 57)
        for days in (s3.CPI_RELEASE_DAYS, s3.EMPSIT_RELEASE_DAYS):
            self.assertTrue(all(a < b for a, b in zip(days, days[1:])))

    def test_all_weekdays(self):
        self.assertTrue(all(d.weekday() < 5 for d in s3.event_days()))

    def test_known_days(self):
        ev = s3.event_days()
        self.assertEqual(ev[date(2024, 6, 12)], ("CPI", "FOMC"))  # CPI and an FOMC statement on the same day
        self.assertEqual(ev[date(2025, 10, 24)], ("CPI",))  # September 2025 CPI, delayed by the shutdown
        self.assertEqual(ev[date(2025, 11, 20)], ("NFP",))
        self.assertEqual(ev[date(2024, 1, 31)], ("FOMC",))
        # No jobs report between the September 2025 release and the delayed one.
        self.assertFalse(any("NFP" in v for d, v in ev.items() if date(2025, 9, 6) <= d <= date(2025, 11, 19)))


class EveTest(unittest.TestCase):
    def test_eves(self):
        # Trading days Mon 2024-01-08 to Fri 01-12 and Tue 01-16 (01-15 is a holiday).
        days = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12),
                date(2024, 1, 16)]
        events = {date(2024, 1, 10): ("CPI",), date(2024, 1, 16): ("NFP",), date(2024, 1, 8): ("FOMC",),
                  date(2024, 2, 1): ("CPI",), date(2024, 1, 15): ("FOMC",)}
        # 01-10 -> eve 01-09; 01-16 and 01-15 -> eve 01-12 (merged); 01-08 is the first day and 02-01 is past the
        # data, so both are skipped.
        self.assertEqual(s3.eves(set(days), events),
                         {date(2024, 1, 9): ("CPI",), date(2024, 1, 12): ("FOMC", "NFP")})

    def test_classify(self):
        d = lambda k: date(2024, 1, k)
        pairs = [(d(8), d(9), []), (d(9), d(10), []), (d(10), d(11), []), (d(11), d(12), [])]
        self.assertEqual(s3.classify(pairs, {d(9), d(12)}), ["in", "out", "clean", "in"])
        self.assertEqual(s3.classify(pairs, {d(9), d(10)}), ["in", "both", "out", "clean"])


class StatsTest(unittest.TestCase):
    def test_event_share(self):
        # Mean 0; squares 1, 1, 9, 9; the two event pairs hold 18 of 20.
        self.assertAlmostEqual(s3.event_share([1.0, -1.0, 3.0, -3.0], ["clean", "clean", "out", "in"]), 0.9)

    def test_medians_by_label(self):
        got = s3.medians_by_label([1.0, 5.0, 3.0, -2.0], ["clean", "clean", "clean", "out"])
        self.assertEqual(got, {"clean": (3.0, 3), "out": (-2.0, 1)})

    def test_comparison_reading(self):
        self.assertTrue(s3.comparison_reading(0.01, 0.2).startswith("the two-factor"))
        self.assertTrue(s3.comparison_reading(-0.3, -0.01).startswith("the rough kernel"))
        self.assertEqual(s3.comparison_reading(-0.1, 0.1), "no clear winner at the one-day end")


class JobHelperTest(unittest.TestCase):
    def test_log_scales(self):
        cols = [[float(v) for v in range(25)]]
        # Median 12, absolute deviations 0..12 (twice each except 0) -> median deviation 6.
        got = job3.log_scales(cols, list(range(25)))
        self.assertAlmostEqual(got[0], __import__("math").log(6 * 1.4826))
        self.assertIsNone(job3.log_scales(cols, list(range(19))))

    def test_predictions_missing(self):
        self.assertIn("error", job3.predictions({}, {}, {"dow": ""}, {}, 0.3, 0.5))

    def test_verdict_prefix(self):
        # R6 relies on this exact prefix of volvol.verdict.
        self.assertTrue(vv.verdict(-0.5, -0.1, -0.2, 0.1).startswith("short-end flattening"))


if __name__ == "__main__":
    unittest.main()
