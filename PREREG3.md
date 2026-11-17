# Pre-registration, stage 3: scheduled macro releases at the one-day end

Registered 2026-11-17, before any result of stage 1 (PREREG.md), stage 2 (PREREG2.md) or stage 3 existed: both earlier
jobs were still waiting in the compute queue when this file was written. The commit timestamp is the registration
time. Nothing below will be edited after results exist; later changes go in dated addenda at the end. PREREG.md and
PREREG2.md are unchanged.

Author: MV2 of Munim, Inc. (mv2@munimtech.com). The design, code and text were made with AI agents working in his name.

## Why a stage 3

The test asks whether the rough power law for vol-of-vol carries on down to a one-day tenor. The models in stages 1
and 2 assume a flat forward-variance curve that diffuses. Scheduled macro releases break that assumption most at the
one-day end:

- VIX1D at the close of day d prices the variance of the next session. If that session holds a CPI or jobs report
  (8:30 ET, before the open) or an FOMC statement (14:00 ET), the index carries the known event variance on the eve,
  and drops it once the release is out. The move is the one-day window rolling across a spike in the forward-variance
  curve. It is not diffusive vol-of-vol, and a rough or Markov kernel does not describe it.
- The size is large. A public working project (github.com/unaifmm/options-event-risk, not peer-reviewed) reports
  VIX1D falling on average from 17.4 to 14.1 across CPI days (a log change of about -0.21) and from 17.5 to 14.6
  across payroll days, against 13.6 to 13.8 on other days. Albers and Kestner (Finance Research Letters 2024) note that
  the largest VIX1D spikes in their sample line up with CPI releases and FOMC press conferences. Implied volatility
  rising before scheduled news and falling after it is long known for the 30-day VIX (Nikkinen and Sahlstrom 2004).
- Longer windows dilute the effect: one event is about 1/9 of VIX9D's window and about 1/30 of VIX's.

Stage 1 drops FOMC days in one sensitivity sample only and relies on the robust scale against CPI and payrolls. With
about three scheduled releases a month, each touching up to two daily pairs, event pairs are far from rare. A robust
scale does not remove them. Stage 3 removes them on purpose and tests the one-day end again.

## Data and calendars (fixed)

- The same six Cboe histories and the same daily pairs as stage 1's primary sample A_daily, fetched again by `job3.py`.
- Event days, in `stage3.py`: the 57 CPI releases and 57 Employment Situation releases from February 2022 to
  November 2026, from the BLS release archives (www.bls.gov/bls/news-release/cpi.htm and empsit.htm). The October
  2025 reports were never published because of the 2025 lapse in appropriations. The September 2025 reports came out
  late, on 2025-10-24 (CPI) and 2025-11-20 (jobs). Also the 40 FOMC statement days of stage 1. Unscheduled events are
  not modelled.
- The eve of an event day E is the last trading day in the data (all six indices have a close) strictly before E.

## Pair labels (fixed)

A daily pair (d1, d2) of consecutive calendar days is labelled **in** if d2 is an eve (an event enters the one-day
window), **out** if d1 is an eve (the event leaves it), **both** if both hold, and **clean** otherwise. The
**clean sample** is the clean pairs only.

## Measurements (fixed)

- On the clean sample, exactly as in stage 2: plain and weekday-centred robust scales of all six indices; the long-end
  rough extrapolation (fitted to 30, 93, 184 and 365 days) and ln R1, ln R9; the one-day hold-out (fit on 9 days to
  1 year, predict 1 day) and the nine-day hold-out, for the rough, TSPL, one-factor and two-factor exponential families
  on stage 1's and stage 2's grids.
- The same point estimates on all pairs, for comparison.
- Descriptive: the median daily log change of every index on in, out, both and clean pairs; VIX1D's median change on
  out pairs by event type; the share of pairs that are not clean; the share of VIX1D's sum of squared deviations
  (from the overall mean daily change) that falls on pairs that are not clean; the plain standard deviation of VIX1D on
  all and on clean pairs.
- Uncertainty: moving-block bootstrap over the clean pairs in time order, block 20, 1,000 draws, seed 20261119; 90%
  intervals are the 5th and 95th percentiles. Removing event pairs breaks a few blocks across gaps; we accept that as an
  approximation, as stage 1 did for its FOMC sensitivity sample.

## Predictions (made before seeing any result of any stage)

- **R1.** The median VIX1D log change on out pairs is below -0.10.
- **R2.** The median VIX1D log change on in pairs is above 0.
- **R3.** Pairs that are not clean carry more than 1.5 times their share of pairs of VIX1D's squared deviations.
- **R4.** The plain robust scale of VIX1D is lower on clean pairs than on all pairs.
- **R5.** The weekday-centred ln R1 is lower on clean pairs than on all pairs (point estimates).
- **R6.** The clean, weekday-centred verdict (stage-1 rule) is "short-end flattening".
- **R7.** On clean pairs, weekday-centred, the two-factor exponential predicts the held-out 1-day value with a smaller
  absolute error than the rough kernel.
- **R8.** The median VIX9D log change on out pairs is negative and smaller in size than VIX1D's.

## How the results will be read (fixed)

- **Primary for stage 3:** the stage-1 verdict rule applied to the 90% intervals of ln R1 and ln R9 on the clean
  sample, weekday-centred. This sample removes both known artefacts at the one-day end: the methodology-driven weekday
  pattern and the scheduled event roll. It is a pre-registered robustness check of stage 1, not a replacement for it.
- **If stage 1 and stage 3 agree,** the one-day answer does not come from scheduled events. **If they disagree,**
  `RESULTS.md` reports the disagreement as the main finding about the one-day end, namely that the scheduled release
  calendar drives the one-day vol-of-vol, and gives both verdicts.
- **Model comparison:** the 90% interval of |error rough| - |error two-factor| at the held-out 1-day tenor on the
  clean, weekday-centred sample, read with the stage-2 rule (above 0: two-factor better; below 0: rough better;
  otherwise no clear winner).

Every prediction will be graded and published, right or wrong, in `RESULTS.md`, together with stages 1 and 2.

## Known limitations (fixed in advance)

- As in stages 1 and 2: first-order Gaussian model, calendar-day time, one 4.5-year regime (2022-2026).
- Only three release types. PCE, GDP, retail sales, large earnings (such as NVIDIA) and unscheduled news are not
  removed. VIX9D and VIX also roll events in and out of their windows. They are not cleaned, because one event weighs
  much less in a longer window.
- The calendars are as published by BLS and the Federal Reserve. A release moved at short notice is taken at the date
  it actually came out.
- Clean pairs are fewer (roughly two thirds of A_daily), so the intervals are wider than in stage 1.

This is research on public market data. It is not investment advice.
