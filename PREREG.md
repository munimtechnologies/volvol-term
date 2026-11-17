# Pre-registration: is volatility rough all the way down to one day?

Registered 2026-11-17, before any of the analysis below was run on data. This file is committed before the job runs;
the commit timestamp is the registration time. Nothing below will be edited after results exist; any later change
goes in a dated addendum at the end.

Author: MV2 of Munim, Inc. (mv2@munimtech.com). The design, code and text were made with AI agents working in his name.

## Question

"Rough volatility" (Gatheral, Jaisson and Rosenbaum, *Quantitative Finance* 2018) models the log of instantaneous
variance with a fractional kernel k(x) = x^(H - 1/2), H about 0.1. One of its sharpest predictions is a **power law
in maturity**: the volatility of variance-swap rates (vol-of-vol) of tenor Delta scales like Delta^(H - 1/2), and keeps
growing as the tenor shrinks to zero. Practitioners have long seen such a power law between about 1 month and a few
years: Bergomi (Smile Dynamics II and *Stochastic Volatility Modeling*, 2016) reports exponents of about 0.3 to 0.6 for
equity indices, which is one of the motivations for rough models.

What happens below one month is contested. For the other famous power law of rough volatility, the at-the-money skew,
Guyon and El Amrani (*Risk*, 2023, "Does the term-structure of equity at-the-money skew really follow a power law?")
found the fit breaks down below about three weeks: the short-end skew grows more slowly than the power law, and
shifted or capped power laws (finite at zero maturity) fit better. Other studies (Rømer 2022; "Volatility models in
practice: rough, path-dependent or Markovian?", 2024) also find rough models' short-end shape too steep. Bandi, Fusari,
Gazzani and Renò (2026) report strong oscillations in the ultra-short-term ATM volatility term structure.

The vol-of-vol term structure is a second, independent observable of the same kernel, and it can now be measured
from public data down to one day: Cboe publishes VIX1D (1-day, history from May 2022), VIX9D, VIX, VIX3M, VIX6M and
VIX1Y. We found no study that measures the vol-of-vol term structure from 1 day to 1 year and tests the rough
extrapolation at the one-day end. (Earlier work estimates the time-series roughness of single implied-volatility
series, e.g. Le Floc'h, "Roughness of the implied volatility"; Da Fonseca and Zhang, "Volatility of volatility is
(also) rough"; that is a different measurement.)

**The "wow", if it works:** a public-data answer to whether volatility is rough at the shortest horizons. Either the
vol-of-vol power law fitted between 1 month and 1 year carries on down to one day (strong new support for rough
volatility from an observable that was not used to build it), or it breaks below a few weeks, as Guyon and El Amrani
found for the skew, which would be a second, independent sign that the kernel is not a pure power law at short lags.

## Data (all public; fetched by `job.py`)

Cboe daily index histories, `cdn-api.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv`, close column,
for NAME = VIX1D (tenor 1 day), VIX9D (9), VIX (30), VIX3M (93), VIX6M (184), VIX1Y (365). SHA-256 hashes of every
downloaded file go in `results.json`. Only derived statistics are published, never Cboe's raw data.

## Measurement (fixed)

- Change: r = ln(close on day d2 / close on day d1). Daily step: d2 = d1 + 1 calendar day (Mon-Tue to Thu-Fri only;
  weekends and holidays never enter). Weekly step: d2 = d1 + 7 calendar days. All indices must have a positive close
  on both days. Pairs that end on the run day are dropped.
- Vol-of-vol of each index: **robust scale = 1.4826 x median absolute deviation** of its changes (primary, chosen to
  limit the effect of scheduled-event spikes); the plain standard deviation is reported too.
- Lag-1 autocorrelation: Pearson correlation of a change with the next change that starts where it ended.

## Samples (fixed)

- **A_daily (primary):** all six indices, daily step, every date where all six exist (VIX1D's history starts
  2022-05-13).
- A_daily_noFOMC: as A_daily, dropping every pair that touches an FOMC statement day or the weekday before it
  (40 statement days 2022-2026, from federalreserve.gov).
- A_weekly: as A_daily with the 7-day step.
- A_daily_live: as A_daily from 2023-04-24 (VIX1D's launch; earlier values are Cboe's back-calculation). Reported
  only.
- B_daily_2011: the five tenors from 9 days to 1 year, daily step, from 2011-01-01.

## Models (fixed)

The log forward variance moves as Z_t(u) = integral_{-inf}^{t} k(u - s) dW_s; the log of an index squared is the
average of Z_t(u) over its window [t, t + Delta]. To first order the change over a step delta is Gaussian with
variance proportional to I(Delta, delta), computed exactly by numerical integration (`volvol.moments`). This matters
at the one-day tenor, where the window rolls over completely in one step. The level of vol-of-vol is a free constant
in every fit. Kernels:

- **rough:** k(x) = x^(H - 1/2); H on the grid 0.01, 0.02, ..., 0.99.
- **TSPL (time-shifted power law):** k(x) = (x + eps)^(H - 1/2); the same H grid, eps in {0, 0.25, 0.5, 1, 2, 4, 8, 16} days.
- **exponential (one-factor Markov):** k(x) = exp(-kappa x); kappa = 10^(-4 + 0.025 i), i = 0..200, per day.

Fits minimize the sum of squared differences between ln(robust scale) and the model curve over the tenors. **Long-end
rough fit:** the rough model fitted to the 30, 93, 184 and 365-day tenors only, then extrapolated to 1 and 9 days:
R1 = observed / extrapolated at 1 day, R9 at 9 days.

Uncertainty: moving-block bootstrap over the pairs (block 20 pairs, 1,000 draws, seed 20261117 + sample number),
refitting everything in each draw; 90% intervals are the 5th and 95th percentiles.

## Predictions (made before seeing any result)

Our background knowledge is the literature above, which leans towards a power law above one month and a break below.

- **P1.** In A_daily, the robust vol-of-vol falls strictly from each tenor to the next longer one.
- **P2.** The long-end rough fit in A_daily gives 0.02 <= H <= 0.30.
- **P3.** In A_daily, R1 < 1: one-day vol-of-vol is below the rough extrapolation from the long end.
- **P4.** In A_daily, the best TSPL fit has eps > 0.
- **P5.** In A_daily, the rough model fits the six tenors better (lower SSE) than the one-factor exponential.
- **P6.** In A_daily, the lag-1 autocorrelation of daily changes is negative for VIX1D and for VIX9D.
- **P7.** The rough H fitted to all five tenors of B_daily_2011 is within 0.10 of the long-end H of P2.
- **P8.** ln R1 has the same sign in A_daily, A_daily_noFOMC and A_weekly.

## How the result will be read (fixed; primary sample A_daily, 90% bootstrap intervals)

- **"Short-end flattening":** the interval for ln R1 lies below 0. One-day vol-of-vol is smaller than the rough
  power law predicts from the long end. Scheduled events (FOMC, CPI, payrolls) and index noise inflate short-tenor
  changes, so they bias R1 *upwards*; a flattening verdict is therefore conservative.
- **"Short end steeper":** the interval lies above 0. Because of the upward biases just named, this is read as
  inconclusive about roughness unless it also holds in A_daily_noFOMC and A_weekly.
- **"Power law holds down to one day":** the intervals for ln R1 and ln R9 both contain 0.
- Otherwise: "mixed", reported with the numbers.

Every prediction will be graded and published, right or wrong, in `RESULTS.md` next to this file.

## Known limitations (fixed in advance)

- The VIX indices approximate variance-swap volatilities (strike discretization and truncation). VIX1D is built from
  options with one or two days to expiry, and its tenor is about one trading day.
- First-order Gaussian model with a flat forward-variance curve and constant vol-of-vol. Real vol-of-vol rises with
  the volatility level; using unconditional robust scales across tenors measured on the same days limits the impact on
  the *shape*.
- Calendar-day time throughout; weekends are excluded rather than modelled.
- CPI and payroll release days are not excluded (only FOMC days in the sensitivity sample); the robust scale is the
  main defence.
- About 4.5 years of VIX1D data (2022-2026): one volatility regime sample.

This is research on public market data. It is not investment advice.
