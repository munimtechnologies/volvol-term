# Pre-registration, stage 2: weekday seasonality, volatility regime and a two-factor Markov rival

Registered 2026-11-17, before any result of stage 1 (PREREG.md) or stage 2 existed: the stage-1 job was still waiting
in the compute queue when this file was written. The commit timestamp is the registration time. Nothing below will be
edited after results exist; later changes go in dated addenda at the end. PREREG.md is unchanged.

Author: MV2 of Munim, Inc. (mv2@munimtech.com). The design, code and text were made with AI agents working in his name.

## Why a stage 2

Stage 1 asks whether the rough-volatility power law for vol-of-vol, fitted between 1 month and 1 year, carries on
down to 1 day. Three things could make its answer misleading, and each is tested here.

1. **Weekday seasonality in the short indices.** Albers and Kestner ("The daily rise and fall of the VIX1D: causes and
   solutions of its overnight bias", *Finance Research Letters*, 2024; SSRN 4650419) report mean close-to-close log
   changes of VIX1D of about -0.087 into Monday, +0.071 into Tuesday, -0.020 into Wednesday, +0.067 into Thursday
   and -0.057 into Friday (May 2022 to November 2023). They trace it to the index methodology (business-time weighting
   of the next-day options), and find smaller weekday effects in VIX9D and VIX. A fixed weekly zig-zag is not
   volatility of volatility, but it widens the spread of daily changes, so it inflates the measured one-day
   vol-of-vol and pushes stage 1's R1 upwards.
2. **Volatility regime.** Rough Bergomi-type models make the *shape* of the log vol-of-vol term structure independent
   of the volatility level. If the short-end shape changes with the VIX level, a single unconditional answer hides it.
3. **The Markovian alternative.** Practitioners mostly use the two-factor Bergomi model, whose kernel is a mix of two
   exponentials and is finite at zero lag. Abi Jaber and Li ("Volatility models in practice: rough, path-dependent or
   Markovian?", *Mathematical Finance* 35(4), 2025; arXiv 2401.03345) find that a two-factor Markovian model fits
   SPX smiles from one week to three years better than its rough counterpart, with three to four parameters. Their
   test uses option smiles; this one uses the time-series volatility of variance-swap rates, a different observable.
   Stage 1 compares the rough kernel only with one exponential, which is a weak rival.

## Data and sample (fixed)

The same six Cboe daily histories as stage 1, fetched again by `job2.py`. The sample is stage 1's primary sample
A_daily: consecutive calendar days (Mon-Tue to Thu-Fri) on which all six indices have a positive close, ending the day
before the run. Changes are daily log changes; the weekday of a pair is the weekday of its first day.

## Measurements (fixed)

- **Plain:** robust scale = 1.4826 x median absolute deviation from the overall median (as in stage 1).
- **Weekday-centred:** each change minus the median change of its own weekday (four groups), then 1.4826 x median of
  the absolute centred changes.
- **VIX terciles:** pairs split by the VIX close on the first day at the 1/3 and 2/3 percentiles of that close over the
  sample (cutoffs fixed on the full sample, also inside the bootstrap). Plain robust scales in each tercile.
- **Long-end rough extrapolation** exactly as in stage 1: the rough kernel (H = 0.01..0.99) fitted to the 30, 93, 184
  and 365-day tenors, extrapolated to 1 and 9 days; ln R1 and ln R9 = ln(observed / extrapolated). Computed for plain,
  weekday-centred and each tercile.
- **Hold-out prediction.** Four kernel families, each with a free level: rough, TSPL and one-factor exponential (stage-1
  grids) and the **two-factor exponential** k(x) = (1 - theta) exp(-k1 x) + theta exp(-k2 x), factors correlated rho,
  with k1 = 10^(-2.5 + 0.125 i), i = 0..28 (per day), k2 = 10^(-4.5 + 0.125 j), j = 0..28, k2 < k1,
  theta = 0.1..0.9 by 0.1 and rho in {0, 0.5, 1} (20,250 shapes). Its moments are exact closed forms (`stage2.py`).
  The one-day hold-out fits each family to the five tenors from 9 days to 1 year and predicts the 1-day value; the
  nine-day hold-out fits the four tenors from 30 days and predicts 9 days. The error is observed minus predicted
  (in ln robust scale). Done for plain and weekday-centred scales.
- The full six-tenor fits of all four families are reported too (descriptive only: the families have different
  numbers of parameters, so in-sample SSE is not used to rank them).
- Uncertainty: moving-block bootstrap over pairs, block 20, 1,000 draws, seed 20261118; 90% intervals are the 5th and
  95th percentiles.

## Predictions (made before seeing any stage-1 or stage-2 result)

- **Q1.** The stage-1 verdict rule applied to the weekday-centred scale gives the same verdict as applied to the plain
  scale (both from this run's bootstrap draws).
- **Q2.** The median daily log change of VIX1D differs across the four weekdays by more than 0.05 (max minus min).
- **Q3.** Weekday centring lowers ln R1 (point estimates).
- **Q4.** The plain ln R1 has the same sign in all three VIX terciles.
- **Q5.** The long-end rough H of the three VIX terciles differ by at most 0.10 (max minus min).
- **Q6.** The robust vol-of-vol of the 30-day VIX is higher in the top VIX tercile than in the bottom one.
- **Q7.** In the weekday-centred one-day hold-out, the two-factor exponential predicts the 1-day value with a smaller
  absolute error than the rough kernel.
- **Q8.** In the weekday-centred one-day hold-out, TSPL predicts the 1-day value with a smaller absolute error than
  the rough kernel.

## How the results will be read (fixed)

- **Seasonality:** if Q1 holds, stage 1's verdict is robust to the VIX1D weekday pattern. If not, the weekday-centred
  verdict is the one to trust, because the weekday pattern is a known artefact of the index methodology.
- **Model comparison (primary for stage 2):** the 90% interval of |error rough| - |error two-factor| at the held-out
  one-day tenor, weekday-centred. Above 0: "the two-factor Markov kernel predicts the one-day end better than the
  rough kernel". Below 0: the reverse. Otherwise: "no clear winner at the one-day end". A rough win here would be
  notable, since the two-factor family has three shape parameters to the rough kernel's one.
- **Regime:** if Q4 or Q5 fails, the short-end shape depends on the volatility level, and the unconditional verdict is
  reported with that caveat.

Every prediction will be graded and published, right or wrong, in `RESULTS.md`, together with stage 1.

## Known limitations (fixed in advance)

- As in stage 1: first-order Gaussian model, flat forward-variance curve, calendar-day time, one 4.5-year regime.
- The two-factor fit uses a grid, not a continuous optimizer; the best grid point is the fit.
- Weekday centring removes a constant pattern per weekday; if the pattern changed over 2022-2026 (Cboe changed nothing
  in the VIX1D formula that we know of), some seasonality remains.
- Terciles have about a third of the pairs each, so their intervals are wider.

This is research on public market data. It is not investment advice.
