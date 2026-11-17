# volvol-term: is volatility rough all the way down to one day?

An open, pre-registered test of one of the sharpest predictions of rough volatility, using only public Cboe data.

Rough-volatility models predict that the volatility of variance-swap rates (vol-of-vol) follows a power law in
maturity, growing without limit as the maturity shrinks. Between one month and a few years, such a power law is a
well-known stylized fact. Below one month it is contested: for the at-the-money skew, Guyon and El Amrani (Risk, 2023)
found that the power law breaks below about three weeks. Cboe's VIX1D (1 day), VIX9D, VIX, VIX3M, VIX6M and VIX1Y
indices now allow the vol-of-vol term structure to be measured from **1 day to 1 year**. This repo measures it, fits
rough, time-shifted power-law and one-factor Markov kernels with the exact finite-step model, and asks whether the
power law fitted between 1 month and 1 year carries on down to 1 day.

## Stage 1

- [PREREG.md](PREREG.md): the question, the fixed design, eight predictions and how the result will be read, all
  committed before the analysis was run.
- `volvol.py`: data handling, the exact model moments (numerical integration), fitting and the block bootstrap.
  Python standard library only.
- `job.py`: downloads the six Cboe histories and writes `results.json` and `results.md` (derived statistics only).
- `test_volvol.py`: offline tests, including checks of the numerical integrals against closed forms.

## Stage 2

[PREREG2.md](PREREG2.md), registered before any stage-1 result existed, adds three checks:

- **Weekday seasonality.** VIX1D has a known day-of-week zig-zag caused by its methodology (Albers and Kestner, Finance
  Research Letters 2024). It inflates measured one-day vol-of-vol, so the analysis is repeated with each index's
  changes centred on their own weekday's median.
- **Volatility regime.** The term structure and the long-end extrapolation are measured in each VIX tercile.
- **A two-factor Markov rival.** The two-factor Bergomi kernel (a mix of two exponentials, finite at zero lag) is
  fitted with exact closed-form moments. Every family is fitted with the one-day tenor held out and judged on how
  well it predicts it.

Files: `stage2.py` (closed forms and statistics), `job2.py` (writes `results2.json` and `results2.md`),
`test_stage2.py` (offline tests, including the closed form against the stage-1 numerical integrator).

## Run it

    python -m unittest -v test_volvol test_stage2
    python job.py --data-dir data --out-dir out      # stage 1, about 5 minutes on one CPU
    python job2.py --data-dir data --out-dir out     # stage 2, about 10 minutes on one CPU

Results, with every prediction graded right or wrong, will be added as `RESULTS.md`.

Made with AI agents working for MV2 of Munim, Inc. (mv2@munimtech.com). Code is Apache-2.0; text and derived results
are CC BY 4.0. Index data belong to Cboe and are not redistributed here. This is research, not investment advice.
