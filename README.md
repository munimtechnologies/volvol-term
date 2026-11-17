# volvol-term: is volatility rough all the way down to one day?

An open, pre-registered test of one of the sharpest predictions of rough volatility, using only public Cboe data.

Rough-volatility models predict that the volatility of variance-swap rates (vol-of-vol) follows a power law in
maturity, growing without limit as the maturity shrinks. Between one month and a few years, such a power law is a
well-known stylized fact. Below one month it is contested: for the at-the-money skew, Guyon and El Amrani (Risk, 2023)
found that the power law breaks below about three weeks. Cboe's VIX1D (1 day), VIX9D, VIX, VIX3M, VIX6M and VIX1Y
indices now allow the vol-of-vol term structure to be measured from **1 day to 1 year**. This repo measures it, fits
rough, time-shifted power-law and one-factor Markov kernels with the exact finite-step model, and asks whether the
power law fitted between 1 month and 1 year carries on down to 1 day.

- [PREREG.md](PREREG.md): the question, the fixed design, eight predictions and how the result will be read, all
  committed before the analysis was run.
- `volvol.py`: data handling, the exact model moments (numerical integration), fitting and the block bootstrap.
  Python standard library only.
- `job.py`: downloads the six Cboe histories and writes `results.json` and `results.md` (derived statistics only).
- `test_volvol.py`: offline tests (`python -m unittest -v test_volvol`), including checks of the numerical integrals
  against closed forms.

Run it: `python job.py --data-dir data --out-dir out` (about 5 minutes on one CPU).

Results, with every prediction graded right or wrong, will be added as `RESULTS.md`.

Made with AI agents working for MV2 of Munim, Inc. (mv2@munimtech.com). Code is Apache-2.0; text and derived results
are CC BY 4.0. Index data belong to Cboe and are not redistributed here. This is research, not investment advice.
