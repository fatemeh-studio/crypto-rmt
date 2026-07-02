# crypto-rmt — Random Matrix Theory on Cryptocurrency Correlations

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Separating genuine market structure from statistical noise in a basket of
> cryptocurrencies, using Random Matrix Theory (RMT). The empirical correlation
> spectrum is compared against the Marchenko–Pastur law to identify which
> correlations are real, which eigenvectors carry market-wide signal, and how
> the assets cluster.

---

## Overview

Correlation matrices estimated from finite time series are noisy: with `N`
assets and `T` observations, most of the eigenvalue spectrum is indistinguishable
from that of a random matrix. RMT provides a principled null model — the
**Marchenko–Pastur distribution** — for the bulk of eigenvalues you'd see from
pure noise. Eigenvalues that fall *outside* that bulk carry genuine information:
the largest typically encodes a market-wide mode, and the next few often map onto
sectors or groups of co-moving assets.

This project applies that toolkit to **[FILL: N] cryptocurrencies** sampled at
**hourly resolution**, and asks:

- How much of the measured correlation structure is real vs. noise?
- Is there a dominant "market mode," and which assets load onto it?
- Do the deviating eigenvectors reveal interpretable clusters?

## Method

1. **Data & returns** — load aligned hourly price series, handle missing values,
   and compute log-returns `r_t = ln(p_t / p_{t-1})`.
2. **Correlation matrix** — build the `N × N` cross-correlation matrix `C` of the
   standardized returns.
3. **Eigenspectrum vs. Marchenko–Pastur** — diagonalize `C` and compare the
   empirical eigenvalue density to the MP law with bounds
   `λ± = (1 ± √(N/T))²`. Eigenvalues above `λ₊` are candidate signal.
4. **Inverse participation ratio (IPR / NPR)** — measure how localized each
   eigenvector is, i.e. how many assets contribute to each mode.
5. **Shuffled-matrix null** — independently shuffle each return series, rebuild
   the correlation matrix, and characterize the distribution of its largest
   eigenvalue as an empirical noise baseline.
6. **Hierarchical clustering** — cluster assets from the correlation distances
   and visualize the structure as a dendrogram / cluster map.

## Key results

> Fill these from your original report — keep them specific.

- **[FILL]** of **[FILL: N]** eigenvalues lie above the Marchenko–Pastur upper
  bound `λ₊ ≈ [FILL]`, indicating genuine (non-random) correlation structure.
- The largest eigenvalue, `λ_max ≈ [FILL]`, corresponds to a market-wide mode in
  which **[FILL: describe which assets load onto it]**.
- The IPR analysis shows **[FILL: localized vs. delocalized findings]**.
- Hierarchical clustering separates the assets into **[FILL: number/description]**
  groups: **[FILL]**.

| Figure | Description |
|--------|-------------|
| `figures/spectrum_vs_mp.png` | Empirical eigenvalue density vs. Marchenko–Pastur |
| `figures/ipr.png` | Inverse participation ratio per eigenvalue |
| `figures/cluster_map.png` | Correlation-based hierarchical cluster map |

## Project structure

```
crypto-rmt/
├── src/crypto_rmt/
│   ├── io.py           # load, align, clean series; build the returns matrix
│   ├── rmt.py          # correlation matrix, eigen-decomposition,
│   │                   #   Marchenko–Pastur bounds, IPR/NPR, shuffled null
│   ├── clustering.py   # hierarchical clustering / cluster map
│   └── plotting.py     # spectrum, IPR, and cluster-map figures
├── notebooks/
│   └── demo.ipynb      # imports the package, reproduces the figures above
├── data/               # raw price series (gitignored — see Data)
├── figures/            # generated figures
├── tests/              # pytest smoke tests
├── AGENTS.md           # refactoring rules (for the Cursor agent)
├── README.md
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/[FILL: your-username]/crypto-rmt.git
cd crypto-rmt
pip install -e .
```

## Usage

```python
from crypto_rmt import io, rmt, plotting

returns = io.load_returns("data/", tickers=[...])   # N × T returns matrix
C = rmt.correlation_matrix(returns)
eigenvalues, eigenvectors = rmt.eigen(C)
lam_minus, lam_plus = rmt.marchenko_pastur_bounds(n=returns.shape[0], t=returns.shape[1])

plotting.spectrum_vs_mp(eigenvalues, lam_minus, lam_plus, save="figures/spectrum_vs_mp.png")
```

Or open `notebooks/demo.ipynb` to reproduce the full analysis end to end.

## Data

Hourly price series for **[FILL: N]** cryptocurrencies, stored as JSON
(`[{"t": <unix_seconds>, "v": <price | null>}, ...]`). Raw data files are **not
committed** (size + redistribution); `data/` is gitignored.

**[FILL: where you obtained the data — source/API, date range, e.g. "12,001 hours
ending [date]". Add a short script or instructions so the analysis is
reproducible.]**

## References

- L. Laloux, P. Cizeau, J.-P. Bouchaud, M. Potters, *Noise dressing of financial
  correlation matrices*, Phys. Rev. Lett. **83**, 1467 (1999).
- V. Plerou, P. Gopikrishnan, B. Rosenow, L. A. N. Amaral, T. Guhr, H. E. Stanley,
  *Random matrix approach to cross correlations in financial data*,
  Phys. Rev. E **65**, 066126 (2002).
- V. A. Marchenko, L. A. Pastur, *Distribution of eigenvalues for some sets of
  random matrices*, Math. USSR-Sbornik **1**, 457 (1967).

## License

Released under the MIT License — see `LICENSE`.
