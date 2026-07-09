"""Random Matrix Theory core for the crypto-RMT analysis.

This module owns the correlation/eigenvalue machinery: it builds the
cross-correlation matrix of the z-scored returns, diagonalizes it, quantifies
how localized each eigenvector is (inverse participation ratio), and
characterizes the noise band via a shuffled-correlation null model. No data
loading or plotting logic lives here.

Notes
-----
Two correctness fixes from the original notebook are baked in here:

* Eigendecomposition uses :func:`numpy.linalg.eigh` on the real-symmetric
  correlation matrix (no ``sympy``), so eigenvalues are real by construction.
* The inverse participation ratio is computed from the eigenvectors of the
  correlation matrix ``C`` (the columns returned by :func:`eigsystem`), **not**
  from the eigenvectors of a diagonal matrix. The latter are standard basis
  vectors, for which ``sum_i v[i]**4`` is identically ``1`` for every mode
  (a vacuous result).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "SHUFFLES_PER_PAIR",
    "correlation_matrix",
    "eigsystem",
    "ipr",
    "participation_ratio",
    "shuffle_correlation",
    "null_spectrum",
    "null_threshold",
]

#: Number of shuffled matrices generated per distinct asset pair. With ``N``
#: assets there are ``N*(N-1)//2`` pairs, so the default shuffle count is
#: ``SHUFFLES_PER_PAIR * N*(N-1)//2`` (765 for ``N = 18``, matching the
#: notebook's ``5 * 153``).
SHUFFLES_PER_PAIR: int = 5


def correlation_matrix(
    returns: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the cross-correlation matrix of the return series.

    Parameters
    ----------
    returns : numpy.ndarray
        Array of shape ``(N, T)`` whose rows are the (z-scored) return series
        of ``N`` assets over ``T`` observations.

    Returns
    -------
    numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)`` with unit diagonal.
    """
    return np.corrcoef(returns)


def eigsystem(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Diagonalize the correlation matrix, sorted by descending eigenvalue.

    Parameters
    ----------
    C : numpy.ndarray
        Real-symmetric correlation matrix of shape ``(N, N)``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        The ``N`` eigenvalues in **descending** order.
    eigenvectors : numpy.ndarray
        Array of shape ``(N, N)`` whose **columns** are the unit-norm
        eigenvectors, ordered to match ``eigenvalues`` (column ``k``
        corresponds to ``eigenvalues[k]``).

    Notes
    -----
    Uses :func:`numpy.linalg.eigh`, which is designed for real-symmetric (and
    Hermitian) matrices and returns real eigenvalues in ascending order. The
    output is reversed so the market mode (largest eigenvalue) is first, and the
    eigenvector columns are reordered identically to stay aligned.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def ipr(
    eigenvectors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the inverse participation ratio (IPR) of each mode.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape ``(N, N)`` whose columns are unit-norm eigenvectors, as
        returned by :func:`eigsystem`.

    Returns
    -------
    numpy.ndarray
        Array of length ``N`` with ``ipr[k] = sum_i eigenvectors[i, k]**4``.

    Notes
    -----
    The IPR is large (approaching ``1``) for a mode localized on a single
    asset and small (approaching ``1 / N``) for a mode spread uniformly across
    all assets. It must be computed from the eigenvectors of the correlation
    matrix, not from the standard basis vectors of a diagonal matrix.
    """
    return (eigenvectors**4).sum(axis=0)


def participation_ratio(
    eigenvectors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the participation ratio of each mode.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape ``(N, N)`` whose columns are unit-norm eigenvectors, as
        returned by :func:`eigsystem`.

    Returns
    -------
    numpy.ndarray
        Array of length ``N`` equal to ``1 / ipr(eigenvectors)``, i.e. the
        effective number of assets participating in each mode.
    """
    return 1.0 / ipr(eigenvectors)


def shuffle_correlation(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Randomly permute the off-diagonal entries of a correlation matrix.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)`` with unit diagonal.
    rng : numpy.random.Generator
        Random generator supplying the permutation.

    Returns
    -------
    numpy.ndarray
        A new symmetric matrix whose upper off-diagonal entries are a random
        permutation of those of ``C``, mirrored to the lower triangle. The
        diagonal is left untouched.

    Notes
    -----
    Only the ``N*(N-1)//2`` upper off-diagonal entries are shuffled (once) and
    mirrored, which preserves symmetry and the unit diagonal while destroying
    any genuine correlation structure. This is the empirical noise baseline for
    the eigenvalue spectrum.
    """
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    shuffled = C.copy()
    values = rng.permutation(C[iu])
    shuffled[iu] = values
    shuffled[(iu[1], iu[0])] = values
    return shuffled


def null_spectrum(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
    n_shuffles: int | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Pool the eigenvalues of many shuffled correlation matrices.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)``.
    n_shuffles : int, optional
        Number of shuffled matrices to generate. Defaults to
        ``SHUFFLES_PER_PAIR * N*(N-1)//2`` (765 for ``N = 18``).
    rng : numpy.random.Generator, optional
        Random generator. Defaults to :func:`numpy.random.default_rng`.

    Returns
    -------
    numpy.ndarray
        Flat array of all ``n_shuffles * N`` eigenvalues from the shuffled
        matrices, forming the empirical noise band.

    Notes
    -----
    Eigenvalues are obtained with :func:`numpy.linalg.eigvalsh`, which returns
    real values for the symmetric shuffled matrices.
    """
    n = C.shape[0]
    if n_shuffles is None:
        n_shuffles = SHUFFLES_PER_PAIR * (n * (n - 1) // 2)
    if rng is None:
        rng = np.random.default_rng()

    spectra = [
        np.linalg.eigvalsh(shuffle_correlation(C, rng)) for _ in range(n_shuffles)
    ]
    return np.concatenate(spectra)


def null_threshold(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
    n_shuffles: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the upper edge of the shuffled-null eigenvalue band.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)``.
    n_shuffles : int, optional
        Number of shuffled matrices. See :func:`null_spectrum`.
    rng : numpy.random.Generator, optional
        Random generator. Defaults to :func:`numpy.random.default_rng`.

    Returns
    -------
    float
        The maximum eigenvalue observed across all shuffled matrices. A real
        eigenvalue exceeding this threshold is evidence of genuine structure.
    """
    return float(null_spectrum(C, n_shuffles=n_shuffles, rng=rng).max())
