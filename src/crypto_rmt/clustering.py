"""Hierarchical clustering of the crypto correlation matrix.

This module turns the cross-correlation matrix produced by
:mod:`crypto_rmt.rmt` into a hierarchical clustering, ready to drive the
cluster-map figure. It owns *only* the distance/linkage/label machinery; the
dendrogram and heatmap rendering live in :mod:`crypto_rmt.plotting`.

Notes
-----
The original notebook called ``seaborn.clustermap`` directly on the raw
correlation matrix, which clusters the rows under the default Euclidean metric
on correlation *values* -- not a proper distance between assets. This module
replaces that with the standard correlation distance
``d_ij = sqrt(2 * (1 - C_ij))`` (Mantegna 1999), an ultrametric-friendly,
methodologically grounded metric: perfectly correlated assets are at distance
``0`` and perfectly anti-correlated assets at distance ``2``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "correlation_distance",
    "linkage_from_correlation",
    "cluster_labels",
]


def correlation_distance(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
) -> npt.NDArray[np.float64]:
    """Convert a correlation matrix into a metric distance matrix.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)`` with unit diagonal and
        entries in ``[-1, 1]``.

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape ``(N, N)`` with
        ``d_ij = sqrt(2 * (1 - C_ij))``, an exact zero diagonal, and all
        entries ``>= 0``.

    Notes
    -----
    This is the Mantegna (1999) correlation distance. Floating-point overshoot
    of ``2 * (1 - C_ij)`` below zero (possible near ``C_ij = 1``) is clipped to
    ``0`` before the square root. The diagonal is forced to an exact zero and
    the result is symmetrized so it is a valid metric matrix that
    :func:`scipy.spatial.distance.squareform` will accept.
    """
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))  # noqa: N806
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0  # noqa: N806
    return D


def linkage_from_correlation(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
    method: str = "average",
) -> npt.NDArray[np.float64]:
    """Build a hierarchical linkage from a correlation matrix.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)``.
    method : str, optional
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`
        (e.g. ``"average"``, ``"complete"``, ``"ward"``). Defaults to
        ``"average"``.

    Returns
    -------
    numpy.ndarray
        The linkage matrix of shape ``(N - 1, 4)`` as returned by
        :func:`scipy.cluster.hierarchy.linkage`.

    Notes
    -----
    The correlation matrix is first mapped to the Mantegna distance via
    :func:`correlation_distance`, then condensed with
    :func:`scipy.spatial.distance.squareform` (``checks=False``) into the
    upper-triangular vector that :func:`scipy.cluster.hierarchy.linkage`
    expects.
    """
    condensed = squareform(correlation_distance(C), checks=False)
    return linkage(condensed, method=method)


def cluster_labels(
    Z: npt.NDArray[np.float64],  # noqa: N803  (linkage matrix, standard symbol)
    k: int,
) -> npt.NDArray[np.int32]:
    """Cut a linkage into exactly ``k`` flat clusters.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix of shape ``(N - 1, 4)`` as returned by
        :func:`linkage_from_correlation`.
    k : int
        Desired number of clusters.

    Returns
    -------
    numpy.ndarray
        Array of length ``N`` of integer cluster labels (one per asset),
        obtained via :func:`scipy.cluster.hierarchy.fcluster` with
        ``criterion="maxclust"``.
    """
    return fcluster(Z, t=k, criterion="maxclust")
