"""
Bsplines
========

B-spline toolbox for constructing, evaluating, and fitting spline curves
to demographic or actuarial data.

Overview
--------
A **B-spline of degree** :math:`p` over a knot vector
:math:`t_0 \\le t_1 \\le \\ldots \\le t_{n+p}` is defined recursively by the
**Cox–de Boor recurrence**:

.. math::

    B_{i,0}(x) =
    \\begin{cases}
        1 & \\text{if } t_i \\le x < t_{i+1} \\\\
        0 & \\text{otherwise}
    \\end{cases}

.. math::

    B_{i,p}(x) =
      \\frac{x - t_i}{t_{i+p} - t_i}\\, B_{i,p-1}(x)
    + \\frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}}\\, B_{i+1,p-1}(x)

A **spline curve** is the weighted sum of :math:`n` basis functions:

.. math::

    S(x) = \\sum_{i=0}^{n-1} c_i\\, B_{i,p}(x)

where :math:`c_i` are the **B-spline coefficients** (control points).

Key properties:

* **Local support** — :math:`B_{i,p}` is non-zero only on
  :math:`[t_i,\\, t_{i+p+1})`, so each coefficient influences the curve
  only locally.
* **Partition of unity** — :math:`\\sum_i B_{i,p}(x) = 1` for all :math:`x`
  in the domain.
* **Non-negativity** — :math:`B_{i,p}(x) \\ge 0` for all :math:`x`.

Knot vector convention (scipy)
-------------------------------
Given :math:`m` equally spaced internal knots in :math:`[x_{\\min}, x_{\\max}]`
and a spline degree :math:`p`, the **full knot vector** is built as:

.. math::

    \\underbrace{x_{\\min}, \\ldots, x_{\\min}}_{p \\text{ times}},\\;
    t_1, \\ldots, t_m,\\;
    \\underbrace{x_{\\max}, \\ldots, x_{\\max}}_{p \\text{ times}}

which yields :math:`n_{\\text{basis}} = m + p - 1` basis functions.

Finite difference penalty (P-splines)
--------------------------------------
To regularise a fit, the :math:`k`-th order **difference matrix**
:math:`D_k \\in \\mathbb{R}^{(n-k) \\times n}` is used to penalise
rough coefficient sequences:

.. math::

    \\text{penalty} = \\lambda \\|D_k\\, \\mathbf{c}\\|^2

The first-order difference operator is:

.. math::

    D_1 = \\begin{pmatrix}
        -1 &  1 &  0 & \\cdots \\\\
         0 & -1 &  1 & \\cdots \\\\
         \\vdots & & \\ddots &
    \\end{pmatrix}

and the :math:`k`-th order matrix is obtained by applying :math:`D_1`
recursively :math:`k` times, with coefficients given by:

.. math::

    (D_k)_{i,j} = (-1)^j \\binom{k}{j}, \\quad j = 0, \\ldots, k

Visual illustration
-------------------
.. image:: ../../_static/bspline_illustration.png
   :width: 700px
   :align: center
   :alt: B-spline basis functions and resulting curve

   *Left* — Individual B-spline basis functions :math:`B_{i,3}(x)` of
   degree 3 with 6 internal knots. The dashed vertical lines mark the knot
   positions. The basis functions form a **partition of unity**.
   *Right* — Resulting spline curve :math:`S(x) = \\sum_i c_i B_{i,3}(x)`
   (thick dark line) as the weighted sum of the shaded basis functions.
   Red dots are the control points :math:`c_i`.

**Dependencies**: ``numpy``, ``scipy``, ``matplotlib``, ``math``.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln
import pandas as pd
from math import comb
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.linalg import lstsq


# ════════════════════════════════════════════════════════════════════
# FINITE DIFFERENCE MATRIX
# ════════════════════════════════════════════════════════════════════

def difference_matrix(n, k):
    """
    Construct the finite-difference matrix of order *k* for vectors of
    length *n*.

    The :math:`k`-th order difference matrix :math:`D_k` of size
    :math:`(n-k) \\times n` is defined entry-wise as:

    .. math::

        (D_k)_{i,\\,i+j} = (-1)^j \\binom{k}{j}, \\quad j = 0, \\ldots, k

    so that :math:`(D_k \\mathbf{c})_i = \\Delta^k c_i`, the :math:`k`-th
    forward difference of the coefficient sequence.

    It is used in **P-spline** regression to penalise roughness:

    .. math::

        \\text{penalty} = \\lambda\\, \\|D_k \\mathbf{c}\\|^2

    :param n: Length of the coefficient vector (number of columns).
    :type n: int
    :param k: Order of the finite difference operator.
    :type k: int
    :returns: Difference matrix :math:`D_k` of shape ``(n-k, n)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``k >= n``.

    Examples
    --------
    First-order difference matrix for n=4::

        D = difference_matrix(4, 1)
        # [[-1,  1,  0,  0],
        #  [ 0, -1,  1,  0],
        #  [ 0,  0, -1,  1]]

    Second-order (curvature) matrix for n=5::

        D = difference_matrix(5, 2)
        # [[ 1, -2,  1,  0,  0],
        #  [ 0,  1, -2,  1,  0],
        #  [ 0,  0,  1, -2,  1]]
    """
    if k >= n:
        raise ValueError("Order k must be less than n.")

    # Coefficients of the k-th order difference using binomial coefficients
    coeffs = np.array([(-1)**j * comb(k, j) for j in range(k + 1)])

    # Initialise the (n-k) x n matrix
    D = np.zeros((n - k, n))

    # Fill each row with the shifted coefficient pattern
    for i in range(n - k):
        D[i, i:i + k + 1] = coeffs

    return D


# ════════════════════════════════════════════════════════════════════
# B-SPLINE TOOLBOX  (legacy / low-level helpers)
# ════════════════════════════════════════════════════════════════════

def plot_Bsplines(all_knots, degree, xmin, xmax):
    """
    Plot all B-spline basis functions defined by a full knot vector.

    For a knot vector :math:`\\mathbf{t}` of length :math:`n + p + 1`,
    there are :math:`n = |\\mathbf{t}| - p - 1` basis functions
    :math:`B_{0,p}, \\ldots, B_{n-1,p}`.  Each is obtained by setting
    the corresponding coefficient to 1 and all others to 0:

    .. math::

        B_{j,p}(x) = \\text{BSpline}(\\mathbf{t},\\, \\mathbf{e}_j,\\, p)(x)

    :param all_knots: Full knot vector (including boundary repetitions).
    :type all_knots: array-like
    :param degree: Spline degree :math:`p`.
    :type degree: int
    :param xmin: Left boundary of the evaluation domain.
    :type xmin: float
    :param xmax: Right boundary of the evaluation domain.
    :type xmax: float

    .. note::
        The function calls ``plt.show()`` directly. For non-interactive use,
        call ``plt.savefig()`` before ``plt.show()``, or suppress display with
        ``matplotlib.use("Agg")``.
    """
    x = np.arange(xmin, xmax, 0.1)
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))

    for j in range(ncoef):
        # Activate basis function j by setting its coefficient to 1
        c = np.zeros(ncoef)
        c[j] = 1.0
        bs = BSpline(all_knots, c, degree, extrapolate=False)
        B[:, j] = bs(x)
        plt.plot(x, B[:, j], label=str(j))

    plt.grid()
    plt.legend()
    plt.show()


def basis_matrix_from_knots(all_knots, degree, x):
    """
    Compute the B-spline basis matrix for a given full knot vector and
    evaluation points.

    Each column :math:`j` of the returned matrix contains the values of
    the :math:`j`-th basis function evaluated at all points in *x*:

    .. math::

        B[i,\\, j] = B_{j,p}(x_i)

    :param all_knots: Full knot vector :math:`\\mathbf{t}` (with boundary
        multiplicity).
    :type all_knots: array-like
    :param degree: Spline degree :math:`p`.
    :type degree: int
    :param x: Evaluation points.
    :type x: array-like
    :returns: Basis matrix of shape ``(len(x), n_basis)`` where
        ``n_basis = len(all_knots) - degree - 1``.
    :rtype: numpy.ndarray
    """
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))

    for j in range(ncoef):
        # Activate basis function j by setting its coefficient to 1
        c = np.zeros(ncoef)
        c[j] = 1.0
        bs = BSpline(all_knots, c, degree, extrapolate=False)
        B[:, j] = bs(x)

    return B


def Bspline_approx(coeffs, x, degree, m, xmin, xmax):
    """
    Evaluate a B-spline approximation from coefficients and an equispaced
    internal knot grid.

    The full knot vector is constructed with spacing
    :math:`\\delta = (x_{\\max} - x_{\\min}) / m` as:

    .. math::

        \\mathbf{t} = \\bigl[
            x_{\\min} - p\\,\\delta,\\;
            x_{\\min} - (p-1)\\,\\delta,\\;
            \\ldots,\\;
            x_{\\max} + p\\,\\delta
        \\bigr]

    so that the spline has support over :math:`[x_{\\min}, x_{\\max}]`.

    The approximated curve is:

    .. math::

        \\hat{y}(x) = B(x)\\,\\mathbf{c}

    where :math:`B(x)` is the basis matrix and :math:`\\mathbf{c}` the
    coefficient vector.

    :param coeffs: B-spline coefficient vector :math:`\\mathbf{c}`.
    :type coeffs: numpy.ndarray
    :param x: Evaluation points.
    :type x: array-like
    :param degree: Spline degree :math:`p`.
    :type degree: int
    :param m: Number of equally-spaced internal knot intervals.
        Internal knots do **not** include :math:`x_{\\min}` or
        :math:`x_{\\max}`.
    :type m: int
    :param xmin: Left boundary of the spline domain.
    :type xmin: float
    :param xmax: Right boundary of the spline domain.
    :type xmax: float
    :returns: Tuple ``(yhat, all_knots, B)`` where

        * ``yhat``      — approximated values at *x*, shape ``(len(x),)``.
        * ``all_knots`` — full knot vector used.
        * ``B``         — basis matrix of shape ``(len(x), n_basis)``.

    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    dx = (xmax - xmin) / m
    all_knots = xmin - dx * degree + dx * np.arange(0, m - 1 + 2 * (degree + 1))

    B    = basis_matrix_from_knots(all_knots, degree, x)
    yhat = B.dot(coeffs)

    return yhat, all_knots, B


# ════════════════════════════════════════════════════════════════════
# SCIPY-BASED B-SPLINE CONSTRUCTION
# ════════════════════════════════════════════════════════════════════

def make_bspline_basis(xv, degree, n_knots, xmin=None, xmax=None):
    """
    Build the B-spline basis matrix using ``scipy.interpolate.BSpline``.

    The **full knot vector** follows the scipy convention:

    .. math::

        \\mathbf{t} =
        \\underbrace{x_{\\min},\\,\\ldots,\\,x_{\\min}}_{p},\\;
        t_1,\\,\\ldots,\\,t_{n_{\\text{knots}}},\\;
        \\underbrace{x_{\\max},\\,\\ldots,\\,x_{\\max}}_{p}

    where :math:`t_1, \\ldots, t_{n_{\\text{knots}}}` are equally spaced
    internal knots (including the boundary values).

    The number of basis functions is:

    .. math::

        n_{\\text{basis}} = n_{\\text{knots}} + p - 1

    The basis matrix entry :math:`B[i,j]` is the value of the :math:`j`-th
    basis function at the :math:`i`-th evaluation point:

    .. math::

        B[i,\\, j] = B_{j,\\, p}(x_i)

    :param xv: Evaluation points (e.g. ages 0 to 82).
    :type xv: array-like
    :param degree: Spline degree :math:`p`.
    :type degree: int
    :param n_knots: Number of internal knots, including the two boundary
        knots :math:`x_{\\min}` and :math:`x_{\\max}`.
    :type n_knots: int
    :param xmin: Left boundary of the spline domain (default: ``min(xv)``).
    :type xmin: float, optional
    :param xmax: Right boundary of the spline domain (default: ``max(xv)``).
    :type xmax: float, optional
    :returns: Tuple ``(B, knots, n_basis)`` where

        * ``B``       — basis matrix of shape ``(len(xv), n_basis)``.
        * ``knots``   — full knot vector.
        * ``n_basis`` — number of basis functions.

    :rtype: tuple(numpy.ndarray, numpy.ndarray, int)

    Notes
    -----
    Values outside the support :math:`[x_{\\min}, x_{\\max}]` are set to 0
    (NaN produced by ``extrapolate=False`` are replaced).

    Examples
    --------
    ::

        ages   = np.arange(0, 83)
        B, t, n = make_bspline_basis(ages, degree=3, n_knots=10)
        # B.shape == (83, 12)   since n_basis = 10 + 3 - 1 = 12
    """
    if xmin is None:
        xmin = float(np.min(xv))

    if xmax is None:
        xmax = float(np.max(xv))

    # Equally spaced internal knots (boundary values included)
    internal_knots = np.linspace(xmin, xmax, n_knots)

    # Full knot vector with boundary multiplicity (scipy convention)
    knots = np.concatenate([
        [xmin] * degree,        # left boundary repetition
        internal_knots,
        [xmax] * degree         # right boundary repetition
    ])

    n_basis = len(knots) - degree - 1

    # Build the basis matrix column by column
    B = np.zeros((len(xv), n_basis))

    for i in range(n_basis):
        # Basis function i: coefficient vector is the i-th standard basis vector
        coef = np.zeros(n_basis)
        coef[i] = 1.0

        spline   = BSpline(knots, coef, degree, extrapolate=False)
        B[:, i]  = spline(xv)

    # Replace NaN values with 0 (points outside the support)
    B = np.nan_to_num(B, nan=0.0)

    return B, knots, n_basis


def eval_bspline_from_coef(coef, xv, knots, degree):
    """
    Evaluate a B-spline curve from its coefficient vector.

    Given coefficients :math:`\\mathbf{c} = (c_0, \\ldots, c_{n-1})` and a
    full knot vector :math:`\\mathbf{t}`, the curve is:

    .. math::

        S(x) = \\sum_{i=0}^{n-1} c_i\\, B_{i,p}(x)

    :param coef: B-spline coefficient vector :math:`\\mathbf{c}`, length
        equal to ``len(knots) - degree - 1``.
    :type coef: numpy.ndarray
    :param xv: Evaluation points.
    :type xv: array-like
    :param knots: Full knot vector :math:`\\mathbf{t}` (with boundary
        multiplicity).
    :type knots: array-like
    :param degree: Spline degree :math:`p`.
    :type degree: int
    :returns: Curve values :math:`S(x_i)` at each point in *xv*, shape
        ``(len(xv),)``. Points outside the support return 0.
    :rtype: numpy.ndarray

    Examples
    --------
    ::

        B, knots, n = make_bspline_basis(ages, degree=3, n_knots=10)
        coef = np.linalg.lstsq(B, y_obs, rcond=None)[0]
        y_fit = eval_bspline_from_coef(coef, ages, knots, degree=3)
    """
    spline = BSpline(knots, coef, degree, extrapolate=False)
    y      = spline(xv)

    # Replace NaN (outside support) with 0
    return np.nan_to_num(y, nan=0.0)