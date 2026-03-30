# -*- coding: utf-8 -*-
"""
LiLee — Object-oriented architecture
===============================================================

Multi-population mortality models based on the Li-Lee family.

Class hierarchy::

    LiLee                              ← general base class
    ├── LiLee.Parametric               ← parametric variants (B-splines + P-splines)
    │   ├── LiLee.Parametric.FullModel ← full Li-Lee model (two time indices)
    │   └── LiLee.Parametric.Variant   ← Lee & Li model (single common time index)
    └── LiLee.Classic                  ← classic Li-Lee (gradient descent)

Model equations:

.. math::

    \\text{FullModel:} \\quad
    \\ln(\\mu_{x,t,g}) = \\alpha_{x,g} + \\beta_x \\cdot \\kappa_t
                        + \\beta_{x,g} \\cdot \\kappa_{g,t}

    \\text{Variant:} \\quad
    \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_{x,g} \\cdot \\kappa_t

USAGE::

    # Full Li-Lee parametric model
    model = LiLee.Parametric.FullModel(degree=3, n_knots=6, lam=0.1)
    results = model.fit(alpha_coef_init, beta_coef_init, beta_g_coef_init,
                        kappa_init, kappa_g_init, Extg, Dxtg, xv, tv)

    # Lee & Li parametric model
    model = LiLee.Parametric.Variant(n_knots=10, lam=0.1)
    results = model.fit(ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv)

    # Classic Li-Lee model
    model = LiLee.Classic(nb_iter=500)
    results = model.fit(ax, bx, bx_gr, kappa, kappa_gr, Extg, Dxtg, Muxtg, xv, tv)

Dependencies:

- :class:`scipy.interpolate.BSpline`
- :func:`scipy.special.gammaln`
- :mod:`numpy`

Reference: Hainaut (2025), equation (1) page 3
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import gammaln
from math import comb
import matplotlib.pyplot as plt

from morta_nuts2.model.Bsplines.Bsplines import make_bspline_basis, eval_bspline_from_coef


# =============================================================================
# BASE CLASS — LiLee
# Contains:
#   - shared static methods (penalty matrix, log-likelihood, fit stats...)
#   - nested sub-classes: Parametric and Classic
# =============================================================================

class LiLee:
    """
    General Li-Lee class.

    Groups together:

    - shared mathematical tools (static methods)
    - the two main model families as nested sub-classes:

      - :class:`LiLee.Parametric` — two parametric variants (B-splines, P-splines)
      - :class:`LiLee.Classic`    — non-parametric gradient descent

    Example::

        >>> model = LiLee.Parametric.FullModel(degree=3, n_knots=6, lam=0.1, verbose=True)
        >>> results = model.fit(alpha0, beta0, beta_g0, kappa0, kappa_g0, Extg, Dxtg, xv, tv)
    """

    # =========================================================================
    # SHARED STATIC METHODS
    # =========================================================================

    # -------------------------------------------------------------------------
    # P-SPLINES PENALTY MATRIX — numpy
    # -------------------------------------------------------------------------
    @staticmethod
    def make_penalty_matrix(n_basis, diff_order=2):
        """
        Build the P-splines penalty matrix :math:`D^T D`.

        The penalty is used to enforce smoothness on the spline coefficients
        :math:`c` via the penalized log-likelihood:

        .. math::

            \\ell_p(\\theta) = \\ell(\\theta) - \\lambda \\, c^\\top D^\\top D \\, c

        where :math:`D` is the finite difference matrix of order ``diff_order``
        applied to an identity of size ``n_basis``.

        :param n_basis: Number of spline basis functions.
        :type n_basis: int
        :param diff_order: Order of the finite difference operator.
            ``1`` penalizes slope changes; ``2`` penalizes curvature.
        :type diff_order: int

        :returns:
            - **DtD** (*numpy.ndarray*, shape ``(n_basis, n_basis)``) — penalty matrix :math:`D^\\top D`.
            - **diag_DtD** (*numpy.ndarray*, shape ``(n_basis,)``) — diagonal of ``DtD``,
              used for diagonal Hessian preconditioning.
        """
        D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
        DtD = D.T @ D
        return DtD, np.diag(DtD)

    # -------------------------------------------------------------------------
    # POISSON LOG-LIKELIHOOD — scipy.special.gammaln
    # -------------------------------------------------------------------------
    @staticmethod
    def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
        """
        Compute the Poisson log-likelihood for the Li-Lee model.

        Under the assumption
        :math:`D_{x,t}^{(g)} \\sim \\text{Poisson}(E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)})`,
        the log-likelihood is:

        .. math::

            \\ell(\\theta) = \\sum_{x,t,g} \\Bigl[
                D_{x,t}^{(g)} \\ln \\mu_{x,t}^{(g)}
                - E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)}
                + D_{x,t}^{(g)} \\ln E_{x,t}^{(g)}
                - \\ln D_{x,t}^{(g)}!
            \\Bigr]

        :param Dxtg: Observed death counts, shape ``(nb_ages, nb_years, nb_regions)``.
        :type Dxtg: numpy.ndarray
        :param Extg: Exposures, shape ``(nb_ages, nb_years, nb_regions)``.
        :type Extg: numpy.ndarray
        :param logmu: Log mortality rates :math:`\\ln \\mu_{x,t}^{(g)}`, same shape as ``Dxtg``.
        :type logmu: numpy.ndarray
        :param logDxtgFact: Pre-computed :math:`\\ln(D_{x,t,g}!)` via ``scipy.special.gammaln``.
        :type logDxtgFact: numpy.ndarray

        :returns:
            - **lnL** (*float*) — scalar log-likelihood :math:`\\ell(\\theta)`.
            - **exp_logmu** (*numpy.ndarray*) — :math:`\\mu_{x,t}^{(g)} = \\exp(\\ln \\mu)`.
            - **weighted_exp** (*numpy.ndarray*) — expected deaths :math:`E_{x,t}^{(g)} \\mu_{x,t}^{(g)}`.
            - **residual** (*numpy.ndarray*) — Pearson residuals :math:`D_{x,t}^{(g)} - E_{x,t}^{(g)} \\mu_{x,t}^{(g)}`.
        """
        exp_logmu    = np.exp(logmu)
        weighted_exp = Extg * exp_logmu
        residual     = Dxtg - weighted_exp
        lnL = float(np.sum(
            Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
        ))
        return lnL, exp_logmu, weighted_exp, residual

    # -------------------------------------------------------------------------
    # EFFECTIVE DEGREES OF FREEDOM — trace of the hat matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def effective_dof_spline_block(B, w_flat, lam, DtD):
        """
        Effective degrees of freedom for one B-spline block under P-splines penalty.

        For a penalized Poisson GLM, the effective dofs of a single spline block
        are the trace of its hat-matrix contribution:

        .. math::

            \\text{dof}_\\text{eff}
            = \\text{tr}\\!\\left[
                \\left(B^T W B + \\lambda D^T D\\right)^{-1} B^T W B
              \\right]

        where :math:`W = \\text{diag}(w)` are the final Poisson weights.
        When :math:`\\lambda = 0` this reduces to :math:`n_{\\text{basis}}` exactly.

        :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
        :type B: numpy.ndarray
        :param w_flat: Aggregated Poisson weights per age, shape ``(nb_ages,)``.
            Typically ``weighted_exp[:, :, g].sum(axis=1)`` for region ``g``,
            or ``weighted_exp.sum(axis=(1, 2))`` for a common block.
        :type w_flat: numpy.ndarray
        :param lam: P-splines penalty weight :math:`\\lambda`.
        :type lam: float
        :param DtD: Penalty matrix :math:`D^T D`, shape ``(n_basis, n_basis)``.
        :type DtD: numpy.ndarray

        :returns: Effective dofs for this block (scalar).
        :rtype: float
        """
        if lam == 0.0:
            return float(B.shape[1])          # n_basis, exact
        BtWB = B.T @ (w_flat[:, None] * B)    # (n_basis, n_basis)
        A    = BtWB + lam * DtD               # penalized Hessian
        try:
            AinvBtWB = np.linalg.solve(A, BtWB)
        except np.linalg.LinAlgError:
            AinvBtWB = np.linalg.lstsq(A, BtWB, rcond=None)[0]
        return float(np.trace(AinvBtWB))

    # -------------------------------------------------------------------------
    # FIT STATISTICS — numpy
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, dofs):
        """
        Compute Poisson deviance, AIC, and BIC given pre-computed effective dofs.

        **Poisson deviance**

        .. math::

            \\mathcal{D} = 2\\bigl(\\ell_{\\text{sat}} - \\ell(\\hat{\\theta})\\bigr)

        where the saturated log-likelihood is:

        .. math::

            \\ell_{\\text{sat}} = \\sum_{x,t,g} \\left[
                D_{x,t}^{(g)} \\ln\\!\\frac{D_{x,t}^{(g)}}{E_{x,t}^{(g)}}
                - D_{x,t}^{(g)}
            \\right]

        **Information criteria**

        .. math::

            \\text{AIC} = 2 \\cdot \\text{dofs} - 2\\,\\ell(\\hat{\\theta}), \\qquad
            \\text{BIC} = \\text{dofs} \\cdot \\ln(N) - 2\\,\\ell(\\hat{\\theta})

        The effective dofs must be computed by the calling ``fit()`` method
        using :meth:`LiLee.effective_dof_spline_block`, so that the P-splines
        penalty is properly accounted for.

        :param Dxtg: Death counts, shape ``(nb_ages, nb_years, nb_regions)``.
        :type Dxtg: numpy.ndarray
        :param Extg: Exposures, same shape as ``Dxtg``.
        :type Extg: numpy.ndarray
        :param logmu: Fitted log mortality rates, same shape as ``Dxtg``.
        :type logmu: numpy.ndarray
        :param logDxtgFact: Pre-computed :math:`\\ln(D!)`, same shape as ``Dxtg``.
        :type logDxtgFact: numpy.ndarray
        :param dofs: Effective number of parameters (float).
        :type dofs: float

        :returns: DataFrame with columns ``["N", "dofs", "lnL", "deviance", "AIC", "BIC"]``.
        :rtype: pandas.DataFrame
        """
        exp_logmu = np.exp(logmu)
        lnL = float(np.sum(
            Dxtg * logmu - Extg * exp_logmu + Dxtg * np.log(Extg) - logDxtgFact
        ))

        safe_Dxtg = np.where(Dxtg > 0, Dxtg, 1.0)
        lnL_sat = float(np.sum(np.where(
            Dxtg > 0,
            Dxtg * np.log(safe_Dxtg / np.maximum(Extg, 1e-12)) - Dxtg,
            0.0
        )))
        deviance = 2.0 * (lnL_sat - lnL)

        nb_obs = int(Dxtg.size)
        AIC = 2.0 * dofs - 2.0 * lnL
        BIC = dofs * np.log(nb_obs) - 2.0 * lnL

        return pd.DataFrame(
            [[nb_obs, round(dofs, 2),
              round(lnL, 2), round(deviance, 2), round(AIC, 2), round(BIC, 2)]],
            columns=["N", "dofs", "lnL", "deviance", "AIC", "BIC"]
        )

    # -------------------------------------------------------------------------
    # FINITE DIFFERENCE MATRIX — used by Classic variant
    # -------------------------------------------------------------------------
    @staticmethod
    def difference_matrix(n, k):
        """
        Construct the finite difference matrix of order ``k`` for vectors of length ``n``.

        The :math:`k`-th order finite difference operator applied to a vector
        :math:`v \\in \\mathbb{R}^n` yields a vector of length :math:`n - k`
        whose :math:`i`-th entry is:

        .. math::

            (\\Delta^k v)_i = \\sum_{j=0}^{k} (-1)^j \\binom{k}{j} \\, v_{i+j}

        The matrix :math:`D` of shape :math:`(n-k, n)` encodes this operation
        as :math:`D v = \\Delta^k v`.

        :param n: Length of the input vector.
        :type n: int
        :param k: Difference order. Must be strictly less than ``n``.
        :type k: int

        :returns: Finite difference matrix of shape ``(n - k, n)``.
        :rtype: numpy.ndarray

        :raises ValueError: If ``k >= n``.
        """
        if k >= n:
            raise ValueError("Order k must be less than n.")
        # Compute coefficients of k-th difference using binomial coefficients
        coeffs = np.array([(-1)**j * comb(k, j) for j in range(k + 1)])
        # Initialize matrix
        D = np.zeros((n - k, n))
        # Fill each row with shifted coefficients
        for i in range(n - k):
            D[i, i:i + k + 1] = coeffs
        return D


    # =========================================================================
    # SUB-CLASS : Parametric
    # Contains two nested model variants: FullModel and LeeAndLi
    # =========================================================================

    class Parametric:
        """
        Parametric Li-Lee model family — B-splines + P-splines penalty.

        This class acts as a **namespace** grouping two parametric variants:

        :class:`LiLee.Parametric.FullModel`
            Full Li-Lee model with region-specific baselines :math:`\\alpha_{x,g}`,
            a common sensitivity :math:`\\beta_x`, and regional deviations
            :math:`\\beta_{x,g}`:

            .. math::

                \\ln(\\mu_{x,t,g}) = \\alpha_{x,g} + \\beta_x \\cdot \\kappa_t
                                    + \\beta_{x,g} \\cdot \\kappa_{g,t}

        :class:`LiLee.Parametric.Variant`
            Simplified Li & Lee model with a **common** baseline :math:`\\alpha_x`
            and region-specific sensitivities :math:`\\beta_{x,g}` driven by
            a **single** time index:

            .. math::

                \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_{x,g} \\cdot \\kappa_t

        .. note::

            Both variants share the same B-spline basis construction and
            P-splines roughness penalty. All age curves are smooth by construction.
        """

        # =====================================================================
        # FullModel
        # Full Li-Lee model: α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}
        # =====================================================================

        class FullModel:
            """
            Full parametric Li-Lee model.

            .. math::

                \\ln(\\mu_{x,t,g}) = \\alpha_{x,g} + \\beta_x \\cdot \\kappa_t
                                    + \\beta_{x,g} \\cdot \\kappa_{g,t}

            where:

            - :math:`\\alpha_{x,g}` — region-specific age baseline (B-splines)
            - :math:`\\beta_x`      — common age sensitivity (B-splines)
            - :math:`\\kappa_t`     — common time index
            - :math:`\\beta_{x,g}` — regional age sensitivity (B-splines)
            - :math:`\\kappa_{g,t}` — regional time index

            :param degree: B-spline degree (3 recommended).
            :type degree: int
            :param n_knots: Number of internal knots (6 recommended).
            :type n_knots: int
            :param xmin: Minimum age bound (inferred from data if ``None``).
            :type xmin: float or None
            :param xmax: Maximum age bound (inferred from data if ``None``).
            :type xmax: float or None
            :param lam: P-splines penalty weight.
            :type lam: float
            :param diff_order: Difference order for the penalty matrix (2 recommended).
            :type diff_order: int
            :param nb_iter: Maximum number of NR iterations.
            :type nb_iter: int
            :param eta0: Initial learning rate (0.30 recommended).
            :type eta0: float
            :param tol: Convergence tolerance.
            :type tol: float
            :param verbose: Whether to display iteration progress.
            :type verbose: bool
            """

            def __init__(
                self,
                degree=3,
                n_knots=6,
                xmin=None,
                xmax=None,
                lam=0.0,
                diff_order=2,
                nb_iter=800,
                eta0=0.30,
                tol=1e-3,
                verbose=False,
            ):
                self.degree     = degree
                self.n_knots    = n_knots
                self.xmin       = xmin
                self.xmax       = xmax
                self.lam        = lam
                self.diff_order = diff_order
                self.nb_iter    = nb_iter
                self.eta0       = eta0
                self.tol        = tol
                self.verbose    = verbose

            # -----------------------------------------------------------------
            # ln(µ) RECONSTRUCTION — FULL LI-LEE MODEL
            # -----------------------------------------------------------------
            @staticmethod
            def compute_logmu_lilee(
                alpha_coef,   # (nb_regions, n_basis)  α_{x,g}
                beta_coef,    # (n_basis,)              β_x common
                beta_g_coef,  # (nb_regions, n_basis)  β_{x,g}
                kappa,        # (nb_years,)             κ_t common
                kappa_g,      # (nb_regions, nb_years)  κ_{g,t}
                xv, B, knots, degree
            ):
                """
                Reconstruct :math:`\\ln(\\mu_{x,t,g})` for the full Li-Lee model:

                .. math::

                    \\ln(\\mu_{x,t,g}) = \\alpha_{x,g} + \\beta_x \\cdot \\kappa_t
                                        + \\beta_{x,g} \\cdot \\kappa_{g,t}

                :param alpha_coef: B-spline coefficients of :math:`\\alpha_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type alpha_coef: numpy.ndarray
                :param beta_coef: B-spline coefficients of :math:`\\beta_x`,
                    shape ``(n_basis,)``.
                :type beta_coef: numpy.ndarray
                :param beta_g_coef: B-spline coefficients of :math:`\\beta_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type beta_g_coef: numpy.ndarray
                :param kappa: Common time index :math:`\\kappa_t`, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param kappa_g: Regional time indices :math:`\\kappa_{g,t}`,
                    shape ``(nb_regions, nb_years)``.
                :type kappa_g: numpy.ndarray
                :param xv: Age vector, shape ``(nb_ages,)``.
                :type xv: numpy.ndarray
                :param B: Pre-computed B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param knots: Full knot vector.
                :type knots: numpy.ndarray
                :param degree: B-spline degree.
                :type degree: int

                :returns:
                    - **logmu** (*numpy.ndarray*, shape ``(nb_ages, nb_years, nb_regions)``)
                    - **alpha** (*numpy.ndarray*, shape ``(nb_ages, nb_regions)``)
                      — evaluated :math:`\\alpha_{x,g}`
                    - **beta** (*numpy.ndarray*, shape ``(nb_ages,)``)
                      — evaluated :math:`\\beta_x`
                    - **beta_g** (*numpy.ndarray*, shape ``(nb_ages, nb_regions)``)
                      — evaluated :math:`\\beta_{x,g}`
                """
                nb_regions = alpha_coef.shape[0]
                nb_ages    = len(xv)

                # Evaluate B-spline curves
                alpha  = np.zeros((nb_ages, nb_regions))
                beta_g = np.zeros((nb_ages, nb_regions))
                for g in range(nb_regions):
                    alpha[:, g]  = B @ alpha_coef[g]
                    beta_g[:, g] = B @ beta_g_coef[g]

                beta = B @ beta_coef

                # Build ln(µ) by broadcasting
                # Dimensions: (nb_ages, nb_years, nb_regions)
                logmu = (
                    alpha[:, None, :]                                # (nb_ages, 1, nb_regions)
                    + beta[:, None, None] * kappa[None, :, None]    # β_x · κ_t
                    + beta_g[:, None, :] * kappa_g.T[None, :, :]   # β_{x,g} · κ_{g,t}
                )

                return logmu, alpha, beta, beta_g

            # -----------------------------------------------------------------
            # NR UPDATES — FULL LI-LEE MODEL
            # -----------------------------------------------------------------
            @staticmethod
            def update_alpha_coef(alpha_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of :math:`\\alpha_{x,g}` (region-specific baseline,
                one coefficient vector per region).

                For each region :math:`g` and basis function :math:`j`:

                .. math::

                    \\text{num}_{j,g} = \\sum_{x,t} r_{x,t}^{(g)} B_j(x)
                        - 2\\lambda (D^\\top D \\, c^{\\alpha}_g)_j

                .. math::

                    \\text{den}_{j,g} = \\sum_{x,t} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} B_j(x)^2
                        + 2\\lambda (D^\\top D)_{jj}

                .. math::

                    c^{\\alpha}_{g,j} \\leftarrow c^{\\alpha}_{g,j}
                        + \\eta \\, \\frac{\\text{num}_{j,g}}{\\text{den}_{j,g}}

                :param alpha_coef: Current coefficients, shape ``(nb_regions, n_basis)``.
                :type alpha_coef: numpy.ndarray
                :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float
                :param lam: P-spline penalty :math:`\\lambda`.
                :type lam: float
                :param DtD: Penalty matrix :math:`D^\\top D`, shape ``(n_basis, n_basis)``.
                :type DtD: numpy.ndarray
                :param diag_DtD: Diagonal of ``DtD``, shape ``(n_basis,)``.
                :type diag_DtD: numpy.ndarray

                :returns: Updated ``alpha_coef``, shape ``(nb_regions, n_basis)``.
                :rtype: numpy.ndarray
                """
                nb_regions     = alpha_coef.shape[0]
                n_basis        = alpha_coef.shape[1]
                alpha_coef_new = alpha_coef.copy()

                for g in range(nb_regions):
                    pen_grad = 2.0 * lam * (DtD @ alpha_coef[g]) if lam > 0 else np.zeros(n_basis)
                    for j in range(n_basis):
                        Bj3d = B[:, j][:, None, None]
                        num  = float(np.sum(residual[:, :, g:g+1]     * Bj3d)) - pen_grad[j]
                        den  = float(np.sum(weighted_exp[:, :, g:g+1] * Bj3d**2))
                        if lam > 0:
                            den += 2.0 * lam * diag_DtD[j]
                        if den != 0:
                            alpha_coef_new[g, j] += eta * num / den
                return alpha_coef_new

            @staticmethod
            def update_beta_coef(beta_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of :math:`\\beta_x` (common sensitivity,
                a single coefficient vector for all regions).

                The gradient is summed over all regions :math:`g`:

                .. math::

                    \\text{num}_j = \\sum_{x,t,g} r_{x,t}^{(g)} B_j(x) \\kappa_t
                        - 2\\lambda (D^\\top D \\, c^{\\beta})_j

                .. math::

                    \\text{den}_j = \\sum_{x,t,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)}
                        \\bigl(B_j(x) \\kappa_t\\bigr)^2
                        + 2\\lambda (D^\\top D)_{jj}

                .. math::

                    c^{\\beta}_j \\leftarrow c^{\\beta}_j
                        + \\eta \\, \\frac{\\text{num}_j}{\\text{den}_j}

                :param beta_coef: Current coefficients, shape ``(n_basis,)``.
                :type beta_coef: numpy.ndarray
                :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param kappa: Common time index :math:`\\kappa_t`, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float
                :param lam: P-spline penalty :math:`\\lambda`.
                :type lam: float
                :param DtD: Penalty matrix :math:`D^\\top D`, shape ``(n_basis, n_basis)``.
                :type DtD: numpy.ndarray
                :param diag_DtD: Diagonal of ``DtD``, shape ``(n_basis,)``.
                :type diag_DtD: numpy.ndarray

                :returns: Updated ``beta_coef``, shape ``(n_basis,)``.
                :rtype: numpy.ndarray
                """
                n_basis       = len(beta_coef)
                nb_ages       = B.shape[0]
                beta_coef_new = beta_coef.copy()
                pen_grad      = 2.0 * lam * (DtD @ beta_coef) if lam > 0 else np.zeros(n_basis)

                kappaM = np.repeat(kappa[None, :], nb_ages, axis=0)

                for j in range(n_basis):
                    BjKappa = B[:, j][:, None] * kappaM
                    BjK3d   = BjKappa[:, :, None]

                    num = float(np.sum(residual     * BjK3d)) - pen_grad[j]
                    den = float(np.sum(weighted_exp * BjK3d**2))
                    if lam > 0:
                        den += 2.0 * lam * diag_DtD[j]
                    if den != 0:
                        beta_coef_new[j] += eta * num / den
                return beta_coef_new

            @staticmethod
            def update_beta_g_coef(beta_g_coef, B, kappa_g, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of :math:`\\beta_{x,g}` (regional sensitivity,
                one coefficient vector per region).

                For each region :math:`g` and basis function :math:`j`:

                .. math::

                    \\text{num}_{j,g} = \\sum_{x,t} r_{x,t}^{(g)} B_j(x) \\kappa_{g,t}
                        - 2\\lambda (D^\\top D \\, c^{\\beta_g}_g)_j

                .. math::

                    \\text{den}_{j,g} = \\sum_{x,t} E_{x,t}^{(g)} \\mu_{x,t}^{(g)}
                        \\bigl(B_j(x) \\kappa_{g,t}\\bigr)^2
                        + 2\\lambda (D^\\top D)_{jj}

                .. math::

                    c^{\\beta_g}_{g,j} \\leftarrow c^{\\beta_g}_{g,j}
                        + \\eta \\, \\frac{\\text{num}_{j,g}}{\\text{den}_{j,g}}

                :param beta_g_coef: Current coefficients, shape ``(nb_regions, n_basis)``.
                :type beta_g_coef: numpy.ndarray
                :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param kappa_g: Regional time indices :math:`\\kappa_{g,t}`,
                    shape ``(nb_regions, nb_years)``.
                :type kappa_g: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float
                :param lam: P-spline penalty :math:`\\lambda`.
                :type lam: float
                :param DtD: Penalty matrix :math:`D^\\top D`, shape ``(n_basis, n_basis)``.
                :type DtD: numpy.ndarray
                :param diag_DtD: Diagonal of ``DtD``, shape ``(n_basis,)``.
                :type diag_DtD: numpy.ndarray

                :returns: Updated ``beta_g_coef``, shape ``(nb_regions, n_basis)``.
                :rtype: numpy.ndarray
                """
                nb_regions      = beta_g_coef.shape[0]
                nb_ages         = B.shape[0]
                beta_g_coef_new = beta_g_coef.copy()

                for g in range(nb_regions):
                    pen_grad = 2.0 * lam * (DtD @ beta_g_coef[g]) if lam > 0 else np.zeros(beta_g_coef.shape[1])
                    kappaM   = np.repeat(kappa_g[g, None, :], nb_ages, axis=0)  # (nb_ages, nb_years)

                    for j in range(beta_g_coef.shape[1]):
                        BjKappa = B[:, j][:, None] * kappaM
                        BjK3d   = BjKappa[:, :, None]

                        num = float(np.sum(residual[:, :, g:g+1]     * BjK3d)) - pen_grad[j]
                        den = float(np.sum(weighted_exp[:, :, g:g+1] * BjK3d**2))
                        if lam > 0:
                            den += 2.0 * lam * diag_DtD[j]
                        if den != 0:
                            beta_g_coef_new[g, j] += eta * num / den
                return beta_g_coef_new

            @staticmethod
            def update_kappa(kappa, beta, residual, weighted_exp, eta):
                """
                Newton-Raphson update of :math:`\\kappa_t` (common time factor).

                The score and Fisher information, summed over all ages and regions:

                .. math::

                    \\text{num}_t = \\sum_{x,g} r_{x,t}^{(g)} \\beta_x

                .. math::

                    \\text{den}_t = \\sum_{x,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} \\beta_x^2

                .. math::

                    \\kappa_t \\leftarrow \\kappa_t
                        + \\eta \\, \\frac{\\text{num}_t}{\\text{den}_t}

                :param kappa: Current time index, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param beta: Evaluated common sensitivity :math:`\\beta_x`, shape ``(nb_ages,)``.
                :type beta: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float

                :returns: Updated :math:`\\kappa_t`, shape ``(nb_years,)``.
                :rtype: numpy.ndarray
                """
                beta3d  = beta[:, None, None]
                num_k   = np.sum(residual     * beta3d, axis=(0, 2))
                den_k   = np.sum(weighted_exp * beta3d**2, axis=(0, 2))
                kappa_new       = kappa.copy()
                mask            = den_k != 0
                kappa_new[mask] += eta * num_k[mask] / den_k[mask]
                return kappa_new

            @staticmethod
            def update_kappa_g(kappa_g, beta_g, residual, weighted_exp, eta):
                """
                Newton-Raphson update of :math:`\\kappa_{g,t}` (regional time factors).

                For each region :math:`g` and time :math:`t`:

                .. math::

                    \\text{num}_{g,t} = \\sum_x r_{x,t}^{(g)} \\beta_{x,g}

                .. math::

                    \\text{den}_{g,t} = \\sum_x E_{x,t}^{(g)} \\mu_{x,t}^{(g)} \\beta_{x,g}^2

                .. math::

                    \\kappa_{g,t} \\leftarrow \\kappa_{g,t}
                        + \\eta \\, \\frac{\\text{num}_{g,t}}{\\text{den}_{g,t}}

                :param kappa_g: Current regional time indices,
                    shape ``(nb_regions, nb_years)``.
                :type kappa_g: numpy.ndarray
                :param beta_g: Evaluated regional sensitivity :math:`\\beta_{x,g}`,
                    shape ``(nb_ages, nb_regions)``.
                :type beta_g: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float

                :returns: Updated :math:`\\kappa_{g,t}`, shape ``(nb_regions, nb_years)``.
                :rtype: numpy.ndarray
                """
                nb_regions  = kappa_g.shape[0]
                kappa_g_new = kappa_g.copy()

                for g in range(nb_regions):
                    beta_g3d = beta_g[:, g][:, None, None]
                    num_k    = np.sum(residual[:, :, g:g+1]     * beta_g3d, axis=0).squeeze()
                    den_k    = np.sum(weighted_exp[:, :, g:g+1] * beta_g3d**2, axis=0).squeeze()
                    mask     = den_k != 0
                    kappa_g_new[g, mask] += eta * num_k[mask] / den_k[mask]

                return kappa_g_new

            # -----------------------------------------------------------------
            # NORMALIZATION (IDENTIFIABILITY CONSTRAINTS)
            # -----------------------------------------------------------------
            @staticmethod
            def normalize_lilee(alpha_coef, beta_coef, beta_g_coef, kappa, kappa_g, B):
                """
                Enforce Li-Lee identifiability constraints.

                Three constraints are applied to ensure the model is identified:

                **1. Common factor scale** — :math:`\\sum_x \\beta_x = 1`

                .. math::

                    s = \\sum_x \\beta_x, \\quad
                    \\beta_x \\leftarrow \\frac{\\beta_x}{s}, \\quad
                    \\kappa_t \\leftarrow s \\, \\kappa_t

                **2. Regional factor scale** — :math:`\\sum_x \\beta_{x,g} = 1 \\; \\forall g`

                .. math::

                    s_g = \\sum_x \\beta_{x,g}, \\quad
                    \\beta_{x,g} \\leftarrow \\frac{\\beta_{x,g}}{s_g}, \\quad
                    \\kappa_{g,t} \\leftarrow s_g \\, \\kappa_{g,t}

                **3. Common index centering** — :math:`\\overline{\\kappa}_t = 0`

                .. math::

                    \\bar{\\kappa} = \\frac{1}{T}\\sum_t \\kappa_t, \\quad
                    \\kappa_t \\leftarrow \\kappa_t - \\bar{\\kappa}, \\quad
                    \\alpha_{x,g} \\leftarrow \\alpha_{x,g} + \\beta_x \\, \\bar{\\kappa}

                so that :math:`\\ln \\mu_{x,t,g}` remains invariant.
                """
                # ----------------------------------------------------------
                # 1. Σ_x β_x = 1 : scale of the common factor
                #    Identical to classic Lee-Carter
                # ----------------------------------------------------------
                beta      = B @ beta_coef
                scal_beta = float(np.sum(beta))
                if scal_beta != 0:
                    beta_coef = beta_coef / scal_beta
                    kappa     = kappa * scal_beta

                # ----------------------------------------------------------
                # 2. Σ_x β_{x,g} = 1 ∀g : scale of the regional factors
                #    Symmetric to the constraint on β_x
                # ----------------------------------------------------------
                nb_regions = beta_g_coef.shape[0]
                for g in range(nb_regions):
                    beta_g  = B @ beta_g_coef[g]
                    scal_bg = float(np.sum(beta_g))
                    if scal_bg != 0:
                        beta_g_coef[g] = beta_g_coef[g] / scal_bg
                        kappa_g[g]     = kappa_g[g] * scal_bg

                # ----------------------------------------------------------
                # 3. mean(κ_t) = 0 : centre κ_t
                #    Compensation in α_{x,g} via β_x (non-zero, sum=1)
                #    logmu invariant: α_{x,g} += β_x * mean(κ_t)
                # ----------------------------------------------------------
                mean_kappa = float(np.mean(kappa))
                if mean_kappa != 0:
                    kappa -= mean_kappa
                    beta_x = B @ beta_coef                              # (nb_ages,)
                    alpha_adjustment = beta_x * mean_kappa              # (nb_ages,)
                    adj_coef = np.linalg.lstsq(B, alpha_adjustment, rcond=None)[0]
                    for g in range(nb_regions):
                        alpha_coef[g] += adj_coef                       # same adjustment for all regions

                return alpha_coef, beta_coef, beta_g_coef, kappa, kappa_g

            # -----------------------------------------------------------------
            # ROBUST INITIALISATION — B-spline projection of SVD estimates
            # -----------------------------------------------------------------
            def init_params(self, Dxtg, Extg, xv):
                """
                Robust initialisation for the parametric Li-Lee model.

                Strategy:

                1. Aggregate data across regions → fit a common Lee-Carter (SVD)
                   to extract :math:`\\alpha_{\\text{common}}`, :math:`\\beta_x`, :math:`\\kappa_t`.
                2. For each region, compute the residual log-rate after removing
                   the common component, then apply SVD to extract
                   :math:`\\beta_{x,g}`, :math:`\\kappa_{g,t}`.
                3. Project all age curves onto the B-spline basis via least squares.

                :param Dxtg: Death counts, shape ``(nb_ages, nb_years, nb_regions)``.
                :type Dxtg: numpy.ndarray
                :param Extg: Exposures, same shape as ``Dxtg``.
                :type Extg: numpy.ndarray
                :param xv: Age vector, shape ``(nb_ages,)``.
                :type xv: numpy.ndarray

                :returns:
                    - **alpha_coef** (*numpy.ndarray*, shape ``(nb_regions, n_basis)``)
                    - **beta_coef** (*numpy.ndarray*, shape ``(n_basis,)``)
                    - **beta_g_coef** (*numpy.ndarray*, shape ``(nb_regions, n_basis)``)
                    - **kappa** (*numpy.ndarray*, shape ``(nb_years,)``)
                    - **kappa_g** (*numpy.ndarray*, shape ``(nb_regions, nb_years)``)
                """
                nb_ages, nb_years, nb_regions = Dxtg.shape

                xmin = self.xmin if self.xmin is not None else float(np.min(xv))
                xmax = self.xmax if self.xmax is not None else float(np.max(xv))

                # =====================================================
                # Step 1 — AGGREGATED RATES → common component
                # Equal weighting across regions (prevents large regions
                # such as FR10 from dominating the common factor)
                # =====================================================
                Mxtg_all = Dxtg / np.maximum(Extg, 1e-12)   # (nb_ages, nb_years, nb_regions)
                Mxtg_all = np.maximum(Mxtg_all, 1e-12)

                # Simple average of rates across regions — each region has weight 1/G
                Mxt  = Mxtg_all.mean(axis=2)                 # (nb_ages, nb_years)
                Mxt  = np.maximum(Mxt, 1e-12)

                logM = np.log(Mxt)

                # Common alpha (age baseline averaged over time)
                alpha_common = np.mean(logM, axis=1)

                M_centered = logM - alpha_common[:, None]

                # Common SVD
                U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

                beta_common  = U[:, 0]
                kappa_common = S[0] * Vt[0, :]

                # Sign correction: force kappa_common to have a negative drift
                # (mortality decreasing over time)
                sign_common = np.sign(np.mean(np.diff(kappa_common)))
                if sign_common > 0:   # kappa rises → flip
                    beta_common  *= -1
                    kappa_common *= -1

                # Standard LC normalisation
                beta_common  /= np.sum(beta_common)
                kappa_common *= np.sum(beta_common)

                # =====================================================
                # Step 2 — REGIONAL DEVIATIONS
                # =====================================================
                beta_g  = np.zeros((nb_ages, nb_regions))
                kappa_g = np.zeros((nb_regions, nb_years))

                for g in range(nb_regions):

                    Mxtg   = Dxtg[:, :, g] / np.maximum(Extg[:, :, g], 1e-12)
                    Mxtg   = np.maximum(Mxtg, 1e-12)
                    logM_g = np.log(Mxtg)

                    # Remove common component
                    residual = (
                        logM_g
                        - alpha_common[:, None]
                        - np.outer(beta_common, kappa_common)
                    )

                    U_g, S_g, Vt_g = np.linalg.svd(residual, full_matrices=False)

                    # Sign correction only: force beta_g to be positive on average
                    sign = np.sign(np.sum(U_g[:, 0]))
                    if sign == 0:
                        sign = 1.0
                    U_g[:, 0]  *= sign
                    Vt_g[0, :] *= sign

                    beta_g[:, g]  = U_g[:, 0]
                    kappa_g[g, :] = S_g[0] * Vt_g[0, :]

                    # Original normalisation (identical behaviour to the original)
                    beta_g[:, g]  /= np.sum(beta_g[:, g])
                    kappa_g[g, :] *= np.sum(beta_g[:, g])

                # =====================================================
                # Step 3 — PROJECTION ONTO B-SPLINE BASIS
                # =====================================================
                B, knots, n_basis = make_bspline_basis(xv, self.degree, self.n_knots, xmin, xmax)

                # Alpha per region (starting point: common alpha replicated)
                alpha_coef = np.zeros((nb_regions, n_basis))
                for g in range(nb_regions):
                    alpha_coef[g] = np.linalg.lstsq(B, alpha_common, rcond=None)[0]

                # Common beta
                beta_coef = np.linalg.lstsq(B, beta_common, rcond=None)[0]

                # Regional beta
                beta_g_coef = np.zeros((nb_regions, n_basis))
                for g in range(nb_regions):
                    beta_g_coef[g] = np.linalg.lstsq(B, beta_g[:, g], rcond=None)[0]

                return alpha_coef, beta_coef, beta_g_coef, kappa_common, kappa_g

            # -----------------------------------------------------------------
            # MAIN FITTING METHOD — FullModel
            # -----------------------------------------------------------------
            def fit(
                self,
                alpha_coef_init,   # (nb_regions, n_basis)
                beta_coef_init,    # (n_basis,)
                beta_g_coef_init,  # (nb_regions, n_basis)
                kappa_init,        # (nb_years,)
                kappa_g_init,      # (nb_regions, nb_years)
                Extg,              # (nb_ages, nb_years, nb_regions)
                Dxtg,              # (nb_ages, nb_years, nb_regions)
                xv,                # (nb_ages,)
                tv,                # (nb_years,)
            ):
                """
                Calibrate the full parametric Li-Lee model:

                .. math::

                    \\ln(\\mu_{x,t,g}) = \\alpha_{x,g} + \\beta_x \\cdot \\kappa_t
                                        + \\beta_{x,g} \\cdot \\kappa_{g,t}

                :param alpha_coef_init: Initial coefficients of :math:`\\alpha_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type alpha_coef_init: numpy.ndarray
                :param beta_coef_init: Initial coefficients of :math:`\\beta_x`,
                    shape ``(n_basis,)``.
                :type beta_coef_init: numpy.ndarray
                :param beta_g_coef_init: Initial coefficients of :math:`\\beta_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type beta_g_coef_init: numpy.ndarray
                :param kappa_init: Initial common time index, shape ``(nb_years,)``.
                :type kappa_init: numpy.ndarray
                :param kappa_g_init: Initial regional time indices,
                    shape ``(nb_regions, nb_years)``.
                :type kappa_g_init: numpy.ndarray
                :param Extg: Exposures, shape ``(nb_ages, nb_years, nb_regions)``.
                :type Extg: numpy.ndarray
                :param Dxtg: Death counts, same shape as ``Extg``.
                :type Dxtg: numpy.ndarray
                :param xv: Age vector, shape ``(nb_ages,)``.
                :type xv: numpy.ndarray
                :param tv: Year vector, shape ``(nb_years,)``.
                :type tv: numpy.ndarray

                :returns: Dictionary with keys:

                    - ``"parameters"`` — ``alpha_coef``, ``beta_coef``,
                      ``beta_g_coef``, ``kappa``, ``kappa_g``
                    - ``"curves"`` — ``alpha_xg``, ``beta_x``, ``beta_xg``
                    - ``"fitted_values"`` — ``log_mu``, ``mu``
                    - ``"fit_statistics"`` — :class:`pandas.DataFrame`

                :rtype: dict
                :raises ValueError: If the shapes of the initial coefficient arrays
                    are inconsistent with the computed B-spline basis.
                """
                nb_years   = len(tv)
                nb_regions = Extg.shape[2]

                xmin = self.xmin if self.xmin is not None else float(np.min(xv))
                xmax = self.xmax if self.xmax is not None else float(np.max(xv))

                # Build B-spline basis matrix
                B, knots, n_basis = make_bspline_basis(xv, self.degree, self.n_knots, xmin, xmax)

                # Dimension checks
                if alpha_coef_init.shape != (nb_regions, n_basis):
                    raise ValueError(f"alpha_coef_init must have shape ({nb_regions}, {n_basis})")
                if len(beta_coef_init) != n_basis:
                    raise ValueError(f"beta_coef_init must have {n_basis} elements")
                if beta_g_coef_init.shape != (nb_regions, n_basis):
                    raise ValueError(f"beta_g_coef_init must have shape ({nb_regions}, {n_basis})")

                alpha_coef  = alpha_coef_init.copy()
                beta_coef   = beta_coef_init.copy()
                beta_g_coef = beta_g_coef_init.copy()
                kappa       = kappa_init.copy()
                kappa_g     = kappa_g_init.copy()

                # P-splines penalty
                DtD, diag_DtD = LiLee.make_penalty_matrix(n_basis, self.diff_order)

                # Log-factorial constant
                logDxtgFact = gammaln(Dxtg + 1)

                lnL       = 0.0
                Delta_lnL = -1000.0
                flag      = 0
                it        = -1
                eta       = self.eta0

                # Debug: track regional contribution (index 0 = first region)
                # Change _debug_idx if FR10 is not at index 0
                _debug_idx     = 0
                _debug_contrib = []

                if self.verbose:
                    print("=" * 70)
                    print("PARAMETRIC LI-LEE MODEL CALIBRATION")
                    print("=" * 70)
                    print(f"Parameters: degree={self.degree}, n_knots={self.n_knots}, lam={self.lam}")
                    print(f"Data: {Dxtg.shape[0]} ages × {nb_years} years × {nb_regions} regions")
                    print(f"Number of basis functions: {n_basis}")
                    print("=" * 70)

                # NR loop
                while (it < self.nb_iter) and (flag < 4):
                    it += 1

                    # Adaptive learning rate
                    if Delta_lnL < 0:
                        eta *= 0.5
                    else:
                        eta = min(eta * 1.05, 2.0)

                    # Stopping criterion
                    if np.abs(Delta_lnL) < self.tol:
                        flag += 1
                    else:
                        flag = 0

                    # Reconstruct ln(µ)
                    logmu, alpha, beta, beta_g = self.compute_logmu_lilee(
                        alpha_coef, beta_coef, beta_g_coef,
                        kappa, kappa_g, xv, B, knots, self.degree,
                    )

                    # Log-likelihood
                    lnL_new, _, weighted_exp, residual = LiLee.poisson_lnL(
                        Dxtg, Extg, logmu, logDxtgFact
                    )

                    Delta_lnL = lnL_new - lnL
                    lnL       = lnL_new

                    if self.verbose and (it % 10 == 0):
                        print(f"It {it:4d} | lnL = {lnL:,.2f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

                    # Sequential NR updates
                    alpha_coef = self.update_alpha_coef(
                        alpha_coef, B, residual, weighted_exp, eta, self.lam, DtD, diag_DtD
                    )
                    beta_coef = self.update_beta_coef(
                        beta_coef, B, kappa, residual, weighted_exp, eta, self.lam, DtD, diag_DtD
                    )
                    beta_g_coef = self.update_beta_g_coef(
                        beta_g_coef, B, kappa_g, residual, weighted_exp, eta, self.lam, DtD, diag_DtD
                    )
                    kappa   = self.update_kappa(kappa, beta, residual, weighted_exp, eta)
                    kappa_g = self.update_kappa_g(kappa_g, beta_g, residual, weighted_exp, eta)

                    # Final normalization
                    alpha_coef, beta_coef, beta_g_coef, kappa, kappa_g = self.normalize_lilee(
                        alpha_coef, beta_coef, beta_g_coef, kappa, kappa_g, B
                    )

                    # Debug: track regional contribution
                    _bxg     = B @ beta_g_coef[_debug_idx]
                    _kg      = kappa_g[_debug_idx, :]
                    _contrib = (_bxg[:, None] * _kg[None, :]).mean()
                    _debug_contrib.append(_contrib)

                # Final reconstruction
                logmu_final, alpha, beta, beta_g = self.compute_logmu_lilee(
                    alpha_coef, beta_coef, beta_g_coef,
                    kappa, kappa_g, xv, B, knots, self.degree,
                )

                # Effective dofs — evaluated at final Poisson weights
                w_final = Extg * np.exp(logmu_final)          # (nb_ages, nb_years, nb_regions)

                dof_alpha = sum(                               # α_{x,g} : one block per region
                    LiLee.effective_dof_spline_block(B, w_final[:, :, g].sum(axis=1), self.lam, DtD)
                    for g in range(nb_regions)
                )
                dof_beta = LiLee.effective_dof_spline_block(   # β_x : single common block
                    B, w_final.sum(axis=(1, 2)), self.lam, DtD
                )
                dof_beta_g = sum(                              # β_{x,g} : one block per region
                    LiLee.effective_dof_spline_block(B, w_final[:, :, g].sum(axis=1), self.lam, DtD)
                    for g in range(nb_regions)
                )
                dof_kappa   = nb_years                        # κ_t    : unpenalized
                dof_kappa_g = nb_years * nb_regions           # κ_{g,t}: unpenalized
                # identifiability constraints (exact regardless of λ)
                dof_constraints = 1 + nb_regions + nb_regions  # Σβ_x=1, Σβ_{x,g}=0 ∀g, Σκ_{g,t}=0 ∀g

                dofs = (dof_alpha + dof_beta + dof_beta_g
                        + dof_kappa + dof_kappa_g
                        - dof_constraints)

                # Statistics
                Fit_stat = LiLee.compute_fit_stats(
                    Dxtg, Extg, logmu_final, logDxtgFact, dofs,
                )
                if self.verbose:
                    print("\n" + "=" * 70)
                    print("CALIBRATION COMPLETE")
                    print("=" * 70)
                    print(f"Convergence reached after {it + 1} iterations")
                    print("\nFinal statistics:")
                    print(Fit_stat.to_string(index=False))
                    print("=" * 70)

                return {
                    "parameters": {
                        "alpha_coef":  alpha_coef,
                        "beta_coef":   beta_coef,
                        "beta_g_coef": beta_g_coef,
                        "kappa":       kappa,
                        "kappa_g":     kappa_g,
                    },
                    "curves": {
                        "alpha_xg": alpha,   # (nb_ages, nb_regions)
                        "beta_x":   beta,    # (nb_ages,)
                        "beta_xg":  beta_g,  # (nb_ages, nb_regions)
                    },
                    "fitted_values": {
                        "log_mu": logmu_final,
                        "mu":     np.exp(logmu_final),
                    },
                    "fit_statistics": Fit_stat,
                    "debug_contrib":  _debug_contrib,  # contribution tracking for region _debug_idx
                }


        # =====================================================================
        # VARIANT : LeeAndLi
        # Simplified Lee & Li model: α_x + β_{x,g}·κ_t
        # =====================================================================

        class Variant:
            """
            Simplified Lee & Li parametric model — single common time index.

            .. math::

                \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_{x,g} \\cdot \\kappa_t

            where:

            - :math:`\\alpha_x`    — **common** age baseline (no region dimension)
            - :math:`\\beta_{x,g}` — **region-specific** age sensitivity
              (one B-spline curve per region :math:`g`)
            - :math:`\\kappa_t`    — **single** common time index

            .. note::

                This variant was previously implemented as
                ``LeeCarter.Parametric.Multiregion`` in ``lcp_class.py``.

            :param degree: B-spline degree.
            :type degree: int
            :param n_knots: Number of internal knots.
            :type n_knots: int
            :param xmin: Minimum age bound (inferred from data if ``None``).
            :type xmin: float or None
            :param xmax: Maximum age bound (inferred from data if ``None``).
            :type xmax: float or None
            :param lam: P-splines penalty parameter.
            :type lam: float
            :param diff_order: Order of the penalty difference operator.
            :type diff_order: int
            :param nb_iter: Maximum number of iterations.
            :type nb_iter: int
            :param eta0: Initial learning rate.
            :type eta0: float
            :param tol: Convergence tolerance on the log-likelihood increment.
            :type tol: float
            :param verbose: Whether to print iteration progress.
            :type verbose: bool
            """

            def __init__(
                self,
                degree=2,
                n_knots=10,
                xmin=None,
                xmax=None,
                lam=0.0,
                diff_order=2,
                nb_iter=800,
                eta0=0.2,
                tol=1e-3,
                verbose=False,
            ):
                self.degree     = degree
                self.n_knots    = n_knots
                self.xmin       = xmin
                self.xmax       = xmax
                self.lam        = lam
                self.diff_order = diff_order
                self.nb_iter    = nb_iter
                self.eta0       = eta0
                self.tol        = tol
                self.verbose    = verbose

            # -----------------------------------------------------------------
            # ln(µ) RECONSTRUCTION — LEE & LI MODEL
            # -----------------------------------------------------------------
            @staticmethod
            def compute_logmu(ax_coef, bx_coef, kappa, xv, B, knots, degree):
                """
                Reconstruct :math:`\\ln(\\mu_{x,t,g})` for the Lee & Li model:

                .. math::

                    \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_{x,g} \\cdot \\kappa_t

                :param ax_coef: B-spline coefficients of :math:`\\alpha_x`,
                    shape ``(n_basis,)``.
                :type ax_coef: numpy.ndarray
                :param bx_coef: B-spline coefficients of :math:`\\beta_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type bx_coef: numpy.ndarray
                :param kappa: Common time index :math:`\\kappa_t`, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param xv: Age vector, shape ``(nb_ages,)``.
                :type xv: numpy.ndarray
                :param B: Pre-computed B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param knots: Full knot vector.
                :type knots: numpy.ndarray
                :param degree: B-spline degree.
                :type degree: int

                :returns:
                    - **logmu** (*numpy.ndarray*, shape ``(nb_ages, nb_years, nb_regions)``)
                    - **ax** (*numpy.ndarray*, shape ``(nb_ages,)``)
                      — evaluated :math:`\\alpha_x`
                    - **bx_reg** (*numpy.ndarray*, shape ``(nb_ages, nb_regions)``)
                      — evaluated :math:`\\beta_{x,g}`
                """
                nb_regions = bx_coef.shape[0]
                nb_ages    = len(xv)

                # Evaluate α_x (common baseline)
                ax = B @ ax_coef

                # Evaluate β_{x,g} for each region
                bx_reg = np.zeros((nb_ages, nb_regions))
                for g in range(nb_regions):
                    bx_reg[:, g] = B @ bx_coef[g]

                # Build ln(µ) by broadcasting
                logmu = (
                    ax[:, None, None]
                    + bx_reg[:, None, :] * kappa[None, :, None]
                )
                return logmu, ax, bx_reg

            # -----------------------------------------------------------------
            # NR UPDATE — bx_coef (one curve per region)
            # -----------------------------------------------------------------
            @staticmethod
            def update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of :math:`\\beta_{x,g}` (one B-spline curve per region).

                For each region :math:`g` and basis function :math:`j`:

                .. math::

                    \\text{num}_{j,g} = \\sum_{x,t} r_{x,t}^{(g)} B_j(x) \\kappa_t
                        - 2\\lambda (D^\\top D \\, c^{\\beta_g}_g)_j

                .. math::

                    \\text{den}_{j,g} = \\sum_{x,t} E_{x,t}^{(g)} \\mu_{x,t}^{(g)}
                        \\bigl(B_j(x) \\kappa_t\\bigr)^2
                        + 2\\lambda (D^\\top D)_{jj}

                .. math::

                    c^{\\beta_g}_{g,j} \\leftarrow c^{\\beta_g}_{g,j}
                        + \\eta \\, \\frac{\\text{num}_{j,g}}{\\text{den}_{j,g}}

                :param bx_coef: Current coefficients, shape ``(nb_regions, n_basis)``.
                :type bx_coef: numpy.ndarray
                :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param kappa: Common time index :math:`\\kappa_t`, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float
                :param lam: P-spline penalty :math:`\\lambda`.
                :type lam: float
                :param DtD: Penalty matrix :math:`D^\\top D`, shape ``(n_basis, n_basis)``.
                :type DtD: numpy.ndarray
                :param diag_DtD: Diagonal of ``DtD``, shape ``(n_basis,)``.
                :type diag_DtD: numpy.ndarray

                :returns: Updated ``bx_coef``, shape ``(nb_regions, n_basis)``.
                :rtype: numpy.ndarray
                """
                nb_regions  = bx_coef.shape[0]
                nb_ages     = B.shape[0]
                bx_coef_new = bx_coef.copy()
                kappaM      = np.repeat(kappa[None, :], nb_ages, axis=0)

                for g in range(nb_regions):
                    pen_grad = 2.0 * lam * (DtD @ bx_coef[g]) if lam > 0 else np.zeros(bx_coef.shape[1])
                    for j in range(bx_coef.shape[1]):
                        BjKappa = B[:, j][:, None] * kappaM
                        BjK3d   = BjKappa[:, :, None]
                        num = float(np.sum(residual[:, :, g:g+1]     * BjK3d)) - pen_grad[j]
                        den = float(np.sum(weighted_exp[:, :, g:g+1] * BjK3d**2))
                        if lam > 0:
                            den += 2.0 * lam * diag_DtD[j]
                        if den != 0:
                            bx_coef_new[g, j] += eta * num / den
                return bx_coef_new

            # -----------------------------------------------------------------
            # NR UPDATE — ax_coef (common baseline)
            # -----------------------------------------------------------------
            @staticmethod
            def update_ax_coef(ax_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of the B-spline coefficients of :math:`\\alpha_x`.

                Since :math:`\\alpha_x` is common to all regions, the gradient
                is summed over all :math:`g`:

                .. math::

                    \\text{num}_j = \\sum_{x,t,g} r_{x,t}^{(g)} B_j(x)
                        - 2\\lambda (D^\\top D \\, c^{\\alpha})_j

                .. math::

                    \\text{den}_j = \\sum_{x,t,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} B_j(x)^2
                        + 2\\lambda (D^\\top D)_{jj}

                .. math::

                    c^{\\alpha}_j \\leftarrow c^{\\alpha}_j
                        + \\eta \\, \\frac{\\text{num}_j}{\\text{den}_j}

                :param ax_coef: Current coefficients, shape ``(n_basis,)``.
                :type ax_coef: numpy.ndarray
                :param B: B-spline basis matrix, shape ``(nb_ages, n_basis)``.
                :type B: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float
                :param lam: P-spline penalty :math:`\\lambda`.
                :type lam: float
                :param DtD: Penalty matrix :math:`D^\\top D`, shape ``(n_basis, n_basis)``.
                :type DtD: numpy.ndarray
                :param diag_DtD: Diagonal of ``DtD``, shape ``(n_basis,)``.
                :type diag_DtD: numpy.ndarray

                :returns: Updated ``ax_coef``, shape ``(n_basis,)``.
                :rtype: numpy.ndarray
                """
                n_basis      = len(ax_coef)
                ax_coef_new  = ax_coef.copy()
                pen_grad     = 2.0 * lam * (DtD @ ax_coef) if lam > 0 else np.zeros(n_basis)

                for j in range(n_basis):
                    Bj3d = B[:, j][:, None, None]
                    num  = float(np.sum(residual    * Bj3d)) - pen_grad[j]
                    den  = float(np.sum(weighted_exp * Bj3d**2))
                    if lam > 0:
                        den += 2.0 * lam * diag_DtD[j]
                    if den != 0:
                        ax_coef_new[j] += eta * num / den
                return ax_coef_new

            # -----------------------------------------------------------------
            # NR UPDATE — kappa (single common time index)
            # -----------------------------------------------------------------
            @staticmethod
            def update_kappa(kappa, bx_reg, residual, weighted_exp, eta):
                """
                Newton-Raphson update of the common time index :math:`\\kappa_t`.

                For the Lee & Li variant, all regions share :math:`\\kappa_t`,
                so the gradient is summed over both ages and regions:

                .. math::

                    \\text{num}_t = \\sum_{x,g} r_{x,t}^{(g)} \\beta_{x,g}

                .. math::

                    \\text{den}_t = \\sum_{x,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} \\beta_{x,g}^2

                .. math::

                    \\kappa_t \\leftarrow \\kappa_t
                        + \\eta \\, \\frac{\\text{num}_t}{\\text{den}_t}

                :param kappa: Current time index, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray
                :param bx_reg: Evaluated regional sensitivity :math:`\\beta_{x,g}`,
                    shape ``(nb_ages, nb_regions)``.
                :type bx_reg: numpy.ndarray
                :param residual: Residuals :math:`D - E\\mu`, shape ``(nb_ages, nb_years, nb_regions)``.
                :type residual: numpy.ndarray
                :param weighted_exp: Expected deaths :math:`E\\mu`, same shape as ``residual``.
                :type weighted_exp: numpy.ndarray
                :param eta: Learning rate.
                :type eta: float

                :returns: Updated :math:`\\kappa_t`, shape ``(nb_years,)``.
                :rtype: numpy.ndarray
                """
                bx3d  = bx_reg[:, None, :]
                num_k = np.sum(residual    * bx3d, axis=(0, 2))
                den_k = np.sum(weighted_exp * bx3d**2, axis=(0, 2))
                kappa_new       = kappa.copy()
                mask            = den_k != 0
                kappa_new[mask] += eta * num_k[mask] / den_k[mask]
                return kappa_new

            # -----------------------------------------------------------------
            # FINAL RESCALING
            # -----------------------------------------------------------------
            @staticmethod
            def rescale_bx_kappa(bx_coef, bx_reg, kappa):
                """
                Normalize :math:`\\beta_{x,g}` via the average across regions.

                The identifiability constraint enforced is:

                .. math::

                    \\sum_x \\bar{\\beta}_x = 1, \\qquad
                    \\bar{\\beta}_x = \\frac{1}{G} \\sum_g \\beta_{x,g}

                This is achieved by computing the scale factor
                :math:`s = \\sum_x \\bar{\\beta}_x` and applying:

                .. math::

                    \\beta_{x,g} \\leftarrow \\frac{\\beta_{x,g}}{s}, \\quad
                    c^{\\beta_g} \\leftarrow \\frac{c^{\\beta_g}}{s}, \\quad
                    \\kappa_t \\leftarrow s \\, \\kappa_t

                so that the product :math:`\\beta_{x,g} \\kappa_t` is preserved.

                :param bx_coef: Coefficients of :math:`\\beta_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type bx_coef: numpy.ndarray
                :param bx_reg: Evaluated :math:`\\beta_{x,g}`,
                    shape ``(nb_ages, nb_regions)``.
                :type bx_reg: numpy.ndarray
                :param kappa: Time index, shape ``(nb_years,)``.
                :type kappa: numpy.ndarray

                :returns: Rescaled ``(bx_coef, bx_reg, kappa)``.
                :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
                """
                bx_avg      = np.mean(bx_reg, axis=1)
                scal_factor = float(np.sum(bx_avg))
                if scal_factor == 0:
                    return bx_coef, bx_reg, kappa
                return bx_coef / scal_factor, bx_reg / scal_factor, kappa * scal_factor

            # -----------------------------------------------------------------
            # MAIN FITTING METHOD — LeeAndLi
            # -----------------------------------------------------------------
            def fit(self, ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv):
                """
                Fit the Lee & Li parametric model:

                .. math::

                    \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_{x,g} \\cdot \\kappa_t

                :param ax_coef_init: Initial coefficients of :math:`\\alpha_x`,
                    shape ``(n_basis,)``.
                :type ax_coef_init: numpy.ndarray
                :param bx_coef_init: Initial coefficients of :math:`\\beta_{x,g}`,
                    shape ``(nb_regions, n_basis)``.
                :type bx_coef_init: numpy.ndarray
                :param kappa_init: Initial time index :math:`\\kappa_t`,
                    shape ``(nb_years,)``.
                :type kappa_init: numpy.ndarray
                :param Extg: Exposures, shape ``(nb_ages, nb_years, nb_regions)``.
                :type Extg: numpy.ndarray
                :param Dxtg: Death counts, same shape as ``Extg``.
                :type Dxtg: numpy.ndarray
                :param xv: Age vector, shape ``(nb_ages,)``.
                :type xv: numpy.ndarray
                :param tv: Year vector, shape ``(nb_years,)``.
                :type tv: numpy.ndarray

                :returns: Dictionary with keys:

                    - ``"parameters"`` — ``ax_coef``, ``bx_coef``, ``kappa``
                    - ``"curves"`` — ``alpha_x``, ``beta_xg``
                    - ``"fitted_values"`` — ``log_mu``, ``mu``
                    - ``"fit_statistics"`` — :class:`pandas.DataFrame`

                :rtype: dict
                :raises ValueError: If the shapes of the initial coefficient arrays
                    are inconsistent with the computed B-spline basis.
                """
                nb_years   = len(tv)
                nb_regions = Extg.shape[2]

                xmin = self.xmin if self.xmin is not None else float(np.min(xv))
                xmax = self.xmax if self.xmax is not None else float(np.max(xv))

                B, knots, n_basis = make_bspline_basis(xv, self.degree, self.n_knots, xmin, xmax)

                if len(ax_coef_init) != n_basis:
                    raise ValueError(f"ax_coef_init must have {n_basis} elements, got {len(ax_coef_init)}")
                if bx_coef_init.shape[1] != n_basis:
                    raise ValueError(f"bx_coef_init must have {n_basis} columns, got {bx_coef_init.shape[1]}")

                ax_coef = ax_coef_init.copy()
                bx_coef = bx_coef_init.copy()
                kappa   = kappa_init.copy()

                DtD, diag_DtD = LiLee.make_penalty_matrix(n_basis, self.diff_order)
                logDxtgFact   = gammaln(Dxtg + 1)

                lnL       = -np.inf
                Delta_lnL = 0.0
                eta       = self.eta0
                it        = -1

                # Patience-based early stopping
                best_lnL  = -np.inf
                patience  = 40           # max iterations without improvement
                min_delta = 1e-2         # minimum significant improvement
                wait      = 0

                while it < self.nb_iter:
                    it += 1

                    # Adaptive learning rate
                    if Delta_lnL < 0:
                        eta *= 0.5
                    else:
                        eta = min(eta * 1.05, 2.0)

                    # Reconstruction
                    logmu, ax, bx_reg = self.compute_logmu(
                        ax_coef, bx_coef, kappa, xv, B, knots, self.degree
                    )

                    # Log-likelihood
                    lnL_new, _, weighted_exp, residual = LiLee.poisson_lnL(
                        Dxtg, Extg, logmu, logDxtgFact
                    )

                    Delta_lnL = lnL_new - lnL

                    if self.verbose and (it % 10 == 0):
                        print(f"It {it:4d} | lnL = {lnL_new:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

                    # Early stopping logic
                    if lnL_new > best_lnL + min_delta:
                        best_lnL = lnL_new
                        wait     = 0
                    else:
                        wait += 1

                    if wait >= patience:
                        if self.verbose:
                            print("\nEarly stopping: no more significant improvement.")
                        break

                    # Classic stopping criterion
                    if abs(Delta_lnL) < self.tol:
                        if self.verbose:
                            print("\nConvergence reached (tolerance).")
                        break

                    lnL = lnL_new

                    # Updates
                    ax_coef = self.update_ax_coef(ax_coef, B, residual, weighted_exp, eta, self.lam, DtD, diag_DtD)
                    bx_coef = self.update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, self.lam, DtD, diag_DtD)
                    kappa   = self.update_kappa(kappa, bx_reg, residual, weighted_exp, eta)

                # Final rescaling
                bx_coef, bx_reg, kappa = self.rescale_bx_kappa(bx_coef, bx_reg, kappa)

                logmu_final, ax, bx_reg = self.compute_logmu(
                    ax_coef, bx_coef, kappa, xv, B, knots, self.degree
                )

                # Effective dofs — evaluated at final Poisson weights
                w_final = Extg * np.exp(logmu_final)          # (nb_ages, nb_years, nb_regions)

                dof_alpha = LiLee.effective_dof_spline_block(  # α_x : single common block
                    B, w_final.sum(axis=(1, 2)), self.lam, DtD
                )
                dof_beta_g = sum(                              # β_{x,g} : one block per region
                    LiLee.effective_dof_spline_block(B, w_final[:, :, g].sum(axis=1), self.lam, DtD)
                    for g in range(nb_regions)
                )
                dof_kappa = nb_years                          # κ_t : unpenalized
                # identifiability constraint: Σ_x mean_g(β_{x,g}) = 1
                dof_constraints = 1

                dofs = dof_alpha + dof_beta_g + dof_kappa - dof_constraints

                Fit_stat = LiLee.compute_fit_stats(
                    Dxtg, Extg, logmu_final, logDxtgFact, dofs
                )

                if self.verbose:
                    print("\n" + "="*70)
                    print("FINAL STATISTICS")
                    print("="*70)
                    print(Fit_stat.to_string(index=False))

                return {
                    "parameters": {
                        "ax_coef": ax_coef,
                        "bx_coef": bx_coef,
                        "kappa":   kappa
                    },
                    "curves": {
                        "alpha_x": ax,
                        "beta_xg": bx_reg
                    },
                    "fitted_values": {
                        "log_mu": logmu_final,
                        "mu":     np.exp(logmu_final)
                    },
                    "fit_statistics": Fit_stat
                }


    # =========================================================================
    # SUB-CLASS : Classic
    # Classic non-parametric Li-Lee (gradient descent)
    # First fits a Lee-Carter baseline, then fits the regional residual terms
    # =========================================================================

    class Classic:
        """
        Classic (non-parametric) Li-Lee model fitted by gradient descent.

        Step 1 — fits a standard Lee-Carter model (:math:`\\alpha_x`, :math:`\\beta_x`,
        :math:`\\kappa_t`) on the data.

        Step 2 — fits the regional residual terms (:math:`\\beta_{x,g}`,
        :math:`\\kappa_{g,t}`) on top:

        .. math::

            \\ln(\\mu_{x,t,g}) = \\alpha_x + \\beta_x \\cdot \\kappa_t
                                + \\beta_{x,g} \\cdot \\kappa_{g,t}

        :param nb_iter: Number of gradient descent iterations.
        :type nb_iter: int
        :param h: P-splines roughness penalty for :math:`\\beta_{x,g}`.
        :type h: float
        :param z: Difference order for the penalty matrix.
        :type z: int
        :param verbose: Whether to display plots during fitting.
        :type verbose: bool
        """

        def __init__(self, nb_iter=500, h=0.0, z=2, verbose=False):
            self.nb_iter = nb_iter
            self.h       = h
            self.z       = z
            self.verbose = verbose

        # ---------------------------------------------------------------------
        # INTERNAL HELPER — classic Lee-Carter fit (used as baseline)
        # ---------------------------------------------------------------------
        @staticmethod
        def _lc_fit(ax, bx, kappa, Extg, Dxtg, xv, tv, nb_iter):
            """
            Classic Lee-Carter gradient descent — used as Step 1 baseline.

            Fits :math:`\\alpha_x`, :math:`\\beta_x`, :math:`\\kappa_t` on the
            full dataset by maximising the Poisson log-likelihood via
            coordinate-wise gradient ascent.

            At each iteration the three blocks are updated in sequence:

            1. :math:`\\alpha_x` — gradient step.
            2. :math:`\\beta_x` — gradient step followed by normalization
               :math:`\\sum_x \\beta_x = 1`.
            3. :math:`\\kappa_t` — gradient step followed by centering
               :math:`\\bar{\\kappa} = 0`; the mean is absorbed into :math:`\\alpha_x`:

               .. math::

                   \\alpha_x \\leftarrow \\alpha_x + \\bar{\\kappa} \\, \\beta_x

            :param ax: Initial age baseline :math:`\\alpha_x`, shape ``(nb_ages, 1)``.
            :type ax: numpy.ndarray
            :param bx: Initial sensitivity :math:`\\beta_x`, shape ``(nb_ages, 1)``.
            :type bx: numpy.ndarray
            :param kappa: Initial period index :math:`\\kappa_t`, shape ``(nb_years,)``.
            :type kappa: numpy.ndarray
            :param Extg: Exposures, shape ``(nb_ages, nb_years, nb_regions)``.
            :type Extg: numpy.ndarray
            :param Dxtg: Death counts, same shape as ``Extg``.
            :type Dxtg: numpy.ndarray
            :param xv: Age vector.
            :type xv: numpy.ndarray
            :param tv: Year vector.
            :type tv: numpy.ndarray
            :param nb_iter: Number of gradient descent iterations.
            :type nb_iter: int

            :returns: Updated ``(ax, bx, kappa, Fit_stat)``.
            :rtype: tuple
            """
            #gradient descent step size
            eta = 1
            for it in range(nb_iter):
                for ct_opt in np.arange(0, 3):
                    ax = ax.reshape(-1, 1) ; bx = bx.reshape(-1, 1)
                    nb_regions = Extg.shape[2]
                    axM    = np.repeat(ax, len(tv), axis=1)
                    bxM    = np.repeat(bx, len(tv), axis=1)
                    kappaM = np.repeat(kappa.reshape(1, -1), len(xv), axis=0)
                    logmuxt_baseline = axM + bxM * kappaM
                    logmuxt_grp = np.zeros((len(xv), len(tv), nb_regions))
                    #computation of log(mu(x,t,g))
                    for ct in range(nb_regions):
                        logmuxt_grp[:, :, ct] = logmuxt_baseline.copy()
                    #baseline for update
                    dlnL_baseline = (Dxtg - Extg * np.exp(logmuxt_grp))
                    if (ct_opt == 0):
                        #--------------- ax --------------------
                        ax_new    = np.zeros_like(ax)
                        dlnL_dpar = (np.sum(dlnL_baseline, axis=(1, 2)) /
                                     np.sum(Extg * np.exp(logmuxt_grp), axis=(1, 2)))
                        ax_new = ax + eta * dlnL_dpar.reshape(-1, 1)
                        #update
                        ax = ax_new.copy()
                    if (ct_opt == 1):
                        #--------------- bx --------------------
                        bx_new = np.zeros_like(bx)
                        kappaM = np.repeat(kappa.reshape(1, -1), len(xv), axis=0)
                        kappaM = np.expand_dims(kappaM, axis=2)
                        kappaM = np.repeat(kappaM, nb_regions, axis=2)
                        dlnL_dpar = (np.sum(dlnL_baseline * kappaM, axis=(1, 2)) /
                                     (np.sum(Extg * np.exp(logmuxt_grp) * kappaM**2, axis=(1, 2))))
                        bx_new = bx + eta * dlnL_dpar.reshape(-1, 1)
                        #normalise
                        scal_bx = np.sum(bx_new)
                        bx_new  = bx_new / scal_bx
                        kappa   = kappa * scal_bx
                        bx      = bx_new.copy()
                    if (ct_opt == 2):
                        #---------------Kappa-----------------
                        # warning we use the old betax(x)
                        kappa_new = np.zeros_like(kappa)
                        bxM = np.repeat(bx, len(tv), axis=1)
                        bxM = np.expand_dims(bxM, axis=2)
                        bxM = np.repeat(bxM, nb_regions, axis=2)
                        dlnL_dpar = (np.sum(dlnL_baseline * bxM, axis=(0, 2)) /
                                     (np.sum(Extg * np.exp(logmuxt_grp) * bxM**2, axis=(0, 2))))
                        kappa_new = kappa + eta * dlnL_dpar
                        #rescale
                        kappa_avg = np.mean(kappa_new)
                        kappa_new = (kappa_new - kappa_avg)  #*np.sum(bx)
                        ax        = ax + kappa_avg * bx
                        #update
                        kappa = kappa_new.copy()
            #end loop
            # we recompute log-mort. rates
            ax = ax.reshape(-1, 1) ; bx = bx.reshape(-1, 1)
            axM    = np.repeat(ax, len(tv), axis=1)
            bxM    = np.repeat(bx, len(tv), axis=1)
            kappaM = np.repeat(kappa.reshape(1, -1), len(xv), axis=0)
            logmuxt_grp = axM + bxM * kappaM
            logmuxt_grp = np.repeat(logmuxt_grp[:, :, np.newaxis], nb_regions, axis=2)
            #log-likelihood
            exp_logmuxt = np.exp(logmuxt_grp)
            logDxtgFact = gammaln(Dxtg + 1)
            lnL = np.sum(Dxtg * logmuxt_grp - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)
            #degrees of freedom and number of records
            nb_obs = Dxtg.size
            # Discrete parameters, no penalty.
            # Constraints: Σ_x β_x = 1 → -1 ; mean(κ_t) = 0 → -1
            dofs   = len(ax) + len(bx) + len(kappa) - 1 - 1
            AIC    = 2 * dofs - 2 * lnL
            BIC    = dofs * np.log(nb_obs) - 2 * lnL
            #dataframe with goodness-of-fit statistics
            Fit_stat = [[nb_obs, 'NA', 'NA', dofs, np.round(lnL, 2), np.round(AIC, 2), np.round(BIC, 2)]]
            #print the dataframe
            Fit_stat         = pd.DataFrame(Fit_stat)
            Fit_stat.columns = ["N", "m", "degree", "dofs", "lnL", "AIC", "BIC"]
            #return ax, bx, kappa and statistics
            return ax, bx, kappa, Fit_stat

        # ---------------------------------------------------------------------
        # MAIN FITTING METHOD
        # ---------------------------------------------------------------------
        def fit(self, ax, bx, bx_gr, kappa, kappa_gr, Extg, Dxtg, Muxtg, xv, tv):
            """
            Fit the classic Li-Lee model by two-step gradient descent.

            **Step 1** — fits a standard Lee-Carter model on the data
            using :meth:`_lc_fit` to obtain the common components
            :math:`(\\alpha_x, \\beta_x, \\kappa_t)`.

            **Step 2** — fixes the common component and fits the regional
            residual terms :math:`(\\beta_{x,g}, \\kappa_{g,t})` by gradient
            ascent on the Poisson log-likelihood of:

            .. math::

                \\ln \\mu_{x,t,g} = \\alpha_x + \\beta_x \\kappa_t
                                   + \\beta_{x,g} \\kappa_{g,t}

            At each iteration:

            - :math:`\\beta_{x,g}` — gradient step followed by normalization
              :math:`\\sum_x \\beta_{x,g} = 1 \\; \\forall g`, with a P-spline
              roughness penalty of strength ``h``.
            - :math:`\\kappa_{g,t}` — gradient step (no centering constraint).

            :param ax: Initial age baseline :math:`\\alpha_x`, shape ``(nb_ages, 1)``.
            :type ax: numpy.ndarray
            :param bx: Initial common sensitivity :math:`\\beta_x`, shape ``(nb_ages, 1)``.
            :type bx: numpy.ndarray
            :param bx_gr: Initial regional sensitivity :math:`\\beta_{x,g}`,
                shape ``(nb_ages, nb_regions)``.
            :type bx_gr: numpy.ndarray
            :param kappa: Initial common time factor :math:`\\kappa_t`,
                shape ``(nb_years,)``.
            :type kappa: numpy.ndarray
            :param kappa_gr: Initial regional time factors :math:`\\kappa_{g,t}`,
                shape ``(nb_years, nb_regions)``.
            :type kappa_gr: numpy.ndarray
            :param Extg: Exposures, shape ``(nb_ages, nb_years, nb_regions)``.
            :type Extg: numpy.ndarray
            :param Dxtg: Death counts, same shape as ``Extg``.
            :type Dxtg: numpy.ndarray
            :param Muxtg: Observed mortality rates, same shape as ``Extg``.
            :type Muxtg: numpy.ndarray
            :param xv: Age vector, shape ``(nb_ages,)``.
            :type xv: numpy.ndarray
            :param tv: Year vector, shape ``(nb_years,)``.
            :type tv: numpy.ndarray

            :returns: Dictionary with keys ``"parameters"``, ``"fitted_values"``,
                ``"fit_statistics"``.
            :rtype: dict
            """
            #difference matrix of order z
            Kz    = LiLee.difference_matrix(len(ax), self.z)
            KTK   = Kz.T @ Kz
            IdKTK = np.diag(KTK)

            # ax and bx are computed with the Poisson LC
            ax, bx, kappa, _ = self._lc_fit(ax, bx, kappa, Extg, Dxtg, xv, tv, self.nb_iter)

            #gradient descent step size
            eta = 0.80

            for it in range(self.nb_iter):
                for ct_opt in np.arange(0, 2):

                    axM       = np.repeat(ax, len(tv), axis=1)
                    bxM       = np.repeat(bx, len(tv), axis=1)
                    bx_grM    = np.expand_dims(bx_gr, axis=1)
                    bx_grM    = np.repeat(bx_grM, len(tv), axis=1)
                    kappaM    = np.repeat(kappa.reshape(1, -1), len(xv), axis=0)
                    kappa_grM = np.expand_dims(kappa_gr, axis=0)
                    kappa_grM = np.repeat(kappa_grM, len(xv), axis=0)

                    logmuxt_baseline = axM + bxM * kappaM
                    nb_regions = Extg.shape[2]
                    logmuxt_gr = np.zeros((len(xv), len(tv), nb_regions))

                    #computation of log(mu(x,t,g))
                    for ct in range(nb_regions):
                        logmuxt_gr[:, :, ct] = (logmuxt_baseline + bx_grM[:, :, ct] * kappa_grM[:, :, ct])

                    #baseline for update
                    # r_{x,t,g} = D_{x,t}^{(g)} - E_{x,t}^{(g)} * exp(ln mu_{x,t,g})
                    dlnL_baseline = (Dxtg - Extg * np.exp(logmuxt_gr))

                    if (ct_opt == 0):
                        #--------------- bx gr -----------------
                        # Penalized Newton-Raphson update for β_{x,g}:
                        #
                        #   ∂ℓ/∂β_{x,g}  = Σ_t r_{x,t,g} · κ_{g,t}  - 2h (K^T K β_g)_x
                        #   ∂²ℓ/∂β²_{x,g} = -Σ_t E_{x,t,g} μ_{x,t,g} κ_{g,t}²  + 2h (diag K^T K)_x
                        #
                        #   β_{x,g} ← β_{x,g} + η · (∂ℓ/∂β) / |∂²ℓ/∂β²|
                        bx_gr_new = np.zeros_like(bx_gr)

                        dlnL_dpar = ((np.sum(dlnL_baseline * kappa_grM, axis=1) - 2 * self.h * (KTK @ bx_gr)) /
                                     (np.sum(Extg * np.exp(logmuxt_gr) * kappa_grM**2, axis=1)
                                      - 2 * self.h * np.repeat(IdKTK.reshape(-1, 1), nb_regions, axis=1)))

                        bx_gr_new = bx_gr + eta * dlnL_dpar

                        #----scaling----
                        scal_bx   = np.sum(bx_gr_new, axis=0).reshape((1, -1))
                        scal_bx   = np.repeat(scal_bx, len(xv), axis=0)
                        bx_gr_new = bx_gr_new / scal_bx
                        scal_kap  = scal_bx[0, :].reshape((1, -1))
                        scal_kap  = np.repeat(scal_kap, len(tv), axis=0)

                        if self.verbose:
                            plt.plot(bx_gr_new, label='bx gdp new')
                            plt.title("bx_gr")
                            plt.show()

                        bx_gr = bx_gr_new

                    if (ct_opt == 1):
                        #--------------- kappa gr -----------------
                        # Newton-Raphson update for κ_{g,t}:
                        #
                        #   ∂ℓ/∂κ_{g,t}  = Σ_x r_{x,t,g} · β_{x,g}
                        #   ∂²ℓ/∂κ²_{g,t} = -Σ_x E_{x,t,g} μ_{x,t,g} β_{x,g}²
                        #
                        #   κ_{g,t} ← κ_{g,t} + η · (∂ℓ/∂κ) / |∂²ℓ/∂κ²|
                        kappa_gr_new = np.zeros_like(kappa_gr)

                        dlnL_dpar = (np.sum(dlnL_baseline * bx_grM, axis=0) /
                                     (np.sum(Extg * np.exp(logmuxt_gr) * bx_grM**2, axis=0)))

                        kappa_gr_new = kappa_gr + eta * dlnL_dpar

                        if self.verbose:
                            plt.plot(kappa_gr_new, label='kappa new')
                            plt.title("kappa_gr")
                            plt.show()

                        kappa_gr = kappa_gr_new.copy()

            #end loop

            # Final log(mu) and mu
            axM       = np.repeat(ax, len(tv), axis=1)
            bxM       = np.repeat(bx, len(tv), axis=1)
            bx_grM    = np.expand_dims(bx_gr, axis=1)
            bx_grM    = np.repeat(bx_grM, len(tv), axis=1)
            kappaM    = np.repeat(kappa.reshape(1, -1), len(xv), axis=0)
            kappa_grM = np.expand_dims(kappa_gr, axis=0)
            kappa_grM = np.repeat(kappa_grM, len(xv), axis=0)

            nb_regions  = Extg.shape[2]
            logmu_final = np.zeros((len(xv), len(tv), nb_regions))

            for ct in range(nb_regions):
                logmu_final[:, :, ct] = (
                    axM + bxM * kappaM +
                    bx_grM[:, :, ct] * kappa_grM[:, :, ct]
                )

            mu_final = np.exp(logmu_final)

            # Log-likelihood
            exp_logmuxt = mu_final
            logDxtgFact = gammaln(Dxtg + 1)

            lnL = np.sum(Dxtg * logmu_final
                         - Extg * exp_logmuxt
                         + Dxtg * np.log(Extg)
                         - logDxtgFact)

            nb_obs = Dxtg.size
            # Discrete parameters, no penalty.
            # Constraints: Σ_x β_x = 1 → -1 ; mean(κ_t) = 0 → -1 ; Σ_x β_{x,g} = 1 ∀g → -G
            dofs   = (len(ax) + len(bx) + np.size(bx_gr) + np.size(kappa_gr) + np.size(kappa)
                      - 1 - 1 - nb_regions)
            AIC    = 2 * dofs - 2 * lnL
            BIC    = dofs * np.log(nb_obs) - 2 * lnL

            Fit_stat = [[nb_obs, 'NA', 'NA', dofs, np.round(lnL, 2), np.round(AIC, 2), np.round(BIC, 2)]]
            Fit_stat = pd.DataFrame(Fit_stat)
            Fit_stat.columns = ["N", "m", "degree", "dofs", "lnL", "AIC", "BIC"]

            return {
                "parameters": {
                    "ax":       ax,
                    "bx":       bx,
                    "bx_gr":    bx_gr,
                    "kappa":    kappa,
                    "kappa_gr": kappa_gr,
                },
                "fitted_values": {
                    "log_mu": logmu_final,
                    "mu":     mu_final,
                },
                "fit_statistics": Fit_stat,
            }

