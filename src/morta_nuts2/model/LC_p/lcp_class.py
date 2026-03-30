# -*- coding: utf-8 -*-
"""
Lee-Carter — Object-oriented architecture
===============================================================

This module implements the Lee-Carter mortality model and its parametric
extension using B-splines and P-splines.

The core model decomposes the log force of mortality as:

.. math::

    \\ln \\mu_{x,t} = \\alpha_x + \\beta_x \\, \\kappa_t

where :math:`\\alpha_x` is the age-specific baseline, :math:`\\beta_x` the
age sensitivity pattern, and :math:`\\kappa_t` the period index.

Class hierarchy
---------------
::

    LeeCarter                                  ← general base class
    ├── LeeCarter.Parametric                   ← parametric Lee-Carter (B-splines + P-splines)
    │   └── LeeCarter.Parametric.National      ← β_x shared across all regions
    └── LeeCarter.Classic                      ← classic Lee-Carter (gradient descent)

Usage
-----
::

    # National parametric model
    model = LeeCarter.Parametric.National(n_knots=10, lam=0.1)
    results = model.fit(ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv)

    # Classic model
    model = LeeCarter.Classic(nb_iter=500)
    results = model.fit(ax, bx, kappa, Extg, Dxtg, xv, tv)

Dependencies
------------
- ``scipy.interpolate.BSpline``         — B-spline construction
- ``scipy.interpolate.make_lsq_spline`` — least-squares spline fit
- ``scipy.special.gammaln``             — log-factorial for Poisson likelihood
- ``scipy.linalg``                      — linear algebra utilities
- ``numpy``                             — matrix computation
"""
import sys, os
sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.special import gammaln
from scipy.linalg import lstsq

from morta_nuts2.model.Bsplines.Bsplines import make_bspline_basis, eval_bspline_from_coef


# =============================================================================
# BASE CLASS — LeeCarter
# Contains :
#   - shared static methods (penalty matrix, log-likelihood, fit stats...)
#   - nested sub-classes : Parametric and Classic
# =============================================================================

class LeeCarter:
    """
    General Lee-Carter base class.

    The Lee-Carter model expresses log-mortality as:

    .. math::

        \\ln \\mu_{x,t} = \\alpha_x + \\beta_x \\, \\kappa_t + \\varepsilon_{x,t}

    where :math:`\\alpha_x` is the age baseline, :math:`\\beta_x` the age
    sensitivity, :math:`\\kappa_t` the period index, and
    :math:`\\varepsilon_{x,t}` a zero-mean error term.

    This class groups together:

    - Shared mathematical tools (static methods): penalty matrix,
      Poisson log-likelihood, Newton-Raphson updates, fit statistics.
    - Two main model families as nested sub-classes:

      - :class:`LeeCarter.Parametric` — B-splines + P-splines smoothing.
      - :class:`LeeCarter.Classic`    — non-parametric gradient descent.

    Example
    -------
    >>> model = LeeCarter.Parametric.National(n_knots=10, lam=0.1, verbose=True)
    >>> results = model.fit(ax0, bx0, kappa0, Extg, Dxtg, xv, tv)
    """

    # =========================================================================
    # SHARED STATIC METHODS
    # =========================================================================

    # -------------------------------------------------------------------------
    # 2. P-SPLINES PENALTY MATRIX — numpy
    # -------------------------------------------------------------------------
    @staticmethod
    def make_penalty_matrix(n_basis, diff_order=2):
        """
        Build the **P-spline penalty matrix** :math:`D^T D`.

        The penalty matrix is constructed from a finite difference operator
        applied to the spline coefficients. It is used to enforce smoothness
        in the spline representation.

        The penalization term is:

        .. math::

            P(c) = \\lambda \\, c^T (D^T D) c

        where:

        - :math:`c` are the spline coefficients
        - :math:`D` is the finite difference matrix
        - :math:`\\lambda` is the smoothing parameter

        The matrix :math:`D` approximates discrete derivatives of the spline
        coefficients and is constructed using ``numpy.diff``.

        Parameters
        ----------
        n_basis : int
            Number of spline basis functions.

        diff_order : int
            Order of the finite difference operator.

            - ``1`` → first-order differences (penalizes slope changes)
            - ``2`` → second-order differences (penalizes curvature)

        Returns
        -------
        DtD : numpy.ndarray of shape (n_basis, n_basis)
            Quadratic penalty matrix used in the penalized likelihood.

        diag_DtD : numpy.ndarray of shape (n_basis,)
            Diagonal of :math:`D^T D`, often used for Hessian approximation
            or diagonal preconditioning.

        Notes
        -----
        This matrix is used in the penalized likelihood:

        .. math::

            \\ell_p(\\theta) = \\ell(\\theta) - \\lambda c^T D^T D c

        which controls the smoothness of the spline coefficients.
        """
        # Difference matrix of order diff_order
        D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
        DtD = D.T @ D
        return DtD, np.diag(DtD)

    # -------------------------------------------------------------------------
    # 4. POISSON LOG-LIKELIHOOD — scipy.special.gammaln
    # -------------------------------------------------------------------------
    @staticmethod
    def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
        """
        Compute the Poisson log-likelihood for the Lee-Carter model.

        Under a Poisson assumption, the number of deaths
        :math:`D_{x,t}^{(g)} \\sim \\text{Poisson}(E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)})`,
        and the log-likelihood is:

        .. math::

            \\ell(\\theta) = \\sum_{x,t,g}
            \\Bigl[
                D_{x,t}^{(g)} \\ln \\mu_{x,t}^{(g)}
                - E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)}
                + D_{x,t}^{(g)} \\ln E_{x,t}^{(g)}
                - \\ln D_{x,t}^{(g)}!
            \\Bigr]

        Parameters
        ----------
        Dxtg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Observed death counts.
        Extg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Observed exposures (person-years).
        logmu : ndarray, broadcastable to ``(nb_ages, nb_years, nb_regions)``
            Log force of mortality :math:`\\ln \\mu_{x,t}^{(g)}`.
        logDxtgFact : ndarray
            Pre-computed :math:`\\ln(D_{x,t}^{(g)}!)` via ``scipy.special.gammaln``.

        Returns
        -------
        lnL : float
            Total Poisson log-likelihood.
        exp_logmu : ndarray
            :math:`\\mu_{x,t}^{(g)} = \\exp(\\ln \\mu_{x,t}^{(g)})`.
        weighted_exp : ndarray
            :math:`E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)}` — expected deaths.
        residual : ndarray
            :math:`D_{x,t}^{(g)} - E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)}` — Pearson residuals.
        """
        exp_logmu    = np.exp(logmu)
        weighted_exp = Extg * exp_logmu
        residual     = Dxtg - weighted_exp
        lnL = float(np.sum(
            Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
        ))
        return lnL, exp_logmu, weighted_exp, residual

    # -------------------------------------------------------------------------
    # 5. NR UPDATE — ax_coef (shared by both parametric variants)
    # -------------------------------------------------------------------------
    @staticmethod
    def update_ax_coef(ax_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
        """
        Newton-Raphson update of the B-spline coefficients of :math:`\\alpha_x`.

        For each basis coefficient :math:`c_j^{\\alpha}`, the penalized
        score and Fisher information are:

        .. math::

            \\text{num}_j = \\sum_{x,t,g} r_{x,t}^{(g)} \\, B_j(x)
                           - 2\\lambda \\, (D^T D \\, c^\\alpha)_j

        .. math::

            \\text{den}_j = \\sum_{x,t,g} \\hat{\\mu}_{x,t}^{(g)} E_{x,t}^{(g)} \\, B_j(x)^2
                           + 2\\lambda \\, (D^T D)_{jj}

        The update step is:

        .. math::

            c_j^{\\alpha} \\leftarrow c_j^{\\alpha}
                + \\eta \\, \\frac{\\text{num}_j}{\\text{den}_j}

        where :math:`r_{x,t}^{(g)} = D_{x,t}^{(g)} - E_{x,t}^{(g)} \\mu_{x,t}^{(g)}`
        is the residual and :math:`\\eta` is the learning rate.

        Parameters
        ----------
        ax_coef : ndarray, shape ``(n_basis,)``
            Current B-spline coefficients of :math:`\\alpha_x`.
        B : ndarray, shape ``(nb_ages, n_basis)``
            B-spline basis matrix evaluated at observed ages.
        residual : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Current Pearson residuals :math:`D - E\\mu`.
        weighted_exp : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Expected deaths :math:`E \\mu`.
        eta : float
            Learning rate.
        lam : float
            P-spline penalty strength :math:`\\lambda`.
        DtD : ndarray, shape ``(n_basis, n_basis)``
            Penalty matrix :math:`D^T D`.
        diag_DtD : ndarray, shape ``(n_basis,)``
            Diagonal of :math:`D^T D`.

        Returns
        -------
        ndarray, shape ``(n_basis,)``
            Updated B-spline coefficients of :math:`\\alpha_x`.
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

    # -------------------------------------------------------------------------
    # 5. NR UPDATE — kappa (shared by both parametric variants)
    # -------------------------------------------------------------------------
    @staticmethod
    def update_kappa(kappa, bx_reg, residual, weighted_exp, eta):
        """
        Newton-Raphson update of the period index :math:`\\kappa_t`.

        For each time point :math:`t`, the score and Fisher information are:

        .. math::

            \\text{num}_t = \\sum_{x,g} r_{x,t}^{(g)} \\, \\beta_x^{(g)}

        .. math::

            \\text{den}_t = \\sum_{x,g} E_{x,t}^{(g)} \\, \\mu_{x,t}^{(g)}
                           \\, (\\beta_x^{(g)})^2

        The update is:

        .. math::

            \\kappa_t \\leftarrow \\kappa_t
                + \\eta \\, \\frac{\\text{num}_t}{\\text{den}_t}

        Parameters
        ----------
        kappa : ndarray, shape ``(nb_years,)``
            Current period index values.
        bx_reg : ndarray, shape ``(nb_ages,)`` or ``(nb_ages, nb_regions)``
            Age-sensitivity pattern :math:`\\beta_x` (or :math:`\\beta_x^{(g)}`).
        residual : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Current Pearson residuals.
        weighted_exp : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Expected deaths :math:`E \\mu`.
        eta : float
            Learning rate.

        Returns
        -------
        ndarray, shape ``(nb_years,)``
            Updated period index :math:`\\kappa_t`.
        """
        bx3d  = bx_reg[:, None, :]
        num_k = np.sum(residual    * bx3d, axis=(0, 2))
        den_k = np.sum(weighted_exp * bx3d**2, axis=(0, 2))
        kappa_new       = kappa.copy()
        mask            = den_k != 0
        kappa_new[mask] += eta * num_k[mask] / den_k[mask]
        return kappa_new

    # -------------------------------------------------------------------------
    # 7. FIT STATISTICS — numpy
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, n_basis, nb_years, nb_regions,
                          lam=0.0, DtD=None, B=None, weighted_exp=None):
        """
        Compute Poisson goodness-of-fit statistics: deviance, AIC, BIC.

        **Poisson deviance**

        .. math::

            \\mathcal{D} = 2 \\bigl( \\ell_{\\text{sat}} - \\ell(\\hat{\\theta}) \\bigr)

        where the saturated log-likelihood is:

        .. math::

            \\ell_{\\text{sat}} = \\sum_{x,t,g}
            \\left[ D_{x,t}^{(g)} \\ln\\!
                \\frac{D_{x,t}^{(g)}}{E_{x,t}^{(g)}} - D_{x,t}^{(g)}
            \\right]

        **Effective degrees of freedom (P-spline case)**

        When :math:`\\lambda > 0`, the nominal parameter count is replaced by
        the effective degrees of freedom (edf), computed as the trace of the
        hat matrix:

        .. math::

            \\text{edf} = \\operatorname{tr}\\!\\bigl(
                B \\,(B^T W B + 2\\lambda D^T D)^{-1} B^T W
            \\bigr)

        where :math:`W = \\operatorname{diag}\\!\\left(
        \\sum_{t,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} \\right)`.

        The total model degrees of freedom are:

        .. math::

            \\text{dofs} = \\text{edf}_{\\alpha_x}
                          + \\text{edf}_{\\beta_x}
                          + T_{\\kappa}

        where :math:`T_{\\kappa} = \\text{nb\\_years}` (unpenalized).

        **Information criteria**

        .. math::

            \\text{AIC} = 2 \\cdot \\text{dofs} - 2\\,\\ell(\\hat{\\theta})

        .. math::

            \\text{BIC} = \\text{dofs} \\cdot \\ln(N) - 2\\,\\ell(\\hat{\\theta})

        Parameters
        ----------
        Dxtg : ndarray
            Observed death counts.
        Extg : ndarray
            Observed exposures.
        logmu : ndarray
            Fitted log force of mortality.
        logDxtgFact : ndarray
            Pre-computed :math:`\\ln(D!)`.
        n_basis : int
            Number of B-spline basis functions (used when ``lam=0``).
        nb_years : int
            Number of observation years (number of :math:`\\kappa_t` parameters).
        nb_regions : int
            Number of regions.
        lam : float, optional
            P-spline penalty :math:`\\lambda` (default ``0.0``).
        DtD : ndarray or None, optional
            Penalty matrix :math:`D^T D` (required when ``lam > 0``).
        B : ndarray or None, optional
            B-spline basis matrix (required when ``lam > 0``).
        weighted_exp : ndarray or None, optional
            Expected deaths :math:`E \\mu` (required when ``lam > 0``).

        Returns
        -------
        pandas.DataFrame
            One-row DataFrame with columns
            ``["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]``.
        """
        exp_logmu = np.exp(logmu)
        lnL = float(np.sum(
            Dxtg * logmu - Extg * exp_logmu + Dxtg * np.log(Extg) - logDxtgFact
        ))

        # Poisson deviance
        safe_Dxtg = np.where(Dxtg > 0, Dxtg, 1.0)
        lnL_sat   = float(np.sum(np.where(
            Dxtg > 0,
            Dxtg * np.log(safe_Dxtg / np.maximum(Extg, 1e-12)) - Dxtg,
            0.0
        )))
        deviance = 2.0 * (lnL_sat - lnL)

        nb_obs = int(Dxtg.size)

        # ------------------------------------------------------------------
        # Effective degrees of freedom (edf) when penalization is active
        # ------------------------------------------------------------------
        if lam > 0 and B is not None and DtD is not None and weighted_exp is not None:
            # Weight vector: sum weighted_exp over years and regions → (nb_ages,)
            W = np.sum(weighted_exp, axis=(1, 2)) if weighted_exp.ndim == 3 \
                else np.sum(weighted_exp, axis=1)

            BtWB = B.T @ (W[:, None] * B)                    # (n_basis, n_basis)
            A    = BtWB + 2.0 * lam * DtD                    # penalized info matrix
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            # trace(H) = trace(B A^{-1} B^T W) = sum_i [B A^{-1} B^T]_ii * W_i
            BA_inv  = B @ A_inv                               # (nb_ages, n_basis)
            H_diag  = np.einsum('ij,ij->i', BA_inv, W[:, None] * B)
            edf     = float(np.sum(H_diag))

            # α_x and β_x share the same basis B and penalty → edf counted twice
            edf_ax   = edf
            edf_bx   = edf
            dofs_mod = edf_ax + edf_bx + nb_years
        else:
            # lam = 0 : no penalization → nominal dofs
            dofs_mod = 2.0 * n_basis + nb_years

        AIC = 2.0 * dofs_mod - 2.0 * lnL
        BIC = dofs_mod * np.log(nb_obs) - 2.0 * lnL

        return pd.DataFrame(
            [[nb_obs, n_basis, round(dofs_mod, 2),
              round(lnL, 2), round(deviance, 2), round(AIC, 2), round(BIC, 2)]],
            columns=["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]
        )

    # -------------------------------------------------------------------------
    # UTILITY — build input matrices from a long-format DataFrame
    # -------------------------------------------------------------------------
    @staticmethod
    def build_input_from_dataframe(df):
        """
        Convert a long-format DataFrame into 3-D arrays compatible with ``.fit()``.

        The input DataFrame must contain one row per (region, year, age)
        combination.  The function pivots it into 3-D NumPy arrays of shape
        ``(nb_ages, nb_years, nb_regions)`` using vectorised index mapping
        (no triple loop).

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format table with columns:
            ``['region', 'year', 'age', 'deaths', 'exposure', 'mortality_rate']``.

        Returns
        -------
        Muxtg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Observed mortality rates :math:`\\mu_{x,t}^{(g)}`.
        Dxtg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Observed death counts :math:`D_{x,t}^{(g)}` (clipped to 0).
        Extg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
            Observed exposures :math:`E_{x,t}^{(g)}` (clipped to ``1e-12``).
        xv : ndarray
            Sorted age vector.
        tv : ndarray
            Sorted year vector.
        regions : ndarray
            Sorted region labels.
        """
        # Sort for safety
        df = df.sort_values(["age", "year", "region"]).copy()

        xv      = np.sort(df["age"].unique())
        tv      = np.sort(df["year"].unique())
        regions = np.sort(df["region"].unique())

        nb_ages    = len(xv)
        nb_years   = len(tv)
        nb_regions = len(regions)

        # Index mapping
        age_idx  = {a: i for i, a in enumerate(xv)}
        year_idx = {y: i for i, y in enumerate(tv)}
        reg_idx  = {r: i for i, r in enumerate(regions)}

        # Allocation
        Dxtg  = np.zeros((nb_ages, nb_years, nb_regions))
        Extg  = np.zeros_like(Dxtg)
        Muxtg = np.zeros_like(Dxtg)

        # Vectorization without triple loop
        Dxtg[ df.age.map(age_idx), df.year.map(year_idx), df.region.map(reg_idx)] = df.deaths.values
        Extg[ df.age.map(age_idx), df.year.map(year_idx), df.region.map(reg_idx)] = df.exposure.values
        Muxtg[df.age.map(age_idx), df.year.map(year_idx), df.region.map(reg_idx)] = df.mortality_rate.values

        # Numerical safety
        Extg  = np.maximum(Extg, 1e-12)
        Dxtg  = np.maximum(Dxtg, 0.0)

        return Muxtg, Dxtg, Extg, xv, tv, regions


    # =========================================================================
    # SUB-CLASS : Parametric
    # Parametric Lee-Carter with B-splines + P-splines (national variant only)
    # NOTE : the multi-region variant (Lee & Li) has been moved to llp_class.py
    # =========================================================================

    class Parametric:
        """
        Parametric Lee-Carter model using B-splines and P-splines — national variant.

        Age effects :math:`\\alpha_x` and :math:`\\beta_x` are represented as
        linear combinations of B-spline basis functions:

        .. math::

            \\alpha_x = \\sum_j c_j^{\\alpha} \\, B_j(x), \\qquad
            \\beta_x  = \\sum_j c_j^{\\beta}  \\, B_j(x)

        so that the log force of mortality is:

        .. math::

            \\ln \\mu_{x,t,g} = \\alpha_x + \\beta_x \\, \\kappa_t

        where :math:`\\beta_x` is **common to all regions** (no :math:`g`
        dimension).

        Smoothness is enforced by a P-spline penalty on the spline
        coefficients:

        .. math::

            \\ell_p(\\theta) = \\ell(\\theta)
                - \\lambda \\bigl( c^{\\alpha \\top} D^T D \\, c^{\\alpha}
                                 + c^{\\beta  \\top} D^T D \\, c^{\\beta} \\bigr)

        Notes
        -----
        The multi-region variant (:math:`\\beta_{x,g}` per region),
        corresponding to the Lee–Li model, has been moved to
        ``llp_class.py → LeeAndLi``.
        """

        # ---------------------------------------------------------------------
        # NATIONAL VARIANT
        # ---------------------------------------------------------------------
        class National:
            """
            Parametric Lee-Carter — national variant (common :math:`\\beta_x`).

            Fits the model:

            .. math::

                \\ln \\mu_{x,t,g} = \\alpha_x + \\beta_x \\, \\kappa_t

            where :math:`\\alpha_x` and :math:`\\beta_x` are B-spline curves
            (shared across all regions), and :math:`\\kappa_t` is the period
            index estimated for each year.

            Parameters
            ----------
            degree : int, optional
                B-spline polynomial degree (default ``2``).
            n_knots : int, optional
                Number of internal knots (default ``10``).
            xmin : float or None, optional
                Lower age bound for the spline domain (inferred from data if ``None``).
            xmax : float or None, optional
                Upper age bound for the spline domain (inferred from data if ``None``).
            lam : float, optional
                P-spline penalty strength :math:`\\lambda` (default ``0.0``).
            diff_order : int, optional
                Order of the finite difference penalty operator (default ``2``).
            nb_iter : int, optional
                Maximum number of Newton-Raphson iterations (default ``800``).
            eta0 : float, optional
                Initial learning rate (default ``0.2``).
            tol : float, optional
                Convergence tolerance on :math:`\\Delta \\ell` (default ``1e-3``).
            verbose : bool, optional
                If ``True``, print iteration logs (default ``False``).
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
            # COMPUTE LOGMU — national version (common β_x)
            # -----------------------------------------------------------------
            @staticmethod
            def compute_logmu_national(ax_coef, bx_coef, kappa, xv, B):
                """
                Reconstruct log-mortality from B-spline coefficients (national variant).

                Evaluates the B-spline curves and applies the Lee-Carter identity:

                .. math::

                    \\alpha_x = B(x)\\, c^{\\alpha}, \\qquad
                    \\beta_x  = B(x)\\, c^{\\beta}

                .. math::

                    \\ln \\mu_{x,t,g} = \\alpha_x + \\beta_x \\, \\kappa_t

                Since :math:`\\beta_x` is common to all regions, the output is
                broadcast to shape ``(nb_ages, nb_years, 1)`` for automatic
                broadcasting over :math:`g`.

                Parameters
                ----------
                ax_coef : ndarray, shape ``(n_basis,)``
                    B-spline coefficients of :math:`\\alpha_x`.
                bx_coef : ndarray, shape ``(n_basis,)``
                    B-spline coefficients of :math:`\\beta_x` (no region dimension).
                kappa : ndarray, shape ``(nb_years,)``
                    Period index :math:`\\kappa_t`.
                xv : ndarray, shape ``(nb_ages,)``
                    Observed age vector.
                B : ndarray, shape ``(nb_ages, n_basis)``
                    B-spline basis matrix.

                Returns
                -------
                logmu : ndarray, shape ``(nb_ages, nb_years, 1)``
                    Log force of mortality, broadcast-ready over regions.
                ax : ndarray, shape ``(nb_ages,)``
                    Evaluated age baseline :math:`\\alpha_x`.
                bx : ndarray, shape ``(nb_ages,)``
                    Evaluated age sensitivity :math:`\\beta_x`.
                """
                ax = B @ ax_coef   # (nb_ages,)
                bx = B @ bx_coef   # (nb_ages,)

                # broadcast: no region dimension → identical for all g
                logmu = ax[:, None] + bx[:, None] * kappa[None, :]   # (nb_ages, nb_years)
                logmu = logmu[:, :, None]  # (nb_ages, nb_years, 1) → broadcast over nb_regions

                return logmu, ax, bx

            # -----------------------------------------------------------------
            # UPDATE BX — national version
            # -----------------------------------------------------------------
            @staticmethod
            def update_bx_coef_national(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """
                Newton-Raphson update of the B-spline coefficients of :math:`\\beta_x` (national).

                Since :math:`\\beta_x` is common to all regions, the gradient
                and Fisher information are summed over regions :math:`g`.

                For each basis coefficient :math:`c_j^{\\beta}`:

                .. math::

                    \\text{num}_j = \\sum_{x,t,g}
                        r_{x,t}^{(g)} \\, B_j(x) \\, \\kappa_t
                        - 2\\lambda \\,(D^T D\\, c^{\\beta})_j

                .. math::

                    \\text{den}_j = \\sum_{x,t,g}
                        E_{x,t}^{(g)} \\mu_{x,t}^{(g)}
                        \\bigl(B_j(x)\\,\\kappa_t\\bigr)^2
                        + 2\\lambda \\,(D^T D)_{jj}

                .. math::

                    c_j^{\\beta} \\leftarrow c_j^{\\beta}
                        + \\eta \\, \\frac{\\text{num}_j}{\\text{den}_j}

                Parameters
                ----------
                bx_coef : ndarray, shape ``(n_basis,)``
                    Current B-spline coefficients of :math:`\\beta_x`.
                B : ndarray, shape ``(nb_ages, n_basis)``
                    B-spline basis matrix.
                kappa : ndarray, shape ``(nb_years,)``
                    Current period index :math:`\\kappa_t`.
                residual : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                    Current Pearson residuals.
                weighted_exp : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                    Expected deaths :math:`E \\mu`.
                eta : float
                    Learning rate.
                lam : float
                    P-spline penalty :math:`\\lambda`.
                DtD : ndarray, shape ``(n_basis, n_basis)``
                    Penalty matrix.
                diag_DtD : ndarray, shape ``(n_basis,)``
                    Diagonal of the penalty matrix.

                Returns
                -------
                ndarray, shape ``(n_basis,)``
                    Updated B-spline coefficients of :math:`\\beta_x`.
                """
                n_basis     = len(bx_coef)
                nb_ages     = B.shape[0]
                bx_coef_new = bx_coef.copy()

                pen_grad = 2.0 * lam * (DtD @ bx_coef) if lam > 0 else np.zeros(n_basis)

                # κ_t repeated over age axis: (nb_ages, nb_years)
                kappaM = np.broadcast_to(kappa[None, :], (nb_ages, len(kappa)))

                for j in range(n_basis):
                    # B[:, j] * κ_t  →  (nb_ages, nb_years)
                    BjKappa = B[:, j][:, None] * kappaM          # (nb_ages, nb_years)
                    BjK3d   = BjKappa[:, :, None]                # (nb_ages, nb_years, 1) → broadcast over g

                    # Sum over (ages, years, regions)
                    num = float(np.sum(residual    * BjK3d)) - pen_grad[j]
                    den = float(np.sum(weighted_exp * BjK3d**2))
                    if lam > 0:
                        den += 2.0 * lam * diag_DtD[j]
                    if den != 0:
                        bx_coef_new[j] += eta * num / den

                return bx_coef_new

            # -----------------------------------------------------------------
            # RESCALING — national version
            # -----------------------------------------------------------------
            @staticmethod
            def rescale_bx_kappa_national(bx_coef, bx, kappa):
                """
                Normalize the identifiability constraint :math:`\\sum_x \\beta_x = 1`.

                The Lee-Carter model is not identifiable without constraints.
                The standard normalization rescales :math:`\\beta_x` and
                :math:`\\kappa_t` simultaneously so that:

                .. math::

                    \\sum_x \\beta_x = 1

                This is achieved by computing the scale factor
                :math:`s = \\sum_x \\beta_x` and applying:

                .. math::

                    \\beta_x \\leftarrow \\frac{\\beta_x}{s}, \\qquad
                    c^{\\beta} \\leftarrow \\frac{c^{\\beta}}{s}, \\qquad
                    \\kappa_t \\leftarrow s \\, \\kappa_t

                so that the product :math:`\\beta_x \\kappa_t` is preserved.

                Parameters
                ----------
                bx_coef : ndarray, shape ``(n_basis,)``
                    B-spline coefficients of :math:`\\beta_x`.
                bx : ndarray, shape ``(nb_ages,)``
                    Evaluated :math:`\\beta_x` curve.
                kappa : ndarray, shape ``(nb_years,)``
                    Period index :math:`\\kappa_t`.

                Returns
                -------
                bx_coef : ndarray
                    Rescaled B-spline coefficients.
                bx : ndarray
                    Rescaled :math:`\\beta_x`.
                kappa : ndarray
                    Rescaled :math:`\\kappa_t`.
                """
                scal_factor = float(np.sum(bx))
                if scal_factor == 0:
                    return bx_coef, bx, kappa
                return bx_coef / scal_factor, bx / scal_factor, kappa * scal_factor

            # -----------------------------------------------------------------
            # MAIN FITTING METHOD — national version
            # -----------------------------------------------------------------
            def fit(self, ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv):
                """
                Fit the national parametric Lee-Carter model by penalized Newton-Raphson.

                The algorithm maximises the penalized Poisson log-likelihood:

                .. math::

                    \\ell_p(\\theta) = \\ell(\\theta)
                        - \\lambda \\bigl(
                            c^{\\alpha \\top} D^T D \\, c^{\\alpha}
                          + c^{\\beta  \\top} D^T D \\, c^{\\beta}
                          \\bigr)

                At each iteration the three parameter blocks are updated in
                sequence via Newton-Raphson steps:

                1. :math:`c^{\\alpha}` — via :meth:`LeeCarter.update_ax_coef`.
                2. :math:`c^{\\beta}` — via :meth:`update_bx_coef_national`.
                3. :math:`\\kappa_t` — via :meth:`LeeCarter.update_kappa`.

                A rollback mechanism reverts to the previous state and halves
                the learning rate :math:`\\eta` whenever the likelihood
                decreases.  Early stopping is triggered when no improvement
                greater than ``min_delta`` is observed for ``patience``
                consecutive steps.

                After convergence, :math:`\\beta_x` and :math:`\\kappa_t` are
                renormalized so that :math:`\\sum_x \\beta_x = 1`.

                Parameters
                ----------
                ax_coef_init : ndarray, shape ``(n_basis,)``
                    Initial B-spline coefficients for :math:`\\alpha_x`.
                bx_coef_init : ndarray, shape ``(n_basis,)``
                    Initial B-spline coefficients for :math:`\\beta_x`.
                kappa_init : ndarray, shape ``(nb_years,)``
                    Initial period index :math:`\\kappa_t`.
                Extg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                    Observed exposures.
                Dxtg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                    Observed death counts.
                xv : ndarray, shape ``(nb_ages,)``
                    Observed age vector.
                tv : ndarray, shape ``(nb_years,)``
                    Observation year vector.

                Returns
                -------
                dict
                    - ``parameters``: ``ax_coef``, ``bx_coef``, ``kappa``.
                    - ``curves``: ``alpha_x`` shape ``(nb_ages,)``,
                      ``beta_x`` shape ``(nb_ages,)``.
                    - ``fitted_values``: ``log_mu``, ``mu``,
                      both shape ``(nb_ages, nb_years)``.
                    - ``fit_statistics``: pandas DataFrame with
                      ``N``, ``n_basis``, ``dofs``, ``lnL``,
                      ``deviance``, ``AIC``, ``BIC``.
                """
                nb_years   = len(tv)
                nb_regions = Extg.shape[2]

                xmin = self.xmin if self.xmin is not None else float(np.min(xv))
                xmax = self.xmax if self.xmax is not None else float(np.max(xv))

                B, knots, n_basis = make_bspline_basis(xv, self.degree, self.n_knots, xmin, xmax)

                if len(ax_coef_init) != n_basis:
                    raise ValueError(f"ax_coef_init must have {n_basis} elements, got {len(ax_coef_init)}")
                if len(bx_coef_init) != n_basis:                          # ← 1D check
                    raise ValueError(f"bx_coef_init must have {n_basis} elements, got {len(bx_coef_init)}")

                ax_coef = ax_coef_init.copy()
                bx_coef = bx_coef_init.copy()   # (n_basis,)
                kappa   = kappa_init.copy()

                DtD, diag_DtD = LeeCarter.make_penalty_matrix(n_basis, self.diff_order)
                logDxtgFact   = gammaln(Dxtg + 1)

                lnL       = -np.inf
                Delta_lnL = 0.0
                eta       = self.eta0
                it        = -1

                best_lnL  = -np.inf
                patience  = 40
                min_delta = 1e-2
                wait      = 0

                # Snapshots for rollback when a step degrades the likelihood
                ax_coef_prev = ax_coef.copy()
                bx_coef_prev = bx_coef.copy()
                kappa_prev   = kappa.copy()

                while it < self.nb_iter:
                    it += 1

                    # Reconstruction — common β_x, broadcast over nb_regions
                    logmu, ax, bx = self.compute_logmu_national(ax_coef, bx_coef, kappa, xv, B)

                    # Log-likelihood (logmu automatically broadcast over nb_regions)
                    lnL_new, _, weighted_exp, residual = LeeCarter.poisson_lnL(
                        Dxtg, Extg, logmu, logDxtgFact
                    )

                    Delta_lnL = lnL_new - lnL

                    if self.verbose and (it % 10 == 0):
                        print(f"It {it:4d} | lnL = {lnL_new:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

                    # ----------------------------------------------------------
                    # Step degraded the likelihood → rollback + shrink eta
                    # Do NOT update parameters, retry from previous state
                    # ----------------------------------------------------------
                    if Delta_lnL < 0 and it > 0:
                        ax_coef = ax_coef_prev.copy()
                        bx_coef = bx_coef_prev.copy()
                        kappa   = kappa_prev.copy()
                        eta    *= 0.5
                        continue

                    # ----------------------------------------------------------
                    # Convergence: only triggered on a genuine positive step
                    # ----------------------------------------------------------
                    if 0 <= Delta_lnL < self.tol:
                        if self.verbose:
                            print(f"\nConvergence reached (tolerance) at it {it}.")
                        break

                    # Early stopping
                    if lnL_new > best_lnL + min_delta:
                        best_lnL = lnL_new
                        wait     = 0
                    else:
                        wait += 1

                    if wait >= patience:
                        if self.verbose:
                            print("\nEarly stopping: no more significant improvement.")
                        break

                    # Snapshot current state before updating
                    ax_coef_prev = ax_coef.copy()
                    bx_coef_prev = bx_coef.copy()
                    kappa_prev   = kappa.copy()
                    lnL          = lnL_new

                    # Increase eta prudently after a successful step
                    eta = min(eta * 1.05, 2.0)

                    # Updates
                    ax_coef = LeeCarter.update_ax_coef(
                        ax_coef, B, residual, weighted_exp, eta, self.lam, DtD, diag_DtD
                    )
                    bx_coef = self.update_bx_coef_national(        # ← national version
                        bx_coef, B, kappa, residual, weighted_exp, eta, self.lam, DtD, diag_DtD
                    )
                    kappa = LeeCarter.update_kappa(kappa, bx[:, None], residual, weighted_exp, eta)
                    #                                      ↑ (nb_ages, 1) so that update_kappa works

                # Final rescaling
                bx_coef, bx, kappa = self.rescale_bx_kappa_national(bx_coef, bx, kappa)

                logmu_final, ax, bx = self.compute_logmu_national(ax_coef, bx_coef, kappa, xv, B)

                # Recompute final weighted_exp for edf calculation
                _, _, weighted_exp_final, _ = LeeCarter.poisson_lnL(
                    Dxtg, Extg, logmu_final, logDxtgFact
                )

                Fit_stat = LeeCarter.compute_fit_stats(
                    Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions,
                    lam=self.lam, DtD=DtD, B=B, weighted_exp=weighted_exp_final   # ← edf inputs
                )

                if self.verbose:
                    print("\n" + "="*70)
                    print("FINAL STATISTICS")
                    print("="*70)
                    print(Fit_stat.to_string(index=False))

                return {
                    "parameters": {
                        "ax_coef": ax_coef,
                        "bx_coef": bx_coef,        # (n_basis,)
                        "kappa":   kappa
                    },
                    "curves": {
                        "alpha_x": ax,             # (nb_ages,)
                        "beta_x":  bx,             # (nb_ages,)  ← no g dimension
                    },
                    "fitted_values": {
                        "log_mu": logmu_final[:, :, 0],   # (nb_ages, nb_years)
                        "mu":     np.exp(logmu_final[:, :, 0])
                    },
                    "fit_statistics": Fit_stat
                }


    # =========================================================================
    # SUB-CLASS : Classic
    # Classic non-parametric Lee-Carter (gradient descent)
    # =========================================================================

    class Classic:
        """
        Classic (non-parametric) Lee-Carter model fitted by gradient descent.

        The model is:

        .. math::

            \\ln \\mu_{x,t} = \\alpha_x + \\beta_x \\, \\kappa_t

        Parameters :math:`(\\alpha_x, \\beta_x, \\kappa_t)` are estimated by
        maximising the Poisson log-likelihood using coordinate-wise gradient
        ascent.  At each iteration the three blocks are updated in sequence:

        1. :math:`\\alpha_x` — gradient step on the age baseline.
        2. :math:`\\beta_x`  — gradient step followed by normalization
           :math:`\\sum_x \\beta_x = 1`.
        3. :math:`\\kappa_t` — gradient step followed by centering
           :math:`\\sum_t \\kappa_t = 0` (the mean is absorbed into :math:`\\alpha_x`).

        Parameters
        ----------
        nb_iter : int, optional
            Number of gradient descent iterations (default ``500``).
        eta : float, optional
            Learning rate (default ``1``).
        """

        def __init__(self, nb_iter=500, eta=1):
            self.nb_iter = nb_iter
            self.eta     = eta

        def fit(self, ax, bx, kappa, Extg, Dxtg, xv, tv):
            """
            Fit the classic Lee-Carter model by gradient descent.

            For each parameter block, the gradient of the Poisson log-likelihood
            with respect to that block is computed and a Newton-Raphson step is taken.

            **Update for** :math:`\\alpha_x`:

            .. math::

                \\alpha_x \\leftarrow \\alpha_x + \\eta \\,
                \\frac{\\sum_{t,g} r_{x,t}^{(g)}}{\\sum_{t,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)}}

            **Update for** :math:`\\beta_x` (with normalization):

            .. math::

                \\beta_x \\leftarrow \\frac{\\tilde{\\beta}_x}{\\sum_x \\tilde{\\beta}_x},
                \\qquad
                \\kappa_t \\leftarrow \\kappa_t \\cdot \\sum_x \\tilde{\\beta}_x

            where :math:`\\tilde{\\beta}_x = \\beta_x + \\eta \\,
            \\frac{\\sum_{t,g} r_{x,t}^{(g)} \\kappa_t}{\\sum_{t,g} E_{x,t}^{(g)} \\mu_{x,t}^{(g)} \\kappa_t^2}`.

            **Update for** :math:`\\kappa_t` (with centering):

            .. math::

                \\kappa_t \\leftarrow \\tilde{\\kappa}_t - \\bar{\\kappa},
                \\qquad
                \\alpha_x \\leftarrow \\alpha_x + \\bar{\\kappa} \\, \\beta_x

            where :math:`\\bar{\\kappa} = \\frac{1}{T} \\sum_t \\tilde{\\kappa}_t`.

            Parameters
            ----------
            ax : ndarray, shape ``(nb_ages,)``
                Initial age baseline :math:`\\alpha_x`.
            bx : ndarray, shape ``(nb_ages,)``
                Initial age sensitivity :math:`\\beta_x`.
            kappa : ndarray, shape ``(nb_years,)``
                Initial period index :math:`\\kappa_t`.
            Extg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                Observed exposures.
            Dxtg : ndarray, shape ``(nb_ages, nb_years, nb_regions)``
                Observed death counts.
            xv : ndarray
                Observed age vector.
            tv : ndarray
                Observation year vector.

            Returns
            -------
            dict
                - ``parameters``: ``ax_coef``, ``bx_coef``, ``kappa``
                  (2-D broadcast arrays).
                - ``fitted_values``: ``log_mu``, ``mu``.
                - ``fit_statistics``: pandas DataFrame with
                  ``N``, ``m``, ``degree``, ``dofs``, ``lnL``, ``AIC``, ``BIC``.
            """
            for it in range(self.nb_iter):
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
                        ax_new = ax + self.eta * dlnL_dpar.reshape(-1, 1)
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
                        bx_new = bx + self.eta * dlnL_dpar.reshape(-1, 1)
                        #we normalize
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
                        kappa_new = kappa + self.eta * dlnL_dpar
                        #we rescale
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
            lnL         = np.sum(Dxtg * logmuxt_grp - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)
            #dof's and numbers of records
            nb_obs = Dxtg.size
            dofs   = len(ax) + len(bx) + len(kappa)
            AIC    = 2 * dofs - 2 * lnL
            BIC    = dofs * np.log(nb_obs) - 2 * lnL
            #dataframe with statistics of goodness of fit
            Fit_stat = [[nb_obs, 'NA', 'NA', dofs, np.round(lnL, 2), np.round(AIC, 2), np.round(BIC, 2)]]
            #We print the file
            Fit_stat         = pd.DataFrame(Fit_stat)
            Fit_stat.columns = ["N", "m", "degree", "dofs", "lnL", "AIC", "BIC"]

            results = {
                "parameters": {
                    "ax_coef": axM,
                    "bx_coef": bxM,
                    "kappa":   kappaM
                },
                "fitted_values": {
                    "log_mu": axM + bxM * kappaM,
                    "mu":     np.exp(axM + bxM * kappaM)
                },
                "fit_statistics": Fit_stat
            }
            #we return ax, bx, kappa and stats
            return results
        