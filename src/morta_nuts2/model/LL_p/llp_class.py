# -*- coding: utf-8 -*-
"""
LiLee — Object-oriented architecture
===============================================================
Multi-population Li-Lee model:
    ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

Where:
  α_{x,g}   : mortality baseline by age AND region (B-splines)
  β_x       : common sensitivity across all regions (B-splines)
  κ_t       : common time factor
  β_{x,g}   : regional sensitivity (B-splines)
  κ_{g,t}   : regional time factor

Class hierarchy:

    LiLee                              ← general base class
    ├── LiLee.Parametric               ← parametric Li-Lee (B-splines + P-splines)
    └── LiLee.Classic                  ← classic Li-Lee (gradient descent)

USAGE:
    # Parametric model
    model = LiLee.Parametric(degree=3, n_knots=6, lam=0.1)
    results = model.fit(alpha_coef_init, beta_coef_init, beta_g_coef_init,
                        kappa_init, kappa_g_init, Extg, Dxtg, xv, tv)

    # Classic model
    model = LiLee.Classic(nb_iter=500)
    results = model.fit(ax, bx, bx_gr, kappa, kappa_gr, Extg, Dxtg, Muxtg, xv, tv)

USES ONLY scipy/numpy:
  ✓ scipy.interpolate.BSpline
  ✓ scipy.special.gammaln
  ✓ numpy.diff (penalty matrix)

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
          · LiLee.Parametric   (B-splines, P-splines)
          · LiLee.Classic      (non-parametric, gradient descent)

    Quick usage example
    -------------------
    >>> model = LiLee.Parametric(degree=3, n_knots=6, lam=0.1, verbose=True)
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
        Builds the P-splines penalty matrix D^T D.

        Returns
        --------
        DtD      : (n_basis, n_basis)
        diag_DtD : (n_basis,)   diagonal
        """
        D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
        DtD = D.T @ D
        return DtD, np.diag(DtD)

    # -------------------------------------------------------------------------
    # POISSON LOG-LIKELIHOOD — scipy.special.gammaln
    # -------------------------------------------------------------------------
    @staticmethod
    def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
        """Poisson log-likelihood."""
        exp_logmu    = np.exp(logmu)
        weighted_exp = Extg * exp_logmu
        residual     = Dxtg - weighted_exp
        lnL = float(np.sum(
            Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
        ))
        return lnL, exp_logmu, weighted_exp, residual

    # -------------------------------------------------------------------------
    # FIT STATISTICS — numpy
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, n_basis, nb_years, nb_regions):
        """
        Computes deviance, AIC, BIC for the Li-Lee model.

        Degrees of freedom:
          dofs = n_basis × nb_regions      (α_{x,g})
               + n_basis                    (β_x)
               + n_basis × nb_regions       (β_{x,g})
               + nb_years                   (κ_t)
               + nb_years × nb_regions      (κ_{g,t})
               - nb_regions                 (constraints Σ_x β_{x,g} = 0)
               - nb_regions                 (constraints Σ_t κ_{g,t} = 0)
               - 1                          (constraint Σ_x β_x = 1)
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
        dofs = (n_basis * nb_regions       # α_{x,g}
                + n_basis                   # β_x
                + n_basis * nb_regions      # β_{x,g}
                + nb_years                  # κ_t
                + nb_years * nb_regions     # κ_{g,t}
                - nb_regions                # Σ_x β_{x,g} = 0
                - nb_regions                # Σ_t κ_{g,t} = 0
                - 1)                        # Σ_x β_x = 1

        AIC = 2.0 * dofs - 2.0 * lnL
        BIC = dofs * np.log(nb_obs) - 2.0 * lnL

        return pd.DataFrame(
            [[nb_obs, n_basis, dofs,
              round(lnL, 2), round(deviance, 2), round(AIC, 2), round(BIC, 2)]],
            columns=["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]
        )

    # -------------------------------------------------------------------------
    # FINITE DIFFERENCE MATRIX — used by Classic variant
    # -------------------------------------------------------------------------
    @staticmethod
    def difference_matrix(n, k):
        """
        Construct the finite difference matrix of order k for vectors of length n.
        Returns a (n-k) x n NumPy array.
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
    # Parametric Li-Lee with B-splines + P-splines penalty
    # =========================================================================

    class Parametric:
        """
        Parametric Li-Lee model.

        ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

        All age curves (α, β, β_g) are modeled with B-splines.
        The penalty is a P-splines roughness penalty on the coefficients.

        Parameters
        ----------
        degree     : int   B-spline degree (3 recommended)
        n_knots    : int   number of internal knots (6 recommended)
        xmin/xmax  : float age bounds (inferred from data if None)
        lam        : float P-splines penalty weight
        diff_order : int   difference order (2 recommended)
        nb_iter    : int   max number of NR iterations
        eta0       : float initial learning rate (0.30 recommended)
        tol        : float convergence tolerance
        verbose    : bool  display progress
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

        # ---------------------------------------------------------------------
        # ln(µ) RECONSTRUCTION — LI-LEE MODEL
        # ---------------------------------------------------------------------
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
            Reconstructs ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

            Parameters
            ----------
            alpha_coef  : (nb_regions, n_basis)  α_{x,g} coefficients
            beta_coef   : (n_basis,)             β_x coefficients
            beta_g_coef : (nb_regions, n_basis)  β_{x,g} coefficients
            kappa       : (nb_years,)            common κ_t
            kappa_g     : (nb_regions, nb_years) regional κ_{g,t}

            Returns
            --------
            logmu    : (nb_ages, nb_years, nb_regions)
            alpha    : (nb_ages, nb_regions)  α_{x,g} curves
            beta     : (nb_ages,)             β_x curve
            beta_g   : (nb_ages, nb_regions)  β_{x,g} curves
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

        # ---------------------------------------------------------------------
        # NR UPDATES — LI-LEE MODEL
        # ---------------------------------------------------------------------
        @staticmethod
        def update_alpha_coef(alpha_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
            """
            NR update of α_{x,g} (regional baseline).
            One coefficient vector per region.
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
            NR update of β_x (common sensitivity).
            A single coefficient vector for all regions.
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
            NR update of β_{x,g} (regional sensitivity).
            One coefficient vector per region.
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
            NR update of κ_t (common time factor).
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
            NR update of κ_{g,t} (regional time factors).
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

        # ---------------------------------------------------------------------
        # NORMALIZATION (IDENTIFIABILITY CONSTRAINTS)
        # ---------------------------------------------------------------------
        @staticmethod
        def normalize_lilee(beta_coef, beta_g_coef, kappa, kappa_g, B):
            """
            Li-Lee identifiability constraints:
              1. Σ_x β_x = 1
              2. Σ_x β_{x,g} = 0  for all g
              3. Σ_t κ_{g,t} = 0  for all g

            These constraints ensure uniqueness of the solution.
            """
            # 1. Normalization of β_x : Σ_x β_x = 1
            beta      = B @ beta_coef
            scal_beta = float(np.sum(beta))
            if scal_beta != 0:
                beta_coef = beta_coef / scal_beta
                kappa     = kappa * scal_beta

            # 2. Normalization of β_{x,g} : Σ_x β_{x,g} = 0 for all g
            nb_regions = beta_g_coef.shape[0]
            for g in range(nb_regions):
                beta_g     = B @ beta_g_coef[g]
                sum_beta_g = float(np.sum(beta_g))
                if sum_beta_g != 0:
                    # Subtract mean to center at 0
                    adjustment = sum_beta_g / len(beta_g)
                    beta_g_coef[g] -= adjustment / np.mean(B.sum(axis=0))

            # 3. Normalization of κ_{g,t} : Σ_t κ_{g,t} = 0 for all g
            for g in range(nb_regions):
                mean_kappa_g  = float(np.mean(kappa_g[g]))
                kappa_g[g]   -= mean_kappa_g

            return beta_coef, beta_g_coef, kappa, kappa_g

        # ---------------------------------------------------------------------
        # ROBUST INITIALISATION — B-spline projection of SVD estimates
        # ---------------------------------------------------------------------
        def init_params(self, Dxtg, Extg, xv):
            """
            Robust initialisation for the parametric Li-Lee model.

            Strategy:
              1. Aggregate data across regions → fit a common Lee-Carter (SVD)
                 to extract α_common, β_x, κ_t.
              2. For each region, compute the residual log-rate after removing
                 the common component, then apply SVD to extract β_{x,g}, κ_{g,t}.
              3. Project all age curves onto the B-spline basis via least squares.

            Parameters
            ----------
            Dxtg : (nb_ages, nb_years, nb_regions)   death counts
            Extg : (nb_ages, nb_years, nb_regions)   exposures
            xv   : (nb_ages,)                        age vector

            Returns
            --------
            alpha_coef  : (nb_regions, n_basis)   B-spline coefficients of α_{x,g}
            beta_coef   : (n_basis,)              B-spline coefficients of β_x
            beta_g_coef : (nb_regions, n_basis)   B-spline coefficients of β_{x,g}
            kappa       : (nb_years,)             common time factor κ_t
            kappa_g     : (nb_regions, nb_years)  regional time factors κ_{g,t}
            """
            nb_ages, nb_years, nb_regions = Dxtg.shape

            xmin = self.xmin if self.xmin is not None else float(np.min(xv))
            xmax = self.xmax if self.xmax is not None else float(np.max(xv))

            # =====================================================
            # 1️⃣ AGGREGATED RATES → common component
            # =====================================================
            Dxt = np.sum(Dxtg, axis=2)
            Ext = np.sum(Extg, axis=2)

            Mxt = Dxt / np.maximum(Ext, 1e-12)
            Mxt = np.maximum(Mxt, 1e-12)

            logM = np.log(Mxt)

            # Common alpha (age baseline averaged over time)
            alpha_common = np.mean(logM, axis=1)

            M_centered = logM - alpha_common[:, None]

            # Common SVD
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

            beta_common  = U[:, 0]
            kappa_common = S[0] * Vt[0, :]

            # Standard LC normalisation
            beta_common  /= np.sum(beta_common)
            kappa_common *= np.sum(beta_common)

            # =====================================================
            # 2️⃣ REGIONAL DEVIATIONS
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

                beta_g[:, g]  = U_g[:, 0]
                kappa_g[g, :] = S_g[0] * Vt_g[0, :]

                # Normalisation
                beta_g[:, g]  /= np.sum(beta_g[:, g])
                kappa_g[g, :] *= np.sum(beta_g[:, g])

            # =====================================================
            # 3️⃣ PROJECTION ONTO B-SPLINE BASIS
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

        # ---------------------------------------------------------------------
        # MAIN FITTING METHOD
        # ---------------------------------------------------------------------
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
            Calibrates the parametric Li-Lee model:
                ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

            Parameters
            ----------
            alpha_coef_init  : (nb_regions, n_basis)  init α_{x,g}
            beta_coef_init   : (n_basis,)             init β_x
            beta_g_coef_init : (nb_regions, n_basis)  init β_{x,g}
            kappa_init       : (nb_years,)            init κ_t
            kappa_g_init     : (nb_regions, nb_years) init κ_{g,t}
            Extg             : (nb_ages, nb_years, nb_regions)
            Dxtg             : (nb_ages, nb_years, nb_regions)
            xv               : (nb_ages,)
            tv               : (nb_years,)

            Returns
            --------
            dict with keys: parameters, curves, fitted_values, fit_statistics
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
                beta_coef, beta_g_coef, kappa, kappa_g = self.normalize_lilee(
                    beta_coef, beta_g_coef, kappa, kappa_g, B
                )

            # Final reconstruction
            logmu_final, alpha, beta, beta_g = self.compute_logmu_lilee(
                alpha_coef, beta_coef, beta_g_coef,
                kappa, kappa_g, xv, B, knots, self.degree,
            )

            # Statistics
            Fit_stat = LiLee.compute_fit_stats(
                Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions,
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
                    "alpha_coef": alpha_coef,
                    "beta_coef":  beta_coef,
                    "beta_g_coef": beta_g_coef,
                    "kappa":      kappa,
                    "kappa_g":    kappa_g,
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
            }


    # =========================================================================
    # SUB-CLASS : Classic
    # Classic non-parametric Li-Lee (gradient descent)
    # First fits a Lee-Carter baseline, then fits the regional residual terms
    # =========================================================================

    class Classic:
        """
        Classic (non-parametric) Li-Lee model fitted by gradient descent.

        Step 1 — fits a standard Lee-Carter model (ax, bx, κ_t) on the data.
        Step 2 — fits the regional residual terms (β_{x,g}, κ_{g,t}) on top.

        ln(µ_{x,t,g}) = α_x + β_x·κ_t + β_{x,g}·κ_{g,t}

        Parameters
        ----------
        nb_iter  : int   number of gradient descent iterations
        h        : float P-splines roughness penalty for β_{x,g}
        z        : int   difference order for the penalty matrix
        verbose  : bool  display plots during fitting
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
            Classic Lee-Carter gradient descent.
            Fits α_x, β_x, κ_t on the full dataset.
            Returns updated (ax, bx, kappa, Fit_stat).
            """
            #gradient descent parameter
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
                        kappa_new = kappa + eta * dlnL_dpar
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
            lnL = np.sum(Dxtg * logmuxt_grp - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)
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
            #we return ax, bx, kappa and stats
            return ax, bx, kappa, Fit_stat

        # ---------------------------------------------------------------------
        # MAIN FITTING METHOD
        # ---------------------------------------------------------------------
        def fit(self, ax, bx, bx_gr, kappa, kappa_gr, Extg, Dxtg, Muxtg, xv, tv):
            """
            Fits the classic Li-Lee model.

            Parameters
            ----------
            ax, bx      : (nb_ages, 1)              initial Lee-Carter parameters
            bx_gr       : (nb_ages, nb_regions)     initial regional sensitivity β_{x,g}
            kappa       : (nb_years,)               initial common time factor κ_t
            kappa_gr    : (nb_years, nb_regions)    initial regional time factors κ_{g,t}
            Extg        : (nb_ages, nb_years, nb_regions)
            Dxtg        : (nb_ages, nb_years, nb_regions)
            Muxtg       : (nb_ages, nb_years, nb_regions)
            xv          : (nb_ages,)
            tv          : (nb_years,)

            Returns
            --------
            dict with keys: parameters, fitted_values, fit_statistics
            """
            #matrix of differences, order z
            Kz    = LiLee.difference_matrix(len(ax), self.z)
            KTK   = Kz.T @ Kz
            IdKTK = np.diag(KTK)

            # ax and bx are computed with the Poisson LC
            ax, bx, kappa, _ = self._lc_fit(ax, bx, kappa, Extg, Dxtg, xv, tv, self.nb_iter)

            #gradient descent parameter
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
                    dlnL_baseline = (Dxtg - Extg * np.exp(logmuxt_gr))

                    if (ct_opt == 0):
                        #--------------- bx gr -----------------
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

            # =============================
            # Final log(mu) and mu
            # =============================

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

            # =============================
            # Log-likelihood
            # =============================

            exp_logmuxt = mu_final
            logDxtgFact = gammaln(Dxtg + 1)

            lnL = np.sum(Dxtg * logmu_final
                         - Extg * exp_logmuxt
                         + Dxtg * np.log(Extg)
                         - logDxtgFact)

            nb_obs = Dxtg.size
            dofs   = len(ax) + len(bx) + np.size(bx_gr) + np.size(kappa_gr) + np.size(kappa)
            AIC    = 2 * dofs - 2 * lnL
            BIC    = dofs * np.log(nb_obs) - 2 * lnL

            Fit_stat = [[nb_obs, 'NA', 'NA', dofs, np.round(lnL, 2), np.round(AIC, 2), np.round(BIC, 2)]]
            Fit_stat = pd.DataFrame(Fit_stat)
            Fit_stat.columns = ["N", "m", "degree", "dofs", "lnL", "AIC", "BIC"]

            # =============================
            # Return dictionary
            # =============================

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
