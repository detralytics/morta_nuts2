# -*- coding: utf-8 -*-
"""
Lee-Carter — Object-oriented architecture
===============================================================
Class hierarchy :

    LeeCarter                        ← general base class
    ├── LeeCarter.Parametric         ← parametric Lee-Carter (B-splines + P-splines)
    │   ├── LeeCarter.Parametric.Multiregion   ← β_{x,g} per region
    │   └── LeeCarter.Parametric.National      ← β_x shared across all regions
    └── LeeCarter.Classic            ← classic Lee-Carter (gradient descent)

USAGE :
    # National parametric model
    model = LeeCarter.Parametric.National(n_knots=10, lam=0.1)
    results = model.fit(ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv)

    # Multi-region parametric model
    model = LeeCarter.Parametric.Multiregion(n_knots=10, lam=0.1)
    results = model.fit(ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv)

    # Classic model
    model = LeeCarter.Classic(nb_iter=500)
    results = model.fit(ax, bx, kappa, Extg, Dxtg, xv, tv)

USES STANDARD PYTHON LIBRARIES:
  ✓ scipy.interpolate.BSpline      — B-spline construction
  ✓ scipy.interpolate.make_lsq_spline — least squares fit
  ✓ scipy.special.gammaln          — log-factorial
  ✓ scipy.linalg                   — linear algebra
  ✓ numpy                          — matrix computation
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
    General Lee-Carter class.

    Groups together :
      - shared mathematical tools (static methods)
      - the two main model families as nested sub-classes :
          · LeeCarter.Parametric   (B-splines, P-splines)
          · LeeCarter.Classic      (non-parametric, gradient descent)

    Quick usage example
    -------------------
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
        Builds the P-splines penalty matrix D^T D.
        Uses numpy to build the finite difference matrix.
        
        P(c) = λ · c^T (D^T D) c
        
        Parameters
        ----------
        n_basis    : int   number of basis functions
        diff_order : int   difference order (1 = 1st derivative, 2 = 2nd derivative)
        
        Returns
        --------
        DtD      : (n_basis, n_basis)
        diag_DtD : (n_basis,)   diagonal (for Hessian approximation)
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
        Poisson log-likelihood (equation 2 of the paper).
        Uses scipy.special.gammaln for log(D!).
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
        """NR update of B-spline coefficients of α_x."""
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
        """NR update of κ_t."""
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
    def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, n_basis, nb_years, nb_regions):
        """Computes Poisson deviance, AIC, BIC."""
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
        dofs   = n_basis + n_basis * nb_regions + nb_years
        AIC    = 2.0 * dofs - 2.0 * lnL
        BIC    = dofs * np.log(nb_obs) - 2.0 * lnL

        return pd.DataFrame(
            [[nb_obs, n_basis, dofs,
              round(lnL, 2), round(deviance, 2), round(AIC, 2), round(BIC, 2)]],
            columns=["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]
        )

    # -------------------------------------------------------------------------
    # UTILITY — build input matrices from a long-format DataFrame
    # -------------------------------------------------------------------------
    @staticmethod
    def build_input_from_dataframe(df):
        """
        Transforms a long DataFrame into 3D matrices compatible with .fit().
        
        Parameters
        ----------
        df : DataFrame with columns
             ['region', 'year', 'age', 'deaths', 'exposure', 'mortality_rate']
        
        Returns
        --------
        Muxtg   : (nb_ages, nb_years, nb_regions)
        Dxtg    : (nb_ages, nb_years, nb_regions)
        Extg    : (nb_ages, nb_years, nb_regions)
        xv      : sorted age vector
        tv      : sorted year vector
        regions : list of regions
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
    # Groups the two parametric variants (B-splines + P-splines)
    # =========================================================================

    class Parametric:
        """
        Parametric Lee-Carter with B-splines.
        
        ln(µ_{x,t,g}) = α_x  +  β_{x,g} · κ_t
        
        Contains two nested variants :
          · LeeCarter.Parametric.Multiregion  — β_{x,g} per region
          · LeeCarter.Parametric.National     — β_x shared across all regions
        """

        # ---------------------------------------------------------------------
        # MULTI-REGION VARIANT
        # ---------------------------------------------------------------------
        class Multiregion:
            """
            Parametric Lee-Carter — multi-region variant.
            β_{x,g} has a separate B-spline curve for each region g.

            Parameters
            ----------
            degree     : int   B-spline degree
            n_knots    : int   number of internal knots
            xmin/xmax  : float age bounds (inferred from data if None)
            lam        : float P-splines penalty
            diff_order : int   penalty difference order
            nb_iter    : int   max iterations
            eta0       : float initial learning rate
            tol        : float convergence tolerance
            verbose    : bool
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
            # 3. ln(µ) RECONSTRUCTION — numpy + scipy.interpolate
            # -----------------------------------------------------------------
            @staticmethod
            def compute_logmu(ax_coef, bx_coef, kappa, xv, B, knots, degree):
                """
                Reconstructs ln(µ_{x,t,g}) = α_x + β_{x,g} · κ_t
                using scipy.interpolate.BSpline.
                
                Parameters
                ----------
                ax_coef  : (n_basis,)            B-spline coefficients of α_x
                bx_coef  : (nb_regions, n_basis) B-spline coefficients of β_{x,g}
                kappa    : (nb_years,)           endogenous factor κ_t
                xv       : (nb_ages,)            age vector
                B        : (nb_ages, n_basis)    basis matrix (pre-computed)
                knots    : array                 full knot vector
                degree   : int
                
                Returns
                --------
                logmu  : (nb_ages, nb_years, nb_regions)
                ax     : (nb_ages,)
                bx_reg : (nb_ages, nb_regions)
                """
                nb_regions = bx_coef.shape[0]
                nb_ages    = len(xv)

                # Evaluate α_x with scipy.interpolate.BSpline
                ax = B @ ax_coef  # equivalent to eval_bspline_from_coef(ax_coef, xv, knots, degree)

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
            # 5. NR UPDATE — bx_coef multi-region
            # -----------------------------------------------------------------
            @staticmethod
            def update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
                """NR update of B-spline coefficients of β_{x,g}."""
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
            # 6. FINAL RESCALING — numpy
            # -----------------------------------------------------------------
            @staticmethod
            def rescale_bx_kappa(bx_coef, bx_reg, kappa):
                """Normalization: Σ_x (mean_g β_{x,g}) = 1"""
                bx_avg      = np.mean(bx_reg, axis=1)
                scal_factor = float(np.sum(bx_avg))
                if scal_factor == 0:
                    return bx_coef, bx_reg, kappa
                return bx_coef / scal_factor, bx_reg / scal_factor, kappa * scal_factor

            # -----------------------------------------------------------------
            # MAIN FITTING METHOD
            # -----------------------------------------------------------------
            def fit(self, ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv):
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

                DtD, diag_DtD = LeeCarter.make_penalty_matrix(n_basis, self.diff_order)
                logDxtgFact   = gammaln(Dxtg + 1)

                lnL       = -np.inf
                Delta_lnL = 0.0
                eta       = self.eta0
                it        = -1

                # ===============================
                # 🔵 Patience-based early stopping
                # ===============================
                best_lnL  = -np.inf
                patience  = 40           # max number of iterations without improvement
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
                    lnL_new, _, weighted_exp, residual = LeeCarter.poisson_lnL(
                        Dxtg, Extg, logmu, logDxtgFact
                    )

                    Delta_lnL = lnL_new - lnL

                    if self.verbose and (it % 10 == 0):
                        print(f"It {it:4d} | lnL = {lnL_new:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

                    # ===============================
                    # 🔵 Early stopping logic
                    # ===============================
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
                    ax_coef = LeeCarter.update_ax_coef(ax_coef, B, residual, weighted_exp, eta, self.lam, DtD, diag_DtD)
                    bx_coef = self.update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, self.lam, DtD, diag_DtD)
                    kappa   = LeeCarter.update_kappa(kappa, bx_reg, residual, weighted_exp, eta)

                # ===============================
                # Final rescaling
                # ===============================
                bx_coef, bx_reg, kappa = self.rescale_bx_kappa(bx_coef, bx_reg, kappa)

                logmu_final, ax, bx_reg = self.compute_logmu(
                    ax_coef, bx_coef, kappa, xv, B, knots, self.degree
                )

                Fit_stat = LeeCarter.compute_fit_stats(
                    Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions
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


        # ---------------------------------------------------------------------
        # NATIONAL VARIANT
        # ---------------------------------------------------------------------
        class National:
            """
            Parametric Lee-Carter — national variant.
            β_x is common to all regions (no g dimension).

            Parameters
            ----------
            degree     : int   B-spline degree
            n_knots    : int   number of internal knots
            xmin/xmax  : float age bounds (inferred from data if None)
            lam        : float P-splines penalty
            diff_order : int   penalty difference order
            nb_iter    : int   max iterations
            eta0       : float initial learning rate
            tol        : float convergence tolerance
            verbose    : bool
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
                ln(µ_{x,t,g}) = α_x + β_x · κ_t
                β_x common to all regions.

                Parameters
                ----------
                ax_coef : (n_basis,)
                bx_coef : (n_basis,)      ← no region dimension
                kappa   : (nb_years,)
                xv      : (nb_ages,)
                B       : (nb_ages, n_basis)

                Returns
                --------
                logmu  : (nb_ages, nb_years, 1)   broadcast over g
                ax     : (nb_ages,)
                bx     : (nb_ages,)
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
                NR update of β_x (common to all regions).
                residual / weighted_exp: (nb_ages, nb_years, nb_regions)
                Sum over regions to aggregate the gradient.
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
                """Normalization: Σ_x β_x = 1"""
                scal_factor = float(np.sum(bx))
                if scal_factor == 0:
                    return bx_coef, bx, kappa
                return bx_coef / scal_factor, bx / scal_factor, kappa * scal_factor

            # -----------------------------------------------------------------
            # MAIN FITTING METHOD — national version
            # -----------------------------------------------------------------
            def fit(self, ax_coef_init, bx_coef_init, kappa_init, Extg, Dxtg, xv, tv):
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

                while it < self.nb_iter:
                    it += 1

                    # Adaptive learning rate
                    if Delta_lnL < 0:
                        eta *= 0.5
                    else:
                        eta = min(eta * 1.05, 2.0)

                    # Reconstruction — common β_x, broadcast over nb_regions
                    logmu, ax, bx = self.compute_logmu_national(ax_coef, bx_coef, kappa, xv, B)

                    # Log-likelihood (logmu automatically broadcast over nb_regions)
                    lnL_new, _, weighted_exp, residual = LeeCarter.poisson_lnL(
                        Dxtg, Extg, logmu, logDxtgFact
                    )

                    Delta_lnL = lnL_new - lnL

                    if self.verbose and (it % 10 == 0):
                        print(f"It {it:4d} | lnL = {lnL_new:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

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

                    if abs(Delta_lnL) < self.tol:
                        if self.verbose:
                            print("\nConvergence reached (tolerance).")
                        break

                    lnL = lnL_new

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

                Fit_stat = LeeCarter.compute_fit_stats(
                    Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions
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
        
        ln(µ_{x,t}) = α_x + β_x · κ_t

        Parameters
        ----------
        nb_iter : int   number of gradient descent iterations
        eta     : float learning rate
        """

        def __init__(self, nb_iter=500, eta=1):
            self.nb_iter = nb_iter
            self.eta     = eta

        def fit(self, ax, bx, kappa, Extg, Dxtg, xv, tv):
            #gradient descent parameter
            eta = self.eta
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


