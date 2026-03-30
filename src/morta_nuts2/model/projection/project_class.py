"""
Projection
=====================================================

This module provides classes and utilities for prospective mortality projection,
high-age extrapolation, life expectancy computation, and actuarial annuity pricing.

Main components
---------------
- :func:`kannisto_log_mu` : Low-level Kannisto log-mortality formula.
- :class:`LifeExpectancy` : Computes period life expectancy from mortality rates.
- :class:`ProjectorLC` : Lee-Carter projection.
- :class:`ProjectorLL` : Li-Lee two-factor projection (common + regional).
- :class:`HighAgeExtrapolator` : Extrapolates log-mortality beyond the observed age range.
- :func:`concat_logmu_time` : Concatenates historical and projected log-mortality arrays.
- :func:`Annuity_pricing` : Actuarial present value of life annuities.
- :func:`compute_mae` : Mean absolute error and weighted MAE between observed and modelled rates.
"""


import numpy as np
from scipy.optimize import curve_fit



# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def kannisto_log_mu(x, a, b):
    """
    Evaluate the log force of mortality under the Kannisto model.

    The Kannisto (logistic Gompertz) parametric model expresses the force of
    mortality as a logistic function of age:

    .. math::

        \\mu(x) = \\frac{a \\, e^{bx}}{1 + a \\, e^{bx}}

    This function returns :math:`\\log \\mu(x)` directly for numerical
    stability (avoids computing ``exp`` then ``log``).

    Parameters
    ----------
    x : array-like
        Ages at which to evaluate the model.
    a : float
        Scale parameter (:math:`a > 0`). Controls the level of mortality.
    b : float
        Growth parameter (:math:`b > 0`). Controls the rate of mortality
        increase with age.

    Returns
    -------
    ndarray
        :math:`\\log \\mu(x)` evaluated at each age in ``x``.

    Notes
    -----
    As :math:`x \\to \\infty`, :math:`\\mu(x) \\to 1`, so the model imposes
    an implicit upper bound of 1 on the force of mortality — a key difference
    from the unbounded Gompertz model.
    """
    ebx = np.exp(b * x)
    return np.log(a * ebx / (1.0 + a * ebx))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Life expectancy
# ─────────────────────────────────────────────────────────────────────────────

class LifeExpectancy:
    """
    Computes period life expectancy from an array of force-of-mortality rates.

    Period life expectancy at age :math:`x` is defined as the expected number
    of additional years lived by a cohort subject throughout its life to the
    age-specific mortality rates observed at a given point in time:

    .. math::

        e_x = \\frac{\\sum_{y=x}^{\\omega} {}_y p_0}{{}_{x} p_0}
            = \\sum_{t=0}^{\\omega - x} {}_t p_x

    where the survival probability is derived from the force of mortality via:

    .. math::

        {}_t p_x = \\exp\\!\\left(-\\int_x^{x+t} \\mu(s)\\, ds\\right)
                 \\approx \\prod_{k=0}^{t-1} \\exp(-\\mu_{x+k})

    Parameters
    ----------
    mu_future : ndarray
        Array of force-of-mortality rates.

        - Shape ``(nb_ages, horizon, nb_regions)`` for the deterministic case.
        - Shape ``(nb_ages, horizon, nb_regions, n_sim)`` for the stochastic
          case.
    """

    def __init__(self, mu_future: np.ndarray):
        if mu_future.ndim not in (3, 4):
            raise ValueError("mu_future must be 3D or 4D")
        self.mu_future = mu_future

    # ------------------------------------------------------------------
    def _compute(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute life expectancy from a force-of-mortality array (vectorized).

        Works for both 3-D ``(nb_ages, horizon, nb_regions)`` and 4-D
        ``(nb_ages, horizon, nb_regions, n_sim)`` inputs.

        The algorithm uses the discrete approximation:

        .. math::

            p_x = \\exp(-\\mu_x), \\qquad
            {}_t p_0 = \\prod_{k=0}^{t-1} p_k = \\exp\\!\\left(
                       \\sum_{k=0}^{t-1} \\log p_k \\right)

        Life expectancy at age :math:`x` is then:

        .. math::

            e_x = \\frac{\\displaystyle\\sum_{y \\ge x} {}_y p_0}{{}_{x} p_0}

        which is computed as a reversed cumulative sum of :math:`{}_y p_0`
        divided by :math:`{}_{x-1} p_0` (survival from age 0 to age :math:`x`).

        Parameters
        ----------
        mu : ndarray
            Force-of-mortality array, 3-D or 4-D.

        Returns
        -------
        ndarray
            Life expectancy array with the same shape as ``mu``.
        """
        px        = np.exp(-mu)
        log_px    = np.log(np.maximum(px, 1e-300))
        cumlog_px = np.cumsum(log_px, axis=0)

        # shift cumulative log by 1 to get survival starting at each age x
        cumlog_shifted          = np.zeros_like(cumlog_px)
        cumlog_shifted[1:, ...] = cumlog_px[:-1, ...]

        surv_from_0 = np.exp(cumlog_px)

        # reversed cumulative sum: sum of surv[y] for y >= x
        rev_cumsum = np.cumsum(surv_from_0[::-1, ...], axis=0)[::-1, ...]

        # divisor: survival from age 0 to age x-1
        divisor          = np.ones_like(surv_from_0)
        divisor[1:, ...] = surv_from_0[:-1, ...]

        return rev_cumsum / divisor

    # ------------------------------------------------------------------
    def compute(self) -> np.ndarray:
        """
        Compute life expectancy for the full ``mu_future`` array.

        Returns
        -------
        ndarray
            Life expectancy array with the same shape as ``mu_future``:

            - ``(nb_ages, horizon, nb_regions)`` in the deterministic case.
            - ``(nb_ages, horizon, nb_regions, n_sim)`` in the stochastic case.
        """
        return self._compute(self.mu_future)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lee-Carter SVD projection
# ─────────────────────────────────────────────────────────────────────────────

class ProjectorLC:
    """

    The Lee-Carter model decomposes the log force of mortality as:

    .. math::

        \\ln \\mu(x, t) = \\alpha_x + \\beta_x \\, \\kappa_t + \\varepsilon_{x,t}

    where :math:`\\alpha_x` is the age-specific baseline, :math:`\\beta_x` is
    the age-sensitivity pattern, and :math:`\\kappa_t` is the period index.

    For the **multi-regional** variant, the period indices across regions are
    decomposed via SVD:

    .. math::

        K = U \\, S \\, V^\\top \\approx Z_{\\text{red}} \\, V_{\\text{red}}^\\top

    where :math:`Z_{\\text{red}} = U_{[1:r]} S_{[1:r]}` collects the first
    ``nb_components`` principal scores and :math:`V_{\\text{red}}` the
    corresponding regional loadings.

    Two period-index dynamics are supported:

    **Random walk with drift (``model='rw'``):**

    .. math::

        Z_{T+h} = Z_T + \\sum_{s=1}^{h} \\varepsilon_s, \\qquad
            \\varepsilon_s \\sim \\mathcal{N}(\\hat{\\delta},\\, \\hat{\\Sigma})

    **Linear trend (``model='linear'``):**

    .. math::

        Z_t = \\alpha + \\beta \\, t + \\varepsilon_t, \\qquad
        \\varepsilon_t \\sim \\mathcal{N}(0, \\Sigma_{\\text{res}})

    Parameters
    ----------
    results : dict
        Must contain ``results["curves"]["alpha_x"]``, ``results["curves"]["beta_xg"]``,
        and ``results["parameters"]["kappa"]``.
    tv : array-like
        Observation year vector.
    horizon : int, optional
        Number of projection years (default ``30``).
    exclude_years : list, optional
        Years to exclude before fitting (default ``[2020, 2021]``).
    nb_components : int, optional
        Number of SVD components to retain (default ``1``).
    model : {'rw', 'linear'}, optional
        Projection model for :math:`\\kappa_t` (default ``'rw'``).
    stochastic : bool, optional
        If ``True``, generates ``n_sim`` Monte-Carlo paths (default ``True``).
    n_sim : int, optional
        Number of stochastic simulations (default ``1000``).
    """

    def __init__(
        self,
        results: dict,
        tv,
        horizon: int = 30,
        exclude_years: list = None,
        nb_components: int = 1,
        model: str = "rw",
        stochastic: bool = True,
        n_sim: int = 1000,
    ):
        params = results["parameters"]

        # ── Variante parametric : courbes dans results["curves"] ─────────
        if "curves" in results:
            curves   = results["curves"]
            beta_key = next((k for k in curves if k.startswith("beta")), None)
            if beta_key is None:
                raise KeyError("No beta curve found in results['curves']")
            self.ax        = curves["alpha_x"]          # (nb_ages,)
            self.bx        = curves[beta_key]            # (nb_ages,) ou (nb_ages, nb_regions)
            self.kappa_raw = params["kappa"]             # (nb_years,) ou (nb_regions, nb_years)

        # ── Variante classic : ax/bx dans results["parameters"] ─────────
        # ax[x,g], bx[x,g] : (nb_ages, nb_regions)
        # kappa[x,t]        : (nb_ages, nb_years) mais constant sur x → extraire kappa[0,:]
        else:
            self.ax        = params["ax_coef"]           # (nb_ages, nb_regions)
            self.bx        = params["bx_coef"]           # (nb_ages, nb_regions)
            kappa_raw      = np.asarray(params["kappa"]) # (nb_ages, nb_years) ou (nb_years,)
            # kappa ne dépend que de t : on prend la première ligne si 2D
            self.kappa_raw = kappa_raw[0, :] if kappa_raw.ndim == 2 else kappa_raw  # (nb_years,)
        self.tv            = np.asarray(tv)
        self.horizon       = horizon
        self.exclude_years = exclude_years if exclude_years is not None else [2020, 2021]
        self.nb_components = nb_components
        self.model         = model
        self.stochastic    = stochastic
        self.n_sim         = n_sim

    # ------------------------------------------------------------------
    def _svd_factors(self):
        """
        Build the reduced score matrix ``Zred`` and loading matrix ``Vred`` via SVD.

        Years listed in ``exclude_years`` are removed before decomposition.
        The raw kappa matrix :math:`K` (shape ``(T, nb_regions)``) is
        factorised as:

        .. math::

            K \\approx Z_{\\text{red}} \\, V_{\\text{red}}^\\top

        where :math:`Z_{\\text{red}} = U_{[:,1:r]} \\cdot S_{[1:r]}` and
        :math:`V_{\\text{red}} = V^\\top_{[1:r, :]}{}^\\top`, retaining the
        first ``nb_components`` singular triplets.

        Returns
        -------
        Zred : ndarray of shape ``(T, nb_components)``
            Principal score matrix (time × components).
        Vred : ndarray of shape ``(nb_regions, nb_components)``
            Regional loading matrix.
        """
        X_svd = self.kappa_raw.reshape(-1, 1) if self.kappa_raw.ndim == 1 else self.kappa_raw.T
        X_svd = X_svd[~np.isin(self.tv, self.exclude_years), :]

        U, S, Vt = np.linalg.svd(X_svd, full_matrices=False)
        Vred = Vt[:self.nb_components, :].T                        # (nb_regions, nb_components)
        Zred = U[:, :self.nb_components] * S[:self.nb_components]  # (T, nb_components)
        return Zred, Vred

    # ------------------------------------------------------------------
    def _project_rw(self, Zred, Vred):
        """
        Project the reduced score matrix ``Zred`` with a random walk with drift.

        Each step is drawn from a multivariate normal centred on the empirical
        drift :math:`\\hat{\\delta}`, so the drift is already embedded in the
        distribution of the increments.  The projected score at horizon
        :math:`h` is:

        .. math::

            Z_{T+h} = Z_T + \\sum_{s=1}^{h} \\varepsilon_s, \\qquad
            \\varepsilon_s \\sim \\mathcal{N}(\\hat{\\delta},\\, \\hat{\\Sigma})

        where:

        - :math:`\\hat{\\delta} = \\overline{\\Delta Z}` is the sample mean of
          first differences (annual drift).
        - :math:`\\hat{\\Sigma}` is the sample covariance of first differences.

        In expectation:
        :math:`\\mathbb{E}[Z_{T+h}] = Z_T + h\\,\\hat{\\delta}`,
        and uncertainty grows with the horizon through the cumulative sum
        of independent draws.

        The projected regional kappa is then reconstructed as:

        .. math::

            \\kappa^{\\text{future}} = Z^{\\text{future}} \\, V_{\\text{red}}^\\top

        Parameters
        ----------
        Zred : ndarray of shape ``(T, nb_components)``
        Vred : ndarray of shape ``(nb_regions, nb_components)``

        Returns
        -------
        kappa_future : ndarray
            - ``(horizon, nb_regions)`` if deterministic.
            - ``(horizon, n_sim, nb_regions)`` if stochastic.
        """
        diffs  = np.diff(Zred, axis=0)
        drift  = np.mean(diffs, axis=0)
        cov    = np.cov(diffs, rowvar=False).reshape(self.nb_components, self.nb_components)
        Z_last = Zred[-1, :]

        if not self.stochastic:
            Z_future     = Z_last + drift * np.arange(1, self.horizon + 1)[:, None]
            kappa_future = Z_future @ Vred.T 
        else:
            steps        = np.random.multivariate_normal(drift, cov, size=(self.horizon, self.n_sim))
            Z_future     = Z_last + np.cumsum(steps, axis=0)
            kappa_future = Z_future @ Vred.T
        return kappa_future

    # ------------------------------------------------------------------
    def _project_linear(self, Zred, Vred):
        """
        Project the reduced score matrix ``Zred`` with a linear trend model.

        The score vector is modelled as:

        .. math::

            Z_t = \\alpha + \\beta \\, t + \\varepsilon_t, \\qquad
            \\varepsilon_t \\sim \\mathcal{N}(0,\\, \\hat{\\Sigma}_{\\text{res}})

        Coefficients :math:`(\\alpha, \\beta)` are estimated by ordinary least
        squares. The deterministic projection is :math:`\\hat{Z}_{T+h} = \\alpha
        + \\beta(T+h)`. Stochastic paths add residual noise drawn from
        :math:`\\mathcal{N}(0, \\hat{\\Sigma}_{\\text{res}})`.

        The projected regional kappa is:

        .. math::

            \\kappa^{\\text{future}} = Z^{\\text{future}} \\, V_{\\text{red}}^\\top

        Parameters
        ----------
        Zred : ndarray of shape ``(T, nb_components)``
        Vred : ndarray of shape ``(nb_regions, nb_components)``

        Returns
        -------
        kappa_future : ndarray
            - ``(horizon, nb_regions)`` if deterministic.
            - ``(horizon, n_sim, nb_regions)`` if stochastic.
        """
        T          = Zred.shape[0]
        time_idx   = np.arange(T)
        X_des      = np.column_stack([np.ones(T), time_idx])
        beta_lin   = np.linalg.lstsq(X_des, Zred, rcond=None)[0]

        future_idx = np.arange(T, T + self.horizon)
        Xf         = np.column_stack([np.ones(self.horizon), future_idx])
        Z_det      = Xf @ beta_lin

        if not self.stochastic:
            kappa_future = Z_det @ Vred.T
        else:
            residuals = Zred - X_des @ beta_lin
            cov_res   = np.cov(residuals, rowvar=False).reshape(self.nb_components, self.nb_components)
            noise        = np.random.multivariate_normal(
                np.zeros(self.nb_components), cov_res, size=(self.horizon, self.n_sim)
            )
            Z_future     = Z_det[:, None, :] + noise
            kappa_future = Z_future @ Vred.T
        return kappa_future

    # ------------------------------------------------------------------
    def _reconstruct(self, kappa_future):
        """
        Reconstruct log(μ) and μ from projected kappa trajectories.

        Applies the Lee-Carter identity to recover log-mortality:

        .. math::

            \\ln \\mu(x, t) = \\alpha_x + \\beta_x \\, \\kappa_t

        and exponentiates to obtain :math:`\\mu(x, t) = e^{\\ln \\mu(x,t)}`.

        Expected shapes of ``kappa_future`` depending on the projection path:

        .. list-table::
           :header-rows: 1

           * - Variant
             - Stochastic
             - ``kappa_future`` shape
           * - Classic 1-D
             - No
             - ``(horizon,)``
           * - Classic 1-D
             - Yes
             - ``(horizon, n_sim)``
           * - SVD (≥1 component)
             - No
             - ``(horizon, nb_regions)``
           * - SVD (≥1 component)
             - Yes
             - ``(horizon, n_sim, nb_regions)``

        Parameters
        ----------
        kappa_future : ndarray
            Projected period-index array (see table above).

        Returns
        -------
        dict
            Deterministic path keys: ``kappa_future``, ``logmu_future``,
            ``mu_future``.

            Stochastic path keys: ``kappa_paths``, ``kappa_lower``,
            ``kappa_median``, ``kappa_upper``, ``logmu_sim``, ``mu_sim``,
            ``mu_lower``, ``mu_median``, ``mu_upper``.
        """
        ax, bx  = self.ax, self.bx
        classic = ax.ndim == 2   # classic: ax (nb_ages, nb_regions), parametric: ax (nb_ages,)

        # ── Cas déterministe ────────────────────────────────────────────────
        if not self.stochastic:
            if classic and kappa_future.ndim == 1:
                # classic sans SVD : kappa_future (horizon,)
                # logmu : (nb_ages, horizon, nb_regions)
                logmu_future = (
                    ax[:, None, :]
                    + bx[:, None, :] * kappa_future[None, :, None]
                )
            else:
                # SVD (nb_components ≥ 1) ou parametric :
                # kappa_future (horizon, nb_regions)  →  logmu : (nb_ages, horizon, nb_regions)
                kf = kappa_future[:, None] if kappa_future.ndim == 1 else kappa_future  # (horizon, nb_regions)
                if classic:
                    # ax (nb_ages, nb_regions), bx (nb_ages, nb_regions)
                    logmu_future = ax[:, None, :] + bx[:, None, :] * kf[None, :, :]
                elif bx.ndim == 1:
                    logmu_future = ax[:, None, None] + bx[:, None, None] * kf[None, :, :]
                else:
                    logmu_future = ax[:, None, None] + bx[:, None, :] * kf[None, :, :]
            mu_future = np.exp(logmu_future)
            return {
                "kappa_future": kappa_future,
                "logmu_future": logmu_future,
                "mu_future":    mu_future,
            }

        # ── Cas stochastique ────────────────────────────────────────────────
        else:
            if classic and kappa_future.ndim == 2:
                # classic sans SVD : kappa_future (horizon, n_sim)
                # logmu_sim : (nb_ages, horizon, nb_regions, n_sim)
                logmu_sim = (
                    ax[:, None, :, None]
                    + bx[:, None, :, None] * kappa_future[None, :, None, :]
                )
            else:
                # SVD (nb_components ≥ 1) ou parametric :
                # kappa_future (horizon, n_sim, nb_regions)
                # reshape → (1, horizon, nb_regions, n_sim)
                kf = kappa_future[:, :, None] if kappa_future.ndim == 2 else kappa_future  # (horizon, n_sim, nb_regions)
                kf_4d = kf.transpose(0, 2, 1)[None, :, :, :]  # (1, horizon, nb_regions, n_sim)
                if classic:
                    # ax (nb_ages, nb_regions), bx (nb_ages, nb_regions)
                    logmu_sim = ax[:, None, :, None] + bx[:, None, :, None] * kf_4d
                elif bx.ndim == 1:
                    logmu_sim = ax[:, None, None, None] + bx[:, None, None, None] * kf_4d
                else:
                    logmu_sim = ax[:, None, None, None] + bx[:, None, :, None] * kf_4d
            mu_sim = np.exp(logmu_sim)

            mu_lower  = np.percentile(mu_sim,  2.5, axis=-1)
            mu_median = np.percentile(mu_sim, 50,   axis=-1)
            mu_upper  = np.percentile(mu_sim, 97.5, axis=-1)

            kappa_lower  = np.percentile(kappa_future,  2.5, axis=1)
            kappa_median = np.percentile(kappa_future, 50,   axis=1)
            kappa_upper  = np.percentile(kappa_future, 97.5, axis=1)

            return {
                "kappa_paths":  kappa_future,
                "kappa_lower":  kappa_lower,
                "kappa_median": kappa_median,
                "kappa_upper":  kappa_upper,
                "logmu_sim":    logmu_sim,
                "mu_sim":       mu_sim,
                "mu_lower":     mu_lower,
                "mu_median":    mu_median,
                "mu_upper":     mu_upper,
            }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _project_rw_1d(self):
        """
        Project a scalar kappa series with a random walk with drift.

        Used for the classic (non-SVD) Lee-Carter variant where a single
        :math:`\\kappa_t` drives all regions simultaneously.

        The model is:

        .. math::

            \\kappa_{t+1} = \\kappa_t + \\hat{\\delta} + \\varepsilon_t, \\qquad
            \\varepsilon_t \\sim \\mathcal{N}(0,\\, \\hat{\\sigma}^2)

        where :math:`\\hat{\\delta} = \\overline{\\Delta\\kappa}` is the sample
        mean of annual differences and :math:`\\hat{\\sigma}` is their standard
        deviation.

        Returns
        -------
        ndarray
            - Shape ``(horizon,)`` if deterministic.
            - Shape ``(horizon, n_sim)`` if stochastic.
        """
        kappa_m = self.kappa_raw[~np.isin(self.tv, self.exclude_years)]
        diffs   = np.diff(kappa_m)
        drift   = np.mean(diffs)
        sigma   = np.std(diffs, ddof=1)
        k_last  = kappa_m[-1]
        t_range = np.arange(1, self.horizon + 1)
        if not self.stochastic:
            return k_last + drift * t_range                          # (horizon,)
        else:
            steps = np.random.normal(drift, sigma, size=(self.horizon, self.n_sim))
            return k_last + np.cumsum(steps, axis=0)                 # (horizon, n_sim)

    # ------------------------------------------------------------------
    def _project_linear_1d(self):
        """
        Project a scalar kappa series with a linear trend model.

        Used for the classic (non-SVD) Lee-Carter variant. The model is:

        .. math::

            \\kappa_t = \\alpha + \\beta \\, t + \\varepsilon_t, \\qquad
            \\varepsilon_t \\sim \\mathcal{N}(0,\\, \\hat{\\sigma}_{\\text{res}}^2)

        Coefficients are estimated by OLS. Future values are the deterministic
        trend :math:`\\hat{\\alpha} + \\hat{\\beta}(T+h)` plus optional Gaussian
        residual noise.

        Returns
        -------
        ndarray
            - Shape ``(horizon,)`` if deterministic.
            - Shape ``(horizon, n_sim)`` if stochastic.
        """
        kappa_m = self.kappa_raw[~np.isin(self.tv, self.exclude_years)]
        T        = len(kappa_m)
        time_idx = np.arange(T)
        X_des    = np.column_stack([np.ones(T), time_idx])
        beta_lin = np.linalg.lstsq(X_des, kappa_m, rcond=None)[0]
        Xf       = np.column_stack([np.ones(self.horizon), np.arange(T, T + self.horizon)])
        k_det    = Xf @ beta_lin                                     # (horizon,)
        if not self.stochastic:
            return k_det
        else:
            residuals = kappa_m - X_des @ beta_lin
            sigma     = np.std(residuals, ddof=1)
            noise     = np.random.normal(0, sigma, size=(self.horizon, self.n_sim))
            return k_det[:, None] + noise                            # (horizon, n_sim)

    # ------------------------------------------------------------------
    def project(self) -> dict:
        """
        Run the full Lee-Carter projection and return all outputs.

        Dispatches to the appropriate internal methods depending on whether
        kappa is scalar (classic 1-D variant) or multi-regional (SVD variant),
        then calls :meth:`_reconstruct` to apply the Lee-Carter identity and
        produce mortality arrays.

        Returns
        -------
        dict
            Deterministic keys: ``kappa_future``, ``logmu_future``, ``mu_future``.

            Stochastic keys: ``kappa_paths``, ``kappa_lower``, ``kappa_median``,
            ``kappa_upper``, ``logmu_sim``, ``mu_sim``, ``mu_lower``,
            ``mu_median``, ``mu_upper``.
        """
        classic = self.kappa_raw.ndim == 1   # True for classic LC single-kappa variant

        if classic:
            # No SVD: direct projection of scalar kappa
            if self.model == "rw":
                kappa_future = self._project_rw_1d()
            elif self.model == "linear":
                kappa_future = self._project_linear_1d()
            else:
                raise ValueError("model must be 'rw' or 'linear'")
        else:
            Zred, Vred = self._svd_factors()
            if self.model == "rw":
                kappa_future = self._project_rw(Zred, Vred)
            elif self.model == "linear":
                kappa_future = self._project_linear(Zred, Vred)
            else:
                raise ValueError("model must be 'rw' or 'linear'")

        return self._reconstruct(kappa_future)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lee-Li projection
# ─────────────────────────────────────────────────────────────────────────────

class ProjectorLL:
    """
    Li-Lee prospective mortality projection (common factor + regional components).

    The Li-Lee model extends the Lee-Carter model by decomposing log-mortality
    into a common trend shared by all regions and region-specific deviations:

    .. math::

        \\ln \\mu_{x,t}^{(g)} = \\alpha_x^{(g)}
            + \\beta_x \\, \\kappa_t
            + \\beta_x^{(g)} \\, \\kappa_t^{(g)}

    where:

    - :math:`\\alpha_x^{(g)}` is the age-region baseline log-mortality.
    - :math:`\\beta_x` and :math:`\\kappa_t` are the common age-loading and
      time-index (shared across regions).
    - :math:`\\beta_x^{(g)}` and :math:`\\kappa_t^{(g)}` are the regional
      age-loading and time-index (region :math:`g`-specific).

    The regional kappa series :math:`\\kappa_t^{(g)}` is compressed via SVD
    into ``nb_components`` principal components before projection.

    Parameters
    ----------
    results : dict
        Estimation output.  Must contain:

        - ``results["curves"]`` (parametric variant) with keys
          ``alpha_xg``, ``beta_x``, ``beta_xg``, or
        - ``results["parameters"]`` (classic variant) with keys
          ``ax``, ``bx``, ``bx_gr``, ``kappa``, ``kappa_gr``.
    tv : array-like
        Observation year vector aligned with kappa.
    horizon : int, optional
        Number of projection years (default 30).
    exclude_years : list, optional
        Years excluded from drift/covariance estimation (default [2020, 2021]).
    nb_components : int, optional
        Number of SVD components retained for the regional kappa (default 1).
    model : {'rw', 'linear'}, optional
        Time-series model for kappa projection (default ``'rw'``).
    stochastic : bool, optional
        If ``True``, generate Monte Carlo paths (default ``True``).
    n_sim : int, optional
        Number of stochastic simulations (default 1000).
    """

    def __init__(
        self,
        results: dict,
        tv,
        horizon: int = 30,
        exclude_years: list = None,
        nb_components: int = 1,
        model: str = "rw",
        stochastic: bool = True,
        n_sim: int = 1000,
    ):
        curves = results.get("curves")
        params = results["parameters"]

        # ── Variante parametric : courbes dans results["curves"] ─────────
        if curves is not None:
            self.alpha_xg = curves["alpha_xg"]   # (nb_ages, nb_regions)
            self.beta_x   = curves["beta_x"]     # (nb_ages,)
            self.beta_xg  = curves["beta_xg"]    # (nb_ages, nb_regions)
            self.kappa    = params["kappa"]       # (nb_years,)
            self.kappa_g  = params["kappa_g"]     # (nb_regions, nb_years)

        # ── Variante classic : tout dans results["parameters"] ──────────
        else:
            ax           = np.asarray(params["ax"])    # (nb_ages,) ou (nb_ages, 1) — alpha ne dépend que de x
            bx_gr        = np.asarray(params["bx_gr"]) # (nb_ages, nb_regions)
            nb_regions   = bx_gr.shape[1]
            # normalise ax en (nb_ages,) puis broadcast vers (nb_ages, nb_regions)
            ax_1d         = ax.ravel() if ax.ndim > 1 else ax   # (nb_ages,)
            self.alpha_xg = np.tile(ax_1d[:, None], (1, nb_regions))  # (nb_ages, nb_regions)
            self.beta_x   = np.asarray(params["bx"]).ravel()   # (nb_ages,)
            self.beta_xg  = bx_gr                               # (nb_ages, nb_regions)
            self.kappa    = params["kappa"]                     # (nb_years,)
            kappa_gr      = np.asarray(params["kappa_gr"])
            # kappa_gr peut être (nb_years, nb_regions) ou (nb_regions, nb_years) selon le modèle
            # on normalise en (nb_regions, nb_years) pour ProjectorLeeLi.project()
            self.kappa_g  = kappa_gr.T if kappa_gr.shape[0] == len(np.asarray(params["kappa"])) else kappa_gr
        self.tv            = np.asarray(tv)
        self.horizon       = horizon
        self.exclude_years = exclude_years if exclude_years is not None else [2020, 2021]
        self.nb_components = nb_components
        self.model         = model
        self.stochastic    = stochastic
        self.n_sim         = n_sim

        self.nb_ages, self.nb_regions = self.alpha_xg.shape

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_rw(series: np.ndarray):
        """
        Estimate random-walk parameters from a univariate or multivariate series.

        Computes the sample mean and covariance of the first differences:

        .. math::

            \\hat{\\boldsymbol{\\delta}} = \\overline{\\Delta Z}, \\qquad
            \\hat{\\Sigma} = \\operatorname{Cov}(\\Delta Z)

        where :math:`\\Delta Z_t = Z_t - Z_{t-1}`.

        Parameters
        ----------
        series : ndarray, shape ``(T,)`` or ``(T, nb_regions)``
            Observed time series.

        Returns
        -------
        drift : ndarray
            Sample mean of differences; shape ``(1,)`` for 1-D input or
            ``(nb_regions,)`` for 2-D input.
        cov : ndarray
            Covariance matrix of differences; shape ``(1, 1)`` or
            ``(nb_regions, nb_regions)``.
        last : ndarray
            Last observed value(s).
        """
        diffs = np.diff(series, axis=0)
        drift = np.mean(diffs, axis=0)
        if diffs.ndim == 1:
            cov   = np.array([[np.var(diffs, ddof=1)]])
            drift = np.array([drift])
        else:
            cov = np.cov(diffs, rowvar=False)
        last = series[-1] if series.ndim == 1 else series[-1, :]
        return drift, cov, last

    # ------------------------------------------------------------------
    def _svd_factors_g(self, kappa_gm: np.ndarray):
        """
        Build the reduced SVD factors for the regional kappa matrix.

        The masked regional kappa matrix :math:`K_g` of shape
        ``(T, nb_regions)`` is decomposed as:

        .. math::

            K_g \\approx Z_{\\text{red}} \\, V_{\\text{red}}^\\top

        retaining the first ``nb_components`` singular triplets:

        .. math::

            Z_{\\text{red}} = U_{[:,1:r]} \\cdot S_{[1:r]}, \\qquad
            V_{\\text{red}} = V^\\top_{[1:r,\\,:]}\\,{}^\\top

        Parameters
        ----------
        kappa_gm : ndarray, shape ``(T, nb_regions)``
            Regional kappa series with excluded years already removed.

        Returns
        -------
        Zred : ndarray, shape ``(T, nb_components)``
            Score matrix (time × components).
        Vred : ndarray, shape ``(nb_regions, nb_components)``
            Regional loading matrix.
        """
        U, S, Vt = np.linalg.svd(kappa_gm, full_matrices=False)
        Vred = Vt[:self.nb_components, :].T                        # (nb_regions, nb_components)
        Zred = U[:, :self.nb_components] * S[:self.nb_components]  # (T, nb_components)
        return Zred, Vred

    # ------------------------------------------------------------------
    def _project_rw(self, kappa_m, Zred, Vred):
        """
        Project common and regional factors using independent random walks with drift.

        Both the scalar common index :math:`\\kappa_t` and the reduced regional
        scores :math:`Z_t` follow independent random walks:

        .. math::

            \\kappa_{t+1} = \\kappa_t + \\hat{\\delta}_\\kappa + \\varepsilon_t^\\kappa,
            \\qquad \\varepsilon_t^\\kappa \\sim \\mathcal{N}(0,\\, \\hat{\\sigma}_\\kappa^2)

        .. math::

            Z_{t+1} = Z_t + \\hat{\\boldsymbol{\\delta}}_Z
                      + \\boldsymbol{\\varepsilon}_t^Z,
            \\qquad \\boldsymbol{\\varepsilon}_t^Z
            \\sim \\mathcal{N}(\\mathbf{0},\\, \\hat{\\Sigma}_Z)

        The projected regional kappa is then:

        .. math::

            \\kappa_t^{(g)\\text{future}} = Z_t^{\\text{future}} \\, V_{\\text{red}}^\\top

        Parameters
        ----------
        kappa_m : ndarray, shape ``(T,)``
            Observed common kappa (excluded years removed).
        Zred : ndarray, shape ``(T, nb_components)``
            Observed regional score matrix (excluded years removed).
        Vred : ndarray, shape ``(nb_regions, nb_components)``
            Regional loading matrix from :meth:`_svd_factors_g`.

        Returns
        -------
        k_future : ndarray
            Projected common kappa — ``(horizon,)`` or ``(horizon, n_sim)``.
        kg_future : ndarray
            Projected regional kappa — ``(horizon, nb_regions)`` or
            ``(horizon, n_sim, nb_regions)``.
        """
        drift_k,  cov_k,  k_last  = self._fit_rw(kappa_m)
        drift_z,  cov_z,  z_last  = self._fit_rw(Zred)

        t_range = np.arange(1, self.horizon + 1)

        if not self.stochastic:
            k_future  = k_last + drift_k[0] * t_range                   # (horizon,)
            Z_future  = z_last + drift_z * t_range[:, None]             # (horizon, nb_components)
            kg_future = Z_future @ Vred.T                                # (horizon, nb_regions)
        else:
            steps_k   = np.random.multivariate_normal(drift_k,  cov_k,  size=(self.horizon, self.n_sim))
            k_future  = k_last + np.cumsum(steps_k[:, :, 0], axis=0)   # (horizon, n_sim)
            steps_z   = np.random.multivariate_normal(drift_z, cov_z.reshape(self.nb_components, self.nb_components),   size=(self.horizon, self.n_sim))
            Z_future  = z_last + np.cumsum(steps_z, axis=0)             # (horizon, n_sim, nb_components)
            kg_future = Z_future @ Vred.T                                # (horizon, n_sim, nb_regions)

        return k_future, kg_future

    # ------------------------------------------------------------------
    def _project_linear(self, kappa_m, Zred, Vred):
        """
        Project common and regional factors using independent linear trend models.

        Both the common index and the regional scores are modelled as linear
        functions of time, estimated by OLS:

        .. math::

            \\kappa_t = \\alpha_\\kappa + \\beta_\\kappa \\, t + \\varepsilon_t^\\kappa

        .. math::

            Z_t = \\boldsymbol{\\alpha}_Z + \\boldsymbol{\\beta}_Z \\, t
                  + \\boldsymbol{\\varepsilon}_t^Z

        Deterministic projections use :math:`\\hat{\\alpha} + \\hat{\\beta}(T+h)`;
        stochastic paths add independent Gaussian residual noise drawn from the
        respective estimated residual covariances.

        The projected regional kappa is:

        .. math::

            \\kappa_t^{(g)\\text{future}} = Z_t^{\\text{future}} \\, V_{\\text{red}}^\\top

        Parameters
        ----------
        kappa_m : ndarray, shape ``(T,)``
            Observed common kappa (excluded years removed).
        Zred : ndarray, shape ``(T, nb_components)``
            Observed regional score matrix (excluded years removed).
        Vred : ndarray, shape ``(nb_regions, nb_components)``
            Regional loading matrix.

        Returns
        -------
        k_future : ndarray
            Projected common kappa — ``(horizon,)`` or ``(horizon, n_sim)``.
        kg_future : ndarray
            Projected regional kappa — ``(horizon, nb_regions)`` or
            ``(horizon, n_sim, nb_regions)``.
        """
        T        = kappa_m.shape[0]
        time_idx = np.arange(T)
        X_des    = np.column_stack([np.ones(T), time_idx])
        Xf       = np.column_stack([np.ones(self.horizon), np.arange(T, T + self.horizon)])

        beta_k = np.linalg.lstsq(X_des, kappa_m, rcond=None)[0]
        beta_z = np.linalg.lstsq(X_des, Zred,    rcond=None)[0]

        k_det  = Xf @ beta_k   # (horizon,)
        Z_det  = Xf @ beta_z   # (horizon, nb_components)

        if not self.stochastic:
            k_future  = k_det
            kg_future = Z_det @ Vred.T                                   # (horizon, nb_regions)
        else:
            res_k  = kappa_m - X_des @ beta_k
            res_z  = Zred    - X_des @ beta_z
            cov_k  = np.array([[np.var(res_k, ddof=1)]])
            cov_z  = np.cov(res_z, rowvar=False).reshape(self.nb_components, self.nb_components)

            noise_k  = np.random.multivariate_normal(np.zeros(1),              cov_k, size=(self.horizon, self.n_sim))
            noise_z  = np.random.multivariate_normal(np.zeros(self.nb_components), cov_z, size=(self.horizon, self.n_sim))

            k_future  = k_det[:, None] + noise_k[:, :, 0]               # (horizon, n_sim)
            Z_future  = Z_det[:, None, :] + noise_z                      # (horizon, n_sim, nb_components)
            kg_future = Z_future @ Vred.T                                 # (horizon, n_sim, nb_regions)

        return k_future, kg_future

    # ------------------------------------------------------------------
    def _reconstruct(self, k_future, kg_future):
        """
        Reconstruct log-mortality and mortality from projected common and regional factors.

        Applies the Lee-Li identity:

        .. math::

            \\ln \\mu_{x,t}^{(g)} = \\alpha_x^{(g)}
                + \\beta_x \\, \\kappa_t
                + \\beta_x^{(g)} \\, \\kappa_t^{(g)}

        and exponentiates to obtain :math:`\\mu_{x,t}^{(g)}`.

        In the stochastic case, empirical quantiles at 2.5 %, 50 %, and 97.5 %
        are computed over the simulation axis for both :math:`\\mu` and each
        kappa series.

        Parameters
        ----------
        k_future : ndarray
            Projected common kappa — ``(horizon,)`` or ``(horizon, n_sim)``.
        kg_future : ndarray
            Projected regional kappa — ``(horizon, nb_regions)`` or
            ``(horizon, n_sim, nb_regions)``.

        Returns
        -------
        dict
            Deterministic keys: ``kappa_future``, ``kappa_g_future``,
            ``logmu_future``, ``mu_future``.

            Stochastic keys: ``kappa_paths``, ``kappa_lower``,
            ``kappa_median``, ``kappa_upper``, ``kappa_g_paths``,
            ``kappa_g_lower``, ``kappa_g_median``, ``kappa_g_upper``,
            ``logmu_sim``, ``mu_sim``, ``mu_lower``, ``mu_median``,
            ``mu_upper``.
        """
        alpha_xg = self.alpha_xg
        beta_x   = self.beta_x
        beta_xg  = self.beta_xg

        if not self.stochastic:
            # output shape: (nb_ages, horizon, nb_regions)
            logmu_future = (
                alpha_xg[:, None, :]
                + beta_x[:,  None, None] * k_future[None, :, None]
                + beta_xg[:, None, :]    * kg_future[None, :, :]
            )
            mu_future = np.exp(logmu_future)
            return {
                "kappa_future":   k_future,
                "kappa_g_future": kg_future,
                "logmu_future":   logmu_future,
                "mu_future":      mu_future,
            }
        else:
            # broadcast to (nb_ages, horizon, nb_regions, n_sim)
            k_4d  = k_future[None, :, None, :]
            kg_4d = kg_future.transpose(0, 2, 1)[None, :, :, :]

            logmu_sim = (
                alpha_xg[:, None, :, None]
                + beta_x[:,  None, None, None] * k_4d
                + beta_xg[:, None, :,    None] * kg_4d
            )
            mu_sim = np.exp(logmu_sim)

            mu_lower  = np.percentile(mu_sim,  2.5, axis=-1)
            mu_median = np.percentile(mu_sim, 50,   axis=-1)
            mu_upper  = np.percentile(mu_sim, 97.5, axis=-1)

            k_lower  = np.percentile(k_future,  2.5, axis=1)
            k_median = np.percentile(k_future, 50,   axis=1)
            k_upper  = np.percentile(k_future, 97.5, axis=1)

            kg_lower  = np.percentile(kg_future,  2.5, axis=1)
            kg_median = np.percentile(kg_future, 50,   axis=1)
            kg_upper  = np.percentile(kg_future, 97.5, axis=1)

            return {
                "kappa_paths":    k_future,
                "kappa_lower":    k_lower,
                "kappa_median":   k_median,
                "kappa_upper":    k_upper,
                "kappa_g_paths":  kg_future,
                "kappa_g_lower":  kg_lower,
                "kappa_g_median": kg_median,
                "kappa_g_upper":  kg_upper,
                "logmu_sim":      logmu_sim,
                "mu_sim":         mu_sim,
                "mu_lower":       mu_lower,
                "mu_median":      mu_median,
                "mu_upper":       mu_upper,
            }

    # ------------------------------------------------------------------
    def project(self) -> dict:
        """
        Run the full Lee-Li projection and return all outputs.

        Masks excluded years, compresses regional kappas via SVD, projects
        both the common and regional factors with the selected time-series
        model, then calls :meth:`_reconstruct` to apply the Lee-Li identity
        and build mortality arrays.

        Returns
        -------
        dict
            Deterministic keys: ``kappa_future``, ``kappa_g_future``,
            ``logmu_future``, ``mu_future``.

            Stochastic keys: ``kappa_paths``, ``kappa_lower``,
            ``kappa_median``, ``kappa_upper``, ``kappa_g_paths``,
            ``kappa_g_lower``, ``kappa_g_median``, ``kappa_g_upper``,
            ``logmu_sim``, ``mu_sim``, ``mu_lower``, ``mu_median``,
            ``mu_upper``.
        """
        mask     = ~np.isin(self.tv, self.exclude_years)
        kappa_m  = self.kappa[mask]
        kappa_gm = self.kappa_g[:, mask].T   # (T, nb_regions)

        Zred, Vred = self._svd_factors_g(kappa_gm)

        if self.model == "rw":
            k_future, kg_future = self._project_rw(kappa_m, Zred, Vred)
        elif self.model == "linear":
            k_future, kg_future = self._project_linear(kappa_m, Zred, Vred)
        else:
            raise ValueError("model must be 'rw' or 'linear'")

        return self._reconstruct(k_future, kg_future)


# ─────────────────────────────────────────────────────────────────────────────
# 4. High-age extrapolation
# ─────────────────────────────────────────────────────────────────────────────

class HighAgeExtrapolator:
    """
    Extrapolates log-mortality rates beyond the maximum observed age up to *x_extrap*.

    Two extrapolation methods are available:

    **Linear (no-intercept regression)**

    A slope :math:`s` is estimated by anchored no-intercept OLS on ages
    :math:`[x_{\\text{start}}, x_{\\max}]`:

    .. math::

        \\log\\mu(x) - \\log\\mu(x_{\\max})
            \\approx s \\cdot (x - x_{\\max})

    and the extrapolation beyond :math:`x_{\\max}` is:

    .. math::

        \\log\\mu(x) = \\log\\mu(x_{\\max}) + s \\cdot (x - x_{\\max}),
        \\quad x > x_{\\max}

    **Kannisto (logistic Gompertz)**

    The force of mortality is modelled as:

    .. math::

        \\mu(x) = \\frac{a \\, e^{bx}}{1 + a \\, e^{bx}}

    Parameters :math:`(a, b)` are fitted by nonlinear least squares on ages
    :math:`[x_{\\text{start}}, x_{\\max}]` using the :func:`kannisto_log_mu`
    function.  If the fit fails, the method falls back to the linear approach.
    Monotonicity of log-mortality is enforced beyond :math:`x_{\\max}`.

    Parameters
    ----------
    xv : array-like
        Observed age vector.
    x_extrap : int
        Maximum age to extrapolate to.
    x_extrap_start : int or None
        Starting age of the regression window.  Set to ``None`` and pass
        ``auto_start=True`` to select automatically via leave-one-out CV.
    log_Muxtg : ndarray
        Log force-of-mortality array.  Expected shapes:

        - ``(nb_ages, nb_regions, horizon)`` — deterministic, linear method.
        - ``(nb_ages, nb_regions, horizon, n_sim)`` — stochastic, linear method.
        - ``(nb_ages, horizon, nb_regions)`` — deterministic, Kannisto method.
        - ``(nb_ages, horizon, nb_regions, n_sim)`` — stochastic, Kannisto method.
    method : {'kannisto', 'linear'}, optional
        Extrapolation method (default ``'kannisto'``).
    auto_start : bool, optional
        If ``True``, select *x_extrap_start* automatically (default ``False``).
    fallback_linear : bool, optional
        If ``True``, fall back to the linear method when Kannisto fails
        (default ``True``).
    """

    def __init__(
        self,
        xv,
        x_extrap: int,
        x_extrap_start,
        log_Muxtg: np.ndarray,
        method: str = "kannisto",
        auto_start: bool = False,
        fallback_linear: bool = True,
    ):
        self.xv              = np.asarray(xv)
        self.x_extrap        = x_extrap
        self.x_extrap_start  = x_extrap_start
        self.log_Muxtg       = log_Muxtg
        self.method          = method
        self.auto_start      = auto_start
        self.fallback_linear = fallback_linear
        self.stochastic      = log_Muxtg.ndim == 4

    # ------------------------------------------------------------------
    def _optimal_start(self, x_max: int, min_window: int = 5, max_window: int = 15) -> int:
        """
        Select the regression start age *x_extrap_start* via leave-one-out cross-validation.

        For each candidate window :math:`[x_{\\max} - w,\\, x_{\\max}]`
        (with :math:`w` ranging from *min_window* to *max_window*), a
        no-intercept slope is fitted by leaving out one interior point at a
        time.  The mean squared LOO prediction error is:

        .. math::

            \\text{MSE}(w) = \\frac{1}{|\\text{LOO}|}
            \\sum_{i \\in \\text{LOO}}
            \\left(
                \\hat{s}_{-i} \\cdot \\Delta x_i - \\Delta y_i
            \\right)^2

        where :math:`\\Delta x_i = x_i - x_{\\max}`,
        :math:`\\Delta y_i = \\log\\mu(x_i) - \\log\\mu(x_{\\max})`, and
        :math:`\\hat{s}_{-i}` is the slope estimated without point :math:`i`.

        The window minimising :math:`\\text{MSE}(w)` is selected.

        Parameters
        ----------
        x_max : int
            Maximum observed age.
        min_window : int, optional
            Minimum regression window size (default 5).
        max_window : int, optional
            Maximum regression window size (default 15).

        Returns
        -------
        int
            Optimal *x_extrap_start* age.
        """
        age_to_idx = {int(age): i for i, age in enumerate(self.xv)}
        idx_max    = age_to_idx[x_max]
        anchor     = self.log_Muxtg[idx_max, ...]

        best_start, best_mse = int(x_max) - min_window, np.inf

        for window in range(min_window, max_window + 1):
            start = int(x_max) - window
            if start < int(self.xv.min()):
                break

            xv_reg  = np.arange(start, x_max + 1)
            idx_reg = [age_to_idx[a] for a in xv_reg if a in age_to_idx]
            if len(idx_reg) < 3:
                continue

            dx = (self.xv[idx_reg] - x_max).astype(float)
            Y  = self.log_Muxtg[idx_reg, ...] - anchor

            errors = []
            for leave in range(1, len(idx_reg)):
                mask        = np.ones(len(idx_reg), dtype=bool)
                mask[leave] = False
                dx_train, dx_test = dx[mask], dx[leave]
                Y_train,  Y_test  = Y[mask], Y[leave]
                denom_loo = (dx_train ** 2).sum()
                if denom_loo == 0:
                    continue
                slope = np.einsum('i,i...->...', dx_train, Y_train) / denom_loo
                err   = np.mean((dx_test * slope - Y_test) ** 2)
                errors.append(err)

            if errors:
                mse = np.mean(errors)
                if mse < best_mse:
                    best_mse, best_start = mse, start

        return best_start

    # ------------------------------------------------------------------
    def _extrapolate_linear(self) -> tuple:
        """
        Extrapolate log-mortality beyond *x_max* using anchored no-intercept linear regression.

        For each region and horizon, the slope :math:`s` is estimated on the
        window :math:`[x_{\\text{start}}, x_{\\max}]` by solving:

        .. math::

            s = \\frac{
                \\sum_{x \\in W} (x - x_{\\max})
                \\bigl[\\log\\mu(x) - \\log\\mu(x_{\\max})\\bigr]
            }{
                \\sum_{x \\in W} (x - x_{\\max})^2
            }

        and the extrapolated value at age :math:`x > x_{\\max}` is:

        .. math::

            \\log\\mu(x) = \\log\\mu(x_{\\max}) + s \\cdot (x - x_{\\max})

        Array convention: ``(nb_ages, nb_regions, horizon[, n_sim])``.

        Returns
        -------
        out : ndarray
            Extended log-mortality array with shape
            ``(nb_ages_full, nb_regions, horizon[, n_sim])``.
        xv_full : ndarray
            Complete age vector from ``xv.min()`` to ``x_extrap``.
        """
        x_max   = int(self.xv.max())
        xv_reg  = np.arange(self.x_extrap_start, x_max + 1)
        xv_add  = np.arange(x_max + 1, self.x_extrap + 1)
        xv_full = np.arange(int(self.xv.min()), self.x_extrap + 1)

        dx_reg = (xv_reg - x_max).reshape(-1, 1).astype(float)
        dx_add = (xv_add - x_max).reshape(-1, 1).astype(float)

        mu = self.log_Muxtg
        if self.stochastic:
            nb_ages, nb_regions, horizon, n_sim = mu.shape
            out = np.empty((len(xv_full), nb_regions, horizon, n_sim))
            out[:nb_ages, ...] = mu
            for r in range(nb_regions):
                for h in range(horizon):
                    Y_reg  = mu[xv_reg, r, h, :]
                    anchor = mu[x_max,  r, h, :]
                    slope  = (dx_reg.T @ (Y_reg - anchor)) / (dx_reg.T @ dx_reg)
                    out[nb_ages:, r, h, :] = anchor + dx_add * slope
        else:
            nb_ages, nb_regions, horizon = mu.shape
            out = np.empty((len(xv_full), nb_regions, horizon))
            out[:nb_ages, ...] = mu
            for r in range(nb_regions):
                Y_reg  = mu[xv_reg, r, :]
                anchor = mu[x_max,  r, :]
                slope  = (dx_reg.T @ (Y_reg - anchor)) / (dx_reg.T @ dx_reg)
                out[nb_ages:, r, :] = anchor + dx_add * slope

        return out, xv_full

    # ------------------------------------------------------------------
    def _extrapolate_kannisto(self) -> tuple:
        """
        Extrapolate log-mortality beyond *x_max* using the Kannisto model.

        For each 1-D slice (one region × one horizon × one simulation), the
        Kannisto parameters :math:`(a, b)` are estimated by nonlinear least
        squares via :meth:`_fit_kannisto` on the window
        :math:`[x_{\\text{start}}, x_{\\max}]`.

        The extrapolated log-mortality at ages :math:`x > x_{\\max}` is:

        .. math::

            \\log\\mu(x) = \\log\\!\\left(
                \\frac{a \\, e^{bx}}{1 + a \\, e^{bx}}
            \\right)

        Monotonicity is enforced by:

        .. math::

            \\log\\mu(x) \\leftarrow
            \\max\\bigl(\\log\\mu(x),\\, \\log\\mu(x-1)\\bigr)

        If the Kannisto fit fails and *fallback_linear* is ``True``, the
        linear no-intercept method is used instead for that slice.

        Array convention: ``(nb_ages, horizon, nb_regions[, n_sim])``.

        Returns
        -------
        out : ndarray
            Extended log-mortality array with shape
            ``(nb_ages_full, horizon, nb_regions[, n_sim])``.
        xv_full : ndarray
            Complete age vector from ``xv.min()`` to ``x_extrap``.
        """
        x_max      = int(self.xv.max())
        idx_reg    = np.where((self.xv >= self.x_extrap_start) & (self.xv <= x_max))[0]
        idx_anchor = np.where(self.xv == x_max)[0][0]

        assert len(idx_reg) >= 2, (
            f"Regression window too small: "
            f"x_extrap_start={self.x_extrap_start}, x_max={x_max}"
        )

        xv_fit  = self.xv[idx_reg].astype(float)
        xv_add  = np.arange(x_max + 1, self.x_extrap + 1, dtype=float)
        xv_full = np.concatenate([self.xv, xv_add])
        nb_add  = len(xv_add)

        # precompute distances for the linear fallback
        dx_reg_   = (xv_fit - x_max)
        dx_add_   = (xv_add - x_max)
        denom_lin = float((dx_reg_ ** 2).sum())

        def _linear_1d(Y_reg_1d, anchor_val):
            """Linear fallback on a single 1D slice."""
            if denom_lin == 0:
                return np.full(nb_add, anchor_val)
            slope = max(float(np.dot(dx_reg_, Y_reg_1d - anchor_val) / denom_lin), 0.0)
            return anchor_val + dx_add_ * slope

        def _kannisto_1d(log_mu_1d):
            """Kannisto extrapolation on a single 1D slice."""
            fit_vals   = log_mu_1d[idx_reg]
            anchor_val = log_mu_1d[idx_anchor]

            params = self._fit_kannisto(xv_fit, fit_vals)
            if params is None:
                if self.fallback_linear:
                    return _linear_1d(fit_vals, anchor_val)
                raise RuntimeError("Kannisto fit failed and fallback is disabled.")

            a, b   = params
            extrap = kannisto_log_mu(xv_add, a, b)
            # enforce monotonicity: log-mortality must not decrease at high ages
            extrap = np.maximum.accumulate(np.concatenate([[anchor_val], extrap]))[1:]
            return extrap

        mu = self.log_Muxtg
        if self.stochastic:
            nb_ages, horizon, nb_regions, n_sim = mu.shape
            out = np.empty((len(xv_full), horizon, nb_regions, n_sim))
            out[:nb_ages, ...] = mu
            # flatten trailing dims to iterate over a single axis
            flat        = mu.reshape(nb_ages, -1)
            extrap_flat = np.empty((nb_add, flat.shape[1]))
            for col in range(flat.shape[1]):
                extrap_flat[:, col] = _kannisto_1d(flat[:, col])
            out[nb_ages:, ...] = extrap_flat.reshape(nb_add, horizon, nb_regions, n_sim)
            nan_mask = np.isnan(out).any(axis=(1, 2, 3))
        else:
            nb_ages, horizon, nb_regions = mu.shape
            out = np.empty((len(xv_full), horizon, nb_regions))
            out[:nb_ages, ...] = mu
            flat        = mu.reshape(nb_ages, -1)
            extrap_flat = np.empty((nb_add, flat.shape[1]))
            for col in range(flat.shape[1]):
                extrap_flat[:, col] = _kannisto_1d(flat[:, col])
            out[nb_ages:, ...] = extrap_flat.reshape(nb_add, horizon, nb_regions)
            nan_mask = np.isnan(out).any(axis=(1, 2))

        if nan_mask.any():
            print(f"⚠️ NaN at age indices: {np.where(nan_mask)[0]}")

        return out, xv_full

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_kannisto(xv_fit, log_mu_fit):
        """
        Fit the Kannisto model to a 1-D log-mortality slice.

        Estimates parameters :math:`(a, b)` of:

        .. math::

            \\log\\mu(x) = \\log\\!\\left(
                \\frac{a \\, e^{bx}}{1 + a \\, e^{bx}}
            \\right)

        by nonlinear least squares (``scipy.optimize.curve_fit``).

        Initial values are obtained from a Gompertz log-linear regression
        (which approximates the Kannisto model at moderate ages):

        .. math::

            \\log\\mu(x) \\approx \\log a + b \\, x
            \\quad \\Rightarrow \\quad
            a_0 = e^{\\hat{\\alpha}},\\; b_0 = \\hat{\\beta}

        Parameters
        ----------
        xv_fit : ndarray
            Age values in the regression window.
        log_mu_fit : ndarray
            Corresponding observed log-mortality values.

        Returns
        -------
        tuple or None
            ``(a, b)`` if the fit converged, ``None`` otherwise.
        """
        try:
            # initialize via Gompertz (log-linear regression)
            slope, intercept = np.polyfit(xv_fit, log_mu_fit, 1)
            b0 = max(slope, 1e-4)
            a0 = max(np.exp(intercept), 1e-6)
            popt, _ = curve_fit(
                lambda x, a, b: kannisto_log_mu(x, a, b),
                xv_fit, log_mu_fit,
                p0=[a0, b0],
                bounds=([1e-8, 1e-6], [1e2, 1.0]),
                maxfev=5000,
            )
            return popt
        except Exception:
            return None

    # ------------------------------------------------------------------
    def extrapolate(self) -> tuple:
        """
        Run the high-age extrapolation using the selected method.

        If ``auto_start=True`` or *x_extrap_start* is ``None``, the optimal
        regression start age is selected first via leave-one-out
        cross-validation (:meth:`_optimal_start`).

        Then dispatches to:

        - :meth:`_extrapolate_linear` for ``method='linear'``.
        - :meth:`_extrapolate_kannisto` for ``method='kannisto'``.

        Returns
        -------
        out : ndarray
            Extended log-mortality array covering ages up to *x_extrap*.
        xv_full : ndarray
            Complete age vector from ``xv.min()`` to ``x_extrap``.

        Raises
        ------
        ValueError
            If *method* is not ``'linear'`` or ``'kannisto'``.
        """
        x_max = int(self.xv.max())

        if self.auto_start or self.x_extrap_start is None:
            self.x_extrap_start = self._optimal_start(x_max)
            print(f"x_extrap_start automatically selected: {self.x_extrap_start}")

        if self.method == "linear":
            return self._extrapolate_linear()
        elif self.method == "kannisto":
            return self._extrapolate_kannisto()
        else:
            raise ValueError("method must be 'linear' or 'kannisto'")




#pour stochastique
def concat_logmu_time(logmu_hist, logmu_proj):
    """
    Concatenate historical and projected log-mortality arrays along the time axis.

    In the stochastic case, the 3-D historical array is broadcast to 4-D by
    repeating it *n_sim* times along a new simulation axis before concatenation,
    so that the output always has a consistent shape:

    .. math::

        \\log\\boldsymbol{\\mu}_{\\text{full}} =
        \\bigl[\\,\\log\\boldsymbol{\\mu}_{\\text{hist}}
             \\;\\big|\\;
             \\log\\boldsymbol{\\mu}_{\\text{proj}}\\,\\bigr]_{\\text{axis=1}}

    Parameters
    ----------
    logmu_hist : ndarray, shape ``(nb_ages, nb_years_hist, nb_regions)``
        Observed log-mortality array.
    logmu_proj : ndarray
        Projected log-mortality array.

        - Deterministic: shape ``(nb_ages, horizon, nb_regions)``.
        - Stochastic: shape ``(nb_ages, horizon, nb_regions, n_sim)``.

    Returns
    -------
    ndarray
        - Deterministic: shape ``(nb_ages, nb_years_hist + horizon, nb_regions)``.
        - Stochastic: shape ``(nb_ages, nb_years_hist + horizon, nb_regions, n_sim)``.

    Raises
    ------
    ValueError
        If *logmu_proj* is not 3-D or 4-D.
    """
    if logmu_proj.ndim == 3:
        return np.concatenate([logmu_hist, logmu_proj], axis=1)

    elif logmu_proj.ndim == 4:
        n_sim = logmu_proj.shape[3]
        logmu_hist_expanded = np.repeat(
            logmu_hist[:, :, :, None],   # (nb_ages, nb_years_hist, nb_regions, 1)
            n_sim,
            axis=3
        )
        return np.concatenate([logmu_hist_expanded, logmu_proj], axis=1)

    else:
        raise ValueError("logmu_proj must be 3D (deterministic) or 4D (stochastic)")
    


# #------------------------------------------------------------------------------
# # Function for valuation of annuities
# #------------------------------------------------------------------------------
# def Annuity_pricing(xe,xv,log_Muxtg,duration,rate):
#     _ , nby , nb_reg , nb_simul = log_Muxtg.shape
#     price   = np.zeros((len(xe),nb_reg,nb_simul))    
#     ctx      = 0
#     v = 1/(1+rate)
#     for xs in xe:
#         for i in range(nb_reg):
#             for s in range(nb_simul):
#                 tpx = 1
#                 tax = 0
#                 for j in range(duration):    
#                     #t_p_xs
#                     Muxtg = np.exp(log_Muxtg[xs+j,j,i,s])
#                     px    = np.exp(-Muxtg)  
#                     tpx   = tpx*px
#                     tax   = tax + tpx*v**j
#                     price[ctx,i,s]  = taxcd
#         ctx = ctx +1  
#     return(price)

def Annuity_pricing(xe, xv, log_Muxtg, duration, rate):
    """
    Compute the present value of life annuities using simulated mortality rates.

    This function evaluates the actuarial present value of a life annuity
    for different entry ages, regions, and stochastic mortality scenarios.
    The mortality intensity is provided in logarithmic form and transformed
    to obtain the force of mortality.

    The annuity is valued by computing survival probabilities year by year
    and discounting future payments using a constant interest rate.

    Parameters
    ----------
    xe : array-like
        Vector of entry ages at which the annuity starts.
    xv : array-like
        Vector of ages in the mortality table (not directly used in the function
        but typically included for compatibility with the mortality grid).
    log_Muxtg : numpy.ndarray
        4D array containing the logarithm of mortality intensities with shape:
        (age, projection_year, region, simulation).

        Dimensions:
        - age: index corresponding to the age in the mortality table
        - projection_year: future year index
        - region: geographical region
        - simulation: stochastic mortality scenario
    duration : int
        Maximum duration (in years) of the annuity payments.
    rate : float
        Constant annual interest rate used for discounting.

    Returns
    -------
    numpy.ndarray
        Array containing the annuity prices with shape:
        (len(xe), number_of_regions, number_of_simulations)

        Each value corresponds to the actuarial present value of a life annuity
        for a given entry age, region, and stochastic mortality scenario.

    Notes
    -----
    The survival probability is approximated using the force of mortality:

    .. math::

        p_x = \\exp(-\\mu_x)

    where :math:`\\mu_x` is the force of mortality.

    The annuity value is computed as:

    .. math::

        a_x = \\sum_{t=1}^{T} v^t \\cdot {}_t p_x

    where:

    - :math:`v = \\frac{1}{1+i}` is the discount factor
    - :math:`{}_t p_x` is the survival probability from age :math:`x` to :math:`x+t`
    - :math:`T` is the annuity duration

    The function evaluates the annuity value for each:
    - entry age
    - region
    - stochastic mortality simulation.

    """

    # Extract dimensions from the mortality array
    # nby: number of projection years
    # nb_reg: number of regions
    # nb_simul: number of stochastic simulations
    _, nby, nb_reg, nb_simul = log_Muxtg.shape

    # Initialize output array storing annuity prices
    # Dimensions: entry_age × region × simulation
    price = np.zeros((len(xe), nb_reg, nb_simul))

    # Index for entry age
    ctx = 0

    # Discount factor
    v = 1 / (1 + rate)

    # Loop over entry ages
    for xs in xe:

        # Loop over regions
        for i in range(nb_reg):

            # Loop over stochastic mortality simulations
            for s in range(nb_simul):

                # Survival probability accumulator
                tpx = 1

                # Present value of annuity payments
                tax = 0

                # Loop over annuity duration
                for j in range(duration):

                    # Retrieve mortality intensity for age xs+j
                    # and projection year j
                    Muxtg = np.exp(log_Muxtg[xs + j, j, i, s])

                    # Convert force of mortality into survival probability
                    px = np.exp(-Muxtg)

                    # Update cumulative survival probability
                    tpx = tpx * px

                    # Add discounted payment conditional on survival
                    tax = tax + tpx * v**j

                    # Store intermediate annuity value
                    price[ctx, i, s] = tax

        # Move to next entry age
        ctx = ctx + 1

    return price


################################################################
#                 COMPUTE MAE AND WMAE
################################################################

def compute_mae(mu_obs, mu_model, weights=None):
    """
    Compute the Mean Absolute Error (MAE) and Weighted MAE (WMAE) between
    observed and modelled mortality rates.

    The global MAE is:

    .. math::

        \\text{MAE} = \\frac{1}{A \\cdot T \\cdot G}
        \\sum_{x,t,g} \\bigl| \\hat{\\mu}_{x,t}^{(g)} - \\mu_{x,t}^{(g)} \\bigr|

    When exposure weights :math:`E_{x,t}^{(g)}` are provided, the
    weighted MAE is:

    .. math::

        \\text{WMAE} = \\frac{
            \\sum_{x,t,g}
            E_{x,t}^{(g)} \\,
            \\bigl| \\hat{\\mu}_{x,t}^{(g)} - \\mu_{x,t}^{(g)} \\bigr|
        }{
            \\sum_{x,t,g} E_{x,t}^{(g)}
        }

    Both metrics are also computed marginally over each dimension
    (by region, by age, by year).

    Parameters
    ----------
    mu_obs : ndarray, shape ``(nb_ages, horizon)`` or ``(nb_ages, horizon, nb_regions)``
        Observed mortality rates.
    mu_model : ndarray
        Modelled mortality rates; must be broadcastable to the shape of *mu_obs*.
    weights : ndarray or None, optional
        Non-negative weights (e.g. exposures :math:`E_{x,t}^{(g)}`), same
        shape as *mu_obs*.  If ``None``, only unweighted MAE is computed.

    Returns
    -------
    dict
        Always contains:

        - ``"by_region"`` — MAE averaged over ages and years, shape ``(nb_regions,)``.
        - ``"by_age"``    — MAE averaged over years and regions, shape ``(nb_ages,)``.
        - ``"by_year"``   — MAE averaged over ages and regions, shape ``(horizon,)``.
        - ``"global"``    — scalar global MAE.

        If *weights* is provided, also contains:

        - ``"wmae_by_region"``, ``"wmae_by_age"``, ``"wmae_by_year"``,
          ``"wmae_global"``.
    """
    #  2D  →  broadcast to 3D
    if mu_obs.ndim == 2:
        mu_obs = mu_obs[:, :, None]
    if mu_model.ndim == 2:
        mu_model = mu_model[:, :, None]
    if weights is not None and weights.ndim == 2:
        weights = weights[:, :, None]

    
    if mu_model.shape != mu_obs.shape:
        mu_model = np.broadcast_to(mu_model, mu_obs.shape)

    abs_err = np.abs(mu_model - mu_obs)

    result = {
        "by_region": np.mean(abs_err, axis=(0, 1)),   # (nb_regions,)
        "by_age":    np.mean(abs_err, axis=(1, 2)),   # (nb_ages,)
        "by_year":   np.mean(abs_err, axis=(0, 2)),   # (horizon,)
        "global":    np.sum(abs_err) / abs_err.size,  # scalar
    }

    if weights is not None:
        result.update({
            "wmae_by_region": np.sum(abs_err * weights, axis=(0, 1)) / np.sum(weights, axis=(0, 1)),
            "wmae_by_age":    np.sum(abs_err * weights, axis=(1, 2)) / np.sum(weights, axis=(1, 2)),
            "wmae_by_year":   np.sum(abs_err * weights, axis=(0, 2)) / np.sum(weights, axis=(0, 2)),
            "wmae_global":    np.sum(abs_err * weights) / np.sum(weights),
        })

    return result

