import numpy as np
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def kannisto_log_mu(x, a, b):
    """log(μ(x)) for the Kannisto model: μ(x) = a·exp(b·x) / (1 + a·exp(b·x))"""
    ebx = np.exp(b * x)
    return np.log(a * ebx / (1.0 + a * ebx))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Life expectancy
# ─────────────────────────────────────────────────────────────────────────────

class LifeExpectancy:
    """
    Computes life expectancy from mortality rates.

    Parameters
    ----------
    mu_future : ndarray
        - (ages, horizon, regions)         -> deterministic case
        - (ages, horizon, regions, n_sim)  -> stochastic case
    """

    def __init__(self, mu_future: np.ndarray):
        if mu_future.ndim not in (3, 4):
            raise ValueError("mu_future must be 3D or 4D")
        self.mu_future = mu_future

    # ------------------------------------------------------------------
    def _compute(self, mu: np.ndarray) -> np.ndarray:
        """Vectorized life expectancy computation (works for both 3D and 4D)."""
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
        """Returns ex with the same shape as mu_future."""
        return self._compute(self.mu_future)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lee-Carter SVD projection
# ─────────────────────────────────────────────────────────────────────────────

class ProjectorLC_SVD:
    """
    Multi-regional Lee-Carter prospective projection via SVD.

    Parameters
    ----------
    results : dict
        Must contain results["curves"]["alpha_x"], results["curves"]["beta_xg"],
        and results["parameters"]["kappa"].
    tv : array-like
        Observation year vector.
    horizon : int
    exclude_years : list
    nb_components : int
    model : {'rw', 'linear'}
    stochastic : bool
    n_sim : int
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
        self.ax = results["curves"]["alpha_x"]   # (nb_ages,)
        curves = results["curves"]

        beta_key = next((k for k in curves.keys() if k.startswith("beta")), None)

        if beta_key is None:
            raise KeyError("No beta curve found in results['curves']")

        self.bx = curves[beta_key]  # stocké comme beta_xg dans l'objet
        self.kappa_raw     = results["parameters"]["kappa"]
        self.tv            = np.asarray(tv)
        self.horizon       = horizon
        self.exclude_years = exclude_years if exclude_years is not None else [2020, 2021]
        self.nb_components = nb_components
        self.model         = model
        self.stochastic    = stochastic
        self.n_sim         = n_sim

    # ------------------------------------------------------------------
    def _svd_factors(self):
        """Builds reduced Z and V matrices via SVD after masking excluded years."""
        X_svd = self.kappa_raw.reshape(-1, 1) if self.kappa_raw.ndim == 1 else self.kappa_raw.T
        X_svd = X_svd[~np.isin(self.tv, self.exclude_years), :]

        U, S, Vt = np.linalg.svd(X_svd, full_matrices=False)
        Vred = Vt[:self.nb_components, :].T                        # (nb_regions, nb_components)
        Zred = U[:, :self.nb_components] * S[:self.nb_components]  # (T, nb_components)
        return Zred, Vred

    # ------------------------------------------------------------------
    def _project_rw(self, Zred, Vred):
        """Random walk projection."""
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
        """Linear trend projection."""
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
        """Reconstructs log(μ) and μ from projected kappa."""
        ax, bx = self.ax, self.bx

        if not self.stochastic:
            logmu_future = (
                ax[:, None, None]
                + bx[:, None, :] * kappa_future[None, :, :]
            )
            mu_future = np.exp(logmu_future)
            return {
                "kappa_future": kappa_future,
                "logmu_future": logmu_future,
                "mu_future":    mu_future,
            }
        else:
            # reshape kappa to (1, horizon, nb_regions, n_sim)
            kf_4d = kappa_future.transpose(0, 2, 1)[None, :, :, :]
            logmu_sim = (
                ax[:, None, None, None]
                + bx[:, None, :, None] * kf_4d
            )
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
    def project(self) -> dict:
        """Runs the projection and returns the results dict."""
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

class ProjectorLeeLi:
    """
    Lee-Li prospective projection (common factor + regional components).

    Parameters
    ----------
    results : dict
        Must contain results["curves"] (alpha_xg, beta_x, beta_xg)
        and results["parameters"] (kappa, kappa_g).
    tv : array-like
    horizon : int
    exclude_years : list
    model : {'rw', 'linear'}
    stochastic : bool
    n_sim : int
    """

    def __init__(
        self,
        results: dict,
        tv,
        horizon: int = 30,
        exclude_years: list = None,
        model: str = "rw",
        stochastic: bool = True,
        n_sim: int = 1000,
    ):
        curves = results["curves"]
        params = results["parameters"]

        self.alpha_xg      = curves["alpha_xg"]   # (nb_ages, nb_regions)
        self.beta_x        = curves["beta_x"]     # (nb_ages,)
        self.beta_xg       = curves["beta_xg"]    # (nb_ages, nb_regions)
        self.kappa         = params["kappa"]      # (nb_years,)
        self.kappa_g       = params["kappa_g"]    # (nb_regions, nb_years)
        self.tv            = np.asarray(tv)
        self.horizon       = horizon
        self.exclude_years = exclude_years if exclude_years is not None else [2020, 2021]
        self.model         = model
        self.stochastic    = stochastic
        self.n_sim         = n_sim

        self.nb_ages, self.nb_regions = self.alpha_xg.shape

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_rw(series: np.ndarray):
        """Fits a random walk on series (T,) or (T, nb_regions).
        Returns (drift, covariance, last_value)."""
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
    def _project_rw(self, kappa_m, kappa_gm):
        """Random walk projection for common and regional factors."""
        drift_k,  cov_k,  k_last  = self._fit_rw(kappa_m)
        drift_kg, cov_kg, kg_last = self._fit_rw(kappa_gm.T)

        t_range = np.arange(1, self.horizon + 1)

        if not self.stochastic:
            k_future  = k_last + drift_k[0] * t_range                  # (horizon,)
            kg_future = kg_last + drift_kg * t_range[:, None]          # (horizon, nb_regions)
        else:
            steps_k   = np.random.multivariate_normal(drift_k,  cov_k,  size=(self.horizon, self.n_sim))
            k_future  = k_last + np.cumsum(steps_k[:, :, 0], axis=0)  # (horizon, n_sim)
            steps_kg  = np.random.multivariate_normal(drift_kg, cov_kg, size=(self.horizon, self.n_sim))
            kg_future = kg_last + np.cumsum(steps_kg, axis=0)          # (horizon, n_sim, nb_regions)

        return k_future, kg_future

    # ------------------------------------------------------------------
    def _project_linear(self, kappa_m, kappa_gm):
        """Linear trend projection for common and regional factors."""
        T        = kappa_m.shape[0]
        time_idx = np.arange(T)
        X_des    = np.column_stack([np.ones(T), time_idx])
        Xf       = np.column_stack([np.ones(self.horizon), np.arange(T, T + self.horizon)])

        beta_k  = np.linalg.lstsq(X_des, kappa_m,    rcond=None)[0]
        beta_kg = np.linalg.lstsq(X_des, kappa_gm.T, rcond=None)[0]

        k_det  = Xf @ beta_k    # (horizon,)
        kg_det = Xf @ beta_kg   # (horizon, nb_regions)

        if not self.stochastic:
            k_future  = k_det
            kg_future = kg_det
        else:
            res_k  = kappa_m    - X_des @ beta_k
            res_kg = kappa_gm.T - X_des @ beta_kg
            cov_k  = np.array([[np.var(res_k, ddof=1)]])
            cov_kg = np.cov(res_kg, rowvar=False)

            noise_k   = np.random.multivariate_normal(np.zeros(1),              cov_k,  size=(self.horizon, self.n_sim))
            noise_kg  = np.random.multivariate_normal(np.zeros(self.nb_regions), cov_kg, size=(self.horizon, self.n_sim))

            k_future  = k_det[:, None] + noise_k[:, :, 0]   # (horizon, n_sim)
            kg_future = kg_det[:, None, :] + noise_kg        # (horizon, n_sim, nb_regions)

        return k_future, kg_future

    # ------------------------------------------------------------------
    def _reconstruct(self, k_future, kg_future):
        """Reconstructs log(μ) and μ from projected common and regional factors."""
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
        """Runs the projection and returns the results dict."""
        mask     = ~np.isin(self.tv, self.exclude_years)
        kappa_m  = self.kappa[mask]
        kappa_gm = self.kappa_g[:, mask]

        if self.model == "rw":
            k_future, kg_future = self._project_rw(kappa_m, kappa_gm)
        elif self.model == "linear":
            k_future, kg_future = self._project_linear(kappa_m, kappa_gm)
        else:
            raise ValueError("model must be 'rw' or 'linear'")

        return self._reconstruct(k_future, kg_future)


# ─────────────────────────────────────────────────────────────────────────────
# 4. High-age extrapolation
# ─────────────────────────────────────────────────────────────────────────────

class HighAgeExtrapolator:
    """
    Extrapolates log_Muxtg beyond max(xv) up to x_extrap.

    Two methods available:
      - 'linear'   : no-intercept linear regression (base method)
      - 'kannisto' : Kannisto model with linear fallback

    Expected shape of log_Muxtg:
      - (nb_ages, nb_regions, horizon)         -> deterministic  [linear]
      - (nb_ages, nb_regions, horizon, n_sim)  -> stochastic     [linear]
      - (nb_ages, horizon, nb_regions)         -> deterministic  [kannisto]
      - (nb_ages, horizon, nb_regions, n_sim)  -> stochastic     [kannisto]
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
        """Automatically selects x_extrap_start via leave-one-out cross-validation."""
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
        """No-intercept linear regression (convention: nb_ages, nb_regions, horizon[, n_sim])."""
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
        """Kannisto extrapolation with linear fallback (convention: nb_ages, horizon, nb_regions[, n_sim])."""
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
        """Fits Kannisto model on a 1D vector. Returns (a, b) or None on failure."""
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
        Runs the extrapolation using the selected method.

        Returns (out, xv_full).
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
    Concatenate historical and projected log-mortality along time axis.

    Parameters
    ----------
    logmu_hist : (nb_ages, nb_years_hist, nb_regions)
    logmu_proj :
        - deterministic : (nb_ages, horizon, nb_regions)
        - stochastic    : (nb_ages, horizon, nb_regions, n_sim)

    Returns
    -------
    - deterministic : (nb_ages, nb_years_hist + horizon, nb_regions)
    - stochastic    : (nb_ages, nb_years_hist + horizon, nb_regions, n_sim)
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
    


#------------------------------------------------------------------------------
# Function for valuation of annuities
#------------------------------------------------------------------------------
def Annuity_pricing(xe,xv,log_Muxtg,duration,rate):
    _ , nby , nb_reg , nb_simul = log_Muxtg.shape
    price   = np.zeros((len(xe),nb_reg,nb_simul))    
    ctx      = 0
    v = 1/(1+rate)
    for xs in xe:
        for i in range(nb_reg):
            for s in range(nb_simul):
                tpx = 1
                tax = 0
                for j in range(duration):    
                    #t_p_xs
                    Muxtg = np.exp(log_Muxtg[xs+j,j,i,s])
                    px    = np.exp(-Muxtg)  
                    tpx   = tpx*px
                    tax   = tax + tpx*v**j
                    price[ctx,i,s]  = tax
        ctx = ctx +1  
    return(price)