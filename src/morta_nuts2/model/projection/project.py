import numpy as np

def compute_life_expectancy_all(mu_future):
    """
    mu_future : (ages, horizon, regions)
    retourne  : (ages, horizon, regions)
    """
    ages, horizon, regions = mu_future.shape
    ex = np.zeros((ages, horizon, regions))

    for r in range(regions):
        for t in range(horizon):

            mu = mu_future[:, t, r]          # ✅ axis=1 → horizon, axis=2 → regions
            qx = np.clip(mu / (1 + 0.5 * mu), 0, 1)

            lx      = np.zeros(ages)
            lx[0]   = 100000
            lx[1:]  = lx[0] * np.cumprod(1 - qx[:-1])

            Lx      = np.zeros(ages)
            Lx[:-1] = 0.5 * (lx[:-1] + lx[1:])
            Lx[-1]  = lx[-1]

            Tx         = np.flip(np.cumsum(np.flip(Lx)))
            ex[:, t, r] = Tx / lx             # ✅ cohérent avec (ages, horizon, regions)

    return ex


def compute_life_expectancy_sim(mu_sim):
    """
    mu_sim : (ages, regions, horizon, n_sim)
    
    retourne :
    ex_sim : (ages, regions, horizon, n_sim)
    """
    
    ages, regions, horizon, n_sim = mu_sim.shape
    ex_sim = np.zeros((ages, regions, horizon, n_sim))
    
    for s in range(n_sim):
        for r in range(regions):
            for t in range(horizon):
                
                mu = mu_sim[:, r, t, s]
                qx = mu / (1 + 0.5 * mu)
                qx = np.clip(qx, 0, 1)
                
                lx = np.zeros(ages)
                lx[0] = 100000
                lx[1:] = lx[0] * np.cumprod(1 - qx[:-1])
                
                Lx = np.zeros(ages)
                Lx[:-1] = 0.5 * (lx[:-1] + lx[1:])
                Lx[-1] = lx[-1]
                
                Tx = np.flip(np.cumsum(np.flip(Lx)))
                
                ex_sim[:, r, t, s] = Tx / lx
    
    return ex_sim

def compute_life_expectancy(mu):
    """
    Computes life expectancy table for all ages.

    Handles both cases automatically:
      - Deterministic : mu shape (ages, horizon, regions)
      - Stochastic    : mu shape (ages, regions, horizon, n_sim)

    Returns ex with the same shape as input mu.
    """

    # =====================================================
    # Detect case and reshape to (ages, horizon, regions, [n_sim])
    # =====================================================
    if mu.ndim == 3:
        # Deterministic : (ages, horizon, regions)
        ages, horizon, regions = mu.shape
        n_sim      = None
        mu_work    = mu                              # (ages, horizon, regions)

    elif mu.ndim == 4:
        # Stochastic : (ages, regions, horizon, n_sim)
        ages, regions, horizon, n_sim = mu.shape
        mu_work = mu.transpose(0, 2, 1, 3)          # → (ages, horizon, regions, n_sim)

    else:
        raise ValueError(f"mu must be 3D or 4D, got {mu.ndim}D")

    # =====================================================
    # Core life table computation — vectorized over last axes
    # =====================================================
    def life_table(mu_slice):
        """
        mu_slice : (ages, ...) — any trailing dimensions
        Returns ex of same shape.
        """
        orig_shape = mu_slice.shape
        ages_      = orig_shape[0]
        rest       = int(np.prod(orig_shape[1:]))      # flatten trailing dims

        mu_2d  = mu_slice.reshape(ages_, rest)          # (ages, rest)
        qx     = np.clip(mu_2d / (1 + 0.5 * mu_2d), 0, 1)

        # lx : survival
        lx        = np.zeros_like(qx)
        lx[0, :]  = 100_000
        lx[1:, :] = 100_000 * np.cumprod(1 - qx[:-1, :], axis=0)

        # Lx : person-years lived
        Lx        = np.zeros_like(lx)
        Lx[:-1,:] = 0.5 * (lx[:-1, :] + lx[1:, :])
        Lx[-1, :] = lx[-1, :]

        # Tx : cumulative from above
        Tx = np.flip(np.cumsum(np.flip(Lx, axis=0), axis=0), axis=0)

        ex_2d = Tx / lx
        return ex_2d.reshape(orig_shape)

    # =====================================================
    # Apply and restore original shape convention
    # =====================================================
    if n_sim is None:
        # (ages, horizon, regions)
        ex = life_table(mu_work)                       # (ages, horizon, regions)
        return ex

    else:
        # (ages, horizon, regions, n_sim)
        ex_work = life_table(mu_work)                  # (ages, horizon, regions, n_sim)
        # Restore to (ages, regions, horizon, n_sim)
        return ex_work.transpose(0, 2, 1, 3)


# def compute_life_expectancy_all(mu_future):
#     """
#     mu_future : (ages, regions, horizon)
#     retourne e0 : (regions, horizon)
#     """
    
#     ages, regions, horizon = mu_future.shape
#     e0 = np.zeros((regions, horizon))
    
#     for r in range(regions):
#         for t in range(horizon):
            
#             mu = mu_future[:, r, t]
#             qx = mu / (1 + 0.5 * mu)
            
#             lx = np.zeros(len(qx))
#             lx[0] = 100000
            
#             lx[1:] = lx[0] * np.cumprod(1 - qx[:-1])
            
#             Lx = 0.5 * (lx[:-1] + lx[1:])
#             e0[r, t] = np.sum(Lx) / lx[0]
    
#     return e0


# def compute_life_expectancy_sim(mu_sim):
#     """
#     mu_sim : (ages, regions, horizon, n_sim)
#     retourne e0_sim : (regions, horizon, n_sim)
#     """
    
#     ages, regions, horizon, n_sim = mu_sim.shape
#     e0_sim = np.zeros((regions, horizon, n_sim))
    
#     for s in range(n_sim):
#         for r in range(regions):
#             for t in range(horizon):
                
#                 mu = mu_sim[:, r, t, s]
                
#                 qx = mu / (1 + 0.5 * mu)
#                 qx = np.clip(qx, 0, 1)
                
#                 lx = np.zeros(len(qx))
#                 lx[0] = 100000
                
#                 lx[1:] = lx[0] * np.cumprod(1 - qx[:-1])
                
#                 Lx = 0.5 * (lx[:-1] + lx[1:])
#                 e0_sim[r, t, s] = np.sum(Lx) / lx[0]
    
#     return e0_sim







# def project_LC_prospective_SVD(
#     results,
#     tv,
#     horizon=30,
#     exclude_years=[2020, 2021],
#     nb_components=1,
#     model="rw",
#     stochastic=True,
#     n_sim=1000
# ):
    
#     ax = results["curves"]["alpha_x"]        
#     bx = results["curves"]["beta_xg"]        
#     kappa = results["parameters"]["kappa"]   
    
#     # =====================================================
#     # 1️⃣ Préparer matrice κ
#     # =====================================================
    
#     if kappa.ndim == 1:
#         X_svd = kappa.reshape(-1, 1)
#     else:
#         X_svd = kappa.T
    
#     mask = ~np.isin(tv, exclude_years)
#     X_svd = X_svd[mask, :]
    
#     # =====================================================
#     # 2️⃣ SVD
#     # =====================================================
    
#     U, S, Vt = np.linalg.svd(X_svd, full_matrices=False)
    
#     Ured = U[:, :nb_components]
#     Sred = np.diag(S[:nb_components])
#     Vred = Vt[:nb_components, :].T
    
#     Zred = Ured @ Sred
    
#     T = Zred.shape[0]
#     nb_regions = Vred.shape[0]
    
#     # =====================================================
#     # 3️⃣ Dynamique des facteurs
#     # =====================================================
    
#     if model == "rw":
        
#         diffs = np.diff(Zred, axis=0)
        
#         if nb_components == 1:
#             drift = np.array([np.mean(diffs)])
#             sigma = np.std(diffs)
#             cov = np.array([[sigma**2]])
#         else:
#             drift = np.mean(diffs, axis=0)
#             cov = np.cov(diffs, rowvar=False)
        
#         Z_last = Zred[-1, :]
        
#         if not stochastic:
#             Z_future = np.zeros((horizon, nb_components))
#             last_value = Z_last.copy()
            
#             for t in range(horizon):
#                 last_value = last_value + drift
#                 Z_future[t, :] = last_value
            
#             kappa_future = Z_future @ Vred.T
        
#         else:
#             Z_future = np.zeros((horizon, nb_components, n_sim))
#             kappa_future = np.zeros((horizon, nb_regions, n_sim))
            
#             for i in range(n_sim):
#                 last_value = Z_last.copy()
                
#                 for t in range(horizon):
#                     step = np.random.multivariate_normal(drift, cov)
#                     last_value = last_value + step
#                     Z_future[t, :, i] = last_value
#                     kappa_future[t, :, i] = last_value @ Vred.T
    
#     elif model == "linear":
        
#         time_index = np.arange(T)
#         X = np.vstack([np.ones(T), time_index]).T
        
#         beta_lin = np.linalg.lstsq(X, Zred, rcond=None)[0]
        
#         future_index = np.arange(T, T + horizon)
#         Xf = np.vstack([np.ones(horizon), future_index]).T
        
#         if not stochastic:
#             Z_future = Xf @ beta_lin
#             kappa_future = Z_future @ Vred.T
        
#         else:
#             Z_fit = X @ beta_lin
#             residuals = Zred - Z_fit
            
#             if nb_components == 1:
#                 sigma = np.std(residuals)
#                 cov_res = np.array([[sigma**2]])
#             else:
#                 cov_res = np.cov(residuals.T)
            
#             Z_future = np.zeros((horizon, nb_components, n_sim))
#             kappa_future = np.zeros((horizon, nb_regions, n_sim))
            
#             for i in range(n_sim):
#                 noise = np.random.multivariate_normal(
#                     mean=np.zeros(nb_components),
#                     cov=cov_res,
#                     size=horizon
#                 )
                
#                 Z_sim = Xf @ beta_lin + noise
#                 Z_future[:, :, i] = Z_sim
#                 kappa_future[:, :, i] = Z_sim @ Vred.T
    
#     else:
#         raise ValueError("model must be 'rw' or 'linear'")
    
#     # =====================================================
#     # 4️⃣ Reconstruction des taux
#     # =====================================================
    
#     if not stochastic:
        
#         logmu_future = (
#             ax[:, None, None] +
#             bx[:, :, None] *
#             kappa_future.T[None, :, :]
#         )
        
#         mu_future = np.exp(logmu_future)
#         e0 = compute_life_expectancy_all(mu_future)
        
#         return {
#             "kappa_future": kappa_future,
#             "mu_future": mu_future,
#             "life_expectancy": e0
#         }
    
#     else:
        
#         logmu_sim = (
#             ax[:, None, None, None] +
#             bx[:, :, None, None] *
#             kappa_future.transpose(1, 0, 2)[None, :, :, :]
#         )
        
#         mu_sim = np.exp(logmu_sim)
        
#         kappa_lower = np.percentile(kappa_future, 5, axis=2)
#         kappa_median = np.percentile(kappa_future, 50, axis=2)
#         kappa_upper = np.percentile(kappa_future, 95, axis=2)
        
#         mu_lower = np.percentile(mu_sim, 5, axis=3)
#         mu_median = np.percentile(mu_sim, 50, axis=3)
#         mu_upper = np.percentile(mu_sim, 95, axis=3)
        
#         e0_sim = compute_life_expectancy_sim(mu_sim)
        
#         e0_lower = np.percentile(e0_sim, 5, axis=1)
#         e0_median = np.percentile(e0_sim, 50, axis=1)
#         e0_upper = np.percentile(e0_sim, 95, axis=1)
        
#         return {
#             "kappa_paths": kappa_future,
#             "kappa_lower": kappa_lower,
#             "kappa_median": kappa_median,
#             "kappa_upper": kappa_upper,
#             "mu_sim": mu_sim,
#             "mu_lower": mu_lower,
#             "mu_median": mu_median,
#             "mu_upper": mu_upper,
#             "life_expectancy_median": e0_median,
#             "life_expectancy_lower": e0_lower,
#             "life_expectancy_upper": e0_upper
#         }


def project_LC_prospective_SVD(
    results,
    tv,
    horizon=30,
    exclude_years=[2020, 2021],
    nb_components=1,
    model="rw",
    stochastic=True,
    n_sim=1000,
):
    ax    = results["curves"]["alpha_x"]
    bx    = results["curves"]["beta_xg"]
    kappa = results["parameters"]["kappa"]

    # =====================================================
    # 1️⃣ Préparer matrice κ
    # =====================================================
    X_svd = kappa.reshape(-1, 1) if kappa.ndim == 1 else kappa.T
    X_svd = X_svd[~np.isin(tv, exclude_years), :]

    # =====================================================
    # 2️⃣ SVD
    # =====================================================
    U, S, Vt = np.linalg.svd(X_svd, full_matrices=False)
    Vred = Vt[:nb_components, :].T                    # (nb_regions, nb_components)
    Zred = U[:, :nb_components] * S[:nb_components]   # (T, nb_components)
    T, nb_regions = Zred.shape[0], Vred.shape[0]

    # =====================================================
    # 3️⃣ Dynamique des facteurs
    # =====================================================
    if model == "rw":
        diffs = np.diff(Zred, axis=0)
        drift = np.mean(diffs, axis=0)                # (nb_components,)
        cov   = np.cov(diffs, rowvar=False).reshape(nb_components, nb_components)
        Z_last = Zred[-1, :]

        if not stochastic:
            # Déterministe : cumsum du drift
            Z_future     = Z_last + drift * np.arange(1, horizon + 1)[:, None]  # (horizon, nb_components)
            kappa_future = Z_future @ Vred.T                                     # (horizon, nb_regions)
        else:
            # Vectorisé : (horizon, n_sim, nb_components)
            steps        = np.random.multivariate_normal(drift, cov, size=(horizon, n_sim))
            Z_future     = Z_last + np.cumsum(steps, axis=0)                    # (horizon, n_sim, nb_components)
            kappa_future = Z_future @ Vred.T                                     # (horizon, n_sim, nb_regions)

    elif model == "linear":
        time_idx  = np.arange(T)
        X_des     = np.column_stack([np.ones(T), time_idx])
        beta_lin  = np.linalg.lstsq(X_des, Zred, rcond=None)[0]                # (2, nb_components)

        future_idx = np.arange(T, T + horizon)
        Xf         = np.column_stack([np.ones(horizon), future_idx])
        Z_det      = Xf @ beta_lin                                               # (horizon, nb_components)

        if not stochastic:
            kappa_future = Z_det @ Vred.T                                        # (horizon, nb_regions)
        else:
            residuals = Zred - X_des @ beta_lin
            cov_res   = np.cov(residuals, rowvar=False).reshape(nb_components, nb_components)
            # Vectorisé : (horizon, n_sim, nb_components)
            noise        = np.random.multivariate_normal(np.zeros(nb_components), cov_res, size=(horizon, n_sim))
            Z_future     = Z_det[:, None, :] + noise                            # (horizon, n_sim, nb_components)
            kappa_future = Z_future @ Vred.T                                     # (horizon, n_sim, nb_regions)
    else:
        raise ValueError("model must be 'rw' or 'linear'")

    # =====================================================
    # 4️⃣ Reconstruction log-mortalité (sans espérance de vie)
    # =====================================================
    # ax : (nb_ages,)
    # bx : (nb_ages, nb_regions)
    # kappa_future stochastic : (horizon, n_sim, nb_regions)
    # kappa_future determin.  : (horizon, nb_regions)

    if not stochastic:
        # logmu : (nb_ages, nb_regions, horizon)
        logmu_future = ax[:, None, None] + bx[:, :, None] * kappa_future.T[None, :, :]
        mu_future    = np.exp(logmu_future)

        return {
            "kappa_future": kappa_future,   # (horizon, nb_regions)
            "logmu_future": logmu_future,   # (nb_ages, nb_regions, horizon)
            "mu_future":    mu_future,      # (nb_ages, nb_regions, horizon)
        }
    else:
        # logmu : (nb_ages, nb_regions, horizon, n_sim)
        logmu_sim = (
            ax[:, None, None, None]
            + bx[:, :, None, None] * kappa_future.transpose(2, 1, 0)[None, :, :, :]
        )
        mu_sim = np.exp(logmu_sim)

        # Percentiles sur kappa : (horizon, nb_regions)
        kappa_arr    = kappa_future.transpose(0, 2, 1)   # (horizon, nb_regions, n_sim)
        kappa_lower  = np.percentile(kappa_arr,  5, axis=2)
        kappa_median = np.percentile(kappa_arr, 50, axis=2)
        kappa_upper  = np.percentile(kappa_arr, 95, axis=2)

        # Percentiles sur mu : (nb_ages, nb_regions, horizon)
        mu_lower  = np.percentile(mu_sim,  5, axis=3)
        mu_median = np.percentile(mu_sim, 50, axis=3)
        mu_upper  = np.percentile(mu_sim, 95, axis=3)

        return {
            "kappa_paths":  kappa_future,   # (horizon, n_sim, nb_regions)
            "kappa_lower":  kappa_lower,    # (horizon, nb_regions)
            "kappa_median": kappa_median,
            "kappa_upper":  kappa_upper,
            "logmu_sim":    logmu_sim,      # (nb_ages, nb_regions, horizon, n_sim)
            "mu_sim":       mu_sim,
            "mu_lower":     mu_lower,       # (nb_ages, nb_regions, horizon)
            "mu_median":    mu_median,
            "mu_upper":     mu_upper,
        }


def high_age_extrapolation(
    xv,
    x_extrap,
    x_extrap_start,
    log_Muxtg,          # (nb_ages, nb_regions, horizon, n_sim) OU (nb_ages, nb_regions, horizon)
):
    """
    Extrapole log_Muxtg au-delà de max(xv) jusqu'à x_extrap
    via régression linéaire sans intercept sur [x_extrap_start, max(xv)].

    Fonctionne pour les deux cas :
      - stochastique : log_Muxtg de shape (nb_ages, nb_regions, horizon, n_sim)
      - déterministe : log_Muxtg de shape (nb_ages, nb_regions, horizon)
    """
    stochastic = log_Muxtg.ndim == 4

    x_max       = int(xv.max())
    xv_reg      = np.arange(x_extrap_start, x_max + 1)   # ages pour la régression
    xv_add      = np.arange(x_max + 1, x_extrap + 1)     # ages à extrapoler
    xv_full     = np.arange(int(xv.min()), x_extrap + 1)

    dx_reg  = (xv_reg - x_max).reshape(-1, 1).astype(float)   # centré en x_max
    dx_add  = (xv_add - x_max).reshape(-1, 1).astype(float)

    if stochastic:
        nb_ages, nb_regions, horizon, n_sim = log_Muxtg.shape
        out = np.empty((len(xv_full), nb_regions, horizon, n_sim))
        out[:nb_ages, ...] = log_Muxtg

        for r in range(nb_regions):
            for h in range(horizon):
                # Bloc (nb_ages_reg, n_sim) — régression vectorisée sur les simulations
                Y_reg   = log_Muxtg[xv_reg, r, h, :]          # (nb_ages_reg, n_sim)
                anchor  = log_Muxtg[x_max,  r, h, :]          # (n_sim,)
                Y_cent  = Y_reg - anchor                        # centré en x_max

                # Pente par moindres carrés sans intercept : β = (X'X)^{-1} X'Y
                # X = dx_reg (nb_ages_reg, 1) → solution scalaire
                slope   = (dx_reg.T @ Y_cent) / (dx_reg.T @ dx_reg)   # (1, n_sim)

                # Extrapolation
                Y_extrap = anchor + dx_add * slope                     # (nb_add, n_sim)
                out[nb_ages:, r, h, :] = Y_extrap

    else:
        nb_ages, nb_regions, horizon = log_Muxtg.shape
        out = np.empty((len(xv_full), nb_regions, horizon))
        out[:nb_ages, ...] = log_Muxtg

        for r in range(nb_regions):
            # Bloc (nb_ages_reg, horizon) — vectorisé sur l'horizon
            Y_reg  = log_Muxtg[xv_reg, r, :]           # (nb_ages_reg, horizon)
            anchor = log_Muxtg[x_max,  r, :]            # (horizon,)
            Y_cent = Y_reg - anchor

            slope   = (dx_reg.T @ Y_cent) / (dx_reg.T @ dx_reg)   # (1, horizon)
            Y_extrap = anchor + dx_add * slope                      # (nb_add, horizon)
            out[nb_ages:, r, :] = Y_extrap

    return out, xv_full


def high_age_extrapolation_snd(xv, x_extrap, x_extrap_start, log_Muxtg):

    stochastic  = log_Muxtg.ndim == 4
    x_max       = int(xv.max())

    idx_reg     = np.where((xv >= x_extrap_start) & (xv <= x_max))[0]
    idx_anchor  = np.where(xv == x_max)[0][0]

    # ✅ Guard
    assert len(idx_reg) >= 2, (
        f"Fenêtre de régression trop petite : "
        f"x_extrap_start={x_extrap_start}, x_max={x_max}, "
        f"idx_reg={idx_reg}"
    )

    xv_add  = np.arange(x_max + 1, x_extrap + 1)
    xv_full = np.concatenate([xv, xv_add])

    dx_reg  = (xv[idx_reg] - x_max).reshape(-1, 1).astype(float)
    dx_add  = (xv_add      - x_max).reshape(-1, 1).astype(float)
    denom   = float(dx_reg.T @ dx_reg)

    assert denom != 0, "Dénominateur nul dans la régression"

    if stochastic:
        nb_ages, nb_regions, horizon, n_sim = log_Muxtg.shape
        out = np.empty((len(xv_full), nb_regions, horizon, n_sim))
        out[:nb_ages, ...] = log_Muxtg

        for r in range(nb_regions):
            for h in range(horizon):
                Y_reg   = log_Muxtg[idx_reg,    r, h, :]   # (nb_reg_ages, n_sim)
                anchor  = log_Muxtg[idx_anchor, r, h, :]   # (n_sim,)
                Y_cent  = Y_reg - anchor                    # (nb_reg_ages, n_sim)
                slope   = (dx_reg.T @ Y_cent) / denom      # (1, n_sim)
                out[nb_ages:, r, h, :] = anchor + dx_add * slope  # (nb_add, n_sim)

        # ✅ Vérification post-calcul
        nan_by_age = np.isnan(out).any(axis=(1, 2, 3))
        if nan_by_age.any():
            print(f"⚠️ NaN aux indices d'âge : {np.where(nan_by_age)[0]}")

    else:
        nb_ages, nb_regions, horizon = log_Muxtg.shape
        out = np.empty((len(xv_full), nb_regions, horizon))
        out[:nb_ages, ...] = log_Muxtg

        for r in range(nb_regions):
            Y_reg   = log_Muxtg[idx_reg,    r, :]          # (nb_reg_ages, horizon)
            anchor  = log_Muxtg[idx_anchor, r, :]          # (horizon,)
            Y_cent  = Y_reg - anchor
            slope   = (dx_reg.T @ Y_cent) / denom          # (1, horizon)
            out[nb_ages:, r, :] = anchor + dx_add * slope

    return out, xv_full



def project_LeeLi_prospective(
    results,
    tv,
    horizon=30,
    exclude_years=[2020, 2021],
    model="rw",
    stochastic=True,
    n_sim=1000,
):
    # =====================================================
    # 1️⃣ Extract parameters
    # =====================================================
    alpha_xg = results["curves"]["alpha_xg"]   # (nb_ages, nb_regions)
    beta_x   = results["curves"]["beta_x"]     # (nb_ages,)
    beta_xg  = results["curves"]["beta_xg"]    # (nb_ages, nb_regions)
    kappa    = results["parameters"]["kappa"]  # (nb_years,)        — common factor
    kappa_g  = results["parameters"]["kappa_g"]# (nb_regions, nb_years) — region-specific

    nb_ages, nb_regions = alpha_xg.shape

    # =====================================================
    # 2️⃣ Mask COVID years
    # =====================================================
    mask     = ~np.isin(tv, exclude_years)
    kappa_m  = kappa[mask]                     # (T,)
    kappa_gm = kappa_g[:, mask]                # (nb_regions, T)
    T        = kappa_m.shape[0]

    # =====================================================
    # 3️⃣ Factor dynamics — common kappa
    # =====================================================
    def fit_dynamics(series, model, T):
        """
        series : (T,) or (T, nb_regions)
        Returns drift, cov, last_value
        """
        diffs = np.diff(series, axis=0)
        drift = np.mean(diffs, axis=0)
        if diffs.ndim == 1:
            cov = np.array([[np.var(diffs, ddof=1)]])
            drift = np.array([drift])
        else:
            cov = np.cov(diffs, rowvar=False)
        last = series[-1] if series.ndim == 1 else series[-1, :]
        return drift, cov, last

    if model == "rw":
        # Common factor
        drift_k,  cov_k,  k_last  = fit_dynamics(kappa_m,       model, T)
        # Region-specific factors
        drift_kg, cov_kg, kg_last = fit_dynamics(kappa_gm.T,     model, T)
        # cov_kg : (nb_regions, nb_regions)

    elif model == "linear":
        time_idx = np.arange(T)
        X_des    = np.column_stack([np.ones(T), time_idx])
        Xf       = np.column_stack([np.ones(horizon), np.arange(T, T + horizon)])

        # Common factor linear fit
        beta_k    = np.linalg.lstsq(X_des, kappa_m,   rcond=None)[0]      # (2,)
        beta_kg   = np.linalg.lstsq(X_des, kappa_gm.T, rcond=None)[0]     # (2, nb_regions)

        k_det     = Xf @ beta_k                                             # (horizon,)
        kg_det    = Xf @ beta_kg                                            # (horizon, nb_regions)

        if stochastic:
            res_k  = kappa_m   - X_des @ beta_k                            # (T,)
            res_kg = kappa_gm.T - X_des @ beta_kg                          # (T, nb_regions)
            cov_k  = np.array([[np.var(res_k,  ddof=1)]])
            cov_kg = np.cov(res_kg, rowvar=False)
    else:
        raise ValueError("model must be 'rw' or 'linear'")

    # =====================================================
    # 4️⃣ Projection
    # =====================================================
    if not stochastic:
        if model == "rw":
            # Common kappa : (horizon,)
            k_future  = k_last  + drift_k[0]  * np.arange(1, horizon + 1)

            # Region kappa_g : (horizon, nb_regions)
            kg_future = kg_last + drift_kg     * np.arange(1, horizon + 1)[:, None]

        else:  # linear
            k_future  = k_det                  # (horizon,)
            kg_future = kg_det                 # (horizon, nb_regions)

        # ── Reconstruction logmu : (nb_ages, nb_regions, horizon) ──
        # log(mu) = alpha_xg + beta_x * kappa + beta_xg * kappa_g
        logmu_future = (
            alpha_xg[:, :, None]
            + beta_x[:, None, None]  * k_future[None, None, :]
            + beta_xg[:, :, None]    * kg_future.T[None, :, :]
        )
        mu_future = np.exp(logmu_future)

        return {
            "kappa_future":  k_future,      # (horizon,)
            "kappa_g_future": kg_future,    # (horizon, nb_regions)
            "logmu_future":  logmu_future,  # (nb_ages, nb_regions, horizon)
            "mu_future":     mu_future,     # (nb_ages, nb_regions, horizon)
        }

    else:
        if model == "rw":
            # Common kappa : (horizon, n_sim)
            steps_k   = np.random.multivariate_normal(drift_k,  cov_k,  size=(horizon, n_sim))
            k_future  = k_last  + np.cumsum(steps_k[:, :, 0],  axis=0) # (horizon, n_sim)

            # Region kappa_g : (horizon, n_sim, nb_regions)
            steps_kg  = np.random.multivariate_normal(drift_kg, cov_kg, size=(horizon, n_sim))
            kg_future = kg_last + np.cumsum(steps_kg, axis=0)           # (horizon, n_sim, nb_regions)

        else:  # linear
            noise_k   = np.random.multivariate_normal(np.zeros(1),         cov_k,  size=(horizon, n_sim))
            k_future  = k_det[:, None]    + noise_k[:, :, 0]               # (horizon, n_sim)

            noise_kg  = np.random.multivariate_normal(np.zeros(nb_regions), cov_kg, size=(horizon, n_sim))
            kg_future = kg_det[:, None, :] + noise_kg                       # (horizon, n_sim, nb_regions)

        # ── Reconstruction logmu : (nb_ages, nb_regions, horizon, n_sim) ──
        # k_future  : (horizon, n_sim)         → (1, 1, horizon, n_sim)
        # kg_future : (horizon, n_sim, nb_reg) → (1, nb_reg, horizon, n_sim)
        logmu_sim = (
            alpha_xg[:, :, None, None]
            + beta_x[:,  None, None, None] * k_future.T[None, None, :, :]
            + beta_xg[:, :,    None, None] * kg_future.transpose(2, 1, 0)[None, :, :, :]
        )
        mu_sim = np.exp(logmu_sim)

        # Percentiles kappa commun : (horizon,)
        k_lower  = np.percentile(k_future,  5,  axis=1)
        k_median = np.percentile(k_future,  50, axis=1)
        k_upper  = np.percentile(k_future,  95, axis=1)

        # Percentiles kappa_g : (horizon, nb_regions)
        kg_arr    = kg_future                                    # (horizon, n_sim, nb_regions)
        kg_lower  = np.percentile(kg_arr,  5,  axis=1)
        kg_median = np.percentile(kg_arr,  50, axis=1)
        kg_upper  = np.percentile(kg_arr,  95, axis=1)

        # Percentiles mu : (nb_ages, nb_regions, horizon)
        mu_lower  = np.percentile(mu_sim,  5,  axis=3)
        mu_median = np.percentile(mu_sim,  50, axis=3)
        mu_upper  = np.percentile(mu_sim,  95, axis=3)

        return {
            "kappa_paths":    k_future,     # (horizon, n_sim)
            "kappa_lower":    k_lower,      # (horizon,)
            "kappa_median":   k_median,
            "kappa_upper":    k_upper,
            "kappa_g_paths":  kg_future,    # (horizon, n_sim, nb_regions)
            "kappa_g_lower":  kg_lower,     # (horizon, nb_regions)
            "kappa_g_median": kg_median,
            "kappa_g_upper":  kg_upper,
            "logmu_sim":      logmu_sim,    # (nb_ages, nb_regions, horizon, n_sim)
            "mu_sim":         mu_sim,
            "mu_lower":       mu_lower,     # (nb_ages, nb_regions, horizon)
            "mu_median":      mu_median,
            "mu_upper":       mu_upper,
        }