# -*- coding: utf-8 -*-
"""
LiLee_fit — Parametric Li-Lee model with B-splines/P-splines
================================================================
Multi-population Li-Lee model:
    ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

Where:
  α_{x,g}   : mortality baseline by age AND region (B-splines)
  β_x       : common sensitivity across all regions (B-splines)
  κ_t       : common time factor
  β_{x,g}   : regional sensitivity (B-splines)
  κ_{g,t}   : regional time factor

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
from morta_nuts2.model.Bsplines.Bsplines import make_bspline_basis,eval_bspline_from_coef




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


# =============================================================================
# 2. ln(µ) RECONSTRUCTION — LI-LEE MODEL
# =============================================================================

def compute_logmu_lilee(
    alpha_coef,  # (nb_regions, n_basis)  α_{x,g}
    beta_coef,   # (n_basis,)             β_x common
    beta_g_coef, # (nb_regions, n_basis)  β_{x,g}
    kappa,       # (nb_years,)            κ_t common
    kappa_g,     # (nb_regions, nb_years) κ_{g,t}
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
    nb_ages = len(xv)
    
    # Evaluate B-spline curves
    alpha = np.zeros((nb_ages, nb_regions))
    beta_g = np.zeros((nb_ages, nb_regions))
    for g in range(nb_regions):
        alpha[:, g] = B @ alpha_coef[g]
        beta_g[:, g] = B @ beta_g_coef[g]
    
    beta = B @ beta_coef
    
    # Build ln(µ) by broadcasting
    # Dimensions: (nb_ages, nb_years, nb_regions)
    logmu = (
        alpha[:, None, :]                                # (nb_ages, 1, nb_regions)
        + beta[:, None, None] * kappa[None, :, None]     # β_x · κ_t
        + beta_g[:, None, :] * kappa_g.T[None, :, :]     # β_{x,g} · κ_{g,t}
    )
    
    return logmu, alpha, beta, beta_g


# =============================================================================
# 3. POISSON LOG-LIKELIHOOD
# =============================================================================

def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
    """Poisson log-likelihood."""
    exp_logmu = np.exp(logmu)
    weighted_exp = Extg * exp_logmu
    residual = Dxtg - weighted_exp
    lnL = float(np.sum(
        Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
    ))
    return lnL, exp_logmu, weighted_exp, residual


# =============================================================================
# 4. NR UPDATES — LI-LEE MODEL
# =============================================================================

def update_alpha_coef(alpha_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """
    NR update of α_{x,g} (regional baseline).
    One coefficient vector per region.
    """
    nb_regions = alpha_coef.shape[0]
    n_basis = alpha_coef.shape[1]
    alpha_coef_new = alpha_coef.copy()
    
    for g in range(nb_regions):
        pen_grad = 2.0 * lam * (DtD @ alpha_coef[g]) if lam > 0 else np.zeros(n_basis)
        for j in range(n_basis):
            Bj3d = B[:, j][:, None, None]
            num = float(np.sum(residual[:, :, g:g+1] * Bj3d)) - pen_grad[j]
            den = float(np.sum(weighted_exp[:, :, g:g+1] * Bj3d**2))
            if lam > 0:
                den += 2.0 * lam * diag_DtD[j]
            if den != 0:
                alpha_coef_new[g, j] += eta * num / den
    return alpha_coef_new


def update_beta_coef(beta_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """
    NR update of β_x (common sensitivity).
    A single coefficient vector for all regions.
    """
    n_basis = len(beta_coef)
    nb_ages = B.shape[0]
    beta_coef_new = beta_coef.copy()
    pen_grad = 2.0 * lam * (DtD @ beta_coef) if lam > 0 else np.zeros(n_basis)
    
    kappaM = np.repeat(kappa[None, :], nb_ages, axis=0)
    
    for j in range(n_basis):
        BjKappa = B[:, j][:, None] * kappaM
        BjK3d = BjKappa[:, :, None]
        
        num = float(np.sum(residual * BjK3d)) - pen_grad[j]
        den = float(np.sum(weighted_exp * BjK3d**2))
        if lam > 0:
            den += 2.0 * lam * diag_DtD[j]
        if den != 0:
            beta_coef_new[j] += eta * num / den
    return beta_coef_new


def update_beta_g_coef(beta_g_coef, B, kappa_g, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """
    NR update of β_{x,g} (regional sensitivity).
    One coefficient vector per region.
    """
    nb_regions = beta_g_coef.shape[0]
    nb_ages = B.shape[0]
    beta_g_coef_new = beta_g_coef.copy()
    
    for g in range(nb_regions):
        pen_grad = 2.0 * lam * (DtD @ beta_g_coef[g]) if lam > 0 else np.zeros(beta_g_coef.shape[1])
        kappaM = np.repeat(kappa_g[g, None, :], nb_ages, axis=0)  # (nb_ages, nb_years)
        
        for j in range(beta_g_coef.shape[1]):
            BjKappa = B[:, j][:, None] * kappaM
            BjK3d = BjKappa[:, :, None]
            
            num = float(np.sum(residual[:, :, g:g+1] * BjK3d)) - pen_grad[j]
            den = float(np.sum(weighted_exp[:, :, g:g+1] * BjK3d**2))
            if lam > 0:
                den += 2.0 * lam * diag_DtD[j]
            if den != 0:
                beta_g_coef_new[g, j] += eta * num / den
    return beta_g_coef_new


def update_kappa(kappa, beta, residual, weighted_exp, eta):
    """
    NR update of κ_t (common time factor).
    """
    beta3d = beta[:, None, None]
    num_k = np.sum(residual * beta3d, axis=(0, 2))
    den_k = np.sum(weighted_exp * beta3d**2, axis=(0, 2))
    kappa_new = kappa.copy()
    mask = den_k != 0
    kappa_new[mask] += eta * num_k[mask] / den_k[mask]
    return kappa_new


def update_kappa_g(kappa_g, beta_g, residual, weighted_exp, eta):
    """
    NR update of κ_{g,t} (regional time factors).
    """
    nb_regions = kappa_g.shape[0]
    kappa_g_new = kappa_g.copy()
    
    for g in range(nb_regions):
        beta_g3d = beta_g[:, g][:, None, None]
        num_k = np.sum(residual[:, :, g:g+1] * beta_g3d, axis=0).squeeze()
        den_k = np.sum(weighted_exp[:, :, g:g+1] * beta_g3d**2, axis=0).squeeze()
        mask = den_k != 0
        kappa_g_new[g, mask] += eta * num_k[mask] / den_k[mask]
    
    return kappa_g_new


# =============================================================================
# 5. NORMALIZATION (IDENTIFIABILITY CONSTRAINTS)
# =============================================================================

def normalize_lilee(beta_coef, beta_g_coef, kappa, kappa_g, B):
    """
    Li-Lee identifiability constraints:
      1. Σ_x β_x = 1
      2. Σ_x β_{x,g} = 0  for all g
      3. Σ_t κ_{g,t} = 0  for all g
    
    These constraints ensure uniqueness of the solution.
    """
    # 1. Normalization of β_x : Σ_x β_x = 1
    beta = B @ beta_coef
    scal_beta = float(np.sum(beta))
    if scal_beta != 0:
        beta_coef = beta_coef / scal_beta
        kappa = kappa * scal_beta
    
    # 2. Normalization of β_{x,g} : Σ_x β_{x,g} = 0 for all g
    nb_regions = beta_g_coef.shape[0]
    for g in range(nb_regions):
        beta_g = B @ beta_g_coef[g]
        sum_beta_g = float(np.sum(beta_g))
        if sum_beta_g != 0:
            # Subtract mean to center at 0
            adjustment = sum_beta_g / len(beta_g)
            beta_g_coef[g] -= adjustment / np.mean(B.sum(axis=0))
    
    # 3. Normalization of κ_{g,t} : Σ_t κ_{g,t} = 0 for all g
    for g in range(nb_regions):
        mean_kappa_g = float(np.mean(kappa_g[g]))
        kappa_g[g] -= mean_kappa_g
    
    return beta_coef, beta_g_coef, kappa, kappa_g


# =============================================================================
# 6. FIT STATISTICS
# =============================================================================

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


# =============================================================================
# 7. MAIN FUNCTION — LI-LEE MODEL
# =============================================================================

def LiLee_p_fit(
    alpha_coef_init,  # (nb_regions, n_basis)
    beta_coef_init,   # (n_basis,)
    beta_g_coef_init, # (nb_regions, n_basis)
    kappa_init,       # (nb_years,)
    kappa_g_init,     # (nb_regions, nb_years)
    Extg,             # (nb_ages, nb_years, nb_regions)
    Dxtg,             # (nb_ages, nb_years, nb_regions)
    xv,               # (nb_ages,)
    tv,               # (nb_years,)
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
    """
    Calibrates the parametric Li-Lee model:
        ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}
    
    USES ONLY scipy/numpy:
      ✓ scipy.interpolate.BSpline
      ✓ scipy.special.gammaln
      ✓ numpy.diff
    
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
    degree           : int   B-spline degree (3 recommended)
    n_knots          : int   number of internal knots (6 recommended)
    lam              : float P-splines penalty weight
    diff_order       : int   difference order (2 recommended)
    nb_iter          : int   max number of iterations
    eta0             : float initial learning rate (0.30 recommended)
    tol              : float convergence tolerance
    verbose          : bool  display progress
    
    Returns
    --------
    alpha_coef  : (nb_regions, n_basis)  α_{x,g} coefficients
    beta_coef   : (n_basis,)             β_x coefficients
    beta_g_coef : (nb_regions, n_basis)  β_{x,g} coefficients
    kappa       : (nb_years,)            κ_t
    kappa_g     : (nb_regions, nb_years) κ_{g,t}
    alpha       : (nb_ages, nb_regions)  α_{x,g} curves
    beta        : (nb_ages,)             β_x curve
    beta_g      : (nb_ages, nb_regions)  β_{x,g} curves
    logmu_final : ln(µ_{x,t,g})
    Fit_stat    : pd.DataFrame           statistics
    """
    nb_years = len(tv)
    nb_regions = Extg.shape[2]

    if xmin is None:
        xmin = float(np.min(xv))
    if xmax is None:
        xmax = float(np.max(xv))

    # Build B-spline basis matrix
    B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)

    # Dimension checks
    if alpha_coef_init.shape != (nb_regions, n_basis):
        raise ValueError(f"alpha_coef_init must have shape ({nb_regions}, {n_basis})")
    if len(beta_coef_init) != n_basis:
        raise ValueError(f"beta_coef_init must have {n_basis} elements")
    if beta_g_coef_init.shape != (nb_regions, n_basis):
        raise ValueError(f"beta_g_coef_init must have shape ({nb_regions}, {n_basis})")

    alpha_coef = alpha_coef_init.copy()
    beta_coef = beta_coef_init.copy()
    beta_g_coef = beta_g_coef_init.copy()
    kappa = kappa_init.copy()
    kappa_g = kappa_g_init.copy()

    # P-splines penalty
    DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)

    # Log-factorial constant
    logDxtgFact = gammaln(Dxtg + 1)

    lnL = 0.0
    Delta_lnL = -1000.0
    flag = 0
    it = -1
    eta = eta0

    if verbose:
        print("=" * 70)
        print("PARAMETRIC LI-LEE MODEL CALIBRATION")
        print("=" * 70)
        print(f"Parameters: degree={degree}, n_knots={n_knots}, lam={lam}")
        print(f"Data: {Dxtg.shape[0]} ages × {nb_years} years × {nb_regions} regions")
        print(f"Number of basis functions: {n_basis}")
        print("=" * 70)

    # NR loop
    while (it < nb_iter) and (flag < 4):
        it += 1

        # Adaptive learning rate
        if Delta_lnL < 0:
            eta *= 0.5
        else:
            eta = min(eta * 1.05, 2.0)

        # Stopping criterion
        if np.abs(Delta_lnL) < tol:
            flag += 1
        else:
            flag = 0

        # Reconstruct ln(µ)
        logmu, alpha, beta, beta_g = compute_logmu_lilee(
            alpha_coef,
            beta_coef,
            beta_g_coef,
            kappa,
            kappa_g,
            xv,
            B,
            knots,
            degree,
        )

        # Log-likelihood
        lnL_new, _, weighted_exp, residual = poisson_lnL(
            Dxtg, Extg, logmu, logDxtgFact
        )

        Delta_lnL = lnL_new - lnL
        lnL = lnL_new

        if verbose and (it % 10 == 0):
            print(f"It {it:4d} | lnL = {lnL:,.2f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

        # Sequential NR updates
        alpha_coef = update_alpha_coef(
            alpha_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD
        )

        beta_coef = update_beta_coef(
            beta_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD
        )

        beta_g_coef = update_beta_g_coef(
            beta_g_coef, B, kappa_g, residual, weighted_exp, eta, lam, DtD, diag_DtD
        )

        kappa = update_kappa(kappa, beta, residual, weighted_exp, eta)

        kappa_g = update_kappa_g(
            kappa_g, beta_g, residual, weighted_exp, eta
        )

        # Final normalization
        beta_coef, beta_g_coef, kappa, kappa_g = normalize_lilee(
            beta_coef, beta_g_coef, kappa, kappa_g, B
        )

    # Final reconstruction
    logmu_final, alpha, beta, beta_g = compute_logmu_lilee(
        alpha_coef,
        beta_coef,
        beta_g_coef,
        kappa,
        kappa_g,
        xv,
        B,
        knots,
        degree,
    )

    # Statistics
    Fit_stat = compute_fit_stats(
        Dxtg,
        Extg,
        logmu_final,
        logDxtgFact,
        n_basis,
        nb_years,
        nb_regions,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"Convergence reached after {it + 1} iterations")
        print("\nFinal statistics:")
        print(Fit_stat.to_string(index=False))
        print("=" * 70)

    results = {

        "parameters": {
            "alpha_coef": alpha_coef,
            "beta_coef": beta_coef,
            "beta_g_coef": beta_g_coef,
            "kappa": kappa,
            "kappa_g": kappa_g,
        },

        "curves": {
            "alpha_xg": alpha,
            "beta_x": beta,
            "beta_xg": beta_g,
        },

        "fitted_values": {
            "log_mu": logmu_final,
            "mu": np.exp(logmu_final),
        },

        "fit_statistics": Fit_stat,
    }

    return results



import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.special import gammaln
from scipy.linalg import lstsq
import matplotlib.pyplot as plt



# =============================================================================
# 1. LEE-CARTER MODEL CONSTRUCTION: inspired by Donatien
# =============================================================================

#%% Smoothing procedure
from math import comb
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

def LC_fit(ax, bx,kappa,Extg,Dxtg,xv,tv,nb_iter):
    #gradient descent parameter
    eta = 1
    for it in range(nb_iter):
        for ct_opt in np.arange(0,3):
            ax = ax.reshape(-1,1) ; bx = bx.reshape(-1,1) 
            nb_regions = Extg.shape[2]
            axM     = np.repeat(ax,len(tv),axis=1)
            bxM     = np.repeat(bx,len(tv),axis=1)            
            kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
            logmuxt_baseline = axM+bxM*kappaM
            logmuxt_grp  = np.zeros((len(xv),len(tv),nb_regions))    
            #computation of log(mu(x,t,g))
            for ct in range(nb_regions):                
                logmuxt_grp[:,:,ct] = logmuxt_baseline.copy()
            #baseline for update
            dlnL_baseline  = (Dxtg - Extg*np.exp(logmuxt_grp))
            if (ct_opt==0):
                #--------------- ax --------------------
                ax_new    = np.zeros_like(ax)             
                dlnL_dpar = (np.sum(dlnL_baseline,axis=(1,2))/
                            np.sum(Extg*np.exp(logmuxt_grp),axis=(1,2)) )
                ax_new = ax + eta* dlnL_dpar.reshape(-1,1)                
                #update
                ax     = ax_new.copy()            
            if (ct_opt==1):
                #--------------- bx --------------------
                bx_new = np.zeros_like(bx)            
                kappaM= np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
                kappaM= np.expand_dims(kappaM,axis=2)
                kappaM= np.repeat(kappaM,nb_regions,axis=2)                
                dlnL_dpar = (np.sum(dlnL_baseline*kappaM,axis=(1,2))/(
                            np.sum(Extg*np.exp(logmuxt_grp)*kappaM**2,axis=(1,2))))
                bx_new = bx + eta*dlnL_dpar.reshape(-1,1)     
                #we normalize
                scal_bx   = np.sum(bx_new)
                bx_new    = bx_new /scal_bx
                kappa     = kappa*scal_bx
                bx        = bx_new.copy()                
            if (ct_opt==2):
                #---------------Kappa-----------------    
                # warning we use the old betax(x)
                kappa_new = np.zeros_like(kappa)
                bxM = np.repeat(bx,len(tv),axis=1)
                bxM = np.expand_dims(bxM,axis=2)
                bxM = np.repeat(bxM,nb_regions,axis=2)                
                dlnL_dpar = (np.sum(dlnL_baseline*bxM,axis=(0,2))/(
                      np.sum(Extg*np.exp(logmuxt_grp)*bxM**2,axis=(0,2))))
                kappa_new = kappa + eta*dlnL_dpar
                #we rescale
                kappa_avg = np.mean(kappa_new)
                kappa_new = (kappa_new - kappa_avg) #*np.sum(bx)
                ax        =  ax + kappa_avg*bx               
                #update
                kappa  = kappa_new.copy()         
    #end loop
    # we recompute log-mort. rates
    ax = ax.reshape(-1,1)  ;  bx = bx.reshape(-1,1)         
    axM     = np.repeat(ax,len(tv),axis=1)
    bxM     = np.repeat(bx,len(tv),axis=1)                
    kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
    logmuxt_grp = axM+bxM*kappaM  
    logmuxt_grp = np.repeat(logmuxt_grp[:,:,np.newaxis],nb_regions,axis=2)
    #log-likelihood      
    exp_logmuxt = np.exp(logmuxt_grp)    
    logDxtgFact = gammaln(Dxtg + 1)
    lnL         = np.sum(Dxtg * logmuxt_grp - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)          
    #dof's and numbers of records
    nb_obs  = Dxtg.size 
    dofs    = len(ax) + len(bx) + len(kappa) 
    AIC     = 2*dofs - 2*lnL    
    BIC     = dofs*np.log(nb_obs)  - 2*lnL
    #dataframe with statistics of goodness of fit
    Fit_stat = [[nb_obs,'NA','NA',dofs,np.round(lnL,2),np.round(AIC,2),np.round(BIC,2)] ]
    #We print the file
    Fit_stat         = pd.DataFrame(Fit_stat)
    Fit_stat.columns = ["N","m","degree","dofs","lnL","AIC","BIC"]    
    #we return ax, bx, kappa and stats    
    return ax, bx , kappa , Fit_stat

#%% Lee and Li
# def LandL_fit(ax, bx , bx_gr , kappa, kappa_gr, 
#               Extg, Dxtg,Muxtg, xv, tv, nb_iter,h,z, verbose):
#     #matrix of differences, order z    
#     Kz = difference_matrix(len(ax), z)
#     KTK   = Kz.T @ Kz 
#     IdKTK = np.diag(KTK)    
#     # ax and bx are computed with the Poisson LC
#     ax, bx, kappa , _ = LC_fit(ax, bx,kappa,Extg,Dxtg,xv,tv,nb_iter)
#     #gradient descent parameter
#     eta = 0.80
#     for it in range(nb_iter):
#         for ct_opt in np.arange(0,2):
#             axM     = np.repeat(ax,len(tv),axis=1)
#             bxM     = np.repeat(bx,len(tv),axis=1)            
#             bx_grM  = np.expand_dims(bx_gr,axis=1)
#             bx_grM  = np.repeat(bx_grM,len(tv),axis=1)                        
#             kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
#             kappa_grM = np.expand_dims(kappa_gr,axis=0)
#             kappa_grM = np.repeat(kappa_grM,len(xv),axis=0)                        
#             logmuxt_baseline = axM+bxM*kappaM
#             nb_regions = Extg.shape[2]    
#             logmuxt_gr  = np.zeros((len(xv),len(tv),nb_regions))    
#             #computation of log(mu(x,t,g))
#             for ct in range(nb_regions):                
#                 logmuxt_gr[:,:,ct] = (logmuxt_baseline + bx_grM[:,:,ct]*kappa_grM[:,:,ct])               
#             #baseline for update
#             dlnL_baseline  = (Dxtg - Extg*np.exp(logmuxt_gr))
#             if (ct_opt==0):                
#                 #--------------- bx gr-----------------    
#                 bx_gr_new = np.zeros_like(bx_gr)                  
#                 dlnL_dpar = ( (np.sum(dlnL_baseline*kappa_grM,axis=1) -2*h*(KTK@bx_gr) ) / (                    
#                             np.sum(Extg*np.exp(logmuxt_gr)*kappa_grM**2,axis=1)
#                             -2*h*np.repeat(IdKTK.reshape(-1,1),nb_regions,axis=1) ) )  
#                 bx_gr_new= bx_gr + eta*dlnL_dpar
#                 #----scaling----
#                 #we normalize
#                 scal_bx   = np.sum(bx_gr_new,axis=0).reshape((1,-1))
#                 scal_bx   = np.repeat(scal_bx,len(xv),axis=0)
#                 bx_gr_new    = bx_gr_new /scal_bx
#                 scal_kap     = scal_bx[0,:].reshape((1,-1))
#                 scal_kap     = np.repeat(scal_kap,len(tv),axis=0)
#                 # kappa_gr     = kappa_gr*scal_kap
#                 #-----plot------     
#                 if verbose:
#                     plt.plot(bx_gr_new,label='bx gdp new')
#                     #plt.plot(bx_gr,label='bx gdp')
#                     plt.title("bx_gr")
#                     #plt.legend()
#                     plt.show()
#                 #update
#                 bx_gr  = bx_gr_new
#             if (ct_opt==1):                
#                 #--------------- kappa gr-----------------    
#                 kappa_gr_new = np.zeros_like(kappa_gr)
#                 dlnL_dpar    = (np.sum(dlnL_baseline*bx_grM,axis=0)/(
#                       np.sum(Extg*np.exp(logmuxt_gr)*bx_grM**2,axis=0)))
#                 kappa_gr_new = kappa_gr + eta*dlnL_dpar
#                 #---- Do not scale!
#                 #-----plot----
#                 if verbose:
#                     plt.plot(kappa_gr_new,label='kappa new')
#                     plt.title("kappa_gr")
#                     # plt.plot(kappa_gr,label='kappa')
#                     # plt.legend()
#                     plt.show()
#                 #update
#                 kappa_gr  = kappa_gr_new.copy()     
#     #end loop   
#     #log-likelihood      
#     exp_logmuxt = np.exp(logmuxt_gr)    
#     logDxtgFact = gammaln(Dxtg + 1)
#     lnL         = np.sum(Dxtg * logmuxt_gr - Extg * exp_logmuxt + Dxtg * np.log(Extg) - logDxtgFact)          
#     #dof's and numbers of records
#     nb_obs  = Dxtg.size 
#     dofs    = len(ax)+len(bx)+np.size(bx_gr)+np.size(kappa_gr)+np.size(kappa)
#     AIC     = 2*dofs - 2*lnL    
#     BIC     = dofs*np.log(nb_obs)  - 2*lnL
#     #dataframe with statistics of goodness of fit
#     Fit_stat = [[nb_obs,'NA','NA',dofs,np.round(lnL,2),np.round(AIC,2),np.round(BIC,2)] ]
#     #We print the file
#     Fit_stat         = pd.DataFrame(Fit_stat)
#     Fit_stat.columns = ["N","m","degree","dofs","lnL","AIC","BIC"]    
#     # Return updated coefficients and stats
#     
#     return ax, bx , bx_gr , kappa ,kappa_gr , Fit_stat


def LandL_fit(ax, bx , bx_gr , kappa, kappa_gr, 
              Extg, Dxtg,Muxtg, xv, tv, nb_iter,h,z, verbose):

    #matrix of differences, order z    
    Kz = difference_matrix(len(ax), z)
    KTK   = Kz.T @ Kz 
    IdKTK = np.diag(KTK)    

    # ax and bx are computed with the Poisson LC
    ax, bx, kappa , _ = LC_fit(ax, bx,kappa,Extg,Dxtg,xv,tv,nb_iter)

    #gradient descent parameter
    eta = 0.80

    for it in range(nb_iter):
        for ct_opt in np.arange(0,2):

            axM     = np.repeat(ax,len(tv),axis=1)
            bxM     = np.repeat(bx,len(tv),axis=1)            
            bx_grM  = np.expand_dims(bx_gr,axis=1)
            bx_grM  = np.repeat(bx_grM,len(tv),axis=1)                        
            kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
            kappa_grM = np.expand_dims(kappa_gr,axis=0)
            kappa_grM = np.repeat(kappa_grM,len(xv),axis=0)                        

            logmuxt_baseline = axM+bxM*kappaM
            nb_regions = Extg.shape[2]    
            logmuxt_gr  = np.zeros((len(xv),len(tv),nb_regions))    

            #computation of log(mu(x,t,g))
            for ct in range(nb_regions):                
                logmuxt_gr[:,:,ct] = (logmuxt_baseline + bx_grM[:,:,ct]*kappa_grM[:,:,ct])               

            #baseline for update
            dlnL_baseline  = (Dxtg - Extg*np.exp(logmuxt_gr))

            if (ct_opt==0):                

                #--------------- bx gr-----------------    
                bx_gr_new = np.zeros_like(bx_gr)                  

                dlnL_dpar = ( (np.sum(dlnL_baseline*kappa_grM,axis=1) -2*h*(KTK@bx_gr) ) / (                    
                            np.sum(Extg*np.exp(logmuxt_gr)*kappa_grM**2,axis=1)
                            -2*h*np.repeat(IdKTK.reshape(-1,1),nb_regions,axis=1) ) )  

                bx_gr_new= bx_gr + eta*dlnL_dpar

                #----scaling----
                scal_bx   = np.sum(bx_gr_new,axis=0).reshape((1,-1))
                scal_bx   = np.repeat(scal_bx,len(xv),axis=0)
                bx_gr_new    = bx_gr_new /scal_bx
                scal_kap     = scal_bx[0,:].reshape((1,-1))
                scal_kap     = np.repeat(scal_kap,len(tv),axis=0)

                if verbose:
                    plt.plot(bx_gr_new,label='bx gdp new')
                    plt.title("bx_gr")
                    plt.show()

                bx_gr  = bx_gr_new

            if (ct_opt==1):                

                #--------------- kappa gr-----------------    
                kappa_gr_new = np.zeros_like(kappa_gr)

                dlnL_dpar    = (np.sum(dlnL_baseline*bx_grM,axis=0)/(
                      np.sum(Extg*np.exp(logmuxt_gr)*bx_grM**2,axis=0)))

                kappa_gr_new = kappa_gr + eta*dlnL_dpar

                if verbose:
                    plt.plot(kappa_gr_new,label='kappa new')
                    plt.title("kappa_gr")
                    plt.show()

                kappa_gr  = kappa_gr_new.copy()     

    #end loop   

    # =============================
    # Final log(mu) and mu
    # =============================

    axM     = np.repeat(ax,len(tv),axis=1)
    bxM     = np.repeat(bx,len(tv),axis=1)            
    bx_grM  = np.expand_dims(bx_gr,axis=1)
    bx_grM  = np.repeat(bx_grM,len(tv),axis=1)                        
    kappaM  = np.repeat(kappa.reshape(1,-1),len(xv),axis=0)
    kappa_grM = np.expand_dims(kappa_gr,axis=0)
    kappa_grM = np.repeat(kappa_grM,len(xv),axis=0)                        

    nb_regions = Extg.shape[2]
    logmu_final  = np.zeros((len(xv),len(tv),nb_regions))

    for ct in range(nb_regions):                
        logmu_final[:,:,ct] = (
            axM + bxM*kappaM +
            bx_grM[:,:,ct]*kappa_grM[:,:,ct]
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

    nb_obs  = Dxtg.size 
    dofs    = len(ax)+len(bx)+np.size(bx_gr)+np.size(kappa_gr)+np.size(kappa)

    AIC     = 2*dofs - 2*lnL    
    BIC     = dofs*np.log(nb_obs)  - 2*lnL

    Fit_stat = [[nb_obs,'NA','NA',dofs,np.round(lnL,2),np.round(AIC,2),np.round(BIC,2)] ]
    Fit_stat = pd.DataFrame(Fit_stat)
    Fit_stat.columns = ["N","m","degree","dofs","lnL","AIC","BIC"]    

    # =============================
    # Return dictionary
    # =============================

    results = {
        "parameters": {
            "ax": ax,
            "bx": bx,
            "bx_gr": bx_gr,
            "kappa": kappa,
            "kappa_gr": kappa_gr,
        },
        "fitted_values": {
            "log_mu": logmu_final,
            "mu": mu_final,
        },
        "fit_statistics": Fit_stat
    }

    return results


