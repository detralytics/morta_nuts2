# -*- coding: utf-8 -*-
"""
LiLee_fit — Modèle Li-Lee paramétrique avec B-splines/P-splines
================================================================
Modèle Li-Lee multi-population :
    ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}

Où :
  α_{x,g}   : baseline mortalité par âge ET région (B-splines)
  β_x       : sensibilité commune à toutes les régions (B-splines)
  κ_t       : facteur temporel commun
  β_{x,g}   : sensibilité régionale (B-splines)
  κ_{g,t}   : facteur temporel régional

UTILISE UNIQUEMENT scipy/numpy :
  ✓ scipy.interpolate.BSpline
  ✓ scipy.special.gammaln
  ✓ numpy.diff (matrice de pénalité)

Référence : Hainaut (2025), équation (1) page 3
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import gammaln
#from morta_nuts2.model.Bsplines import make_bspline_basis,eval_bspline_from_coef



# =============================================================================
# 1. CONSTRUCTION DES B-SPLINES — scipy.interpolate.BSpline
# =============================================================================

def make_bspline_basis(xv, degree, n_knots, xmin=None, xmax=None):
    """
    Construit la matrice de base B-spline avec scipy.interpolate.BSpline.
    
    Paramètres
    ----------
    xv       : array   points d'évaluation (ex: âges 0 à 82)
    degree   : int     degré des B-splines (3 recommandé)
    n_knots  : int     nombre de nœuds internes (équivalent à m+1 du code source)
    xmin     : float   borne inférieure (défaut: min(xv))
    xmax     : float   borne supérieure (défaut: max(xv))
    
    Retourne
    --------
    B      : (len(xv), n_basis)   matrice de base B-spline
    knots  : array                vecteur de nœuds complet (avec multiplicité aux bords)
    n_basis: int                  nombre de fonctions de base
    
    Notes
    -----
    scipy.interpolate.BSpline utilise la convention :
      - nœuds internes : np.linspace(xmin, xmax, n_knots)
      - nœuds complets : répétition degree+1 fois aux bords
      - n_basis = n_knots + degree - 1
    """
    if xmin is None:
        xmin = float(np.min(xv))
    if xmax is None:
        xmax = float(np.max(xv))
    
    # Nœuds internes équidistants
    internal_knots = np.linspace(xmin, xmax, n_knots)
    
    # Nœuds complets avec multiplicité aux bords (convention scipy)
    knots = np.concatenate([
        [xmin] * degree,        # répétition à gauche
        internal_knots,
        [xmax] * degree         # répétition à droite
    ])
    
    n_basis = len(knots) - degree - 1
    
    # Construction de la matrice de base (chaque colonne = une fonction de base)
    B = np.zeros((len(xv), n_basis))
    for i in range(n_basis):
        # Fonction de base i : tous les coeffs = 0 sauf le i-ème = 1
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        spline = BSpline(knots, coef, degree, extrapolate=False)
        B[:, i] = spline(xv)
    
    # Remplacer NaN par 0 (extrapolation hors support)
    B = np.nan_to_num(B, nan=0.0)
    
    return B, knots, n_basis


def eval_bspline_from_coef(coef, xv, knots, degree):
    """
    Évalue une courbe B-spline à partir de ses coefficients.
    Utilise scipy.interpolate.BSpline.
    
    Paramètres
    ----------
    coef   : array  coefficients de la B-spline
    xv     : array  points d'évaluation
    knots  : array  vecteur de nœuds complet
    degree : int    degré
    
    Retourne
    --------
    y : array   courbe évaluée aux points xv
    """
    spline = BSpline(knots, coef, degree, extrapolate=False)
    y = spline(xv)
    return np.nan_to_num(y, nan=0.0)




def make_penalty_matrix(n_basis, diff_order=2):
    """
    Construit la matrice de pénalité P-splines D^T D.
    
    Retourne
    --------
    DtD      : (n_basis, n_basis)
    diag_DtD : (n_basis,)   diagonale
    """
    D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    DtD = D.T @ D
    return DtD, np.diag(DtD)


# =============================================================================
# 2. RECONSTRUCTION ln(µ) — MODÈLE LI-LEE
# =============================================================================

def compute_logmu_lilee(
    alpha_coef,  # (nb_regions, n_basis)  α_{x,g}
    beta_coef,   # (n_basis,)             β_x commun
    beta_g_coef, # (nb_regions, n_basis)  β_{x,g}
    kappa,       # (nb_years,)            κ_t commun
    kappa_g,     # (nb_regions, nb_years) κ_{g,t}
    xv, B, knots, degree
):
    """
    Reconstruit ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}
    
    Paramètres
    ----------
    alpha_coef  : (nb_regions, n_basis)  coefficients α_{x,g}
    beta_coef   : (n_basis,)             coefficients β_x
    beta_g_coef : (nb_regions, n_basis)  coefficients β_{x,g}
    kappa       : (nb_years,)            κ_t commun
    kappa_g     : (nb_regions, nb_years) κ_{g,t} régionaux
    
    Retourne
    --------
    logmu    : (nb_ages, nb_years, nb_regions)
    alpha    : (nb_ages, nb_regions)  courbes α_{x,g}
    beta     : (nb_ages,)             courbe β_x
    beta_g   : (nb_ages, nb_regions)  courbes β_{x,g}
    """
    nb_regions = alpha_coef.shape[0]
    nb_ages = len(xv)
    
    # Évaluation des courbes B-splines
    alpha = np.zeros((nb_ages, nb_regions))
    beta_g = np.zeros((nb_ages, nb_regions))
    for g in range(nb_regions):
        alpha[:, g] = B @ alpha_coef[g]
        beta_g[:, g] = B @ beta_g_coef[g]
    
    beta = B @ beta_coef
    
    # Construction de ln(µ) par broadcast
    # Dimensions : (nb_ages, nb_years, nb_regions)
    logmu = (
        alpha[:, None, :]                                # (nb_ages, 1, nb_regions)
        + beta[:, None, None] * kappa[None, :, None]     # β_x · κ_t
        + beta_g[:, None, :] * kappa_g.T[None, :, :]     # β_{x,g} · κ_{g,t}
    )
    
    return logmu, alpha, beta, beta_g


# =============================================================================
# 3. LOG-VRAISEMBLANCE POISSON
# =============================================================================

def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
    """Log-vraisemblance de Poisson."""
    exp_logmu = np.exp(logmu)
    weighted_exp = Extg * exp_logmu
    residual = Dxtg - weighted_exp
    lnL = float(np.sum(
        Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
    ))
    return lnL, exp_logmu, weighted_exp, residual


# =============================================================================
# 4. MISES À JOUR NR — MODÈLE LI-LEE
# =============================================================================

def update_alpha_coef(alpha_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """
    Mise à jour NR de α_{x,g} (baseline régional).
    Un vecteur de coefficients par région.
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
    Mise à jour NR de β_x (sensibilité commune).
    Un seul vecteur de coefficients pour toutes les régions.
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
    Mise à jour NR de β_{x,g} (sensibilité régionale).
    Un vecteur de coefficients par région.
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
    Mise à jour NR de κ_t (facteur temporel commun).
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
    Mise à jour NR de κ_{g,t} (facteurs temporels régionaux).
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
# 5. NORMALISATION (CONTRAINTES D'IDENTIFIABILITÉ)
# =============================================================================

def normalize_lilee(beta_coef, beta_g_coef, kappa, kappa_g, B):
    """
    Contraintes d'identifiabilité Li-Lee :
      1. Σ_x β_x = 1
      2. Σ_x β_{x,g} = 0  pour tout g
      3. Σ_t κ_{g,t} = 0  pour tout g
    
    Ces contraintes garantissent l'unicité de la solution.
    """
    # 1. Normalisation de β_x : Σ_x β_x = 1
    beta = B @ beta_coef
    scal_beta = float(np.sum(beta))
    if scal_beta != 0:
        beta_coef = beta_coef / scal_beta
        kappa = kappa * scal_beta
    
    # 2. Normalisation de β_{x,g} : Σ_x β_{x,g} = 0 pour tout g
    nb_regions = beta_g_coef.shape[0]
    for g in range(nb_regions):
        beta_g = B @ beta_g_coef[g]
        sum_beta_g = float(np.sum(beta_g))
        if sum_beta_g != 0:
            # Soustraire la moyenne pour centrer à 0
            adjustment = sum_beta_g / len(beta_g)
            beta_g_coef[g] -= adjustment / np.mean(B.sum(axis=0))
    
    # 3. Normalisation de κ_{g,t} : Σ_t κ_{g,t} = 0 pour tout g
    for g in range(nb_regions):
        mean_kappa_g = float(np.mean(kappa_g[g]))
        kappa_g[g] -= mean_kappa_g
    
    return beta_coef, beta_g_coef, kappa, kappa_g


# =============================================================================
# 6. STATISTIQUES FIT
# =============================================================================

def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, n_basis, nb_years, nb_regions):
    """
    Calcule déviance, AIC, BIC pour le modèle Li-Lee.
    
    Degrés de liberté :
      dofs = n_basis × nb_regions      (α_{x,g})
           + n_basis                    (β_x)
           + n_basis × nb_regions       (β_{x,g})
           + nb_years                   (κ_t)
           + nb_years × nb_regions      (κ_{g,t})
           - nb_regions                 (contraintes Σ_x β_{x,g} = 0)
           - nb_regions                 (contraintes Σ_t κ_{g,t} = 0)
           - 1                          (contrainte Σ_x β_x = 1)
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
# 7. FONCTION PRINCIPALE — MODÈLE LI-LEE
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
    tol=1e-6,
    verbose=False,
):
    """
    Calibre le modèle Li-Lee paramétrique :
        ln(µ_{x,t,g}) = α_{x,g} + β_x·κ_t + β_{x,g}·κ_{g,t}
    
    UTILISE UNIQUEMENT scipy/numpy :
      ✓ scipy.interpolate.BSpline
      ✓ scipy.special.gammaln
      ✓ numpy.diff
    
    Paramètres
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
    degree           : int   degré B-splines (3 recommandé)
    n_knots          : int   nombre de nœuds internes (6 recommandé)
    lam              : float poids pénalité P-splines
    diff_order       : int   ordre différences (2 recommandé)
    nb_iter          : int   nombre max d'itérations
    eta0             : float learning rate initial (0.30 recommandé)
    tol              : float tolérance convergence
    verbose          : bool  afficher progression
    
    Retourne
    --------
    alpha_coef  : (nb_regions, n_basis)  coefficients α_{x,g}
    beta_coef   : (n_basis,)             coefficients β_x
    beta_g_coef : (nb_regions, n_basis)  coefficients β_{x,g}
    kappa       : (nb_years,)            κ_t
    kappa_g     : (nb_regions, nb_years) κ_{g,t}
    alpha       : (nb_ages, nb_regions)  courbes α_{x,g}
    beta        : (nb_ages,)             courbe β_x
    beta_g      : (nb_ages, nb_regions)  courbes β_{x,g}
    logmu_final : ln(µ_{x,t,g}) 
    Fit_stat    : pd.DataFrame           statistiques
    """
    nb_years = len(tv)
    nb_regions = Extg.shape[2]

    if xmin is None:
        xmin = float(np.min(xv))
    if xmax is None:
        xmax = float(np.max(xv))

    # Construction de la matrice de base B-spline
    B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)

    # Vérification des dimensions
    if alpha_coef_init.shape != (nb_regions, n_basis):
        raise ValueError(f"alpha_coef_init doit avoir shape ({nb_regions}, {n_basis})")
    if len(beta_coef_init) != n_basis:
        raise ValueError(f"beta_coef_init doit avoir {n_basis} éléments")
    if beta_g_coef_init.shape != (nb_regions, n_basis):
        raise ValueError(f"beta_g_coef_init doit avoir shape ({nb_regions}, {n_basis})")

    alpha_coef = alpha_coef_init.copy()
    beta_coef = beta_coef_init.copy()
    beta_g_coef = beta_g_coef_init.copy()
    kappa = kappa_init.copy()
    kappa_g = kappa_g_init.copy()

    # Pénalité P-splines
    DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)

    # Constante log-factorielle
    logDxtgFact = gammaln(Dxtg + 1)

    lnL = 0.0
    Delta_lnL = -1000.0
    flag = 0
    it = -1
    eta = eta0

    if verbose:
        print("=" * 70)
        print("CALIBRATION MODÈLE LI-LEE PARAMÉTRIQUE")
        print("=" * 70)
        print(f"Paramètres : degree={degree}, n_knots={n_knots}, lam={lam}")
        print(f"Données : {Dxtg.shape[0]} âges × {nb_years} années × {nb_regions} régions")
        print(f"Nombre de fonctions de base : {n_basis}")
        print("=" * 70)

    # Boucle NR
    while (it < nb_iter) and (flag < 4):
        it += 1

        # Learning rate adaptatif
        if Delta_lnL < 0:
            eta *= 0.5
        else:
            eta = min(eta * 1.05, 2.0)

        # Critère d'arrêt
        if np.abs(Delta_lnL) < tol:
            flag += 1
        else:
            flag = 0

        # Reconstruction ln(µ)
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

        # Log-vraisemblance
        lnL_new, _, weighted_exp, residual = poisson_lnL(
            Dxtg, Extg, logmu, logDxtgFact
        )

        Delta_lnL = lnL_new - lnL
        lnL = lnL_new

        if verbose and (it % 10 == 0):
            print(f"It {it:4d} | lnL = {lnL:,.2f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

        # Mises à jour NR séquentielles
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

    # Normalisation finale
    beta_coef, beta_g_coef, kappa, kappa_g = normalize_lilee(
        beta_coef, beta_g_coef, kappa, kappa_g, B
    )

    # Reconstruction finale
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

    # Statistiques
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
        print("CALIBRATION TERMINÉE")
        print("=" * 70)
        print(f"Convergence atteinte après {it + 1} itérations")
        print("\nStatistiques finales :")
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
























