# -*- coding: utf-8 -*-
"""
LCp_fit — Utilise uniquement les bibliothèques Python standard
===============================================================
Modèle Lee-Carter paramétrique B-splines :
    ln(µ_{x,t,g}) = α_x  +  β_{x,g} · κ_t

UTILISE LES BIBLIOTHÈQUES PYTHON STANDARD :
  ✓ scipy.interpolate.BSpline      — construction B-splines
  ✓ scipy.interpolate.make_lsq_spline — fit par moindres carrés
  ✓ scipy.special.gammaln          — log-factorielle
  ✓ scipy.linalg                   — algèbre linéaire
  ✓ numpy                          — calcul matriciel

PAS de réimplémentation manuelle — tout vient de scipy/numpy.

Référence scipy B-splines :
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
"""
import sys, os
sys.path.append(os.path.abspath(".."))
import geopandas as gpd
import sys
sys.executable
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.special import gammaln
from scipy.linalg import lstsq

from morta_nuts2.model.Bsplines.Bsplines import make_bspline_basis, eval_bspline_from_coef

#from morta_nuts2.model.Bsplines.Bsplines import*

# =============================================================================
# 1. CONSTRUCTION DES B-SPLINES — scipy.interpolate.BSpline
# =============================================================================

# def make_bspline_basis(xv, degree, n_knots, xmin=None, xmax=None):
#     """
#     Construit la matrice de base B-spline avec scipy.interpolate.BSpline.
    
#     Paramètres
#     ----------
#     xv       : array   points d'évaluation (ex: âges 0 à 82)
#     degree   : int     degré des B-splines (3 recommandé)
#     n_knots  : int     nombre de nœuds internes (équivalent à m+1 du code source)
#     xmin     : float   borne inférieure (défaut: min(xv))
#     xmax     : float   borne supérieure (défaut: max(xv))
    
#     Retourne
#     --------
#     B      : (len(xv), n_basis)   matrice de base B-spline
#     knots  : array                vecteur de nœuds complet (avec multiplicité aux bords)
#     n_basis: int                  nombre de fonctions de base
    
#     Notes
#     -----
#     scipy.interpolate.BSpline utilise la convention :
#       - nœuds internes : np.linspace(xmin, xmax, n_knots)
#       - nœuds complets : répétition degree+1 fois aux bords
#       - n_basis = n_knots + degree - 1
#     """
#     if xmin is None:
#         xmin = float(np.min(xv))
#     if xmax is None:
#         xmax = float(np.max(xv))
    
#     # Nœuds internes équidistants
#     internal_knots = np.linspace(xmin, xmax, n_knots)
    
#     # Nœuds complets avec multiplicité aux bords (convention scipy)
#     knots = np.concatenate([
#         [xmin] * degree,        # répétition à gauche
#         internal_knots,
#         [xmax] * degree         # répétition à droite
#     ])
    
#     n_basis = len(knots) - degree - 1
    
#     # Construction de la matrice de base (chaque colonne = une fonction de base)
#     B = np.zeros((len(xv), n_basis))
#     for i in range(n_basis):
#         # Fonction de base i : tous les coeffs = 0 sauf le i-ème = 1
#         coef = np.zeros(n_basis)
#         coef[i] = 1.0
#         spline = BSpline(knots, coef, degree, extrapolate=False)
#         B[:, i] = spline(xv)
    
#     # Remplacer NaN par 0 (extrapolation hors support)
#     B = np.nan_to_num(B, nan=0.0)
    
#     return B, knots, n_basis


# def eval_bspline_from_coef(coef, xv, knots, degree):
#     """
#     Évalue une courbe B-spline à partir de ses coefficients.
#     Utilise scipy.interpolate.BSpline.
    
#     Paramètres
#     ----------
#     coef   : array  coefficients de la B-spline
#     xv     : array  points d'évaluation
#     knots  : array  vecteur de nœuds complet
#     degree : int    degré
    
#     Retourne
#     --------
#     y : array   courbe évaluée aux points xv
#     """
#     spline = BSpline(knots, coef, degree, extrapolate=False)
#     y = spline(xv)
#     return np.nan_to_num(y, nan=0.0)




# =============================================================================
# 2. MATRICE DE PÉNALITÉ P-SPLINES — numpy
# =============================================================================

def make_penalty_matrix(n_basis, diff_order=2):
    """
    Construit la matrice de pénalité P-splines D^T D.
    Utilise numpy pour construire la matrice de différences finies.
    
    P(c) = λ · c^T (D^T D) c
    
    Paramètres
    ----------
    n_basis    : int   nombre de fonctions de base
    diff_order : int   ordre des différences (1 = dérivée 1ère, 2 = dérivée 2nde)
    
    Retourne
    --------
    DtD      : (n_basis, n_basis)
    diag_DtD : (n_basis,)   diagonale (pour approx. Hessien)
    """
    # Matrice de différences d'ordre diff_order
    D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    DtD = D.T @ D
    return DtD, np.diag(DtD)


# =============================================================================
# 3. RECONSTRUCTION DE ln(µ) — numpy + scipy.interpolate
# =============================================================================

def compute_logmu(ax_coef, bx_coef, kappa, xv, B, knots, degree):
    """
    Reconstruit ln(µ_{x,t,g}) = α_x + β_{x,g} · κ_t
    en utilisant scipy.interpolate.BSpline.
    
    Paramètres
    ----------
    ax_coef  : (n_basis,)            coefficients B-spline de α_x
    bx_coef  : (nb_regions, n_basis) coefficients B-spline de β_{x,g}
    kappa    : (nb_years,)           facteur endogène κ_t
    xv       : (nb_ages,)            vecteur des âges
    B        : (nb_ages, n_basis)    matrice de base (précalculée)
    knots    : array                 nœuds complets
    degree   : int
    
    Retourne
    --------
    logmu  : (nb_ages, nb_years, nb_regions)
    ax     : (nb_ages,)            courbe α_x évaluée
    bx_reg : (nb_ages, nb_regions) courbes β_{x,g} évaluées
    """
    nb_regions = bx_coef.shape[0]
    nb_ages = len(xv)
    
    # Évaluation de α_x avec scipy.interpolate.BSpline
    ax = B @ ax_coef  # équivalent à eval_bspline_from_coef(ax_coef, xv, knots, degree)
    
    # Évaluation de β_{x,g} pour chaque région
    bx_reg = np.zeros((nb_ages, nb_regions))
    for g in range(nb_regions):
        bx_reg[:, g] = B @ bx_coef[g]
    
    # Construction de ln(µ) par broadcast
    logmu = (
        ax[:, None, None]
        + bx_reg[:, None, :] * kappa[None, :, None]
    )
    return logmu, ax, bx_reg


# =============================================================================
# 4. LOG-VRAISEMBLANCE DE POISSON — scipy.special.gammaln
# =============================================================================

def poisson_lnL(Dxtg, Extg, logmu, logDxtgFact):
    """
    Log-vraisemblance de Poisson (équation 2 du papier).
    Utilise scipy.special.gammaln pour log(D!).
    """
    exp_logmu = np.exp(logmu)
    weighted_exp = Extg * exp_logmu
    residual = Dxtg - weighted_exp
    lnL = float(np.sum(
        Dxtg * logmu - weighted_exp + Dxtg * np.log(Extg) - logDxtgFact
    ))
    return lnL, exp_logmu, weighted_exp, residual


# =============================================================================
# 5. MISES À JOUR NR — numpy
# =============================================================================

def update_ax_coef(ax_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """Mise à jour NR des coefficients B-splines de α_x."""
    n_basis = len(ax_coef)
    ax_coef_new = ax_coef.copy()
    pen_grad = 2.0 * lam * (DtD @ ax_coef) if lam > 0 else np.zeros(n_basis)
    
    for j in range(n_basis):
        Bj3d = B[:, j][:, None, None]
        num = float(np.sum(residual * Bj3d)) - pen_grad[j]
        den = float(np.sum(weighted_exp * Bj3d**2))
        if lam > 0:
            den += 2.0 * lam * diag_DtD[j]
        if den != 0:
            ax_coef_new[j] += eta * num / den
    return ax_coef_new


def update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD):
    """Mise à jour NR des coefficients B-splines de β_{x,g}."""
    nb_regions = bx_coef.shape[0]
    nb_ages = B.shape[0]
    bx_coef_new = bx_coef.copy()
    kappaM = np.repeat(kappa[None, :], nb_ages, axis=0)
    
    for g in range(nb_regions):
        pen_grad = 2.0 * lam * (DtD @ bx_coef[g]) if lam > 0 else np.zeros(bx_coef.shape[1])
        for j in range(bx_coef.shape[1]):
            BjKappa = B[:, j][:, None] * kappaM
            BjK3d = BjKappa[:, :, None]
            num = float(np.sum(residual[:, :, g:g+1] * BjK3d)) - pen_grad[j]
            den = float(np.sum(weighted_exp[:, :, g:g+1] * BjK3d**2))
            if lam > 0:
                den += 2.0 * lam * diag_DtD[j]
            if den != 0:
                bx_coef_new[g, j] += eta * num / den
    return bx_coef_new


def update_kappa(kappa, bx_reg, residual, weighted_exp, eta):
    """Mise à jour NR de κ_t."""
    bx3d = bx_reg[:, None, :]
    num_k = np.sum(residual * bx3d, axis=(0, 2))
    den_k = np.sum(weighted_exp * bx3d**2, axis=(0, 2))
    kappa_new = kappa.copy()
    mask = den_k != 0
    kappa_new[mask] += eta * num_k[mask] / den_k[mask]
    return kappa_new


# =============================================================================
# 6. RESCALING FINAL — numpy
# =============================================================================

def rescale_bx_kappa(bx_coef, bx_reg, kappa):
    """Normalisation : Σ_x (moyenne_g β_{x,g}) = 1"""
    bx_avg = np.mean(bx_reg, axis=1)
    scal_factor = float(np.sum(bx_avg))
    if scal_factor == 0:
        return bx_coef, bx_reg, kappa
    return bx_coef / scal_factor, bx_reg / scal_factor, kappa * scal_factor


# =============================================================================
# 7. STATISTIQUES DE FIT — numpy
# =============================================================================

def compute_fit_stats(Dxtg, Extg, logmu, logDxtgFact, n_basis, nb_years, nb_regions):
    """Calcule déviance Poisson, AIC, BIC."""
    exp_logmu = np.exp(logmu)
    lnL = float(np.sum(
        Dxtg * logmu - Extg * exp_logmu + Dxtg * np.log(Extg) - logDxtgFact
    ))
    
    # Déviance Poisson
    safe_Dxtg = np.where(Dxtg > 0, Dxtg, 1.0)
    lnL_sat = float(np.sum(np.where(
        Dxtg > 0,
        Dxtg * np.log(safe_Dxtg / np.maximum(Extg, 1e-12)) - Dxtg,
        0.0
    )))
    deviance = 2.0 * (lnL_sat - lnL)
    
    nb_obs = int(Dxtg.size)
    dofs = n_basis + n_basis * nb_regions + nb_years
    AIC = 2.0 * dofs - 2.0 * lnL
    BIC = dofs * np.log(nb_obs) - 2.0 * lnL
    
    return pd.DataFrame(
        [[nb_obs, n_basis, dofs,
          round(lnL, 2), round(deviance, 2), round(AIC, 2), round(BIC, 2)]],
        columns=["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]
    )


# =============================================================================
# 8. FONCTION PRINCIPALE
# =============================================================================

# def LCp_fit(
#     ax_coef_init,
#     bx_coef_init,
#     kappa_init,
#     Extg,
#     Dxtg,
#     xv,
#     tv,
#     degree=2,
#     n_knots=10,       # nombre de nœuds internes (équivalent à m+1)
#     xmin=None,
#     xmax=None,
#     lam=0.0,
#     diff_order=2,
#     nb_iter=800,
#     eta0=0.2,
#     tol=1e-6,
#     verbose=False,
# ):
#     """
#     Calibre le modèle Lee-Carter paramétrique :
#         ln(µ_{x,t,g}) = α_x  +  β_{x,g} · κ_t

#     UTILISE UNIQUEMENT LES BIBLIOTHÈQUES PYTHON STANDARD :
#       ✓ scipy.interpolate.BSpline
#       ✓ scipy.special.gammaln
#       ✓ numpy
    
#     Paramètres
#     ----------
#     ax_coef_init : (n_basis,)            init coefficients B-spline de α_x
#     bx_coef_init : (nb_regions, n_basis) init coefficients B-spline de β_{x,g}
#     kappa_init   : (nb_years,)           init κ_t
#     Extg         : (nb_ages, nb_years, nb_regions)  expositions
#     Dxtg         : (nb_ages, nb_years, nb_regions)  décès
#     xv           : (nb_ages,)            vecteur des âges
#     tv           : (nb_years,)           vecteur des années
#     degree       : int   degré B-splines (défaut: 3)
#     n_knots      : int   nombre de nœuds internes (défaut: 6)
#     xmin, xmax   : float bornes (défaut: min/max de xv)
#     lam          : float poids pénalité P-splines (0 = B-splines purs)
#     diff_order   : int   ordre différences (1 ou 2)
    
#     Retourne
#     --------
#     ax_coef     : (n_basis,)            coefficients B-splines de α_x
#     bx_coef     : (nb_regions, n_basis) coefficients B-splines de β_{x,g}
#     kappa       : (nb_years,)           κ_t calibré
#     ax          : (nb_ages,)            courbe α_x évaluée
#     bx          : (nb_ages, nb_regions) courbes β_{x,g} évaluées
#     logmu_final : ln(µ_{x,t,g})
#     Fit_stat    : pd.DataFrame          statistiques
#     """
#     nb_years = len(tv)
#     nb_regions = Extg.shape[2]
    
#     if xmin is None:
#         xmin = float(np.min(xv))
#     if xmax is None:
#         xmax = float(np.max(xv))
    
#     # Construction de la matrice de base B-spline avec scipy
#     B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)
    
#     # Vérification des dimensions d'initialisation
#     if len(ax_coef_init) != n_basis:
#         raise ValueError(f"ax_coef_init doit avoir {n_basis} éléments, reçu {len(ax_coef_init)}")
#     if bx_coef_init.shape[1] != n_basis:
#         raise ValueError(f"bx_coef_init doit avoir {n_basis} colonnes, reçu {bx_coef_init.shape[1]}")
    
#     ax_coef = ax_coef_init.copy()
#     bx_coef = bx_coef_init.copy()
#     kappa = kappa_init.copy()
    
#     # Pénalité P-splines
#     DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)
    
#     # Constante log-factorielle
#     logDxtgFact = gammaln(Dxtg + 1)
    
#     lnL = 0.0
#     Delta_lnL = -1000.0
#     flag = 0
#     it = -1
#     eta = eta0
    
#     # Boucle NR
#     while (it < nb_iter) and (flag < 4):
#         it += 1
        
#         # Learning rate adaptatif
#         if Delta_lnL < 0:
#             eta *= 0.5
#         else:
#             eta = min(eta * 1.05, 2.0)
        
#         # Critère d'arrêt
#         if np.abs(Delta_lnL) < tol:
#             flag += 1
#         else:
#             flag = 0
        
#         # Reconstruction ln(µ) avec scipy.interpolate.BSpline
#         logmu, ax, bx_reg = compute_logmu(
#             ax_coef, bx_coef, kappa, xv, B, knots, degree
#         )
        
#         # Log-vraisemblance
#         lnL_new, _, weighted_exp, residual = poisson_lnL(
#             Dxtg, Extg, logmu, logDxtgFact
#         )
#         Delta_lnL = lnL_new - lnL
#         lnL = lnL_new
        
#         if verbose and (it % 10 == 0):
#             print(f"It {it:4d} | lnL = {lnL:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")
        
#         # Mises à jour NR
#         ax_coef = update_ax_coef(ax_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD)
#         bx_coef = update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD)
#         kappa = update_kappa(kappa, bx_reg, residual, weighted_exp, eta)
    
#     # Rescaling final
#     bx_coef, bx_reg, kappa = rescale_bx_kappa(bx_coef, bx_reg, kappa)

#     # Reconstruction finale
#     logmu_final, ax, bx_reg = compute_logmu(
#         ax_coef, bx_coef, kappa, xv, B, knots, degree
#     )
    
#     # Statistiques
#     Fit_stat = compute_fit_stats(
#         Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions
#     )
    
#     if verbose:
#         print("\n" + "="*70)
#         print("STATISTIQUES FINALES (scipy/numpy uniquement)")
#         print("="*70)
#         print(Fit_stat.to_string(index=False))


#     results = {
    
#     "parameters": {
#         "ax_coef": ax_coef,
#         "bx_coef": bx_coef,
#         "kappa": kappa
#     },
    
#     "curves": {
#         "alpha_x": ax,
#         "beta_xg": bx_reg
#     },
    
#     "fitted_values": {
#         "log_mu": logmu_final,
#         "mu": np.exp(logmu_final)
#     },
    
#     "fit_statistics": Fit_stat
#     }


 
    
#     return results


def LCp_multiregion_fit(
    ax_coef_init,
    bx_coef_init,
    kappa_init,
    Extg,
    Dxtg,
    xv,
    tv,
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
    nb_years = len(tv)
    nb_regions = Extg.shape[2]
    
    if xmin is None:
        xmin = float(np.min(xv))
    if xmax is None:
        xmax = float(np.max(xv))
    
    B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)
    
    if len(ax_coef_init) != n_basis:
        raise ValueError(f"ax_coef_init doit avoir {n_basis} éléments, reçu {len(ax_coef_init)}")
    if bx_coef_init.shape[1] != n_basis:
        raise ValueError(f"bx_coef_init doit avoir {n_basis} colonnes, reçu {bx_coef_init.shape[1]}")
    
    ax_coef = ax_coef_init.copy()
    bx_coef = bx_coef_init.copy()
    kappa = kappa_init.copy()
    
    DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)
    logDxtgFact = gammaln(Dxtg + 1)
    
    lnL = -np.inf
    Delta_lnL = 0.0
    eta = eta0
    it = -1
    
    # ===============================
    # 🔵 Early stopping type patience
    # ===============================
    best_lnL = -np.inf
    patience = 40           # nombre max d'itérations sans amélioration
    min_delta = 1e-2        # amélioration minimale significative
    wait = 0
    
    while it < nb_iter:
        it += 1
        
        # Learning rate adaptatif
        if Delta_lnL < 0:
            eta *= 0.5
        else:
            eta = min(eta * 1.05, 2.0)
        
        # Reconstruction
        logmu, ax, bx_reg = compute_logmu(
            ax_coef, bx_coef, kappa, xv, B, knots, degree
        )
        
        # Log-vraisemblance
        lnL_new, _, weighted_exp, residual = poisson_lnL(
            Dxtg, Extg, logmu, logDxtgFact
        )
        
        Delta_lnL = lnL_new - lnL
        
        if verbose and (it % 10 == 0):
            print(f"It {it:4d} | lnL = {lnL_new:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")
        
        # ===============================
        # 🔵 Early stopping logique
        # ===============================
        if lnL_new > best_lnL + min_delta:
            best_lnL = lnL_new
            wait = 0
        else:
            wait += 1
        
        if wait >= patience:
            if verbose:
                print("\nArrêt anticipé : plus d'amélioration significative.")
            break
        
        # Critère d'arrêt classique
        if abs(Delta_lnL) < tol:
            if verbose:
                print("\nConvergence atteinte (tolérance).")
            break
        
        lnL = lnL_new
        
        # Updates
        ax_coef = update_ax_coef(ax_coef, B, residual, weighted_exp, eta, lam, DtD, diag_DtD)
        bx_coef = update_bx_coef(bx_coef, B, kappa, residual, weighted_exp, eta, lam, DtD, diag_DtD)
        kappa = update_kappa(kappa, bx_reg, residual, weighted_exp, eta)
    
    # ===============================
    # Rescaling final
    # ===============================
    bx_coef, bx_reg, kappa = rescale_bx_kappa(bx_coef, bx_reg, kappa)

    logmu_final, ax, bx_reg = compute_logmu(
        ax_coef, bx_coef, kappa, xv, B, knots, degree
    )
    
    Fit_stat = compute_fit_stats(
        Dxtg, Extg, logmu_final, logDxtgFact, n_basis, nb_years, nb_regions
    )
    
    if verbose:
        print("\n" + "="*70)
        print("STATISTIQUES FINALES")
        print("="*70)
        print(Fit_stat.to_string(index=False))

    results = {
        "parameters": {
            "ax_coef": ax_coef,
            "bx_coef": bx_coef,
            "kappa": kappa
        },
        "curves": {
            "alpha_x": ax,
            "beta_xg": bx_reg
        },
        "fitted_values": {
            "log_mu": logmu_final,
            "mu": np.exp(logmu_final)
        },
        "fit_statistics": Fit_stat
    }

    return results



# import numpy as np
# import pandas as pd
# from scipy.special import gammaln
# from scipy.interpolate import BSpline


# # =============================================================================
# # 1. BASE B-SPLINE
# # =============================================================================

# def make_bspline_basis(x, degree=2, n_knots=10, xmin=None, xmax=None):
#     if xmin is None:
#         xmin = float(np.min(x))
#     if xmax is None:
#         xmax = float(np.max(x))

#     knots_internal = np.linspace(xmin, xmax, n_knots)
#     knots = np.concatenate((
#         np.repeat(xmin, degree),
#         knots_internal,
#         np.repeat(xmax, degree)
#     ))

#     n_basis = len(knots) - degree - 1

#     B = np.zeros((len(x), n_basis))
#     for i in range(n_basis):
#         coef = np.zeros(n_basis)
#         coef[i] = 1.0
#         spline = BSpline(knots, coef, degree)
#         B[:, i] = spline(x)

#     return B, knots, n_basis


# # =============================================================================
# # 2. MATRICE DE PÉNALITÉ
# # =============================================================================

# def make_penalty_matrix(n_basis, diff_order=2):
#     D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
#     DtD = D.T @ D
#     return DtD, np.diag(DtD)


# # =============================================================================
# # 3. RECONSTRUCTION ln(μ)
# # =============================================================================

# def compute_logmu(ax_coef, bx_coef, kappa, B):
#     ax = B @ ax_coef
#     bx = B @ bx_coef

#     logmu = ax[:, None] + bx[:, None] * kappa[None, :]
#     return logmu, ax, bx


# # =============================================================================
# # 4. LOG-VRAISEMBLANCE POISSON
# # =============================================================================

# def poisson_lnL(Dxt, Ext, logmu, logDxtFact):
#     exp_logmu = np.exp(logmu)
#     weighted_exp = Ext * exp_logmu
#     residual = Dxt - weighted_exp

#     lnL = float(np.sum(
#         Dxt * logmu
#         - weighted_exp
#         + Dxt * np.log(np.maximum(Ext, 1e-12))
#         - logDxtFact
#     ))

#     return lnL, exp_logmu, weighted_exp, residual


# # =============================================================================
# # 5. UPDATE α
# # =============================================================================

# def update_ax_coef(ax_coef, B, residual, weighted_exp,
#                    eta, lam, DtD, diag_DtD):

#     n_basis = len(ax_coef)
#     ax_new = ax_coef.copy()

#     pen_grad = 2.0 * lam * (DtD @ ax_coef) if lam > 0 else np.zeros(n_basis)

#     for j in range(n_basis):
#         Bj = B[:, j][:, None]

#         num = float(np.sum(residual * Bj)) - pen_grad[j]
#         den = float(np.sum(weighted_exp * Bj**2))

#         if lam > 0:
#             den += 2.0 * lam * diag_DtD[j]

#         if den != 0:
#             ax_new[j] += eta * num / den

#     return ax_new


# # =============================================================================
# # 6. UPDATE β
# # =============================================================================

# def update_bx_coef(bx_coef, B, kappa, residual, weighted_exp,
#                    eta, lam, DtD, diag_DtD):

#     n_basis = len(bx_coef)
#     bx_new = bx_coef.copy()

#     nb_ages = B.shape[0]
#     kappaM = np.repeat(kappa[None, :], nb_ages, axis=0)

#     pen_grad = 2.0 * lam * (DtD @ bx_coef) if lam > 0 else np.zeros(n_basis)

#     for j in range(n_basis):
#         BjK = B[:, j][:, None] * kappaM

#         num = float(np.sum(residual * BjK)) - pen_grad[j]
#         den = float(np.sum(weighted_exp * BjK**2))

#         if lam > 0:
#             den += 2.0 * lam * diag_DtD[j]

#         if den != 0:
#             bx_new[j] += eta * num / den

#     return bx_new


# # =============================================================================
# # 7. UPDATE κ
# # =============================================================================

# def update_kappa(kappa, bx, residual, weighted_exp, eta):

#     bx2d = bx[:, None]

#     num_k = np.sum(residual * bx2d, axis=0)
#     den_k = np.sum(weighted_exp * bx2d**2, axis=0)

#     kappa_new = kappa.copy()
#     mask = den_k != 0
#     kappa_new[mask] += eta * num_k[mask] / den_k[mask]

#     return kappa_new


# # =============================================================================
# # 8. RESCALING (Σ β_x = 1)
# # =============================================================================

# def rescale_bx_kappa(bx_coef, bx, kappa):
#     scale = float(np.sum(bx))
#     if scale == 0:
#         return bx_coef, bx, kappa

#     return bx_coef / scale, bx / scale, kappa * scale


# # =============================================================================
# # 9. STATISTIQUES
# # =============================================================================

# def compute_fit_stats(Dxt, Ext, logmu, logDxtFact,
#                       n_basis, nb_years):

#     exp_logmu = np.exp(logmu)

#     lnL = float(np.sum(
#         Dxt * logmu
#         - Ext * exp_logmu
#         + Dxt * np.log(np.maximum(Ext, 1e-12))
#         - logDxtFact
#     ))

#     safe_Dxt = np.where(Dxt > 0, Dxt, 1.0)

#     lnL_sat = float(np.sum(np.where(
#         Dxt > 0,
#         Dxt * np.log(safe_Dxt / np.maximum(Ext, 1e-12)) - Dxt,
#         0.0
#     )))

#     deviance = 2.0 * (lnL_sat - lnL)

#     nb_obs = int(Dxt.size)
#     dofs = 2 * n_basis + nb_years

#     AIC = 2.0 * dofs - 2.0 * lnL
#     BIC = dofs * np.log(nb_obs) - 2.0 * lnL

#     return pd.DataFrame(
#         [[nb_obs, n_basis, dofs,
#           round(lnL, 2), round(deviance, 2),
#           round(AIC, 2), round(BIC, 2)]],
#         columns=["N", "n_basis", "dofs",
#                  "lnL", "deviance", "AIC", "BIC"]
#     )


# # =============================================================================
# # 10. FIT COMPLET
# # =============================================================================

# def LCp_fit(
#     ax_coef_init,
#     bx_coef_init,
#     kappa_init,
#     Ext,
#     Dxt,
#     xv,
#     tv,
#     degree=2,
#     n_knots=10,
#     lam=0.0,
#     diff_order=2,
#     nb_iter=800,
#     eta0=0.2,
#     tol=1e-3,
#     verbose=False,
# ):

#     nb_years = len(tv)

#     B, knots, n_basis = make_bspline_basis(
#         xv, degree, n_knots
#     )

#     ax_coef = ax_coef_init.copy()
#     bx_coef = bx_coef_init.copy()
#     kappa = kappa_init.copy()

#     DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)
#     logDxtFact = gammaln(Dxt + 1)

#     lnL = -np.inf
#     Delta_lnL = 0.0
#     eta = eta0
#     it = -1

#     best_lnL = -np.inf
#     patience = 40
#     min_delta = 1e-2
#     wait = 0

#     while it < nb_iter:
#         it += 1

#         if Delta_lnL < 0:
#             eta *= 0.5
#         else:
#             eta = min(eta * 1.05, 2.0)

#         logmu, ax, bx = compute_logmu(
#             ax_coef, bx_coef, kappa, B
#         )

#         lnL_new, _, weighted_exp, residual = poisson_lnL(
#             Dxt, Ext, logmu, logDxtFact
#         )

#         Delta_lnL = lnL_new - lnL

#         if verbose and it % 10 == 0:
#             print(f"It {it:4d} | lnL={lnL_new:.4f} "
#                   f"| Δ={Delta_lnL:+.6f} | η={eta:.4f}")

#         if lnL_new > best_lnL + min_delta:
#             best_lnL = lnL_new
#             wait = 0
#         else:
#             wait += 1

#         if wait >= patience or abs(Delta_lnL) < tol:
#             break

#         lnL = lnL_new

#         ax_coef = update_ax_coef(
#             ax_coef, B, residual, weighted_exp,
#             eta, lam, DtD, diag_DtD
#         )

#         bx_coef = update_bx_coef(
#             bx_coef, B, kappa, residual, weighted_exp,
#             eta, lam, DtD, diag_DtD
#         )

#         kappa = update_kappa(
#             kappa, bx, residual, weighted_exp, eta
#         )

#     bx_coef, bx, kappa = rescale_bx_kappa(
#         bx_coef, bx, kappa
#     )

#     logmu_final, ax, bx = compute_logmu(
#         ax_coef, bx_coef, kappa, B
#     )

#     Fit_stat = compute_fit_stats(
#         Dxt, Ext, logmu_final,
#         logDxtFact, n_basis, nb_years
#     )

#     results = {
#         "parameters": {
#             "ax_coef": ax_coef,
#             "bx_coef": bx_coef,
#             "kappa": kappa
#         },
#         "curves": {
#             "alpha_x": ax,
#             "beta_x": bx
#         },
#         "fitted_values": {
#             "log_mu": logmu_final,
#             "mu": np.exp(logmu_final)
#         },
#         "fit_statistics": Fit_stat
#     }

#     return results



# def LCp_fit_snd(
#     ax_coef_init,
#     bx_coef_init,
#     kappa_init,
#     Extg,
#     Dxtg,
#     xv,
#     tv,
#     degree=2,
#     n_knots=10,
#     xmin=None,
#     xmax=None,
#     lam=0.0,
#     diff_order=2,
#     nb_iter=800,
#     eta0=0.2,
#     tol=1e-6,
#     verbose=False,
# ):

#     nb_years = len(tv)
#     nb_regions = Extg.shape[2]

#     if xmin is None:
#         xmin = float(np.min(xv))
#     if xmax is None:
#         xmax = float(np.max(xv))

#     B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)

#     ax_coef = ax_coef_init.copy()
#     bx_coef = bx_coef_init.copy()
#     kappa = kappa_init.copy()

#     DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)
#     logDxtgFact = gammaln(Dxtg + 1)

#     eta = min(eta0, 1.0)
#     flag = 0
#     it = 0

#     # loglik initiale
#     logmu, ax, bx_reg = compute_logmu(
#         ax_coef, bx_coef, kappa, xv, B, knots, degree
#     )
#     lnL, _, _, _ = poisson_lnL(Dxtg, Extg, logmu, logDxtgFact)

#     while (it < nb_iter) and (flag < 4):

#         it += 1

#         # recalcul courant
#         logmu, ax, bx_reg = compute_logmu(
#             ax_coef, bx_coef, kappa, xv, B, knots, degree
#         )

#         lnL_current, _, weighted_exp, residual = poisson_lnL(
#             Dxtg, Extg, logmu, logDxtgFact
#         )

#         # tentative update
#         eta_try = eta
#         accepted = False

#         while not accepted:

#             ax_new = update_ax_coef(
#                 ax_coef, B, residual, weighted_exp,
#                 eta_try, lam, DtD, diag_DtD
#             )

#             bx_new = update_bx_coef(
#                 bx_coef, B, kappa, residual, weighted_exp,
#                 eta_try, lam, DtD, diag_DtD
#             )

#             kappa_new = update_kappa(
#                 kappa, bx_reg, residual, weighted_exp,
#                 eta_try
#             )

#             # --- Stabilisation forte κ ---
#             for _ in range(3):
#                 logmu_tmp, _, bx_reg_tmp = compute_logmu(
#                     ax_new, bx_new, kappa_new, xv, B, knots, degree
#                 )
#                 _, _, weighted_exp_tmp, residual_tmp = poisson_lnL(
#                     Dxtg, Extg, logmu_tmp, logDxtgFact
#                 )
#                 kappa_new = update_kappa(
#                     kappa_new, bx_reg_tmp, residual_tmp,
#                     weighted_exp_tmp, 0.5
#                 )

#             # centrage κ
#             kappa_new -= np.mean(kappa_new)

#             # renormalisation β / κ
#             logmu_tmp, _, bx_reg_tmp = compute_logmu(
#                 ax_new, bx_new, kappa_new, xv, B, knots, degree
#             )
#             bx_avg = np.mean(bx_reg_tmp, axis=1)
#             scale = np.sum(bx_avg)

#             if abs(scale) > 1e-10:
#                 bx_new /= scale
#                 kappa_new *= scale

#             # recalcul loglik candidate
#             logmu_new, _, _ = compute_logmu(
#                 ax_new, bx_new, kappa_new,
#                 xv, B, knots, degree
#             )

#             lnL_new, _, _, _ = poisson_lnL(
#                 Dxtg, Extg, logmu_new, logDxtgFact
#             )

#             if lnL_new >= lnL_current:
#                 accepted = True
#                 ax_coef = ax_new
#                 bx_coef = bx_new
#                 kappa = kappa_new
#                 lnL = lnL_new
#                 eta = min(eta_try * 1.05, 1.0)
#             else:
#                 eta_try *= 0.5
#                 if eta_try < 1e-8:
#                     accepted = True

#         Delta_lnL = lnL - lnL_current

#         if abs(Delta_lnL) < tol:
#             flag += 1
#         else:
#             flag = 0

#         if verbose and (it % 20 == 0):
#             print(f"It {it:4d} | lnL = {lnL:.6f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

#     # reconstruction finale
#     logmu_final, ax, bx_reg = compute_logmu(
#         ax_coef, bx_coef, kappa, xv, B, knots, degree
#     )

#     Fit_stat = compute_fit_stats(
#         Dxtg, Extg, logmu_final,
#         logDxtgFact, n_basis,
#         nb_years, nb_regions
#     )

#     return {
#         "parameters": {
#             "ax_coef": ax_coef,
#             "bx_coef": bx_coef,
#             "kappa": kappa
#         },
#         "curves": {
#             "alpha_x": ax,
#             "beta_xg": bx_reg
#         },
#         "fitted_values": {
#             "log_mu": logmu_final,
#             "mu": np.exp(logmu_final)
#         },
#         "fit_statistics": Fit_stat
#     }



# def LCp_fit_second(
#     ax_coef_init,
#     bx_coef_init,
#     kappa_init,
#     Extg,
#     Dxtg,
#     xv,
#     tv,
#     degree=2,
#     n_knots=10,
#     xmin=None,
#     xmax=None,
#     lam=0.0,
#     diff_order=2,
#     nb_iter=800,
#     eta0=0.2,
#     tol=1e-6,
#     verbose=False,
# ):

#     nb_years = len(tv)
#     nb_regions = Extg.shape[2]

#     if xmin is None:
#         xmin = float(np.min(xv))
#     if xmax is None:
#         xmax = float(np.max(xv))

#     B, knots, n_basis = make_bspline_basis(xv, degree, n_knots, xmin, xmax)

#     ax_coef = ax_coef_init.copy()
#     bx_coef = bx_coef_init.copy()
#     kappa = kappa_init.copy()

#     DtD, diag_DtD = make_penalty_matrix(n_basis, diff_order)
#     logDxtgFact = gammaln(Dxtg + 1)

#     eta = min(eta0, 1.0)
#     flag = 0
#     it = 0

#     # log-likelihood initial
#     logmu, ax, bx_reg = compute_logmu(
#         ax_coef, bx_coef, kappa, xv, B, knots, degree
#     )
#     lnL, _, _, _ = poisson_lnL(Dxtg, Extg, logmu, logDxtgFact)

#     while (it < nb_iter) and (flag < 4):

#         it += 1

#         # --- calcul gradient local ---
#         logmu, ax, bx_reg = compute_logmu(
#             ax_coef, bx_coef, kappa, xv, B, knots, degree
#         )
#         lnL_current, _, weighted_exp, residual = poisson_lnL(
#             Dxtg, Extg, logmu, logDxtgFact
#         )

#         # --- tentative de mise à jour ---
#         eta_try = eta
#         accepted = False

#         while not accepted:

#             ax_new = update_ax_coef(
#                 ax_coef, B, residual, weighted_exp,
#                 eta_try, lam, DtD, diag_DtD
#             )

#             bx_new = update_bx_coef(
#                 bx_coef, B, kappa, residual, weighted_exp,
#                 eta_try, lam, DtD, diag_DtD
#             )

#             kappa_new = update_kappa(
#                 kappa, bx_reg, residual, weighted_exp,
#                 eta_try
#             )

#             # recalcul loglik candidate
#             logmu_new, _, _ = compute_logmu(
#                 ax_new, bx_new, kappa_new,
#                 xv, B, knots, degree
#             )

#             lnL_new, _, _, _ = poisson_lnL(
#                 Dxtg, Extg, logmu_new, logDxtgFact
#             )

#             if lnL_new >= lnL_current:
#                 accepted = True
#                 ax_coef = ax_new
#                 bx_coef = bx_new
#                 kappa = kappa_new
#                 lnL = lnL_new
#                 eta = min(eta_try * 1.05, 1.0)
#             else:
#                 eta_try *= 0.5
#                 if eta_try < 1e-8:
#                     accepted = True  # abandon amélioration

#         Delta_lnL = lnL - lnL_current

#         if abs(Delta_lnL) < tol:
#             flag += 1
#         else:
#             flag = 0

#         if verbose and (it % 10 == 0):
#             print(f"It {it:4d} | lnL = {lnL:.4f} | Δ = {Delta_lnL:+.6f} | η = {eta:.5f}")

#     # Rescaling final
#     bx_coef, bx_reg, kappa = rescale_bx_kappa(bx_coef, bx_reg, kappa)

#     logmu_final, ax, bx_reg = compute_logmu(
#         ax_coef, bx_coef, kappa, xv, B, knots, degree
#     )

#     Fit_stat = compute_fit_stats(
#         Dxtg, Extg, logmu_final,
#         logDxtgFact, n_basis,
#         nb_years, nb_regions
#     )

#     results = {
#         "parameters": {
#             "ax_coef": ax_coef,
#             "bx_coef": bx_coef,
#             "kappa": kappa
#         },
#         "curves": {
#             "alpha_x": ax,
#             "beta_xg": bx_reg
#         },
#         "fitted_values": {
#             "log_mu": logmu_final,
#             "mu": np.exp(logmu_final)
#         },
#         "fit_statistics": Fit_stat
#     }

#     return results



def build_input_from_dataframe(df):
    """
    Transforme un DataFrame long en matrices 3D compatibles LCp_fit.
    
    Paramètres
    ----------
    df : DataFrame avec colonnes
         ['region', 'year', 'age', 'deaths', 'exposure','mortality_rate']
    
    Retourne
    --------
    Muxtg: (nb_ages, nb_years, nb_regions)
    Dxtg : (nb_ages, nb_years, nb_regions)
    Extg : (nb_ages, nb_years, nb_regions)
    xv   : vecteur des âges triés
    tv   : vecteur des années triées
    regions : liste des régions
    """
    
    # Tri pour sécurité
    df = df.sort_values(["age", "year", "region"]).copy()
    
    xv = np.sort(df["age"].unique())
    tv = np.sort(df["year"].unique())
    regions = np.sort(df["region"].unique())
    
    nb_ages = len(xv)
    nb_years = len(tv)
    nb_regions = len(regions)
    
    # Mapping index
    age_idx = {a:i for i,a in enumerate(xv)}
    year_idx = {y:i for i,y in enumerate(tv)}
    reg_idx = {r:i for i,r in enumerate(regions)}
    
    # Allocation
    Dxtg = np.zeros((nb_ages, nb_years, nb_regions))
    Extg = np.zeros_like(Dxtg)
    Muxtg = np.zeros_like(Dxtg)
    
    # Vectorisation sans boucle triple
    Dxtg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.deaths.values
    
    Extg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.exposure.values

    Muxtg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.mortality_rate.values
    
    # Sécurité numérique
    Extg = np.maximum(Extg, 1e-12)
    Dxtg = np.maximum(Dxtg, 0.0)
    
    return Muxtg, Dxtg, Extg, xv, tv, regions


#### Lee carter classique 

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

    results = {
        "parameters": {
            "ax_coef": axM,
            "bx_coef": bxM,
            "kappa": kappaM
        },

        "fitted_values": {
            "log_mu": axM+bxM*kappaM,
            "mu": np.exp(axM+bxM*kappaM)
        },
        "fit_statistics": Fit_stat
    }    
    #we return ax, bx, kappa and stats    
    return results



# Modèle Lee carter paramétrique

def make_bspline_basis_national(x, degree=2, n_knots=10):

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    knots_internal = np.linspace(xmin, xmax, n_knots)

    knots = np.concatenate((
        np.repeat(xmin, degree),
        knots_internal,
        np.repeat(xmax, degree)
    ))

    n_basis = len(knots) - degree - 1

    B = np.zeros((len(x), n_basis))

    for i in range(n_basis):
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        spline = BSpline(knots, coef, degree)
        B[:, i] = spline(x)

    return B, n_basis

def make_penalty_matrix_national(n_basis, diff_order=2):

    D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    DtD = D.T @ D

    return DtD, np.diag(DtD)

def compute_logmu_national(ax_coef, bx_coef,
                           kappa, B,
                           nb_regions):

    ax = B @ ax_coef
    bx = B @ bx_coef

    logmu_2d = ax[:, None] + bx[:, None] * kappa[None, :]

    logmu_3d = np.repeat(
        logmu_2d[:, :, np.newaxis],
        nb_regions,
        axis=2
    )

    return logmu_3d, logmu_2d, ax, bx

def poisson_lnL_national(Dxtg, Extg,
                         logmu_3d,
                         logDxtFact):

    exp_logmu = np.exp(logmu_3d)
    weighted_exp = Extg * exp_logmu
    residual = Dxtg - weighted_exp

    lnL = float(np.sum(
        Dxtg * logmu_3d
        - weighted_exp
        + Dxtg * np.log(np.maximum(Extg, 1e-12))
        - logDxtFact
    ))

    return lnL, weighted_exp, residual


def update_ax_coef_national(ax_coef, B,
                            residual,
                            weighted_exp,
                            eta, lam,
                            DtD, diag_DtD):

    n_basis = len(ax_coef)
    ax_new = ax_coef.copy()

    pen_grad = 2.0 * lam * (DtD @ ax_coef) if lam > 0 else np.zeros(n_basis)

    for j in range(n_basis):

        Bj = B[:, j][:, None]

        num = float(np.sum(residual * Bj)) - pen_grad[j]
        den = float(np.sum(weighted_exp * Bj**2))

        if lam > 0:
            den += 2.0 * lam * diag_DtD[j]

        if den != 0:
            ax_new[j] += eta * num / den

    return ax_new


def update_bx_coef_national(bx_coef, B, kappa,
                            residual,
                            weighted_exp,
                            eta, lam,
                            DtD, diag_DtD):

    n_basis = len(bx_coef)
    bx_new = bx_coef.copy()

    nb_ages = B.shape[0]
    nb_regions = residual.shape[2]

    kappaM = np.repeat(kappa[None, :], nb_ages, axis=0)
    kappaM = np.expand_dims(kappaM, axis=2)
    kappaM = np.repeat(kappaM, nb_regions, axis=2)

    pen_grad = 2.0 * lam * (DtD @ bx_coef) if lam > 0 else np.zeros(n_basis)

    for j in range(n_basis):

        BjK = B[:, j][:, None, None] * kappaM

        num = float(np.sum(residual * BjK)) - pen_grad[j]
        den = float(np.sum(weighted_exp * BjK**2))

        if lam > 0:
            den += 2.0 * lam * diag_DtD[j]

        if den != 0:
            bx_new[j] += eta * num / den

    return bx_new


def update_kappa_national(kappa, bx,
                          residual,
                          weighted_exp,
                          eta):

    nb_regions = residual.shape[2]

    bxM = bx[:, None]
    bxM = np.expand_dims(bxM, axis=2)
    bxM = np.repeat(bxM, nb_regions, axis=2)

    num_k = np.sum(residual * bxM, axis=(0, 2))
    den_k = np.sum(weighted_exp * bxM**2, axis=(0, 2))

    kappa_new = kappa.copy()

    mask = den_k != 0
    kappa_new[mask] += eta * num_k[mask] / den_k[mask]

    return kappa_new


def LCp_fit(
    ax_coef_init,
    bx_coef_init,
    kappa_init,
    Extg,
    Dxtg,
    xv,
    tv,
    degree=2,
    n_knots=10,
    lam=0.0,
    diff_order=2,
    nb_iter=500,
    eta=0.2,
    tol=1e-4,
    verbose=False
):

    nb_regions = Extg.shape[2]

    B, n_basis = make_bspline_basis_national(
        xv, degree, n_knots
    )

    DtD, diag_DtD = make_penalty_matrix_national(
        n_basis, diff_order
    )

    ax_coef = ax_coef_init.copy()
    bx_coef = bx_coef_init.copy()
    kappa = kappa_init.copy()

    logDxtFact = gammaln(Dxtg + 1)

    lnL = -np.inf

    for it in range(nb_iter):

        logmu_3d, logmu_2d, ax, bx = compute_logmu_national(
            ax_coef, bx_coef, kappa, B, nb_regions
        )

        lnL_new, weighted_exp, residual = poisson_lnL_national(
            Dxtg, Extg, logmu_3d, logDxtFact
        )

        if abs(lnL_new - lnL) < tol:
            break

        lnL = lnL_new

        ax_coef = update_ax_coef_national(
            ax_coef, B,
            residual, weighted_exp,
            eta, lam,
            DtD, diag_DtD
        )

        bx_coef = update_bx_coef_national(
            bx_coef, B, kappa,
            residual, weighted_exp,
            eta, lam,
            DtD, diag_DtD
        )

        kappa = update_kappa_national(
            kappa, bx,
            residual, weighted_exp,
            eta
        )

    # Rescale Σβ = 1
    scale = np.sum(bx)
    if scale != 0:
        bx_coef /= scale
        bx /= scale
        kappa *= scale

    logmu_3d, logmu_2d, ax, bx = compute_logmu_national(
        ax_coef, bx_coef, kappa, B, nb_regions
    )

    results = {
        "parameters": {
            "ax_coef": ax_coef,
            "bx_coef": bx_coef,
            "kappa": kappa
        },
        "curves": {
            "alpha_x": ax,
            "beta_x": bx
        },
        "fitted_values": {
            "log_mu": logmu_2d,
            "mu": np.exp(logmu_2d)
        }
    }

    return results