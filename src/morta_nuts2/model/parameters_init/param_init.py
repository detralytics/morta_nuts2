import sys, os
sys.path.append(os.path.abspath(".."))
import geopandas as gpd
import sys
sys.executable
import numpy as np
import pandas as pd
from morta_nuts2.model.Bsplines.Bsplines import make_bspline_basis

def lcp_bspline_initialisation(Dxtg, Extg, xv, degree, n_knots):
    """
    Initialisation Lee-Carter robuste compatible B-splines.
    Multi-régions.
    """
    
    # 1️⃣ Taux agrégés toutes régions
    Dxt = np.sum(Dxtg, axis=2)
    Ext = np.sum(Extg, axis=2)
    
    Mxt = Dxt / np.maximum(Ext, 1e-12)
    Mxt = np.maximum(Mxt, 1e-12)
    
    logM = np.log(Mxt)
    
    # 2️⃣ alpha_x = moyenne par âge
    ax = np.mean(logM, axis=1)
    
    # 3️⃣ centrage
    M_centered = logM - ax[:, None]
    
    # 4️⃣ SVD
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    
    bx_raw = U[:, 0]
    kappa = S[0] * Vt[0, :]
    
    # 5️⃣ normalisation standard Lee-Carter
    bx_raw = bx_raw / np.sum(bx_raw)
    kappa = kappa * np.sum(bx_raw)
    
    # 6️⃣ Projection sur base spline
    B, knots, n_basis = make_bspline_basis(xv, degree, n_knots)
    
    # moindres carrés pour trouver coefficients spline
    ax_coef = np.linalg.lstsq(B, ax, rcond=None)[0]
    bx_coef_1d = np.linalg.lstsq(B, bx_raw, rcond=None)[0]
    
    # Répliquer β pour chaque région (point de départ neutre)
    nb_regions = Dxtg.shape[2]
    bx_coef = np.tile(bx_coef_1d, (nb_regions, 1))
    
    return ax_coef, bx_coef, kappa


def lileep_bspline_initialisation(Dxtg, Extg, xv, degree, n_knots):
    """
    Initialisation robuste Li-Lee compatible B-splines pénalisées.
    """

    nb_ages, nb_years, nb_regions = Dxtg.shape

    # =====================================================
    # 1️⃣ TAUX AGRÉGÉS → composante commune
    # =====================================================
    Dxt = np.sum(Dxtg, axis=2)
    Ext = np.sum(Extg, axis=2)

    Mxt = Dxt / np.maximum(Ext, 1e-12)
    Mxt = np.maximum(Mxt, 1e-12)

    logM = np.log(Mxt)

    # alpha commun
    alpha_common = np.mean(logM, axis=1)

    M_centered = logM - alpha_common[:, None]

    # SVD commune
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

    beta_common = U[:, 0]
    kappa_common = S[0] * Vt[0, :]

    # normalisation LC standard
    beta_common /= np.sum(beta_common)
    kappa_common *= np.sum(beta_common)

    # =====================================================
    # 2️⃣ Déviations régionales
    # =====================================================
    beta_g = np.zeros((nb_ages, nb_regions))
    kappa_g = np.zeros((nb_regions, nb_years))

    for g in range(nb_regions):

        Mxtg = Dxtg[:, :, g] / np.maximum(Extg[:, :, g], 1e-12)
        Mxtg = np.maximum(Mxtg, 1e-12)
        logM_g = np.log(Mxtg)

        # retirer composante commune
        residual = (
            logM_g
            - alpha_common[:, None]
            - np.outer(beta_common, kappa_common)
        )

        U_g, S_g, Vt_g = np.linalg.svd(residual, full_matrices=False)

        beta_g[:, g] = U_g[:, 0]
        kappa_g[g, :] = S_g[0] * Vt_g[0, :]

        # normalisation
        beta_g[:, g] /= np.sum(beta_g[:, g])
        kappa_g[g, :] *= np.sum(beta_g[:, g])

    # =====================================================
    # 3️⃣ Projection sur B-splines
    # =====================================================
    B, knots, n_basis = make_bspline_basis(xv, degree, n_knots)

    # alpha par région (point de départ : alpha commun répliqué)
    alpha_coef = np.zeros((nb_regions, n_basis))
    for g in range(nb_regions):
        alpha_coef[g] = np.linalg.lstsq(B, alpha_common, rcond=None)[0]

    beta_coef = np.linalg.lstsq(B, beta_common, rcond=None)[0]

    beta_g_coef = np.zeros((nb_regions, n_basis))
    for g in range(nb_regions):
        beta_g_coef[g] = np.linalg.lstsq(B, beta_g[:, g], rcond=None)[0]

    return (
        alpha_coef,
        beta_coef,
        beta_g_coef,
        kappa_common,
        kappa_g
    )

