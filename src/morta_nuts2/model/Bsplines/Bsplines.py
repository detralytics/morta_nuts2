import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln 
import pandas as pd
from math import comb
from scipy.interpolate import BSpline



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
# --------------------------------------------------------------------------------
# %% B spline  toolbox
# --------------------------------------------------------------------------------
def plot_Bsplines(all_knots, degree, xmin, xmax):
    x=np.arange(xmin,xmax,0.1)
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))
    for j in range(ncoef):
        c    = np.zeros(ncoef)
        c[j] = 1.0
        bs   = BSpline(all_knots, c, degree , extrapolate=False)
        B[:, j] = bs(x)
        plt.plot(x,B[:, j],label=str(j))    
    plt.grid()
    plt.legend()    
    plt.show()

def basis_matrix_from_knots(all_knots, degree, x):
    """Compute B-spline basis matrix for given knots and x-values."""
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))
    for j in range(ncoef):
        c    = np.zeros(ncoef)
        c[j] = 1.0
        bs   = BSpline(all_knots, c, degree , extrapolate=False)
        B[:, j] = bs(x)
    return B

def Bspline_approx(coeffs, x, degree , m, xmin, xmax):
    # coeffs    : Bspline weights
    # x         : input
    # degree    : Bspline order
    # minternal : number of internal knots, equidistant between xmin and xmax          
    # internal knots do not contain x_min and x_max
    dx         = (xmax-xmin)/m    
    all_knots  = xmin-dx*(degree) + dx*np.arange(0, m-1+2*(degree+1))    
    B          = basis_matrix_from_knots(all_knots, degree, x)
    yhat       = B.dot(coeffs)    
    return yhat , all_knots , B


import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.special import gammaln
from scipy.linalg import lstsq


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


