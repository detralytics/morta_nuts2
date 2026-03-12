import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln 
import pandas as pd
from math import comb
#from scipy.interpolate import BSpline
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.linalg import lstsq



def difference_matrix(n, k):
    """
    Construct the finite difference matrix of order k for vectors of length n.
    Returns a (n-k) x n NumPy array.
    """
    if k >= n:
        raise ValueError("Order k must be less than n.")
    # Compute the coefficients of the k-th order difference using binomial coefficients
    coeffs = np.array([(-1)**j * comb(k, j) for j in range(k + 1)])
    
    # Initialize the matrix
    D = np.zeros((n - k, n))
    
    # Fill each row with shifted coefficients
    for i in range(n - k):
        D[i, i:i + k + 1] = coeffs
        
    return D        

# --------------------------------------------------------------------------------
# %% B-spline toolbox
# --------------------------------------------------------------------------------

def plot_Bsplines(all_knots, degree, xmin, xmax):
    x = np.arange(xmin, xmax, 0.1)
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))
    
    for j in range(ncoef):
        c = np.zeros(ncoef)
        c[j] = 1.0
        bs = BSpline(all_knots, c, degree, extrapolate=False)
        B[:, j] = bs(x)
        plt.plot(x, B[:, j], label=str(j))
        
    plt.grid()
    plt.legend()    
    plt.show()


def basis_matrix_from_knots(all_knots, degree, x):
    """Compute the B-spline basis matrix for given knots and x-values."""
    
    ncoef = len(all_knots) - degree - 1
    B = np.zeros((len(x), ncoef))
    
    for j in range(ncoef):
        c = np.zeros(ncoef)
        c[j] = 1.0
        bs = BSpline(all_knots, c, degree, extrapolate=False)
        B[:, j] = bs(x)
        
    return B


def Bspline_approx(coeffs, x, degree, m, xmin, xmax):
    """
    coeffs    : B-spline weights
    x         : input values
    degree    : B-spline degree
    m         : number of internal knots, equally spaced between xmin and xmax
    internal knots do not include xmin and xmax
    """
    
    dx = (xmax - xmin) / m    
    all_knots = xmin - dx * (degree) + dx * np.arange(0, m - 1 + 2 * (degree + 1))    
    
    B = basis_matrix_from_knots(all_knots, degree, x)
    yhat = B.dot(coeffs)    
    
    return yhat, all_knots, B




# =============================================================================
# 1. B-SPLINE CONSTRUCTION — scipy.interpolate.BSpline
# =============================================================================

def make_bspline_basis(xv, degree, n_knots, xmin=None, xmax=None):
    """
    Build the B-spline basis matrix using scipy.interpolate.BSpline.
    
    Parameters
    ----------
    xv       : array   evaluation points (e.g. ages 0 to 82)
    degree   : int     spline degree (3 recommended)
    n_knots  : int     number of internal knots (equivalent to m+1 in the original code)
    xmin     : float   lower bound (default: min(xv))
    xmax     : float   upper bound (default: max(xv))
    
    Returns
    -------
    B       : (len(xv), n_basis)   B-spline basis matrix
    knots   : array                full knot vector (with boundary multiplicity)
    n_basis : int                  number of basis functions
    
    Notes
    -----
    scipy.interpolate.BSpline uses the convention:
      - internal knots : np.linspace(xmin, xmax, n_knots)
      - full knots     : boundaries repeated degree+1 times
      - n_basis = n_knots + degree - 1
    """
    
    if xmin is None:
        xmin = float(np.min(xv))
        
    if xmax is None:
        xmax = float(np.max(xv))
    
    # Equally spaced internal knots
    internal_knots = np.linspace(xmin, xmax, n_knots)
    
    # Full knot vector with boundary multiplicity (scipy convention)
    knots = np.concatenate([
        [xmin] * degree,        # left boundary repetition
        internal_knots,
        [xmax] * degree         # right boundary repetition
    ])
    
    n_basis = len(knots) - degree - 1
    
    # Build the basis matrix (each column = one basis function)
    B = np.zeros((len(xv), n_basis))
    
    for i in range(n_basis):
        
        # Basis function i: all coefficients are zero except the i-th
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        
        spline = BSpline(knots, coef, degree, extrapolate=False)
        B[:, i] = spline(xv)
    
    # Replace NaN values with 0 (outside the support)
    B = np.nan_to_num(B, nan=0.0)
    
    return B, knots, n_basis


def eval_bspline_from_coef(coef, xv, knots, degree):
    """
    Evaluate a B-spline curve from its coefficients.
    Uses scipy.interpolate.BSpline.
    
    Parameters
    ----------
    coef   : array  B-spline coefficients
    xv     : array  evaluation points
    knots  : array  full knot vector
    degree : int    spline degree
    
    Returns
    -------
    y : array   curve evaluated at xv
    """
    
    spline = BSpline(knots, coef, degree, extrapolate=False)
    y = spline(xv)
    
    return np.nan_to_num(y, nan=0.0)