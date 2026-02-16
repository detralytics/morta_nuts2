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


