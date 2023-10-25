import mpmath
import math
import pandas as pd
import numpy as np
#import csv
import statsmodels.api as sm
import scipy.stats
#import matplotlib.pyplot as plt
#from statsmodels.compat.python import Literal, lzip
#from collections import defaultdict
#import datetime
#import statsmodels.tsa.arima.model
from statsmodels.tsa.arima.model import ARIMA
from copy import deepcopy
from copy import copy


def autocov_est(vhat,j):
    vhat = np.copy(vhat)
    if j >= 0:
        T = len(vhat)
        v_t = vhat[j:T]
        v_tm1 = vhat[0:T-j]
        gamma_hat_j = np.dot(v_t.T,v_tm1)/T
        return gamma_hat_j
    else:
        raise ValueError("j cannot be negative number")
        
def reg_estimating_equation_v1(data, lag):
    """
    Estimating equtation for autocorrelation, the first option

    Parameters:
    - data (np.ndarray): The input data array. y_t from t=1 to T
    - p (lag) (int): The lag order estimating equation

    Returns:
    - y (np.ndarray): y_t from t=p+1 to T
    - x (np.ndarray): y_t-p
    - nmp (int): The length of x (T - p).
    - X (np.ndarray): T x 2 matrix with intercept and x
    - coeffs (np.ndarray): The estimated coefficients from the estimating equation
    """
    data = np.copy(data)
    p = np.copy(lag)
    
    T_data = len(data)  # T

    # Create y and x based on the lag order p
    y = np.copy(data[p:T_data, :])
    x = np.copy(data[0:T_data - p, :])

    # Calculate nmp (T - p)
    nmp = len(x)

    # Create a constant term for the regression
    cons = np.ones((nmp, 1))

    # Create the design matrix X
    X = np.hstack((cons, x))

    # Calculate the coefficients based on the ordinary least squares (OLS) formula
    coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    return y, x, nmp, X, coeffs

def AD_band(vhat):
    vhat = np.copy(vhat)
    #vhat = np.copy(xuhat)
    if vhat.shape[1] == 1:
        pass
    elif vhat.shape[1] == 2:
        vhat = np.copy(np.reshape(vhat[:,1],(len(vhat[:,1]),1))) 
    
    T = len(vhat)

    vhat_t = np.copy(vhat[1:])
    vhat_tm1 = np.copy(vhat[:-1])
    y2 = np.copy(vhat_t)
    X2 = np.copy(vhat_tm1)
    rho_hat = np.linalg.inv(X2.transpose().dot(X2)).dot(X2.transpose()).dot(y2) #without constant
    #print (rho_hat)
    #if (np.abs(rho_hat-1) < 0.001) & (rho_hat>=1):
    #    rho_hat = 1.001
    #elif (np.abs(rho_hat-1) < 0.001) & (rho_hat<1):
    #    rho_hat = 0.999
        
    #if np.abs(rho_hat) >= 0.97:
    #    rho_hat = 0.97*np.sign(rho_hat)

    alpha_hat_2 = (4*(rho_hat)**2)/((1-rho_hat)**4) #univariate version CLP
    ST = 2.6614*(T*alpha_hat_2)**(0.2) #bandwidth
    ST = np.float64(ST)
    
    if ST > T:
        ST = np.copy(T)
    
    return ST

def noncen_chisq(k,j,x):
    #k is df
    #j is non-centrality parameter
    pdf_v = 0.5*np.exp(-0.5*(x+j))*((x/j)**(k/4-0.5))*mpmath.besseli(0.5*k-1, np.sqrt(j*x), derivative=0)
    pdf_v2 = np.float64(pdf_v)
    return pdf_v2

def chisq_dfone(x):
    pdf_v = (np.exp(-x/2))/(np.sqrt(2*math.pi*x))
    pdf_v2 = np.float64(pdf_v)
    return pdf_v2

def SPJ_band(vhat,w,z_a=1.96,delta=2,q=2,g=6,c=0.539):
    #for Parzen
    #z_a = 1.96 # w is tuning parameter
    vhat = np.copy(vhat)

    if vhat.shape[1] == 1:
        pass
    elif vhat.shape[1] == 2:
        vhat = np.copy(np.reshape(vhat[:,1],(len(vhat[:,1]),1))) 
    
    T = len(vhat)

    vhat_t = np.copy(vhat[1:])
    vhat_tm1 = np.copy(vhat[:-1])
    y2 = np.copy(vhat_t)
    X2 = np.copy(vhat_tm1)
    rho_hat = np.linalg.inv(X2.transpose().dot(X2)).dot(X2.transpose()).dot(y2) #without constant
    
    #if np.abs(rho_hat) >= 0.97:
    #    rho_hat = 0.97*np.sign(rho_hat)
    
    d_hat = (2*rho_hat)/(1-rho_hat)**2
    d_hat = np.float64(d_hat)
    x_val = z_a**2
    
    ### calculation of b_hat
    G_0 = chisq_dfone(x=x_val)
    G_d = noncen_chisq(k=1,j=delta**2,x=x_val)
    k_d = ((delta**2)/(2*x_val))*noncen_chisq(k=3,j=delta**2,x=x_val)
    
    if d_hat*(w*G_0 - G_d) > 0:
        b_hat = (((q*g*d_hat*(w*G_0 - G_d))/(c*x_val*k_d))**(1/3))*(T**(-2/3))
        #print (b_hat)
    elif d_hat*(w*G_0 - G_d) <= 0:
        b_hat = np.log(T)/T
    else:
        raise Exception("Error")
    spj_band = b_hat * T
    spj_band = np.float64(spj_band)
    
    if spj_band > T:
        spj_band = np.copy(T)    
    
    return spj_band


def Parzen_vec(x):
    x = np.copy(x)
    
    kx = np.zeros_like(x)
    
    mask1 = (0 <= np.abs(x)) & (np.abs(x) <= 0.5)
    mask2 = (0.5 < np.abs(x)) & (np.abs(x) <= 1)
    
    kx[mask1] = 1 - 6 * (x[mask1] ** 2) + 6 * (np.abs(x[mask1]) ** 3)
    kx[mask2] = 2 * ((1 - np.abs(x[mask2])) ** 3)
    
    return kx

def newLRV_vec(vhat, M_n):
    vhat = np.copy(vhat)
    
    n, k = vhat.shape
    LRV = autocov_est(vhat, 0)
    
    j_values = np.arange(1, n)
    parzen_values = Parzen_vec(j_values / M_n)
    
    autocov_values = np.array([autocov_est(vhat, j) for j in j_values])
    
    for j, p_val in zip(j_values, parzen_values):
        LRV += p_val * (autocov_values[j-1] + autocov_values[j-1].T)

    return LRV


def calc_fixed_cv_org(fixed_b, Tnum, fixedb_coeffs):
    """
    Compute fixed-b critical values for t-test

    Args:
    - fixed_b: value of bandwidth (i.e. M)
    - Tnum (int): length of time series, in this autocorrelation testing case, T-k
    - fixedb_coeffs: List of fixed-b cv coefficients [c1, c2, ..., c9]

    Returns:
    - Scalar, fixed-b critical value
    """
    c = copy(fixedb_coeffs) #shallow copy for list prevent mistake
    
    return 1.96 + c[0]*(fixed_b/Tnum*1.96) + c[1]*((fixed_b/Tnum)*(1.96**2)) + c[2]*((fixed_b/Tnum)*(1.96**3)) + \
        c[3]*(((fixed_b/Tnum)**2)*1.96) + c[4]*(((fixed_b/Tnum)**2)*(1.96**2)) + c[5]*(((fixed_b/Tnum)**2)*(1.96**3)) + \
        c[6]*(((fixed_b/Tnum)**3)*(1.96**1)) + c[7]*(((fixed_b/Tnum)**3)*(1.96**2)) + c[8]*(((fixed_b/Tnum)**3)*(1.96**3))

def create_spj_function(w, z_a=1.96, delta=2, q=2, g=6, c=0.539):
    def spj_function(vhat):
        return SPJ_band(vhat, w=w, z_a=z_a, delta=delta, q=q, g=g, c=c)
    return spj_function

#usage 
#spj_function = create_spj_function(w=10)

def process_tstat_fixedbcv(vhat, M_n, X, nmp, true_acf1, coeffs1, fixedb_coeffs):
    vhat = np.copy(vhat)
    X =  np.copy(X)
    fixedb_coeffs = copy(fixedb_coeffs)
    
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = newLRV_vec(vhat, M_n)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp
    fixed_cv = calc_fixed_cv_org(M_n, nmp, fixedb_coeffs)
    t_stat = (coeffs1 - true_acf1) / np.sqrt(np.float64(var[1, 1]))

    return t_stat, fixed_cv, np.float64(var[1, 1])

def initialize_stats_dict():
    return {
        "rej": 0,
        "var": 0,
        "band":0,
    }

def update_stats_dict(vhat, X, nmp, true_acf1, coeffs1, fixedb_coeffs, band_set, stats, spj_band_function):
    vhat = np.copy(vhat)
    X = np.copy(X)
    fixedb_coeffs = copy(fixedb_coeffs)
    band_set = copy(band_set)
    
    
    # For constb
    for b_coord in range(len(band_set)):
        M_n_constb = np.float64(band_set[b_coord]) * len(X)
        t_stat_constb, fixed_cv_constb, var_constb = process_tstat_fixedbcv(vhat, M_n_constb, X, nmp, true_acf1, coeffs1, fixedb_coeffs)
        
        if np.abs(t_stat_constb) > 1.96:
            stats['constb_normal'][b_coord]['rej'] += 1
        if np.abs(t_stat_constb) > fixed_cv_constb:
            stats['constb_fixedb'][b_coord]['rej'] += 1
        
        # save var
        stats['constb_fixedb'][b_coord]['var'] += var_constb

    # For SPJ
    M_n_spj = spj_band_function(vhat)  # Here, we use the passed-in function to compute M_n_spj
    t_stat_spj, fixed_cv_spj, var_spj = process_tstat_fixedbcv(vhat, M_n_spj, X, nmp, true_acf1, coeffs1, fixedb_coeffs)

    if np.abs(t_stat_spj) > 1.96:
        stats['SPJ_normal']['rej'] += 1
    if np.abs(t_stat_spj) > fixed_cv_spj:
        stats['SPJ_fixedb']['rej'] += 1
     
    #save var
    stats['SPJ_normal']['var'] += var_spj
    stats['SPJ_fixedb']['var'] += var_spj
    
    #save band
    stats['SPJ_normal']['band'] += M_n_spj/nmp
    stats['SPJ_fixedb']['band'] += M_n_spj/nmp

    # For AD
    M_n_ad = AD_band(vhat)
    t_stat_ad, fixed_cv_ad, var_ad = process_tstat_fixedbcv(vhat, M_n_ad, X, nmp, true_acf1, coeffs1, fixedb_coeffs)

    if np.abs(t_stat_ad) > 1.96:
        stats['AD_normal']['rej'] += 1
    if np.abs(t_stat_ad) > fixed_cv_ad:
        stats['AD_fixedb']['rej'] += 1
    
    # save var
    stats['AD_normal']['var'] += var_ad
    stats['AD_fixedb']['var'] += var_ad
    
    # save band
    stats['AD_normal']['band'] += M_n_ad/nmp
    stats['AD_fixedb']['band'] += M_n_ad/nmp
    
    return stats

def initialize_stats_dict_comp():
    return {
        "rej": 0,
        "var": 0,
        "replication": 0,
    }

def update_stats_comp_dict(ori_rho_hat, true_acf1, var_list, stats_comp):
    var_list = copy(var_list)
    
    # Ensure the length of var_list and the keys in stats_comp are the same
    if len(var_list) != len(stats_comp):
        raise ValueError("The length of var_list and the number of keys in stats_comp must be the same.")
    
    # Iterate over each var and its corresponding key in stats_comp
    for var, key in zip(var_list, stats_comp.keys()):
        t_stat = (ori_rho_hat - true_acf1) / np.sqrt(var)
        
        if var > 0:
            stats_comp[key]['replication'] += 1
            if np.abs(t_stat) > 1.96:
                stats_comp[key]['rej'] += 1

            # Update variance
            stats_comp[key]['var'] += var

    return stats_comp

# Original function
def compute_Lambda_original(f, ell):
    f = np.copy(f)
    
    T, d = f.shape  
    Lambda_ell = np.zeros((d, 1))  
    
    for t in range(1, T + 1):
        r = t / T
        phi = Phi_ell(ell, r)
        
        Lambda_ell += phi * f[t-1, :, np.newaxis]  # Reshape f[t-1] to be (d, 1)
    
    Lambda_ell = Lambda_ell / np.sqrt(T)
    
    return Lambda_ell

# Vectorized function
def compute_Lambda_vectorized(f, ell):
    f = np.copy(f)
    
    T, d = f.shape  
    
    r_values = np.arange(1, T+1) / T
    phi_values = Phi_ell(ell, r_values)
    Lambda_ell = np.sum(phi_values[:, np.newaxis] * f, axis=0) / np.sqrt(T)
    
    return Lambda_ell.reshape((d, 1))

# Function for Phi_ell
def Phi_ell(ell, r):
    if ell % 2 == 0:
        return np.sqrt(2) * np.sin(np.pi * ell * r)
    else:
        return np.sqrt(2) * np.cos(np.pi * (ell + 1) * r)

# Function to compute Omega_OS
def compute_Omega_OS(f, K):
    '''
    args
    - f (array): f is T x d (or n x d) array is vhat (zhat in online appendix of Pellatt and Sun (2022, JOE) ) 
    - K (integer) : K is MSE optimally chosen smoothing parameter  
    return
    - Omega_OS (LRV)
    '''
    f = np.copy(f)
    
    T, d = f.shape
    Omega_OS = np.zeros((d, d))
    
    for ell in range(1, K + 1):
        Lambda_ell = compute_Lambda_vectorized(f, ell)
        Omega_OS += np.dot(Lambda_ell, Lambda_ell.T)  # Outer product
    
    Omega_OS = Omega_OS / K  # Average
    
    return Omega_OS

def compute_khat_D(zhat, W_mat, integer_type):

    '''
    args :
    - zhat (n x d array) : zhat = x_i*\hat{u_i} in Online Appendix of Pellatt and Sun (2022, JOE)
    - W_mat ((d,) shape array) : Weighting Matrix. See Andrews (1991).
    - integer type: 'any' or 'even'. Even gives Khat_D as even.
    
    Return :
    - MSE mininized Khat_D (scalar)
    '''
    zhat = np.copy(zhat)
    W_mat = np.copy(W_mat)
    
    n, d = zhat.shape
    
    if W_mat.shape != (d,):
        raise ValueError("Weight vector W must have shape (d,), where d is the number of columns in zhat.")
    
    # Compute rho_hat
    Z_i = zhat[1:, :]  # Elements from i=2 to n
    Z_im1 = zhat[:-1, :]  # Elements from i=1 to n-1
    numerator = np.sum(Z_i * Z_im1, axis=0)
    denominator = np.sum(Z_im1 ** 2, axis=0)
    rho_hat = numerator / denominator  # This will be a d-length array, cf) rho_hat.reshape((d, 1)) => dx1 array
    
    # Compute sigma_squared_hat
    diff = Z_i - rho_hat * Z_im1
    
    sigma_squared_hat = np.sum(diff ** 2, axis=0) / n  # This will also be a d-length array, sigma_squared_hat.reshape((d, 1)) => dx1
    
    ## Reshape to 1D arrays for easier calculations (only if you are using (d,1) array. Don't need flatten if using (d,) array)
    #rho_hat = rho_hat.flatten()
    #sigma_squared_hat = sigma_squared_hat.flatten()
    
    # Compute kappa_D
    c_phi_2 = (np.pi**2) / 6
    term1 = W_mat * (rho_hat ** 2) * (sigma_squared_hat ** 2) / ((1 - rho_hat) ** 8)
    term2 = W_mat * (sigma_squared_hat ** 2) / ((1 - rho_hat) ** 4)
    sum1 = np.sum(term1)
    sum2 = np.sum(term2)
    kappa_D = (1 / (8 * (c_phi_2 ** 2))) * (sum2 / sum1)
    
    Khat_D = (kappa_D**(1/5))*(n**(4/5))

    if integer_type == 'any':
        Khat_D = np.ceil(Khat_D) # round up to next ineger, following Pellatt and Sun (2022)        
    elif integer_type == 'even':
        floor_Khat_D = np.floor(Khat_D)
        ceil_Khat_D = np.ceil(Khat_D)
        if floor_Khat_D % 2 == 0:
            Khat_D = floor_Khat_D
        else:
            Khat_D = ceil_Khat_D
    else:
        raise ValueError("Invalid value for integer_type. Expected 'any' or 'even'.")
        
    if isinstance(Khat_D, float):
        if not Khat_D.is_integer():
            raise ValueError("calculated Khat is a float but not an integer-valued float. Something about the caluclation is wrong")
            
    Khat_D = max(Khat_D, 2) #Truncation avoid division by zero
    
    return int(Khat_D)

def process_tstat_fixedK(vhat, Khat_D, X, true_acf1, coeffs1):
    vhat = np.copy(vhat)
    X = np.copy(X)
    
    nmp = len(X)
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = compute_Omega_OS(f=vhat, K=Khat_D)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp
    
    t_stat = (coeffs1 - true_acf1) / np.sqrt(np.float64(var[1, 1]))

    return t_stat, np.float64(var[1, 1])

def initialize_stats_dict_fixedK():
    return {
        "rej": 0,
        "var": 0,
        "Khat": 0, 
    }

def update_stats_fixedK_dict(vhat, X, true_acf1, coeffs1, W_mat, integer_type, stats_fixedK):
    
    '''
    Args:
    - vhat (np.array): T x d vhat (product of regressors and residual)
    - X (np.array): T x d regressors
    - true_acf1 (scalar) : Null value
    - coeffs1 : autocorrelation estimator
    - W_mat : weighting for MSE optimal smoothing parameter calculation, usually [0,1] for d = 2
    - integer_type: 'even' or 'any'
    - stats_fixedK : dict to update
    
    Returns:
    - updated dict (updated stats_fixedK)
    '''
    vhat = np.copy(vhat)
    X = np.copy(X)
    W_mat = np.copy(W_mat)
    
    # calculating Khat_D and t-stat with var
    nmp = len(X)
    Khat_D = compute_khat_D(zhat=vhat, W_mat=W_mat, integer_type=integer_type) 
    t_stat_fixedK, var_fixedK = process_tstat_fixedK(vhat=vhat, Khat_D=Khat_D, X=X, true_acf1=true_acf1, coeffs1=coeffs1)
    
    # Single restriction, p = 1, then sqrt of F(1,K) is student T distribution with K. So T_K is used for CV. See Pellat and Sun (2022)
    alpha_test = 0.05
    cv_fixedK = scipy.stats.t.ppf(1-alpha_test/2, Khat_D)
    
    if np.abs(t_stat_fixedK) > cv_fixedK:
        stats_fixedK['FixedK']['rej'] += 1

    #save var
    stats_fixedK['FixedK']['var'] += var_fixedK

    #save band
    stats_fixedK['FixedK']['Khat'] += Khat_D

    
    return stats_fixedK