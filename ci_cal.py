import mpmath
import math
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
import scipy.stats
#import matplotlib.pyplot as plt
#from statsmodels.compat.python import Literal, lzip
#from collections import defaultdict
import datetime
from statsmodels.compat.python import Literal, lzip
import cmath
from statistics import mean

def quadratic_roots(c2, c1, c0):
    """
    Calculate the roots of the equation: c2*a^2 + 2*c1*a + c0 = 0

    Args:
    - c2 (float): Coefficient of a^2
    - 2*c1 (float): Coefficient of a
    - c0 (float): Constant term

    Returns:
    - tuple: Roots of the equation (root1, root2)
    """
    
    # Calculating the discriminant
    D = cmath.sqrt((2*c1)**2 - 4*c2*c0)

    # Calculating the roots
    root1 = (-2*c1 + D) / (2*c2)
    root2 = (-2*c1 - D) / (2*c2)

    return root1, root2

## Example:
#c2, c1, c0 = 1, 1, 1
#roots = quadratic_roots(c2, c1, c0)
#print(f"The roots are: {roots[0]} and {roots[1]}")

def Parzen(x):
    if 0<=np.abs(x)<=0.5:
        kx = 1-6*(x**2)+6*((np.abs(x))**3)
        return kx
    elif 0.5<=np.abs(x)<=1:
        kx = 2*((1-np.abs(x))**3)
        return kx
    else:
        kx = 0
        return kx

def compute_sum_all(y, k, band, cv):
    # initialize a variable to store the sum
    sum1 = 0
    sum2 = 0
    T = len(y)

    y_tk_mean = np.mean(y[0:len(y)-k])
    dy = y - y_tk_mean
    # iterate over t and s
    for t in range(k, T):
        for s in range(k, T):
            sum1 += (dy[t-k]**2)*(dy[s-k]**2)
            sum2 += (dy[t-k]**2)*Parzen(np.abs(t-s)/band)*(dy[s-k]**2)
            

    return 1 - sum2/sum1*(cv**2), sum2/(len(y)-k)

def compute_omega_lrv(y, M, k_function, k, a0):
    y = y.reshape(-1, 1)
    T = y.shape[0]
    
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    y_tilda = y[k:] - ybar_k_plus_1_T
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k
    
    v_til = ((y_tilda * y_tilda_minus_k) - a0 * (y_tilda_minus_k ** 2)).squeeze()
    
    indices_matrix = np.abs(np.arange(k+1, T+1)[:, None] - np.arange(k+1, T+1))
    K = k_function(indices_matrix / M)
    

    omega = np.sum(v_til[:, None] * K * v_til[None, :]) / (T-k)
    
    return omega

def compute_var(y, k, Q2_val, lrv_val):
    return (Q2_val**-2)*lrv/(len(y)-k)


def tr_k_function(x):
    """Kernel Function."""
    if abs(x) <= 1:
        return 1
    else:
        return 0
    
def tr_k_function_vectorized(x):
    """Vectorized Kernel Function for tr_k_function."""
    
    # Defining the condition
    condition = [abs(x) <= 1]
    
    # Defining the output for the condition
    output = [1]
    
    # Using np.select to vectorize the condition
    return np.select(condition, output, default=0)

def qs_k_function(x):
    if x == 0:
        return 1
    elif x > 0:
        m = (6 * np.pi * x) / 5
        return (3 / m**2) * ((np.sin(m) / m) - np.cos(m))
    else:
        return 0

def qs_k_function_vectorized(x):
    """Vectorized QS Kernel Function."""
    
    m = (6 * np.pi * x) / 5
    
    conditions = [
        (x == 0),
        (x > 0)
    ]
    
    outputs = [
        1,
        (3 / m**2) * ((np.sin(m) / m) - np.cos(m))
    ]
    
    # Return 0 for x < 0 (default value) unless you need a different behavior for x < 0.
    return np.select(conditions, outputs, default=0)

def pc_k_function(x):
    """Kernel Function."""
    if x <= 1:
        return 1 / (1 + x**2)
    else:
        return 0

def pc_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    
    conditions = [
        (x <= 1)
    ]
    
    outputs = [
        1 / (1 + x**2)
    ]
    
    # Return 0 for x > 1 (default value).
    return np.select(conditions, outputs, default=0)


def pg_k_function(x):
    """Kernel Function."""
    if x <= 1:
        return 1 / (1 + x)
    else:
        return 0

def pg_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    
    conditions = [
        (x <= 1)
    ]
    
    outputs = [
        1 / (1 + x)
    ]
    
    # Return 0 for x > 1 (default value).
    return np.select(conditions, outputs, default=0)


def th_k_function(x):
    """Kernel Function."""
    if x <= 1:
        return 0.5 + 0.5 * np.cos(np.pi * x)
    else:
        return 0

def th_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    
    conditions = [
        (x <= 1)
    ]
    
    outputs = [
        0.5 + 0.5 * np.cos(np.pi * x)
    ]
    
    # Return 0 for x > 1 (default value).
    return np.select(conditions, outputs, default=0)

def tz_k_function(x):
    
    c = 0.4
    """Kernel Function."""
    abs_x = abs(x)
    if abs_x <= c:
        return 1
    elif c < abs_x <= 1:
        return (abs_x - 1) / (c - 1)
    else:
        return 0

def tz_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    
    c = 0.4
    abs_x = np.abs(x)
    
    conditions = [
        (abs_x <= c),
        (c < abs_x) & (abs_x <= 1)
    ]
    
    outputs = [
        np.ones_like(x),
        (abs_x - 1) / (c - 1)
    ]
    
    # Return 0 for other cases (default value).
    return np.select(conditions, outputs, default=0)

def dn_k_function(x):
    """Function to compute kappa value for a given x."""
    if x == 0:
        return 1
    else:
        return np.sin(np.pi * x) / (np.pi * x)


def dn_k_function_vectorized(x_array):
    """Vectorized version of the dn_k_function."""
    conditions = [
        x_array == 0
    ]
    
    outputs = [
        1
    ]
    
    result = np.select(conditions, outputs, default=np.sin(np.pi * x_array) / (np.pi * x_array))
    
    return result


def parzen_k_function(x):
    """Kernel Function."""
    if abs(x) <= 0.5:
        return 1 - 6 * x**2 + 6 * abs(x)**3
    elif 0.5 <= abs(x) <= 1:
        return 2 * (1 - abs(x))**3
    else:
        return 0
    
def Bart_k_function(x):
    """Kernel Function."""
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0
    
def parzen_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    conditions = [
        (abs(x) <= 0.5),
        (abs(x) > 0.5) & (abs(x) <= 1)
    ]
    
    outputs = [
        1 - 6 * x**2 + 6 * abs(x)**3,
        2 * (1 - abs(x))**3
    ]
    
    return np.select(conditions, outputs, default=0)
    
def Bart_k_function_vectorized(x):
    """Vectorized Bart Kernel Function."""
    condition = (abs(x) <= 1)
    
    output = 1 - abs(x)
    
    return np.where(condition, output, 0)

def constant_k_function_vectorized(x):
    """Vectorized Constant Kernel Function."""
    return np.ones_like(x)

def constant_k_function(x):
    """Vectorized Constant Kernel Function."""
    return 1
    

def compute_omegas_loops_demean(y, M, k_function, k):
    """
    Compute Omega values using double loops.
    
    Args:
    - y (np.array): Input data array (time-series)
    - M (int): Bandwidth
    - k_function (function): Kernel function
    - a (int): null value; H_0: rho_k = a
    - k (int): lag k; i.e. rho_k
    
    Returns:
    - tuple: Omega values (omega_11, omega_12, omega_22)
    """
    
    T = len(y)  # Compute the length of y
    
    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    # Initialize the omegas
    omega_11 = 0
    omega_12 = 0
    omega_22 = 0
    
    # Containers for v1 and v2 values
    v1_values = []
    v2_values = []

    # Demeaning y values and computing v1 and v2
    for t in range(k+1, T+1):
        y_tilda_t = y[t-1] - ybar_k_plus_1_T
        y_tilda_t_minus_k = y[t-1-k] - ybar_1_T_minus_k

        v1_values.append(y_tilda_t * y_tilda_t_minus_k)
        v2_values.append(y_tilda_t_minus_k**2)

    # Calculate the means of v1 and v2 
    mean_v1 = np.mean(v1_values)
    mean_v2 = np.mean(v2_values)

    # Compute Omega values
    for t in range(k+1, T+1):
        for s in range(k+1, T+1):
            y_tilda_t = y[t-1] - ybar_k_plus_1_T
            y_tilda_s = y[s-1] - ybar_k_plus_1_T
            
            y_tilda_t_minus_k = y[t-1-k] - ybar_1_T_minus_k
            y_tilda_s_minus_k = y[s-1-k] - ybar_1_T_minus_k
            
            v1_t = y_tilda_t * y_tilda_t_minus_k - mean_v1
            v1_s = y_tilda_s * y_tilda_s_minus_k - mean_v1
            
            v2_t = y_tilda_t_minus_k**2 - mean_v2
            v2_s = y_tilda_s_minus_k**2 - mean_v2
            
            k_value = k_function(abs(t-s) / M)
            
            omega_11 += v1_t * k_value * v1_s
            omega_12 += v1_t * k_value * v2_s
            omega_22 += v2_t * k_value * v2_s

    omega_11 /= (T-k)
    omega_12 /= (T-k)
    omega_22 /= (T-k)

    return np.float64(omega_11), np.float64(omega_12), np.float64(omega_22)

def compute_omegas_loops_nodemean(y, M, k_function, k):
    """
    Compute Omega values using double loops.
    
    Args:
    - y (np.array): Input data array
    - M (int): Scaling factor for kernel function
    - k_function (function): Kernel function
    - a (int): Parameter for adjustment
    - k (int): Parameter for indexing
    
    Returns:
    - tuple: Omega values (omega_11, omega_12, omega_22)
    """
    
    T = len(y)  # Compute the length of y
    
    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    # Initialize the omegas
    omega_11 = 0
    omega_12 = 0
    omega_22 = 0
    
    # Containers for v1 and v2 values
    v1_values = []
    v2_values = []

    ## Demeaning y values and computing v1 and v2
    #for t in range(k+1, T+1):
    #    y_tilda_t = y[t-1] - ybar_k_plus_1_T
    #    y_tilda_t_minus_k = y[t-1-k] - ybar_1_T_minus_k

    #    v1_values.append(y_tilda_t * y_tilda_t_minus_k)
    #    v2_values.append(y_tilda_t_minus_k**2)

    ## Calculate the means of v1 and v2 
    #mean_v1 = np.mean(v1_values)
    #mean_v2 = np.mean(v2_values)

    # Compute Omega values
    for t in range(k+1, T+1):
        for s in range(k+1, T+1):
            y_tilda_t = y[t-1] - ybar_k_plus_1_T
            y_tilda_s = y[s-1] - ybar_k_plus_1_T
            
            y_tilda_t_minus_k = y[t-1-k] - ybar_1_T_minus_k
            y_tilda_s_minus_k = y[s-1-k] - ybar_1_T_minus_k
            
            #v1_t = y_tilda_t * y_tilda_t_minus_k - mean_v1
            #v1_s = y_tilda_s * y_tilda_s_minus_k - mean_v1
            
            #v2_t = y_tilda_t_minus_k**2 - mean_v2
            #v2_s = y_tilda_s_minus_k**2 - mean_v2
            
            v1_t = y_tilda_t * y_tilda_t_minus_k 
            v1_s = y_tilda_s * y_tilda_s_minus_k 
            
            v2_t = y_tilda_t_minus_k**2 
            v2_s = y_tilda_s_minus_k**2
            
            k_value = k_function(abs(t-s) / M)
            
            omega_11 += v1_t * k_value * v1_s
            omega_12 += v1_t * k_value * v2_s
            omega_22 += v2_t * k_value * v2_s

    omega_11 /= (T-k)
    omega_12 /= (T-k)
    omega_22 /= (T-k)

    return np.float64(omega_11), np.float64(omega_12), np.float64(omega_22)

def compute_omegas_vectorized_nodemean(y, M, k_function, k):
    y = y.reshape(-1, 1)
    T = y.shape[0]
    
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    y_tilda = y[k:] - ybar_k_plus_1_T
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    v1 = (y_tilda * y_tilda_minus_k).squeeze()
    v2 = (y_tilda_minus_k ** 2).squeeze()

    indices_matrix = np.abs(np.arange(k+1, T+1)[:, None] - np.arange(k+1, T+1))
    K = k_function(indices_matrix / M)

    omega_11 = np.sum(v1[:, None] * K * v1[None, :]) / (T-k)
    omega_12 = np.sum(v1[:, None] * K * v2[None, :]) / (T-k)
    omega_22 = np.sum(v2[:, None] * K * v2[None, :]) / (T-k)
    
    return omega_11, omega_12, omega_22

def compute_omegas_vectorized_demean(y, M, k_function, k):
    y = y.reshape(-1, 1)
    T = y.shape[0]

    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    # Demeaning y values and computing v1 and v2
    y_tilda = y[k:] - ybar_k_plus_1_T
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    v1 = (y_tilda * y_tilda_minus_k).squeeze()
    v2 = (y_tilda_minus_k ** 2).squeeze()

    # Calculate the means of v1 and v2 
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)

    # Demean v1 and v2
    v1_demeaned = v1 - mean_v1
    v2_demeaned = v2 - mean_v2

    # Create the kernel matrix
    indices_matrix = np.abs(np.arange(k+1, T+1)[:, None] - np.arange(k+1, T+1))
    K = k_function(indices_matrix / M)

    # Compute Omega values using demeaned v1 and v2
    omega_11 = np.sum(v1_demeaned[:, None] * K * v1_demeaned[None, :]) / (T-k)
    omega_12 = np.sum(v1_demeaned[:, None] * K * v2_demeaned[None, :]) / (T-k)
    omega_22 = np.sum(v2_demeaned[:, None] * K * v2_demeaned[None, :]) / (T-k)

    return omega_11, omega_12, omega_22

def compute_Q2(y, k):
    """
    Compute Q2 value.
    
    Args:
    - y (np.array): Input data array (time-series)
    - k (int): lag k; i.e. rho_k
    
    Returns:
    - float: Q2 value
    """
    
    T = len(y)

    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])

    # Demeaning y values
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    # Compute Q2
    Q2 = np.mean(y_tilda_minus_k**2)
    
    return Q2

def compute_rho_k_tilda(y, k):
    """
    Compute the rho_k_tilda value.
    
    Args:
    - y (np.array): Input data array (time-series)
    - k (int): lag k; i.e. rho_k
    
    Returns:
    - float: rho_k_tilda value
    """
    
    T = len(y)
    
    # Compute the means for the relevant parts of y
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    # Demeaning y values
    y_tilda = y[k:T] - ybar_k_plus_1_T
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    # Compute the numerator and denominator for rho_k_tilda
    numerator = np.sum(y_tilda * y_tilda_minus_k)
    denominator = np.sum(y_tilda_minus_k**2)
    
    rho_k_tilda = numerator / denominator
    
    return rho_k_tilda

def compute_v_tilde_k_data_depen_nodemean(y, k, a0):
    """
    Compute \widetilde{v}_t^{(k)} for data dependent methods to calculate bandwidth (nodemean)
    for estimating equation : y_t = c + rho_k*y_t-1 + eta_t
    
    Args:
    - y (np.array): Input data array (time-series) of shape (T, 1)
    - k (int): lag k
    - a0 (float): estimates (depending on null (null value) or no null imposed (rho_k_tilde))
    
    Returns:
    - np.array: Array of \widetilde{v}_t^{(k)} values of shape (T-k, 1)
    """
    
    T = len(y)
    
    # Compute \widetilde{y}_{t-k} and \widetilde{y}_t
    y_tilda_t_minus_k = y[:T-k] - np.mean(y[:T-k])
    y_tilda_t = y[k:] - np.mean(y[k:])
    
    # Compute \widetilde{v}_t^{(k)}
    #v_tilda_k = y_tilda_t_minus_k * (y_tilda_t - a * y_tilda_t_minus_k)
    v_tilda_k = y[:T-k] * (y_tilda_t - a0 * y_tilda_t_minus_k)
    
    return v_tilda_k

def compute_v_tilde_k_data_depen_demean(y, k, a0):
    """
    Compute \widetilde{v}_t^{(k)} for data dependent methods to calculate bandwidth (demeaned \widetilde{v}_t^{(k)})
    for estimating equation : y_t = c + rho_k*y_t-1 + eta_t
    
    Args:
    - y (np.array): Input data array (time-series) of shape (T, 1)
    - k (int): lag k
    - a0 (float): estimates (depending on null (null value) or no null imposed (rho_k_tilde))
    
    Returns:
    - np.array: Array of \widetilde{v}_t^{(k)} values of shape (T-k, 1)
    """
    
    T = len(y)
    
    # Compute \widetilde{y}_{t-k} and \widetilde{y}_t
    y_tilda_t_minus_k = y[:T-k] - np.mean(y[:T-k])
    y_tilda_t = y[k:] - np.mean(y[k:])
    
    # Compute \widetilde{v}_t^{(k)}
    v_tilda_k = y[:T-k] * (y_tilda_t - a0 * y_tilda_t_minus_k)
    v_tilda_k_demean = v_tilda_k - np.mean(v_tilda_k)
    
    return v_tilda_k_demean

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
    #z_a = 1.96 # w is tuning parameter
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

def calc_fixed_cv(fixed_b, Tnum, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    """"
    compute fixed-b critical values for t-test
    
    Args:
    - fixed_b: value of bandwidth (i.e. M)
    - Tnum (int): length of time series, in this autocorrelation testing case, T-k
    - c1,..,c9: fixed-b cv coefficient
    
    Returns:
    - Scalar, fixed-b critical value
    """
    
    return 1.96 + c1*(fixed_b/Tnum*1.96) + c2*((fixed_b/Tnum)*(1.96**2)) + c3*((fixed_b/Tnum)*(1.96**3)) + \
        c4*(((fixed_b/Tnum)**2)*1.96) + c5*(((fixed_b/Tnum)**2)*(1.96**2)) + c6*(((fixed_b/Tnum)**2)*(1.96**3)) + \
        c7*(((fixed_b/Tnum)**3)*(1.96**1)) + c8*(((fixed_b/Tnum)**3)*(1.96**2)) + c9*(((fixed_b/Tnum)**3)*(1.96**3))    

def compute_c_coefficients(omegas, cv, y, k_val, rho_k_til, Q_2):
    """
    Compute the coefficients c_0_nd, c_1_nd, and c_2_nd.
    
    Args:
    - omegas (tuple): Omega values (omegas_nd11, omegas_nd12, omegas_nd22)
    - cv (float): Value of cv
    - y (np.array): Input data array
    - k_val (int): Parameter k value
    - rho_k_til (float): Value of rho_k_til
    - Q_2 (float): Value of Q_2
    
    Returns:
    - tuple: Coefficients (c_0_nd, c_1_nd, c_2_nd)
    """
    
    omegas_nd11, omegas_nd12, omegas_nd22 = omegas

    T_minus_k = len(y) - k_val
    Q_2_inv_squared = Q_2 ** (-2)
    common_factor = (1 / T_minus_k) * Q_2_inv_squared * (cv ** 2)

    c_2_nd = 1 - common_factor * omegas_nd22
    c_1_nd = common_factor * omegas_nd12 - rho_k_til
    c_0_nd = (rho_k_til ** 2) - common_factor * omegas_nd11

    return c_2_nd, c_1_nd, c_0_nd

def initialize_stats_dict():
    return {
        "count_complex": 0,
        "count_real": 0,
        "count_cases": np.zeros((4, 1)),
        "ECP_sum_cases": np.zeros((4, 1)),
        "CI_set_cases": [[], [], [], []],
        "lower_root_set_cases": [[], [], [], []],
        "upper_root_set_cases": [[], [], [], []],
        "spj_band_sum_cases": np.zeros((4, 1)),
    }

def update_stats_SW_cases_dict(lower, upper, roots, c_coef, a_val, Tnum, stats_dict):
    """
    Update results

    Args:
    - lower : lower bound of confidence for unrestricted estimator
    - upper : upper bound of confidence for unrestricted estimator
    - roots (tuple) : Roots of the equation (root1, root2)
    - c_coef (tuple) : Coefficients (c_0_nd, c_1_nd, c_2_nd)
    - a_val (float) : null value
    - Tnum : len(y) - k; length of y minus lag k
    
    - stats_dict (dict): Dictionary containing statistics ex) stats['unres_nd_normal_for_res_nd_normal']
    
    Returns:
    - Updated stats_dict
    """
    
    ## saving part
    is_first_root_complex = roots[0].imag != 0 
    is_second_root_complex = roots[1].imag != 0
    
    if is_first_root_complex or is_second_root_complex:
        stats_dict['count_complex'] += 1
        index = 2 if c_coef[0] < 0 else 3
    else:
        stats_dict['count_real'] += 1
        index = 0 if c_coef[0] > 0 else 1
    
    # counts
    stats_dict['count_cases'][index] += 1
    
    # Truncation
    if abs(lower) > 1:
        lower = np.sign(lower)
    if abs(upper) > 1:
        upper = np.sign(upper)

    # ECP update
    if lower <= a_val <= upper:
        stats_dict['ECP_sum_cases'][index] += 1

    # CI and sets update
    CI = abs(upper - lower)
    stats_dict['CI_set_cases'][index].append(CI)                

    stats_dict['lower_root_set_cases'][index].append(lower)
    stats_dict['upper_root_set_cases'][index].append(upper)

    # Bandwidth saving
    #stats_dict['spj_band_sum_cases'][index] += M_band / Tnum
                    
    return stats_dict

def update_stats_unrestricted(lower, upper, a_val, ECP_sum, CI_set, lower_set, upper_set):
    """
    Update the lower and upper bounds, check and update ECP, and append values to the CI, lower, and upper lists.
    """
    
    # Truncation
    if abs(lower) > 1:
        lower = np.sign(lower)
    if abs(upper) > 1:
        upper = np.sign(upper)
    
    # ECP update
    if lower <= a_val <= upper:
        ECP_sum += 1
        
    # CI and sets update
    CI = abs(upper - lower)
    CI_set.append(CI)
    
    lower_set.append(lower)
    upper_set.append(upper)
    
    return ECP_sum, CI_set, lower_set, upper_set

def update_stats_unres_cases_dict(lower, upper, roots, c_coef, a_val, Tnum, M_band, stats_dict):
    """
    Update results

    Args:
    - lower : lower bound of confidence for unrestricted estimator
    - upper : upper bound of confidence for unrestricted estimator
    - roots (tuple) : Roots of the equation (root1, root2)
    - c_coef (tuple) : Coefficients (c_0_nd, c_1_nd, c_2_nd)
    - a_val (float) : null value
    - Tnum : len(y) - k; length of y minus lag k
    - M_band: bandwidth M (usually SPJ)
    
    - stats_dict (dict): Dictionary containing statistics ex) stats['unres_nd_normal_for_res_nd_normal']
    
    Returns:
    - Updated stats_dict
    """
    
    ## saving part
    is_first_root_complex = roots[0].imag != 0 
    is_second_root_complex = roots[1].imag != 0
    
    if is_first_root_complex or is_second_root_complex:
        stats_dict['count_complex'] += 1
        index = 2 if c_coef[0] < 0 else 3
    else:
        stats_dict['count_real'] += 1
        index = 0 if c_coef[0] > 0 else 1
    
    # counts
    stats_dict['count_cases'][index] += 1
    
    # Truncation
    if abs(lower) > 1:
        lower = np.sign(lower)
    if abs(upper) > 1:
        upper = np.sign(upper)

    # ECP update
    if lower <= a_val <= upper:
        stats_dict['ECP_sum_cases'][index] += 1

    # CI and sets update
    CI = abs(upper - lower)
    stats_dict['CI_set_cases'][index].append(CI)                

    stats_dict['lower_root_set_cases'][index].append(lower)
    stats_dict['upper_root_set_cases'][index].append(upper)

    # Bandwidth saving
    stats_dict['spj_band_sum_cases'][index] += M_band / Tnum
                    
    return stats_dict

def update_stats_restricted_cases_dict(roots, c_coef, a_val, Tnum, M_band, stats_dict):
    """
    Update results

    Args:
    - roots (tuple) : Roots of the equation (root1, root2)
    - c_coef (tuple) : Coefficients (c_0_nd, c_1_nd, c_2_nd)
    - a_val (float) : null value
    - Tnum : len(y) - k; length of y minus lag k
    - M_band: bandwidth M (usually SPJ)
    - stats_dict (dict) : Dictionary containing statistical information, ex) stats['res_nd_normal']

    Returns:
    - Updated stats_dict
    """
    
    is_first_root_complex = roots[0].imag != 0
    is_second_root_complex = roots[1].imag != 0
    
    if is_first_root_complex or is_second_root_complex:
        stats_dict["count_complex"] += 1
        index = 2 if c_coef[0] < 0 else 3
        
        if index == 3:
            raise ValueError("Open upward, vertex is above 0, where there is no such collection of null value rejecting null hypothesis. Impossible.")
        
        stats_dict["count_cases"][index] += 1
        stats_dict["ECP_sum_cases"][index] += 1
        stats_dict["spj_band_sum_cases"][index] += M_band / Tnum
        
        upper = 1
        lower = -1
        CI = abs(upper - lower)
        
        stats_dict["CI_set_cases"][index].append(CI)
        stats_dict["lower_root_set_cases"][index].append(lower)
        stats_dict["upper_root_set_cases"][index].append(upper)

    else:
        stats_dict["count_real"] += 1
        index = 0 if c_coef[0] > 0 else 1

        stats_dict["count_cases"][index] += 1

        lower = min(roots[0].real, roots[1].real)
        upper = max(roots[0].real, roots[1].real)

        # truncation
        if abs(lower) > 1:
            lower = np.sign(lower)
        if abs(upper) > 1:
            upper = np.sign(upper)

        if index == 0:
            CI = abs(upper - lower)
        else:
            CI = abs(lower + 1) + abs(1 - upper) # disjoint case where index = 1

        stats_dict["CI_set_cases"][index].append(CI)
        stats_dict["lower_root_set_cases"][index].append(lower)
        stats_dict["upper_root_set_cases"][index].append(upper)
        
        if index == 0:
            if lower <= a_val <= upper:
                stats_dict["ECP_sum_cases"][index] += 1
        elif index == 1:
            if -1 <= a_val <= lower or upper <= a_val <= 1:
                stats_dict["ECP_sum_cases"][index] += 1

        stats_dict["spj_band_sum_cases"][index] += M_band / Tnum
    
    return stats_dict

def calculate_list_means(class_score):
    """
    Calculate the mean for each inner list in a list of lists.
    Returns a list of means, where 'None' indicates an empty inner list.
    """
    mean_scores = []
    for class_list in class_score:
        if len(class_list) == 0:
            mean_scores.append(None)  # or 0 or "N/A" or whatever you like
        else:
            avg = mean(class_list)
            mean_scores.append(avg)
    return mean_scores

def calculate_whole_list_mean(class_score):
    """
    Calculate the mean of all elements across a list of lists.
    Returns 'None' if there are no elements.
    """
    total_elements = 0  # For the whole list mean
    total_sum = 0       # Sum of all elements for the whole list mean
    for class_list in class_score:
        total_elements += len(class_list)
        total_sum += sum(class_list)
        
    if total_elements == 0:
        return None  # or 0 or "N/A" or whatever you like
    else:
        return total_sum / total_elements