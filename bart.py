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

def AR_Bart(y,ij):
    y = np.copy(y)
    
    i = np.copy(ij)
    rho_y = sm.tsa.stattools.acf(y,nlags=1,fft=False)
    rho_1 = rho_y[1]
    w_ii_result = (1-(rho_1**(2*i)))*(1+rho_1**2)/(1-rho_1**2) - 2*i*(rho_1**(2*i))
        
    return w_ii_result

def MA_Bart(MA_n,y,ij):
    y = np.copy(y)
    
    q = np.copy(MA_n)
    if ij <= q:
        rho_y = sm.tsa.stattools.acf(y,nlags=1,fft=False)
        w_ii_result = 1-3*(rho_y[1]**2)+4*(rho_y[1]**4)
    
    if ij > q:
        w_ii = 0
        rho_y = sm.tsa.stattools.acf(y,nlags=q,fft=False)
        for l in range(-abs(q),abs(q)+1): # -1, 0, 1 for MA(1)
            w_ii = w_ii + (rho_y[abs(l)]**2)
        w_ii_result = w_ii
        
    return w_ii_result  

def MA_GB(MA_n,y,eps,ij):
    y = np.copy(y)
    eps = np.copy(eps)
    
    q = np.copy(MA_n) # MA_n = 1 if assuming y is following MA(1)
    if ij <= q:
        gam_eps = sm.tsa.stattools.acovf(eps, adjusted=False, demean=True, fft=False, nlag=1)[0]
        gam_eps_sq = sm.tsa.stattools.acovf(eps**2, adjusted=False, demean=True, fft=False, nlag=2)
        rho_y = sm.tsa.stattools.acf(y,nlags=1,fft=False)
        aa1 = (gam_eps_sq[1]*(1-4*(rho_y[1]**2)+4*(rho_y[1]**4)))/(gam_eps**2)
        aa2 = (gam_eps_sq[2]*(rho_y[1]**2))/(gam_eps**2)
        w_ii_result = aa1 + aa2 ### MA1, w_ii
    
    if ij > q:
        w_ii = 0
        rho_y = sm.tsa.stattools.acf(y,nlags=q,fft=False)
        gam_eps_sq = sm.tsa.stattools.acovf(eps**2, adjusted=False, demean=True, fft=False, nlag=10)
        for l in range(-abs(q),abs(q)+1): # -1, 0, 1 for MA(1)
            w_ii = w_ii + gam_eps_sq[abs(ij-l)]*(rho_y[abs(l)]**2)
        gam_eps = sm.tsa.stattools.acovf(eps, adjusted=False, demean=True, fft=False, nlag=1)[0]
        w_ii_result = w_ii/(gam_eps**2)
        
    return w_ii_result

def White_GB(y, lag):
    y = np.copy(y)
    
    data = np.copy(y)
    nd = np.copy(data)
    p = np.copy(lag)
    nom = sm.tsa.stattools.acovf(nd**2, adjusted=False, nlag=p, demean=True, fft=False)[p]
    denom = (sm.tsa.stattools.acovf(nd, adjusted=False, nlag=1, demean=True, fft=False)[0])**2 # both 1/n times for autocov
        
    return nom/denom

def var_taylor_cal(y,lag,ori_rho_hat):
    data = np.copy(y)
    p = np.copy(lag)
    T_data = len(data) # T

    y2 = np.copy(data[p:T_data,:]) - np.mean(data)
    x2 = np.copy(data[0:T_data-p,:]) - np.mean(data)

    t_tilda_nom = np.sum(y2*x2)
    t_tilda_denom = np.sqrt(np.sum((y2*x2)**2))
    t_tilda = t_tilda_nom/t_tilda_denom
    c_hat = t_tilda/ori_rho_hat
    var_taylor = 1/(c_hat**2)
    
    return var_taylor

def stat_pack_var(y,lag_number):
    data = np.copy(y)
    acf2 = sm.tsa.stattools.acf(x=data, nlags=lag_number,fft=False)
    nobs = len(data)
    
    varacf = np.ones_like(acf2) / nobs
    varacf[0] = 0
    varacf[1] = 1.0 / nobs
    varacf[2:] *= 1 + 2 * np.cumsum(acf2[1:-1] ** 2)
    
    return varacf