import mpmath
import math
import pandas as pd
import numpy as np
#import csv
import statsmodels.api as sm
import scipy.stats
import heapq
from copy import deepcopy
from copy import copy
from HAC import *

def process_tstat_only_HAC(vhat, M_n, X, nmp, true_acf1, coeffs1):
    vhat = np.copy(vhat)
    X = np.copy(X)
    
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = newLRV_vec(vhat, M_n)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp
    #fixed_cv = calc_fixed_cv_org(M_n, nmp, fixedb_coeffs)
    t_stat = (float(coeffs1) - true_acf1) / np.sqrt(np.float64(var[1, 1]))

    return t_stat

def process_tstat_var_HAC(vhat, M_n, X, nmp, true_acf1, coeffs1):
    vhat = np.copy(vhat)
    X = np.copy(X)
    
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = newLRV_vec(vhat, M_n)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp
    #fixed_cv = calc_fixed_cv_org(M_n, nmp, fixedb_coeffs)
    t_stat = (float(coeffs1) - true_acf1) / np.sqrt(np.float64(var[1, 1]))

    return t_stat, np.float64(var[1, 1])

def append_tstat_size_adj_HAC(vhat, X, nmp, true_acf1, coeffs1, band_set, stats, spj_band_function):
    vhat = np.copy(vhat)
    X = np.copy(X)
    band_set = copy(band_set)
    
    # For constb
    for b_coord in range(len(band_set)):
        M_n_constb = np.float64(band_set[b_coord]) * len(X)
        t_stat_constb = process_tstat_only_HAC(vhat, M_n_constb, X, nmp, true_acf1, coeffs1)
        
        stats['constb_normal'][b_coord]['t_stat_append'].append(t_stat_constb)
        stats['constb_fixedb'][b_coord]['t_stat_append'].append(t_stat_constb)

    # For SPJ
    M_n_spj = spj_band_function(vhat)  # Here, we use the passed-in function to compute M_n_spj
    t_stat_spj = process_tstat_only_HAC(vhat, M_n_spj, X, nmp, true_acf1, coeffs1)

    stats['SPJ_normal']['t_stat_append'].append(t_stat_spj)
    stats['SPJ_fixedb']['t_stat_append'].append(t_stat_spj)

    # For AD
    M_n_ad = AD_band(vhat)
    t_stat_ad = process_tstat_only_HAC(vhat, M_n_ad, X, nmp, true_acf1, coeffs1)
    
    stats['AD_normal']['t_stat_append'].append(t_stat_ad)
    stats['AD_fixedb']['t_stat_append'].append(t_stat_ad)
    
    return stats

def append_tstat_size_adj_fixedK(vhat, X, true_acf1, coeffs1, W_mat, integer_type, stats_fixedK):
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
    
    stats_fixedK['FixedK']['t_stat_append'].append(t_stat_fixedK)

    return stats_fixedK

def append_tstat_size_adj_comp(ori_rho_hat, true_acf1, var_list, stats_comp):
    var_list = copy(var_list)
    
    # Ensure the length and order of var_list and the keys in stats_comp are the same
    if len(var_list) != len(stats_comp):
        raise ValueError("The length of var_list and the number of keys in stats_comp must be the same.")
    
    # Iterate over each var and its corresponding key in stats_comp
    for var, key in zip(var_list, stats_comp.keys()):
        t_stat = (float(ori_rho_hat) - true_acf1) / np.sqrt(var)
        if var > 0:
            stats_comp[key]['t_stat_append'].append(t_stat) 

    return stats_comp

def upper_lower_size_adj_cv(my_list, sig_level):
    my_list = copy(my_list)
    
    num_elements = round(len(my_list) * sig_level)
    
    # Check if the list has enough elements for the given significance level
    if len(my_list) < 40:
        raise ValueError("The list is too small for the given significance level, e.g. two sided 5%")
    
    upper_cv = heapq.nlargest(num_elements, my_list)[-1]
    lower_cv = heapq.nsmallest(num_elements, my_list)[-1]

    return upper_cv, lower_cv

def update_stats_power_dict(vhat, X, nmp, true_acf1, coeffs1, band_set, stats, spj_band_function, sig_level):
    vhat = np.copy(vhat)
    X = np.copy(X)
    band_set = copy(band_set)
    
    # sig level = 0.025 if 5% two sided test
    ## For constb
    for b_coord in range(len(band_set)):
        M_n_constb = np.float64(band_set[b_coord]) * len(X)
        t_stat_constb, var_constb = process_tstat_var_HAC(vhat, M_n_constb, X, nmp, true_acf1, coeffs1)
        
        upper_cv_constb, lower_cv_constb = upper_lower_size_adj_cv(my_list=stats['constb_normal'][b_coord]['t_stat_append'], sig_level=sig_level)
        
        if t_stat_constb > upper_cv_constb or t_stat_constb < lower_cv_constb:
            stats['constb_normal'][b_coord]['rej'] += 1
        if t_stat_constb > upper_cv_constb or t_stat_constb < lower_cv_constb:
            stats['constb_fixedb'][b_coord]['rej'] += 1
        
        # save var
        stats['constb_normal'][b_coord]['var'] += var_constb
        stats['constb_fixedb'][b_coord]['var'] += var_constb

    ## For SPJ
    M_n_spj = spj_band_function(vhat)  # Here, we use the passed-in function to compute M_n_spj
    t_stat_spj, var_spj = process_tstat_var_HAC(vhat, M_n_spj, X, nmp, true_acf1, coeffs1)
    
    upper_cv_spj, lower_cv_spj = upper_lower_size_adj_cv(my_list=stats['SPJ_normal']['t_stat_append'], sig_level=sig_level)

    if t_stat_spj > upper_cv_spj or t_stat_spj < lower_cv_spj:
        stats['SPJ_normal']['rej'] += 1
    if t_stat_spj > upper_cv_spj or t_stat_spj < lower_cv_spj:
        stats['SPJ_fixedb']['rej'] += 1
     
    #save var
    stats['SPJ_normal']['var'] += var_spj
    stats['SPJ_fixedb']['var'] += var_spj
    
    #save band
    stats['SPJ_normal']['band'] += M_n_spj/nmp
    stats['SPJ_fixedb']['band'] += M_n_spj/nmp

    ## For AD
    M_n_ad = AD_band(vhat)
    t_stat_ad, var_ad = process_tstat_var_HAC(vhat, M_n_ad, X, nmp, true_acf1, coeffs1)
    
    upper_cv_ad, lower_cv_ad = upper_lower_size_adj_cv(my_list=stats['AD_normal']['t_stat_append'], sig_level=sig_level)

    if t_stat_ad > upper_cv_ad or t_stat_ad < lower_cv_ad:
        stats['AD_normal']['rej'] += 1
    if t_stat_ad > upper_cv_ad or t_stat_ad < lower_cv_ad:
        stats['AD_fixedb']['rej'] += 1
    
    # save var
    stats['AD_normal']['var'] += var_ad
    stats['AD_fixedb']['var'] += var_ad
    
    # save band
    stats['AD_normal']['band'] += M_n_ad/nmp
    stats['AD_fixedb']['band'] += M_n_ad/nmp
    
    return stats

def update_stats_fixedK_power_dict(vhat, X, true_acf1, coeffs1, W_mat, integer_type, stats_fixedK, sig_level):
    
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
    
    upper_cv_fixedK, lower_cv_fixedK = upper_lower_size_adj_cv(my_list=stats_fixedK['FixedK']['t_stat_append'], sig_level=sig_level)
    
    if t_stat_fixedK > upper_cv_fixedK or t_stat_fixedK < lower_cv_fixedK:
        stats_fixedK['FixedK']['rej'] += 1

    #save var
    stats_fixedK['FixedK']['var'] += var_fixedK

    #save band
    stats_fixedK['FixedK']['Khat'] += Khat_D

    
    return stats_fixedK

def update_stats_comp_power_dict(ori_rho_hat, true_acf1, var_list, stats_comp, sig_level):
    var_list = copy(var_list)
    
    # Ensure the length of var_list and the keys in stats_comp are the same
    if len(var_list) != len(stats_comp):
        raise ValueError("The length of var_list and the number of keys in stats_comp must be the same.")
    
    # Iterate over each var and its corresponding key in stats_comp
    for var, key in zip(var_list, stats_comp.keys()):
        t_stat_comp = (float(ori_rho_hat) - true_acf1) / np.sqrt(var)
        
        upper_cv_comp, lower_cv_comp = upper_lower_size_adj_cv(my_list=stats_comp[key]['t_stat_append'], sig_level=sig_level)
        
        if var > 0:
            stats_comp[key]['replication'] += 1
            if t_stat_comp > upper_cv_comp or t_stat_comp < lower_cv_comp:
                stats_comp[key]['rej'] += 1

            # Update variance
            stats_comp[key]['var'] += var

    return stats_comp

def initialize_stats_dict_power():
    return {
        "rej": 0,
        "var": 0,
        "band":0,
        "t_stat_append":[]
    }

def initialize_stats_dict_power_fixedK():
    return {
        "rej": 0,
        "var": 0,
        "Khat": 0,
        "t_stat_append": [],
    }

def initialize_stats_dict_power_comp():
    return {
        "rej": 0,
        "var": 0,
        "replication": 0,
        "t_stat_append": [],
    }

def update_dict_to_zero_HAC(d, data_dep_categories, constb_categories, band_set):
    data_dep_categories = copy(data_dep_categories)
    constb_categories = copy(constb_categories)
    band_set = copy(band_set)
    
    for key in d:
        if key in data_dep_categories:
            if isinstance(d[key], dict):
                for subkey in d[key]:
                    if subkey != 't_stat_append':
                        d[key][subkey] = 0
        elif key in constb_categories:
            for b_coord in range(len(band_set)):
                if isinstance(d[key][b_coord], dict):
                    for sub_subkey in d[key][b_coord]:
                        if sub_subkey != 't_stat_append':
                            d[key][b_coord][sub_subkey] = 0
    return d

def update_dict_to_zero(d):
    for key in d:
        if isinstance(d[key], dict):
            for subkey in d[key]:
                if subkey != 't_stat_append':
                    d[key][subkey] = 0
    return d