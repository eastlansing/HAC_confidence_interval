import mpmath
import math
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
import scipy.stats
import matplotlib.pyplot as plt
#from statsmodels.compat.python import Literal, lzip
from collections import defaultdict
import datetime
import statsmodels.tsa.arima.model
from statsmodels.tsa.arima.model import ARIMA
import copy
import argparse
import pickle
import os
from copy import deepcopy

from bart import *
from DGP_burn import *
from HAC import *
from power_ftn import *

def save_metrics_HAC(stats, data_dep_categories, constb_categories, band_set, replication):
    result_dict = {}  # Initialize an empty dictionary to store results
    
    for prefix1 in data_dep_categories:
        # Initialize sub-dictionaries for each category if not already present
        if prefix1 not in result_dict:
            result_dict[prefix1] = {}
            for metric1 in ['rej', 'var', 'band']:
                result_dict[prefix1][metric1] = {}

        stats_case = stats[prefix1]
        replication_factor = replication  # Assuming this is a constant you have

        # Calculate and store the results for this prefix in the result_dict
        result_dict[prefix1]['rej'] = stats_case['rej'] / replication_factor
        result_dict[prefix1]['var'] = stats_case['var'] / replication_factor
        result_dict[prefix1]['band'] = stats_case['band'] / replication_factor
        
    for prefix2 in constb_categories:
        # Initialize sub-dictionaries for each category if not already present
        if prefix2 not in result_dict:
            result_dict[prefix2] = {}
            for b_const in range(len(band_set)):
                result_dict[prefix2][b_const] = {}
                for metric1 in ['rej', 'var', 'band']:
                    result_dict[prefix2][b_const][metric1] = {}
        
        for b_const in range(len(band_set)):
            stats_case = stats[prefix2][b_const]

                # Calculate and store the results for this prefix in the result_dict
            result_dict[prefix2][b_const]['rej'] = stats_case['rej'] / replication_factor
            result_dict[prefix2][b_const]['var'] = stats_case['var'] / replication_factor

    return result_dict

def save_metrics_comp(stats_comp, comp_categories):
    result_dict = {}  # Initialize an empty dictionary to store results
    
    for prefix1 in comp_categories:
        # Initialize sub-dictionaries for each category if not already present
        if prefix1 not in result_dict:
            result_dict[prefix1] = {}
            for metric1 in ['rej', 'var']:
                result_dict[prefix1][metric1] = {}

        stats_case = stats_comp[prefix1]

        # Calculate and store the results for this prefix in the result_dict
        result_dict[prefix1]['rej'] = stats_case['rej'] / stats_case['replication']
        result_dict[prefix1]['var'] = stats_case['var'] / stats_case['replication']

    return result_dict

def save_metrics_fixedK(stats_fixedK, fixedK_categories, replication):
    result_dict = {}  # Initialize an empty dictionary to store results
    
    for prefix1 in fixedK_categories:
        # Initialize sub-dictionaries for each category if not already present
        if prefix1 not in result_dict:
            result_dict[prefix1] = {}
            for metric1 in ['rej', 'var', 'Khat']:
                result_dict[prefix1][metric1] = {}

        stats_case = stats_fixedK[prefix1]
        replication_factor = replication 

        # Calculate and store the results for this prefix in the result_dict
        result_dict[prefix1]['rej'] = stats_case['rej'] / replication_factor
        result_dict[prefix1]['var'] = stats_case['var'] / replication_factor
        result_dict[prefix1]['Khat'] = stats_case['Khat'] / replication_factor

    return result_dict


# Create a mapping of DGP types to classes
DGP_CLASSES = {
    'AR1-IID': AR1_iid,
    'AR1-MDS': AR1_MDS,
    'AR1-GARCH': AR1_GARCH,
    'AR1-WN': AR1_WN,
    'AR1-WN-gam-v': AR1_WN_gam_v,
    'AR1-WN-gam-v-minus': AR1_WN_gam_v_minus,
    'AR1-non-md1': AR1_non_md1,
    'AR1-NLMA': AR1_NLMA,
    'AR1-bilinear': AR1_bilinear,
    'ARMA11-IID': ARMA11_iid,
    'ARMA11-MDS': ARMA11_MDS,
    'ARMA11-GARCH': ARMA11_GARCH,
    'ARMA11-WN': ARMA11_WN,
    'ARMA11-WN-gam-v': ARMA11_WN_gam_v,
    'ARMA11-WN-gam-v-minus': ARMA11_WN_gam_v_minus,
    'ARMA11-non-md1': ARMA11_non_md1,
    'ARMA11-NLMA': ARMA11_NLMA,
    'ARMA11-bilinear': ARMA11_bilinear,
    'MA1-IID': MA1_iid,
    'MA1-MDS': MA1_MDS,
    'MA1-GARCH': MA1_GARCH,
    'MA1-WN': MA1_WN,
    'MA1-WN-gam-v': MA1_gam_v,
    'MA1-WN-gam-v-minus': MA1_gam_v_minus,
    'MA1-non-md1': MA1_non_md1,
    'MA1-NLMA': MA1_NLMA,
    'MA1-bilinear': MA1_bilinear,
}

# Define your data generation function
def generate_data(dgp_type: str, **kwargs: Any) -> np.ndarray:
    dgp_class = DGP_CLASSES.get(dgp_type)
    if dgp_class is None:
        raise ValueError(f"Unknown DGP type: {dgp_type}")

    dgp_instance = dgp_class(**kwargs)
    return dgp_instance.generate()

# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DGP', type=str, default='AR1-IID')
    parser.add_argument('--lag_set', type=str, default='1')
    parser.add_argument('--size_set', type=str, default='50,100')
    parser.add_argument('--null_phi', type=float, default=0.5)
    parser.add_argument('--delta_set', type=str, default='0.62,0.59,0.56,0.53,0.47,0.44,0.41,0.38')
    parser.add_argument('--replication', type=int, default=1000)
    parser.add_argument('--size_adj_replication', type=int, default=5000)
    parser.add_argument('--two_sided_sig_level', type=float, default=0.025)
    
    args = parser.parse_args()

    # Convert comma-separated strings to lists of integers or floats
    rho_set = list(map(int, args.lag_set.split(',')))
    size_set = list(map(int, args.size_set.split(',')))
    phi_set = list(map(float, args.delta_set.split(','))) # this assigns delta in our working paper that local alternative values
    
    # Assign the integer value for replication
    replication = args.replication
    cv_replication = args.size_adj_replication
    null = args.null_phi
    
    #Assign significance level (two sided)
    sig_level = args.two_sided_sig_level
      
    # Assign the string value for DGP type
    DGP_type = args.DGP
    print(DGP_type)
    
    ## for example
    #phi_set = [0.1, 0.5, 0.9]
    #size_set = [100, 200, 300]
    #replication = 10
    #DGP_type = 'AR-IID'
    
    band_set = [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
    #data_set = {}
    
    # set dicts
    result_save_hac_opt1 = {}
    result_save_hac_opt2 = {}
    result_save_hac_opt3 = {}
    result_save_comp = {}
    result_save_fixedK_opt1 = {}
    result_save_fixedK_opt2 = {}
    result_save_fixedK_opt3 = {}
    
    #unchanged parameters
    #null = 0.5
    theta = 0.7
    
    if null != 0:
        raise ValueError("Error: null_phi must be zero for this .py program.")

    if not DGP_type.startswith('AR1'):
        raise ValueError("Error: DGP_type must start with 'AR1' for this .py program")

    for m in range(len(rho_set)):
        for j in range(len(size_set)):

            data_dep_categories = ['SPJ_normal', 'SPJ_fixedb', 'AD_normal', 'AD_fixedb']
            stats = {category: initialize_stats_dict_power() for category in data_dep_categories}

            constb_categories = ['constb_normal', 'constb_fixedb']
            constb_stats = {category: {} for category in constb_categories}

            for w1 in constb_categories:
                for w2 in range(len(band_set)):
                    constb_stats[w1][w2] = initialize_stats_dict_power()

            #update
            stats.update(constb_stats)

            stats_opt1 = deepcopy(stats)
            stats_opt2 = deepcopy(stats)
            stats_opt3 = deepcopy(stats)

            ## getting stats comp dict to save results 
            comp_categories = ['Bart', 'GBMA', 'GBW', 'Taylor']
            stats_comp = {category: initialize_stats_dict_power_comp() for category in comp_categories}

            ## FixedK
            fixedK_categories = ['FixedK']
            stats_fixedK = {category: initialize_stats_dict_power_fixedK() for category in fixedK_categories}

            stats_fixedK_opt1 = deepcopy(stats_fixedK)
            stats_fixedK_opt2 = deepcopy(stats_fixedK)
            stats_fixedK_opt3 = deepcopy(stats_fixedK)

            ## calculate true autocorrelation for null, H_0 value
            p = rho_set[m] # lag k

            if DGP_type.startswith('AR1'):
                true_acf1 = null**p 
            elif DGP_type.startswith('ARMA11'):
                true_acf1 = (((1+theta*null)*(null+theta))/(1+2*theta*null+theta**2))*(null**(p-1))
            elif DGP_type.startswith('MA1'):
                if p == 1:
                    true_acf1 = null/(1+(null**2)) ## for MA1
                else:
                    true_acf1 = 0
            else:
                raise ValueError("Invalid DGP_type")

            ## Calculate size adjusted critical values for each method      
            for i in range(50000,50000+cv_replication):
                np.random.seed(size_set[j]*100 + i)
                data = generate_data(DGP_type, phi=null, size=size_set[j], shape=0.3, scale=0.4, ma1=theta)
                #p = rho_set[m]

                y, x, nmp, X, coeffs = reg_estimating_equation_v1(data=np.copy(data), lag=p) # first estimating equation

                ori_rho_hat = sm.tsa.stattools.acf(np.copy(data), nlags=p, fft=False)[p] # cal from traditional autocorrelation calculation


                ##### start calculating variance/t-stat for the empirical null rejction probabilities #####
                #null not imposed (opt1), null imposed-nodemean (opt2), null imposed-demean (opt3)
                uhat_opt1 = y - np.dot(X,coeffs)  # null is not imposed
                uhat_opt2 = (y - np.mean(y)) - true_acf1*(x - np.mean(x)) # null imposed
                uhat_opt3 = (y - np.mean(y)) - true_acf1*(x - np.mean(x)) # null imposed + demeaning

                # calculate vhat    
                vhat_opt1 = np.copy(X*uhat_opt1)
                vhat_opt2 = np.copy(X*uhat_opt2)
                vhat_opt3 = np.copy(X*uhat_opt3)

                vhat_opt3 = vhat_opt3 - vhat_opt3.mean(axis=0) ### Stock Watson demeaning

                # calculate t statistics and append
                spj_function = create_spj_function(w=10)
                coeffs1 = np.float64(coeffs[1])
                stats_opt1 = append_tstat_size_adj_HAC(np.copy(vhat_opt1), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt1, spj_function)
                stats_opt2 = append_tstat_size_adj_HAC(np.copy(vhat_opt2), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt2, spj_function)
                stats_opt3 = append_tstat_size_adj_HAC(np.copy(vhat_opt3), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt3, spj_function)


                # calculate t statistics and append for fixed K
                W_mat = np.ones(X.shape[1],)
                W_mat[0] = 0 # intercept weighting is zero
                integer_type = 'even'
                stats_fixedK_opt1 = append_tstat_size_adj_fixedK(np.copy(vhat_opt1), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt1)
                stats_fixedK_opt2 = append_tstat_size_adj_fixedK(np.copy(vhat_opt2), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt2)
                stats_fixedK_opt3 = append_tstat_size_adj_fixedK(np.copy(vhat_opt3), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt3)

                DGP_two_chars = DGP_type[:2]
                if DGP_two_chars == 'AR':
                    var_ba = 1
                    #var_ba = AR_Bart(y=data,ij=rho_set[m])
                #elif DGP_two_chars == 'MA':
                #    var_ba = MA_Bart(MA_n=1,y=data,ij=rho_set[m])
                #else:
                #    raise ValueError("Invalid DGP_type")

                ## Generalized Bartlett - MA1
                var_ba_MA1 = MA_Bart(MA_n=1,y=np.copy(data),ij=rho_set[m])

                arma_mod = ARIMA(np.copy(data), order=(0, 0, 1),trend="c") # MA1 with constant
                arma_res = arma_mod.fit()
                eps_hat_GB = arma_res.resid
                eps_hat_GB = np.reshape(eps_hat_GB,(eps_hat_GB.shape[0],1))

                var_gba = var_ba_MA1 + MA_GB(MA_n=1,y=np.copy(data),eps=eps_hat_GB,ij=rho_set[m])

                ## Generalized Bartlett - White noise
                var_bam = 1
                var_gbam = var_bam + White_GB(y=np.copy(data), lag=p)

                ## Taylor

                var_taylor = var_taylor_cal(y=np.copy(data),lag=p,ori_rho_hat=ori_rho_hat)

                ## cal real variance and update result to dictionary

                var_ba_real = var_ba/(nmp+p)
                var_gbma_real = var_gba/(nmp+p)
                var_gbw_real = var_gbam/(nmp+p)

                var_list = [var_ba_real, var_gbma_real, var_gbw_real, var_taylor] #['Bart', 'GBMA', 'GBW', 'Taylor'] order
                stats_comp = append_tstat_size_adj_comp(ori_rho_hat, true_acf1, var_list, stats_comp)

            ## Compute power for each alternative values  
            for q in range(len(phi_set)):
                ## update rejection prob to zero for new phi_set value
                stats_opt1 = update_dict_to_zero_HAC(stats_opt1, data_dep_categories, constb_categories, band_set.copy())
                stats_opt2 = update_dict_to_zero_HAC(stats_opt2, data_dep_categories, constb_categories, band_set.copy())
                stats_opt3 = update_dict_to_zero_HAC(stats_opt3, data_dep_categories, constb_categories, band_set.copy())

                stats_fixedK_opt1 = update_dict_to_zero(stats_fixedK_opt1)
                stats_fixedK_opt2 = update_dict_to_zero(stats_fixedK_opt2)
                stats_fixedK_opt3 = update_dict_to_zero(stats_fixedK_opt3)

                stats_comp = update_dict_to_zero(stats_comp)

                ## calculate power
                for i in range(50000,50000+replication):
                    np.random.seed(size_set[j]*100 + i)
                    #alter_val = null + phi_set[q]
                    alter_val = phi_set[q]
                    data = generate_data(DGP_type, phi=alter_val, size=size_set[j], shape=0.3, scale=0.4, ma1=theta) # DGP under alternative 

                    y, x, nmp, X, coeffs = reg_estimating_equation_v1(data=np.copy(data), lag=p) # first estimating equation

                    ori_rho_hat = sm.tsa.stattools.acf(np.copy(data), nlags=p, fft=False)[p] # cal from traditional autocorrelation calculation

                    ##### start calculating variance/t-stat for the empirical null rejction probabilities #####
                    #null not imposed (opt1), null imposed-nodemean (opt2), null imposed-demean (opt3)
                    uhat_opt1 = y - np.dot(X,coeffs)  # null is not imposed
                    uhat_opt2 = (y - np.mean(y)) - true_acf1*(x - np.mean(x)) # null imposed
                    uhat_opt3 = (y - np.mean(y)) - true_acf1*(x - np.mean(x)) # null imposed + demeaning

                    # calculate vhat    
                    vhat_opt1 = np.copy(X*uhat_opt1)
                    vhat_opt2 = np.copy(X*uhat_opt2)
                    vhat_opt3 = np.copy(X*uhat_opt3)

                    vhat_opt3 = vhat_opt3 - vhat_opt3.mean(axis=0) ### Stock Watson demeaning

                    ### HAC - normal, fixedb but taking size adjusted critical values here
                    spj_function = create_spj_function(w=10)
                    coeffs1 = np.float64(coeffs[1])
                    stats_opt1 = update_stats_power_dict(np.copy(vhat_opt1), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt1, spj_function, sig_level)
                    stats_opt2 = update_stats_power_dict(np.copy(vhat_opt2), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt2, spj_function, sig_level)
                    stats_opt3 = update_stats_power_dict(np.copy(vhat_opt3), np.copy(X), nmp, true_acf1, coeffs1, band_set.copy(), stats_opt3, spj_function, sig_level)



                    ### EWC - FixedK with MSE optimal K, taking size adjusted critical values here
                    W_mat = np.ones(X.shape[1],)
                    W_mat[0] = 0 # intercept weighting is zero
                    integer_type = 'even'
                    stats_fixedK_opt1 = update_stats_fixedK_power_dict(np.copy(vhat_opt1), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt1,sig_level)
                    stats_fixedK_opt2 = update_stats_fixedK_power_dict(np.copy(vhat_opt2), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt2,sig_level)
                    stats_fixedK_opt3 = update_stats_fixedK_power_dict(np.copy(vhat_opt3), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt3,sig_level)

                    #### other methods for variance for autocorrelation test calculations ####
                    ## Bartlett for given DGP
                    DGP_two_chars = DGP_type[:2]
                    if DGP_two_chars == 'AR':
                        var_ba = 1
                        #var_ba = AR_Bart(y=data,ij=rho_set[m])
                    #elif DGP_two_chars == 'MA':
                    #    var_ba = MA_Bart(MA_n=1,y=data,ij=rho_set[m])
                    #else:
                    #    raise ValueError("Invalid DGP_type")

                    ## Generalized Bartlett - MA1
                    var_ba_MA1 = MA_Bart(MA_n=1,y=np.copy(data),ij=rho_set[m])

                    arma_mod = ARIMA(np.copy(data), order=(0, 0, 1),trend="c") # MA1 with constant
                    arma_res = arma_mod.fit()
                    eps_hat_GB = arma_res.resid
                    eps_hat_GB = np.reshape(eps_hat_GB,(eps_hat_GB.shape[0],1))

                    var_gba = var_ba_MA1 + MA_GB(MA_n=1,y=np.copy(data),eps=eps_hat_GB,ij=rho_set[m])

                    ## Generalized Bartlett - White noise
                    var_bam = 1
                    var_gbam = var_bam + White_GB(y=np.copy(data), lag=p)

                    ## Taylor

                    var_taylor = var_taylor_cal(y=np.copy(data),lag=p,ori_rho_hat=ori_rho_hat)

                    ## cal real variance and update result to dictionary

                    var_ba_real = var_ba/(nmp+p)
                    var_gbma_real = var_gba/(nmp+p)
                    var_gbw_real = var_gbam/(nmp+p)

                    var_list = [var_ba_real, var_gbma_real, var_gbw_real, var_taylor] #['Bart', 'GBMA', 'GBW', 'Taylor'] order
                    stats_comp = update_stats_comp_power_dict(ori_rho_hat, true_acf1, var_list, stats_comp, sig_level)

                result_save_hac_opt1[m,q,j] = save_metrics_HAC(stats_opt1, data_dep_categories, constb_categories, band_set.copy(), replication)
                result_save_hac_opt2[m,q,j] = save_metrics_HAC(stats_opt2, data_dep_categories, constb_categories, band_set.copy(), replication)
                result_save_hac_opt3[m,q,j] = save_metrics_HAC(stats_opt3, data_dep_categories, constb_categories, band_set.copy(), replication)
                result_save_comp[m,q,j] = save_metrics_comp(stats_comp, comp_categories)
                result_save_fixedK_opt1[m,q,j] = save_metrics_fixedK(stats_fixedK_opt1, fixedK_categories, replication)
                result_save_fixedK_opt2[m,q,j] = save_metrics_fixedK(stats_fixedK_opt2, fixedK_categories, replication)
                result_save_fixedK_opt3[m,q,j] = save_metrics_fixedK(stats_fixedK_opt3, fixedK_categories, replication)
                
    result_all_dict = {'result_save_hac_opt1': result_save_hac_opt1, 'result_save_hac_opt2': result_save_hac_opt2, 
                  'result_save_hac_opt3': result_save_hac_opt3, 'result_save_comp': result_save_comp, 
                   'result_save_fixedK_opt1' : result_save_fixedK_opt1, 'result_save_fixedK_opt2': result_save_fixedK_opt2,
                   'result_save_fixedK_opt3' : result_save_fixedK_opt3}

    # Check if the folder exists, and create one if it doesn't
    if not os.path.exists('power_result'):
        os.makedirs('power_result')

    # Dumping the master_dict
    filename = f'power_result/result_power_{DGP_type}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(result_all_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(null, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(phi_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(size_set, f, protocol=pickle.HIGHEST_PROTOCOL)