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
import argparse
import pickle
import os
from copy import copy
from copy import deepcopy
from bart import *
from DGP_burn import *
from HAC import *

def whLRV(vhat):
    n = len(vhat)
    LRV = 0
    LRV = LRV + autocov_est(vhat,0)
    return LRV

def process_wh_var(vhat, X, nmp):
    vhat = np.copy(vhat)
    X =  np.copy(X)
    
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = whLRV(vhat)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp

    return np.float64(var[1, 1])


def update_stats_comp_dict_with_White(ori_rho_hat, coeffs1, true_acf1, var_list, stats_comp):
    var_list = copy(var_list)
    
    # Ensure the length of var_list and the keys in stats_comp are the same
    if len(var_list) != len(stats_comp):
        raise ValueError("The length of var_list and the number of keys in stats_comp must be the same.")
    
    # Iterate over each var and its corresponding key in stats_comp
    for var, key in zip(var_list, stats_comp.keys()):
        if key == 'White':
            t_stat = (coeffs1 - true_acf1) / np.sqrt(var)
        else:    
            t_stat = (ori_rho_hat - true_acf1) / np.sqrt(var)
        
        if var > 0:
            stats_comp[key]['replication'] += 1
            if np.abs(t_stat) > 1.96:
                stats_comp[key]['rej'] += 1

            # Update variance
            stats_comp[key]['var'] += var

    return stats_comp


def save_metrics_HAC(stats, data_dep_categories, constb_categories, band_set, replication):
    data_dep_categories = copy(data_dep_categories)
    constb_categories = copy(constb_categories)
    band_set = copy(band_set)
    
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
    comp_categories = copy(comp_categories)
    
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
    fixedK_categories = copy(fixedK_categories)
    
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
    parser.add_argument('--phi_set', type=str, default='0.1,0.3,0.5,0.7,0.9,-0.1,-0.3,-0.5,-0.7,-0.9')
    parser.add_argument('--replication', type=int, default=1000)
    
    args = parser.parse_args()

    # Convert comma-separated strings to lists of integers or floats
    rho_set = list(map(int, args.lag_set.split(',')))
    size_set = list(map(int, args.size_set.split(',')))
    phi_set = list(map(float, args.phi_set.split(',')))
    
    # Assign the integer value for replication
    replication = args.replication
      
    # Assign the string value for DGP type
    DGP_type = args.DGP
    print(DGP_type)
    
    #### prepare fixed-b critical values ##
    c1 = 0.43754
    c2 = 0.11912
    c3 = 0.08640
    c4 = 0.49629
    c5 = -0.57879
    c6 = 0.43266
    c7 = 0.02543
    c8 = -0.02379
    c9 = -0.02376

    fixedb_coeffs = [c1, c2, c3, c4, c5, c6, c7, c8, c9]

    s1 = 0.74330
    s2 = -0.30350
    s3 = -0.15730
    s4 = 0.47990
    s5 = -1.34310
    s6 = 0.59400
    s7 = -0.35820
    s8 = 0.74630
    s9 = -0.28670

    fixedb_coeffs_nodemean = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    
    ## for example
    #phi_set = [0.1, 0.5, 0.9]
    #size_set = [100, 200, 300]
    #replication = 10
    #DGP_type = 'AR-IID'
    
    band_set = [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
    data_set = {}
    
    if phi_set != [0]:
        raise ValueError("Error: phi_set must be [0] for this .py program")
        
    if not DGP_type.startswith('AR1'):
        raise ValueError("Error: DGP_type must start with 'AR1' for this .py program")
    
    ## Monte Carlo simulation data generation
    for q in range(len(phi_set)):
        for j in range(len(size_set)):
            for i in range(50000,50000+replication):
                np.random.seed(size_set[j]*100 + int(phi_set[q])*1000000000 + i)
                theta = 0.7
                data = generate_data(DGP_type, phi=phi_set[q], size=size_set[j], shape=0.3, scale=0.4, ma1=theta)
                data_set[q,j,i] = np.copy(data)
    
    # set dicts
    result_save_hac_opt1 = {}
    result_save_hac_opt2 = {}
    result_save_hac_opt3 = {}
    result_save_comp = {}
    result_save_fixedK_opt1 = {}
    result_save_fixedK_opt2 = {}
    result_save_fixedK_opt3 = {}

    for m in range(len(rho_set)):
        for q in range(len(phi_set)):
            for j in range(len(size_set)):

                ## getting stats dict to save results
                data_dep_categories = ['SPJ_normal', 'SPJ_fixedb', 'AD_normal', 'AD_fixedb']
                stats = {category: initialize_stats_dict() for category in data_dep_categories}

                constb_categories = ['constb_normal', 'constb_fixedb']
                constb_stats = {category: {} for category in constb_categories}

                for w1 in constb_categories:
                    for w2 in range(len(band_set)):
                        constb_stats[w1][w2] = initialize_stats_dict()

                #update
                stats.update(constb_stats)

                stats_opt1 = deepcopy(stats)
                stats_opt2 = deepcopy(stats)
                stats_opt3 = deepcopy(stats)

                ## getting stats comp dict to save results 
                comp_categories = ['Bart', 'GBMA', 'GBW', 'Taylor', 'White']
                stats_comp = {category: initialize_stats_dict_comp() for category in comp_categories}

                ## FixedK
                fixedK_categories = ['FixedK']
                stats_fixedK = {category: initialize_stats_dict_fixedK() for category in fixedK_categories}

                stats_fixedK_opt1 = deepcopy(stats_fixedK)
                stats_fixedK_opt2 = deepcopy(stats_fixedK)
                stats_fixedK_opt3 = deepcopy(stats_fixedK)

                for i in range(50000,50000+replication):
                    np.random.seed(size_set[j]*100 + int(phi_set[q])*1000000000 + i)
                    phi = phi_set[q]
                    p = rho_set[m] #rho(p) (lag k in the paper)

                    ## data assign
                    data = np.copy(data_set[q,j,i])
                
                    ## autocorrelation coeffs calculation
                    #T_data = len(data) # T

                    #y = np.copy(data[p:T_data,:])
                    #x = np.copy(data[0:T_data-p,:])

                    #nmp = len(x) # T-p
                    #cons = np.ones((nmp,1))
                    #X = np.hstack((cons,x))
                    
                    #coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y) #cal from the regression type estimating equation
                    
                    y, x, nmp, X, coeffs = reg_estimating_equation_v1(data=np.copy(data), lag=p) # first estimating equation
                    
                    ori_rho_hat = sm.tsa.stattools.acf(np.copy(data), nlags=p, fft=False)[p] # cal from traditional autocorrelation calculation
                    
                    ## calculate true autocorrelation for DGP
                    if DGP_type.startswith('AR1'):
                        true_acf1 = phi**p 
                    elif DGP_type.startswith('ARMA11'):
                        true_acf1 = (((1+theta*phi)*(phi+theta))/(1+2*theta*phi+theta**2))*(phi**(p-1))
                    elif DGP_type.startswith('MA1'):
                        if p == 1:
                            true_acf1 = phi/(1+(phi**2)) ## for MA1
                        else:
                            true_acf1 = 0
                    else:
                        raise ValueError("Invalid DGP_type")
                    

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

                    ### HAC - normal, fixedb
                    spj_function = create_spj_function(w=10)
                    coeffs1 = np.float64(coeffs[1])
                    stats_opt1 = update_stats_dict(np.copy(vhat_opt1), np.copy(X), nmp, true_acf1, coeffs1, fixedb_coeffs.copy(), band_set.copy(), stats_opt1, spj_function)
                    stats_opt2 = update_stats_dict(np.copy(vhat_opt2), np.copy(X), nmp, true_acf1, coeffs1, fixedb_coeffs_nodemean.copy(), band_set.copy(), stats_opt2, spj_function)
                    stats_opt3 = update_stats_dict(np.copy(vhat_opt3), np.copy(X), nmp, true_acf1, coeffs1, fixedb_coeffs.copy(), band_set.copy(), stats_opt3, spj_function)

                    ### EWC - FixedK with MSE optimal K
                    W_mat = np.ones(X.shape[1],)
                    W_mat[0] = 0 # intercept weighting is zero
                    integer_type = 'even'
                    stats_fixedK_opt1 = update_stats_fixedK_dict(np.copy(vhat_opt1), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt1)
                    stats_fixedK_opt2 = update_stats_fixedK_dict(np.copy(vhat_opt2), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt2)
                    stats_fixedK_opt3 = update_stats_fixedK_dict(np.copy(vhat_opt3), np.copy(X), true_acf1, coeffs1, np.copy(W_mat), integer_type, stats_fixedK_opt3)

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
                    
                    ## White
                    
                    var_white = process_wh_var(np.copy(vhat_opt3), np.copy(X), nmp)

                    ## cal real variance and update result to dictionary

                    var_ba_real = var_ba/(nmp+p)
                    var_gbma_real = var_gba/(nmp+p)
                    var_gbw_real = var_gbam/(nmp+p)

                    var_list = [var_ba_real, var_gbma_real, var_gbw_real, var_taylor, var_white] #['Bart', 'GBMA', 'GBW', 'Taylor', 'White'] order
                    stats_comp = update_stats_comp_dict_with_White(ori_rho_hat, coeffs1, true_acf1, var_list, stats_comp)
                    

                    ################# End of Replication ###################
                
                ##### Save result to dictionary #####

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
    if not os.path.exists('null_rej_result'):
        os.makedirs('null_rej_result')

    # Dumping the master_dict
    filename = f'null_rej_result/result_enj_{DGP_type}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(result_all_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(phi_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(size_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(lag_set, f, protocol=pickle.HIGHEST_PROTOCOL)