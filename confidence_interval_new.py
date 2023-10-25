import numpy as np
import mpmath
import math
import pandas as pd
import argparse
import pickle
import os
from DGP_burn import *
from ci_cal import *

def save_metrics(stats, categories, replication):
    result_dict = {}  # Initialize an empty dictionary to store results
    
    for prefix in categories:
        # Initialize sub-dictionaries for each category if not already present
        if prefix not in result_dict:
            result_dict[prefix] = {}
            for metric in ['count_cases', 'ECP_sum_cases', 'ECP_sum_cases_all', 
                           'CI_set_cases', 'CI_set_cases_all', 'lower_root_set_cases', 
                           'lower_root_set_cases_all', 'upper_root_set_cases',
                           'upper_root_set_cases_all', 'spj_band_sum_cases',
                           'spj_band_sum_cases_all']:
                result_dict[prefix][metric] = {}

        stats_case = stats[prefix]
        replication_factor = replication  # Assuming this is a constant you have

        # Calculate and store the results for this prefix in the result_dict
        result_dict[prefix]['count_cases'] = stats_case['count_cases'] / replication_factor
        result_dict[prefix]['ECP_sum_cases'] = stats_case['ECP_sum_cases'] / stats_case['count_cases']
        result_dict[prefix]['ECP_sum_cases_all'] = np.sum(stats_case['ECP_sum_cases']) / np.sum(stats_case['count_cases'])
        result_dict[prefix]['CI_set_cases'] = calculate_list_means(stats_case['CI_set_cases'])
        result_dict[prefix]['CI_set_cases_all'] = calculate_whole_list_mean(stats_case['CI_set_cases'])
        result_dict[prefix]['lower_root_set_cases'] = calculate_list_means(stats_case['lower_root_set_cases'])
        result_dict[prefix]['lower_root_set_cases_all'] = calculate_whole_list_mean(stats_case['lower_root_set_cases'])
        result_dict[prefix]['upper_root_set_cases'] = calculate_list_means(stats_case['upper_root_set_cases'])
        result_dict[prefix]['upper_root_set_cases_all'] = calculate_whole_list_mean(stats_case['upper_root_set_cases'])
        result_dict[prefix]['spj_band_sum_cases'] = stats_case['spj_band_sum_cases'] / stats_case['count_cases']
        result_dict[prefix]['spj_band_sum_cases_all'] = np.sum(stats_case['spj_band_sum_cases']) / np.sum(stats_case['count_cases'])

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DGP', type=str, default='AR1-IID')
    parser.add_argument('--lag_set', type=str, default='1,2')
    parser.add_argument('--size_set', type=str, default='50,100')
    parser.add_argument('--phi_set', type=str, default='0.1,0.3,0.5,0.7,0.9,-0.1,-0.3,-0.5,-0.7,-0.9')
    parser.add_argument('--replication', type=int, default=1000)
    
    args = parser.parse_args()

    # Convert comma-separated strings to lists of integers or floats
    size_set = list(map(int, args.size_set.split(',')))
    phi_set = list(map(float, args.phi_set.split(',')))
    lag_set = list(map(float, args.lag_set.split(',')))
    
    # Assign the integer value for replication, lag value
    replication = args.replication
    #lag_val = args.lag
      
    # Assign the string value for DGP type
    DGP_type = args.DGP
    print(DGP_type)
    
    # Example
    #rho_set = 
    #phi_set = [-0.5, 0, 0.5]
    #size_set = [100, 200, 500, 600]
    #replication = 300

    #usual fixed-b critical value
    c1 = 0.43754
    c2 = 0.11912
    c3 = 0.08640
    c4 = 0.49629
    c5 = -0.57879
    c6 = 0.43266
    c7 = 0.02543
    c8 = -0.02379
    c9 = -0.02376

    #this is not usual fixed-b critical value. This is fixed-b for nodemeaned v^k_tilde
    s1 = 0.74330
    s2 = -0.30350
    s3 = -0.15730
    s4 = 0.47990
    s5 = -1.34310
    s6 = 0.59400
    s7 = -0.35820
    s8 = 0.74630
    s9 = -0.28670

    CI_set_result_sw = {}
    ECP_sum_result_sw = {}

    ECP_sum_result_unres_normal = {}
    CI_set_result_unres_normal = {}
    lower_set_result_unres_normal = {}
    upper_set_result_unres_normal = {}

    ECP_sum_result_unres_fixedb = {}
    CI_set_result_unres_fixedb = {}
    lower_set_result_unres_fixedb = {}
    upper_set_result_unres_fixedb = {}

    result_dict_save = {}
    
    for kk in range(len(lag_set)):
        for pp in range(len(phi_set)):
            for ss in range(len(size_set)):

                CI_set_sw = []
                ECP_sum_sw= 0

                ECP_sum_unres_normal = 0
                ECP_sum_unres_fixedb = 0

                spj_band_sum_unres = 0
                spj_band_sum_res_nd = 0
                spj_band_sum_res_dm = 0

                CI_set_unres_normal = []
                lower_set_unres_normal = []
                upper_set_unres_normal = []

                CI_set_unres_fixedb = []
                lower_set_unres_fixedb = []
                upper_set_unres_fixedb = []

                categories = ['res_nd_normal', 'unres_nd_normal_for_res_nd_normal', 
                              'res_nd_fixedb', 'unres_nd_fixedb_for_res_nd_fixedb', 
                              'res_dm_normal', 'unres_nd_normal_for_res_dm_normal', 
                              'res_dm_fixedb', 'unres_nd_fixedb_for_res_dm_fixedb',
                              'sw_for_res_nd_normal', 'sw_for_res_dm_normal',]

                stats = {category: initialize_stats_dict() for category in categories}

                for w in range(replication):
                    np.random.seed(w)
                    phi = phi_set[pp]
                    size = size_set[ss]

                    # data generating
                    theta = 0.7
                    y = generate_data(DGP_type, phi=phi_set[pp], size=size_set[ss], shape=0.3, scale=0.4, ma1=theta)
                    
                    k_val = int(lag_set[kk])
                    cv_n = 1.96
                    k_function_vectorized = parzen_k_function_vectorized

                    # setting a_val (null value)
                    if DGP_type.startswith('AR1'):
                        a_val = phi**k_val 
                    elif DGP_type.startswith('ARMA11'):
                        a_val = (((1+theta*phi)*(phi+theta))/(1+2*theta*phi+theta**2))*(phi**(k_val-1))
                    elif DGP_type.startswith('MA1'):
                        if k_val == 1:
                            a_val = phi/(1+(phi**2)) ## for MA1
                        else:
                            a_val = 0
                    else:
                        raise ValueError("Invalid DGP_type")


                    ##### software built in function ###

                    acf, confint_sw = sm.tsa.stattools.acf(y, nlags=k_val, alpha=0.05, bartlett_confint=True)
                    lower_sw = confint_sw[k_val,0]
                    upper_sw = confint_sw[k_val,1]

                    if abs(lower_sw) > 1:
                        lower_sw = np.sign(lower_sw)
                    if abs(upper_sw) > 1:
                        upper_sw = np.sign(upper_sw)
                    CI_sw = upper_sw - lower_sw
                    CI_set_sw.append(CI_sw)

                    if lower_sw <= a_val <= upper_sw:
                        ECP_sum_sw += 1


                    ######## unrestricted, nodemean ##############

                    rho_k_til = compute_rho_k_tilda(y,k_val)
                    Q_2 = compute_Q2(y, k_val)

                    v_hat_unres_nodemean = compute_v_tilde_k_data_depen_nodemean(y=y, k=k_val, a0=rho_k_til)
                    M_SPJ_unres_nodemean = SPJ_band(vhat=v_hat_unres_nodemean,w=10,z_a=1.96,delta=2,q=2,g=6,c=0.539)

                    Tnum1 = len(y) - k_val
                    cv_unres_fixedb = calc_fixed_cv(M_SPJ_unres_nodemean, Tnum1, c1, c2, c3, c4, c5, c6, c7, c8, c9)

                    spj_band_sum_unres = spj_band_sum_unres + M_SPJ_unres_nodemean/Tnum1 # save spj band

                    omegas_nd = compute_omegas_vectorized_nodemean(y, M_SPJ_unres_nodemean, k_function_vectorized, k_val) # Parze, SPJ
                    omegas_nd11 = omegas_nd[0] #11
                    omegas_nd12 = omegas_nd[1] #12
                    omegas_nd22 = omegas_nd[2] #22

                    var_not_null_imp = (Q_2**(-2))*(omegas_nd11 - 2*rho_k_til*omegas_nd12 + (rho_k_til**2)*omegas_nd22)/(len(y)-k_val)

                    ##normal cv
                    lower_unres_normal = rho_k_til - cv_n*np.sqrt(var_not_null_imp)
                    upper_unres_normal = rho_k_til + cv_n*np.sqrt(var_not_null_imp)

                    ECP_sum_unres_normal, CI_set_unres_normal, lower_set_unres_normal, upper_set_unres_normal = update_stats_unrestricted(
                                            lower_unres_normal, upper_unres_normal, a_val, 
                                            ECP_sum_unres_normal, CI_set_unres_normal, lower_set_unres_normal, upper_set_unres_normal)

                    ##fixed-b
                    lower_unres_fixedb = rho_k_til - cv_unres_fixedb*np.sqrt(var_not_null_imp)
                    upper_unres_fixedb = rho_k_til + cv_unres_fixedb*np.sqrt(var_not_null_imp)

                    ECP_sum_unres_fixedb, CI_set_unres_fixedb, lower_set_unres_fixedb, upper_set_unres_fixedb = update_stats_unrestricted(
                                            lower_unres_fixedb, upper_unres_fixedb, a_val, 
                                            ECP_sum_unres_fixedb, CI_set_unres_fixedb, lower_set_unres_fixedb, upper_set_unres_fixedb)

                    ####### restricted, nodemean ###################

                    rho_k_til = compute_rho_k_tilda(y,k_val)
                    Q_2 = compute_Q2(y, k_val)

                    v_til_res_nd = compute_v_tilde_k_data_depen_nodemean(y=y, k=k_val, a0=a_val)
                    M_SPJ_res_nd = SPJ_band(vhat=v_til_res_nd,w=10,z_a=1.96,delta=2,q=2,g=6,c=0.539)

                    Tnum1 = len(y) - k_val
                    cv_res_nd_fixedb = calc_fixed_cv(M_SPJ_res_nd, Tnum1, s1, s2, s3, s4, s5, s6, s7, s8, s9)

                    spj_band_sum_res_nd = spj_band_sum_res_nd + M_SPJ_res_nd/Tnum1 # save spj band

                    omegas_res_nd = compute_omegas_vectorized_nodemean(y, M_SPJ_res_nd, k_function_vectorized, k_val)
                    #omegas_res_nd11 = omegas_res_nd[0] #11
                    #omegas_res_nd12 = omegas_res_nd[1] #12
                    #omegas_res_nd22 = omegas_res_nd[2] #22

                    # don't need to calculate long run variance for CI. Just need C2, C1, C0.

                    ## Normal CV
                    c_coef_res_nd_normal = compute_c_coefficients(omegas=omegas_res_nd, cv=cv_n, y=y, k_val=k_val, rho_k_til=rho_k_til, Q_2=Q_2)
                    roots_res_nd_normal = quadratic_roots(c_coef_res_nd_normal[0], c_coef_res_nd_normal[1], c_coef_res_nd_normal[2]) # c_coef_nd[0] is coef of a^2

                    stats['res_nd_normal'] = update_stats_restricted_cases_dict(roots_res_nd_normal, 
                                                                           c_coef_res_nd_normal, 
                                                                           a_val, 
                                                                           Tnum1, 
                                                                           M_SPJ_res_nd, 
                                                                           stats['res_nd_normal'])

                    stats['unres_nd_normal_for_res_nd_normal'] = update_stats_unres_cases_dict(lower_unres_normal, upper_unres_normal,
                                                                           roots_res_nd_normal, 
                                                                           c_coef_res_nd_normal, 
                                                                           a_val, 
                                                                           Tnum1, 
                                                                           M_SPJ_res_nd, 
                                                                           stats['unres_nd_normal_for_res_nd_normal'])

                    stats['sw_for_res_nd_normal'] = update_stats_SW_cases_dict(lower_sw, upper_sw, roots_res_nd_normal, c_coef_res_nd_normal, a_val, Tnum1, stats['sw_for_res_nd_normal'])


                    ## Fixed-b CV
                    c_coef_res_nd_fixedb = compute_c_coefficients(omegas=omegas_res_nd, cv=cv_res_nd_fixedb, y=y, k_val=k_val, rho_k_til=rho_k_til, Q_2=Q_2)
                    roots_res_nd_fixedb = quadratic_roots(c_coef_res_nd_fixedb[0], c_coef_res_nd_fixedb[1], c_coef_res_nd_fixedb[2])
                    #if c_coef_res_nd_fixedb[1]**2 - c_coef_res_nd_fixedb[0]*c_coef_res_nd_fixedb[2] < 0:
                    #    print ('raise error')

                    stats['res_nd_fixedb'] = update_stats_restricted_cases_dict(roots_res_nd_fixedb, 
                                                                                c_coef_res_nd_fixedb, 
                                                                                a_val, 
                                                                                Tnum1, 
                                                                                M_SPJ_res_nd, 
                                                                                stats['res_nd_fixedb'])

                    # Using the refactored functions for the "unres_nd_fixedb_for_res_nd_fixedb" case
                    stats['unres_nd_fixedb_for_res_nd_fixedb'] = update_stats_unres_cases_dict(lower_unres_fixedb, upper_unres_fixedb,
                                                                                          roots_res_nd_fixedb, 
                                                                                          c_coef_res_nd_fixedb, 
                                                                                          a_val, 
                                                                                          Tnum1, 
                                                                                          M_SPJ_unres_nodemean, 
                                                                                          stats['unres_nd_fixedb_for_res_nd_fixedb'])


                    ####### restricted, demean ###################

                    rho_k_til = compute_rho_k_tilda(y,k_val)
                    Q_2 = compute_Q2(y, k_val)

                    v_til_res_dm = compute_v_tilde_k_data_depen_demean(y=y, k=k_val, a0=a_val)
                    M_SPJ_res_dm = SPJ_band(vhat=v_til_res_dm,w=10,z_a=1.96,delta=2,q=2,g=6,c=0.539)

                    Tnum1 = len(y) - k_val
                    cv_res_dm_fixedb = calc_fixed_cv(M_SPJ_res_dm, Tnum1, c1, c2, c3, c4, c5, c6, c7, c8, c9)

                    spj_band_sum_res_dm = spj_band_sum_res_dm + M_SPJ_res_dm/Tnum1 # save spj band b-ratio

                    omegas_res_dm = compute_omegas_vectorized_demean(y, M_SPJ_res_dm, k_function_vectorized, k_val)
                    #omegas_res_dm11 = omegas_res_dm[0] #11
                    #omegas_res_dm12 = omegas_res_dm[1] #12
                    #omegas_res_dm22 = omegas_res_dm[2] #22

                    # don't need to calculate long run variance for CI. Just need C2, C1, C0.

                    ## Normal CV
                    c_coef_res_dm_normal = compute_c_coefficients(omegas=omegas_res_dm, cv=cv_n, y=y, k_val=k_val, rho_k_til=rho_k_til, Q_2=Q_2)
                    roots_res_dm_normal = quadratic_roots(c_coef_res_dm_normal[0], c_coef_res_dm_normal[1], c_coef_res_dm_normal[2]) # c_coef_nd[0] is coef of a^2    

                    stats['res_dm_normal'] = update_stats_restricted_cases_dict(roots_res_dm_normal, 
                                                                    c_coef_res_dm_normal, 
                                                                    a_val, 
                                                                    Tnum1, 
                                                                    M_SPJ_res_dm, 
                                                                    stats['res_dm_normal'])


                    stats['unres_nd_normal_for_res_dm_normal'] = update_stats_unres_cases_dict(lower_unres_normal, upper_unres_normal,
                                                                                               roots_res_dm_normal, 
                                                                                               c_coef_res_dm_normal, 
                                                                                               a_val, 
                                                                                               Tnum1, 
                                                                                               M_SPJ_unres_nodemean, 
                                                                                               stats['unres_nd_normal_for_res_dm_normal'])

                    stats['sw_for_res_dm_normal'] = update_stats_SW_cases_dict(lower_sw, upper_sw, roots_res_dm_normal, c_coef_res_dm_normal, a_val, Tnum1, stats['sw_for_res_dm_normal'])


                    ## Fixed-b CV
                    c_coef_res_dm_fixedb = compute_c_coefficients(omegas=omegas_res_dm, cv=cv_res_dm_fixedb, y=y, k_val=k_val, rho_k_til=rho_k_til, Q_2=Q_2)
                    roots_res_dm_fixedb = quadratic_roots(c_coef_res_dm_fixedb[0], c_coef_res_dm_fixedb[1], c_coef_res_dm_fixedb[2]) # c_coef_nd[0] is coef of a^2

                    stats['res_dm_fixedb'] = update_stats_restricted_cases_dict(roots_res_dm_fixedb, 
                                                                                c_coef_res_dm_fixedb, 
                                                                                a_val, 
                                                                                Tnum1, 
                                                                                M_SPJ_res_dm, 
                                                                                stats['res_dm_fixedb'])


                    stats['unres_nd_fixedb_for_res_dm_fixedb'] = update_stats_unres_cases_dict(lower_unres_fixedb, upper_unres_fixedb,
                                                                                               roots_res_dm_fixedb, 
                                                                                               c_coef_res_dm_fixedb, 
                                                                                               a_val, 
                                                                                               Tnum1, 
                                                                                               M_SPJ_unres_nodemean, 
                                                                                               stats['unres_nd_fixedb_for_res_dm_fixedb'])


                # Saving results

                #CI_set_result_sw[pp,ss] = np.mean(np.array(CI_set_sw))
                #ECP_sum_result_sw[pp,ss] = ECP_sum_sw/replication

                #ECP_sum_result_unres_normal[pp,ss] = ECP_sum_unres_normal/replication
                #CI_set_result_unres_normal[pp,ss] = np.mean(np.array(CI_set_unres_normal))
                #lower_set_result_unres_normal[pp,ss] = np.mean(np.array(lower_set_unres_normal))
                #upper_set_result_unres_normal[pp,ss] = np.mean(np.array(upper_set_unres_normal))

                #ECP_sum_result_unres_fixedb[pp,ss] = ECP_sum_unres_fixedb/replication
                #CI_set_result_unres_fixedb[pp,ss] = np.mean(np.array(CI_set_unres_fixedb))
                #lower_set_result_unres_fixedb[pp,ss] = np.mean(np.array(lower_set_unres_fixedb))
                #upper_set_result_unres_fixedb[pp,ss] = np.mean(np.array(upper_set_unres_fixedb))

                result_dict_save[kk,pp,ss] = save_metrics(stats, categories, replication)
            
            # dump to pkl
    #result_all_dict_CI = {
    #    'CI_set_result_sw': CI_set_result_sw,
    #    'ECP_sum_result_sw': ECP_sum_result_sw,
    #    'ECP_sum_result_unres_normal': ECP_sum_result_unres_normal,
    #    'CI_set_result_unres_normal': CI_set_result_unres_normal,
    #    'lower_set_result_unres_normal': lower_set_result_unres_normal,
    #    'upper_set_result_unres_normal': upper_set_result_unres_normal,
    #    'ECP_sum_result_unres_fixedb': ECP_sum_result_unres_fixedb,
    #    'CI_set_result_unres_fixedb': CI_set_result_unres_fixedb,
    #    'lower_set_result_unres_fixedb': lower_set_result_unres_fixedb,
    #    'upper_set_result_unres_fixedb': upper_set_result_unres_fixedb,
    #    'result_dict_save': result_dict_save  # Assuming (pp, ss) is the key you want to use
    #     }

        # Check if the folder exists, and create one if it doesn't
    if not os.path.exists('CI_result'):
        os.makedirs('CI_result')

    # Dumping the master_dict
    filename = f'CI_result/result_CI_{DGP_type}_lag_{lag_set}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(result_dict_save, f, protocol=4)
        pickle.dump(phi_set, f, protocol=4)
        pickle.dump(size_set, f, protocol=4)
        pickle.dump(lag_set, f, protocol=4)
        
    print("Complete without error")