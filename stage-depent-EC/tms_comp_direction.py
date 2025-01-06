# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:42:36 2023

@author: XJM

使用PSI 进行有向分析
"""

import numpy as np
from scipy.io import loadmat, savemat
from mne_connectivity import phase_slope_index
from utils import dpli_transform, psl_transform, directed_net_summary


#################################
#### Import time-weight data V
result = loadmat('.../evoked_recon_resort.mat')
V = result['V_resort'].T
num_comp = 10




#################################
#### PSI method
## all time during (Static analysis)
list_V = []
list_V.append(V)


psl = phase_slope_index(
    list_V, mode='multitaper', 
    sfreq=1000, fmin=1, fmax=100)
psl_vec = psl.get_data()
psl_conn = psl.get_data('dense').squeeze()
psl_conn_tril = psl_conn[np.tril_indices(n=num_comp,k=-1)]

psl_conn_full = psl_transform(psl_conn, psl_vec, 1)
psl_conn_norm = psl_transform(psl_conn, psl_vec, 0.3)

save_name = '.../PSI_static.mat'
save_obj = {'PSI_static':psl_conn_norm}
savemat(save_name, save_obj)

save_name = '.../PSI_static_full.mat'
save_obj = {'PSI_static_full':psl_conn_full}
savemat(save_name, save_obj)



########################################################################
####### Main results (Stage-dependent EC)
time_seg = loadmat('.../state_seg_resort.mat')
states_idx = time_seg['states_idx'].squeeze()
num_states = len(np.unique(states_idx))
V_seg = [V[:,states_idx == i] for i in np.unique(states_idx)]


### PSI stage 2 and 3
V_stage_2_3 = np.concatenate((V_seg[1],V_seg[2]),axis=1)
V_stage_2_3_list = []
V_stage_2_3_list.append(V_stage_2_3)

pls_stage_2_3 = phase_slope_index(
                V_stage_2_3_list,  mode='multitaper', 
                sfreq=1000, fmin=1, fmax=100)

psl_conn_stage_2_3 = pls_stage_2_3.get_data('dense').squeeze()
psl_conn_vec_stage_2_3 = pls_stage_2_3.get_data()
psl_conn_norm_stage_2_3 = psl_transform(psl_conn_stage_2_3,psl_conn_vec_stage_2_3, 0.3)
psl_conn_full_stage_2_3 = psl_transform(psl_conn_stage_2_3,psl_conn_vec_stage_2_3, 1)


save_name = '.../PSI_stage_2_3.mat'
save_obj = {'psl_conn_norm_stage_2_3':psl_conn_norm_stage_2_3}
savemat(save_name, save_obj)

save_name = '.../PSI_stage_2_3_full.mat'
save_obj = {'psl_conn_full_stage_2_3':psl_conn_full_stage_2_3}
savemat(save_name, save_obj)


### PSI stage 3 and 4
V_stage_3_4 = np.concatenate((V_seg[2],V_seg[3]),axis=1)
V_stage_3_4_list = []
V_stage_3_4_list.append(V_stage_3_4)

pls_stage_3_4 = phase_slope_index(
                V_stage_3_4_list,  mode='multitaper', 
                sfreq=1000, fmin=1, fmax=100)

psl_conn_stage_3_4 = pls_stage_3_4.get_data('dense').squeeze()
psl_conn_vec_stage_3_4 = pls_stage_3_4.get_data()
psl_conn_norm_stage_3_4 = psl_transform(psl_conn_stage_3_4,psl_conn_vec_stage_3_4, 0.3)
psl_conn_full_stage_3_4 = psl_transform(psl_conn_stage_3_4,psl_conn_vec_stage_3_4, 1)



save_name = '.../PSI_stage_3_4.mat'
save_obj = {'psl_conn_norm_stage_3_4':psl_conn_norm_stage_3_4}
savemat(save_name, save_obj)

save_name = 'D:/DATA/TMS-EEG/important/tensor/process/NF_model/Final_model_resort/PSI_stage_3_4_full.mat'
save_obj = {'psl_conn_full_stage_3_4':psl_conn_full_stage_3_4}
savemat(save_name, save_obj)









