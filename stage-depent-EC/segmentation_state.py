# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 20:14:33 2023

@author: XJM
"""

import numpy as np
from neural_segment import GSBS
from scipy.io import loadmat, savemat
import pickle
import netneurotools.plotting

path = '.../evoked_recon_resort.mat'
evoke = loadmat(path)

V = evoke['V_resort']


#####################################################
## Run state segmentation
states = GSBS(kmax=10,x=V,blocksize=5)
states.fit()


nstates = states.nstates
bounds = states.bounds
all_bounds = states.all_bounds
tdists = states.tdists
deltas = states.deltas

states_idx = states.get_states()
state_patterns = states.get_state_patterns()
strengths = states.get_strengths()

save_path = '...'
save_model = save_path + 'seg_model_resort.pickle'
file = open(save_model, "wb")
pickle.dump(states, file)
file.close()

save_seg = save_path + 'state_seg_resort.mat'
state_seg = {'nstates':nstates, 'bounds':bounds, 'all_bounds':all_bounds,'tdists':tdists,
             'states_idx':states_idx, 'state_patterns':state_patterns, 'strengths':strengths}
savemat(save_seg, state_seg)




