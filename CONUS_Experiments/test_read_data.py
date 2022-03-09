# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 2022
@author: Onlyrich-Ryzen

Test reading CONUS dataset
"""

import shutil
import numpy as np
import torch
import random
from copy import copy
from IPython.display import clear_output
from matplotlib import pylab as plt
from CONUS_Experiments.q_models_gpu import DQAgent # Using GPU for training
from CONUS_Experiments.edf_env_v2_02 import EdfEnv
from CONUS_Experiments.data_utils_02 import retrieve_sfc_req_from_json, retrieve_sfc_req_from_json_CONUS
import os
from datetime import datetime
from pathlib import Path
import json


### Training Settings
IS_BINARY_STATE = True # Using Binary encoded state
IS_NEIGHBOR_MASK = False# Neighborhood mask Default = False
IS_CONTINUE_TRAINING = False # Continue training the DQN from the current DQN' weights
STATE_NOISE_SCALE = 50.
RESOURCE_SCALER = 0.65

#### Build the EDF network environment
DATA_FOLDER = "SOF_Data_Sets-2022_Mar_07\complete_data_sets\Data_18"# Training Dataset
req_json_file = os.path.join(DATA_FOLDER, "non_uniform_CONUS36_reordered_traffic_1000_slots_2_con.tra")  # service request traffic
# Create EDF environment
edf = EdfEnv(data_path=DATA_FOLDER, 
             net_file='non_uniformCONUS36_1000_slots_2_con.net', 
             sfc_file='CONUS36_sfc_file.sfc', 
             resource_scaler=RESOURCE_SCALER)

# Using Binary state or Fractional state renderer
if IS_BINARY_STATE:
    edf.binary_state_dim()
else:
    edf.fractional_state_dim()
    
#### Obtain service requests
n_req, req_list = retrieve_sfc_req_from_json_CONUS(req_json_file)