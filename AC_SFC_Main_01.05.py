# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:45:10 2021

@author: Onlyrich-Ryzen

Implement Advantage Actor Critic (A2C) learning main loop
    + Service Function Chain Embedding problem
    + EDF network environment

- Changelog from version 01.04:
    + Integrate LSTM/GRU cell into the shared_net_AC model
    
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import random
from copy import copy
from IPython.display import clear_output
from matplotlib import pylab as plt
from actor_critic_models_SFC_v05 import ACNet, A3CWorker, SharedAdam, SharedRMSProp
from edf_env_A3C_v2_02 import EdfEnv
from data_utils_A3C_02 import retrieve_sfc_req_from_json, mov_window_avg
import os
from datetime import datetime
from pathlib import Path
import json
import shutil


def read_reward_data(log_file):
    data = []
    for f in log_file:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        vals = [float(l) for l in lines]
        data.append(vals)
    return np.array(data) 

def read_loss_data(log_files):
    """
    log_files: list 
        A list of loss_log files
        Each line of a loss_log file: actor_loss critic_loss sum_weighted_loss
    
    Return:
        A array where each row is the sum_weighted_loss items corresponding to each loss_log file
    """
    data = []
    for f in log_files:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        vals = [float(l.split()[-1]) for l in lines]
        data.append(vals)
    return np.array(data) 


def worker(a3c_worker, g_counter):
    a3c_worker.run(g_counter)

def main(is_train, network_arch):
    ### Training Settings
    IS_BINARY_STATE = True # Using Binary encoded state
    IS_NEIGHBOR_MASK = False# Neighborhood mask Default = False
    IS_CONTINUE_TRAINING = False # Continue training the DQN from the current DQN' weights
    STATE_NOISE_SCALE = 100.
    RESOURCE_SCALER = 1.0

    OPERATION_MODE = {'train_mode': 1, 
                      'test_mode': 0}

    NET_ARCH = {'shared_net': 1,
                'shared_net_w_RNN':2,
                'separated_net':3}
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Train/Test mode '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    # is_train = OPERATION_MODE["test_mode"]
    
    #### Training parameters
    N_EPOCHS = 500_000 if not IS_CONTINUE_TRAINING else 500_000
    N_STEPS = 10 # N-step training for actor-critic, N-step=0 --> use Monte-Carlo sampling
    MAX_MOVE = 25 # Max number of steps per episode
    N_CPUS = 1#int(mp.cpu_count()/2)
    
    #### Create Model Folder 
    MODEL_NAME = "AC_Agent"
    MODEL_DIR = 'A3C_SFCE_models'
    now = datetime.now()
    # Training directory
    TRAIN_DIR = 'train_results__' + now.strftime("%Y-%B-%d__%H-%M")
    if IS_BINARY_STATE:
        TRAIN_DIR += "_BINARY_state"
    else:
        TRAIN_DIR += "_FRACTIONAL_state"
    TRAIN_DIR = os.path.join(MODEL_DIR, TRAIN_DIR)

    # Testing directory
    TEST_DIR = 'test_results__' + now.strftime("%Y-%B-%d__%H-%M")
    if IS_BINARY_STATE:
        TEST_DIR += "_BINARY_state"
    else:
        TEST_DIR += "_FRACTIONAL_state"    
    TEST_DIR = os.path.join(MODEL_DIR, TEST_DIR)

    if is_train == OPERATION_MODE["train_mode"]:    
        Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    else:
        Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
    
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training Dataset Folder
    DATA_FOLDER = "SOF_Data_Sets-v09\complete_data_sets"
    # Create a temporary edf_env for information extraction
    temp_edf = EdfEnv(data_path=DATA_FOLDER,
                      net_file="ibm_15000_slots_1_con.net",
                      sfc_file="sfc_file.sfc",
                      resource_scaler=RESOURCE_SCALER)
    
    # Using Binary state or Fractional state renderer
    if IS_BINARY_STATE:
        temp_edf.binary_state_dim()
    else:
        temp_edf.fractional_state_dim()
        
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Configuration for the neural network
    # Neural net layers' neurons
    # INPUT_DIMS = temp_edf.n_state_dims
    INPUT_DIMS = temp_edf.n_state_dims
    ACTOR_OUTPUT_DIMS = 2 * temp_edf.n_nodes
    L_COMMON_1 = 256
    L_COMMON_2 = 128
    L_CRITIC_1 = 128
    L_RNN = 128
    if network_arch == NET_ARCH['shared_net']:
        # FC_HID_LAYERS = np.array([L_COMMON_1, L_COMMON_2, L_CRITIC_1])
        FC_HID_LAYERS = {'common_1': L_COMMON_1,
                         'common_2': L_COMMON_2,
                         'critic_1': L_CRITIC_1}
    elif network_arch == NET_ARCH['shared_net_w_RNN']:
        # FC_HID_LAYERS = np.array([L_COMMON_1, L_COMMON_2, L_RNN])
        FC_HID_LAYERS = {'common_1': L_COMMON_1,
                         'rnn_1': L_RNN}
    elif network_arch == NET_ARCH['separated_net']:
        FC_HID_LAYERS = {'l1': L_COMMON_1,
                         'l2': L_COMMON_2}
    
    #### Learning parameters
    GAMMA = 0.99
    TAU = 1.0
    LEARNING_RATE = 7e-4
    ACTOR_FACTOR = 1
    CRITIC_FACTOR = 0.5
    ENTROPY_FACTOR = 0.01

    #### RMSProp's parameters
    RMS_EPSILON = 1e-5
    RMS_WEIGHT_DECAY = 0
    RMS_ALPHA = 0.99
    RMS_MOMENTUM = 0.0
    RMS_CENTERED = False

    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Export the parameter setting into a file
    if is_train == OPERATION_MODE["train_mode"]:
        param_setting_file = os.path.join(TRAIN_DIR, "parameter_settings.txt")
    else:
        param_setting_file = os.path.join(TEST_DIR, "parameter_settings.txt")

    with open(param_setting_file, 'a') as fp:
        print("SYSTEM PARAMETERS", file=fp)
        if OPERATION_MODE["train_mode"]:
            print(f"training directory: {TRAIN_DIR} \n", file=fp)
        else:
            print(f"testing directory: {TEST_DIR} \n", file=fp)
        
        print(f"IS_BINARY_STATE = {IS_BINARY_STATE}", file=fp)
        print(f"IS_NEIGHBOR_MASK = {IS_NEIGHBOR_MASK}", file=fp)
        print(f"IS_CONTINUE_TRAINING = {IS_CONTINUE_TRAINING}", file=fp)
        print(f'STATE_NOISE_SCALE = {STATE_NOISE_SCALE}', file=fp)
        
        print("NEURAL NETWORK SETTINGS", file=fp)
        print(f"INPUT_DIMS = {INPUT_DIMS}", file=fp)
        print(f"ACTOR_OUTPUT_DIMS = {ACTOR_OUTPUT_DIMS} \n", file=fp)
        if network_arch == NET_ARCH['shared_net']:
            print(f"Network Architecture = Shared_Net", file=fp)
            print(f"FC_COMMON_1 = {L_COMMON_1}, FC_COMMON_2 = {L_COMMON_2}, FC_CRITIC_1 = {L_CRITIC_1}", file=fp)
        elif network_arch == NET_ARCH['shared_net_w_RNN']:
            print(f"Network Architecture = Shared_Net with RNN", file=fp)
            print(f"FC_COMMON_1 = {L_COMMON_1}, RNN_CELL = {L_RNN}", file=fp)
        elif network_arch == NET_ARCH['separated_net']:
            print(f"Network Architecture = Separated_Net", file=fp)
            print(f"FC_ACTOR_1 = {L_COMMON_1}, FC_ACTOR_2 = {L_COMMON_2}", file=fp)
            print(f"FC_CRITIC_1 = {L_COMMON_1}, FC_CRITIC_2 = {L_COMMON_2}", file=fp)
        # print(f"FC_HID_LAYERS  = {FC_HID_LAYERS}", file=fp)
        
        print("LEARNING PARAMETERS", file=fp)
        print(f"GAMMA = {GAMMA}", file=fp)
        print(f"TAU = {TAU}", file=fp)
        print(f"LEARNING_RATE = {LEARNING_RATE}", file=fp)
        print(f"CRITIC_FACTOR = {CRITIC_FACTOR}", file=fp)
        print(f"ACTOR_FACTOR = {ACTOR_FACTOR}", file=fp)
        print(f"ENTROPY_FACTOR = {ENTROPY_FACTOR}", file=fp)

        print("Optimizer parameters", file=fp)
        print(f"Opt_Learning_Rate = {LEARNING_RATE}", file=fp)
        print(f"Opt_Epsilon = {RMS_EPSILON}", file=fp)
        print(f"Opt_Weight_Decay = {RMS_WEIGHT_DECAY}", file=fp)
        print(f"Opt_Alpha = {RMS_ALPHA}", file=fp)
        print(f"Opt_Momentum = {RMS_MOMENTUM}", file=fp)
        print(f"Opt_Centered = {RMS_CENTERED}", file=fp)
        
        print("Training parameters", file=fp)
        print(f"N_EPOCHS = {N_EPOCHS}", file=fp)
           
    print("Export system parameter setting into text files ... DONE!")
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    #### Create a Master Actor-Critic Agent
    Before_Train_Model = ACNet(name=MODEL_NAME, model_dir=MODEL_DIR, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=network_arch)
    
    #### Load the trained network params
    if IS_CONTINUE_TRAINING:
        Before_Train_Model.load_params()
        print("CONTINUE TRAINING FROM PREVIOUS CHECKPOINT")
        print(f"Model directory: = {MODEL_DIR}")
        print(f"Model name: {MODEL_NAME}")
    else:
        print("TRAINING THE NEURAL NETWORK FROM SCRATCH!")
    
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training & Testing datasets
    sfc_spec_file = 'sfc_file.sfc'
    
    # {'net_topo' : "ibm_200000_slots_1_con.net",
    #                  'traffic' : "reordered_traffic_200000_slots_1_con.tra"},
    train_dataset_list = [{'net_topo' : "ibm_500000_slots_1_con.net",
                     'traffic' : "reordered_traffic_500000_slots_1_con.tra"},
                    \
                    {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-06-04.net",
                     'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-04.tra"},
                    \
                    {'net_topo': "ibm_200000_slots_1_con_2021-September-21__16-06-36.net",
                     'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-36.tra"},
                    \
                    {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-06-10.net",
                     'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-10.tra"},
                    \
                    {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-08-42.net",
                     'traffic': "reordered_traffic_200000_slots_1_con_2021-September-21__16-08-42.tra"},
                    \
                    {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-09-30.net",
                     'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-09-30.tra"}
                    ]
    
    test_dataset_list = [{"net_topo": "ibm_15000_slots_1_con.net",
                           "traffic": "reordered_traffic_15000_slots_1_con.tra"}]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training & Testing params
    train_params = {'epochs': N_EPOCHS,
                    'n_workers': N_CPUS,
                    'sfc_spec': sfc_spec_file,
                    'datasets': train_dataset_list,
                    'data_folder': DATA_FOLDER,
                    'model_dir': MODEL_DIR,
                    'train_dir': TRAIN_DIR,
                    'test_dir': TEST_DIR,
                    'hidden_layers': FC_HID_LAYERS,
                    'input_dims': INPUT_DIMS,
                    'actor_dims': ACTOR_OUTPUT_DIMS,
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA,
                    'tau': TAU,
                    'critic_factor': CRITIC_FACTOR,
                    'actor_factor': ACTOR_FACTOR,
                    'entropy_factor': ENTROPY_FACTOR,
                    'N_steps': N_STEPS,
                    'max_moves': MAX_MOVE,
                    'resource_scaler': RESOURCE_SCALER,
                    'is_binary_state': IS_BINARY_STATE,
                    'state_noise_scale': STATE_NOISE_SCALE,
                    'opt_epsilon': RMS_EPSILON,
                    'opt_weight_decay': RMS_WEIGHT_DECAY,
                    'opt_alpha': RMS_ALPHA,
                    'opt_momentum': RMS_MOMENTUM,
                    'opt_centered': RMS_CENTERED
                    }
    
    test_params = {'epochs': 15_000,
                    'n_workers': N_CPUS,
                    'sfc_spec': sfc_spec_file,
                    'datasets': test_dataset_list,
                    'data_folder': DATA_FOLDER,
                    'model_dir': MODEL_DIR,
                    'train_dir': TRAIN_DIR,
                    'test_dir': TEST_DIR,
                    'hidden_layers': FC_HID_LAYERS,
                    'input_dims': INPUT_DIMS,
                    'actor_dims': ACTOR_OUTPUT_DIMS,
                    'learning_rate': LEARNING_RATE,
                    'gamma': GAMMA,
                    'tau': TAU,
                    'critic_factor': CRITIC_FACTOR,
                    'actor_factor': ACTOR_FACTOR,
                    'entropy_factor': ENTROPY_FACTOR,
                    'N_steps': N_STEPS,
                    'max_moves': MAX_MOVE,
                    'resource_scaler': RESOURCE_SCALER,
                    'is_binary_state': IS_BINARY_STATE,
                    'state_noise_scale': STATE_NOISE_SCALE,
                    'opt_epsilon': RMS_EPSILON,
                    'opt_weight_decay': RMS_WEIGHT_DECAY,
                    'opt_alpha': RMS_ALPHA,
                    'opt_momentum': RMS_MOMENTUM,
                    'opt_centered': RMS_CENTERED
                    }
    

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Prepare the traffic request datasets '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    print("Preparing traffic request datasets")
    all_req_lists = []
    all_n_reqs = []
    #### TODO: implement it in parallel in the future
    if is_train == OPERATION_MODE["train_mode"]:
        for idx in range(train_params['n_workers']):
            req_json_file = os.path.join(DATA_FOLDER, train_dataset_list[idx]['traffic'])
            n_req, req_list = retrieve_sfc_req_from_json(req_json_file)
            print(f"Dataset {idx+1} is ")
            print(f"{train_dataset_list[idx]['net_topo']}")
            print(f"{train_dataset_list[idx]['traffic']}")
            print(f"Dataset {idx+1} has {len(req_list)} requests")
            all_n_reqs.append(n_req)
            all_req_lists.append(req_list)
    else:
        for idx in range(train_params['n_workers']):
            req_json_file = os.path.join(DATA_FOLDER, test_dataset_list[idx]['traffic'])
            n_req, req_list = retrieve_sfc_req_from_json(req_json_file)
            print(f"Dataset {idx+1} is ")
            print(f"{test_dataset_list[idx]['net_topo']}")
            print(f"{test_dataset_list[idx]['traffic']}")
            print(f"Dataset {idx+1} has {len(req_list)} requests")
            all_n_reqs.append(n_req)
            all_req_lists.append(req_list)
        

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' A3C_Worker's task '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Original Global_Model '''
    Global_Model = ACNet(name=MODEL_NAME, model_dir=MODEL_DIR, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=network_arch)

    Global_Model.load_state_dict(Before_Train_Model.state_dict())

    # os.environ['OMP_NUM_THREADS'] = '1' # 1 thread per CPU core
    #### Training mode
    if is_train == OPERATION_MODE["train_mode"]:
        print("Training the A3C RL-agent")
        # the model's parameters are globally shared among processes
        Global_Model.share_memory()
        global_optimizer = SharedRMSProp(Global_Model.parameters(), lr=LEARNING_RATE,
                                        eps=RMS_EPSILON, weight_decay=RMS_WEIGHT_DECAY,
                                        alpha=RMS_ALPHA, momentum=RMS_MOMENTUM, centered=RMS_CENTERED)
        # global_optimizer = SharedAdam(Global_Model.parameters(), lr=train_params['learning_rate'])
        processes = []
        g_counter = mp.Value('i', 0) # global counter shared among processes
        
        # create a3c_workers
        all_a3c_workers = []
        for idx in range(train_params['n_workers']):
            print(f'Start worker #{idx}')
            # Create an A3C worker instance
            a3c_worker = A3CWorker(worker_id=idx, 
                                   global_model=Global_Model,
                                   optimizer=global_optimizer,
                                   params=train_params,
                                   traffic_file=all_req_lists[idx], 
                                   counter=g_counter,
                                   net_arch=network_arch,
                                   is_train=True)
            all_a3c_workers.append(a3c_worker)

        # assign each a3c_woker to a process
        for worker in all_a3c_workers:
            worker.start()
            processes.append(worker)
            
        # for idx in range(train_params['n_workers']):    
        #     # instantiate a process invoking a worker who does the training
        #     p = mp.Process(target=all_a3c_workers[idx].run, args=(g_counter,))
        #     p.start()
        #     processes.append(p)

        # Wait for each process to be done before returning to the main thread
        [p.join() for p in processes]
        # Terminate each process
        [p.terminate() for p in processes]
        # Global counter and the first process's exit code
        print(f"g_counter = {g_counter.value}")
        
        exit_codes = []
        [exit_codes.append(processes[i].exitcode) for i in range(train_params["n_workers"])]
            
        # Save the model
        if 1 in exit_codes:
            print("Training Process Failed at some worker(s)!")
            print(f"exit_codes = {exit_codes}")
        else:
            print("Training Process Sucess for all worker(s)!")
        Global_Model.save_params()
        # Copy trained network model to the train_directory
        abs_path = os.getcwd()
        full_path = os.path.join(abs_path, MODEL_DIR)
        src_file = os.path.join(full_path, MODEL_NAME + '.pt')
        shutil.copy2(src_file, TRAIN_DIR)

        #### Plotting results #######################################################################
        print("Plotting some results")
        # Get the names of reward_log and loss_log files
        reward_logs, loss_logs = [], []
        for worker_id in range(train_params['n_workers']):
            reward_log = os.path.join(train_params['train_dir'], "W_" + str(worker_id) + "_ep_reward.log")
            reward_logs.append(reward_log)
            loss_log = os.path.join(train_params['train_dir'], "W_" + str(worker_id) + "_losses.log")
            loss_logs.append(loss_log)
        
        ''' Plot Episode Rewards'''
        # Read reward_log file data
        reward_log_data = read_reward_data(reward_logs)
        # Plot episode rewards
        plt.figure(figsize=(10,7.5))
        plt.grid()
        plt.title("Reward over training epochs")
        plt.xlabel("Epochs",fontsize=22)
        plt.ylabel("Reward",fontsize=22)
        mov_avg_reward = mov_window_avg(reward_log_data.mean(axis=0), window_size=1000)
        plt.plot(mov_avg_reward, color='b')
        fig_name = "Avg_Train_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
        fig_name = os.path.join(train_params['train_dir'], fig_name)
        plt.savefig(fig_name)


        ''' Plot loss values'''
        # Read loss_log file data
        loss_log_data = read_loss_data(loss_logs)
        # Plot loss values over training steps
        plt.figure(figsize=(10,7.5))
        plt.grid()
        plt.title("Loss over training epochs")
        plt.xlabel("Epochs",fontsize=22)
        plt.ylabel("Loss",fontsize=22)
        mov_avg_loss = mov_window_avg(loss_log_data.mean(axis=0), window_size=1000)
        plt.plot(mov_avg_loss, color='tab:orange')
        plt.plot(loss_log_data.mean(axis=1), color='b')
        fig_name = "Avg_Train_Loss_" + now.strftime("%Y-%B-%d__%H-%M")
        fig_name = os.path.join(train_params['train_dir'], fig_name)   
        plt.savefig(fig_name)
        #############################################################################################

    #### Testing mode
    else:
        print("Testing the trained A2C RL-agent")
        Global_Model = ACNet(name=MODEL_NAME, model_dir=MODEL_DIR, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=network_arch)
        Global_Model.load_params()
        Global_Model.share_memory()
        global_optimizer = SharedRMSProp(Global_Model.parameters(), lr=train_params['learning_rate'])
        processes = []
        g_counter = mp.Value('i', 0) # global counter shared among processes
        a3c_worker = A3CWorker(worker_id=idx, 
                                global_model=Global_Model,
                                optimizer=global_optimizer,
                                params=test_params,
                                traffic_file=all_req_lists[idx], 
                                counter=g_counter,
                                net_arch=network_arch,
                                is_train=False)
        a3c_worker.start()
        a3c_worker.join()
        a3c_worker.terminate()

        #### TODO: Plot system throughput

    #############################################################################################

    # Check Global_Model vs. Before_Train_Model
    print("Return the Before_Train and After_Train models")
    return Before_Train_Model, Global_Model

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

#### Show model's weights
def get_model_weights(model):
    weights = dict()
    for layer_name, params in model.named_parameters():
        weights[layer_name] = params.detach().to('cpu').numpy()
    return weights

#### Difference between before and after training
def diff_weights(model1, model2):
    print("Calculate the weights differences between the Before_Train and After_Train models")
    diffs = dict()
    w1 = get_model_weights(model1)
    w2 = get_model_weights(model2)
    for key in w1:
        diffs[key] = w1[key] - w2[key]
    return diffs, w1, w2
    
        
    
#### Run the script
if __name__ == "__main__":
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
    
    # Choose neural network architecture
    network_arch = NET_ARCH['shared_net_w_RNN']
    operating_mode = OPERATION_MODE['train_mode']

    Before_Model, After_Model = main(operating_mode, network_arch)
    
    diff, before_weights, after_weights = diff_weights(Before_Model, After_Model)
    if network_arch == NET_ARCH['shared_net']:
        diff1 = np.sum(np.sqrt(diff['fc1.weight']**2)) + np.sum(np.sqrt(diff['fc1.bias']**2))
        diff2 = np.sum(np.sqrt(diff['fc2.weight']**2)) + np.sum(np.sqrt(diff['fc2.bias']**2))
        diff3 = np.sum(np.sqrt(diff['fc3.weight']**2)) + np.sum(np.sqrt(diff['fc3.bias']**2))
        diff_actor_lin = np.sum(np.sqrt(diff['actor_lin1.weight']**2)) + np.sum(np.sqrt(diff['actor_lin1.bias']**2))
        diff_critic_lin = np.sum(np.sqrt(diff['critic_lin1.weight']**2)) + np.sum(np.sqrt(diff['critic_lin1.bias']**2))

        print(f"diff fc1 = {diff1}")
        print(f"diff fc2 = {diff2}")
        print(f"diff fc3 = {diff3}")
        print(f"diff actor_lin1 = {diff_actor_lin}")
        print(f"diff critic_lin1 = {diff_critic_lin}")
    elif network_arch == NET_ARCH['shared_net_w_RNN']:
        # TODO: implement diff
        pass

    elif network_arch == NET_ARCH['separated_net']:
        # Actor
        a_diff1 = np.sum(np.sqrt(diff['actor_fc1.weight']**2)) + np.sum(np.sqrt(diff['actor_fc1.bias']**2))
        a_diff2 = np.sum(np.sqrt(diff['actor_fc2.weight']**2)) + np.sum(np.sqrt(diff['actor_fc2.bias']**2))

        # Critic
        c_diff1 = np.sum(np.sqrt(diff['critic_fc1.weight']**2)) + np.sum(np.sqrt(diff['critic_fc1.bias']**2))
        c_diff2 = np.sum(np.sqrt(diff['critic_fc2.weight']**2)) + np.sum(np.sqrt(diff['critic_fc2.bias']**2))
        c_diff3 = np.sum(np.sqrt(diff['critic_v.weight']**2)) + np.sum(np.sqrt(diff['critic_v.bias']**2))
    else:
        print("Network architect not supported!")



