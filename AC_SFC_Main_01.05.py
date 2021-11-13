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
import os

from plot_results import plot_episode_rwd
os.environ['OMP_NUM_THREADS'] = '2' # 2 thread per CPU core for numpy
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
from datetime import datetime
from pathlib import Path
import json
import shutil
import argparse
from dataset_and_params import get_sfc_spec_file, get_train_datasets, get_test_datasets, get_train_test_params
from plot_results import plot_episode_rwd, plot_loss_val, plot_accum_accept_req


def read_reward_data(log_file):
    '''Read data from multiple sub-files in log_file
        Return a list where each sub-list contains data from each sub-file
    '''
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

def read_accept_ratio_data(log_file):
    '''Read data from multiple sub-files in log_file
        Return a list where each sub-list contains data from each sub-file
    '''
    data = []
    for f in log_file:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        vals = [float(l) for l in lines[:-1]] # do not read the last line
        data.append(vals)
    return np.array(data)


def worker(a3c_worker, g_counter):
    a3c_worker.run(g_counter)

# def main(is_train, network_arch, adv_style):
def main(args):
    ### Training Settings
    IS_NEIGHBOR_MASK = False# Neighborhood mask Default = False
    IS_CONTINUE_TRAINING = False # Continue training the DQN from the current DQN' weights

    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
    ADV_STYLE = {'n_step_return': 1, 'gae': 2}
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Train/Test mode '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    # is_train = OPERATION_MODE["test_mode"]
    
    #### Training parameters
    N_EPOCHS = args.epochs if not IS_CONTINUE_TRAINING else 200_000
    if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
        args.n_workers = args.n_workers #mp.cpu_count()
    else:
        args.n_workers = 1
    
    #### Create Model Folder 
    now = datetime.now()
    # Training directory
    TRAIN_DIR = 'train_results__' + now.strftime("%Y-%B-%d__%H-%M")
    if args.is_binary_state:
        TRAIN_DIR += "_BINARY_state"
    else:
        TRAIN_DIR += "_FRACTIONAL_state"
    TRAIN_DIR = os.path.join(args.model_dir, TRAIN_DIR)

    # Testing directory
    TEST_DIR = 'test_results__' + now.strftime("%Y-%B-%d__%H-%M")
    if args.is_binary_state:
        TEST_DIR += "_BINARY_state"
    else:
        TEST_DIR += "_FRACTIONAL_state"    
    TEST_DIR = os.path.join(args.model_dir, TEST_DIR)

    if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:    
        Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    else:
        Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
    
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training Dataset Folder
    # args.data_folder = "SOF_Data_Sets-v09\complete_data_sets"
    # args.data_folder = args.data_folder
    # Create a temporary edf_env for information extraction
    temp_edf = EdfEnv(data_path=args.data_folder,
                      net_file="ibm_15000_slots_1_con.net",
                      sfc_file=args.sfc_spec,
                      resource_scaler=args.rsc_scaler)
    
    # Using Binary state or Fractional state renderer
    if args.is_binary_state:
        temp_edf.binary_state_dim()
    else:
        temp_edf.fractional_state_dim()
        
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Configuration for the neural network
    # Neural net layers' neurons
    # INPUT_DIMS = temp_edf.n_state_dims
    INPUT_DIMS = temp_edf.n_state_dims
    ACTOR_OUTPUT_DIMS = 2 * temp_edf.n_nodes
    L_COMMON_1 = 128
    L_COMMON_2 = 128
    L_CRITIC_1 = 128
    L_RNN = 128
    if NET_ARCH[args.net_arch] == NET_ARCH['shared_net']:
        FC_HID_LAYERS = {'common_1': L_COMMON_1,
                         'common_2': L_COMMON_2,
                         'critic_1': L_CRITIC_1}
    elif NET_ARCH[args.net_arch] == NET_ARCH['shared_net_w_RNN']:
        FC_HID_LAYERS = {'common_1': L_COMMON_1,
                         'rnn_1': L_RNN}
    elif NET_ARCH[args.net_arch] == NET_ARCH['separated_net']:
        FC_HID_LAYERS = {'l1': L_COMMON_1,
                         'l2': L_COMMON_2}
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Export the parameter setting into a file
    if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
        param_setting_file = os.path.join(TRAIN_DIR, "parameter_settings.txt")
    else:
        param_setting_file = os.path.join(TEST_DIR, "parameter_settings.txt")

    with open(param_setting_file, 'a') as fp:
        print("SYSTEM PARAMETERS", file=fp)
        if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
            print(f"training directory: {TRAIN_DIR} \n", file=fp)
        else:
            print(f"testing directory: {TEST_DIR} \n", file=fp)
        
        print(f"args.is_binary_state = {args.is_binary_state}", file=fp)
        print(f"IS_NEIGHBOR_MASK = {IS_NEIGHBOR_MASK}", file=fp)
        print(f"IS_CONTINUE_TRAINING = {IS_CONTINUE_TRAINING}", file=fp)
        print(f'args.state_noise_scale = {args.state_noise_scale}', file=fp)
        
        print("NEURAL NETWORK SETTINGS", file=fp)
        print(f"INPUT_DIMS = {INPUT_DIMS}", file=fp)
        print(f"ACTOR_OUTPUT_DIMS = {ACTOR_OUTPUT_DIMS} \n", file=fp)
        if NET_ARCH[args.net_arch] == NET_ARCH['shared_net']:
            print(f"Network Architecture = Shared_Net", file=fp)
            print(f"FC_COMMON_1 = {L_COMMON_1}, FC_COMMON_2 = {L_COMMON_2}, FC_CRITIC_1 = {L_CRITIC_1}", file=fp)
        elif NET_ARCH[args.net_arch] == NET_ARCH['shared_net_w_RNN']:
            print(f"Network Architecture = Shared_Net with RNN", file=fp)
            print(f"FC_COMMON_1 = {L_COMMON_1}, RNN_CELL = {L_RNN}", file=fp)
        elif NET_ARCH[args.net_arch] == NET_ARCH['separated_net']:
            print(f"Network Architecture = Separated_Net", file=fp)
            print(f"FC_ACTOR_1 = {L_COMMON_1}, FC_ACTOR_2 = {L_COMMON_2}", file=fp)
            print(f"FC_CRITIC_1 = {L_COMMON_1}, FC_CRITIC_2 = {L_COMMON_2}", file=fp)
        # print(f"FC_HID_LAYERS  = {FC_HID_LAYERS}", file=fp)
        
        print("\nLearning parameters", file=fp)
        print(f"args.gamma = {args.gamma}", file=fp)
        print(f"args.tau = {args.tau}", file=fp)
        print(f"args.lr = {args.lr}", file=fp)
        print(f"args.critic_factor = {args.critic_factor}", file=fp)
        print(f"args.actor_factor = {args.actor_factor}", file=fp)
        print(f"args.entropy_factor = {args.entropy_factor}", file=fp)
        print(f"args.entropy_decay_val = {args.entropy_decay_val}", file=fp)
        print(f"args.entropy_decay_freq = {args.entropy_decay_freq}", file=fp)
        print(f"args.entropy_min = {args.entropy_min}", file=fp)

        print("\nOptimizer parameters", file=fp)
        print(f"Opt_Learning_Rate = {args.opt_lr}", file=fp)
        print(f"Opt_Epsilon = {args.opt_epsilon}", file=fp)
        print(f"Opt_Weight_Decay = {args.opt_weight_decay}", file=fp)
        print(f"Opt_Alpha = {args.opt_alpha}", file=fp)
        print(f"Opt_Momentum = {args.opt_momentum}", file=fp)
        print(f"Opt_Centered = {args.opt_centered}", file=fp)
        
        print("\nCost factors betas", file=fp)
        print(f"args.betas = {args.betas}", file=fp)
        print(f"Big_Reward = {args.big_rwd}", file=fp)

        if OPERATION_MODE["train_mode"]:
            print("\nTraining parameters", file=fp)
            print(f"args.n_workers = {args.n_workers}", file=fp)
            print(f"N_EPOCHS = {N_EPOCHS}", file=fp)
            print(f"args.train_freq = {args.train_freq}", file=fp)
            if ADV_STYLE[args.adv_style] == ADV_STYLE['n_step_return']:
                print("ADVANTAGE_STYLE = N_step return", file=fp)
            elif ADV_STYLE[args.adv_style] == ADV_STYLE['gae']:
                print(f"ADVANTAGE_STYLE = GAE", file=fp)

           
    print("Export system parameter setting into text files ... DONE!")
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    #### Create a Master Actor-Critic Agent
    Before_Train_Model = ACNet(name=args.model_name, model_dir=args.model_dir, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=NET_ARCH[args.net_arch])
    
    #### Load the trained network params
    if IS_CONTINUE_TRAINING:
        Before_Train_Model.load_params()
        print("CONTINUE TRAINING FROM PREVIOUS CHECKPOINT")
        print(f"Model directory: = {args.model_dir}")
        print(f"Model name: {args.model_name}")
    else:
        if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
            print("TRAINING THE NEURAL NETWORK FROM SCRATCH!")
    
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training & Testing datasets
    sfc_spec_file = get_sfc_spec_file()
    train_dataset_list = get_train_datasets()
    test_dataset_list = get_test_datasets(args.test_size)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Prepare the traffic request datasets '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    print("Preparing traffic request datasets")
    all_req_lists = []
    all_n_reqs = []
    #### TODO: implement it in parallel in the future
    if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
        for idx in range(args.n_workers):
            req_json_file = os.path.join(args.data_folder, train_dataset_list[idx]['traffic'])
            n_req, req_list = retrieve_sfc_req_from_json(req_json_file)
            print(f"Dataset {idx+1} is ")
            print(f"{train_dataset_list[idx]['net_topo']}")
            print(f"{train_dataset_list[idx]['traffic']}")
            print(f"Dataset {idx+1} has {len(req_list)} requests")
            all_n_reqs.append(n_req)
            all_req_lists.append(req_list)
    else:
        for idx in range(args.n_workers):
            req_json_file = os.path.join(args.data_folder, test_dataset_list[idx]['traffic'])
            n_req, req_list = retrieve_sfc_req_from_json(req_json_file)
            print(f"Dataset {idx+1} is ")
            print(f"{test_dataset_list[idx]['net_topo']}")
            print(f"{test_dataset_list[idx]['traffic']}")
            print(f"Dataset {idx+1} has {len(req_list)} requests")
            all_n_reqs.append(n_req)
            all_req_lists.append(req_list)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    #### Training & Testing params
    max_epochs = min([N_EPOCHS,] * args.n_workers, all_n_reqs)
    print(f"max_epochs = {max_epochs}")
    train_params, test_params = \
        get_train_test_params(max_epochs, sfc_spec_file, 
        train_dataset_list, TRAIN_DIR,
        test_dataset_list, TEST_DIR,
        FC_HID_LAYERS, INPUT_DIMS, ACTOR_OUTPUT_DIMS, 
        args)
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' A3C_Worker's task '''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    ''' Original Global_Model '''
    Global_Model = ACNet(name=args.model_name, model_dir=args.model_dir, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=NET_ARCH[args.net_arch])

    Global_Model.load_state_dict(Before_Train_Model.state_dict())

    #### Training mode
    if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
        print("Training the A3C RL-agent")
        # the model's parameters are globally shared among processes
        Global_Model.share_memory()
        global_optimizer = SharedRMSProp(Global_Model.parameters(), lr=args.opt_lr,
                                        eps=args.opt_epsilon, weight_decay=args.opt_weight_decay,
                                        alpha=args.opt_alpha, momentum=args.opt_momentum, centered=args.opt_centered)
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
                                   net_arch=NET_ARCH[args.net_arch],
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
        full_path = os.path.join(abs_path, args.model_dir)
        src_file = os.path.join(full_path, args.model_name + '.pt')
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
        plot_episode_rwd(reward_log_data, now, train_params)
        # # Plot episode rewards
        # plt.figure(figsize=(10,7.5))
        # plt.grid()
        # plt.title("Reward over training epochs")
        # plt.xlabel("Epochs",fontsize=22)
        # plt.ylabel("Reward",fontsize=22)
        # mov_avg_reward = mov_window_avg(reward_log_data.mean(axis=0), window_size=1000)
        # plt.plot(mov_avg_reward, color='b')
        # fig_name = "Avg_Train_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
        # fig_name = os.path.join(train_params['train_dir'], fig_name)
        # plt.savefig(fig_name)

        ''' Plot loss values'''
        # Read loss_log file data
        loss_log_data = read_loss_data(loss_logs)
        plot_loss_val(loss_log_data, now, train_params)
        # # Plot loss values over training steps
        # plt.figure(figsize=(10,7.5))
        # plt.grid()
        # plt.title("Loss over training epochs")
        # plt.xlabel("Epochs",fontsize=22)
        # plt.ylabel("Loss",fontsize=22)
        # mov_avg_loss = mov_window_avg(loss_log_data.mean(axis=0), window_size=1000)
        # plt.plot(mov_avg_loss, color='tab:orange')
        # plt.plot(loss_log_data.mean(axis=1), color='b')
        # fig_name = "Avg_Train_Loss_" + now.strftime("%Y-%B-%d__%H-%M")
        # fig_name = os.path.join(train_params['train_dir'], fig_name)
        # plt.savefig(fig_name)

        #############################################################################################

    #### Testing mode
    else:
        print("Testing the trained A2C RL-agent")
        Global_Model = ACNet(name=args.model_name, model_dir=args.model_dir, 
                       fc_hid_layers=FC_HID_LAYERS, 
                       input_dims=INPUT_DIMS, 
                       actor_dims=ACTOR_OUTPUT_DIMS,
                       net_arch=NET_ARCH[args.net_arch])
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
                                net_arch=NET_ARCH[args.net_arch],
                                is_train=False)
        a3c_worker.start()
        a3c_worker.join()
        a3c_worker.terminate()

        #### TODO: Plot system throughput

    #############################################################################################

    ''' Plot accum_n_accepted_req over epochs'''
    # Read accept_ratio.log files
    ar_logs = []
    for w in range(args.n_workers):
        if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
            fp = os.path.join(train_params['train_dir'], "W_" + str(w) + "_accept_ratio.log")
        else:
            fp = os.path.join(train_params['test_dir'], "W_" + str(w) + "_test_accept_ratio.log")
        ar_logs.append(fp)
    ar_log_datas = read_accept_ratio_data(ar_logs)

    accum_accept_reqs = plot_accum_accept_req(ar_log_datas, args.mode, train_params, now)
    # # Accumulate n_accepted requests
    # accum_accept_reqs = []
    # for f in ar_log_datas:
    #     accum_count = 0
    #     counts = np.zeros(len(f))
    #     for i in range(len(f)):
    #         counts[i] = accum_count = f[i] + accum_count
    #     accum_accept_reqs.append(counts)
    
    # legs = ['W_' + str(i) for i in range(args.n_workers)]
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'fuchsia', 'teal', 'lime', 'pink']
    # plt.figure(figsize=(10,7.5))
    # plt.grid()
    # plt.title("Accumulated accepted request over epochs")
    # plt.xlabel("Epochs",fontsize=22)
    # plt.ylabel("Num. of requests",fontsize=22)
    # [plt.plot(accum_accept_reqs[i], color=colors[i]) for i in range(args.n_workers)]
    # plt.legend(legs)
    # if OPERATION_MODE[args.mode] == OPERATION_MODE["train_mode"]:
    #     fig_name = os.path.join(train_params['train_dir'], 
    #                             "Train_Accum_Accept_Reqs_" + now.strftime("%Y-%B-%d__%H-%M"))   
    # else:
    #     fig_name = os.path.join(train_params['test_dir'], 
    #                             "Test_Accum_Accept_Reqs_" + now.strftime("%Y-%B-%d__%H-%M"))
    # plt.savefig(fig_name)

    # Print the avg acceptance ratio 
    print(f"Average request acceptance ratio of {args.n_workers} workers = {np.mean(ar_log_datas)*100}%")
    print("Return the Before_Train and After_Train models")
    return Before_Train_Model, Global_Model
#############################################################################################

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
    
#############################################################################################

def calc_model_diff(before_model, after_model, net_arch):
    NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
    diff, before_weights, after_weights = diff_weights(before_model, after_model)
    if NET_ARCH[net_arch] == NET_ARCH['shared_net']:
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
    elif NET_ARCH[net_arch] == NET_ARCH['shared_net_w_RNN']:
        # TODO: implement diff
        pass

    elif NET_ARCH[net_arch] == NET_ARCH['separated_net']:
        # Actor
        a_diff1 = np.sum(np.sqrt(diff['actor_fc1.weight']**2)) + np.sum(np.sqrt(diff['actor_fc1.bias']**2))
        a_diff2 = np.sum(np.sqrt(diff['actor_fc2.weight']**2)) + np.sum(np.sqrt(diff['actor_fc2.bias']**2))

        # Critic
        c_diff1 = np.sum(np.sqrt(diff['critic_fc1.weight']**2)) + np.sum(np.sqrt(diff['critic_fc1.bias']**2))
        c_diff2 = np.sum(np.sqrt(diff['critic_fc2.weight']**2)) + np.sum(np.sqrt(diff['critic_fc2.bias']**2))
        c_diff3 = np.sum(np.sqrt(diff['critic_v.weight']**2)) + np.sum(np.sqrt(diff['critic_v.bias']**2))
    else:
        print("Network architect not supported!")

#### Run the script
if __name__ == "__main__":
    #### Parsing arguments
    parser = argparse.ArgumentParser(description="A3C-based SFC Embedding and Routing")

    parser.add_argument("mode", type=str, help="Support options: train_mode, test_mode")
    parser.add_argument("--net_arch", default="shared_net", type=str, 
                        help="Neural network architecture. Support options: shared_net, shared_net_w_RNN, separated_net")
    parser.add_argument("--epochs", default=200_000, type=int, help="Number of episodes/epochs")
    parser.add_argument("--n_workers", default=12, type=int, help="Number of parallel workers. Support upto 12")
    parser.add_argument("--sfc_spec", default="sfc_file.sfc", help="SFC specification file")
    parser.add_argument("--data_folder", default="SOF_Data_Sets-v09\complete_data_sets", type=str, help="Dataset directory")
    parser.add_argument("--model_dir", default="A3C_SFCE_models", type=str, help="Directory storing the model")
    parser.add_argument("--model_name", default="AC_Agent", type=str, help="Name of the model")
    parser.add_argument("--train_freq", default=5, type=int, help="Num of episode per training")
    parser.add_argument("--adv_style", default="n_step_return", type=str, 
                        help="Method for calculating advantage. Support options: n_step_return, gae")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--gamma", default=0.88, type=float, help="Discount factor")
    parser.add_argument("--tau", default=1, type=float, help="GAE factor")
    parser.add_argument("--critic_factor", default=1, type=float, help="Weight associated to the Critic head")
    parser.add_argument("--actor_factor", default=0.25, type=float, help="Weight associated to the Actor head")
    parser.add_argument("--entropy_factor", default=0.1, type=float, help="Initial weight associated with the entropy term")
    parser.add_argument("--entropy_decay_val", default=0.01, type=float, help="Entropy decay value")
    parser.add_argument("--entropy_decay_freq", default=12_000, type=int, help="Entropy decay frequency")
    parser.add_argument("--entropy_min", default=0.01, type=float, help="Min value of the entropy factor")
    parser.add_argument("--betas", default=[1.0, 25, 15, 0.0], type=list, help="List of beta weights in the reward function")
    parser.add_argument("--big_rwd", default=3.0, type=float, help="Big reward value in the reward function")
    parser.add_argument("--n_steps", default=15, type=int, help="N-step values")
    parser.add_argument("--max_moves", default=25, type=int, help="Maximum number of steps per episode")
    parser.add_argument("--rsc_scaler", default=1, type=float, help="Resource scaler value in [0, 1]")
    parser.add_argument("--is_binary_state", default=True, type=bool, help="Using binary state or fractional state render")
    parser.add_argument("--state_noise_scale", default=100, type=float, help="Scale of the noise added to the state")
    parser.add_argument("--opt_lr", default=5e-4, type=float, help="Optimizer's learning rate")
    parser.add_argument("--opt_epsilon", default=1e-5, type=float, help="Optimizer's epsilon value")
    parser.add_argument("--opt_weight_decay", default=0, type=float, help="Optimizer's weight decay value")
    parser.add_argument("--opt_alpha", default=0.99, type=float, help="Optimizer's alpha value")
    parser.add_argument("--opt_momentum", default=0.2, type=float, help="Optimizer's momentum value")
    parser.add_argument("--opt_centered", default=False, type=bool, help="Whether using centered or not for the optimzer")
    parser.add_argument("--test_size", default="15k", type=str, 
                        help="Choose 15k-dataset or 100k-dataset for testing. Support options: 15k, 100k")

    args = parser.parse_args()
    if args.n_workers <= 0: 
        raise ValueError("A positive value is required.")
    if args.n_workers > 12: 
        raise ValueError("Support upto 12 workers.")
    
    if args.mode == "test_mode":
        args.n_workers = 1
        input("Test mode supports only 1 worker! Press Enter to continue testing the model...")
        
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
    ADV_STYLE = {'n_step_return': 1, 'gae': 2}# n-step return or GAE

    Before_Model, After_Model = main(args)
    calc_model_diff(Before_Model, After_Model, args.net_arch)

