# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 22:31:10 2021

@author: Onlyrich-Ryzen

Functions for plotting results    
"""

import matplotlib.pyplot as plt
import numpy as np
from data_utils_A3C_02 import mov_window_avg
import os

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
        vals = [float(l) for l in lines[:-6]] # do not read the 6 last lines
        data.append(vals)
    return np.array(data)

def read_throughputs(log_file):
    """Read data from multiple sub-files in log_file
        Return offered_load and real_throughput data from each sub-file

    Args:
        log_file ([string]): Full path to the log file

    Returns:
        data [list]: list of dictionary items, each corresponding to data from a sub-file
    """
    data = []
    for f in log_file:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        vals = {'offered_load':[], 'real_tp': []}
        # export data to vals, ignore the header line
        vals['offered_load'] = [float(l.split()[1]) for l in lines[1:]] 
        vals['real_tp'] = [float(l.split()[2]) for l in lines[1:]]
        data.append(vals)
    return data

def read_rsc_usages(log_file):
    """Read CPU, RAM, STORAGE, and BANDWIDTH usages data from sub-files in log_file
        
    Args:
        log_file (list): A list of sub-files containing resource usages of each worker

    Returns:
        data [type]: [description]
    """
    data = []
    for f in log_file:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        vals = {'cpu_usage':[], 'ram_usage': [], 'sto_usage': [], 'bw_usage': []}
        # export data to vals, ignore the header line
        vals['cpu_usage'] = [float(l.split()[0]) for l in lines[1:]] 
        vals['ram_usage'] = [float(l.split()[1]) for l in lines[1:]]
        vals['sto_usage'] = [float(l.split()[2]) for l in lines[1:]] 
        vals['bw_usage'] = [float(l.split()[3]) for l in lines[1:]]
        data.append(vals)
    return data


''' Plot Episode Rewards'''
def plot_episode_rwd(time, train_params):
    # Get the names of reward_log and loss_log files
    reward_logs = []
    for worker_id in range(train_params['n_workers']):
        reward_log = os.path.join(train_params['train_dir'], "W_" + str(worker_id) + "_ep_reward.log")
        reward_logs.append(reward_log)

    # Read reward data from log files
    reward_log_data = read_reward_data(reward_logs)

    # Plot average reward over episodes
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Reward over training episodes")
    plt.xlabel("Episodes",fontsize=22)
    plt.ylabel("Reward",fontsize=22)
    mov_avg_reward = mov_window_avg(reward_log_data.mean(axis=0), window_size=1000)
    plt.plot(mov_avg_reward, color='b')
    fig_name = "Avg_Train_Reward_" + time.strftime("%Y-%B-%d__%H-%M")
    fig_name = os.path.join(train_params['train_dir'], fig_name)
    plt.savefig(fig_name)
    return reward_log_data
    
def plot_loss_val(time, train_params):
    # Get the names of reward_log and loss_log files
    loss_logs = []
    for worker_id in range(train_params['n_workers']):
        loss_log = os.path.join(train_params['train_dir'], "W_" + str(worker_id) + "_losses.log")
        loss_logs.append(loss_log)
    
    # Read loss data from log files
    loss_log_data = read_loss_data(loss_logs)

    # Plot loss values over update_params steps
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Loss over training episodes")
    plt.xlabel("Update param steps",fontsize=22)
    plt.ylabel("Loss",fontsize=22)
    mov_avg_loss = mov_window_avg(loss_log_data.mean(axis=0), window_size=1000)
    plt.plot(mov_avg_loss, color='tab:orange')
    plt.plot(loss_log_data.mean(axis=1), color='b')
    fig_name = "Avg_Train_Loss_" + time.strftime("%Y-%B-%d__%H-%M")
    fig_name = os.path.join(train_params['train_dir'], fig_name)
    plt.savefig(fig_name)
    return loss_log_data

def plot_accum_accept_req(mode, params, time):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}

    # Read accept_ratio.log files
    ar_logs = []
    for w in range(params['n_workers']):
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fp = os.path.join(params['train_dir'], "W_" + str(w) + "_accept_ratio.log")
        else:
            fp = os.path.join(params['test_dir'], "W_" + str(w) + "_test_accept_ratio.log")
        ar_logs.append(fp)
    ar_log_datas = read_accept_ratio_data(ar_logs)
    
    # Accumulate n_accepted requests
    accum_accept_reqs = []
    for f in ar_log_datas:
        accum_count = 0
        counts = np.zeros(len(f))
        for i in range(len(f)):
            counts[i] = accum_count = f[i] + accum_count
        accum_accept_reqs.append(counts)
    
    # Plot results
    legs = ['W_' + str(i) for i in range(params['n_workers'])]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'fuchsia', 'teal', 'lime', 'pink']
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Accumulated accepted request over episodes")
    plt.xlabel("Episodes",fontsize=22)
    plt.ylabel("Num. of requests",fontsize=22)
    [plt.plot(accum_accept_reqs[i], color=colors[i]) for i in range(params['n_workers'])]
    plt.legend(legs)
    if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
        fig_name = os.path.join(params['train_dir'], 
                                "Train_Accum_Accept_Reqs_" + time.strftime("%Y-%B-%d__%H-%M"))   
    else:
        fig_name = os.path.join(params['test_dir'], 
                                "Test_Accum_Accept_Reqs_" + time.strftime("%Y-%B-%d__%H-%M"))
    plt.savefig(fig_name)
    return accum_accept_reqs, np.mean(ar_log_datas)*100


def plot_throughputs(mode, params, time):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}

    # Read throughputs log files
    tp_logs = []
    for w in range(params['n_workers']):
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fp = os.path.join(params['train_dir'], 'W_' + str(w) + "_throughputs.log")
        else:
            fp = os.path.join(params['test_dir'], 'W_' + str(w) + "_test_throughputs.log")
        tp_logs.append(fp)
    
    tp_log_datas = read_throughputs(tp_logs)
    
    # Plot results
    for i in range(params['n_workers']):
        # Real throughput vs. Offered load
        y = tp_log_datas[i]
        plt.figure(figsize=(10, 7.5))
        plt.plot(y['offered_load'], color='tab:blue')
        plt.plot(y['real_tp'], color='tab:orange')
        plt.xlabel("Time slot", fontsize=22)
        plt.ylabel("Throughput [bw unit]", fontsize=22)
        plt.grid()
        plt.legend(('Perfect Throughput (Offered load)', 'Real Throughput'), fontsize=20) 
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fig_name = os.path.join(params['train_dir'], "W_" + str(i) +
                                    "_Train_Throughput_" + time.strftime("%Y-%B-%d__%H-%M"))
        else:
            fig_name = os.path.join(params['test_dir'], "W_" + str(i) +
                                    "_Test_Throughput_" + time.strftime("%Y-%B-%d__%H-%M"))
        plt.savefig(fig_name)

        # Percentage gap between Real throughput vs. Offered load
        tp_gap = (np.array(y['real_tp']) - np.array(y['offered_load'])) / np.array(y['offered_load']) * 100
        plt.figure(figsize=(10, 7.5))
        plt.plot(tp_gap)
        plt.xlabel("Time slot", fontsize=22)
        plt.ylabel("Throughput gap [%]", fontsize=22)
        plt.grid()
        # plt.ylim((0, 10))
        # plt.legend(('Perfect Throughput (Offered load)', 'Real Throughput'), fontsize=20) 
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fig_name = os.path.join(params['train_dir'], "W_" + str(i) +
                                    "_Train_Throughput_Gap_" + time.strftime("%Y-%B-%d__%H-%M"))
        else:
            fig_name = os.path.join(params['test_dir'], "W_" + str(i) +
                                    "_Test_Throughput_Gap_" + time.strftime("%Y-%B-%d__%H-%M"))
        plt.savefig(fig_name)

    return tp_log_datas

def plot_rsc_usages(mode, params, time):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    # Read resources usage log files
    rsc_usage_logs = []
    for w in range(params['n_workers']):
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fp = os.path.join(params['train_dir'], 'W_' + str(w) + "_rsc_usage.log")
        else:
            fp = os.path.join(params['test_dir'], 'W_' + str(w) + "_test_rsc_usage.log")
        rsc_usage_logs.append(fp)
    
    rsc_usage_data = read_rsc_usages(rsc_usage_logs)

    # Plot CPU, RAM, STO usage rates [%] in one Figure for each worker
    for i in range(params['n_workers']):
        y = rsc_usage_data[i]
        for key in y:
            y[key] = np.array(y[key]) * 100 #convert to percentage
        plt.figure(figsize=(10, 7.5))
        plt.plot(y['cpu_usage'], color='red')
        plt.plot(y['ram_usage'], color='magenta')
        plt.plot(y['sto_usage'], color='green')
        plt.plot(y['bw_usage'], color='blue')
        plt.xlabel("Time slots", fontsize=22)
        plt.ylabel("Resource usage rates [%]", fontsize=22)
        plt.grid()
        plt.legend(('CPU', 'RAM', 'STO', 'BW'), fontsize=20)
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fig_name = os.path.join(params['train_dir'], "W_" + str(i) +
                                    "_Train_Resource_Usage_" + time.strftime("%Y-%B-%d__%H-%M"))
        else:
            fig_name = os.path.join(params['test_dir'], "W_" + str(i) +
                                    "_Test_Resource_Usage_" + time.strftime("%Y-%B-%d__%H-%M"))
        plt.savefig(fig_name)
    
    return rsc_usage_data

def read_delay_stress_data(log_files):
    datas = []
    for fname in log_files:
        x = []
        with open(fname, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            x.append(float(lines[i].split()[0]))
        datas.append(x)
    return datas

def plot_delay_stress_level(mode, params, time):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    
    delay_files = []
    for w in range(params['n_workers']):
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fp = os.path.join(params['train_dir'], 'W_' + str(w) + "_delay_stress.log")
        else:
            fp = os.path.join(params['test_dir'], 'W_' + str(w) + "_test_delay_stress.log")
        delay_files.append(fp)
    delay_datas = read_delay_stress_data(delay_files)

    for i in range(params['n_workers']):
        x = delay_datas[i]
        plt.figure(figsize=(10, 7.5))
        plt.xlabel("Delay stress level [%]", fontsize=22)
        plt.ylabel("Num. of requests", fontsize=22)
        plt.hist(x, bins=10)
        if OPERATION_MODE[mode] == OPERATION_MODE["train_mode"]:
            fig_name = os.path.join(params['train_dir'], 
                                'W_' + str(i) + "_delay_stress_" + time.strftime("%Y-%B-%d__%H-%M"))
        else:
            fig_name = os.path.join(params['test_dir'], 
                                'W_' + str(i) + "_test_delay_stress_" + time.strftime("%Y-%B-%d__%H-%M"))
        plt.savefig(fig_name)
    


