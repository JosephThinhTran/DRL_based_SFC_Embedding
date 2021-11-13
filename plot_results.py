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

''' Plot Episode Rewards'''
def plot_episode_rwd(reward_log_data, time, train_params):
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Reward over training epochs")
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Reward",fontsize=22)
    mov_avg_reward = mov_window_avg(reward_log_data.mean(axis=0), window_size=1000)
    plt.plot(mov_avg_reward, color='b')
    fig_name = "Avg_Train_Reward_" + time.strftime("%Y-%B-%d__%H-%M")
    fig_name = os.path.join(train_params['train_dir'], fig_name)
    plt.savefig(fig_name)
    return 0
    
def plot_loss_val(loss_log_data, time, params):
    # Plot loss values over update_params steps
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Loss over training epochs")
    plt.xlabel("Update param steps",fontsize=22)
    plt.ylabel("Loss",fontsize=22)
    mov_avg_loss = mov_window_avg(loss_log_data.mean(axis=0), window_size=1000)
    plt.plot(mov_avg_loss, color='tab:orange')
    plt.plot(loss_log_data.mean(axis=1), color='b')
    fig_name = "Avg_Train_Loss_" + time.strftime("%Y-%B-%d__%H-%M")
    fig_name = os.path.join(params['train_dir'], fig_name)
    plt.savefig(fig_name)
    return 0

def plot_accum_accept_req(ar_log_datas, mode, params, time):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    
    # Accumulate n_accepted requests
    accum_accept_reqs = []
    for f in ar_log_datas:
        accum_count = 0
        counts = np.zeros(len(f))
        for i in range(len(f)):
            counts[i] = accum_count = f[i] + accum_count
        accum_accept_reqs.append(counts)
    
    # Plot result
    legs = ['W_' + str(i) for i in range(params['n_workers'])]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'fuchsia', 'teal', 'lime', 'pink']
    plt.figure(figsize=(10,7.5))
    plt.grid()
    plt.title("Accumulated accepted request over epochs")
    plt.xlabel("Epochs",fontsize=22)
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
    return accum_accept_reqs
