# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 23:38:22 2021

@author: Onlyrich-Ryzen

Plot the (moving average) reward and CDF of accepted requests 
over the training episodes
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import data_utils
from datetime import datetime
from pathlib import Path
import argparse

#### Plot the reward and accumulated accepted request in a same figure
# N_EPOCHS = 10
# x = np.linspace(0, N_EPOCHS, 11)
# moving_avg_reward = np.linspace(10,20, 11)
# cdf_accum = np.linspace(10_000, 0, 11)


# # reward curve
# plt.figure(figsize=(10,7.5))
# fig, ax = plt.subplots()
# ax.plot(moving_avg_reward, color='red', linestyle='-')
# ax.set_xlabel("Episodes", fontsize=20)
# ax.set_ylabel("Reward", fontsize=20)


# # accum accepted request curve
# ax2 = ax.twinx()
# ax2.plot(cdf_accum, color='blue', linestyle = '-.')
# ax2.set_ylabel("Num. of Requests", fontsize=18)
# plt.grid(axis='both')


def main():
    now = datetime.now()
    # RESULT_DIR = args.result_dir
    RESULT_DIR = "results_comparison_2021_Dec_02"
    NTimeSlots = 200_000
    
    
    # Read the A3C Accepted request log
    A3cAccReqFile = os.path.join(RESULT_DIR, 'W_0_accept_ratio.log')
    with open(A3cAccReqFile, 'r') as fp:
        A3cAccReqLines = fp.readlines()
    
    A3cCdfAcc = []
    TotAccReq = 0
    for i in range(len(A3cAccReqLines) - 5):
        TotAccReq += float(A3cAccReqLines[i].split()[0])
        A3cCdfAcc.append(TotAccReq)
    
    # Read the A3C Episode reward log
    A3cRwdFile = os.path.join(RESULT_DIR, 'W_0_ep_reward.log')
    with open(A3cRwdFile, 'r') as fp:
        A3cRwdLines = fp.readlines()
    A3cRwdEps = [float(A3cRwdLines[i].split()[0]) for i in range(NTimeSlots)]
    A3cMovRwd = data_utils.mov_window_avg(A3cRwdEps, 1000)

    # px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    dpi = 125
    # Plot results
    plt.figure(figsize=(1500/dpi, 1000/dpi), dpi=dpi)
    fig, ax = plt.subplots()
    ax.plot(A3cMovRwd, color='red', linestyle='-')
    ax.set_xlabel("Episodes", fontsize=22)
    ax.set_ylabel("Reward", fontsize=22, color='red')

    ax2 = ax.twinx()
    ax2.plot(A3cCdfAcc, color='blue', linestyle = '-.')
    ax2.set_ylabel("Num. of Requests", fontsize=22, color='blue')
    # plt.grid(axis='both')
    fig_name = os.path.join(RESULT_DIR, "A3C_Rwd_CDF_Accept_Req_" + now.strftime("%Y-%B-%d__%H-%M"))
    plt.savefig(fig_name, dpi=dpi)

# #### Run the script
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot Reward and CDF Accepted Requests in the same figure")
#     parser.add_argument("result_dir", type=str, help="Folder storing the results")
#     parser.add_argument("--n_time_slots", default=200_000, type=int, 
#                         help="Number of time slots in the training")
#     # parser.add_argument("--start_time", default=0, type=int, help="Starting time slot for throughput calculation")
#     args = parser.parse_args()

#     main(args)
#     print('Done!')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Compare throughput among algoriths: DQN, A3C, RSCP")
    # parser.add_argument("result_dir", type=str, help="Folder storing the results")
    # args = parser.parse_args()
    # main(args)
    main()
    print("Done!")