# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 00:49:01 2021

@author: Onlyrich-Ryzen

Throughput comparison:
    + Offered load
    + Real_tp from DQN-based algo
    + Real_tp from RCSP-based algo (Huy's algorithm)

Change on Mon Nov 15 23:04:05 2021
    + Add a curve showing Real_tp from A3C-based algo

"""

import numpy as np
import torch
import os
from copy import copy
from IPython.display import clear_output
from matplotlib import pylab as plt
import json
import data_utils
from datetime import datetime
from pathlib import Path
import argparse

def main(args):
    now = datetime.now()# time stamp of running the script
    RESULT_DIR = args.result_dir#'results_comparison_2021_Nov_16'
    stime = args.start_time
    n_time_slots = 15_000
    x_vals = np.array(range(n_time_slots - stime))
    t_slots = np.array(range(stime, n_time_slots))

    #### Read DQN simulation results from file
    # Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    DQN_sim_file = os.path.join(RESULT_DIR, 'DQN_sim_results.log')
    dqn_results = {'time_slots':[], 'dqn_offered_load':[], 'dqn_real_tp':[]}
    with open(DQN_sim_file, 'r') as fp:
        dqn_lines = fp.readlines()

    # extract data to lists
    for i in range(1, len(dqn_lines)):
        line = dqn_lines[i].split(sep=' ')
        dqn_results['time_slots'].append(int(line[0]))
        dqn_results['dqn_offered_load'].append(float(line[1]))
        dqn_results['dqn_real_tp'].append(float(line[2]))


    #### Read RCSP simulation results from file
    RCSP_sim_file = os.path.join(RESULT_DIR, 'RCSP_sim_results.log')
    rcsp_results = {'time_slots':[], 'rcsp_offered_load':[], 'rcsp_real_tp':[]}
    with open(RCSP_sim_file, 'r') as fp:
        rcsp_lines = fp.readlines()

    # extract data to lists    
    for i in range(1, len(rcsp_lines)):
        line = rcsp_lines[i].split(sep=' ')
        rcsp_results['time_slots'].append(int(line[0]))
        rcsp_results['rcsp_offered_load'].append(float(line[1]))
        rcsp_results['rcsp_real_tp'].append(float(line[2]))

    # Obtain RCSP_offered_load and RCSP_real_throughput
    run_vals, run_starts, run_lens = data_utils.find_runs(rcsp_results['time_slots'])
    rcsp_offered_load = []
    rcsp_real_tp = []
    for i in range(1, n_time_slots+1):
        o_load = rcsp_results['rcsp_offered_load'][run_starts[i] - 1]
        rcsp_offered_load.append(o_load)
        r_tp = rcsp_results['rcsp_real_tp'][run_starts[i]-1]
        rcsp_real_tp.append(r_tp)


    #### Read A3C simulation results from file
    a3c_sim_file = os.path.join(RESULT_DIR, 'W_0_test_throughputs.log')
    a3c_results = {'time_slots':[], 'a3c_offered_load':[], 'a3c_real_tp':[]}
    with open(a3c_sim_file, 'r') as fp:
        a3c_lines = fp.readlines()

    # extract data to lists
    for i in range(1, len(dqn_lines)):
        line = a3c_lines[i].split(sep=' ')
        a3c_results['time_slots'].append(int(line[0]))
        a3c_results['a3c_offered_load'].append(float(line[1]))
        a3c_results['a3c_real_tp'].append(float(line[2]))


    #### Plot real_value throughput comparison
    font_size = 22
    plt.figure(figsize=(10, 7.5))
    plt.plot(t_slots, dqn_results['dqn_offered_load'][stime:], color='tab:orange', linewidth=2)#offered load
    # plt.plot(rcsp_offered_load, linewidth=4, color='y')
    plt.plot(t_slots, dqn_results['dqn_real_tp'][stime:], color='r', linewidth=2)# DQN real_tp
    plt.plot(t_slots, a3c_results['a3c_real_tp'][stime:], color='g', linewidth=2)# A3C real_tp
    plt.plot(t_slots, rcsp_real_tp[stime:], color='royalblue', linewidth=2)# RCSP real_tp
    # plt.xlim([stime, n_time_slots])
    plt.xticks(np.arange(stime, n_time_slots, 1000))
    plt.legend(('Offered Load', 'DQN Throughput', 'A3C Throughput' ,'RCSP Throughput'), fontsize=font_size)
    plt.xlabel('Time slots', fontsize=font_size)
    plt.ylabel('Throughput [bw unit]', fontsize=font_size)
    plt.grid()
    fig_name = os.path.join(RESULT_DIR, 'throughput_comparison--' + now.strftime("%Y-%B-%d__%H-%M"))
    plt.savefig(fig_name)


    #### Plot throughput rate
    dqn_tp_rate = np.array(dqn_results['dqn_real_tp']) / np.array(dqn_results['dqn_offered_load']) * 100.0
    a3c_tp_rate = np.array(a3c_results['a3c_real_tp']) / np.array(a3c_results['a3c_offered_load']) * 100.0
    rcsp_tp_rate = np.array(rcsp_real_tp) / np.array(rcsp_offered_load) * 100.0
    font_size = 22
    plt.figure(figsize=(10, 7.5))
    plt.plot(t_slots, dqn_tp_rate[stime:], color='r')
    plt.plot(t_slots, a3c_tp_rate[stime:], color='b')
    plt.plot(t_slots, rcsp_tp_rate[stime:], color='k')
    # plt.xlim([stime, n_time_slots])
    plt.xticks(np.arange(stime, n_time_slots, 1000))
    plt.ylim([90, 100])
    plt.xlabel('Time slots', fontsize=font_size)
    plt.ylabel('Rate [%]', fontsize=font_size)
    plt.legend(('DQN algo.', 'A3C algo.' ,'RCSP algo.'), fontsize=font_size)
    plt.grid()
    fig_name = os.path.join(RESULT_DIR, 'throughput_rate_comparison--' + now.strftime("%Y-%B-%d__%H-%M"))
    plt.savefig(fig_name)

    #### Average throughout
    dqn_avg_tp = np.array(dqn_results['dqn_real_tp'][stime:]).mean()
    print(f'Avg. throughput from DQN algo. [bw unit]: {dqn_avg_tp:.3f}')

    a3c_avg_tp = np.array(a3c_results['a3c_real_tp'][stime:]).mean()
    print(f'Avg. throughput from A3C algo. [bw unit]: {a3c_avg_tp:.3f}')

    rcsp_avg_tp = np.array(rcsp_real_tp[stime:]).mean()
    print(f'Avg. throughput from RCSP algo. [bw unit]: {rcsp_avg_tp:.3f}')

    #### Average gap
    dqn_avg_gap = dqn_tp_rate[stime:].mean()
    dqn_var_gap = dqn_tp_rate[stime:].var()
    print(f'Average throughput gap [%] of DQN-based algorithm: {dqn_avg_gap:.3f} (+/- {dqn_var_gap:.3f})')

    a3c_avg_gap = a3c_tp_rate[stime:].mean()
    a3c_var_gap = a3c_tp_rate[stime:].var()
    print(f'Average throughput gap [%] of A3C-based algorithm: {a3c_avg_gap:.3f} (+/- {a3c_var_gap:.3f})')

    rcsp_avg_gap = rcsp_tp_rate[stime:].mean()
    rcsp_var_gap = rcsp_tp_rate[stime:].var()
    print(f'Average throughput gap [%] of RCSP-based algorithm: {rcsp_avg_gap:.3f} (+/- {rcsp_var_gap:.3f})')

    #### Export comparison results to file
    summary_file = os.path.join(RESULT_DIR, 'results_summary_' + now.strftime("%Y-%B-%d__%H-%M") + '.log')
    with open(summary_file, 'a') as fp:
        print(f'Avg. throughput from DQN algo. [bw unit]: {dqn_avg_tp:.3f}', file=fp)
        print(f'Avg. throughput from A3C algo. [bw unit]: {a3c_avg_tp:.3f}', file=fp)
        print(f'Avg. throughput from RCSP algo. [bw unit]: {rcsp_avg_tp:.3f}', file=fp)

        print(f'Average throughput gap [%] of DQN-based algorithm: {dqn_avg_gap:.3f} (+/- {dqn_var_gap:.3f})', file=fp)
        print(f'Average throughput gap [%] of A3C-based algorithm: {a3c_avg_gap:.3f} (+/- {a3c_var_gap:.3f})', file=fp)
        print(f'Average throughput gap [%] of RCSP-based algorithm: {rcsp_avg_gap:.3f} (+/- {rcsp_var_gap:.3f})', file=fp)

#### Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare throughput among algoriths: DQN, A3C, RSCP")
    parser.add_argument("result_dir", type=str, help="Folder storing the results")
    parser.add_argument("--start_time", default=0, type=int, help="Starting time slot for throughput calculation")
    args = parser.parse_args()

    main(args)
    print('Done!')
