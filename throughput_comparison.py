# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 00:49:01 2021

@author: Onlyrich-Ryzen

Throughput comparison:
    + Offered load
    + Real_tp from DQN-based algo
    + Real_tp from RCSP-based algo (Huy's algorithm)
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

now = datetime.now()# time stamp of running the script

#### Read DQN simulation results from file
RESULT_DIR = 'results_comparison_2021_Aug_18'
# Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
DQN_sim_file = os.path.join(RESULT_DIR, 'DQN_sim_results.log')
n_time_slots = 15_000
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

#### Plot real_value throughput comparison
font_size = 22
plt.figure(figsize=(10, 7.5))
plt.plot(dqn_results['dqn_offered_load'], color='orange', linewidth=2)
# plt.plot(rcsp_offered_load, linewidth=4, color='y')
plt.plot(dqn_results['dqn_real_tp'], color='r', linewidth=2)
plt.plot(rcsp_real_tp, color='b', linewidth=2)
# plt.legend(('DQN Offered Load', 'RCSP Offered Load', 'DQN Real Throughput', 'RCSP Real Throughput'), fontsize=font_size)
plt.legend(('Offered Load', 'DQN Throughput', 'RCSP Throughput'), fontsize=font_size)
plt.xlabel('Time slots', fontsize=font_size)
plt.ylabel('Throughput [bw unit]', fontsize=font_size)
plt.grid()
fig_name = os.path.join(RESULT_DIR, 'throughput_comparison--' + now.strftime("%Y-%B-%d__%H-%M"))
plt.savefig(fig_name)


#### Plot throughput rate
dqn_tp_rate = np.array(dqn_results['dqn_real_tp']) / np.array(dqn_results['dqn_offered_load']) * 100.0
rcsp_tp_rate = np.array(rcsp_real_tp) / np.array(rcsp_offered_load) * 100.0
font_size = 22
plt.figure(figsize=(10, 7.5))
plt.plot(dqn_tp_rate, color='r')
plt.plot(rcsp_tp_rate, color='k')
plt.xlabel('Time slots', fontsize=font_size)
plt.ylabel('Rate [%]', fontsize=font_size)
plt.ylim([90, 100])
plt.legend(('DQN algo.', 'RCSP algo.'), fontsize=font_size)
plt.grid()
fig_name = os.path.join(RESULT_DIR, 'throughput_rate_comparison--' + now.strftime("%Y-%B-%d__%H-%M"))
plt.savefig(fig_name)

#### Average throughout
dqn_avg_tp = np.array(dqn_results['dqn_real_tp']).mean()
print(f'Avg. throughput [bw unit]: {dqn_avg_tp:.3f}')
rcsp_avg_tp = np.array(rcsp_real_tp).mean()
print(f'Avg. throughput [bw unit]: {rcsp_avg_tp:.3f}')

#### Average gap
dqn_avg_gap = dqn_tp_rate.mean()
dqn_var_gap = dqn_tp_rate.var()
print(f'Average throughput gap [%] of DQN-based algorithm: {dqn_avg_gap:.3f} (+/- {dqn_var_gap:.3f})')
rcsp_avg_gap = rcsp_tp_rate.mean()
rcsp_var_gap = rcsp_tp_rate.var()
print(f'Average throughput gap [%] of RCSP-based algorithm: {rcsp_avg_gap:.3f} (+/- {rcsp_var_gap:.3f})')

#### Export comparison results to file
summary_file = os.path.join(RESULT_DIR, 'results_summary' + now.strftime("%Y-%B-%d__%H-%M") + '.log')
with open(summary_file, 'a') as fp:
    print(f'Avg. throughput [bw unit]: {dqn_avg_tp:.3f}', file=fp)
    print(f'Avg. throughput [bw unit]: {rcsp_avg_tp:.3f}', file=fp)
    print(f'Average throughput gap [%] of DQN-based algorithm: {dqn_avg_gap:.3f} (+/- {dqn_var_gap:.3f})', file=fp)
    print(f'Average throughput gap [%] of RCSP-based algorithm: {rcsp_avg_gap:.3f} (+/- {rcsp_var_gap:.3f})', file=fp)