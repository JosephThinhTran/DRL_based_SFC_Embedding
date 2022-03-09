# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 2022
@author: Onlyrich-Ryzen

- Testing function for the trained Q-learning agent

- Network topology: CONUS

- Changelog:
    + Based on the "test_model_random_req_order_v01.08.py"
    + Used for training with the CONUS dataset Data_18's parameters
    + Use data_utils_02.py for reading dataset
    + Use edf_env_v2_02.py due to the new dataset's format
"""

import numpy as np
import torch
import os
from copy import copy
from IPython.display import clear_output
from matplotlib import pylab as plt
import json
from edf_env_CONUS_2_02 import EdfEnv
import data_utils_CONUS_02
from datetime import datetime
from pathlib import Path
from q_models_CONUS import DQAgent

#### Test Settings
IS_ORDERED_REQ = True  # Always use ordered sequence of requests
IS_BINARY_STATE = True  # Use binary state by default
IS_NEIGHBOR_MASK = True  # Use Neighborhood mask by default
STATE_NOISE_SCALE = 50.

#### Build the EDF network environment
# Test dataset - Use for performance comparison
DATA_FOLDER = r"SOF_Data_Sets-2022_Mar_07\complete_data_sets\Data_18"
traffic_file_name = "non_uniform_CONUS36_reordered_traffic_20000_slots_1_con.tra"
req_json_file = os.path.join(DATA_FOLDER, traffic_file_name)  # service request traffic
# Create EDF environment
edf = EdfEnv(data_path=DATA_FOLDER, 
             net_file='non_uniformCONUS36_20000_slots_1_con.net', 
             sfc_file='CONUS36_sfc_file.sfc')

# Using Binary state or Fractional state renderer
if IS_BINARY_STATE:
    edf.binary_state_dim()
else:
    edf.fractional_state_dim()

#### Obtain service requests
n_req, req_list = data_utils_CONUS_02.retrieve_sfc_req_from_json_CONUS(req_json_file)

#### Build the Q-learning RL agent'''
# Neural net layers' neurons
INPUT_DIMS = edf.n_state_dims
L2 = 256
L3 = 256
FC_HID_LAYERS = np.array([L2, L3])
OUTPUT_DIMS = 2 * edf.n_nodes

# Learning parameters
GAMMA = 0.785
LEARNING_RATE = 1e-3
EPSILON = 0.6  
DECAY_RATE = 0.9992

# Training parameters
N_EPOCHS = min(20_000, n_req)#  number of requests for testing
BUFFER_SIZE = 10_000
BATCH_SIZE = 200
SYNC_FREQ = 100


#### Testing Folder
MODEL_DIR = 'CONUS_DQN_SFCE_models'
now = datetime.now()
if IS_ORDERED_REQ:
    if np.mod(N_EPOCHS, 1000) == 0:
        test_size = int(N_EPOCHS / 1000)
        TEST_DIR = 'test_results__' + \
            str(test_size) + 'k_' + now.strftime("%Y-%B-%d__%H-%M")
    else:
        TEST_DIR = 'test_results__' + \
            str(N_EPOCHS) + '_' + now.strftime("%Y-%B-%d__%H-%M")
else:
    if np.mod(N_EPOCHS, 1000) == 0:
        test_size = int(N_EPOCHS / 1000)
        TEST_DIR = 'test_results__' + \
            str(test_size) + 'k_random_' + now.strftime("%Y-%B-%d__%H-%M")
    else:
        TEST_DIR = 'test_results__' + \
            str(N_EPOCHS) + '_random_' + now.strftime("%Y-%B-%d__%H-%M")

TEST_DIR = os.path.join(MODEL_DIR, TEST_DIR)
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
test_log = os.path.join(TEST_DIR, 'test.log')

#### Q-agent
q_agent = DQAgent(model_dir=MODEL_DIR,
                  fc_hid_layers=FC_HID_LAYERS,
                  input_dims=INPUT_DIMS,
                  output_dims=OUTPUT_DIMS,
                  gamma=GAMMA,
                  learning_rate=LEARNING_RATE,
                  epsilon=EPSILON,
                  buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE)

#### Load the trained network params
q_agent.load_checkpoint(is_train=False)

#### Data log
losses = []  # Loss values
total_iters = 0
max_moves = 100  # Max number of routing steps (iterations) per episode
reward_hist = []  # list of reward per episode
delay_rate_hist = []  # list of delay ratio  per accepted request

registered_req = {'registered_request': []}  # list of accepted SFC requests
active_req = []  # list of SFC requests still in the EDF
expired_req = []  # list of expired SFC requests
serving_req = [] # list of accepted requests
rejected_req = []  # list of rejected SFC requests

sfc_embed_map = {}
vnf_embed_map = {}
success_embed_count = []  # couting the successful SFC requests

### TODO: REVISE the calculation of resource usages
# CPU, RAM, STORAGE usage_rate over time_slot
tot_cpu, tot_ram, tot_sto = edf.resource_mat.sum(axis=0)#CPU, RAM, STORAGE capacities

# total_bw
real_net_bw = copy(edf.bw_mat)
np.fill_diagonal(real_net_bw, 0) # real system_bw does not take self-connected link bw into account
tot_bw = real_net_bw.sum()
cpu_usage = []
ram_usage = []
sto_usage = []
bw_usage = []

# Throughput [bw unit] per time slot
all_arrive_req = [] # list of perfect SFC requests

# Keep track accepted/rejected requests per time slot
cur_time_slot = -1  # current time slot indicator
# dict of accepted/rejected request per time slot
adm_hist = {}  # key=time_slot, val=adm_per_slot
adm_per_slot = {}  # key=req_id, val=1/0 (accepted/rejected)

#### Main testing loop
start_id = 0
end_id = n_req
# list of req_id in the testing dataset
# req_id_list = list(range(start_id, end_id))
req_id_list = []
for key in req_list.keys():
    req_id_list.append(key)

#starting time slot id
start_time_slot  = req_list[start_id].arrival_time 

for epoch in range(N_EPOCHS):
    # Get a random request_id from the dataset for the epoch-th
    if IS_ORDERED_REQ:
        idx = req_id_list[epoch]
    else:
        idx = np.random.choice(req_id_list)
        req_id_list.remove(idx)  # remove idx from the list

    print(f'Request ID: {idx}')
    req = req_list[idx]  # Get information of the request with idx
    all_arrive_req.append(req.id)
    print(f'Req_Id={req.id},   source={req.source}   destination={req.destination}   {req.sfc_id}:[{edf.sfc_specs[req.sfc_id]}]   bw={req.bw}   delay_req={req.delay_req}')
    with open(test_log, 'a') as log_fp:
        print(f'Req_Id={req.id},   source={req.source}   destination={req.destination}   {req.sfc_id}:[{edf.sfc_specs[req.sfc_id]}]   bw={req.bw}   delay_req={req.delay_req}', file=log_fp)
    
    # Check new time slot
    if req.arrival_time >= cur_time_slot + 1 + start_time_slot:
        cur_time_slot += 1
        adm_hist[cur_time_slot] = {}  # create a dict for the new time slot
        # Calculate resource usage rates per time slot
        # starting at the end of the first time slot
        if cur_time_slot >= 1:
            # CPU, RAM, STO usage rates per time slot
            remain_cpu, remain_ram, remain_sto = edf.resource_mat.sum(axis=0)
            cpu_usage.append((tot_cpu - remain_cpu) / tot_cpu * 100.0)
            ram_usage.append((tot_ram - remain_ram) / tot_ram * 100.0)
            sto_usage.append((tot_sto - remain_sto) / tot_sto * 100.0)
            # BW usage rate per time slot
            remain_real_bw = copy(edf.bw_mat)
            np.fill_diagonal(remain_real_bw, 0)# do not take self-connected link bw into account
            tot_remain_bw = remain_real_bw.sum()
            bw_usage.append((tot_bw - tot_remain_bw) / tot_bw * 100.0)
        # TODO: Only consider the active resource nodes
        
        # Get back resource from the EXPIRED accepted requests
        if cur_time_slot >= 1:
            edf.sojourn_monitor(active_req, req_list, 
                                sfc_embed_map, vnf_embed_map, 
                                cur_time_slot + start_time_slot, expired_req)
            
        
    # Get the required VNF list
    vnf_list = copy(edf.sfc_specs[req.sfc_id])
    vnf_item = 0
    hol_vnf_name = vnf_list[vnf_item]
    cur_node_id = req.source

    # Create list of nodes embedding the current SFC request
    sfc_embed_nodes = [cur_node_id]  # SFC node mapping
    vnf_embed_nodes = {}  # VNF compute node
    success_embed = 0

    # Get delay constraint
    residual_delay = req.delay_req

    # Get a copy of the current resource and bandwidth information
    edf.temp_resource_mat = copy(edf.resource_mat)
    edf.temp_bw_mat = copy(edf.bw_mat)
    
    # Prepare the system state
    if IS_BINARY_STATE:
        state1 = edf.render_state_binary(req, hol_vnf_name, cur_node_id)
    else:
        state1 = edf.render_state_frac(req, hol_vnf_name, cur_node_id)
    state1 = state1.reshape(1, INPUT_DIMS) + \
        np.random.rand(1, INPUT_DIMS) / STATE_NOISE_SCALE
    state1 = torch.from_numpy(state1).float()

    # Perform SFC embedding & Routing
    stop_flag = 0
    mov = 0
    this_episode_rwd = 0

    while(stop_flag < 1):
        total_iters += 1
        mov += 1

        # Obtain raw_action from DQN and convert to real_action
        if IS_NEIGHBOR_MASK:
            neighbor_filter = np.tile(edf.adj_mat[cur_node_id], 2)
            neighbor_filter[neighbor_filter < 1] = -1e6
            action_raw = q_agent.act(state1, neighbor_filter)
        else:
            action_raw = q_agent.act(state1)
        action = edf.to_real_action(action_raw)

        # Update the resources edf.temp_resource_mat, edf.temp_bw_mat
        resource_failed, link_failed = edf.make_move(action, req, hol_vnf_name, cur_node_id)

        # Update delay budget
        residual_delay, added_delay, proc_latency, new_link_latency = \
            edf.cal_residual_delay(
                residual_delay, action, req, hol_vnf_name, cur_node_id)

        # Update the hol_vnf_name
        prev_hol_vnf_name = hol_vnf_name
        if action.is_embed > 0:
            if vnf_item < len(vnf_list):
                # Add to vnf_embed_nodes
                vnf_embed_nodes[hol_vnf_name] = cur_node_id
                # Continue to embed the next vnf
                vnf_item += 1
                if vnf_item < len(vnf_list):
                    hol_vnf_name = vnf_list[vnf_item]
                else:
                    hol_vnf_name = 'VNF_Done'

        # Update the cur_node_id and list sfc_embed_nodes
        prev_node_id = cur_node_id
        cur_node_id = action.next_node
        sfc_embed_nodes.append(cur_node_id)  # SFC node mapping

        # Check SFC embedding and routing accomplishement
        if (cur_node_id == req.destination) and (hol_vnf_name == 'VNF_Done'):
            done_embed = 1
        else:
            done_embed = 0

        # Obtain the next state from the EDF environment
        if IS_BINARY_STATE:
            state2 = edf.render_state_binary(req, hol_vnf_name, cur_node_id)
        else:
            state2 = edf.render_state_frac(req, hol_vnf_name, cur_node_id)
        state2 = state2.reshape(1, INPUT_DIMS) + \
            np.random.rand(1, INPUT_DIMS) / STATE_NOISE_SCALE
        state2 = torch.from_numpy(state2).float()

        # Calculate reward
        reward = edf.reward(action, residual_delay, proc_latency,
                            new_link_latency, done_embed, prev_node_id, prev_hol_vnf_name)

        # Accumulate episode reward
        this_episode_rwd += reward

        # Condition for succesfull SFC embedding
        if residual_delay < 0 or reward < -1:
            fail_embed = 1
            success_embed = 0
        else:
            fail_embed = 0
            success_embed = 1 if done_embed > 0 else 0

        # Update system state for the next step
        state1 = state2

        print(f'Time_{cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req.sfc_id}-{prev_hol_vnf_name}   Action=(is_embed={action.is_embed}; next_node={action.next_node})   Residual_delay={residual_delay:.4f}   Reward={reward:.3f}   Success_embed={success_embed}')
        if resource_failed != 'None': print(resource_failed)
        if link_failed != 'None' : print(link_failed)
        with open(test_log, 'a') as log_fp:
            print(f'Time_{cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req.sfc_id}-{prev_hol_vnf_name}   Action=(is_embed={action.is_embed}; next_node={action.next_node})   Residual_delay={residual_delay:.4f}   Reward={reward:.3f}   Success_embed={success_embed}', file=log_fp)
            if resource_failed != 'None':
                print(resource_failed, file=log_fp)
            if link_failed != 'None':
                print(link_failed, file=log_fp)
        clear_output(wait=True)

        # Stopping criteria
        if success_embed > 0 or fail_embed > 0 or mov > max_moves:
            stop_flag = 1
            mov = 0

        # Register the NEWLY successul embedded request to the serving list
        if success_embed > 0:
            sfc_embed_map[req.id] = sfc_embed_nodes
            vnf_embed_map[req.id] = vnf_embed_nodes
            active_req.append(req.id)
            serving_req.append(req.id)

            # Associate SFC embedding and VNF embedding maps to the req
            # convert namedtuple req to dict type
            accepted_req = dict(req._asdict())
            accepted_req.update({'sfc_map': sfc_embed_nodes})
            accepted_req.update({'vnf_embed_map': vnf_embed_nodes})
            # Add the new accepted_req to a dict
            registered_req['registered_request'].append(accepted_req)

        # Update the resource consumption of this episode to EDF resource availability
        # if the SFC embedding is successfull
        if success_embed > 0:
            edf.resource_mat = copy(edf.temp_resource_mat)
            edf.bw_mat = copy(edf.temp_bw_mat)

        # Count the successful SFC requests
        if stop_flag == 1:
            if success_embed > 0:
                success_embed_count.append(1)
            else:
                success_embed_count.append(0)

        # Calculate the e2e delay ratio for the accepted request
        if success_embed > 0:
            delay_rate = (req.delay_req - residual_delay) / req.delay_req
            delay_rate_hist.append(delay_rate)

        # Keep track of accepted/rejected request per time slot
        if stop_flag == 1:
            if success_embed > 0:
                adm_hist[cur_time_slot].update({req.id: 1})  # accepted request
            else:
                adm_hist[cur_time_slot].update({req.id: 0})  # rejected request
    # End of inner while loop

    # Add episode_reward to the list
    reward_hist.append(this_episode_rwd)

    # END OF OUTER EPOCH LOOP

new_now = datetime.now()
print("start simulation time" + now.strftime("%Y-%B-%d__%H-%M-%S"))
print("start simulation time" + new_now.strftime("%Y-%B-%d__%H-%M-%S"))

#### Export registered_req to JSON file
regis_file = os.path.join(TEST_DIR, 'registered_req_log.json')
with open(regis_file, 'w') as rf:
    json.dump(registered_req, rf, indent=2, default=data_utils_CONUS_02.numpy_encoder)

#### Plot the episode_reward history
reward_hist = np.array(reward_hist)
reward_log_file = os.path.join(TEST_DIR, 'test_episode_rewards.log')
with open(reward_log_file, 'a') as rfp:
    [print(reward_hist[i], file=rfp) for i in range(len(reward_hist))]

moving_avg_reward = data_utils_CONUS_02.mov_window_avg(reward_hist, 1000)
plt.figure(figsize=(10, 7.5))
plt.grid()
# plt.plot(reward_hist, color='tab:blue')
plt.plot(moving_avg_reward, color='tab:orange')
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Reward", fontsize=22)
# plt.legend(("Rewards", "Moving Average Rewards"),fontsize=20)
if IS_BINARY_STATE:
    fig_name = "Test_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "FRAC_Test_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)

# PERFORMANCE EVALUATION
#### Export request acceptance ratio to file
print('TESTING PROCESS...DONE!')
accepted_req_file = os.path.join(TEST_DIR, 'test_accepted_req.log')
accept_ratio = len(sfc_embed_map) / N_EPOCHS
print(f'Total requests arrived at the network: {N_EPOCHS}')
print(f'Total time slots: {cur_time_slot +1}')
if IS_NEIGHBOR_MASK:
    print('Use Neighborhood filter')
print(f'Request acceptance ratio: {accept_ratio*100:.2f}%')
print(f'Average episode reward: {np.mean(reward_hist):.3f}')
with open(accepted_req_file, 'w') as afp:
    print(f'Traffic dataset: {traffic_file_name}', file=afp)
    print(f'Total requests arrived at the network: {N_EPOCHS}', file=afp)
    print(f'Total time slots: {cur_time_slot +1}', file=afp)
    if IS_NEIGHBOR_MASK:
        print('Use Neighborhood filter', file=afp)
    print(f'Request acceptance ratio: {accept_ratio*100:.2f}%', file=afp)
    print(f'Average episode reward: {np.mean(reward_hist):.3f}', file=afp)
clear_output(wait=True)

#### Plot the CDF of accepted SFC requests
accepted_req_id = list(sfc_embed_map.keys())
tot_accept = 0
cdf_accum = []
for i in range(len(success_embed_count)):
    tot_accept += success_embed_count[i]
    cdf_accum.append(tot_accept)

plt.figure(figsize=(10, 7.5))
plt.plot(cdf_accum)
plt.xlabel("Epoch", fontsize=22)
plt.ylabel("Accumulated Num. of Accepted Requests", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_CDF_Accepted_Request_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_CDF_Accepted_Request_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)


#### Avg. path length, i.e,  Average number of links per accepted request, i.e, bandwidth usage
tot_links = 0
for req_item in registered_req['registered_request']:
    # tot_links += len(req_item['sfc_map']) - 1 # old calculation
    # Exclude the self-connected link, i.e., i-i link
    n_diff_nodes, _, _ = data_utils_CONUS_02.find_runs(req_item['sfc_map'])
    tot_links += len(n_diff_nodes) - 1

avg_links = tot_links / tot_accept
print(f'Average number of links per accepted request: {avg_links:.3f}')
with open(accepted_req_file, 'a') as afp:
    print(
        f'Average number of links per accepted request: {avg_links:.3f}', file=afp)

#### E2E delay ratio for accepted requests
print(f'Average end-to-end delay ratio of accepted requests: {np.mean(delay_rate_hist)*100:.3f}% with variance of {np.var(delay_rate_hist)*100:.4f}')
with open(accepted_req_file, 'a') as afp:
    print(f'Average end-to-end delay ratio of accepted requests: {np.mean(delay_rate_hist)*100:.3f}% with variance of {np.var(delay_rate_hist)*100:.4f}', file=afp)



# TODO: REVISE the resources usages as function of Time_slot, NOT Epoch
#### CPU usage rate over epochs
plt.figure(figsize=(10, 7.5))
plt.plot(cpu_usage, color='tomato')
plt.xlabel("Time slots", fontsize=22)
plt.ylabel("CPU usage rate", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_CPU_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_CPU_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)

#### RAM usage rate over epochs
plt.figure(figsize=(10, 7.5))
plt.plot(ram_usage, color='tab:green')
plt.xlabel("Time slots", fontsize=22)
plt.ylabel("RAM usage rate", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_RAM_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_RAM_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)

#### STORAGE usage rate over epochs
plt.figure(figsize=(10, 7.5))
plt.plot(sto_usage, color='tab:blue')
plt.xlabel("Time slots", fontsize=22)
plt.ylabel("Storage usage rate", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_STORAGE_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_STORAGE_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)

#### BW usage rate over epochs
plt.figure(figsize=(10, 7.5))
plt.plot(bw_usage, color='tab:blue')
plt.xlabel("Time slots", fontsize=22)
plt.ylabel("Bandwidth usage rate", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_BW_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_BW_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)


### Plot CPU, RAM, STO usage rates in one Figure
plt.figure(figsize=(10, 7.5))
plt.plot(cpu_usage, color='red')
plt.plot(ram_usage, color='magenta')
plt.plot(sto_usage, color='green')
plt.xlabel("Time slots", fontsize=22)
plt.ylabel("Resource usage rates", fontsize=22)
plt.grid()
plt.legend(('CPU', 'RAM', 'STO'), fontsize=20)
if IS_BINARY_STATE:
    fig_name = "Test_CPU_RAM_STO_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_CPU_RAM_STO_Usage_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)

# Export CPU, RAM, STORAGE, and BW usages to files
cpu_usage_log = os.path.join(TEST_DIR, 'cpu_usage.log')
with open(cpu_usage_log, 'a') as fp:
    [print(cpu_usage[i], file=fp) for i in range(len(cpu_usage))]

ram_usage_log = os.path.join(TEST_DIR, 'ram_usage.log')
with open(ram_usage_log, 'a') as fp:
    [print(ram_usage[i], file=fp) for i in range(len(ram_usage))]

sto_usage_log = os.path.join(TEST_DIR, 'sto_usage.log')
with open(sto_usage_log, 'a') as fp:
    [print(sto_usage[i], file=fp) for i in range(len(sto_usage))]

bw_usage_log = os.path.join(TEST_DIR, 'bw_usage.log')
with open(bw_usage_log, 'a') as fp:
    [print(bw_usage[i], file=fp) for i in range(len(bw_usage))]

#### Calculate the throughput rate & req_accepted_rate per time slot
print("Calculate the throughput rate & req_accepted_rate per time slot")
perfect_tp, real_tp = data_utils_CONUS_02.calc_throughput(adm_hist, req_list, 
                                                 all_arrive_req, serving_req, 
                                                 start_time_slot, 
                                                 edf.sfc_specs, edf.vnf_specs)

#### Plot throughput rate over offered load
plt.figure(figsize=(10, 7.5))
plt.plot(perfect_tp, color='tab:blue')
plt.plot(real_tp, color='tab:orange')
plt.xlabel("Time slot", fontsize=22)
plt.ylabel("Throughput [bw unit]", fontsize=22)
plt.grid()
plt.legend(('Perfect Throughput (Offered load)', 'Real Throughput'), fontsize=20)
if IS_BINARY_STATE:
    fig_name = "Test_Throughput_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_Throughput_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)
mean_real_tp = np.mean(real_tp)
print(f'Avg. bandwidth throughput [bw unit]: {mean_real_tp:.3f}')
with open(accepted_req_file, 'a') as afp:
    print(f'Avg. bandwidth throughput [bw unit]: {mean_real_tp:.3f}', file=afp)

#### Plot % gap between perfect_tp and real_tp
plt.figure(figsize=(10, 7.5))
tp_gap = (np.array(perfect_tp) - np.array(real_tp))/np.array(perfect_tp) * 100
plt.plot(tp_gap)
plt.xlabel("Time slot", fontsize=22)
plt.ylabel("Throughput gap [%]", fontsize=22)
plt.grid()
plt.ylim((0, 10))
if IS_BINARY_STATE:
    fig_name = "Test_Throughput_Gap_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_Throughput_Gap_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)
mean_tp_gap = np.mean(tp_gap)
var_tp_gap = np.var(tp_gap)
print(f'Avg. throughput gap: {mean_tp_gap:.3f}% with variance of {var_tp_gap:.3f}%')
with open(accepted_req_file, 'a') as afp:
    print(f'Avg. throughput gap: {mean_tp_gap:.3f}% with variance of {var_tp_gap:.3f}', file=afp)
    
  
# Export perfect_tp and real_tp into files
perfect_tp_file = os.path.join(TEST_DIR, 'perfect_throughput_DQN.log')
with open(perfect_tp_file, 'w') as fp:
    [print(perfect_tp[i], file=fp) for i in range(len(perfect_tp))]

real_tp_file = os.path.join(TEST_DIR, 'real_throughput_DQN.log')
with open(real_tp_file, 'w') as fp:
    [print(real_tp[i], file=fp) for i in range(len(real_tp))]

# Export time_slot, offered_load, real_tp, CPU_usage_rate, RAM_usage_rate, STO_usage_rate, BW_usage_rate into one log file
# TODO: add CPU, RAM, STO, and BW usage rate after revising the data_utils_02.calc_throughput() function
output_file = os.path.join(TEST_DIR, 'DQN_sim_results.log')
with open(output_file, 'w') as fp:
        # print('Time_slots, Offered_load[bw unit], Real_throughput[bw unit], CPU_usage_rate[%], RAM_usage_rate[%], STO_usage_rate[%], BW_usage_rate[%]', file=fp)
        print('Time_slots, Offered_load[bw unit], Real_throughput[bw unit]', file=fp)
n_time_slots = len(perfect_tp)
with open(output_file, 'a') as fp:
    for i in range(n_time_slots):
        # print(f'{i} {perfect_tp[i]} {real_tp[i]} {cpu_usage[i]} {ram_usage[i]} {sto_usage[i]} {bw_usage[i]}', file=fp)
        print(f'{i} {perfect_tp[i]} {real_tp[i]}', file=fp)
    
    
# Plot request accepted ratio per time slot
adm_rate_per_slot = data_utils_CONUS_02.accept_rate_per_slot(adm_hist)
plt.figure(figsize=(10, 7.5))
plt.plot(adm_rate_per_slot*100)
plt.xlabel("Time slot", fontsize=22)
plt.ylabel("Acceptance rate [%]", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Test_Acceptance_Rate_per_Time_Slot_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Test_FRAC_Acceptance_Rate_per_Time_Slot" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TEST_DIR, fig_name)
plt.savefig(fig_name)


# Export adm_hist to json file
adm_hist_file = os.path.join(TEST_DIR, 'adm_hist.json')
with open(adm_hist_file, 'w') as ahf:
    json.dump(adm_hist, ahf, indent=2)

# Export list of accepted requests into file
with open(accepted_req_file, 'a') as afp:
    [print(accepted_req_id[i], file=afp) for i in range(len(accepted_req_id))]
    


