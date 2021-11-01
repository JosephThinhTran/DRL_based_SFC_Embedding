# -*- coding: utf-8 -*-
"""
Created on Wed July 14 22:37:10 2021
@author: Onlyrich-Ryzen

Implement Q-learning main loop
    + Service Function Chain Embedding problem
    + EDF network environment

- Changelog from version 01.06:
    + The same as in Version 01.06
    + Used for training with the dataset uploaded onto Github on 14-July-2021 (SOF_Data_Sets-v09)
    + Use data_utils_02.py for reading dataset
    + Use edf_env_v2_02.py due to the new dataset's format
"""

import numpy as np
import torch
import random
from copy import copy
from IPython.display import clear_output
from matplotlib import pylab as plt
from q_models_gpu import DQAgent # Using GPU for training
from edf_env_v2_02 import EdfEnv
import data_utils_02
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
DATA_FOLDER = "SOF_Data_Sets-v09\complete_data_sets"# Training Dataset
req_json_file = os.path.join(DATA_FOLDER, "reordered_traffic_500000_slots_1_con.tra")  # service request traffic
# Create EDF environment
edf = EdfEnv(data_path=DATA_FOLDER, 
             net_file='ibm_500000_slots_1_con.net', 
             sfc_file='sfc_file.sfc', 
             resource_scaler=RESOURCE_SCALER)

# Using Binary state or Fractional state renderer
if IS_BINARY_STATE:
    edf.binary_state_dim()
else:
    edf.fractional_state_dim()
    
#### Obtain service requests
n_req, req_list = data_utils_02.retrieve_sfc_req_from_json(req_json_file)

#### Build the Q-learning RL agent
# Neural net layers' neurons
INPUT_DIMS = edf.n_state_dims
L2 = 256
L3 = 256
FC_HID_LAYERS = np.array([L2, L3])
OUTPUT_DIMS = 2 * edf.n_nodes

# Learning parameters
GAMMA = 0.785
LEARNING_RATE = 1e-4
DECAY_RATE = 0.00008
EPSILON = 0.4 if not IS_CONTINUE_TRAINING else 0.3
MIN_EPSILON = 0.02
EPSILON_UPDATE_PERIOD = 10

# Training parameters
# N_EPOCHS = int(n_req*0.4)
N_EPOCHS = 125_000 if not IS_CONTINUE_TRAINING else 100_000
BUFFER_SIZE = 500_000
BATCH_SIZE = 200
SYNC_FREQ = 100

#### Create Q agent
MODEL_DIR = 'DQN_SFCE_models'
now = datetime.now()
TRAIN_DIR = 'train_results__' + now.strftime("%Y-%B-%d__%H-%M")
if IS_BINARY_STATE:
    TRAIN_DIR += "_BINARY_state"
else:
    TRAIN_DIR += "_FRACTIONAL_state"
    
TRAIN_DIR = os.path.join(MODEL_DIR, TRAIN_DIR)
Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)

#### Export the parameter setting into a file
param_setting_file = os.path.join(TRAIN_DIR, "parameter_settings.txt")
with open(param_setting_file, 'a') as fp:
    print("System parameters", file=fp)
    print(f"training directory: {TRAIN_DIR} \n", file=fp)
    
    print(f"IS_BINARY_STATE = {IS_BINARY_STATE}", file=fp)
    print(f"IS_NEIGHBOR_MASK = {IS_NEIGHBOR_MASK}", file=fp)
    print(f"IS_CONTINUE_TRAINING = {IS_CONTINUE_TRAINING}", file=fp)
    print(f'STATE_NOISE_SCALE = {STATE_NOISE_SCALE}', file=fp)
    
    print("Neural network settings", file=fp)
    print(f"INPUT_DIMS = {INPUT_DIMS}", file=fp)
    print(f"L2 = {L2}, L3 = {L3}", file=fp)
    print(f"FC_HID_LAYERS  = {FC_HID_LAYERS}", file=fp)
    print(f"OUTPUT_DIMS = {OUTPUT_DIMS} \n", file=fp)
    
    print("Learning parameters", file=fp)
    print(f"GAMMA = {GAMMA}", file=fp)
    print(f"LEARNING_RATE = {LEARNING_RATE}", file=fp)
    print(f"DECAY_RATE = {DECAY_RATE}", file=fp)
    print(f"EPSILON = {EPSILON}", file=fp)
    print(f"MIN_EPSILON = {MIN_EPSILON}", file=fp)
    print(f"EPSILON_UPDATE_PERIOD = {EPSILON_UPDATE_PERIOD} \n", file=fp)
    
    print("Training parameters", file=fp)
    print(f"N_EPOCHS = {N_EPOCHS}", file=fp)
    print(f"BUFFER_SIZE = {BUFFER_SIZE}", file=fp)
    print(f"BATCH_SIZE = {BATCH_SIZE}", file=fp)
    print(f"SYNC_FREQ = {SYNC_FREQ}", file=fp)
       
print("Export system parameter setting into text files ... DONE!")


train_log = os.path.join(TRAIN_DIR, 'train.log')
q_agent = DQAgent(model_dir=MODEL_DIR, 
                  fc_hid_layers=FC_HID_LAYERS, 
                  input_dims=INPUT_DIMS, 
                  output_dims=OUTPUT_DIMS, 
                  gamma=GAMMA, 
                  learning_rate=LEARNING_RATE,
                  epsilon=EPSILON, 
                  buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, 
                  using_gpu=True)

#### Load the trained network params
if IS_CONTINUE_TRAINING:
    q_agent.load_checkpoint(is_train=True)
    print(f"Continue training the neural network from the current checkpoint with EPSILON = {EPSILON}")
else:
    print("Training the neural network from scratch!")
    print(f'Starting with EPSILON = {EPSILON}')

# input('Press enter to continue!')
#### Loss values
losses = []
total_iters = 0
# Max number of routing steps (iterations) per episode
max_moves = 50
reward_hist = [] # list of reward per episode

registered_req = [] # list of accepted SFC requests
active_req = [] # list of SFC requests still in the EDF
expired_req = [] # list of expired SFC requests
serving_req = [] # list of accepted requests
sfc_embed_map = {}
vnf_embed_map = {}

### Calculation of resource usages
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

cur_time_slot = -1 # current time slot indicator
# dict of accepted/rejected request per time slot
adm_hist = {}# key=time_slot, val=adm_per_slot
adm_per_slot = {}# key=req_id, val=1/0 (accepted/rejected)
rejected_req = {'requests':[]} # list-dict of rejected SFC requests

#### Main training loop
start_id = 0
end_id = N_EPOCHS
# req_id_list = list(range(start_id, end_id))
req_id_list = []
for key in req_list.keys():
    req_id_list.append(key)

#starting time slot id
start_time_slot = req_list[start_id].arrival_time
print(f'Start time slot: {start_time_slot}')

for epoch in range(N_EPOCHS):
    # Get the i-th request
    idx = req_id_list[epoch]
    print(f'Request ID: {idx}')
    req = req_list[idx]
    all_arrive_req.append(req.id)# use for calculating perfect_throughput
    print(f'Req_Id={req.id},   source={req.source}   destination={req.destination}   {req.sfc_id}:[{edf.sfc_specs[req.sfc_id]}]   bw={req.bw}   delay_req={req.delay_req}')
    with open(train_log, 'a') as log_fp:
        print(f'Req_Id={req.id},   source={req.source}   destination={req.destination}   {req.sfc_id}:[{edf.sfc_specs[req.sfc_id]}]   bw={req.bw}   delay_req={req.delay_req}', file=log_fp)
    
    # check new time slot
    if req.arrival_time == cur_time_slot + 1 + start_time_slot:
        cur_time_slot += 1
        adm_hist[cur_time_slot] = {} # create a dict for the new time slot
        
        # Calculate resource usage rates per time slot
        # starting at the end of the first time slot
        if cur_time_slot >= 1:
            # CPU, RAM, STO usage rates per time slot
            remain_cpu, remain_ram, remain_sto = edf.resource_mat.sum(axis=0)
            cpu_usage.append((tot_cpu - remain_cpu) / tot_cpu)
            ram_usage.append((tot_ram - remain_ram) / tot_ram)
            sto_usage.append((tot_sto - remain_sto) / tot_sto)
            # BW usage rate per time slot
            remain_real_bw = copy(edf.bw_mat)
            np.fill_diagonal(remain_real_bw, 0)# do not take self-connected link bw into account
            tot_remain_bw = remain_real_bw.sum()
            bw_usage.append((tot_bw - tot_remain_bw) / tot_bw)
        
        # Get back resource from the EXPIRED accepted requests
        if cur_time_slot >= 1:
            edf.sojourn_monitor(active_req, req_list, 
                                sfc_embed_map, vnf_embed_map, 
                                cur_time_slot + start_time_slot, expired_req)
            
        # Update e-greedy parameter
        if cur_time_slot >= 1 and cur_time_slot % EPSILON_UPDATE_PERIOD == 0:
            # EPSILON = max(DECAY_RATE*EPSILON, 0.01)
            EPSILON = max(EPSILON - DECAY_RATE, MIN_EPSILON)
        
            
    # Get the required VNF list
    vnf_list = copy(edf.sfc_specs[req.sfc_id])
    vnf_item = 0
    hol_vnf_name = vnf_list[vnf_item]
    cur_node_id = req.source
    
    # Create list of nodes embedding the current SFC request
    sfc_embed_nodes = [cur_node_id]# SFC node mapping
    vnf_embed_nodes = {} # VNF compute node
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
    state1 = state1.reshape(1,INPUT_DIMS) + np.random.rand(1,INPUT_DIMS)/STATE_NOISE_SCALE
    state1 = torch.from_numpy(state1).float()
    
    # Perform SFC embedding & Routing
    stop_flag = 0
    mov = 0
    this_episode_rwd = 0
    
    while(stop_flag < 1): 
        total_iters += 1
        mov += 1
        
        #### Epsilon-greedy action sampling
        if IS_NEIGHBOR_MASK:
            neighbor_filter = np.tile(edf.adj_mat[cur_node_id], 2)
        if random.random() < EPSILON:
            if IS_NEIGHBOR_MASK:
                neighbor_idx = np.where(neighbor_filter>0)[0]
                if len(neighbor_idx) >0:
                    action_raw = np.random.choice(neighbor_idx)
                else:
                    action_raw = np.random.randint(0,OUTPUT_DIMS)
            else:
                action_raw = np.random.randint(0,OUTPUT_DIMS)
        else:
            if IS_NEIGHBOR_MASK:
                neighbor_filter[neighbor_filter<1] = -1e6
                action_raw = q_agent.act(state1, neighbor_filter)
            else:
                action_raw = q_agent.act(state1)
        # convert to the real action space
        action = edf.to_real_action(action_raw)

        # Update the resources edf.temp_resource_mat, edf.temp_bw_mat
        resource_failed, link_failed = edf.make_move(action, req, hol_vnf_name, cur_node_id)
        
        # Update delay budget
        residual_delay, added_delay, proc_latency, new_link_latency = \
            edf.cal_residual_delay(residual_delay, action, req, hol_vnf_name, cur_node_id)
        
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
        sfc_embed_nodes.append(cur_node_id) # SFC node mapping
        
        # Check SFC embedding and routing accomplishement
        if (cur_node_id == req.destination) and (hol_vnf_name == 'VNF_Done'):
            done_embed = 1
        else: 
            done_embed = 0
        
        #### Obtain the next state from the EDF environment
        if IS_BINARY_STATE:
            state2 = edf.render_state_binary(req, hol_vnf_name, cur_node_id)
        else:
            state2 = edf.render_state_frac(req, hol_vnf_name, cur_node_id)
        state2 = state2.reshape(1,INPUT_DIMS) + np.random.rand(1,INPUT_DIMS)/STATE_NOISE_SCALE
        state2 = torch.from_numpy(state2).float()
        
        #### Calculate step reward
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
        
        #### Add the experience sample to replay buffer
        exp = (state1, action_raw, reward, state2, success_embed) 
        q_agent.replay_buffer.append(exp)
        state1 = state2
        
        #### Training - Update neural network params
        if len(q_agent.replay_buffer) > BATCH_SIZE:
            loss = q_agent.batch_update()
            losses.append(loss)
            print(f'Time_{cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req.sfc_id}-{prev_hol_vnf_name}   Action=(is_embed={action.is_embed}; next_node={action.next_node})   Residual_delay={residual_delay:.4f}   Reward={reward:.3f}   Success_embed={success_embed}   Training_loss={loss:.3f}   EPSILON={EPSILON:.3f}')
            if resource_failed != 'None' : print(resource_failed)
            if link_failed != 'None' : print(link_failed)
            with open(train_log, 'a') as log_fp:
                print(f'Time_{cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req.sfc_id}-{prev_hol_vnf_name}   Action=(is_embed={action.is_embed}; next_node={action.next_node})   Residual_delay={residual_delay:.4f}   Reward={reward:.3f}   Success_embed={success_embed}   Training_loss={loss:.3f}   EPSILON={EPSILON:.3f}', file=log_fp)
                if resource_failed != 'None':
                    print(resource_failed, file=log_fp)
                if link_failed != 'None':
                    print(link_failed, file=log_fp)
            clear_output(wait=True)
        
        # Sync the target_Q_net with the q_net
        # Save checkpoint
        if total_iters % SYNC_FREQ == 0:
            q_agent.hard_sync()
            q_agent.save_checkpoint()
        
        # # Update the e-greedy parameter
        # if total_iters % 50 == 0:
        #     # EPSILON = max(DECAY_RATE*EPSILON, 0.01)
        #     EPSILON = max(EPSILON - 0.001, 0.02)
                    
        # Stopping criteria
        if success_embed > 0 or fail_embed > 0 or mov > max_moves:
            stop_flag = 1
            mov = 0
        
        # Register the NEWLY successul embedded request to the serving list
        if success_embed > 0:
            registered_req.append(req)
            sfc_embed_map[req.id] = sfc_embed_nodes
            vnf_embed_map[req.id] = vnf_embed_nodes
            active_req.append(req.id)
            serving_req.append(req.id)
            
        # Update the resource consumption of this episode to EDF resource availability
        # if the SFC embedding is successfull
        if success_embed > 0:
            # Update resource from current_added
            edf.resource_mat = copy(edf.temp_resource_mat)
            edf.bw_mat = copy(edf.temp_bw_mat)
            
        # Keep track of accepted/rejected request per time slot
        if stop_flag == 1:
            if success_embed > 0:
                adm_hist[cur_time_slot].update({req.id:1})
            else:
                adm_hist[cur_time_slot].update({req.id:0})
                rejected_req['requests'].append(req._asdict())
                
    # End of inner WHILE loop
    
    # Add episode_reward to the list
    reward_hist.append(this_episode_rwd)
    
    # END OF OUTER EPOCH LOOP


print('TRAINING PROCESS DONE!')
#### Show the training loss values
print(f'Average loss = {np.mean(losses):.4f} \t Loss variance = {np.var(losses):.4f}')
loss_val_file = os.path.join(TRAIN_DIR, 'loss_val.log')
with open(loss_val_file, 'w') as lfp:
    [print(losses[i], file=lfp) for i in range(len(losses))]
with open(loss_val_file, 'a') as lfp:
    print(f'Average loss = {np.mean(losses):.4f} \t Loss variance = {np.var(losses):.4f}', file=lfp)
      
# Plot training loss results
moving_avg_loss = data_utils_02.mov_window_avg(losses, 1000)
plt.figure(figsize=(10,7.5))
plt.grid()
plt.plot(losses, color='tab:blue')
plt.plot(moving_avg_loss, color='tab:orange')
plt.xlabel("Steps",fontsize=22)
plt.ylabel("Loss",fontsize=22)
# plt.legend(("Losses", "Moving Average Losses"),fontsize=20)
plt.ylim([0, 0.6])
if IS_BINARY_STATE:
    fig_name = "Train_Loss_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "FRAC_Train_Loss_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)


#### Plot the episode_reward history
reward_hist = np.array(reward_hist)
reward_log_file = os.path.join(TRAIN_DIR, 'episode_rewards.log')
with open(reward_log_file, 'a') as rfp:
    [print(reward_hist[i], file=rfp) for i in range(len(reward_hist))]

moving_avg_reward = data_utils_02.mov_window_avg(reward_hist, 1000)
plt.figure(figsize=(10,7.5))
plt.grid()
# plt.plot(reward_hist, color='tab:blue')
plt.plot(moving_avg_reward, color='tab:orange')
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Reward",fontsize=22)
# plt.legend(("Rewards", "Moving Average Rewards"),fontsize=20)
if IS_BINARY_STATE:
    fig_name = "Train_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "FRAC_Train_Reward_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)


#### Request acceptance ratio
accept_ratio = len(sfc_embed_map)/N_EPOCHS
print(f'Total requests arrived at the network: {N_EPOCHS}')
print(f'Request acceptance ratio: {accept_ratio*100:.2f}%')
with open(train_log, 'a') as tf:
    print('################# CONCLUSION ############################', file=tf)
    print(f'Total requests arrived at the EDF: {N_EPOCHS}', file=tf)
    print(f'Request acceptance ratio: {accept_ratio*100:.2f}%', file=tf)


#### Plot the CDF of accepted SFC requests
accepted_req_id = list(sfc_embed_map.keys())
tot_accept = 0
cdf_accum = []
for req_id in range(N_EPOCHS):
    if req_id in accepted_req_id:
        tot_accept += 1
    cdf_accum.append(tot_accept)
    
plt.figure(figsize=(10,7.5))
plt.plot(cdf_accum)
plt.xlabel("Epoch",fontsize=22)
plt.ylabel("Num. of Requests",fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "CDF_Accepted_Request_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "FRAC_CDF_Accepted_Request_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)

accepted_req_file = os.path.join(TRAIN_DIR, 'accepted_req.log')
with open(accepted_req_file, 'w') as afp:
    [print(accepted_req_id[i], file=afp) for i in range(len(accepted_req_id))]
    
    
#### Plot the reward and accumulated accepted request in a same figure
# reward curve
plt.figure(figsize=(10, 7.5))
fig, ax = plt.subplots()
ax.plot(moving_avg_reward, color='red', linestyle='-')
ax.set_xlabel("Epochs", fontsize=22)
ax.set_ylabel("Reward", fontsize=22, color='red')
# ax.grid()
# ax.legend(("Reward", "Accepted Requests"))

# accum accepted request curve
ax2 = ax.twinx()
ax2.plot(cdf_accum, color='blue', linestyle = '-.')
ax2.set_ylabel("Number of Requests", fontsize=22, color='blue')
# ax2.legend(("Accum. No. of Accepted Requests"))
# fig.legend(("1", "2"))

if IS_BINARY_STATE:
    fig_name = "Reward_and_CDF_Accept_Req_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "FRAC_Reward_and_CDF_Accept_Req_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
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
fig_name = os.path.join(TRAIN_DIR, fig_name)
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
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)
    
#### Plot the throughput rate & throughput gap
print("Calculate the throughput rate & req_accepted_rate per time slot")
perfect_tp, real_tp = data_utils_02.calc_throughput(adm_hist, req_list, 
                                                 all_arrive_req, serving_req, 
                                                 start_time_slot,
                                                 edf.sfc_specs, edf.vnf_specs)
# Plot throughput rate over offered load
plt.figure(figsize=(10, 7.5))
plt.plot(perfect_tp, color='tab:blue')
plt.plot(real_tp, color='tab:orange')
plt.xlabel("Time slot", fontsize=22)
plt.ylabel("Throughput [bw unit]", fontsize=22)
plt.grid()
plt.legend(('Perfect Throughput (Offered load)', 'Real Throughput'), fontsize=20)
if IS_BINARY_STATE:
    fig_name = "Train_Throughput_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Train_FRAC_Throughput_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)
mean_real_tp = np.mean(real_tp)
print(f'Avg. bandwidth throughput [bw unit]: {mean_real_tp:.3f}')
with open(accepted_req_file, 'a') as afp:
    print(f'Avg. bandwidth throughput [bw unit]: {mean_real_tp:.3f}', file=afp)

# Plot % gap between perfect_tp and real_tp
plt.figure(figsize=(10, 7.5))
tp_gap = (np.array(perfect_tp) - np.array(real_tp))/np.array(perfect_tp) * 100
plt.plot(tp_gap)
plt.xlabel("Time slot", fontsize=22)
plt.ylabel("Throughput gap [%]", fontsize=22)
plt.grid()
if IS_BINARY_STATE:
    fig_name = "Train_Throughput_Gap_" + now.strftime("%Y-%B-%d__%H-%M")
else:
    fig_name = "Train_FRAC_Throughput_Gap_" + now.strftime("%Y-%B-%d__%H-%M")
fig_name = os.path.join(TRAIN_DIR, fig_name)
plt.savefig(fig_name)
mean_tp_rate = np.mean(tp_gap)
print(f'Avg. throughput gap [%]: {mean_tp_rate:.3f}')
with open(accepted_req_file, 'a') as afp:
    print(f'Avg. throughput gap [%]: {mean_tp_rate:.3f}', file=afp)

# # Plot request accepted ratio per time slot
# adm_rate_per_slot = data_utils_02.accept_rate_per_slot(adm_hist)
# plt.figure(figsize=(10, 7.5))
# plt.plot(data_utils_02.mov_window_avg(adm_rate_per_slot*100, 1000))
# plt.xlabel("Time slot", fontsize=22)
# plt.ylabel("Acceptance rate [%]", fontsize=22)
# plt.grid()
# if IS_BINARY_STATE:
#     fig_name = "Train_Acceptance_Rate_per_Time_Slot_" + now.strftime("%Y-%B-%d__%H-%M")
# else:
#     fig_name = "Train_FRAC_Acceptance_Rate_per_Time_Slot" + now.strftime("%Y-%B-%d__%H-%M")
# fig_name = os.path.join(TRAIN_DIR, fig_name)
# plt.savefig(fig_name)

# Export adm_hist to json file
adm_hist_file = os.path.join(TRAIN_DIR, 'adm_hist.json')
with open(adm_hist_file, 'w') as ahf:
    json.dump(adm_hist, ahf, indent=2)

# Export list of accepted requests into file
with open(accepted_req_file, 'a') as afp:
    [print(accepted_req_id[i], file=afp) for i in range(len(accepted_req_id))]


# Export the reject_list_dict to the json file "rejected_traffic_1_con_DATE.tra"    
reject_req_file = "rejected_traffic_1_con_" + now.strftime("%Y_%B_%d__%H_%M") + ".tra"
reject_req_file = os.path.join(DATA_FOLDER, reject_req_file)
with open(reject_req_file, 'w') as fp:
    json.dump(rejected_req, fp, indent=2)
