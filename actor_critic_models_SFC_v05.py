# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:02:30 2021

@author: Onlyrich-Ryzen

Implement Actor-Critic Models

Changes from v04
    + Integrate LSTM/GRU cell into the shared_AC_net model

"""

from networkx.algorithms.triads import is_triad
from numpy.lib import real
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
import os
from collections import deque
from edf_env_A3C_v2_02 import EdfEnv
from data_utils_A3C_02 import mov_window_avg, delete_data, calc_throughput, numpy_encoder, find_runs
from datetime import datetime
from pathlib import Path
import json
from copy import copy
from IPython.display import clear_output
from collections import namedtuple
import gc #garbage collector
from IPython.display import clear_output
from onpolicy_memory import OnPolicyReplay, OnPolicyBatchReplay


# Node_Info = namedtuple('Node_Info', ['node_id', 'cpu_cap', 'cpu_free', 'ram_cap', 'ram_free',
#                                          'sto_cap', 'sto_free', 'employable_vnf'])

# Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
#                                         'bw', 'delay_req',
#                                         'arrival_time', 'end_time', 'co_id'])



# =============================================================================
''' Shared Adam Optimizer'''
# =============================================================================
class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, 
                                         eps=eps, weight_decay=weight_decay)

        # Sharing the memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""'"""""""""""""""""""              

# =============================================================================
''' Shared RMSProp Optimizer'''
# =============================================================================
class SharedRMSProp(optim.RMSprop):
    def __init__(self, params, lr=1e-2, eps=1e-5, weight_decay=0, 
                    alpha=0.99, momentum=0, centered=False):
        super(SharedRMSProp, self).__init__(params, lr=lr, eps=eps, 
                                            weight_decay=weight_decay, alpha=alpha,
                                            momentum=momentum, centered=centered)

        # Sharing the memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['grad_avg'] = torch.zeros_like(p.data)

                state['square_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()
                state['grad_avg'].share_memory_()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""'"""""""""""""""""""   
# =============================================================================
# Actor-Critic Model
# =============================================================================
class ACNet(torch.nn.Module):
    def __init__(self, name, model_dir, fc_hid_layers, input_dims, actor_dims, net_arch):
        '''
        Initialize the Actor-Critic model
        2-head network model : the Actor and Critic share a common part of the neural network

        Parameters
        ----------
        name : String
            NAME OF THE MODEL
        model_dir : String
            DIRECTORY STORING THE MODEL
        fc_hid_layers: np.array int32
            # OF NEURONS PER FULLY-CONNECTED LAYER
            # DEFAUL : 3 LAYERS (2 common fc layers for the actor and critic, 1 more fc layer before output the critic head)
        input_dims : int32
            INPUT DIMENSION (SYSTEM STATE VECTOR DIMENSION)
        actor_dims : int32
            OUTPUT DIMENSION OF THE ACTOR
        
        Returns
        -------
        None.

        '''
        super(ACNet, self).__init__()

        self.NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
        
        # load params to the class
        self.name = name
        self.model_dir = model_dir
        abs_path = os.getcwd()
        full_path = os.path.join(abs_path, self.model_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        self.checkpoint_file = os.path.join(full_path, self.name + '.pt')
        
        
        # Neural network dimensions
        self.net_arch = net_arch
        self.input_dims = input_dims
        self.fc_hid_layers = fc_hid_layers
        self.actor_dims = actor_dims
        self.critic_dims = 1 # just only the critic value
        
        # Configure Neural network structure
        if self.net_arch == self.NET_ARCH['shared_net']:
            # shared_net
            self.fc1 = nn.Linear(self.input_dims, self.fc_hid_layers['common_1'])
            self.fc2 = nn.Linear(self.fc_hid_layers['common_1'], self.fc_hid_layers['common_2'])
            # Actor head
            self.actor_lin1 = nn.Linear(self.fc_hid_layers['common_2'], self.actor_dims)
            # Critic head
            self.fc3 = nn.Linear(self.fc_hid_layers['common_2'], self.fc_hid_layers['critic_1'])#one more fc layer before output the critic head
            self.critic_lin1 = nn.Linear(self.fc_hid_layers['critic_1'], self.critic_dims)# critic head
        
        elif self.net_arch == self.NET_ARCH['shared_net_w_RNN']:
            # shared_net
            self.fc1 = nn.Linear(self.input_dims, self.fc_hid_layers['common_1'])
            self.rnn = nn.GRUCell(self.fc_hid_layers['common_1'], self.fc_hid_layers['rnn_1'])
            # self.fc2 = nn.Linear(self.fc_hid_layers[0], self.fc_hid_layers[1])
            # self.rnn = nn.GRUCell(self.fc_hid_layers[1], self.fc_hid_layers[2])
            # Actor head
            self.actor_lin1 = nn.Linear(self.fc_hid_layers['rnn_1'], self.actor_dims)
            # Critic head
            self.critic_lin1 = nn.Linear(self.fc_hid_layers['rnn_1'], self.critic_dims)

        elif self.net_arch == self.NET_ARCH['separated_net']:
            # Actor network
            self.actor_fc1 = nn.Linear(self.input_dims, self.fc_hid_layers['l1'])
            self.actor_fc2 = nn.Linear(self.fc_hid_layers['l1'], self.fc_hid_layers['l2'])
            # Critic network
            self.critic_fc1 = nn.Linear(self.input_dims, self.fc_hid_layers['l1'])
            self.critic_fc2 = nn.Linear(self.fc_hid_layers['l1'], self.fc_hid_layers['l2'])
            self.critic_v = nn.Linear(self.fc_hid_layers['l2'], self.critic_dims)

        
    def forward(self, x, hx=None):
        if self.net_arch == self.NET_ARCH['shared_net']:
            # x = F.normalize(x, dim=0) #normalize the input data
            y = F.relu(self.fc1(x))
            y = F.relu(self.fc2(y))
            
            # Actor head
            actor_vec = self.actor_lin1(y) # logit vector
            # actor_vec = F.softmax(self.actor_lin1(y), dim=1) # prob. dist. vector
            
            # Critic head
            # z = F.relu(self.fc3(y.detach())) #detach y before feeding to z; do not backpropagate loss from critic head to the fc1 and fc2 layers
            z = F.relu(self.fc3(y))
            # critic_val = torch.tanh(self.critic_lin1(z)) #
            critic_val = self.critic_lin1(z)
            return actor_vec, critic_val

        elif self.net_arch == self.NET_ARCH['shared_net_w_RNN']:
            y = F.relu(self.fc1(x))
            # y = F.relu(self.fc2(y))
            hx = self.rnn(y, (hx))
            # Actor head
            actor_vec = F.softmax(self.actor_lin1(hx), dim=1)
            # Critic head
            critic_val = self.critic_lin1(hx)
            return actor_vec, critic_val, hx

        elif self.net_arch == self.NET_ARCH['separated_net']:
            # Actor
            y = F.relu(self.actor_fc1(x))
            y = F.relu(self.actor_fc2(y))
            actor_vec = F.softmax(y, dim=1)

            # Critic
            z = F.relu(self.critic_fc1(x))
            z = F.relu(self.critic_fc2(z))
            critic_val = self.critic_v(z)
            return actor_vec, critic_val
        
    
    def save_params(self):
        ''' Save network parameters to file'''
        print("Saving " + self.name + "'s network parameters to file")
        torch.save(self.state_dict(), self.checkpoint_file)
            
    def load_params(self):
        ''' Load network parameters from file'''
        print("Loading " + self.name + "'s network parameters from file")
        # torch.load(self.state_dict(), self.checkpoint_file)
        self.load_state_dict(torch.load(self.checkpoint_file))
        
    
        
# =============================================================================
# A3C Worker
# =============================================================================
class A3CWorker(mp.Process):
    ''' Asynchronous Advantage Actor Critic Algorithm '''
    def __init__(self, worker_id, global_model, optimizer, params, traffic_file, counter, 
                net_arch, is_train=True):
        super(A3CWorker, self).__init__()

        self.ADV_STYLE = {'n_step_return': 1, 'gae': 2}
        
        self.worker_id = worker_id
        self.global_model = global_model
        self.net_arch = net_arch
        self.worker_model = ACNet(name='worker_' + str(worker_id), 
                                  model_dir=params['model_dir'], 
                                  fc_hid_layers=params['hidden_layers'], 
                                  input_dims=params['input_dims'], 
                                  actor_dims=params['actor_dims'],
                                  net_arch=self.net_arch)
        if is_train == False:
            self.worker_model.load_state_dict(self.global_model.state_dict())

        self.params = params
        self.g_counter = counter
        self.max_episodes = params['epochs'][self.worker_id]
        
        # self.optimizer = optim.Adam(lr=self.params['learning_rate'], params=global_model.parameters()) # optimizer
        self.optimizer = optimizer
        
        # self.clc = params['critic_factor']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.N_steps = params['N_steps']
        # self.max_moves = params['max_moves']
        
        self.data_folder = params['data_folder']
        self.net_file = params['datasets'][worker_id]['net_topo']
        self.sfc_file = params['sfc_spec']
        
        self.resource_scaler = params['resource_scaler']

        self.betas = params['betas'] #cost factors for the reward function
        self.big_rwd = params['big_rwd']
                
        # Traffic request file
        self.req_json_file = os.path.join(params['data_folder'], 
                                          params['datasets'][worker_id]['traffic'])
        # self.n_req, self.req_list = retrieve_sfc_req_from_json(self.req_json_file)
        self.req_list = traffic_file
        # self.n_req = len(traffic_file)

        self.is_train = is_train

        # Deploy rollout buffer in training mode
        if self.is_train:
            self.train_freq = params['train_freq']
            self.memory = OnPolicyReplay(self.N_steps) # batch of episodes' experiences
        
        # Advantage calculation method
        if self.params['adv_style'] == self.ADV_STYLE['n_step_return']:
            self.calc_advs_v_targets = self.calc_nstep_advs_v_targets
        elif self.params['adv_style'] == self.ADV_STYLE['gae']:
            self.calc_advs_v_targets = self.calc_gae_adv_v_targests
            
        # Create an EDF environment
        self.create_edf_env()
        
        #### Logging
        if self.is_train:
            # self.operation_log = os.path.join(params['train_dir'], "W_" + str(self.worker_id) + "_train.log")
            self.ep_reward_log = os.path.join(params['train_dir'], "W_" + str(self.worker_id) + "_ep_reward.log")
            self.loss_log = os.path.join(params['train_dir'], "W_" + str(self.worker_id) + "_losses.log")
            self.accept_ratio_log = os.path.join(params['train_dir'], 'W_' + str(self.worker_id) + "_accept_ratio.log")
            self.throughput_log = os.path.join(params['train_dir'], 'W_' + str(self.worker_id) + "_throughputs.log")
            self.rsc_usage_log = os.path.join(params['train_dir'], 'W_' + str(self.worker_id) + "_rsc_usage.log")
            self.registerd_req_log = os.path.join(params['train_dir'], 'W_' + str(self.worker_id) + "_registerd_req.log")
        else:
            self.operation_log = os.path.join(params['test_dir'], "W_" + str(self.worker_id) + "_test.log")
            self.ep_reward_log = os.path.join(params['test_dir'], "W_" + str(self.worker_id) + "_ep_reward.log")
            self.accept_ratio_log = os.path.join(params['test_dir'], 'W_' + str(self.worker_id) + "_test_accept_ratio.log")
            self.throughput_log = os.path.join(params['test_dir'], 'W_' + str(self.worker_id) + "_test_throughputs.log")
            self.rsc_usage_log = os.path.join(params['test_dir'], 'W_' + str(self.worker_id) + "_test_rsc_usage.log")
            self.registerd_req_log = os.path.join(params['test_dir'], 'W_' + str(self.worker_id) + "_test_registerd_req.log")
        print(f'Successful Init A3C_Worker {self.worker_id}')
            
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""        
    
    def create_edf_env(self):
        # Create EDF environment
        self.edf = EdfEnv(data_path=self.data_folder,
                          net_file=self.net_file,
                          sfc_file=self.sfc_file,
                          resource_scaler=self.resource_scaler,
                          betas=self.betas, big_rwd=self.big_rwd)
        if self.params['is_binary_state']:
            self.edf.binary_state_dim()
        else:
            self.edf.fractional_state_dim()
            
        #### Loss values
        self.losses = []
        self.total_iters = 0
        # Max number of routing steps (iterations) per episode
        # max_moves = 50
        self.reward_hist = [] # list of reward per episode
        self.delay_rate_hist = [] # list of delay ratio  per accepted request
        
        # self.registered_req = []  # list of accepted SFC requests
        self.registered_req = {'registered_request': []}  # list of accepted SFC requests
        self.active_req = [] # list of SFC requests still in the EDF
        self.expired_req = [] # list of expired SFC requests
        self.serving_req = [] # list of accepted requests

        self.sfc_embed_map = {}
        self.vnf_embed_map = {}
        self.success_embed_count = []  # couting the successful SFC requests
            
        ''' Calculation of resource usage '''
        # CPU, RAM, STORAGE usage_rate over time_slot
        self.tot_cpu, self.tot_ram, self.tot_sto = self.edf.resource_mat.sum(axis=0)#CPU, RAM, STORAGE capacities
        
        # total_bw
        self.real_net_bw = copy(self.edf.bw_mat)
        np.fill_diagonal(self.real_net_bw, 0) # real system_bw does not take self-connected link bw into account
        self.tot_bw = self.real_net_bw.sum()
        self.cpu_usage = []
        self.ram_usage = []
        self.sto_usage = []
        self.bw_usage = []
        
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""    
        
    def run(self):
        '''
        The A3CWorker executes its main task

        Parameters
        ----------
        g_counter : int32
            GLOBAL COUNTER

        Returns
        -------
        None.

        '''
        # Throughput [bw unit] per time slot
        self.all_arrive_req = [] # list of perfect SFC requests
        
        self.cur_time_slot = -1 # current time slot indicator
        # dict of accepted/rejected request per time slot
        self.adm_hist = {}# key=time_slot, val=adm_per_slot
        self.adm_per_slot = {}# key=req_id, val=1/0 (accepted/rejected)
        self.rejected_req = {'requests':[]} # list-dict of rejected SFC requests
        
        # Main training loop
        start_id = 0
        end_id = self.max_episodes
        # req_id_list = list(range(start_id, end_id))
        self.req_id_list = []
        for key in self.req_list.keys():
            self.req_id_list.append(key)
        
        # Starting time slot id
        self.start_time_slot = self.req_list[start_id]['arrival_time']
        #print(f'Start time slot: {self.start_time_slot}')
        
        # Episode_reward list
        self.episode_rewards = []
        self.actor_loss = []
        self.critic_loss = []
        self.losses = []
        self.n_accepted_req = 0

        ''' Run the episodes '''
        # Run an episode and update the neural_network's weights
        for epoch in range(self.max_episodes):
            if self.is_train:
                # update operation log file name
                floor, remainder = np.divmod(epoch, 25_000)
                if remainder == 0:
                    self.operation_log = os.path.join(self.params['train_dir'], "W_" + str(self.worker_id) + 
                                                        "_train_" + str(floor) + ".log")
                # update entropy_factor
                if np.divmod(epoch, self.params['entropy_decay_freq'])[1] == 0:
                    self.params['entropy_factor'] = \
                        max(self.params['entropy_factor'] - self.params['entropy_decay_val'], 
                            self.params['entropy_min'])

            # Play an episode with the env
            print(f"worker_id = {self.worker_id}| local_episode = {epoch}")
            # values, logprobs, entropies, rewards, G, this_episode_rwd = self.run_episode(epoch)
            this_episode_rwd, eplen = self.run_episode(epoch)
            print(f"episode_reward = {this_episode_rwd:.3f}| episode_len = {eplen}\n")
            with open(self.ep_reward_log, 'a') as rwd_fp:
                print(f"{this_episode_rwd:.3f}", file=rwd_fp)
            
            # Worker updates the global model's parameters
            if self.is_train and ((epoch+1) % self.train_freq == 0):
                print(f"UPDATE NEURAL NET's Params at epoch {epoch}")
                actor_loss, critic_loss = self.update_params_batch()
                # Update the global counter
                with self.g_counter.get_lock():
                    self.g_counter.value += 1

            # Garbage collector and clear output
            if np.divmod(epoch, 500)[1] == 0:
                gc.collect()
                clear_output(wait=True)
        # END of FOR loop
    
        with open(self.accept_ratio_log, 'a') as fp:
            print(f"Number of accepted requests = {self.n_accepted_req}", file=fp)
            print(f"Max episodes = {self.max_episodes}", file=fp)
            print(f"Acceptance ratio = {self.n_accepted_req / self.max_episodes * 100}", file=fp)
            print(f"Avg. path len = {self.calc_avg_path_len()}", file=fp)
            avg_delay_rate = np.mean(self.delay_rate_hist) * 100 # percentage
            print(f"Avg E2E delay ratio of accepted requests = {avg_delay_rate:.3f}", file=fp)

        # Calculate real_throughput and offered_load
        self.calc_throughputs()
        self.export_resource_usage()
        self.export_registered_reqs()
        gc.collect()

        ''''''''''''''''''''''''''''''''''''''''''''''''
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""
    
    def run_episode(self, epoch):
        '''
        Play an episode with the given environment

        Returns
        -------
        None.

        '''
        #### Get the i-th request
        idx = self.req_id_list[epoch]
        req = self.req_list[idx]
        self.all_arrive_req.append(req['id'])# use for calculating perfect_throughput
        print(f"Req_Id={req['id']}|   source={req['source']}|   destination={req['destination']}|\
               {req['sfc_id']}:[{self.edf.sfc_specs[req['sfc_id']]}]|\
                      bw={req['bw']}   delay_req={req['delay_req']:.4f}")
        with open(self.operation_log, 'a') as fp:
            print(f"Req_Id={req['id']}|   source={req['source']}|   destination={req['destination']}|\
               {req['sfc_id']}:[{self.edf.sfc_specs[req['sfc_id']]}]|\
                      bw={req['bw']}   delay_req={req['delay_req']:.4f}", file=fp)
        
        #### Check new time slot
        if req['arrival_time'] == self.cur_time_slot + 1 + self.start_time_slot:
            self.cur_time_slot += 1
            self.adm_hist[self.cur_time_slot] = {} # create a dict for the new time slot
            
            # Calculate resource usage rates per time slot
            # starting at the end of the first time slot
            if self.cur_time_slot >= 1:
                # CPU, RAM, STO usage rates per time slot
                remain_cpu, remain_ram, remain_sto = self.edf.resource_mat.sum(axis=0)
                self.cpu_usage.append((self.tot_cpu - remain_cpu) / self.tot_cpu)
                self.ram_usage.append((self.tot_ram - remain_ram) / self.tot_ram)
                self.sto_usage.append((self.tot_sto - remain_sto) / self.tot_sto)
                # BW usage rate per time slot
                remain_real_bw = copy(self.edf.bw_mat)
                np.fill_diagonal(remain_real_bw, 0)# do not take self-connected link bw into account
                tot_remain_bw = remain_real_bw.sum()
                self.bw_usage.append((self.tot_bw - tot_remain_bw) / self.tot_bw)
            
            # Get back resource from the EXPIRED accepted requests
            if self.cur_time_slot >= 1:
                self.edf.sojourn_monitor(self.active_req, self.req_list, 
                                         self.sfc_embed_map, self.vnf_embed_map, 
                                         self.cur_time_slot + self.start_time_slot, self.expired_req)
            
        #### Get the required VNF list
        self.vnf_list = copy(self.edf.sfc_specs[req['sfc_id']])
        self.vnf_item = 0
        hol_vnf_name = self.vnf_list[self.vnf_item]
        cur_node_id = req['source']
        
        # Create list of nodes embedding the current SFC request
        self.sfc_embed_nodes = [cur_node_id]# SFC node mapping
        self.vnf_embed_nodes = {} # VNF compute node
        success_embed = 0
        
        # Get delay constraint
        residual_delay = req['delay_req']
        delay_budget = req['delay_req'] # request's end-to-end delay constraint
        
        # Get a copy of the current resource and bandwidth information
        self.edf.temp_resource_mat = copy(self.edf.resource_mat)
        self.edf.temp_bw_mat = copy(self.edf.bw_mat)
    
        # Prepare the system state
        if self.params['is_binary_state']:
            state1 = self.edf.render_state_binary(req, hol_vnf_name, cur_node_id)
        else:
            state1 = self.edf.render_state_frac(req, hol_vnf_name, cur_node_id)
        state1 = state1.reshape(1,self.worker_model.input_dims) + \
                    np.random.rand(1,self.worker_model.input_dims)/self.params['state_noise_scale']
        state1 = torch.from_numpy(state1).float()
        
        # Perform SFC embedding & Routing
        # values, logprobs, entropies, rewards = [],[],[],[]
        stop_flag = 0
        mov = 0
        this_episode_rwd = 0
        done_embed = 0

        G = torch.Tensor([0]) # Used for N-step training

        if self.net_arch == self.worker_model.NET_ARCH['shared_net_w_RNN']:
            hx = torch.zeros(1, self.params['hidden_layers']['rnn_1'])

        #### While loop (SFC embedding and routing)
        while (stop_flag < 1) and (mov <= self.N_steps):
            self.total_iters += 1
            mov += 1
            
            #### Obtain action_values and value, given the current state
            if self.net_arch != self.worker_model.NET_ARCH['shared_net_w_RNN']:
                action_values, critic_value = self.worker_model(state1) # worker_model.forward(state)
            else:
                action_values, critic_value, hx = self.worker_model(state1, hx) # worker_model.forward(state, hx)
                # print(f"hx = {hx}")
            # print(f"critic_value = {critic_value}")
            # print(f"action_values = {action_values}")
            # values.append(critic_value)
            
            #### Apply action to the env
            # Mask out the unconnected-neighbors of cur_node_id
            inv_adj = np.tile(self.edf.adj_mat[cur_node_id], 2) # duplicate this array
            inv_adj = (1.0 - inv_adj) * (-1e6)
            inv_adj = torch.from_numpy(np.array([inv_adj]))
            masked_action_logits = action_values + inv_adj
            # Transform to prob. dist.
            masked_action_probs = F.softmax(masked_action_logits, dim=1)
            action_dist = torch.distributions.Categorical(probs=masked_action_probs.view(-1))
            # action_dist = torch.distributions.Categorical(probs=action_values.view(-1))

            # sample an action from the action distribution
            action_raw = action_dist.sample() # a tensor
            # if self.is_train:
            #     # sample an action from the action distribution
            #     action_raw = action_dist.sample()
            # else:
            #     # get the action associated with highest prob
            #     action_raw = torch.argmax(action_values)
            # print(f"action_raw = {action_raw}")

            # Calculate log_prob
            log_prob_action = action_dist.log_prob(action_raw)
            # logprobs.append(log_prob_action)
            # print(f"{log_prob_action}")

            # Calculate entropy
            # entropy = -(logprob_dist * action_values).sum(1)
            entropy = action_dist.entropy()
            # entropies.append(entropy)

            # Convert to the real action space before applying to the environment
            action_raw = action_raw.detach().to('cpu').item() # convert from tensor to normal Python data type
            action = self.edf.to_real_action(action_raw)
            # print(f"action_raw = {action_raw}| real_action = {action}")
            
            # Update the resources edf.temp_resource_mat, edf.temp_bw_mat
            resource_failed, link_failed = self.edf.make_move(action, req, hol_vnf_name, cur_node_id)
            
            # Update delay budget
            if (resource_failed == 'None') and (link_failed == 'None'):
                residual_delay, proc_latency, new_link_latency = \
                    self.edf.cal_residual_delay(residual_delay, action, req, hol_vnf_name, cur_node_id, 
                                                self.vnf_item, self.vnf_list, self.sfc_embed_nodes)
            
            else: # resource_failed Or link_failed
                residual_delay = -1 #used for calculate reward below
            
            # Update the hol_vnf_name
            prev_hol_vnf_name = hol_vnf_name
            if action['is_embed'] > 0:
                if self.vnf_item < len(self.vnf_list):
                    # Add to vnf_embed_nodes
                    self.vnf_embed_nodes[hol_vnf_name] = cur_node_id
                    # Continue to embed the next vnf
                    self.vnf_item += 1
                    if self.vnf_item < len(self.vnf_list):
                        hol_vnf_name = self.vnf_list[self.vnf_item]    
                    else:
                        hol_vnf_name = 'VNF_Done'
                
            # Update the cur_node_id and list self.sfc_embed_nodes
            prev_node_id = cur_node_id
            cur_node_id = action['next_node']
            self.sfc_embed_nodes.append(cur_node_id) # SFC node mapping
            
            # Check SFC embedding and routing accomplishement
            if (cur_node_id == req['destination']) and (hol_vnf_name == 'VNF_Done') \
                and (resource_failed == 'None') and (link_failed == 'None'):
                done_embed = 1
            else: 
                done_embed = 0
                
            #### Obtain the next state from the env
            if self.params['is_binary_state']:
                state2 = self.edf.render_state_binary(req, hol_vnf_name, cur_node_id)
            else:
                state2 = self.edf.render_state_frac(req, hol_vnf_name, cur_node_id)
            state2 = state2.reshape(1,self.worker_model.input_dims) + \
                        np.random.rand(1,self.worker_model.input_dims)/self.params['state_noise_scale']
            state2 = torch.from_numpy(state2).float()
        
            #### Calculate step-reward
            if residual_delay >=0:
                reward = self.edf.reward(action, residual_delay, proc_latency, 
                                        new_link_latency, done_embed, 
                                        prev_node_id, prev_hol_vnf_name, delay_budget)
            else:
                reward = -self.edf.big_reward
            # Append to reward list
            # rewards.append(reward)
            # Accumulate episode reward
            this_episode_rwd += reward
            # print(f"action_raw = {action_raw}| real_action = {action}| step_reward = {reward}")
            
            # Condition for succesfull SFC embedding
            if residual_delay < 0 or reward < -1:
                fail_embed = 1
                success_embed = 0
            else:
                fail_embed = 0
                success_embed = 1 if done_embed > 0 else 0

            print(f"Time_{self.cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req['sfc_id']}-{prev_hol_vnf_name}   Action={action}   Residual_delay={residual_delay:.4f}   Step_Reward={reward:.3f}   Success_embed={success_embed}")
            with open(self.operation_log, 'a') as train_fp:
                print(f"Time_{self.cur_time_slot}_Epoch_{epoch}_Step_{mov}   Cur_node={prev_node_id}   {req['sfc_id']}-{prev_hol_vnf_name}   Action={action}   Residual_delay={residual_delay:.4f}   Step_Reward={reward:.3f}   Success_embed={success_embed}", file=train_fp)
        
            # Register the NEWLY successul embedded request to the serving list
            if success_embed > 0:
                self.sfc_embed_map[req['id']] = self.sfc_embed_nodes
                self.vnf_embed_map[req['id']] = self.vnf_embed_nodes
                self.active_req.append(req['id'])
                self.serving_req.append(req['id'])
                
                # Associate SFC embedding and VNF embedding maps to the req
                accepted_req = {'sfc_map': self.sfc_embed_nodes,
                                'vnf_embed_map': self.vnf_embed_nodes}
                self.registered_req['registered_request'].append(accepted_req)

                # Calculate the e2e delay ratio for the accepted request
                delay_rate = (req['delay_req'] - residual_delay) / req['delay_req']
                self.delay_rate_hist.append(delay_rate)

                # Update the resource consumption of this episode to EDF resource availability
                # if the SFC embedding is successfull
                self.edf.resource_mat = copy(self.edf.temp_resource_mat)
                self.edf.bw_mat = copy(self.edf.temp_bw_mat)
                
                # Keep track of accepted request per time slot
                self.adm_hist[self.cur_time_slot].update({req['id']:1})
                # Stopping criteria
                stop_flag = 1

                with open(self.accept_ratio_log, 'a') as fp:
                    print("1", file=fp)
                self.n_accepted_req += 1
                
            if fail_embed:    
                # Keep track of rejected request per time slot
                self.adm_hist[self.cur_time_slot].update({req['id']:0})
                # self.rejected_req['requests'].append(req._asdict())
                self.rejected_req['requests'].append(req)
                # Stopping criteria
                stop_flag = 1

                with open(self.accept_ratio_log, 'a') as fp:
                    print("0", file=fp)

            # G value (Last Return)
            with torch.no_grad():
                if self.net_arch != self.worker_model.NET_ARCH['shared_net_w_RNN']:
                    _, G = self.worker_model(state2)
                else:
                    _, G, _ = self.worker_model(state2, hx)
                G = G.squeeze(0) * (1 - stop_flag)
            
            # Store examples to memory
            if self.is_train:
                self.memory.add_experience(log_prob_action, entropy, critic_value, reward, stop_flag, G)

            # EDF env transits to the next state
            state1 = state2
        # END of inner WHILE loop
        # delete hx
        if self.net_arch == self.worker_model.NET_ARCH['shared_net_w_RNN']:
            delete_data([hx])
        # Add episode-reward to list
        self.episode_rewards.append(this_episode_rwd)
        # # garbage collector
        # gc.collect()
        # clear_output(wait=True)
        # Return results
        # return values, logprobs, entropies, rewards, G, this_episode_rwd
        return this_episode_rwd, mov
    """""""""""""""""""""""""""""""""""""""""""""""""""""""'"""

    def calc_nstep_returns(self, rewards, dones, next_v_pred):
        rets = torch.zeros_like(rewards)
        future_ret = next_v_pred
        not_done = 1 - dones
        n = len(rewards)
        for t in reversed(range(n)):
            rets[t] = future_ret = rewards[t] + self.gamma * future_ret * not_done[t]
        return rets

    def calc_nstep_advs_v_targets(self, batch):
        "Calculate advantages from the batch of episodes' experiences using N-step return"
        advs_list, v_targets_list = [], []
        for epi in range(self.train_freq):
            advs, v_targets = [], []
            cur_epi_data = {k: batch[k][epi] for k in self.memory.data_keys}
            next_v_pred = cur_epi_data['last_returns'][-1] # G
            v_preds = [item.detach() for item in cur_epi_data['v_preds']]# adv does not accumulate gradient
            v_preds = torch.from_numpy(np.array(v_preds))# convert to a new tensor
            # Calculate n-step return
            nstep_rets = self.calc_nstep_returns(cur_epi_data['rewards'], cur_epi_data['dones'], next_v_pred)
            advs = nstep_rets - v_preds
            v_targets = nstep_rets
            advs_list.append(advs)
            v_targets_list.append(v_targets)

        # flatten the list into one tensor
        advs_list = torch.cat(advs_list, dim=0)
        v_targets_list = torch.cat(v_targets_list, dim=0)
        return advs_list, v_targets_list

    def calc_gae(self, rewards, dones, v_preds_all):
        T = len(rewards)
        assert T+1 == len(v_preds_all), f"T+1: {T+1} v.s. v_preds.shape: {v_preds_all.shape}"  # v_preds runs into t+1
        gaes = torch.zeros_like(rewards)
        future_gae = torch.tensor(0.0, dtype=rewards.dtype)
        not_dones = 1 - dones
        deltas = rewards + self.gamma * v_preds_all[1:] * not_dones - v_preds_all[:-1]
        coef = self.gamma * self.tau
        for t in reversed(range(T)):
            gaes[t] = future_gae = deltas[t] + coef * not_dones[t] * future_gae
        return gaes
    
    def standardize(self, v):
        "Method to standardize a rank-1 np array"
        assert len(v) > 1, 'Cannot standardize vector of size 1'
        v_std = (v - v.mean()) / (v.std() + 1e-08)
        return v_std

    def calc_gae_adv_v_targests(self, batch):
        "Calculate advantages from the batch of episodes' experiences using GAE"
        advs_list, v_targets_list = [], []
        for epi in range(self.train_freq):
            advs, v_targets, v_preds_all = [], [], []
            cur_epi_data = {k: batch[k][epi] for k in self.memory.data_keys}
            next_v_pred = cur_epi_data['last_returns'][-1] #G
            next_v_pred = torch.from_numpy(np.array([next_v_pred]))
            v_preds = [item.detach() for item in cur_epi_data['v_preds']]# adv does not accumulate gradient
            v_preds = torch.from_numpy(np.array(v_preds))# convert to a new tensor
            # Calculate GAE return
            v_preds_all = torch.cat((v_preds, next_v_pred), dim=0)
            advs = self.calc_gae(cur_epi_data['rewards'], cur_epi_data['dones'], v_preds_all)
            v_targets = advs + v_preds
            v_targets_list.append(v_targets)
            advs_list.append(advs)
        
        # flatten the list into one tensor
        advs_list = torch.cat(advs_list, dim=0)
        advs_list = self.standardize(advs_list)
        v_targets_list = torch.cat(v_targets_list, dim=0)
        return advs_list, v_targets_list


    def calc_policy_loss(self, batch, adv_list):
        "Calculate the policy loss from the batch of episodes' experiences"
        logprobs_list = torch.cat(batch['logprobs'], dim=0)
        entropy_list = [batch['entropies'][epi][-1].unsqueeze(0) for epi in range(self.train_freq)]
        entropy_list = torch.cat(entropy_list, dim=0)
        policy_loss = - self.params['actor_factor'] * (logprobs_list * adv_list).mean()
        policy_loss += - (self.params['entropy_factor'] * entropy_list.mean())
        return policy_loss

    def calc_critic_loss(self, batch, v_targets_list):
        "Calculate the ciritic loss from the batch of episodes' experiences"
        v_preds_list = torch.cat(batch['v_preds'], dim=0)
        assert v_preds_list.shape == v_targets_list.shape, "v_preds_list.shape == v_targets_list.shape"
        critic_loss = self.params['critic_factor'] * F.mse_loss(v_preds_list, v_targets_list)
        return critic_loss
        
    def to_torch_batch(self, batch):
        "Convert an episodic batch memory into episodic batch of tensors"
        x = {k: [] for k in self.memory.data_keys}
        for k in self.memory.data_keys:
            for epi in range(self.train_freq):
                if torch.is_tensor(batch[k][epi][0]):
                    temp = batch[k][epi]
                    x[k].append(torch.stack(temp, dim=0).view(-1))
                else:
                    temp = np.array(batch[k][epi])
                    x[k].append(torch.from_numpy(temp.astype(np.float32)))
        return x

    def update_params_batch(self):
        """
        Compute and minimizing the loss for updating the model
    
        Parameters
        ----------

        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        batch = self.memory.sample()
        batch = self.to_torch_batch(batch)
        # Calculate the advatanges and v_targets
        advs_list, v_targets_list = self.calc_advs_v_targets(batch) # depending on the setting, this fn will be n-step return Or GAE
        actor_loss = self.calc_policy_loss(batch, advs_list)
        critic_loss = self.calc_critic_loss(batch, v_targets_list)
        loss = actor_loss + critic_loss
        print(f"actor_loss = {actor_loss:g}\t critic_loss = {critic_loss:g}\t loss = {loss:g}")

        # Update neural network's weights using the loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.worker_model.parameters(), 0.5) # clamping gradient from 0 to 50 for avoiding algorithm degeneration
        
        # Ensure the global and local models share the same gradient
        for local_param, global_param in zip(self.worker_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        
        # Optimizing
        self.optimizer.step()

        # Reload the local_model parameter with the newest global_model parameter
        self.worker_model.load_state_dict(self.global_model.state_dict())
        
        # Logging
        self.actor_loss.append(actor_loss.item())
        self.critic_loss.append(critic_loss.item())
        self.losses.append(loss.item())
        with open(self.loss_log, 'a') as lfp:
            print(f"{actor_loss.item():g} {critic_loss.item():g} {loss.item():g}", file=lfp)

        # # garbage collector
        # gc.collect()
        return actor_loss, critic_loss
        
    def update_params(self, values, logprobs, entropies, rewards, G):
        """
        Compute and minimizing the loss for updating the model
    
        Parameters
        ----------

        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        # print(f"values = {values}")
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        # print(f"logprobs = {logprobs}")
        entropies = torch.stack(entropies).flip(dims=(0,)).view(-1)
        # print(f"Entropies = {entropies}")
        
        # Calculate returns
        rewards.reverse()
        n_step_returns = []
        ret_ = G
        for r in range(len(rewards)):
            ret_ = rewards[r] + self.gamma * ret_
            n_step_returns.append(ret_)
        n_step_returns = torch.stack(n_step_returns).view(-1)
        # Returns = F.normalize(Returns,dim=0)
        # print(f"n_step_returns = {n_step_returns}")
        
        #### TODO: Normalize the advantages if using GAE
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)#normalize the advantages

        # n-step actor loss
        advantages = n_step_returns - values.detach()
        # print(f"advantages = {advantages}")
        actor_loss = - self.params['actor_factor']*(logprobs*advantages).mean()
        actor_loss += (- self.params['entropy_factor']*entropies.mean())
        # print(f"actor_loss = {actor_loss}")
        
        # critic_loss = torch.pow(values - Returns, 2)
        critic_loss = self.params['critic_factor'] * F.mse_loss(values, n_step_returns) #v_target = n_step_returns
        # print(f"critic_loss = {critic_loss}")
        # loss = self.params['actor_factor']*actor_loss.sum() + self.params['critic_factor']*critic_loss.sum() # weighted sum of actor_loss and critic_loss
        loss = actor_loss + critic_loss
        print(f"actor_loss = {actor_loss:g}\t critic_loss = {critic_loss:g}\t loss = {loss:g}")
        
        # Update neural network's weights using the loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.worker_model.parameters(), 0.5) # clamping gradient from 0 to 50 for avoiding algorithm degeneration
        
        # Ensure the global and local models share the same gradient
        for local_param, global_param in zip(self.worker_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        
        # Optimizing
        self.optimizer.step()
        
        # Logging
        self.actor_loss.append(actor_loss.item())
        self.critic_loss.append(critic_loss.item())
        self.losses.append(loss.item())
        with open(self.loss_log, 'a') as lfp:
            print(f"{actor_loss.item():g} {critic_loss.item():g} {loss.item():g}", file=lfp)
        
        # Reload the local_model parameter with the newest global_model parameter
        self.worker_model.load_state_dict(self.global_model.state_dict())

        # delete data to save RAM
        delete_data([values, logprobs, entropies, rewards, G])
        # values, logprobs, entropies, rewards, G = [], [], [], [], []
        # # garbage collector
        # gc.collect()
        return actor_loss, critic_loss, len(rewards)

    def update_params_gae(self, values, logprobs, entropies, rewards, G):
        """
        Compute and minimizing the loss for updating the model
    
        Parameters
        ----------

        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        print(f"G = {G}")
        # print(f"values = {values}")
        # values = [v.detach() for v in values]
        # print(f"values = {values}")
        values.append(G.unsqueeze(0))
        print(f"values = {values}")
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        # print(f"values after = {values}")
        
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        # print(f"logprobs = {logprobs}")
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        # print(f"rewards = {rewards}")
        entropies = torch.stack(entropies).flip(dims=(0,)).view(-1)
        # print(f"Entropies = {entropies}")
        
        #### TODO: Bug here --> making negative probability
        # Calculate advantage and normalize it
        # advantages = Returns - values.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)#normalize the advantages

        # Calculate gae
        gaes = []
        gae_ = 0.0
        for i in range(rewards.shape[0]):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae_ = gae_ * self.gamma * self.tau + delta
            gaes.append(gae_)
        gaes = torch.stack(gaes).view(-1)

        # gae actor loss
        actor_loss = -logprobs*gaes - self.params['entropy_factor']*entropies
        # print(f"actor_loss = {actor_loss}")
        
        # Calculate returns 
        Returns = []
        ret_ = G.detach()
        # print(f"ret_ = {ret_}")
            
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + self.gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        # Returns = F.normalize(Returns,dim=0)
        # print(f"Returns = {Returns}")

        critic_loss = 0.5 * torch.pow(values[:-1] - Returns, 2)
        # print(f"critic_loss = {critic_loss}")
        # loss = self.params['actor_factor']*actor_loss.sum() + self.params['critic_factor']*critic_loss.sum() # weighted sum of actor_loss and critic_loss
        loss = (self.params['actor_factor']*actor_loss + self.params['critic_factor']*critic_loss).mean()
        print(f"actor_loss = {actor_loss}| critic_loss = {critic_loss}| total loss = {loss}")
        
        # Update neural network's weights using the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_model.parameters(), 40.0) # clamping gradient from 0 to 50 for avoiding algorithm degeneration
        
        # Ensure the global and local models share the same gradient
        for local_param, global_param in zip(self.worker_model.parameters(), 
                                                self.global_model.parameters()):
            global_param._grad = local_param.grad
        
        # Optimizing
        self.optimizer.step()
        
        # Reload the local_model parameter with the newest global_model parameter
        self.worker_model.load_state_dict(self.global_model.state_dict())

        # Logging
        self.actor_loss.append(actor_loss.mean())
        self.critic_loss.append(critic_loss.mean())
        self.losses.append(loss)
        with open(self.loss_log, 'a') as lfp:
            print(f"{actor_loss.mean():.5f} {critic_loss.mean():.5f} {loss.mean():.5f}", file=lfp)

        return actor_loss, critic_loss, len(rewards)
    
    def calc_throughputs(self):
        """Calculate the real throughput and the offered load
            Export data to log file

        Returns:
            real_tp [list]: list of per-epoch (time slot) throughput obtained from the RL algo
            offered_load [list]: list of per-epoch (time slot) throughput obtained from the perfect scenario
        """
        # Calculate throughputs
        offered_load, real_tp = calc_throughput(self.adm_hist, self.req_list, 
                                              self.all_arrive_req, self.serving_req, 
                                              self.start_time_slot, 
                                              self.edf.sfc_specs, self.edf.vnf_specs)
        # Export data to log file
        with open(self.throughput_log, 'a') as fp:
            print("Time_slots \t Offered_load [bw unit]  \t Real_throughput [bw unit]", file=fp)
            for i in range(len(offered_load)):
                print(f"{i} {offered_load[i]:.3f} {real_tp[i]:.3f}", file=fp)

        return offered_load, real_tp

    def export_resource_usage(self):
        """Export CPU, RAM, STORAGE, and BANDWIDTH usages to log file
        """
        with open(self.rsc_usage_log, 'a') as fp:
            print("CPU \t RAM \t STORAGE \t BANDWIDTH", file=fp)
            [print(f"{self.cpu_usage[i]} {self.ram_usage[i]} {self.sto_usage[i]} {self.bw_usage[i]}", file=fp) \
                for i in range(len(self.cpu_usage))]

    def export_registered_reqs(self):
        """Export registered service requests to JSON file
        """
        with open(self.registerd_req_log, 'w') as fp:
            json.dump(self.registered_req, fp, indent=2, default=numpy_encoder)

    def calc_avg_path_len(self):
        tot_links = 0
        for req_item in self.registered_req['registered_request']:
            # tot_links += len(req_item['sfc_map']) - 1 # old calculation
            # Exclude the self-connected link, i.e., i-i link
            n_diff_nodes, _, _ = find_runs(req_item['sfc_map'])
            tot_links += len(n_diff_nodes) - 1
        avg_path_len = tot_links / self.n_accepted_req
        return avg_path_len
        
# END OF A3CWORKER CLASS=============================================================================

 

# =============================================================================
#### main function
# =============================================================================
# def main():
#     is_training = False
    
#     MODEL_DIR = "A3C_SFCE_models"
#     MODEL_NAME = "a3c_cartpole"
    
#     # Neural network structure
#     input_dims = 4
#     fc_hid_layers = [64, 128, 64]
#     actor_dims = 2
    
#     # Training Datasets
#     net_topo_list = ['ibm_100000_slots_1_con.net', 'ibm_200000_slots_1_con.net']
#     sfc_spec_file = 'sfc_file.sfc'
#     traffic_dataset_list = ['reordered_traffic_100000_slots_1_con.tra', 'reordered_traffic_200000_slots_1_con.tra']
    
#     #### Training phase
#     if is_training:
#         print("Training the A3C RL-agent")
#         MasterNode = ACNet(MODEL_NAME, MODEL_DIR, fc_hid_layers, input_dims, actor_dims) #create an instance of A2C model
#         MasterNode.share_memory() # the model's parameters are globally shared among processes
#         N_steps = 50;
#         processes = [] # A list to store instantiated processes
#         params = {
#             'epochs':2000,
#             'n_workers': 6,
#             'net_topos': net_topo_list,
#             'sfc_spec': sfc_spec_file,
#             'traffic_datasets': traffic_dataset_list
#         }
        
#         # A shared global counter using multiprocessing’s
#         # built-in shared object. The ‘i’ parameter indicates
#         # the type is integer.
#         counter = mp.Value('i',0)  # global counter shared among processes
        
#         for i in range(params['n_workers']):
#             # Create a worker environment
#             worker_env = gym.make("CartPole-v1")
#             # Create an A3C instance
#             #a3c_worker = A3CWorker(i, MasterNode, worker_env, params)
#             a3c_worker = A3CWorker(worker_id=i, worker_model=MasterNode, 
#                                    worker_env=worker_env, params=params,
#                                    N_steps=N_steps, clc=0.1, gamma=0.95)
#             # instantiate a process invoking a worker who does the training
#             # p = mp.Process(target=worker, args=(i, MasterNode, counter, params, worker_env, N_steps))
#             p = mp.Process(target=a3c_worker.run, args=(counter,))
#             p.start() 
#             processes.append(p)
        
#         # Wait for each process to be done before returning to the main thread
#         [p.join() for p in processes] 
#         # Terminate each process
#         [p.terminate() for p in processes]
#         # Global counter and the first process's exit code
#         print(counter.value,processes[1].exitcode) 
        
#         # Save the model
#         print("Training done!")
#         MasterNode.save_params()
#         # save_model(model_dir, model_name, MasterNode)
    
#     #### Testing phase
#     else:
#         print("Testing the trained A2C RL-agent")
#         env = gym.make("CartPole-v1")
#         env.reset()
        
#         # Load neural network model
#         MasterNode = ACNet(MODEL_NAME, MODEL_DIR, fc_hid_layers, input_dims, actor_dims) # create a new instance of neural net
#         # load_model(model_dir, model_name, MasterNode)
#         MasterNode.load_params()
        
#         trial_rewards = []
#         for n_trial in range(20):
#             rewards = []
#             accum_reward = 0
#             done = False
#             total_iter = 0
#             while (done==False and total_iter<500):
#                 total_iter += 1
#                 # observe current state
#                 state_ = np.array(env.env.state)
#                 state = torch.from_numpy(state_).float()
                
#                 # feed state to the neural network
#                 logits, value = MasterNode(state)
                
#                 # make a decision
#                 action_dist = torch.distributions.Categorical(logits=logits)
#                 action = action_dist.sample()
                
#                 # observe next_step, reward, done
#                 state2, reward, done, info = env.step(action.detach().numpy())
#                 # print(f"Iter {total_iter}| Reward = {reward} | Done = {done}")
#                 accum_reward = accum_reward + reward
#                 rewards.append(accum_reward)
#                 # check episode done
#                 if done: 
#                     env.reset()
#                 else:# or continue to play
#                     state_ = np.array(env.env.state)
#                     state = torch.from_numpy(state_).float()
#                 # env.render()
#             print(f"Trial {n_trial}| accum_reward = {accum_reward}")
#             trial_rewards.append(accum_reward)
#             # plt.plot(rewards)
#         avg_trial_reward = np.mean(trial_rewards)
#         print(f"avg_trial_reward = {avg_trial_reward}")
# =============================================================================

# =============================================================================
# MAIN: Perform multiprocessing for training the A2C
# =============================================================================
# if __name__ == "__main__":
#     main()
#     pass