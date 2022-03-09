# -*- coding: utf-8 -*-
"""
Created on Wed July 14 22:37:10 2021

@author: Onlyrich-Ryzen

Implement EDF environment
+ Embed the VNF_type_support for the current HoL VNF in function 
    + binary_state_dim
    + render_state_binary
    
Change log from edf_env_v2_01.py:
    + Integrate into the system_state the shortest distance (from the current node to the destination node)

"""

import numpy as np
import os
import sys
import torch
from copy import copy
import networkx as nx
from data_utils_CONUS_02 import build_net_topo, build_node_info, get_vnf_specs, get_sfc_specs, build_vnf_support
from data_utils_CONUS_02 import build_neighbors, init_resource, init_bw
from data_utils_CONUS_02 import build_net_topo_CONUS, get_vnf_specs_CONUS, get_sfc_specs_CONUS
from collections import namedtuple



class EdfEnv(object):
    ''' EDF network environment'''
    def __init__(self, data_path, net_file='ibm.net', sfc_file='sfc_file.sfc', resource_scaler=1.):
        ''' Initialize the EDF environment
        
        Parameters:
            data_path: string
                FOLDER PATH TO THE DATASET
        '''
        self.topo_file = os.path.join(data_path, net_file)
        self.vnf_sfc_file = os.path.join(data_path, sfc_file)
        
        # self.net_topo, self.n_nodes, self.n_edges = build_net_topo(self.topo_file, resource_scaler)
        self.net_topo, self.n_nodes, self.n_edges = build_net_topo_CONUS(self.topo_file, resource_scaler)
        self.node_info, self.compute_nodes = build_node_info(self.net_topo)
        self.edge_list = list(self.net_topo.edges)
        
        # self.n_vnf_types, self.vnf_specs = get_vnf_specs(self.vnf_sfc_file)
        self.n_vnf_types, self.vnf_specs = get_vnf_specs_CONUS(self.vnf_sfc_file)
        self.n_sfc_types, self.sfc_specs = get_sfc_specs_CONUS(self.vnf_sfc_file)
        self.vnf_support_mat = build_vnf_support(self.net_topo, self.vnf_specs)
        self.adj_mat = build_neighbors(self.net_topo)
        
        self.resource_types = 3 #[cpu, ram, storage] resource types
        self.resource_mat = init_resource(self.net_topo)
        self.bw_mat = init_bw(self.net_topo)
        
        # Modify the adjacency for COMPUTE nodes 
        np.fill_diagonal(self.adj_mat, self.compute_nodes)# Each COMPUTE node is adjacent to itself
        # Self-connected link of COMPUTE nodes have super large bw
        super_bw = np.max(self.bw_mat) * 1e3
        for i in range(self.n_nodes):
            self.bw_mat[i][i] = self.adj_mat[i][i] * super_bw

        # Variables for receiving request's info
        self.temp_resource_mat = copy(self.resource_mat)
        self.temp_bw_mat = copy(self.bw_mat)
        
        # Calculate the shortest paths for all pair of nodes in the network
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path(self.net_topo, weight='delay'))
        
        print('Init EDF environment...done!')
    
    def binary_state_dim(self):
        ''' Calculate the dimension of the BINARY system state
        '''
        System_State_Dims = namedtuple('System_State_Dims', ['dst_node', 'cur_node', 'neighbor_nodes', 
                                                             'hol_vnf', 'sfc_id', 'vnf_available',
                                                             'rsc_vec', 'bw_vec', 'inv_path_length'])
        
        self.state_dim_info = System_State_Dims(dst_node = self.n_nodes,# one-hot vector representing the DST node
                                                cur_node = self.n_nodes,# one-hot vector representing the current node
                                                neighbor_nodes = self.n_nodes,# one-hot vector representing the connected neighbor of current node
                                                hol_vnf = self.n_vnf_types,# one-hot vector representing the current HoL VNF
                                                sfc_id = self.n_sfc_types,#one-hot vector encoding the SFC being used by the request
                                                vnf_available = self.n_nodes,#one-hot vector encoding the support of current Hol VNF
                                                rsc_vec = self.n_nodes,# resource demand binary
                                                bw_vec = self.n_nodes, #bw demand binary
                                                inv_path_length = self.n_nodes #vector encoding the distance from the neighbors of current node to the req.dst
                                                )
        self.n_state_dims = sum(self.state_dim_info)        
        print(f'Using BINARY system state of {self.n_state_dims}-dim vector')
        
    def fractional_state_dim(self):
        ''' Calculate the dimension of the FRACTIONAL system state
        '''
        System_State_Dims = namedtuple('System_State_Dims', ['dst_node', 'cur_node', 'neighbor_nodes', 'hol_vnf', 'rsc_vec', 'bw_vec'])
        self.state_dim_info = System_State_Dims(dst_node = self.n_nodes,# one-hot vector representing the DST node
                                                cur_node = self.n_nodes,# one-hot vector representing the current node
                                                neighbor_nodes = self.n_nodes,# one-hot vector representing the connected neighbor of current node
                                                hol_vnf = self.n_vnf_types,# one-hot vector representing the current HoL VNF
                                                rsc_vec = self.n_nodes * self.resource_types,# resource demand binary
                                                bw_vec = self.n_nodes #bw demand binary
                                                )
        self.n_state_dims = sum(self.state_dim_info)
        print('Using FRACTIONAL system state')
        
    def to_real_action(self, raw_action):
        ''' Interprete the action obtained from the Q-Agent to the SFC embedding and routing decisions
        
        Parameters:
            action : int
                THE OUTPUT FROM THE Q-AGENT
        
        Returns:
            action: namedtupled
                action.is_embed: int
                    THE VNF EMBEDDING DECISION
                action.next_node: int
                    THE ROUTING DECISION
        '''
        # VNF embedding decision, Routing destination
        is_embed, next_node = np.divmod(raw_action, self.n_nodes)
        Real_Action = namedtuple('Real_Action', ['is_embed', 'next_node'])
        action = Real_Action(is_embed=is_embed, next_node=next_node)
        return action
    
    def make_move(self, real_action, req, hol_vnf_name, cur_node_id):
        ''' Update the resource after applying the real_action into the EDF environment
        
        Parameters:
            real_action: namedtuple
                real_action.is_embed: vnf embedding decision
                real_action.next_node: routing decision
        Returns:
            resource_failed : STRING
                NOTIFICATION FOR NEGATIVE RESOURCE
            link_failed : STRING
                NOTIFICATION FOR NEGATIVE LINK BANDWITH
        '''
        # Update resource of the current node if action.is_embed = 1
        resource_failed = 'None'
        if hol_vnf_name != 'VNF_Done':
            if real_action.is_embed > 0:
                # resource_hol_vnf = [self.vnf_specs[hol_vnf_name][i] * req.bw for i in range(2,5)]
                cpu_req = self.vnf_specs[hol_vnf_name].cpu * req.bw
                resource_hol_vnf = [cpu_req, req.RAM_req, req.MEM_req]#[cpu, ram, sto] resources
                self.temp_resource_mat[cur_node_id] -= resource_hol_vnf
                if True in (self.temp_resource_mat[cur_node_id]<0):
                    resource_failed = f'Negative resource at node {cur_node_id}'
                    # print(resource_failed)
            
        # Update the link bw
        link_failed = 'None'
        self.temp_bw_mat[cur_node_id][real_action.next_node] -= req.bw
        # self.temp_bw_mat[cur_node_id][real_action.next_node] = max(self.temp_bw_mat[cur_node_id][real_action.next_node], 0)
        if self.temp_bw_mat[cur_node_id][real_action.next_node] <0:
            link_failed = f'Negative bw at link {cur_node_id}--{real_action.next_node}'
            # print(link_failed)
        
        return resource_failed, link_failed
        
    
    def cal_residual_delay(self, residual_delay, real_action, req, hol_vnf_name, cur_node_id):
        ''' Update the residual_delay after each action
        
        Paramters:
            residual_delay: float
                THE REMAINING DELAY BUDGET OF THE SFC REQUEST
            real_action: list
                THE ACTION BEING APPLIED TO THE EDF ENVIRONEMENT
        '''
        # print("Update residual delay of the current SFC request!")
        # Processing delay
        proc_latency = 0.0
        if hol_vnf_name != 'VNF_Done':
            if real_action.is_embed:
                proc_latency = req.bw * self.vnf_specs[hol_vnf_name].proc_delay
        
        # Link delay
        added_link = (cur_node_id, real_action.next_node)
        if added_link in self.edge_list:
            new_link_latency = self.net_topo.edges[cur_node_id, real_action.next_node]['delay']
        elif cur_node_id == real_action.next_node:# Self-connected link of COMPUTE nodes have zero-latency
            if self.compute_nodes[cur_node_id] > 0:
                new_link_latency = 0.005 if hol_vnf_name != 'VNF_Done' else 0.035 # XXX: Should revise this value
            else: 
                new_link_latency = 1e6
        else:
            new_link_latency = 1e6
        
        # Update delay budget
        if cur_node_id == real_action.next_node and self.compute_nodes[cur_node_id] > 0:
            added_delay = proc_latency #self-connected link does not affect the added_delay
        else:
            added_delay = proc_latency + new_link_latency
        residual_delay -= added_delay
        
        return residual_delay, added_delay, proc_latency, new_link_latency
    
    
    def reward(self, real_action, residual_delay, proc_latency, 
               new_link_latency, done_embed, prev_node_id, prev_hol_vnf_name):
        ''' Calculate the reward, given the EDF's state and current action
        
        Parameters:
            real_action: namedtuple
                real_action.is_embed
                real_action.next_node
            residual_delay: float
                REMAINING DELAY BUDGET
            proc_latency: float
                PROCESSING LATENCY
            new_link_latency: float
                LATENCY OF THE NEWLY ADDED LINK
            done_embed: int (0,1)
                INDICATE THE COMPLETENESS OF THE SFC EMBEDDING
            
        '''
        big_reward = 3
        # Old cost factors
        # beta1 = 0.01
        # beta2 = 0.001
        
        # Cost factors for the new dataset
        beta1 = 25.0
        beta2 = 15.0
        
        # Bonus from VNF embedding
        # cost due to additional link delay
        if proc_latency > 0:
            bonus = 1.0
            cost = beta2 * proc_latency + beta1 * new_link_latency
            # cost = beta2 * proc_latency + beta1 * new_link_latency
        else:
            bonus = 0
            cost = beta1 * new_link_latency
        
        # Check vnf supporting when real_action.is_embed=1
        if real_action.is_embed > 0:
            if prev_hol_vnf_name != 'VNF_Done':
                vnf_id = self.vnf_specs[prev_hol_vnf_name].vnf_id
                # not support prev_hol_vnf_name
                if self.vnf_support_mat[prev_node_id][vnf_id] < 1:
                    print(f'Node {prev_node_id} NOT support {prev_hol_vnf_name}')
                    return -big_reward
        
        # Check negative resource and bandwidth
        neg_resource_check = self.temp_resource_mat < 0
        neg_bw_check = self.temp_bw_mat < 0
        # negative resource Or negative bandwidth
        if (True in neg_resource_check) or (True in neg_bw_check):
            if True in neg_resource_check:
                print(f'Negative resource at node {prev_node_id}')
            if True in neg_bw_check:
                print(f'Negative bandwidth at link {prev_node_id}--{real_action.next_node}')
            return -big_reward
        
        # Reward
        if done_embed > 0: #done SFC embedding and routing
            rwd = big_reward if residual_delay >=0 else -big_reward
        else:
            rwd = bonus - cost if residual_delay >= 0 else -big_reward
        return rwd
    
    def sojourn_monitor(self, active_req, req_list, 
                        sfc_embed_map, vnf_embed_map, 
                        cur_time_slot, expired_req):
        ''' Evict the expired SFC request at the current time slot; 
            Get back the occupied resources from the expired requests

        Parameters:
        -------
            active_req: list 
                LIST OF CURRENTLY ACTIVE ACCEPTED REQUESTS
            req_list: list
                LIST OF REQUEST IN THE DATASET
            sfc_embed_map: dict
                DICT OF SFC_EMBED_NODES
            vnf_embed_map: dict
                DICT OF VNF_HOSTING_NODES
            cur_time_slot: int
                CURRENT TIME SLOT INDEX
            expired_req: list
                LIST OF EXPIRED REQUEST
        Returns
        -------
        None.

        '''
        expired_key = []
        for key in active_req:
            # Check the expiration
            timeout_req = req_list[key]
            if cur_time_slot >= timeout_req.end_time:
                # Get the sfc_node_mapping
                sfc_node_mapping = sfc_embed_map[key]
                vnf_node_mapping = vnf_embed_map[key]
                
                # Get back the resources from the expired requests
                self.getback_resource(timeout_req, sfc_node_mapping, vnf_node_mapping)
                
                # Put serving_req into expired_req list
                expired_req.append(timeout_req)
                
                # Put key into expired_key list
                expired_key.append(key)
                
        # Remove the timeout_req from the sfc_embed_map
        for key in expired_key:
            active_req.remove(key)
        # End of this function
        
    def sojourn_monitor_rand_arrival_req(self, active_req, req_list, 
                        sfc_embed_map, vnf_embed_map, 
                        cur_time_slot, expired_req):
        ''' Evict the expired SFC request at the current time slot; 
            Get back the occupied resources from the expired requests

        Parameters:
        -------
            active_req: list 
                LIST OF CURRENTLY ACTIVE ACCEPTED REQUESTS
            req_list: list
                LIST OF REQUEST IN THE DATASET
            sfc_embed_map: dict
                DICT OF SFC_EMBED_NODES
            vnf_embed_map: dict
                DICT OF VNF_HOSTING_NODES
            cur_time_slot: int
                CURRENT TIME SLOT INDEX
            expired_req: list
                LIST OF EXPIRED REQUEST
        Returns
        -------
        None.

        '''
        expired_key = []
        for key in active_req:
            # Check the expiration
            timeout_req = req_list[key]
            # the duration time of the task
            # use this because the timeout_req.arrival_time is not synchronized with cur_time_slot
            task_duration = timeout_req.end_time - timeout_req.arrival_time 
            if cur_time_slot >= task_duration:
                # Get the sfc_node_mapping
                sfc_node_mapping = sfc_embed_map[key]
                vnf_node_mapping = vnf_embed_map[key]
                
                # Get back the resources from the expired requests
                self.getback_resource(timeout_req, sfc_node_mapping, vnf_node_mapping)
                
                # Put serving_req into expired_req list
                expired_req.append(timeout_req)
                
                # Put key into expired_key list
                expired_key.append(key)
                
        # Remove the timeout_req from the sfc_embed_map
        for key in expired_key:
            active_req.remove(key)
        # End of this function
    
    def render_state_binary(self, req, hol_vnf_name, cur_node_id):
        ''' Render the environment BINARY system state 
        
        Parameters:
            req: namedtuple
                REQUEST INFORMATION
            hol_vnf_name: string
                NAME OF THE CURRENT HEAD-OF-LINE VNF
            cur_node_id: int
                ID OF THE CURRENT NODE THAT IS PROCESSING THE REQUEST
        Return:
            state: binary np.array 
                BINARY SYSTEM STATE
        '''
        # One-hot encoded dst_node
        dst_encode = np.zeros(self.n_nodes)
        dst_encode[req.destination] = 1.0
        
        # One-hot encoded cur_node
        cur_encode = np.zeros(self.n_nodes)
        cur_encode[cur_node_id] = 1.0
        
        # One-hot endcoded neighbor_nodes
        neighbor_encode = copy(self.adj_mat[cur_node_id])
        
        # One-hot encoded hol_vnf
        hol_vnf_encode = np.zeros(self.n_vnf_types)
        if hol_vnf_name != 'VNF_Done': # All VNFs are already embedded
            hol_vnf_id = self.vnf_specs[hol_vnf_name].vnf_id
            hol_vnf_encode[hol_vnf_id] = 1.0
            
        # One-hot encoded SFC_id from SFC_name
        sfc_idx = int(req.sfc_id.split(sep='_')[-1])
        sfc_id_encode = np.zeros(self.n_sfc_types)
        sfc_id_encode[sfc_idx] = 1.0
        
        # One-hot encoded vnf_availability of nodes
        if hol_vnf_name != 'VNF_Done':
            vnf_available_encode = np.zeros(self.n_nodes)
            for node in range(self.n_nodes):
                vnf_available_encode[node] = self.vnf_support_mat[node][hol_vnf_id]
        else:
            vnf_available_encode = np.ones(self.n_nodes)
            
        # One-hot encoded resource condition
        # By comparing the component-wise the resource of all nodes and the hol_vnf
        resource_encode = np.zeros(self.n_nodes)
        if hol_vnf_name != 'VNF_Done':
            # resource_hol_vnf = [self.vnf_specs[hol_vnf_name][i] * req.bw for i in range(2,5)]
            cpu_req = self.vnf_specs[hol_vnf_name].cpu * req.bw
            resource_hol_vnf = [cpu_req, req.RAM_req, req.MEM_req]#[cpu, ram, sto] resources
            for node in range(self.n_nodes):
                node_resource = self.temp_resource_mat[node] * self.vnf_support_mat[node][hol_vnf_id]
                x = node_resource >= resource_hol_vnf
                if False not in x:
                    resource_encode[node] = 1.0 #all resource components of node is larger than those of the hol_vnf
        
        # One-hot encoded bandwidth condition
        bw_encode = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            if self.adj_mat[cur_node_id][j] < 1.0:
                bw_encode[j] = 0.0
            else:
                bw_encode[j] = 1.0 if self.temp_bw_mat[cur_node_id][j] >= req.bw else 0.0
        
        # Encoding the invsersed_distance from the neighbors of current node to the req.dst
        inv_path_length_encode = np.ones(self.n_nodes)
        for neighbor in range(self.n_nodes):
            cur_node_to_neighbor_path_length = len(self.shortest_paths[cur_node_id][neighbor]) - 1
            if cur_node_to_neighbor_path_length <= 1: # neighbor is a connected node of cur_node_id
                neighbor_to_dst_path_length = len(self.shortest_paths[neighbor][req.destination]) - 1
                tot_path_length = cur_node_to_neighbor_path_length + neighbor_to_dst_path_length
                if tot_path_length > 0:
                    inv_path_length_encode[neighbor] = 1. / tot_path_length
                else:
                    inv_path_length_encode[neighbor] = 1.
            else:# neightbor is not ac onnected neighbor node of cur_node_id
                inv_path_length_encode[neighbor] = 0.
            
        # Concatenate all the vector
        state = np.concatenate((dst_encode, cur_encode, neighbor_encode, 
                                hol_vnf_encode, 
                                sfc_id_encode,
                                vnf_available_encode,
                                resource_encode, bw_encode,
                                inv_path_length_encode))
        
        return state
    
    def render_state_frac(self, req, hol_vnf_name, cur_node_id):
# !!!: Remember to revise this function before using it
        ''' Render the environment FRACTIONAL system state 
        
        Parameters:
            req: namedtuple
                REQUEST INFORMATION
            hol_vnf_name: string
                NAME OF THE CURRENT HEAD-OF-LINE VNF
            cur_node_id: int
                ID OF THE CURRENT NODE THAT IS PROCESSING THE REQUEST
        Return:
            state: fractional np.array 
                FRACTIONAL SYSTEM STATE
        '''
        # One-hot encoded dst_node
        dst_encode = np.zeros(self.n_nodes)
        dst_encode[req.destination] = 1.0
        
        # One-hot encoded cur_node
        cur_encode = np.zeros(self.n_nodes)
        cur_encode[cur_node_id] = 1.0
        
        # One-hot endcoded neighbor_nodes
        neighbor_encode = copy(self.adj_mat[cur_node_id])
        
        # One-hot encoded hol_vnf
        hol_vnf_encode = np.zeros(self.n_vnf_types)
        if hol_vnf_name != 'VNF_Done': # All VNFs are already embedded
            hol_vnf_id = self.vnf_specs[hol_vnf_name].vnf_id
            hol_vnf_encode[hol_vnf_id] = 1.0
        
        # Fractional-encoded resource condition
        # By comparing the component-wise the resource of all nodes and the hol_vnf
        resource_encode = np.zeros((self.n_nodes, self.resource_types))
        if hol_vnf_name != 'VNF_Done':
            # resource_hol_vnf = [self.vnf_specs[hol_vnf_name][i] * req.bw for i in range(2,5)]
            cpu_req = self.vnf_specs[hol_vnf_name].cpu * req.bw
            resource_hol_vnf = [cpu_req, req.RAM_req, req.MEM_req]#[cpu, ram, sto] resources
            for node in range(self.n_nodes):
                node_resource = self.temp_resource_mat[node] * self.vnf_support_mat[node][hol_vnf_id]
                rsc_ratio = node_resource / resource_hol_vnf
                resource_encode[node] = rsc_ratio
        resource_encode = np.reshape(resource_encode, self.n_nodes * self.resource_types) # flatten to 1-dim vector
        
        # Fractional-endcoded bandwidth condition
        bw_encode = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            if self.adj_mat[cur_node_id][j] < 1.0:
                bw_encode[j] = 0.0
            else:
                bw_encode[j] = self.temp_bw_mat[cur_node_id][j] / req.bw if req.bw > 0 else 1
                    
        # Concatenate all the vector
        state = np.concatenate((dst_encode, cur_encode, neighbor_encode, hol_vnf_encode, resource_encode, bw_encode))
        
        return state
    
    def getback_resource(self, timeout_req, sfc_node_mapping, vnf_node_mapping):
        ''' Get the resource back from the time_out_request
        
        Parameters:
            timeout_req: namedtuple
                THE EXPIRED REQUEST
            sfc_node_mapping: list
                THE ORDERD LIST OF SFC NODE MAPPING FOR THE timeout_req
            vnf_node_mapping: dict
                key : VNF name
                val : node_id
        Returns:
            None
        '''
        # Get back resources (CPU, RAM, STO)
        for vnf_name in vnf_node_mapping:
            hosting_node_id = vnf_node_mapping[vnf_name]
            # released_resource = [self.vnf_specs[vnf_name][i] * timeout_req.bw for i in range(2,5)]
            cpu_req = self.vnf_specs[vnf_name].cpu * timeout_req.bw
            released_resource = [cpu_req, timeout_req.RAM_req, timeout_req.MEM_req]#[cpu, ram, sto] resources
            if self.compute_nodes[hosting_node_id] <1:
                print("WRONG COMPUTE NODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                self.resource_mat[hosting_node_id] += released_resource
        # Get back link bw
        for i in range(len(sfc_node_mapping)-1):
            start_node = sfc_node_mapping[i]
            end_node = sfc_node_mapping[i+1]
            if self.adj_mat[start_node][end_node] <1:
                print("WRONG LINK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                self.bw_mat[start_node][end_node] += timeout_req.bw
    
    