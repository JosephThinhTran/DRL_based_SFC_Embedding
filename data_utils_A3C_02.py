# -*- coding: utf-8 -*-
"""
Created on Wed July 14 22:37:10 2021

Import data from dataset:
    - *.net: network topology (routers, link_bw, link_latency), compute nodes, supporting VNFs
    - *.sfc: specs of SFCs and VNFs supported by the network
    - *.tra: service request data

@author: Onlyrich-Ryzen

Change log from previous version data_utils.py:
    + Fix function get_vnf_specs() for compatible with the SOF_Data_Sets-v09
        + The VNF_spec now : {vnf_id, cpu, proc_delay}
"""

import networkx as nx
import os
from random import seed     # seed the pseudorandom number generator
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import json
import sys


''' Convert namedtupled to conventional dict for compatible with the multiprocessing '''
#### TODO: Convert namedtuple into dict type

"""
Node_Info = namedtuple('Node_Info', ['node_id', 'cpu_cap', 'cpu_free', 'ram_cap', 'ram_free',
                                          'sto_cap', 'sto_free', 'employable_vnf'])
"""

"""
Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
                                    'bw', 'delay_req',
                                    'arrival_time', 'end_time', 
                                    'RAM_req', 'MEM_req',
                                    'co_id'])
"""


#########################################
def build_net_topo(topo_file, resource_scaler=1.0, is_draw=False):
    ''' Build a network topology from the input file

    Parameters:
        topo_file : string
            FULL_PATH TO THE DATA FILE

    Output:
        g : DiGraph object of networkx
            GRAPH BUILT FROM topo_file
    '''
    if resource_scaler < 0. or resource_scaler > 1.: 
        raise ValueError("resource_scaler argument must be within [0,1]")
    
    file_name = topo_file.split(sep='\\')[-1]
    print(f"Build network topology from '{file_name}'")
    g = nx.DiGraph()

    # read data_file into of lines
    with open(topo_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Add nodes
    n_nodes = int(lines[0].split()[0])
    g.add_nodes_from([n for n in range(n_nodes)])
    # Initialize node' attributes
    for n in range(n_nodes):
        g.nodes[n]['dc_id'] = ''
        g.nodes[n]['cpu_cap'] = g.nodes[n]['cpu_free'] = 0
        g.nodes[n]['ram_cap'] = g.nodes[n]['ram_free'] = 0
        g.nodes[n]['sto_cap'] = g.nodes[n]['sto_free'] = 0
        g.nodes[n]['employable_vnf'] = []

    n_edges = int(lines[1].split()[0])

    # create graph
    print(f"Create graph from file '{file_name}'")
    for i in range(total_lines):
        line = lines[i]
        if ('%' not in line) and ('% list of compute nodes' not in line) and ('DC' not in line):
            sline = line.split()
            # print('Edge info')
            edge_attr = {'bw': float(sline[2]) * resource_scaler,
                         'bw_free': float(sline[2]) * resource_scaler,
                         'delay': float(sline[3])}
            # using node_id as int32
            g.add_edge(int(sline[0]), int(sline[1]), **edge_attr)  # add edges to graph
    # draw graph
    if is_draw:
        nx.draw_networkx(g)

    # add resources to nodes from file
    print(f"Add resources to node from file '{file_name}'")
    for i in range(total_lines):
        line = lines[i]
        if ('%' not in line) and ('% list of compute nodes' not in line) and ('DC' in line):
            sline = line.split()
            # print(sline)
            dc_id = sline[0]
            node_id = int(sline[1])
            g.nodes[node_id]['dc_id'] = dc_id
            g.nodes[node_id]['cpu_cap'] = g.nodes[node_id]['cpu_free'] = float(sline[2]) * resource_scaler
            g.nodes[node_id]['ram_cap'] = g.nodes[node_id]['ram_free'] = float(sline[3]) * resource_scaler
            g.nodes[node_id]['sto_cap'] = g.nodes[node_id]['sto_free'] = float(sline[4]) * resource_scaler

            # list of supporting VNF types
            for vnf_name in sline[6:]:
                g.nodes[node_id]['employable_vnf'].append(vnf_name)

    return g, n_nodes, n_edges
#########################################


#########################################
def build_node_info(g):
    ''' Build a list of named_tuple containing node information

    Parameters:
        g : networkx graph
            GRAPH DATA

    Return:
        node_items : dictionary of named tuples
            INFORMATION OF THE NODES
                ['Node_ID', 'CPU', 'RAM', 'STO', 'employable_vnf'] 
        compute_nodes: binary np.array 
            BINARY LIST OF COMPUTE NODES
    '''
    # create named_tuples of node information
    # Node_Info = namedtuple('Node_Info', ['node_id', 'cpu_cap', 'cpu_free', 'ram_cap', 'ram_free',
    #                                       'sto_cap', 'sto_free', 'employable_vnf'])
    
    node_list = list(g.nodes)
    node_list.sort()
    node_items = []
    compute_nodes = np.zeros(len(node_list))
    # for node_id in node_list:
    #     item = Node_Info(node_id = node_id,
    #                      cpu_cap = g.nodes[node_id]['cpu_cap'],
    #                      cpu_free = g.nodes[node_id]['cpu_free'],
    #                      ram_cap = g.nodes[node_id]['ram_cap'],
    #                      ram_free = g.nodes[node_id]['ram_free'],
    #                      sto_cap = g.nodes[node_id]['sto_cap'],
    #                      sto_free = g.nodes[node_id]['sto_free'],
    #                      employable_vnf=g.nodes[node_id]['employable_vnf'])
    #     node_info[node_id] = item
    for node_id in node_list:
        item = g.nodes[node_id]
        node_item = namedtuple('Node_Info', item.keys())(*item.values())
        node_items.append(node_item._asdict())
        # get binary list of compute node/ Datacenter
        if node_item.dc_id:
            compute_nodes[node_id] = 1

    return node_items, compute_nodes
#########################################


#########################################
def build_edge_info(g):
    ''' Build a list of named_tuple containing edge information

    Parameters:
        g : networkx graph
            GRAPH DATA

    Return:
        edge_info : list of edges
            INFORMATION OF EDGES
                ['src', 'dst', 'bw', 'delay']
    '''
    edge_list = list(g.edges)


#########################################
def build_vnf_support(g, vnf_specs, add_noise=False):
    ''' Build a VNF support matrix

    Parameters:
        g : networkx graph
            GRAPH DATA
        vnf_specs : dict of named_tuple VnF_Info
            DICTIONARY OF VNF_SPECS
        add_noise : bool
            [True] ADD SOME RANDOM NOISE TO AVOID "DEAD NEURON" DURING TRAINING
            [False] DO NOTHING
    Return:
        vnf_support : 2-d float32 {0,1} np.array
            VNF SUPPORT MATRIX
            Each entry vnf_support[i][j]:
                i : node_id
                j : vnf_id
    '''
    print("Build network node-VNF support matrix")
    # get ascending-order list of nodes
    node_list = list(g.nodes)
    node_list.sort()
    n_vnf = len(vnf_specs)
    vnf_names = list(vnf_specs.keys())

    # vnf_id = {'vnf_a':0, 'vnf_b':1, 'vnf_c':2}
    vnf_id = {}
    for name, i in zip(vnf_names, range(n_vnf)):
        vnf_id[name] = i

    # one-hot vnf support matrix
    # each entry (node_id, vnf_id)
    vnf_support = np.zeros((len(node_list), n_vnf), dtype=float)

    for u in node_list:
        if g.nodes[u]['employable_vnf']:
            for vnf_name in g.nodes[u]['employable_vnf']:
                # svnf = vnf_name.split()
                # print(svnf)
                # vnf_id = int(svnf[-1]) #using int32 vnf id
                # vnf_support[u][vnf_id] = 1.0
                vnf_support[u][vnf_id[vnf_name]] = 1.0

    # add noise
    if add_noise:
        vnf_support += np.random.rand(*vnf_support.shape) / 25.0
    return vnf_support
#########################################


#########################################
def build_neighbors(g, add_noise=False):
    ''' Build neighborhood vector for each node in the graph

    Parameters:
        g : networkx graph
            GRAPH DATA
        add_noise : bool
            [True] ADD SOME RANDOM NOISE TO AVOID "DEAD NEURON" DURING TRAINING
            [False] DO NOTHING    
    Return:
        neighbor_mat : 2-d float32 {0,1} np.array
            NODE ADJACENT MATRIX
            Each entry neighbor_mat[i][j] : 
                i : node_ids
                j : node_ids
            Each row neighbor_mat[i] is a one-hot vector of neighbor hoods of i
    '''
    print("Build network nodes adjancy matrix")
    # get ascending-order list of nodes
    sorted_node_list = list(g.nodes)
    sorted_node_list.sort()

    # one-hot adjacency matrix
    # adj_mat = np.zeros((n_nodes, n_nodes))
    adj_mat = nx.linalg.graphmatrix.adjacency_matrix(
        g, nodelist=sorted_node_list)
    adj_mat = adj_mat.toarray().astype(float)

    # Add noise if needed
    if add_noise:
        adj_mat += np.random.rand(*adj_mat.shape) / 25.0
    return adj_mat
#########################################


#########################################
def init_resource(g, add_noise=False):
    ''' Initialize resource matrix for compute nodes from graph g

    Parameters:
        g : networkx graph
            GRAPH DATA
        add_noise : bool
            [True] ADD SOME RANDOM NOISE TO AVOID "DEAD NEURON" DURING TRAINING
            [False] DO NOTHING

    Return:
        resource_matrix : 2-d float np.array
            RESOURCE MATRIX
                resource_mat[i][j]:
                    i : node_id
                    j : resource_type {0:CPU, 1:RAM, 2:STORAGE}
    '''
    print("Init resource to all network nodes")
    # get ascending-order list of nodes
    sorted_node_list = list(g.nodes)
    sorted_node_list.sort()
    n_nodes = len(sorted_node_list)
    n_resource_types = 3  # CPU, RAM, STORAGE

    # resource matrix
    res_mat = np.zeros((n_nodes, n_resource_types), dtype=float)
    for u in sorted_node_list:
        res_mat[u][0] = g.nodes[u]['cpu_cap']
        res_mat[u][1] = g.nodes[u]['ram_cap']
        res_mat[u][2] = g.nodes[u]['sto_cap']

    # add_noise to zero-value entries
    if add_noise:
        for i in sorted_node_list:
            for j in range(n_resource_types):
                if res_mat[i][j] == 0:
                    res_mat[i][j] += np.random.rand() / 25.0
    return res_mat
#########################################


#########################################
# TODO: Fix this function
def get_vnf_specs(vnf_sfc_file):
    ''' Extract the specification of the VNFs and SFCs supported by the network

    Parameters:
        vnf_sfc_file : string
            FULL_PATH TO THE DATA FILE FOR THE VNFs AND SFCs SPECIFICATIONS

    Return:
        n_vnf : int
            NUMBER OF VNF TYPES SUPPORTED BY THE NETWORK
        vnf_specs : dictionary
            DICTIONARY OF VNF SPEC
            vnf_spec[vnf_name] : namedtuple of resource info [CPU, RAM, STORAGE, Proc_delay]
    '''
    # read data_file into of lines
    with open(vnf_sfc_file, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)

    for i in range(total_lines):
        line = lines[i]
        # extract number of VNF types
        if "% number of VNFs" in line:
            sline = line.split()
            n_vnf = int(sline[0])
            vnf_sec_start = i + 1  # the STARTING line index of the VNF spec description
        # the ENDING line index right after the VNF spec description
        elif "% list of SFCs: described using a DAG" in line:
            vnf_sec_end = i
        else:
            pass

    # print(f'the STARTING line index of the VNF spec description {vnf_sec_start}')
    # print(f'the ENDING line index right after the VNF spec description {vnf_sec_end}')

    # Extract VNF resource spec
    file_name = vnf_sfc_file.split(sep='\\')[-1]
    print(f"Extract VNF specs {file_name}")
    Vnf_Info = namedtuple('Vnf_Info', ['vnf_name', 'vnf_id', 'cpu', 'proc_delay'])
    vnf_specs = {}
    for i in range(vnf_sec_start, vnf_sec_end):
        line = lines[i]
        sline = line.split()
        vnf_name = sline[0]
        vnf_item = Vnf_Info(vnf_name=sline[0],
                            vnf_id=int(vnf_name[-1]) - 1,
                            cpu=float(sline[1]),
                            proc_delay=float(sline[2])
                            )
        vnf_specs[vnf_name] = vnf_item._asdict()

    return n_vnf, vnf_specs
#########################################


#########################################
def get_sfc_specs(vnf_sfc_file):
    ''' Extract the specification of the VNFs and SFCs supported by the network

    Parameters:
        vnf_sfc_file : string
            FULL_PATH TO THE DATA FILE FOR THE VNFs AND SFCs SPECIFICATIONS

    Return:
        n_sfc : int
            NUMBER OF SFC TYPES SUPPORTED BY THE NETWORK
        sfc_specs : dictionary
            DICTIONARY OF SFC SPEC
                key : sfc_name
                value : list of vnf_names forming the function chain of sfc_name
    '''
    # read data_file into of lines
    with open(vnf_sfc_file, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)

    for i in range(total_lines):
        line = lines[i]

        # extract number of SFC types
        if "% number of SFCs" in line:
            sline = line.split()
            n_sfc = int(sline[0])
        # the STARTING line index of the SFC spec description
        elif "% list of arcs" in line:
            sfc_sec_start = i + 1
        else:
            pass

    # print(f'the STARTING line index of the SFC spec description {sfc_sec_start}')

    # Extract SFC spec
    file_name = vnf_sfc_file.split(sep='\\')[-1]
    print(f"Extract SFC specs {file_name}")
    sfc_sec_end = total_lines
    sfc_check_list = []
    sfc_specs = {}
    for i in range(sfc_sec_start, sfc_sec_end):
        line = lines[i]
        sline = line.split()
        if "SFC" in line:  # extract SFC_id
            sfc_name = sline[0]
            if sfc_name not in sfc_check_list:
                sfc_check_list.append(sfc_name)
                sfc_specs[sfc_name] = []
        else:  # extract VNF pipeline
            for vnf_name in sline:
                if vnf_name not in sfc_specs[sfc_name]:
                    sfc_specs[sfc_name].append(vnf_name)

    return n_sfc, sfc_specs
#########################################


#########################################
# TODO: Update resource matrix
    # Implement this after done processing the *.tra file
def update_resource(res_mat, req_finish, add_noise=False):
    ''' Update resource matrix for the current time step

    Parameters:
        resource_matrix : 2-d float np.array
            RESOURCE MATRIX
                resource_mat[i][j]:
                    i : node_id
                    j : resource_type {0:CPU, 1:RAM, 2:STORAGE}
        service_done : list 
            LIST OF SERVICE_REQUEST_ID THAT ARE FINISHED AT THE TIME OF UPDATING RESOURCE MATRIX
        add_noise : bool
            [True] ADD SOME RANDOM NOISE TO AVOID "DEAD NEURON" DURING TRAINING
            [False] DO NOTHING

    Return:
        new_res_mat : 2-d float np.array
            UPDATED RESOURCE MATRIX
    '''
    # XXX: Waiting for discussion on how we model the resource dynamics in the network
    pass
#########################################


#########################################
def retrieve_sfc_req_from_txt(req_file):
    ''' Read ALL service requests from the request_file

    Parameters:
        req_file : string
            FULL PATH TO THE FILE RECORDING THE SERVICE REQUESTS

    Return:
        n_req : int
            NUMBER OF SERVICE REQUESTS IN THE FILE
        req_list : dict of namedtuple entries
            A LIST OF SERVICE REQUESTS FROM THE FILE
                REQUIREMENTS OF THE SERVICE REQUEST
                    [id, source, destination, SFC id, bandwidth requirement, delay requirement]
    '''
    print('Read service requests from TEXT file')
    with open(req_file, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)

    req_list = {}

    Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
                                        'bw', 'delay_req',
                                        'arrival_time', 'end_time', 
                                        'RAM_req', 'MEM_req',
                                        'co_id'])

    n_req = int(lines[1].split()[0])
    for line in lines[2:total_lines]:
        sline = line.split()
        req_item = Req_Info(id=int(sline[0]),
                            src=int(sline[1]),
                            dst=int(sline[2]),
                            sfc_id=sline[3],
                            bw=float(sline[4]),
                            delay_req=float(sline[5]),
                            arrival_time=int(sline[6]),
                            end_time=int(sline[7]),
                            co_id=int(sline[8]))
        # req_list.append(req_item)
        req_list[req_item.id] = req_item
    return n_req, req_list
#########################################


#########################################
def retrieve_sfc_req_from_json(req_file):
    ''' Read ALL service requests from the JSON request_file

    Parameters:
        req_file : string
            FULL PATH TO THE json_format FILE RECORDING THE SERVICE REQUESTS

    Return:
        n_req : int
            NUMBER OF SERVICE REQUESTS IN THE FILE
        req_list : dict of namedtuple entries
            A LIST OF SERVICE REQUESTS FROM THE FILE
                REQUIREMENTS OF THE SERVICE REQUEST
                    [id, source, destination, SFC id, bandwidth requirement, delay requirement, start_time, end_time, co_id]
    '''

    """
    Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
                                        'bw', 'delay_req',
                                        'arrival_time', 'end_time', 
                                        'RAM_req', 'MEM_req',
                                        'co_id'])
    """

    print('Read service requests from JSON file')

    with open(req_file, 'r') as f:
        req_data = json.load(f)
        req_data = req_data['requests']

    n_req = len(req_data)
    req_list = {}
    req_order = 0
    for item in req_data:
        req_item = namedtuple('Req_Info', item.keys())(*item.values())
        req_list[req_item.id] = req_item._asdict()
        req_order += 1
    return n_req, req_list
#########################################


#########################################
def retrieve_each_sfc_req_from_text(seq_num, file_name):
    ''' Read EACH service request item from .txt data file

    Parameters:
        seq_num : int
            THE ORDER NUMBER OF THE INQUIRING SFC REQUEST
        file_name : string
            FULL PATH TO DATA FILE

    Return:
        req_item : namedtupled
            INFORMATION OF THE INQUIRED SFC ITEM
    '''
    
    Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
                                        'bw', 'delay_req',
                                        'arrival_time', 'end_time', 
                                        'RAM_req', 'MEM_req',
                                        'co_id'])

    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            if i == seq_num + 2:
                sline = line.split()
                req_item = Req_Info(id=int(sline[0]),
                                    src=int(sline[1]),
                                    dst=int(sline[2]),
                                    sfc_id=sline[3],
                                    bw=float(sline[4]),
                                    delay_req=float(sline[5]),
                                    arrival_time=int(sline[6]),
                                    end_time=int(sline[7]),
                                    co_id=int(sline[8]))
                return req_item
######################################### 

def sfc_req_decode(sfc_req):
    """
    Req_Info = namedtuple('Req_Info', ['id', 'src', 'dst', 'sfc_id',
                                         'bw', 'delay_req',
                                         'arrival_time', 'end_time', 
                                         'RAM_req', 'MEM_req',
                                         'co_id'])
    """
    return namedtuple('Req_Info', sfc_req.keys())(*sfc_req.values())

#########################################


def retrieve_each_sfc_req_from_json(seq_num, req_file):
    ''' Read EACH service request item from .json data file

    Parameters:
        seq_num : int
            THE ORDER NUMBER OF THE INQUIRING SFC REQUEST
        req_file : string
            FULL PATH TO THE json_format FILE RECORDING THE SERVICE REQUESTS

    Return:
        req_item : namedtupled
            INFORMATION OF THE INQUIRED SFC ITEM
    '''
    with open(req_file, 'r') as f:
        req_data = json.load(f)
        req_data = req_data['requests']

    item = req_data[seq_num]
    req_item = namedtuple('Req_Info', item.keys())(*item.values())
    return req_item
#########################################


#########################################
def get_tot_sfc_req(file_name):
    ''' Get the number of SFC requests in the file

    Parameters
    ----------
    file_name : string
        FULL PATH TO DATA FILE

    Returns
    -------
    n_req : int
        TOTAL NUMBER OF SFC REQUESTS IN THE FILE

    '''
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            if "% number of requests" in line:
                sline = line.split()
                return int(sline[0])

#########################################


def link_latency(g, src, dst, show_result=False):
    ''' Calculate the shortest path between nodes i and j in terms of link delay
    '''
    link_lat, link_path = nx.single_source_dijkstra(
        g, src, dst, weight='delay')
    if show_result:
        print(f'{src}-{dst} link latency : {link_lat} ms')
        print(f'{src}-{dst} path : {link_path}')
    return link_lat, link_path
#########################################


#########################################
def link_bw(g, src, dst, show_result=False):
    ''' Calculate the shortest path between nodes i and j in terms of link bandwidth
    '''
    link_bw, link_path = nx.single_source_dijkstra(g, src, dst, weight='bw')
    if show_result:
        print(f'{src}-{dst} link bandwidth : {link_bw} ')
        print(f'{src}-{dst} path : {link_path}')
    return link_bw, link_path
#########################################


#########################################
def booked_req():
    ''' Record the (node, path) of the booked request
        Track the expiration time of the booked requests

     Return:
         expired_req : list
             LIST of requests (id) which are removed from the net due to expired sojourn_time
    '''
    pass
#########################################


#########################################
def init_bw(g):
    ''' Build a matrix of banwdith of network's edges
    '''
    edge_list = list(g.edges)
    n_nodes = g.number_of_nodes()
    bw_mat = np.zeros((n_nodes, n_nodes))

    for edge in edge_list:
        src = edge[0]
        dst = edge[1]
        bw_mat[src, dst] = g.edges[src, dst]['bw']

    # Add super large bw to i-i links (self-link)

    return bw_mat


#########################################
# =============================================================================
# Find runs of consecutive items in an array
# https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065#file-find_runs-py
# =============================================================================
def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


# =============================================================================
# Moving window average
# =============================================================================
def mov_window_avg(data, window_size=1000):
    ''' Calculate moving average values

    Parameters:
        data: list/np.array
            LIST OF DATA
        window_size: int
            THE AVERAGING WINDOW
    Return:
        moving_avg: float list
            MOVING AVERAGE VALUES
    '''
    i = 0
    moving_avg = []
    while i < len(data) - window_size + 1:
        this_window = data[i:i + window_size]
        window_avg = sum(this_window) / window_size
        moving_avg.append(window_avg)
        i += 1
    return moving_avg

# =============================================================================
# Encode numpy.type data before exporting to JSON file
# =============================================================================


def numpy_encoder(object):
    ''' Encode numpy.type data before exporting to JSON file'''
    if isinstance(object, (np.generic, np.ndarray)):
        return object.item()


# =============================================================================
# Calculate CPU, RAM, STORAGE, and BW usage rate
# =============================================================================
def resource_usage_rate(tot_resources_mat, remain_resources_mat, tot_bw, remain_bw_mat):
    '''


    Parameters
    ----------
    tot_resources_mat : np.array
        CPU, RAM, STORAGE RESOURCE CAPACITY
    remain_resources_mat : np.array
        REMAINING CPU, RAM, STORAGE RESOURCES
    tot_bw : float
        TOTAL LINK BW CAPACITY
    remain_bw_mat : np.array
        REMAINING LINK BW

    Returns
    -------
    cpu_usage
    ram_usage
    sto_usage
    bw_usage
    '''
    # calculate the CPU, RAM, STO usage rates per time slot
    tot_cpu, tot_ram, tot_sto = tot_resources_mat
    remain_cpu, remain_ram, remain_sto = remain_resources_mat.sum(axis=0)
    cpu_usage = (tot_cpu - remain_cpu) / tot_cpu
    ram_usage = (tot_ram - remain_ram) / tot_ram
    sto_usage = (tot_sto - remain_sto) / tot_sto
    # calculate the BW usage rates per time slot
    remain_bw = remain_bw_mat.sum()
    bw_usage = (tot_bw - remain_bw) / tot_bw
    return cpu_usage, ram_usage, sto_usage, bw_usage
#########################################


# =============================================================================
# Calculate offered load, real throughput, and request acceptance ratio per time slot
# =============================================================================
def calc_throughput(adm_hist, req_list, all_arrive_req, 
                    serving_req, start_time_slot,
                    sfc_specs, vnf_specs):
    # TODO: calculate CPU, RAM, STO, and BW usage rate as well
    # TODO: fix the bug when req_list is not continuous
    real_tp = []
    perfect_tp = []
    cpu_usage = []
    ram_usage  = []
    sto_usage = []
    
    for time_slot in adm_hist:
        
        # Get all the requests in this time slot
        reqs_per_slot = adm_hist[time_slot]

        # Total perfect_throughput per time slot
        cur_perfect_tp = 0
        for req_item in reqs_per_slot:
            cur_perfect_tp += req_list[req_item]['bw']
        
        # Minus the throughput of expired_requests
        if time_slot > 0:
            cur_perfect_tp += perfect_tp[-1]
            # Minus the expired requests' throughput from the perfect_tp
            remove_perfect_req = []
            for req_id in all_arrive_req:
                if req_list[req_id]['arrival_time'] < time_slot + start_time_slot:
                    if time_slot + start_time_slot == req_list[req_id]['end_time'] + 1:
                        cur_perfect_tp -= req_list[req_id]['bw']
                        remove_perfect_req.append(req_id)
                else:
                    break
            # Remove expired_req from the all_arrive_req list
            for rm_id in remove_perfect_req:
                all_arrive_req.remove(rm_id)
                
        # Append perfect_throughput of the new time slot to the list        
        perfect_tp.append(cur_perfect_tp)
            
        # Total real_throughput per time slot
        cur_real_tp = 0
        for req_item in reqs_per_slot:
            cur_real_tp += reqs_per_slot[req_item] * req_list[req_item]['bw']
        
        # Minus the throughput of expired_requests
        if time_slot > 0:
            cur_real_tp += real_tp[-1]
            # Minus the expired requests' throughput from the real_tp
            remove_real_req = []
            for req_id in serving_req:
                if req_list[req_id]['arrival_time'] < time_slot + start_time_slot:
                    if time_slot + start_time_slot == req_list[req_id]['end_time'] + 1:
                        cur_real_tp -= req_list[req_id]['bw']
                        remove_real_req.append(req_id)
                else:
                    break
            # Remove expired_req from the serving_req list
            for rm_id in remove_real_req:
                serving_req.remove(rm_id)
                
        # Append real_throughput of the new time slot to the list    
        real_tp.append(cur_real_tp)
        
    return perfect_tp, real_tp

#########################################

# =============================================================================
# Acceptance rate per time slot
# =============================================================================
def accept_rate_per_slot(adm_hist):
    # Request acceptance ratio per time slot
    adm_rate_list = []  # list of req accepted rate
    for time_slot in adm_hist:
        reqs_per_slot = adm_hist[time_slot]
        adm_req = 0
        for req_item in reqs_per_slot:
            adm_req += reqs_per_slot[req_item]
        adm_rate = adm_req / len(reqs_per_slot.keys())
        adm_rate_list.append(adm_rate)
    
    return adm_rate_list


#########################################
# ### Main function
# DATA_FOLDER = "SOF_Data_Sets-v04\complete_data_sets"
# topo_file = os.path.join(DATA_FOLDER, "ibm.net")
# g, n_nodes, n_edges= build_net_topo(topo_file, is_draw=False)
# node_info = build_node_info(g)

# vnf_sfc_file = os.path.join(DATA_FOLDER, "sfc_file.sfc")
# n_vnf_types, vnf_specs  = get_vnf_specs(vnf_sfc_file)
# n_sfc_types, sfc_specs = get_sfc_specs(vnf_sfc_file)

# vnf_support_mat = build_vnf_support(g, vnf_specs, add_noise=False)
# # print(vnf_support_mat)

# adj_mat = build_neighbors(g, add_noise=False)
# # print(adj_mat)

# resource_mat = init_resource(g, add_noise=False)
# # print(resource_mat)

# bw_mat = init_bw(g)
# print(bw_mat)

# # req_file = 'traffic.tra'
# # n_req, req_list = retrieve_sfc_req_from_txt(req_file)
# # print(f'Total number of requests: {n_req}')

# # req_file = "..\complete_data_sets\sfc_requests.json"
# # n_req, req_list = retrieve_sfc_req_from_json(req_file)
# # req_txt_file = os.path.join(DATA_FOLDER, "reordered_traffic.txt")
# # n_req, req_list = retrieve_sfc_req_from_txt(req_txt_file)

# # req_json_file = os.path.join(DATA_FOLDER, "reordered_traffic.json")
# # n_req, req_list = retrieve_sfc_req_from_json(req_json_file)
# # req_item = retrieve_each_sfc_req_from_json(seq_num=10, req_file=req_json_file)
