# author: Huy Duong, CRIM
# Note:
# - Dynamic traffic
# - Unbounded Scheme

# import networkx
# import networkx as nx
# import matplotlib.pyplot as plt
import copy
import os
# import networkx
import math

import matplotlib.pyplot as plt
import pandas as pd

import networkx
import networkx as nx
import numpy as np
from networkx import *
# import json
# from numpy import random
from random import seed  # seed the pseudorandom number generator
from random import randint
from simulation import *
from IO import IO
from datetime import datetime
import csv
import geopy.distance
from pyvis.network import Network

import sys


#########################################
def print_graph_to_file(g: networkx.DiGraph, file_name: str):
    f = open(file_name, 'w')

    # print(str(len(g.nodes())) + "\t% number of routers", file=f)
    print(str(len(g.nodes())) + "\t% number of routers", file=f)
    print("% list of nodes", file=f)
    print("% node id, city, state, population", file=f)
    for node_id in g.nodes:
        node = g.nodes[node_id]
        print(node_id, node['city_name'], node['state'].replace(' ', '_'), node['population'],
              node['latitude'],
              node['longitude'],
              file=f)

    print(str(len(g.edges())) + "\t% number of directed links", file=f)
    print("% list of directed links", file=f)
    print("% u, v, bandwidth per direction, delay", file=f)
    for u, v in g.edges():
        print(u, v, str(g[u][v]["bw"]) + " " + str(g[u][v]["delay"]), file=f)
        # print(v, u, str(g[u][v]["bw"]) + " " + str(g[u][v]["delay"]), file=f)

    num_DC = 0  # number of data centers
    for u in g.nodes():
        if len(g.nodes[u]['VNF_list']) > 0:
            num_DC += 1

    info_line = "% list of compute nodes and their resource capacity: " \
                "CPU (#), RAM (GB), storage (GB), # VNFs, VNF list"
    print(info_line, file=f)
    print(num_DC, "% number of data centers", file=f)
    dc_counter = 0
    for u in g.nodes():
        if g.nodes[u]['VNF_list']:
            data_line = "DC" + str(dc_counter) + " " + str(u)
            data_line += " " + str(g.nodes[u]['CPU_cap']) + " " + \
                         str(g.nodes[u]['RAM_cap']) + " " + str(g.nodes[u]['STO_cap'])
            data_line += " " + str(len(g.nodes[u]["VNF_list"]))
            for vnf in g.nodes[u]["VNF_list"]:
                data_line += " " + vnf

            print(data_line, file=f)
            dc_counter += 1

    f.close()
#########################################


#########################################
# this procedure generate completely randomized traffic between all node pairs
def generate_traffic_completely_randomized(g):
    # seed random number generator
    # seed(1)
    # generate some random numbers
    n_nodes = len(g.nodes())
    # number of SFC specifications, it should be read from the SFC file
    # 10 is a random number for the time being
    n_sfc = 10
    max_bw = 10  # maximum bandwidth a connection may require, 10 is a random number for the time being
    # maximum delay a connection may require
    # it should be equal to the minimum link delay of the network, that's why it's 10 according to the network file
    min_delay_req = 10
    max_delay_req = 100  # maximum delay a connection may require, 100 is a random number for the time being

    with open("../complete_data_sets/traffic.tra", 'w') as f:
        num_requests = 1000
        print("% source, destination, SFC's id, bandwidth requirement, delay requirement", file=f)
        print(str(num_requests) + " % number of requests", file=f)
        for i in range(num_requests):
            u = str(randint(0, n_nodes))
            v = str(randint(0, n_nodes))
            while v == u:
                v = str(randint(0, n_nodes))

            sfc_id = "SFC_" + str(randint(0, n_sfc - 1))
            bw = str(randint(1, max_bw))
            delay_req = str(randint(min_delay_req, max_delay_req))

            print(u + ' ' + v + ' ' + sfc_id + " " + bw + " " + delay_req, file=f)


#########################################


#########################################
# After the simulation with unbounded network finished, we create a network according to maximal used resources
def generate_new_network_after_unbound_simulation(simulation, parameters):
    resulted_network = copy.deepcopy(simulation.network)
    range_of_rate_cpu_disk = parameters["expand_rate_CPU_DISK"]
    range_of_rate_ram = parameters["expand_rate_RAM"]
    range_of_rate_bw = parameters['expand_rate_bandwidth']

    for u, v in resulted_network.edges:
        rate = 1 + randint(range_of_rate_bw[0], range_of_rate_bw[1]) / 100.0
        bw = simulation.max_link_used_bandwidth[u][v] * rate
        resulted_network[u][v]['free_bw'] = resulted_network[u][v]['bw'] = round(bw, 2)

    for node in resulted_network.nodes:
        rate = 1 + randint(range_of_rate_cpu_disk[0], range_of_rate_cpu_disk[1]) / 100.0
        max_used_cpu = simulation.max_node_used_cap[node]['CPU']
        resulted_network.nodes[node]['free_CPU'] = resulted_network.nodes[node]['CPU_cap'] = round(max_used_cpu * rate,
                                                                                                   2)

        rate = 1 + randint(range_of_rate_ram[0], range_of_rate_ram[1]) / 100.0
        max_used_ram = simulation.max_node_used_cap[node]['RAM']
        resulted_network.nodes[node]['free_RAM'] = resulted_network.nodes[node]['RAM_cap'] = round(max_used_ram * rate,
                                                                                                   2)

        rate = 1 + randint(range_of_rate_cpu_disk[0], range_of_rate_cpu_disk[1]) / 100.0
        max_used_sto = simulation.max_node_used_cap[node]['STO']
        resulted_network.nodes[node]['free_STO'] = resulted_network.nodes[node]['STO_cap'] = round(max_used_sto * rate,
                                                                                                   2)

    return resulted_network


#########################################


#########################################
# Using an unbounded-capacity network, this procedure generates uniform distributed traffic between all node pairs.
# Given an unbounded-capacity network, requests would be routed on the shortest path wrt delay
def get_delay_of_layered_graph_path(simulation, sfc_request, layered_graph_path):
    delay = 0
    bw = sfc_request['bw']
    sequent_vnfs = simulation.SFC_list[sfc_request['sfc_id']]['sequent_vnfs']

    for i in range(len(layered_graph_path) - 1):
        u, v = layered_graph_path[i], layered_graph_path[i + 1]
        if u[1] == v[1]:
            delay += simulation.network[u[0]][v[0]]['delay']
        else:
            assert u[1] == v[1] - 1, "Not a proper layered graph path"
            processing_vnf = sequent_vnfs[u[1]]
            delay += bw * simulation.SFC_list["vnf_list"][processing_vnf]['PROC_rate']

    return delay


#########################################


# #########################################
# # Reorder the original SFC requests to vary traffic pattern
# def generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation, parameters):
#     rate = parameters["expand_rate"]
#     reordered_sfc_requests = {}
#
#     # id_list is a map of each original request id to a new id
#     id_list = [int(request_id) for request_id in original_sfc_requests.keys()]
#     random.shuffle(id_list)
#
#     # First, reordered SFC requests are the same order with the original SFC requests
#     for request_id in original_sfc_requests:
#         reordered_sfc_requests[request_id] = copy.deepcopy(original_sfc_requests[request_id])
#
#     for e in simulation.processed_events:
#         assert e[1]['type'] in ['new', 'accept', 'terminate'], "Something is wrong with unbounded simulation"
#         if e[1]['type'] == 'accept':
#             r = e[1]['sfc_request']
#             layered_graph_path = e[1]['path']
#             delay = get_delay_of_layered_graph_path(simulation, r, layered_graph_path)
#             assert delay < 1, "delay must be >= 1"
#             new_request = reordered_sfc_requests[r['id']]  # Note that new_request is just an alias
#             new_request['delay_req'] = original_sfc_requests[r['id']]['delay_req'] = delay * 1.2
#             duration = new_request['end_time'] - new_request['arrival_time']
#             exchanged_request_id = id_list[new_request['id']]
#             new_request['arrival_time'] = original_sfc_requests[exchanged_request_id]['arrival_time']
#             new_request['end_time'] = new_request['arrival_time'] + duration
#
#     temp_request_list = [value for key, value in reordered_sfc_requests.items()]
#     reordered_sfc_requests.clear()
#     temp_request_list = sorted(temp_request_list, key=lambda k: k['arrival_time'])
#     average_duration = parameters["average_duration"]
#     fluctuation_period = parameters["fluctuation_period"]
#     offered_load = {}  #
#     for i in range(len(temp_request_list)):
#         request = temp_request_list[i]  # alias of the variable
#         request['co_id'] = request['id']
#         request['id'] = i
#         reordered_sfc_requests[i] = request
#         duration = request['end_time']-request['arrival_time']
#
#         if fluctuation_period <= 0 and i > average_duration:
#             duration = max(int(duration / 4), 2)
#         fluctuation_period -= 1
#         if fluctuation_period == - parameters['fluctuation_period']:
#             fluctuation_period = parameters['fluctuation_period']
#             # dip_level = randint(2, 4)
#
#         for j in range(request['arrival_time'], request['end_time']):
#             offered_load[j] += request['bw']
#
#     time_slot_list = offered_load.keys()
#     offer_load_curve = offered_load.values()
#     plt.plot(time_slot_list, offer_load_curve)
#     plt.show()
#     return reordered_sfc_requests
# #########################################


#########################################
def get_delay_requirement_from_simulation(original_sfc_requests, simulation, parameters):
    delay_req_map = {}
    # rate = 1 + randint(parameters['delay_scaler'][0], parameters['delay_scaler'][1]) / 100.0
    for e in simulation.processed_events:
        assert e[1]['type'] in ['new', 'accept', 'terminate'], "Something is wrong with unbounded simulation"
        if e[1]['type'] == 'accept':
            r = e[1]['sfc_request']
            layered_graph_path = e[1]['path']

            delay = get_delay_of_layered_graph_path(simulation, r, layered_graph_path)
            assert delay < 1, "delay must be >= 1"

            # if the path is 'long', then increase delay requirement
            path = e[1]['path']
            sequent_vnfs = simulation.SFC_list[r["sfc_id"]]['sequent_vnfs']
            if (len(path) - len(sequent_vnfs)) / len(sequent_vnfs) > parameters['short_path_criteria']:
                delay *= 1.2

            delay_req_map[r['id']] = delay

    return delay_req_map


#########################################


#########################################
# Reorder the original SFC requests to vary traffic pattern
def generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation, parameters):
    rate = parameters["expand_rate"]

    # delay_req_map = get_delay_requirement_from_simulation(original_sfc_requests, simulation)

    reordered_sfc_requests = {}

    temp_request_list = list(original_sfc_requests.values())
    random.shuffle(temp_request_list)

    # take the the shuffled order as new id sequence.
    # note that we have to change the arrival time according to the new sequece
    for i in range(len(temp_request_list)):
        r = temp_request_list[i]
        reordered_sfc_requests[i] = copy.deepcopy(r)
        reordered_sfc_requests[i]['id'] = i
        reordered_sfc_requests[i]['co_id'] = r['id']
        duration = reordered_sfc_requests[i]['end_time'] - reordered_sfc_requests[i]['arrival_time']
        reordered_sfc_requests[i]['arrival_time'] = i
        reordered_sfc_requests[i]['end_time'] = i + duration
        reordered_sfc_requests[i]['delay'] = delay_req_map[r['id']] * 1.2

    # add fluctuation to the offered load
    average_duration = parameters["average_duration"]
    fluctuation_period = parameters["fluctuation_period"]
    fluctuated_requests = {}
    id_counter = -1
    for i in reordered_sfc_requests.keys():
        request = copy.deepcopy(reordered_sfc_requests[i])
        duration = request['end_time'] - request['arrival_time']

        if fluctuation_period <= 0 and i > 3 * average_duration and duration < parameters["fluctuation_period"]:
            duration = max(int(duration / 4), 2)
        elif fluctuation_period > 0 and i > 3 * average_duration:  # add requests to increase a peak
            if randint(1, 10) == 1:
                another_request = random.sample(temp_request_list, 1)[0]
                another_request['co_id'] = None
                id_counter += 1
                another_request['id'] = id_counter
                duration_2 = randint(2, parameters[
                    'fluctuation_period'])  # duration should be less than the fluctuation period
                another_request['end_time'] = another_request['arrival_time'] + duration_2
                fluctuated_requests[id_counter] = another_request

        fluctuation_period -= 1
        if fluctuation_period == - parameters['fluctuation_period']:
            fluctuation_period = parameters['fluctuation_period']
            # dip_level = randint(2, 4)

        id_counter += 1
        request['id'] = id_counter
        request['end_time'] = request['arrival_time'] + duration
        fluctuated_requests[request['id']] = request

    plot_offered_load_of_requests(original_sfc_requests, "original")
    plot_offered_load_of_requests(reordered_sfc_requests, "reordered")
    plot_offered_load_of_requests(fluctuated_requests, "fluctuated")
    plt.legend()
    plt.grid()
    plt.show()
    return fluctuated_requests


#########################################


#########################################
def shuffle_request_order(original_sfc_requests, parameters):
    reordered_sfc_requests = {}
    # id_list is a map of each original request id to a new id
    # id_list = [int(request_id) for request_id in original_sfc_requests.keys()]
    id_list = [request['id'] for request in original_sfc_requests]
    random.shuffle(id_list)

    # First, reordered SFC requests are the same order with the original SFC requests
    # for request_id in original_sfc_requests:
    #     reordered_sfc_requests[request_id] = copy.deepcopy(original_sfc_requests[request_id])

    temp_request_list = copy.deepcopy(original_sfc_requests)
    random.shuffle(temp_request_list)

    # get number of requests at each time slot
    num_req_at_slot = {}
    for r in temp_request_list:
        t = r['arrival_time']
        if t not in num_req_at_slot:
            num_req_at_slot[t] = 0
        num_req_at_slot[t] += 1
    # max_T = max(num_req_at_slot.keys())
    # min_T = min(num_req_at_slot.keys())
    reordered_sfc_requests.clear()
    # temp_request_list = sorted(temp_request_list, key=lambda k: k['arrival_time'])
    average_duration = parameters["average_duration"]
    fluctuation_period = None
    if parameters["fluctuation_period"] > 0:
        fluctuation_period = parameters["fluctuation_period"]
    # offered_load = {}  #
    id_counter = -1
    # max_concurrent_req_per_slot = parameters['max_concurrent_request_per_time_slot']
    # num_req_of_this_slot = randint(1, max_concurrent_req_per_slot)
    t = 0
    for r in temp_request_list:
        request = copy.deepcopy(r)  # alias of the variable
        request['co_id'] = request['id']
        id_counter += 1
        request['id'] = id_counter
        duration = request['end_time'] - request['arrival_time']
        # if the number of requests for this time slot is 0, then reset and move to the next slot
        if num_req_at_slot[t] == 0:
            t += 1

        num_req_at_slot[t] -= 1
        request['arrival_time'] = t
        request['end_time'] = request['arrival_time'] + duration
        reordered_sfc_requests[request['id']] = copy.deepcopy(request)

        if fluctuation_period is not None:
            if fluctuation_period <= 0 and t > 3 * average_duration and duration < parameters["fluctuation_period"]:
                duration = max(int(duration / parameters['decrease_rate_during_fluctuation']), 2)
                reordered_sfc_requests[request['id']]['end_time'] = request['arrival_time'] + duration
            elif fluctuation_period > 0 and t > 3 * average_duration:  # add requests to increase a peak
                if randint(1, 10) <= parameters['probability_to_add_request']:
                    another_request = copy.deepcopy(random.sample(temp_request_list, 1)[0])
                    another_request['co_id'] = None
                    id_counter += 1
                    another_request['id'] = id_counter
                    duration_2 = randint(2, parameters['fluctuation_period'])  # duration should be less than the fluctuation period
                    another_request['arrival_time'] = t
                    another_request['end_time'] = another_request['arrival_time'] + duration_2
                    reordered_sfc_requests[id_counter] = copy.deepcopy(another_request)

            fluctuation_period -= 1
            if fluctuation_period == - parameters['fluctuation_period']:
                fluctuation_period = parameters['fluctuation_period']
                # dip_level = randint(2, 4)

        # for j in range(request['arrival_time'], request['end_time']):
        #     if j not in offered_load:
        #         offered_load[j] = 0
        #     offered_load[j] += request['bw']

    # time_slot_list = offered_load.keys()
    # offer_load_curve = offered_load.values()
    # plt.plot(time_slot_list, offer_load_curve)
    # plt.show()
    plt.clf()
    plot_offered_load_of_requests(reordered_sfc_requests.values(), 'fluctuated')
    plot_offered_load_of_requests(original_sfc_requests, 'stable')
    plt.legend()
    plt.grid()
    file_name = "/offer_load_curves_" + str(parameters['#_time_slots']) + ".png"
    plt.savefig(parameters['data_set_folder'] + file_name)
    # plt.show()
    return reordered_sfc_requests


#########################################


#########################################
def plot_offered_load_of_requests(requests, label=None):
    offered_load = {}
    for request in requests:
        for j in range(request['arrival_time'], request['end_time']):
            if j not in offered_load:
                offered_load[j] = 0
            offered_load[j] += request['bw']

    time_slot_list = offered_load.keys()
    offer_load_curve = offered_load.values()
    plt.plot(time_slot_list, offer_load_curve, label=label)


#########################################


#########################################
def build_weighted_random_probabilities_based_on_population_and_distance(network):
    src_dst_pairs = []
    cum_prob_weights = []  # cumulative probability
    for u in network.nodes.keys():
        for v in network.nodes.keys():
            if u == v:
                continue

            # for every node pair whose src != dst
            src_dst_pairs.append((u, v))
            coords_1 = (network.nodes[u]['latitude'], network.nodes[u]['longitude'])
            coords_2 = (network.nodes[v]['latitude'], network.nodes[v]['longitude'])

            dis = geopy.distance.distance(coords_1, coords_2).km
            prob_weight = (network.nodes[u]['population'] * network.nodes[v]['population']) / dis
            cum_prob_weights.append(prob_weight)

    # Change the scale of probability weight using log
    for i in range(len(src_dst_pairs)):
        cum_prob_weights[i] = math.log10(cum_prob_weights[i] + 10)-1

    for i in range(1, len(src_dst_pairs)):
        cum_prob_weights[i] += cum_prob_weights[i - 1]

    print('cum_prob_weights=', cum_prob_weights)

    # pick a random node pair based on its weighted random probability
    # print( random.choices(src_dst_pairs, weights=prob_weights, k=1))
    # src, dst = random.choices(src_dst_pairs, weights=prob_weights, k=1)[0]
    return src_dst_pairs, cum_prob_weights


#########################################


#########################################
def get_random_srt_dst_with_population(network):
    total_pop = 0
    src, dst = None, None

    for node in network.nodes.values():
        total_pop += node['population']

    k = randint(1, total_pop)

    for u in network.nodes:
        k -= network.nodes[u]['population']
        if k <= 0:
            src = u
            break

    k = randint(1, total_pop - network.nodes[src]['population'])
    for u in [x for x in network.nodes if x != src]:
        k -= network.nodes[u]['population']
        if k <= 0:
            dst = u
            break

    return src, dst


#########################################


#########################################
def generate_traffic_with_unbounded_capacity_network(g, sfc_list, parameters, original_traffic_file_name):
    network = copy.deepcopy(g)
    for u in network.nodes:  # make network unbound capacity
        if network.nodes[u]['VNF_list']:
            network.nodes[u]['free_CPU'] = network.nodes[u]['CPU_cap'] = float(1000000.0)
            network.nodes[u]['free_RAM'] = network.nodes[u]['RAM_cap'] = float(1000000.0)
            network.nodes[u]['free_STO'] = network.nodes[u]['STO_cap'] = float(1000000.0)

        for u_1, v in network.edges(u):
            network[u][v]['bw'] = 1000000.0
            network[u][v]['free_bw'] = 1000000.0

    simulation = Simulation(network, sfc_list)
    simulation.simulation_output_file_name = "unbounded_sim.txt"
    simulation.set_routing_algorithm("random_k_shortest")  # "random_k_shortest" or "Dijkstra"
    simulation.parameters['capacity_decision_point'] = parameters['capacity_decision_point']
    original_sfc_requests = {}
    n_sfc = len(sfc_list) - 1
    max_request_bw = 10  # maximal bandwidth that a request can require, 10 is a random number for the time being
    # maximum delay a connection may require,
    # this delay should make sure that delay requirement doesn't prevent provisioning of this request
    max_delay_req = 10 * len(network.edges) * 10
    n_nodes = len(network.nodes)
    simulation_time = parameters["#_time_slots"]  # number of time slots
    # average duration of a request. Note that, for exponential distribution  1/gamma = average
    average_duration = parameters["average_duration"]
    # average number of requests at a time slot
    max_concurrent_per_time_slot = parameters["max_concurrent_request_per_time_slot"]

    request_id = 0
    # fluctuation period is number of consecutive slots,
    # where requests have the same average duration of the random distribution generating requests.
    fluctuation_period = parameters['fluctuation_period']
    dip_level = 2
    src_dst_pairs, cum_prob_weights = build_weighted_random_probabilities_based_on_population_and_distance(network)
    for i in range(simulation_time):  # for each time slot
        for j in range(randint(1, max_concurrent_per_time_slot)):  # j concurrent requests at time slot i
            u, v = None, None
            if parameters['uniform_distribution'] is True:
                u = randint(0, n_nodes - 1)  # random source
                v = randint(0, n_nodes - 1)  # random destination
                while v == u:  # destination has to differ from source
                    v = randint(0, n_nodes - 1)
            else:
                u, v = random.choices(src_dst_pairs, cum_weights=cum_prob_weights, k=1)[0]

            sfc_id = "SFC_" + str(randint(0, n_sfc - 1))
            bw = randint(1, max_request_bw)  # uniform distribution of bandwidth

            # ram_req = parameters['RAM_list'][randint(0, len(parameters['RAM_list'])-1)]
            # delay_req = str(randint(min_delay_req, max_delay_req))
            # sto_req = parameters['Storage_list'][bw-1]
            # sto_req = parameters['Storage_list'][randint(0, len(parameters['Storage_list'])-1)]
            # duration of request has an geometrical distribution
            duration = int(np.random.geometric(1 / average_duration))

            # if fluctuation_period <= 0 and i > average_duration:
            #     duration = max(int(duration / dip_level), 2)
            # fluctuation_period -= 1
            # if fluctuation_period == - parameters['fluctuation_period']:
            #     fluctuation_period = parameters['fluctuation_period']
            #     dip_level = randint(2, 4)

            original_sfc_requests[request_id] = {'id': request_id, 'source': u, 'destination': v, 'sfc_id': sfc_id,
                                                 'bw': bw, 'delay_req': max_delay_req,
                                                 'arrival_time': i, 'end_time': i + duration}

            ram_req, sto_req, cpu_req = None, None, None

            if parameters['RAM_STO_consumption_scheme'] == 'varied_by_request':
                ram_req = parameters['RAM_list'][randint(0, len(parameters['RAM_list']) - 1)]
                sto_req = parameters['STO_list'][randint(0, len(parameters['STO_list']) - 1)]

            for vnf in sfc_list[sfc_id]['sequent_vnfs']:
                cpu_req = simulation.get_cpu_rate_of_vnf(vnf) * bw

                if parameters['RAM_STO_consumption_scheme'] == 'fixed_consumption':
                    ram_req = sfc_list['vnf_list'][vnf]['RAM_rate']
                    sto_req = sfc_list['vnf_list'][vnf]['STO_rate']

                assert ram_req is not None, "RAM_Storage_consumption_scheme is invalid: " \
                                            + parameters['RAM_Storage_consumption_scheme']

                original_sfc_requests[request_id][vnf] = {'CPU_req': cpu_req, 'RAM_req': ram_req, 'STO_req': sto_req}

            event = {"time_point": i, "type": 'new', 'sfc_request': original_sfc_requests[request_id]}
            simulation.add_event(event)
            request_id += 1

    print_traffic_distribution_to_file(network, requests=original_sfc_requests.values(),
                                       file_name=original_traffic_file_name + '_request_dist.png')

    simulation.run(save_network_stats=True, save_processed_events=True, print_simulation_to_file=True,
                   save_curves=False)

    delay_req_map = get_delay_requirement_from_simulation(original_sfc_requests, simulation, parameters)
    rate = 1 + randint(parameters['delay_scaler'][0], parameters['delay_scaler'][1]) / 100.0
    for request_id in original_sfc_requests.keys():
        original_sfc_requests[request_id]['delay_req'] = delay_req_map[request_id] * rate

    resulted_network = generate_new_network_after_unbound_simulation(simulation, parameters)
    # print('Shuffling requests ...')
    # resulted_sfc_requests = generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation,
    #                                                                             parameters)
    # return original_sfc_requests, resulted_sfc_requests, resulted_network, simulation
    return original_sfc_requests, resulted_network, simulation


#########################################

#########################################
# In the varied-by-request scheme, each SFC request requires the same amount of RAM (resp. STO) for all of its VNFs
# For CPU consumption of a VNF in one request, it is the production of the request's required bw and the VNF's CPU rate
#########################################
def generate_SFC_file_for_varied_by_request_consumption_scheme(sfc_file_name: str, g, parameters):
    f = open(sfc_file_name, 'w')
    vnf_list = ['vnf_1', 'vnf_2', 'vnf_3', 'vnf_4']
    max_number_SFCs = 10  # 3 is a random number for the time being
    max_CPU_req = 10  # 10 is a random number for the time being
    # max_RAM_req = 10  # 10 is a random number for the time being
    # max_MEM_req = 10  # 10 is a random number for the time being
    max_processing_delay = 5  # 5 (ms) is a random number, so that sum of VNF processing delays is < 0.1 second

    info_line = "% list of NFVs and their CPU rate per bw. unit and processing delay (s)"
    print(info_line, file=f)
    print(str(len(vnf_list)) + " % number of VNFs", file=f)

    # for i in range(0, max_number_VNFs):
    for vnf_name in vnf_list:
        ram_req = 0
        sto_req = 0
        line1 = vnf_name + " " + str(randint(1, max_CPU_req)) + " " + str(ram_req) + " " + str(sto_req) \
                + " " + str(randint(1, max_processing_delay) / 1000)
        print(line1, file=f)

    info_line = "% list of SFCs: described using a DAG"
    print(info_line, file=f)
    print(str(max_number_SFCs) + " % number of SFCs", file=f)
    print("% SFC id, number of arcs of this SFC", file=f)
    print("% list of arcs", file=f)

    for sfc_counter in range(0, max_number_SFCs):
        num_vnf_pairs = randint(1, len(vnf_list) - 1)  # a random number of VNFs
        line2 = "SFC_" + str(sfc_counter) + " " + str(num_vnf_pairs)
        print(line2, file=f)
        vnf_list_of_sfc = []
        for i in range(num_vnf_pairs + 1):  # build a SFC with a random array of VNFs
            vnf_index = randint(0, len(vnf_list) - 1)
            while vnf_list[vnf_index] in vnf_list_of_sfc:
                vnf_index = randint(0, len(vnf_list) - 1)
            vnf_list_of_sfc.append(vnf_list[vnf_index])

        for i in range(len(vnf_list_of_sfc) - 1):
            line3 = vnf_list_of_sfc[i] + " " + vnf_list_of_sfc[i + 1]
            print(line3, file=f)

    f.close()


#########################################

#########################################
# In the fixed-consumption scheme, for each VNF, it requires a given RAM amount and as well as a given STO amount
# For CPU consumption of a VNF in one request, it is the production of the request's required bw and the VNF's CPU rate
#########################################
def generate_SFC_file_for_fixed_consumption_scheme(sfc_file_name: str, g, parameters):
    f = open(sfc_file_name, 'w')
    vnf_list = ['vnf_1', 'vnf_2', 'vnf_3', 'vnf_4']
    max_number_SFCs = 10  # 3 is a random number for the time being
    max_CPU_req = 10  # 10 is a random number for the time being
    # max_RAM_req = 10  # 10 is a random number for the time being
    # max_MEM_req = 10  # 10 is a random number for the time being
    max_processing_delay = 5  # 5 (ms) is a random number, so that sum of VNF processing delays is < 0.1 second

    info_line = "% list of NFVs and their CPU rate per bw. unit, RAM req., STO req., and processing delay (s)"
    print(info_line, file=f)
    print(str(len(vnf_list)) + " % number of VNFs", file=f)

    # for i in range(0, max_number_VNFs):
    for vnf_name in vnf_list:
        # vnf_list.append("vnf_" + str(i))
        # line = vnf_name + " " + str(randint(1, max_CPU_req)) + " " + str(randint(1, max_RAM_req))
        # line += " " + str(randint(1, max_MEM_req)) + " " + str(randint(1, max_processing_delay) / 1000)
        ram_req = parameters['RAM_list'][randint(0, len(parameters['RAM_list']) - 1)]
        sto_req = parameters['STO_list'][randint(0, len(parameters['STO_list']) - 1)]
        line1 = vnf_name + " " + str(randint(1, max_CPU_req)) + " " + str(ram_req) + " " + str(sto_req) \
                + " " + str(randint(1, max_processing_delay) / 1000)
        print(line1, file=f)

    info_line = "% list of SFCs: described using a DAG"
    print(info_line, file=f)
    print(str(max_number_SFCs) + " % number of SFCs", file=f)
    print("% SFC id, number of arcs of this SFC", file=f)
    print("% list of arcs", file=f)

    for sfc_counter in range(0, max_number_SFCs):
        num_vnf_pairs = randint(1, len(vnf_list) - 1)  # a random number of VNFs
        line2 = "SFC_" + str(sfc_counter) + " " + str(num_vnf_pairs)
        print(line2, file=f)
        vnf_list_of_sfc = []
        for i in range(num_vnf_pairs + 1):  # build a SFC with a random array of VNFs
            vnf_index = randint(0, len(vnf_list) - 1)
            while vnf_list[vnf_index] in vnf_list_of_sfc:
                vnf_index = randint(0, len(vnf_list) - 1)
            vnf_list_of_sfc.append(vnf_list[vnf_index])

        for i in range(len(vnf_list_of_sfc) - 1):
            line3 = vnf_list_of_sfc[i] + " " + vnf_list_of_sfc[i + 1]
            print(line3, file=f)

    f.close()


#########################################


#########################################
def read_graph_from_csv_file(node_list_file_name, link_list_file_name, with_pop):
    node_to_id_map = {}
    links = []
    raw_g = nx.DiGraph()

    # read the list of nodes
    with open(node_list_file_name, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                # from each line, we take id and name (and population if possible) of the city
                city_id, city_name, state = row[0], row[1], row[2]
                raw_g.add_node(city_id)
                raw_g.nodes[city_id]['city_name'] = city_name
                raw_g.nodes[city_id]['city_id'] = city_id
                raw_g.nodes[city_id]['state'] = state
                if with_pop:
                    raw_g.nodes[city_id]['population'] = row[5]
                    raw_g.nodes[city_id]['latitude'] = row[3]
                    raw_g.nodes[city_id]['longitude'] = row[4]

                node_to_id_map[city_name] = city_id

            line_count += 1

    # read edge list
    min_distance = sys.maxsize
    with open(link_list_file_name, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                # from each line, we take u, v, and distance between u-v
                u_name, v_name, dis = row[0], row[1], float(row[2])
                u_id, v_id = node_to_id_map[u_name], node_to_id_map[v_name]
                raw_g.add_edge(u_id, v_id, distance=dis)
                raw_g.add_edge(v_id, u_id, distance=dis)
                min_distance = min(min_distance, dis)
                # nodes.append({'head': row[0], 'tail': row[1], 'distance': })

            line_count += 1

    return raw_g


#########################################


#########################################
# get the shortest edge of a graph
def get_min_distance(g):
    min_distance = sys.maxsize

    for e in g.edges:
        print(e[0], e[1], g[e[0]][e[1]])
        min_distance = min(min_distance, g[e[0]][e[1]]['distance'])

    return min_distance


#########################################


#########################################
# Remove all nodes whose degree of 2.
# Then reindexing node ids
def remove_degree_2_nodes(original_graph):
    g = copy.deepcopy(original_graph)

    removed_node = -1  # a dummy number
    while removed_node is not None:
        removed_node = None
        node_list = list(g.nodes.keys())
        for node in node_list:  # looking for a node whose degree of 2
            if g.out_degree(node) == 2:
                e1 = list(g.out_edges(node))[0]
                e2 = list(g.out_edges(node))[1]
                v1, v2 = e1[1], e2[1]
                g.add_edge(v1, v2, distance=g[v1][node]['distance'] + g[node][v2]['distance'])
                g.add_edge(v2, v1, distance=g[v2][node]['distance'] + g[node][v1]['distance'])
                g.remove_node(node)
                removed_node = node
                break

    return g


#########################################


#########################################
def preprocess_raw_graph(raw_g):
    # pyvis_net = Network('1000px', '1000px')
    # pyvis_net.from_nx(raw_g)
    # pyvis_net.show('original_graph.html')

    no_degree_2_graph = remove_degree_2_nodes(raw_g)

    # pyvis_net_2 = Network('1000px', '1000px')
    # pyvis_net_2.from_nx(no_degree_2_graph)
    # pyvis_net_2.show('no_degree_2_graph.html')

    # proper_g = networkx.DiGraph()

    # node capacity modification
    for node in no_degree_2_graph.nodes:
        no_degree_2_graph.nodes[node]['RAM_cap'] = sys.maxsize
        no_degree_2_graph.nodes[node]['CPU_cap'] = sys.maxsize
        no_degree_2_graph.nodes[node]['STO_cap'] = sys.maxsize
        # g.nodes[node]['cap'] = 0
        no_degree_2_graph.nodes[node]['VNF_list'] = []
        no_degree_2_graph.nodes[node]['color'] = '#00ff1e'  # green

    # vnf_node_list = ['2', '3', '5', '7', '12', '13', '15', '17']
    # employable_vnfs_node_list = {"vnf_1": ['3', '5', '12', '13', '15'],
    #                              "vnf_2": ['3', '7', '12', '15', '17'],
    #                              "vnf_3": ['2', '5', '7', '12', '13', '17'],
    #                              "vnf_4": ['2', '5', '12', '15', '17']}

    # vnf_node_list = ['1', '5', '8', '10', '14', '17', '20', '40', '41', '50', '51', '53', '54', '57', '58', '63',
    #                 '66', '67', '69', '70']

    # vnf_node_list = random.sample(list(no_degree_2_graph.nodes.keys()), 10)
    vnf_node_list = ['61', '32', '18', '28', '19', '38', '55', '44', '39', '72']
    vnf_list = ['vnf_1', 'vnf_2', 'vnf_3', 'vnf_4']
    # employable_vnfs_node_list = {}
    # for vnf in vnf_list:
    #     employable_vnfs_node_list[vnf] = []  # random.sample(vnf_node_list, 10)
    #
    # for vnf_name, node_list in employable_vnfs_node_list.items():
    #     for node in node_list:
    #         no_degree_2_graph.nodes[node]["VNF_list"].append(vnf_name)
    #         no_degree_2_graph.nodes[node]['color'] = '#dd4b39'  # red

    for node in vnf_node_list:
        k = random.randint(2, 4)
        no_degree_2_graph.nodes[node]["VNF_list"] = random.sample(vnf_list, k)
        no_degree_2_graph.nodes[node]['color'] = '#dd4b39'  # red
        print(node, ' ', ': ', no_degree_2_graph.nodes[node]["VNF_list"])
        no_degree_2_graph.nodes[node]['RAM_cap'] = sys.maxsize
        no_degree_2_graph.nodes[node]['CPU_cap'] = sys.maxsize
        no_degree_2_graph.nodes[node]['STO_cap'] = sys.maxsize

    # for node in vnf_a_node_list:
    #    g.nodes[node]['cap'] = 2000

    # link capacity modification
    min_distance = get_min_distance(no_degree_2_graph)
    for u, v in no_degree_2_graph.edges():
        # Note that g is a multi-directed graph, so it has the third index to address an edge
        # Since we haven't introduce DC links, we only have communication links so far
        no_degree_2_graph[u][v]['bw'] = sys.maxsize  # Gbps
        no_degree_2_graph[u][v]['delay'] = 0.001 * no_degree_2_graph[u][v]['distance'] / min_distance  # 0.001 second
        no_degree_2_graph[u][v]['link_type'] = 'communication'

    # relabel nodes so that they are indexed from 0
    relabeled_g = no_degree_2_graph.copy()
    index = -1
    node_mapping = {}
    for node in relabeled_g.nodes:
        index += 1
        node_mapping[node] = index

    relabeled_g = nx.relabel_nodes(relabeled_g, node_mapping)

    return relabeled_g


#########################################
def read_graph_from_graphml_file(file_name):
    undirected_g = nx.read_graphml(file_name)
    g = undirected_g.to_directed()
    # networkx.draw_networkx(g)
    # # Set margins for the axes so that nodes aren't clipped
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()

    # node capacity modification
    for node in g.nodes:
        g.nodes[node]['RAM_cap'] = 0
        g.nodes[node]['CPU_cap'] = 0
        g.nodes[node]['STO_cap'] = 0
        # g.nodes[node]['cap'] = 0
        g.nodes[node]['VNF_list'] = []

    vnf_node_list = ['2', '3', '5', '7', '12', '13', '15', '17']
    employable_vnfs_node_list = {"vnf_1": ['3', '5', '12', '13', '15'],
                                 "vnf_2": ['3', '7', '12', '15', '17'],
                                 "vnf_3": ['2', '5', '7', '12', '13', '17'],
                                 "vnf_4": ['2', '5', '12', '15', '17']}
    for node in vnf_node_list:
        g.nodes[node]['RAM_cap'] = 0
        g.nodes[node]['CPU_cap'] = 0
        g.nodes[node]['STO_cap'] = 0

    for vnf_name, node_list in employable_vnfs_node_list.items():
        for node in node_list:
            g.nodes[node]["VNF_list"].append(vnf_name)

    # for node in vnf_a_node_list:
    #    g.nodes[node]['cap'] = 2000

    # link capacity modification
    for u, v in g.edges():
        g[u][v]['bw'] = 0  # Gbps
        g[u][v]['delay'] = 0.01  # 0.01 second
        g[u][v].pop("LinkType", None)
        g[u][v].pop("LinkLabel", None)
        g[u][v].pop("LinkNote", None)
        g[u][v].pop("key", None)

    return g


#########################################


# #########################################
# # procedure to read network from text file
# def read_graph_from_text_file(network_file_name):
#     file = open(network_file_name, 'r')
#
#     line_list = file.readlines()
#     num_nodes = int(line_list[0].split()[0])
#     num_links = int(line_list[1].split()[0])
#
#     # initialize graph's nodes
#     network = networkx.DiGraph()
#     network.add_nodes_from([x for x in range(num_nodes)])
#     for u in network.nodes:
#         network.nodes[u]["VNF_list"] = []
#     # read edge list
#     for i in range(4, 4 + num_links):
#         u, v, bw, delay = [float(x) for x in line_list[i].split()]  # full-duplex edge
#         free_cap = bw
#         network.add_edge(int(u), int(v), bw=bw, delay=delay, free_bw=free_cap)  # first direction
#         # network.add_edge(v, u, bw=bw, delay=delay, free_bw=free_cap)  # second direction
#
#     # read nodes whose deployable VNFs
#     # formation: DC's id, connected router, CPU (#), RAM (GB), storage (GB), # VNFs, VNF list
#     num_data_centers = int(line_list[5 + num_links].split()[0])
#     start_line_of_DC_list = 6 + num_links
#     for i in range(start_line_of_DC_list, start_line_of_DC_list + num_data_centers):
#         dc_id, v, cpu, ram, disk, num_vnf = line_list[i].split()[0:6]
#         num_vnf = int(num_vnf)
#         v = int(v)
#         network.nodes[v]["DC_ID"] = dc_id
#         network.nodes[v]["free_CPU"] = network.nodes[v]["CPU_cap"] = float(cpu)
#         network.nodes[v]["free_RAM"] = network.nodes[v]["RAM_cap"] = float(ram)
#         network.nodes[v]["free_MEM"] = network.nodes[v]["MEM_cap"] = float(disk)
#         network.nodes[v]["VNF_list"] = line_list[i].split()[6:6 + num_vnf]
#
#     file.close()
#     return network
# #########################################


# #########################################
# def read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc):
#     file = open(sfc_file_name, 'r')
#     lines = file.readlines()
#
#     sfc_list = {'vnf_list': {}}
#     num_vnfs = int(lines[1].split()[0])
#     for i in range(2, num_vnfs + 2):
#         vnf_id, cpu_rate, ram_rate, mem_rate, proc_rate = lines[i].split()[0:5]
#         sfc_list['vnf_list'][vnf_id] = {'CPU_rate': int(cpu_rate), 'RAM_rate': int(ram_rate),
#                                         'MEM_rate': int(mem_rate), 'PROC_rate': float(proc_rate)}
#
#     num_sfcs = int(lines[num_vnfs + 3].split()[0])
#     start_line_of_sfc_list = num_vnfs + 6
#     j = start_line_of_sfc_list
#     for i in range(num_sfcs):
#         sfc_id, num_vnfs_in_sfc = lines[j].split()[0:2]
#         j += 1
#         num_vnfs_in_sfc = int(num_vnfs_in_sfc)
#         sfc_list[sfc_id] = {'SFC_ID': sfc_id, 'relations': {}}
#         for k in range(0, num_vnfs_in_sfc):
#             pre_vnf, succeeding_vnf = lines[j].split()[0:2]
#             j += 1
#             sfc_list[sfc_id]['relations'][pre_vnf] = succeeding_vnf
#
#         if is_sequent_sfc:  # if each SFC is a sequence of VNFs, then we build a ordered list
#             temp_relations = copy.copy(sfc_list[sfc_id]['relations'])
#             first_vnf_marker = [x for x in temp_relations.keys()]
#             for pre_vnf, succeeding_vnf in temp_relations.items():
#                 if succeeding_vnf in first_vnf_marker:
#                     first_vnf_marker.remove(succeeding_vnf)
#
#             assert len(first_vnf_marker) == 1, "relations are inconsistent"
#
#             v = first_vnf_marker[0]
#             sfc_list[sfc_id]['sequent_vnfs'] = [v]
#             while len(temp_relations) > 0:
#                 sfc_list[sfc_id]['sequent_vnfs'].append(temp_relations[v])
#                 vv = temp_relations[v]
#                 temp_relations.pop(v)
#                 v = vv
#
#     file.close()
#     return sfc_list
# #########################################


#########################################
def check_generated_traffic_with_unbounded_capacity_network(network, sfc_list,
                                                            sfc_requests, unbounded_simulation):
    verifying_simulation = Simulation(network, sfc_list=sfc_list)
    for sfc_request in sfc_requests:
        event = {"time_point": sfc_request["arrival_time"], "type": 'new', 'sfc_request': sfc_request}
        verifying_simulation.add_event(event)

    verifying_simulation.set_routing_algorithm("RCSP")
    verifying_simulation.simulation_output_file_name = "CONUS75_verifying_sim.txt"
    verifying_simulation.run(save_network_stats=True, save_processed_events=True, print_simulation_to_file=True,
                             save_curves=True)

    # for edge in simulation.network.edges.data():
    #     assert edge['free_cap'] == 0, "free capacity must be 0."


#########################################


#########################################
def print_traffic_to_file(traffic_file_name, sfc_requests, parameters):
    data = {'requests': []}

    if parameters is not None:
        for para, value in parameters.items():
            data[para] = value

    for request_id, sfc_request in sfc_requests.items():
        data['requests'].append(sfc_request)

    file = open(traffic_file_name, 'w')
    json.dump(data, file, indent=2)
    file.close()

#########################################


#########################################
def read_traffic_from_text_file(traffic_file_name):
    file = open(traffic_file_name, 'r')
    data = json.load(file)
    sfc_requests = data['requests']
    # print(sfc_requests)
    return sfc_requests


#########################################


#########################################
def plot_network(netx_network):
    # pyvis_net = Network()
    # populates the nodes and edges data structures
    # pyvis_net.from_nx(netx_network)
    # pyvis_net.show('nx.html')

    # pyvis_net = Network('1000px', '1000px')
    colored_net = networkx.DiGraph()
    labels_dict = {}
    # color_map = {}
    for node in netx_network.nodes:
        if 'VNF_list' in netx_network.nodes[node] and len(netx_network.nodes[node]['VNF_list']) > 0:
            # color_map[node] = 'red'
            label = [str(node) + ": ", netx_network.nodes[node]['VNF_list']]
            labels_dict[node] = label
            colored_net.add_node(node, label=label)
        else:
            # color_map[node] = 'green'
            label = str(node)
            labels_dict[node] = label
            colored_net.add_node(node, label=label)

    for u, v in netx_network.edges:
        colored_net.add_edge(u, v, weight=netx_network[u][v]['delay'])

    pos = nx.planar_layout(colored_net)

    nx.draw(colored_net, pos, labels=labels_dict, with_labels=True)
    plt.show()
    # pyvis_net.show('nx.html')

    print("Finish plotting!")
    # networkx.draw_networkx(network, with_labels=True, labels=labels_dict)
    # # Set margins for the axes so that nodes aren't clipped
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()


#########################################


#########################################
def test_shuffling_function():
    parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 2000, '#_time_slots': 20_000,
                  'expand_rate': [-10, 20],
                  'RAM_list': [8, 16, 32], 'STO_list': [20, 40, 100, 200],
                  'RAM_STO_consumption_scheme': 'varied_by_request',
                  # options: ['fixed_consumption', 'varied_by_request']
                  # if 'print_support_files' parameter is True then generator would
                  # generate several supporting files which are not necessary
                  'print_support_files': True,
                  'fluctuation_period': 1000,
                  'sfc_file_name': 'CONUS36_sfc_file.sfc'}
    data_set_folder = '../complete_data_sets/' + parameters['RAM_STO_consumption_scheme']
    sfc_file_name = data_set_folder + '/' + parameters['sfc_file_name']
    simulation_file = 'unbounded_sim.txt'
    events = IO.read_events_from_simulation(simulation_file)

    raw_preprocessed_network_file_name = "../raw_topologies/preprocessed_CONUS_75.net"
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name, with_population=True)
    # read SFC file
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)
    num_con_current = str(parameters["max_concurrent_request_per_time_slot"])
    num_time_slots = str(parameters["#_time_slots"])
    original_traffic_file_name = data_set_folder + '/CONUS75_original_traffic_' + num_time_slots + '_slots_' + num_con_current + '_con.tra'
    original_request_list = IO.read_traffic_from_json_file(original_traffic_file_name)
    original_requests = {}
    for r in original_request_list:
        original_requests[r['id']] = r

    fluctuated_requests = generate_sfc_requests_from_unbounded_delay_requests(original_requests, events, parameters)
    exit(0)
#########################################


#########################################
def generate_shuffled_traffic(original_traffic_file_name, parameters):
    original_sfc_requests = IO.read_traffic_from_json_file(original_traffic_file_name)
    print('Shuffling requests ...')
    resulted_sfc_requests = generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation,
                                                                                parameters)


#########################################


#########################################
def print_traffic_distribution_to_file(network, requests, file_name):
    number_requests_map = {}
    for request in requests:
        s, d = request['source'], request['destination']
        if s not in number_requests_map:
            number_requests_map[s] = {}

        if d not in number_requests_map[s]:
            number_requests_map[s][d] = 0

        number_requests_map[s][d] += 1

    print('number_requests_map:', number_requests_map)

    n_nodes = network.number_of_nodes()
    data = np.zeros((n_nodes, n_nodes))
    for s in number_requests_map:
        for d in number_requests_map[s]:
            data[s][d] = number_requests_map[s][d]

    # data = pd.DataFrame.from_dict(number_requests_map, orient='columns', dtype=None)
    # print(data)
    # s = 0
    # for d1 in data:
    #     for pop in data[d1]:
    #         s += pop

    # plt.figure(0)
    plt.clf()
    plt.imshow(data)
    # plt.title("2-D Heat Map")
    plt.colorbar()
    plt.savefig(file_name)
    print('s =', s)
#########################################


#########################################
def main():
    print("Hello World!")
    # set up parameters
    # the original is 
    # 'max_concurrent_request_per_time_slot': 1
    # '#_time_slots': 20_000
    parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 2000, '#_time_slots': 20_000,
                  'expand_rate_CPU_DISK': [-10, 5],
                  'expand_rate_RAM': [-10, 0],
                  'expand_rate_bandwidth': [-20, 0],
                  'delay_scaler': [20, 20],
                  'RAM_list': [8, 16, 32], 'STO_list': [20, 40, 100, 200],
                  'RAM_STO_consumption_scheme': 'varied_by_request',
                  # options: ['fixed_consumption', 'varied_by_request']
                  # if 'print_support_files' parameter is True then generator would
                  # generate several supporting files which are not necessary
                  'print_support_files': True,
                  'fluctuation_period': 1000,
                  'sfc_file_name': 'CONUS36_sfc_file.sfc',
                  'generate_original_traffic': True,
                  'generate_reordered_traffic': True,
                  'uniform_distribution': False,
                  'decrease_rate_during_fluctuation': 5,
                  'probability_to_add_request': 4,
                  'seed': 3,
                  'short_path_criteria': 4,
                  'capacity_decision_point': 20000,
                  'data_set_folder': '../complete_data_sets/' + 'Data_18'}

    seed(parameters['seed'])  # seed of random generator
    random.seed(parameters['seed'])

    # test_shuffling_function()

    # graphml_file_name = "../raw_topologies/ibm.graphml.xml"  # read graphml file
    # g = read_graph_from_graphml_file(graphml_file_name)

    node_list_file_name = '../raw_topologies/preprocessed_CONUS_75_node_list.csv'
    link_list_file_name = '../raw_topologies/preprocessed_CONUS_75_link_list.csv'
    g = read_graph_from_csv_file(node_list_file_name, link_list_file_name, with_pop=True)

    preprocessed_g = preprocess_raw_graph(g)
    # plot_network(preprocessed_g)

    # change raw network to our desired format
    raw_preprocessed_network_file_name = "../raw_topologies/preprocessed_CONUS_75.net"
    print_graph_to_file(preprocessed_g, raw_preprocessed_network_file_name)

    # # data_set_folder = '../complete_data_sets/' + parameters['RAM_STO_consumption_scheme']
    data_set_folder = parameters['data_set_folder']

    try:
        os.mkdir(data_set_folder)
    except OSError as error:
        print('Folder existed')

    # generating SFC file
    sfc_file_name = data_set_folder + '/' + parameters['sfc_file_name']
    if parameters['RAM_STO_consumption_scheme'] == 'fixed_consumption':
        # generate_SFC_file_for_fixed_consumption_scheme(sfc_file_name, g, parameters)
        pass
    elif parameters['RAM_STO_consumption_scheme'] == 'varied_by_request':
        generate_SFC_file_for_varied_by_request_consumption_scheme(sfc_file_name, g, parameters)
    else:
        assert False, '\'RAM_STO_consumption_scheme\' is invalid: ' + parameters['RAM_STO_consumption_scheme']

    # read graph file
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name, with_population=True)
    # plot_network(network)
    # print(network.edges)

    # read SFC file
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)

    num_con_current = str(parameters["max_concurrent_request_per_time_slot"])
    num_time_slots = str(parameters["#_time_slots"])
    original_traffic_file_name = data_set_folder + (
        '/uniform' if parameters['uniform_distribution'] else '/non_uniform')
    original_traffic_file_name += '_CONUS36_original_traffic_' + num_time_slots + '_slots_' + num_con_current + '_con.tra'

    if parameters['generate_original_traffic']:
        # generating traffic for unbounded capacity network
        print("Generating original data ... ", datetime.now().time())
        network.graph["is_unbounded"] = True
        original_sfc_requests, resulted_network, unbounded_simulation \
            = generate_traffic_with_unbounded_capacity_network(network, sfc_list, parameters, original_traffic_file_name)
        resulted_network.graph["is_unbounded"] = False

        network_file_name = data_set_folder + ('/uniform' if parameters['uniform_distribution'] else '/non_uniform') \
                            + 'CONUS36_' + num_time_slots + '_slots_' + num_con_current + '_con.net'
        print_graph_to_file(resulted_network, network_file_name)

        print_traffic_to_file(original_traffic_file_name, original_sfc_requests, parameters)

        plot_offered_load_of_requests(original_sfc_requests.values(), 'original_traffic')
        # plt.legend()
        # plt.grid()
        # plt.show()

    # reuse the generated data to run simulation again to see how algorithm and network behave
    # print('Checking originally generated requests ... ', datetime.now().time())
    # imported_sfc_requests = read_traffic_from_text_file(original_traffic_file_name)
    # check_generated_traffic_with_unbounded_capacity_network(resulted_network, sfc_list,
    #                                                        imported_sfc_requests, unbounded_simulation)

    if parameters['generate_reordered_traffic']:
        print("Generating fluctuated traffic ... ")
        original_sfc_requests = IO.read_traffic_from_json_file(original_traffic_file_name)
        reordered_sfc_requests = shuffle_request_order(original_sfc_requests, parameters)
        reordered_traffic_file_name = data_set_folder + (
            '/uniform' if parameters['uniform_distribution'] else '/non_uniform') \
                                      + '_CONUS36_reordered_traffic_' + num_time_slots + '_slots_' + num_con_current + '_con.tra'
        traffic_dis_file_name = reordered_traffic_file_name + 'request_dist.png'
        print_traffic_distribution_to_file(network, reordered_sfc_requests.values(),
                                           traffic_dis_file_name)
        print_traffic_to_file(reordered_traffic_file_name, reordered_sfc_requests, parameters)

    print("Nicely done !!!", datetime.now().time())


#########################################


#########################################
if __name__ == "__main__":
    main()
