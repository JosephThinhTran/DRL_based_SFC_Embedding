# author: Huy Duong, CRIM
# Note:
# - Dynamic traffic
# - Unbounded Scheme

# import networkx
# import networkx as nx
# import matplotlib.pyplot as plt
# import os
# import networkx
import numpy
from networkx import *
# import json
from numpy import random
from random import seed  # seed the pseudorandom number generator
from random import randint
from simulation import *
from IO import IO
from datetime import datetime
import sys


#########################################
def print_graph_to_file(g: networkx.DiGraph, file_name: str):
    f = open(file_name, 'w')

    # print(str(len(g.nodes())) + "\t% number of routers", file=f)
    print(str(len(g.nodes())) + "\t% number of routers", file=f)
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
            data_line += " " + str(g.nodes[u]['CPU_cap']) + " " +\
                         str(g.nodes[u]['RAM_cap']) + " " + str(g.nodes[u]['MEM_cap'])
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
    range_of_rate = parameters["expand_rate"]

    for u, v in resulted_network.edges:
        rate = 1 + randint(range_of_rate[0], range_of_rate[1]) / 100.0
        resulted_network[u][v]['free_bw'] = resulted_network[u][v]['bw'] \
            = simulation.max_link_used_bandwidth[u][v] * rate

    for node in resulted_network.nodes:
        rate = 1 + randint(range_of_rate[0], range_of_rate[1]) / 100.0
        max_used_cpu = simulation.max_node_used_cap[node]['CPU']
        resulted_network.nodes[node]['free_CPU'] = resulted_network.nodes[node]['CPU_cap'] = max_used_cpu * rate

        rate = 1 + randint(range_of_rate[0], range_of_rate[1]) / 100.0
        max_used_ram = simulation.max_node_used_cap[node]['RAM']
        resulted_network.nodes[node]['free_RAM'] = resulted_network.nodes[node]['RAM_cap'] = max_used_ram * rate

        rate = 1 + randint(range_of_rate[0], range_of_rate[1]) / 100.0
        max_used_mem = simulation.max_node_used_cap[node]['MEM']
        resulted_network.nodes[node]['free_MEM'] = resulted_network.nodes[node]['MEM_cap'] = max_used_mem * rate

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
        u, v = layered_graph_path[i], layered_graph_path[i+1]
        if u[1] == v[1]:
            delay += simulation.network[u[0]][v[0]]['delay']
        else:
            assert u[1] == v[1] - 1, "Not a proper layered graph path"
            processing_vnf = sequent_vnfs[u[1]]
            delay += bw * simulation.SFC_list["vnf_list"][processing_vnf]['PROC_rate']

    return delay
#########################################


#########################################
# Reorder the original SFC requests to vary traffic pattern
def generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation, parameters):
    rate = parameters["expand_rate"]
    reordered_sfc_requests = {}

    # id_list is a map of each original request id to a new id
    id_list = [int(request_id) for request_id in original_sfc_requests.keys()]
    random.shuffle(id_list)

    # First, reordered SFC requests are the same order with the original SFC requests
    for request_id in original_sfc_requests:
        reordered_sfc_requests[request_id] = copy.deepcopy(original_sfc_requests[request_id])

    for e in simulation.processed_events:
        assert e[1]['type'] in ['new', 'accept', 'terminate'], "Something is wrong with unbounded simulation"
        if e[1]['type'] == 'accept':
            r = e[1]['sfc_request']
            layered_graph_path = e[1]['path']
            delay = get_delay_of_layered_graph_path(simulation, r, layered_graph_path)
            assert delay < 1, "delay is >= 1"
            new_request = reordered_sfc_requests[r['id']]  # Note that new_request is just an alias
            new_request['delay_req'] = original_sfc_requests[r['id']]['delay_req'] = delay * 1.2
            duration = new_request['end_time'] - new_request['arrival_time']
            exchanged_request_id = id_list[new_request['id']]
            new_request['arrival_time'] = original_sfc_requests[exchanged_request_id]['arrival_time']
            new_request['end_time'] = new_request['arrival_time'] + duration

    temp_request_list = [value for key, value in reordered_sfc_requests.items()]
    reordered_sfc_requests.clear()
    temp_request_list = sorted(temp_request_list, key=lambda k: k['arrival_time'])
    for i in range(len(temp_request_list)):
        request = temp_request_list[i]  # alias of the variable
        request['co_id'] = request['id']
        request['id'] = i
        reordered_sfc_requests[i] = request

    return reordered_sfc_requests
#########################################


#########################################
def generate_traffic_with_unbounded_capacity_network(g, sfc_list, parameters):
    network = copy.deepcopy(g)
    for u in network.nodes:  # make network unbound capacity
        if network.nodes[u]['VNF_list']:
            network.nodes[u]['free_CPU'] = network.nodes[u]['CPU_cap'] = sys.maxsize
            network.nodes[u]['free_RAM'] = network.nodes[u]['RAM_cap'] = sys.maxsize
            network.nodes[u]['free_MEM'] = network.nodes[u]['MEM_cap'] = sys.maxsize

        for u_1, v in network.edges(u):
            network[u][v]['bw'] = sys.maxsize
            network[u][v]['free_bw'] = sys.maxsize

    simulation = Simulation(network, sfc_list)
    simulation.simulation_output_file_name = "unbounded_sim.txt"
    simulation.set_routing_algorithm("random_k_shortest")  # "random_k_shortest" or "Dijkstra"
    original_sfc_requests = {}
    n_sfc = len(sfc_list)-1
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
    for i in range(simulation_time):  # for each time slot
        for j in range(randint(1, max_concurrent_per_time_slot)):  # j concurrent requests at time slot i
            u = randint(0, n_nodes - 1)  # random source
            v = randint(0, n_nodes - 1)  # random destination
            while v == u:  # destination has to differ from source
                v = randint(0, n_nodes - 1)

            sfc_id = "SFC_" + str(randint(0, n_sfc - 1))
            bw = randint(1, max_request_bw)  # uniform distribution of bandwidth
            ram_req = parameters['RAM_list'][randint(0, len(parameters['RAM_list'])-1)]
            # delay_req = str(randint(min_delay_req, max_delay_req))
            # sto_req = parameters['Storage_list'][bw-1]
            sto_req = parameters['Storage_list'][randint(0, len(parameters['Storage_list'])-1)]
            # duration of request has an geometrical distribution
            duration = numpy.random.geometric(1 / average_duration)
            original_sfc_requests[request_id] = {'id': request_id, 'source': u, 'destination': v, 'sfc_id': sfc_id,
                                                 'bw': bw, 'delay_req': max_delay_req,
                                                 'arrival_time': i, 'end_time': i + duration,
                                                 'RAM_req': ram_req, 'MEM_req': sto_req}
            event = {"time_point": i, "type": 'new', 'sfc_request': original_sfc_requests[request_id]}
            simulation.add_event(event)
            request_id += 1

    simulation.run(save_network_stats=True, save_processed_events=True, print_simulation_to_file=True, save_curves=False)

    # resulted_paths = {}
    # for event in simulation.processed_events:
    #     if event['type'] == 'accept':
    #         sfc_request = event['sfc_request']
    #         resulted_paths[sfc_request['id']] = event['path']

    resulted_network = generate_new_network_after_unbound_simulation(simulation, parameters)
    print('Shuffling requests ...')
    resulted_sfc_requests = generate_sfc_requests_from_unbounded_delay_requests(original_sfc_requests, simulation, parameters)
    return original_sfc_requests, resulted_sfc_requests, resulted_network, simulation
#########################################


#########################################
def generate_SFC_file(sfc_file_name: str, g):
    f = open(sfc_file_name, 'w')
    vnf_list = ['vnf_1', 'vnf_2', 'vnf_3', 'vnf_4']
    max_number_SFCs = 10  # 3 is a random number for the time being
    max_CPU_req = 10  # 10 is a random number for the time being
    # max_RAM_req = 10  # 10 is a random number for the time being
    # max_MEM_req = 10  # 10 is a random number for the time being
    max_processing_delay = 5  # 5 (ms) is a random number, so that sum of VNF processing delays is < 0.1 second

    info_line = "% list of NFVs and their bandwidth unit requirements: CPU, processing delay (s)"
    print(info_line, file=f)
    print(str(len(vnf_list)) + " % number of VNFs", file=f)

    # for i in range(0, max_number_VNFs):
    for vnf_name in vnf_list:
        # vnf_list.append("vnf_" + str(i))
        # line = vnf_name + " " + str(randint(1, max_CPU_req)) + " " + str(randint(1, max_RAM_req))
        # line += " " + str(randint(1, max_MEM_req)) + " " + str(randint(1, max_processing_delay) / 1000)
        line1 = vnf_name + " " + str(randint(1, max_CPU_req)) + " " + str(randint(1, max_processing_delay) / 1000)
        print(line1, file=f)

    info_line = "% list of SFCs: described using a DAG"
    print(info_line, file=f)
    print(str(max_number_SFCs) + " % number of SFCs", file=f)
    print("% SFC id, number of arcs of this SFC", file=f)
    print("% list of arcs", file=f)

    for sfc_counter in range(0, max_number_SFCs):
        num_vnf_pairs = randint(1, len(vnf_list)-1)  # a random number of VNFs
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
        g.nodes[node]['MEM_cap'] = 0
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
        g.nodes[node]['MEM_cap'] = 0

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
    verifying_simulation.simulation_output_file_name = "verifying_sim.txt"
    verifying_simulation.run(save_network_stats=True, save_processed_events=True, print_simulation_to_file=True)

    # for edge in simulation.network.edges.data():
    #     assert edge['free_cap'] == 0, "free capacity must be 0."
#########################################


#########################################
def print_traffic_to_file(traffic_file_name, sfc_requests, parameters):
    file = open(traffic_file_name, 'w')
    data = {'requests': []}
    for para, value in parameters.items():
        data[para] = value

    for request_id, sfc_request in sfc_requests.items():
        data['requests'].append(sfc_request)

    json.dump(data, file, indent=2)
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
def main():
    print("Hello World!")
    graphml_file_name = "../raw_topologies/ibm.graphml.xml"  # read graphml file
    g = read_graph_from_graphml_file(graphml_file_name)

    # change raw network to our desired format
    raw_preprocessed_network_file_name = "../raw_topologies/preprocessed_ibm.net"
    print_graph_to_file(g, raw_preprocessed_network_file_name)

    seed(1)     # seed of random generator

    # generating SFC file
    sfc_file_name = '../complete_data_sets/sfc_file.sfc'
    #generate_SFC_file(sfc_file_name, g)

    # read graph file
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name)
    # print(network.edges)

    # read SFC file
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)

    # set up parameters
    parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 2000, '#_time_slots': 200_000,
                  'expand_rate': [-10, 20],
                  'RAM_list': [8, 16, 32], 'Storage_list': [20, 40, 100, 200]}

    # generating traffic for unbounded capacity network
    print(parameters)
    log_time = datetime.now().strftime("%Y-%B-%d__%H-%M-%S")
    print("Generating dataset at time " + log_time)
    network.graph["is_unbounded"] = True
    original_sfc_requests, reordered_sfc_requests, resulted_network, unbounded_simulation \
        = generate_traffic_with_unbounded_capacity_network(network, sfc_list, parameters)
    resulted_network.graph["is_unbounded"] = False

    num_con_current = str(parameters["max_concurrent_request_per_time_slot"])
    num_time_slots = str(parameters["#_time_slots"])
    original_traffic_file_name = '../complete_data_sets/original_traffic_'+num_time_slots+'_slots_' + num_con_current \
                                    + '_con_' + log_time + '.tra'
    print_traffic_to_file(original_traffic_file_name, original_sfc_requests, parameters)

    reordered_traffic_file_name = '../complete_data_sets/reordered_traffic_'+num_time_slots+'_slots_' + num_con_current \
                                + '_con_' + log_time + '.tra'
    print_traffic_to_file(reordered_traffic_file_name, reordered_sfc_requests, parameters)

    network_file_name = '../complete_data_sets/ibm_' + num_time_slots + '_slots_' + num_con_current \
                                + '_con_' + log_time + '.net'
    print_graph_to_file(resulted_network, network_file_name)

    # reuse the generated data to run simulation again to see how algorithm and network behave
    # print('Checking originally generated requests ... ', datetime.now().time())
    # imported_sfc_requests = read_traffic_from_text_file(original_traffic_file_name)
    # check_generated_traffic_with_unbounded_capacity_network(resulted_network, sfc_list,
    #                                                        imported_sfc_requests, unbounded_simulation)

    print("Nicely done !!!", datetime.now().time())
#########################################


#########################################
if __name__ == "__main__":
    main()
