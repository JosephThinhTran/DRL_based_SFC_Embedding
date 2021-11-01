#########################################
# Author: Huy Duong, CRIM
# Simulation to run a dynamic process
#########################################

# packages
# import networkx
import time
import copy
import json
import random
import heapq
from datetime import datetime

import matplotlib.pyplot as plt
import queue
# from gurobipy import *
import networkx
# import pylgrim

# from RCSP_model import *


########################################################################
def append_values_to_curves(sim, t, curves, throughput, offered_load_every_slot_list, total_resource_capacities):
    total_cpu_cap = total_resource_capacities['total_cpu_cap']
    total_ram_cap = total_resource_capacities['total_ram_cap']
    total_disk_cap = total_resource_capacities['total_disk_cap']
    total_bw_cap = total_resource_capacities['total_bw_cap']

    cpu_usage, ram_usage, disk_usage, bw_usage = sim.get_resource_usage()

    curves['time_points'].append(t)
    curves['cpu_usage_rate'].append(cpu_usage / total_cpu_cap * 100.0)
    curves['ram_usage_rate'].append(ram_usage / total_ram_cap * 100.0)
    curves['disk_usage_rate'].append(disk_usage / total_disk_cap * 100.0)
    curves['bw_usage_rate'].append(bw_usage / total_bw_cap * 100.0)
    curves['throughput'].append(throughput)
    curves['offered_load'].append(offered_load_every_slot_list[t])
    if offered_load_every_slot_list[t] > 0:
        curves['throughput_rate'].append(throughput / offered_load_every_slot_list[t] * 100.0)
    else:
        assert throughput == 0, 'Throughput must be 0 !!!'
        curves['throughput_rate'].append(0)
########################################################################


class Simulation:
    """Simulation to run a dynamic process"""

    ########################################################################
    # methods

    ########################################################################
    # initialization of an object
    def __init__(self, network, sfc_list):
        self.network = copy.deepcopy(network)
        self.empty_network = copy.deepcopy(network)  # keep an empty network for auxiliary functions
        self.SFC_list = sfc_list
        self.routing_algorithm = "ILP"
        self.current_connects = []  # set of current connections
        self.event_queue = queue.PriorityQueue(0)
        self.event_saving_flag = False      # if flag=True, all processed events are saved in self.processed_events
        self.processed_events = []
        self.max_link_used_bandwidth = None
        self.max_node_used_cap = None
        self.simulation_output_file_name = None
        self.average_new_request_processing_time = {'num_requests': 0, 'total_time': 0}
        self.shortest_distances = None

    ########################################################################
    def add_event(self, event):
        self.event_queue.put(((event["time_point"], event['sfc_request']['id']), event))

    ########################################################################
    def run(self, save_network_stats, save_processed_events, print_simulation_to_file, save_curves):
        if print_simulation_to_file:
            file = open(self.simulation_output_file_name, 'w')
            file.close()

        if save_network_stats:  # if we want to collect stats of simulation
            self.initialize_network_stats()

        if save_processed_events:
            self.event_saving_flag = True

        # precompute the shortest path distances between all pairs or nodes
        self.compute_shortest_distances_between_all_pairs()

        while not self.event_queue.empty():
            self.process_next_event(print_simulation_to_file)

            if save_network_stats:
                self.collect_running_network_stats()

            if print_simulation_to_file and len(self.processed_events) % 3000 == 0:
                self.print_simulation_to_file()

            if save_curves and len(self.processed_events) % 3000 == 0:
                self.print_curves_to_file()
            # if print_simulation_to_file:
            #     file = open(self.simulation_output_file_name, 'a')
            #     # print(next_event, file=file)
            #     print(self.max_node_used_cap, file=file)
            #     file.close()

        if print_simulation_to_file:
            self.print_simulation_to_file()

        if save_curves:
            self.print_curves_to_file()

    ########################################################################
    # process a 'new' event, i.e., when a sfc request arrives
    def print_simulation_to_file(self):
        average_new_request_proc_time = self.average_new_request_processing_time['total_time'] / \
                                        self.average_new_request_processing_time['num_requests']
        data = {'average_routing_processing (s)': average_new_request_proc_time,
                'events_size': len(self.processed_events), 'events': [], }
        file = open(self.simulation_output_file_name, 'w')
        for e in self.processed_events:
            data['events'].append(e)

        json.dump(data, file, indent=2)
        # print(self.network.nodes[15], file=file)
        # print(self.max_node_used_cap, file=file)
        file.close()
    ########################################################################
    # ########################################################################
    # def run_with_hardcoded_paths(self, save_network_stats, save_processed_events):
    #     if save_network_stats:  # if we want to collect stats of simulation
    #         self.initialize_network_stats()
    #
    #     if save_processed_events:
    #         self.event_saving_flag = True
    #
    #     while not self.event_queue.empty():
    #         self.process_next_event()
    #         if save_network_stats:
    #             self.collect_running_network_stats()

    ########################################################################
    # process a 'new' event, i.e., when a sfc request arrives
    def process_new_request_event(self, an_event):
        # find a shortest end-to-end delay path
        start_time = time.time()

        path = None
        if self.routing_algorithm == "Dijkstra":
            path = self.find_shortest_e_2_e_delay_path_using_dijkstra_on_layered_graph(an_event['sfc_request'])
        elif self.routing_algorithm == "random_k_shortest":
            path = self.find_a_random_path_among_k_shortest_paths_on_layered_graph(an_event['sfc_request'])
        elif self.routing_algorithm == 'RCSP':
            # path = self.find_resource_constrained_shortest_path_on_layered_graph(an_event['sfc_request'])
            path = self.find_resource_constrained_shortest_path_on_layered_graph_with_heap(an_event['sfc_request'])
        # elif self.routing_algorithm == "ILP":
        #     path = self.find_shortest_e_2_e_delay_path_using_ILP_model(an_event[1]['sfc_request'])
        else:
            assert False, print(self.routing_algorithm, "is not defined.")

        self.average_new_request_processing_time['num_requests'] += 1
        self.average_new_request_processing_time['total_time'] += time.time()-start_time

        # if a path is found, then the simulation puts an 'accept' event to the queue,
        # else, puts an 'reject' event to the event queue
        if path is not None:
            # formation of an "accept event": time point, event's code, sfc request, a path
            accept_event = {"time_point": an_event['time_point'], "type": "accept",
                            "sfc_request": an_event['sfc_request'], "path": path}
            self.add_event(accept_event)

            # add the corresponding terminating event
            terminating_event = {"time_point": an_event['sfc_request']['end_time'], "type": "terminate",
                                 "sfc_request": an_event['sfc_request'], "path": path}
            self.add_event(terminating_event)
        else:
            # formation of an "reject" event: time point, event's code, sfc request
            reject_event = {"time_point": an_event["time_point"], "type": "reject",
                            "sfc_request": an_event['sfc_request']}
            self.add_event(reject_event)

    ########################################################################
    # process 'accept' event
    def process_accept_event(self, accept_event):
        # time_point = accept_event['time_point']
        # accept_event = accept_event[1]
        assert accept_event["type"] == 'accept', "event's type is not 'accept', but " + accept_event['type']
        sfc_request = accept_event["sfc_request"]
        path = accept_event["path"]
        sequent_vnfs = self.SFC_list[sfc_request["sfc_id"]]['sequent_vnfs']
        num_vnfs = len(sequent_vnfs)     # number of layers = number of VNFs + 1
        # for i in range(num_vnfs+1):
        for i in range(len(path)-1):    # reserve bandwidth and compute resources along the path
            uu, vv = path[i], path[i+1]
            u, v = uu[0], vv[0]
            if uu[1] == vv[1]:  # uu and vv are at the same layer
                self.network[u][v]["free_bw"] -= sfc_request["bw"]
                assert self.network[u][v]["free_bw"] >= 0, "free capacity cannot be less than 0"
            else:  # uu and vv are at different layers. It means a VNF is installed at u
                vnf_location = u    # location of (i)-th VNF
                vnf = self.SFC_list['vnf_list'][sequent_vnfs[uu[1]]]

                bw = sfc_request["bw"]  # for the time being, consumption is proportional to bandwidth
                self.network.nodes[vnf_location]["free_CPU"] -= bw * vnf["CPU_rate"]
                # self.network.nodes[vnf_location]["free_RAM"] -= bw * vnf["RAM_rate"]
                self.network.nodes[vnf_location]["free_RAM"] -= sfc_request['RAM_req']
                # self.network.nodes[vnf_location]["free_MEM"] -= bw * vnf["MEM_rate"]
                self.network.nodes[vnf_location]["free_MEM"] -= sfc_request["MEM_req"]
                assert self.network.nodes[vnf_location]["free_CPU"] >= 0
                assert self.network.nodes[vnf_location]["free_RAM"] >= 0
                assert self.network.nodes[vnf_location]["free_MEM"] >= 0

    ########################################################################
    # process 'reject' event
    def process_reject_event(self, reject_event):
        pass

    ########################################################################
    # process next event
    def process_next_event(self, print_simulation_to_file):
        assert not self.event_queue.empty(), "Event queue is empty"
        next_event = self.event_queue.get()  # get the next (front) event, i.e., event whose smallest time point

        if self.event_saving_flag or print_simulation_to_file:      # saving the event to the list
            self.processed_events.append(next_event)

        if next_event[1]['type'] == 'new':
            self.process_new_request_event(next_event[1])
        elif next_event[1]['type'] == 'accept':
            self.process_accept_event(next_event[1])
        elif next_event[1]['type'] == 'reject':
            self.process_reject_event(next_event[1])
        elif next_event[1]['type'] == 'terminate':
            self.process_terminating_event(next_event[1])

        # if print_simulation_to_file:
        #     file = open(self.simulation_output_file_name, 'a')
        #     print(next_event, file=file)
        #     # print(self.network.nodes[15], file=file)
        #     # print(self.max_node_used_cap, file=file)
        #     file.close()

    ########################################################################
    # find the shortest e-2-e delay path for a sfc request
    def find_shortest_e_2_e_delay_path_using_ILP_model(self, sfc_request):
        pass

        model = RCSP_Model()     # initialize an empty model
        model.set_network(self.network)
        model.generate_model(sfc_request)

        path = None
        return path
    ########################################################################

    ########################################################################
    # build a layered graph from the original physical network
    def build_layered_graph(self, sfc_request):
        layered_graph = networkx.DiGraph()
        # layered_graph = networkx.DiGraph()
        sequent_vnfs = self.SFC_list[sfc_request['sfc_id']]['sequent_vnfs']
        for i in range(len(sequent_vnfs)+1):  # build i-th layer in (# vnfs) + 1 layers
            for u in self.network.nodes:
                node_name = self.make_node_name_in_layered_graph(u, i)  # node u at layer i
                layered_graph.add_node(node_name, layer=i)

                if i > 0 and sequent_vnfs[i-1] in self.network.nodes[u]['VNF_list']:
                    pre_layer_node = self.make_node_name_in_layered_graph(u, i-1)  # node u at layer i-1
                    processing_vnf = sequent_vnfs[i-1]
                    processing_delay_rate = self.SFC_list["vnf_list"][processing_vnf]['PROC_rate']
                    delay = sfc_request['bw'] * processing_delay_rate  # processing delay of i-th VNF
                    layered_graph.add_edge(pre_layer_node, node_name, delay=delay)

            for u, v in self.network.edges:
                uu = self.make_node_name_in_layered_graph(u, i)
                vv = self.make_node_name_in_layered_graph(v, i)
                layered_graph.add_edge(uu, vv)
                for key in self.network[u][v]:
                    layered_graph[uu][vv][key] = self.network[u][v][key]

        return layered_graph
    ########################################################################

    ########################################################################
    # find shortest path using Dijkstra's algorithm on layered graph without resource constraints
    def find_shortest_e_2_e_delay_path_using_dijkstra_on_layered_graph(self, sfc_request):
        layered_graph = self.build_layered_graph(sfc_request)
        s = sfc_request['source']
        d = sfc_request['destination']
        sequent_vnfs = self.SFC_list[sfc_request["sfc_id"]]['sequent_vnfs']
        s = self.make_node_name_in_layered_graph(int(s), 0)
        d = self.make_node_name_in_layered_graph(int(d), len(sequent_vnfs))
        path = None
        try:
            path = networkx.dijkstra_path(layered_graph, s, d, weight='delay')
            # print(path)
            return path
        except networkx.NetworkXException:
            networkx.draw_networkx(layered_graph)
            # Set margins for the axes so that nodes aren't clipped
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.show()
            assert False, "no path exists."
    ########################################################################

    ########################################################################
    # find k-shortest paths, then choose a random one, on layered graph without resource constraints
    def find_a_random_path_among_k_shortest_paths_on_layered_graph(self, sfc_request):
        layered_graph = self.build_layered_graph(sfc_request)
        s = sfc_request['source']
        d = sfc_request['destination']
        sequent_vnfs = self.SFC_list[sfc_request["sfc_id"]]['sequent_vnfs']
        s = self.make_node_name_in_layered_graph(int(s), 0)
        d = self.make_node_name_in_layered_graph(int(d), len(sequent_vnfs))
        paths = None
        k = random.randint(0, 9)
        try:
            paths = networkx.shortest_simple_paths(layered_graph, s, d, weight='delay')  # all paths
            # print(path)
            for i, path in enumerate(paths):
                if i == k:
                    return path
        except networkx.NetworkXException:
            networkx.draw_networkx(layered_graph)
            # Set margins for the axes so that nodes aren't clipped
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.show()
            assert False, "no path exists."
    ########################################################################

    ########################################################################
    def get_cpu_rate_of_vnf(self, vnf):
        return self.SFC_list['vnf_list'][vnf]["CPU_rate"]
    ########################################################################

    ########################################################################
    def get_ram_rate_of_vnf(self, vnf):
        return self.SFC_list['vnf_list'][vnf]["RAM_rate"]
    ########################################################################

    ########################################################################
    def get_storage_rate_of_vnf(self, vnf):
        return self.SFC_list['vnf_list'][vnf]["MEM_rate"]
    ########################################################################

    ########################################################################
    def make_new_label_GLSA(self, layered_graph, sfc_request, u, v, prefix_path_cost_and_weight, detailed_path, link_usage, node_usage):
        new_label = copy.deepcopy(prefix_path_cost_and_weight)
        # new_link_usage = None
        # new_node_usage = None
        new_link_usage = copy.deepcopy(link_usage)
        new_node_usage = copy.deepcopy(node_usage)
        none_value_set = None, None, None, None

        # firstly, check delay constraints
        lower_bound_delay = prefix_path_cost_and_weight[1] + layered_graph[u][v]['delay'] + self.get_remaining_delay(v[0], v[1], sfc_request)
        if lower_bound_delay > sfc_request['delay_req']:
            return none_value_set

        if u[1] == v[1]:  # u and v are on the same layer
            uu, vv = u[0], v[0]
            # secondly, check bandwidth constrains
            if link_usage[uu][vv] + sfc_request['bw'] > self.network[uu][vv]['free_bw']:
                return none_value_set

            # new_link_usage = copy.deepcopy(link_usage)
            # new_node_usage = copy.deepcopy(node_usage)
            new_link_usage[uu][vv] += sfc_request['bw']

        else:  # u and v are not on the same layer -> vnf is installed here.
            original_node = u[0]
            sfc = self.SFC_list[sfc_request['sfc_id']]
            vnf = sfc['sequent_vnfs'][u[1]]

            if vnf not in self.network.nodes()[original_node]['VNF_list']:
                return none_value_set

            cpu_usage = node_usage[original_node]['CPU'] + self.get_cpu_rate_of_vnf(vnf)*sfc_request['bw']
            # cpu_usage = node_usage[original_node]['CPU'] + sfc_request['RAM_req']
            if cpu_usage > self.network.nodes()[original_node]['free_CPU']:
                return none_value_set

            # ram_usage = node_usage[original_node]['RAM'] + self.get_ram_rate_of_vnf(vnf) * sfc_request['bw']
            ram_usage = node_usage[original_node]['RAM'] + sfc_request['RAM_req']
            if ram_usage > self.network.nodes[original_node]['free_RAM']:
                return none_value_set

            # check storage usage
            # storage_usage = node_usage[original_node]['MEM'] + self.get_storage_rate_of_vnf(vnf) * sfc_request['bw']
            storage_usage = node_usage[original_node]['MEM'] + sfc_request['MEM_req']
            if storage_usage > self.network.nodes[original_node]['free_MEM']:
                return none_value_set

            new_node_usage[original_node] = {'CPU': cpu_usage, 'RAM': ram_usage, 'MEM': storage_usage}

        # make the new label
        new_label = (prefix_path_cost_and_weight[0] + 1, prefix_path_cost_and_weight[1] + layered_graph[u][v]['delay'])
        new_detailed_path = copy.deepcopy(detailed_path)
        new_detailed_path.append(v)
        return new_label, new_detailed_path, new_link_usage, new_node_usage
    ########################################################################

    ########################################################################
    def initialize_GLSA(self, layered_graph, s):
        L = {}
        link_usage_records, node_usage_records = {}, {}
        # T = {}
        untreated_list = {}
        max_id_list = {}
        detailed_paths = {}

        # initialization of each node
        for u in layered_graph.nodes:
            if s != u:
                L[u] = {}
                detailed_paths[u] = {}
                untreated_list[u] = set()
                max_id_list[u] = -1
                link_usage_records[u] = {}
                node_usage_records[u] = {}

        # initialization of the source node
        L[s] = {0: (0, 0)}  # 0 (id): 0 (hop), 0 (delay) at source node
        detailed_paths[s] = {0: [s]}
        max_id_list[s] = 0
        untreated_list[s] = {0}  # path whose id=0 is untreated
        link_usage_records[s] = {0: {}}
        node_usage_records[s] = {0: {}}
        for u in self.network.nodes():
            link_usage_records[s][0][u] = {}
            node_usage_records[s][0][u] = {'CPU': 0, 'RAM': 0, 'MEM': 0}
            for v in self.network.neighbors(u):
                link_usage_records[s][0][u][v] = 0  # empty path at the source node

        return L, detailed_paths, link_usage_records, node_usage_records, untreated_list, max_id_list
    ########################################################################

    ########################################################################
    # solve resource constrained shortest path problem using heuristic
    def find_resource_constrained_shortest_path_on_layered_graph(self, sfc_request):
        layered_graph = self.build_layered_graph(sfc_request)
        s = sfc_request['source']
        d = sfc_request['destination']
        sequent_vnfs = self.SFC_list[sfc_request["sfc_id"]]['sequent_vnfs']
        s = self.make_node_name_in_layered_graph(int(s), 0)
        d = self.make_node_name_in_layered_graph(int(d), len(sequent_vnfs))
        L, detailed_paths, link_usage_records, node_usage_records, untreated_list, max_id_list = \
            self.initialize_GLSA(layered_graph, s)

        min_lexico_path = (0, 0)  # a dummy number
        least_cost = None  # least cost from s to d
        while min_lexico_path is not None:  # repeat until no more untreated path
            # find the lexicographically-minimum path
            min_lexico_path = None

            for u in layered_graph.nodes:
                for k in untreated_list[u]:
                    if min_lexico_path is None or L[min_lexico_path[0]][min_lexico_path[1]] > L[u][k]:
                        min_lexico_path = (u, k)

            if min_lexico_path is None:     # if there is no untreated path (label), then stop
                continue

            u, min_path_id = min_lexico_path
            untreated_list[u].remove(min_path_id)   # remove the label from the untreated set
            prefix_path = L[u][min_path_id]
            link_usage, node_usage = link_usage_records[u][min_path_id], node_usage_records[u][min_path_id]

            # expand the current minimum lexicographically path
            for v in layered_graph.neighbors(u):
                new_label, new_detailed_path, new_link_usage, new_node_usage = \
                    self.make_new_label_GLSA(layered_graph, sfc_request, u, v, prefix_path, detailed_paths[u][min_path_id],
                                             link_usage, node_usage)

                if new_label is None:
                    continue

                is_dominated = False
                for path_id_2 in L[v]:  # checking if the new label is dominated
                    if L[v][path_id_2][0] < new_label[0] and L[v][path_id_2][1] < new_label[1]:
                        is_dominated = True
                        break

                # if the new label is dominated or its cost is > least cost from s to d so far, then skip
                if is_dominated or (least_cost is not None and least_cost < new_label[0]):
                    continue

                # remove dominated paths, by the new path, from the untreated label list
                for path_id in untreated_list[v]:
                    if L[v][path_id][0] > new_label[0] and L[v][path_id][1] > new_label[1]:
                        del L[v][path_id]
                        del detailed_paths[v][path_id]
                        del link_usage_records[v][path_id]
                        del node_usage_records[v][path_id]
                        untreated_list[v].remove(path_id)

                # add the new label to list L and untreated list
                max_id_list[v] += 1
                new_path_id = max_id_list[v]
                L[v][new_path_id] = new_label
                detailed_paths[v][new_path_id] = new_detailed_path
                untreated_list[v].append(new_path_id)
                link_usage_records[v][new_path_id] = new_link_usage
                node_usage_records[v][new_path_id] = new_node_usage
                if v == d and (least_cost is None or least_cost > new_label[0]):
                    least_cost = new_label[0]

                # print('node: ', v)
                # for path_id, path in L[v].items():
                #     print(path_id, path)

        # end while

        if not L[d]:    # if L[d] is empty, i.e., no path is found, then return None
            return None

        # find the "best" path
        shortest_path_id = None
        for path_id in L[d]:
            if shortest_path_id is None or L[d][shortest_path_id][0] > L[d][path_id][0]:
                shortest_path_id = path_id

        return copy.deepcopy(detailed_paths[d][shortest_path_id])

    ########################################################################

    ########################################################################
    # solve resource constrained shortest path problem using heuristic
    def find_resource_constrained_shortest_path_on_layered_graph_with_heap(self, sfc_request):
        layered_graph = self.build_layered_graph(sfc_request)
        s = sfc_request['source']
        d = sfc_request['destination']
        sequent_vnfs = self.SFC_list[sfc_request["sfc_id"]]['sequent_vnfs']
        s = self.make_node_name_in_layered_graph(int(s), 0)
        d = self.make_node_name_in_layered_graph(int(d), len(sequent_vnfs))
        L, detailed_paths, link_usage_records, node_usage_records, untreated_list, max_id_list = \
            self.initialize_GLSA(layered_graph, s)

        path_heap = []
        heapq.heappush(path_heap, (L[s][0], s, 0))

        min_lexico_path = (0, 0)  # a dummy number
        least_cost = None  # least cost from s to d
        while path_heap:  # repeat until no more untreated path
            # find the lexicographically-minimum path
            min_lexico_path = heapq.heappop(path_heap)

            u, min_path_id = min_lexico_path[1:3]
            if min_path_id not in untreated_list[u]:
                continue

            untreated_list[u].remove(min_path_id)  # remove the label from the untreated set
            prefix_path = L[u][min_path_id]
            link_usage, node_usage = link_usage_records[u][min_path_id], node_usage_records[u][min_path_id]

            # expand the current minimum lexicographically path
            for v in layered_graph.neighbors(u):
                new_label, new_detailed_path, new_link_usage, new_node_usage = \
                    self.make_new_label_GLSA(layered_graph, sfc_request, u, v, prefix_path,
                                             detailed_paths[u][min_path_id],
                                             link_usage, node_usage)

                if new_label is None:
                    continue

                is_dominated = False
                for path_id_2 in L[v]:  # checking if the new label is dominated
                    if L[v][path_id_2][0] < new_label[0] and L[v][path_id_2][1] < new_label[1]:
                        is_dominated = True
                        break

                # if the new label is dominated or its cost is > least cost from s to d so far, then skip
                if is_dominated or (least_cost is not None and least_cost < new_label[0]):
                    continue

                # remove dominated paths, by the new path, from the untreated label list
                for path_id in untreated_list[v]:
                    if L[v][path_id][0] > new_label[0] and L[v][path_id][1] > new_label[1]:
                        del L[v][path_id]
                        del detailed_paths[v][path_id]
                        del link_usage_records[v][path_id]
                        del node_usage_records[v][path_id]
                        untreated_list[v].remove(path_id)

                # add the new label to list L and untreated list
                max_id_list[v] += 1
                new_path_id = max_id_list[v]
                L[v][new_path_id] = new_label
                detailed_paths[v][new_path_id] = new_detailed_path
                untreated_list[v].add(new_path_id)
                link_usage_records[v][new_path_id] = new_link_usage
                node_usage_records[v][new_path_id] = new_node_usage
                heapq.heappush(path_heap, (new_label, v, new_path_id))
                if v == d and (least_cost is None or least_cost > new_label[0]):
                    least_cost = new_label[0]

                # print('node: ', v)
                # for path_id, path in L[v].items():
                #     print(path_id, path)

        # end while

        if not L[d]:  # if L[d] is empty, i.e., no path is found, then return None
            return None

        # find the "best" path
        shortest_path_id = None
        for path_id in L[d]:
            if shortest_path_id is None or L[d][shortest_path_id][0] > L[d][path_id][0]:
                shortest_path_id = path_id

        return copy.deepcopy(detailed_paths[d][shortest_path_id])
    ########################################################################

    ########################################################################
    # find shortest path using Dijkstra's algorithm on layered graph without resource constraints
    def set_routing_algorithm(self, algo):
        self.routing_algorithm = algo

    ########################################################################
    # make the name of a node in layered graph from it's original node and layer
    @staticmethod
    def make_node_name_in_layered_graph(node, layer_index):
        return node, layer_index
        # return str(node) + "_l_" + str(layer_index)

    # def get_origiral_node_name_and_layer(self, uu):
    #    return uu

    ########################################################################
    # Record network's stats during simulation
    def collect_running_network_stats(self):
        for u, v in self.network.edges:
            used_bw = self.network[u][v]['bw'] - self.network[u][v]['free_bw']
            self.max_link_used_bandwidth[u][v] = max(self.max_link_used_bandwidth[u][v], used_bw)

        for node in self.network.nodes:
            if self.network.nodes[node]['VNF_list']:
                used_cpu = self.network.nodes[node]['CPU_cap'] - self.network.nodes[node]['free_CPU']
                self.max_node_used_cap[node]['CPU'] = max(self.max_node_used_cap[node]['CPU'], used_cpu)

                used_ram = self.network.nodes[node]['RAM_cap'] - self.network.nodes[node]['free_RAM']
                self.max_node_used_cap[node]['RAM'] = max(self.max_node_used_cap[node]['RAM'], used_ram)

                used_mem = self.network.nodes[node]['MEM_cap'] - self.network.nodes[node]['free_MEM']
                self.max_node_used_cap[node]['MEM'] = max(self.max_node_used_cap[node]['MEM'], used_mem)
    ########################################################################

    ########################################################################
    # Initialize stats of simulation
    def initialize_network_stats(self):
        self.max_link_used_bandwidth = {}
        for u, v in self.network.edges:
            if u not in self.max_link_used_bandwidth:
                self.max_link_used_bandwidth[u] = {}

            self.max_link_used_bandwidth[u][v] = 0

        self.max_node_used_cap = {}
        for node in self.network.nodes:
            self.max_node_used_cap[node] = {}
            self.max_node_used_cap[node]['CPU'] = 0
            self.max_node_used_cap[node]['RAM'] = 0
            self.max_node_used_cap[node]['MEM'] = 0

    ########################################################################
    def process_terminating_event(self, event):
        sfc_request = event['sfc_request']
        bw = event['sfc_request']['bw']
        sequent_vnfs = self.SFC_list[sfc_request['sfc_id']]['sequent_vnfs']
        layered_graph_path = event['path']

        for i in range(len(layered_graph_path) - 1):
            u, v = layered_graph_path[i], layered_graph_path[i + 1]
            if u[1] == v[1]:
                self.network[u[0]][v[0]]['free_bw'] += bw
                assert self.network[u[0]][v[0]]['free_bw'] <= self.network[u[0]][v[0]]['bw']
            else:
                assert u[1] == v[1] - 1, "Not a proper layered graph path"
                processing_vnf = self.SFC_list['vnf_list'][sequent_vnfs[u[1]]]
                vnf_location = u[0]
                self.network.nodes[vnf_location]["free_CPU"] += bw * processing_vnf["CPU_rate"]
                # self.network.nodes[vnf_location]["free_RAM"] += bw * processing_vnf["RAM_rate"]
                self.network.nodes[vnf_location]["free_RAM"] += sfc_request["RAM_req"]
                # self.network.nodes[vnf_location]["free_MEM"] += bw * processing_vnf["MEM_rate"]
                self.network.nodes[vnf_location]["free_MEM"] += sfc_request["MEM_req"]
                assert self.network.nodes[vnf_location]["free_CPU"] <= self.network.nodes[vnf_location]["CPU_cap"]
                assert self.network.nodes[vnf_location]["free_RAM"] <= self.network.nodes[vnf_location]["RAM_cap"]
                assert self.network.nodes[vnf_location]["free_MEM"] <= self.network.nodes[vnf_location]["MEM_cap"]
    ########################################################################

    ########################################################################
    def run_and_print_stats(self):
        pass
    ########################################################################

    ########################################################################
    def get_total_resource_capacities(self):
        total_cpu_cap, total_ram_cap, total_disk_cap, total_bw_cap = 0, 0, 0, 0
        for u in self.network.nodes:
            if not self.network.nodes[u]['VNF_list']:
                continue

            total_cpu_cap += self.network.nodes[u]['CPU_cap']
            total_ram_cap += self.network.nodes[u]['RAM_cap']
            total_disk_cap += self.network.nodes[u]['MEM_cap']

        for u, v in self.network.edges:
            total_bw_cap += self.network[u][v]['bw']

        return total_cpu_cap, total_ram_cap, total_disk_cap, total_bw_cap
    ########################################################################

    ########################################################################
    def get_resource_usage(self):
        total_cpu_usage, total_ram_usage, total_disk_usage, total_bw_usage = 0, 0, 0, 0
        for u in self.network.nodes:
            if not self.network.nodes[u]['VNF_list']:
                continue

            total_cpu_usage += self.network.nodes[u]['CPU_cap'] - self.network.nodes[u]['free_CPU']
            total_ram_usage += self.network.nodes[u]['RAM_cap'] - self.network.nodes[u]['free_RAM']
            total_disk_usage += self.network.nodes[u]['MEM_cap'] - self.network.nodes[u]['free_MEM']

        for u, v in self.network.edges:
            total_bw_usage += self.network[u][v]['bw'] - self.network[u][v]['free_bw']

        return total_cpu_usage, total_ram_usage, total_disk_usage, total_bw_usage
    ########################################################################

    ########################################################################
    def compute_shortest_distances_between_all_pairs(self):
        self.shortest_distances = {}
        for s in self.network.nodes:
            self.shortest_distances[s] = {}
            for d in self.network.nodes:
                if s != d:
                    try:
                        self.shortest_distances[s][d] = networkx.dijkstra_path_length(self.network, s, d, weight='delay')  # the shortest length
                    except networkx.NetworkXException:
                        networkx.draw_networkx(self.network)
                        # Set margins for the axes so that nodes aren't clipped
                        ax = plt.gca()
                        ax.margins(0.20)
                        plt.axis("off")
                        plt.show()
                        assert False, "no path exists."
                else:
                    self.shortest_distances[s][d] = 0
                # end if

    ########################################################################

    ########################################################################
    def print_curves_to_file(self):
        # read graph file
        # raw_preprocessed_network_file_name = '../complete_data_sets/ibm_10000_slots_1_con.net'
        # network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name)
        # print(network.edges)

        # read SFC file
        # sfc_file_name = '../complete_data_sets/sfc_file.sfc'
        # sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)
        sim = Simulation(copy.deepcopy(self.empty_network), self.SFC_list)

        # read simulation's event
        simulation_file = self.simulation_output_file_name

        event_list = self.processed_events
        # set up parameters
        parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 1000, '#_time_slots': 10000}

        curves = {'time_points': [], 'cpu_usage_rate': [], 'ram_usage_rate': [], 'disk_usage_rate': [],
                  'bw_usage_rate': [], 'throughput_rate': [], 'offered_load': [], 'throughput': []}
        offered_load_every_slot_list = []

        total_cpu_cap, total_ram_cap, total_disk_cap, total_bw_cap = sim.get_total_resource_capacities()
        total_resource_capacities = {'total_cpu_cap': total_cpu_cap, 'total_ram_cap': total_ram_cap,
                                     'total_disk_cap': total_disk_cap, 'total_bw_cap': total_bw_cap}
        offered_load, through_put = 0, 0

        print('Simulation for curves is running ... ', datetime.now().time())  # this simulation run with 'accept' and 'terminate' events only

        for event_time_and_request_id, event in event_list:
            t = event_time_and_request_id[0]
            assert t == event['time_point'], "event_time_and_request_id is not the right one of this event !!!"
            if event['type'] == 'accept':
                sim.add_event(event)
                sim.process_next_event(print_simulation_to_file=False)
                # cpu_usage, ram_usage, disk_usage, bw_usage = sim.get_resource_usage()
                # time_slots.append(t)
                # cpu_usage_rate.append(cpu_usage / total_cpu_cap * 100.0)
                # ram_usage_rate.append(ram_usage / total_ram_cap * 100.0)
                # disk_usage_rate.append(disk_usage / total_disk_cap * 100.0)
                # bw_usage_rate.append(bw_usage / total_bw_cap * 100.0)
                through_put += event['sfc_request']['bw']
                # throughput_curve.append(through_put)
            elif event['type'] == 'terminate':
                sim.add_event(event)
                sim.process_next_event(print_simulation_to_file=False)
                through_put -= event['sfc_request']['bw']
            elif event['type'] == 'new':
                bw = event['sfc_request']['bw']
                arrival_time, end_time = event['sfc_request']['arrival_time'], event['sfc_request']['end_time']
                if end_time > len(offered_load_every_slot_list):
                    for i in range(len(offered_load_every_slot_list), end_time + 1):
                        offered_load_every_slot_list.append(0)

                for i in range(arrival_time, end_time):
                    offered_load_every_slot_list[i] += bw

            if event['type'] == 'accept':  # or event['type'] == 'terminate':
                append_values_to_curves(sim, t, curves, through_put, offered_load_every_slot_list, total_resource_capacities)

        # plot lines
        plt.rcParams.update({'font.size': 14})
        plt.figure(1)
        plt.clf()
        plt.plot(curves['time_points'], curves['cpu_usage_rate'], label="CPU usage")
        plt.plot(curves['time_points'], curves['ram_usage_rate'], label="RAM usage")
        plt.plot(curves['time_points'], curves['disk_usage_rate'], label="Storage usage")
        plt.plot(curves['time_points'], curves['bw_usage_rate'], label="Bandwidth usage")
        plt.plot(curves['time_points'], curves['throughput_rate'], label="Throughput/offered load rate")
        plt.legend()
        plt.ylabel("Usage rate (%)")
        plt.xlabel("Time slot")
        plt.savefig(simulation_file + '_main_curves.png')
        # plt.show()

        plt.figure(2)
        plt.clf()
        plt.plot(curves['time_points'], curves['throughput'], label="Throughput")
        plt.plot(curves['time_points'], curves['offered_load'], label="Offered load")
        plt.legend()
        plt.ylabel("Throughput (Gbps)")
        plt.xlabel("Time slot")
        plt.savefig(simulation_file + '_throughput_offered_load.png')
        # plt.show()
    ########################################################################

    ########################################################################
    # This function calculate the lower bound of delay assuming that the prefix part of the path is give, i.e., s->v
    # Node 'v' in layer 'layer_index'
    def get_remaining_delay(self, v, layer_index, sfc_request):
        sequent_vnfs = self.SFC_list[sfc_request['sfc_id']]['sequent_vnfs']
        min_remaining_delay = self.shortest_distances[v][sfc_request['destination']]
        for i in range(layer_index, len(self.SFC_list[sfc_request['sfc_id']]) - 1):
            processing_vnf = sequent_vnfs[i]
            processing_delay_rate = self.SFC_list["vnf_list"][processing_vnf]['PROC_rate']
            delay = sfc_request['bw'] * processing_delay_rate  # processing delay of i-th VNF
            min_remaining_delay += delay

        return min_remaining_delay
    ########################################################################
