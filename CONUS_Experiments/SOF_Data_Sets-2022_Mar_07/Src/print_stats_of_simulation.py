#########################################
# Author: Huy Duong, CRIM
# Simulation to run a dynamic process
#########################################

import copy
# import sys
import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from IO import IO
from datetime import datetime
from data_generator import get_delay_of_layered_graph_path
import pandas as pd
# from iteration_utilities import duplicates
#########################################


########################################################################
def append_values_to_curves(sim, t, curves, throughput, offered_load_every_slot_list,
                            total_resource_capacities,
                            # total_accepted_requests, total_requests,
                            # window_size
                            ):
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
    # if t % window_size == 0:
    #     curves['acceptance_check_points'].append(t)
    #     curves['acceptance'].append(total_accepted_requests / total_requests * 100.0)

    if offered_load_every_slot_list[t] > 0:
        curves['throughput_rate'].append(throughput / offered_load_every_slot_list[t] * 100.0)
    else:
        assert throughput == 0, 'Throughput must be 0 !!!'
        curves['throughput_rate'].append(0)
########################################################################


########################################################################
def plot_delay_requirement_stress(simulation, sorted_event_list, simulation_file):
    delay_stress_list = []

    for _, event in sorted_event_list:
        if event['type'] != 'accept':
            continue

        delay_req = event['sfc_request']['delay_req']
        path_delay = get_delay_of_layered_graph_path(simulation, event['sfc_request'], event['path'])
        delay_stress = path_delay / delay_req * 100.0
        # we only consider accepted events for this plot
        delay_stress_list.append(delay_stress)

    plt.figure(4)
    plt.clf()
    plt.hist(delay_stress_list, bins=10, range=[0, 100])
    # plt.plot(curves['time_points'], curves['offered_load'], label="Offered load")
    # plt.legend()
    # plt.ylabel("")
    plt.xlabel("Delay stress level")
    plt.savefig(simulation_file + '_delay_stress.png')

########################################################################


########################################################################
def main():
    parameters = {'#_slots': 20000,
                  'network_file': '../complete_data_sets/Data_18/non_uniformCONUS36_20000_slots_2_con.net',
                  'sfc_file': '../complete_data_sets/Data_18/CONUS36_sfc_file.sfc',
                  'simulation_file': '../complete_data_sets/Data_18/non_uniform_CONUS36_reordered_traffic_20000_slots_2_con_RCSP_sim.txt',
                  'window_size': 500}
    # read graph file
    # raw_preprocessed_network_file_name = '../complete_data_sets/varied_by_request/CONUS75_15000_slots_1_con.net'
    raw_preprocessed_network_file_name = parameters['network_file']
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name, with_population=True)
    # for node in network.nodes.values():
    #     node["STO_cap"] = node["free_STO"] = sys.maxsize
    #     node["RAM_cap"] = node["free_RAM"] = sys.maxsize
    #     node["CPU_cap"] = node["free_CPU"] = sys.maxsize
    # print(network.edges)

    # read SFC file
    sfc_file_name = parameters['sfc_file']
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)
    sim = Simulation(network, sfc_list)
    window_size = parameters['window_size']

    # read simulation's event
    # simulation_file = './reordered_traffic_10000_slots_1_con_RCSP_sim_backup.txt'
    # simulation_file = './reordered_traffic_10000_slots_1_con_RCSP_sim.txt'
    simulation_file = parameters['simulation_file']
    event_list = IO.read_events_from_simulation(simulation_file)
    # set up parameters
    # parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 2000, '#_time_slots': 15000}

    curves = {'time_points': [], 'cpu_usage_rate': [], 'ram_usage_rate': [], 'disk_usage_rate': [],
              'bw_usage_rate': [], 'throughput_rate': [], 'offered_load': [], 'throughput': [],
              'acceptance_check_points': [], 'acceptance': []}
    offered_load_every_slot_list = []

    total_cpu_cap, total_ram_cap, total_disk_cap, total_bw_cap = sim.get_total_resource_capacities()
    total_resource_capacities = {'total_cpu_cap': total_cpu_cap, 'total_ram_cap': total_ram_cap,
                                 'total_disk_cap': total_disk_cap, 'total_bw_cap': total_bw_cap}
    offered_load, through_put = 0, 0
    segment_accepted_requests, segment_arrival_requests = 0, 0
    overall_requests, overall_acc_requests = 0, 0

    print('Simulation is running ... ', )  # this simulation run with 'accept' and 'terminate' events only
    sorted_event_list = copy.deepcopy(event_list)
    sorted_event_list = sorted(sorted_event_list, key=lambda k: k[0])
    number_requests_map = {}

    for event_time_and_request_id, event in sorted_event_list:
        t = event_time_and_request_id[0]
        assert t == event['time_point'], "event_time_and_request_id is not the right one of this event !!!"
        if event['type'] == 'accept':
            overall_acc_requests += 1
            # last_accept_point = t
            sim.add_event(event)
            sim.process_next_event(print_simulation_to_file=False)

            segment_accepted_requests += 1
            through_put += event['sfc_request']['bw']
            append_values_to_curves(sim, t, curves, through_put, offered_load_every_slot_list,
                                    total_resource_capacities,
                                    # segment_accepted_requests, segment_arrival_requests,
                                    # window_size=window_size
                                    )
        elif event['type'] == 'terminate':
            sim.add_event(event)
            sim.process_next_event(print_simulation_to_file=False)
            through_put -= event['sfc_request']['bw']
        elif event['type'] == 'new':
            request = event['sfc_request']
            s, d = request['source'], request['destination']
            if s not in number_requests_map:
                number_requests_map[s] = {}

            if d not in number_requests_map[s]:
                number_requests_map[s][d] = 0

            number_requests_map[s][d] += 1

            # print(t)
            if t == 6000 and t != curves['acceptance_check_points'][-1]:
                overall_requests = overall_acc_requests = 0
            if t > 0 and t % window_size == 0 and \
                    (len(curves['acceptance_check_points']) == 0 or t != curves['acceptance_check_points'][-1]):  # reset counting at a steady point
                curves['acceptance_check_points'].append(t)
                if segment_accepted_requests <= 0:
                    assert False, 'accepted requests cannot be 0'
                curves['acceptance'].append(segment_accepted_requests / segment_arrival_requests * 100.0)
                segment_accepted_requests = segment_arrival_requests = 0

            overall_requests += 1
            segment_arrival_requests += 1
            bw = event['sfc_request']['bw']
            arrival_time, end_time = event['sfc_request']['arrival_time'], event['sfc_request']['end_time']
            if end_time > len(offered_load_every_slot_list):
                for i in range(len(offered_load_every_slot_list), end_time + 1):
                    offered_load_every_slot_list.append(0)

            for i in range(arrival_time, end_time):
                offered_load_every_slot_list[i] += bw

    if curves['time_points'][-1] != parameters['#_slots']:
        t = parameters['#_slots']
        # append_values_to_curves(sim, t, curves, through_put, offered_load_every_slot_list,
        #                         total_resource_capacities,
        #                         # segment_accepted_requests, segment_arrival_requests,
        #                         # window_size=window_size
        #                         )
        curves['acceptance_check_points'].append(t)
        curves['acceptance'].append(segment_accepted_requests / segment_arrival_requests * 100.0)

    print('number_requests_map:', number_requests_map)

    # todo: print heat maps of node and link average usage

    # n_nodes = network.number_of_nodes()
    # data = np.zeros((n_nodes, n_nodes))
    # for s in number_requests_map:
    #     for d in number_requests_map[d]:
    #         data[s][d] = number_requests_map[s][d]
    # data = pd.DataFrame.from_dict(number_requests_map, orient='columns', dtype=None)
    # print(data)
    # s = 0
    # for d1 in data:
    #     for d2 in data[d1]:
    #         if data[d1][d2] is not None:
    #             s += data[d1][d2]
    # print(s)
    # plt.figure(0)
    # plt.imshow(data)
    # # # plt.title("2-D Heat Map")
    # plt.colorbar()
    # plt.savefig(simulation_file + '_request_dist.png')

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
    plt.ylabel("rate (%)")
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

    overall_acc_rate = round(overall_acc_requests / overall_requests * 100.0, 2)
    plt.figure(3)
    plt.clf()
    plt.plot(curves['acceptance_check_points'], curves['acceptance'], label="acceptance rate")
    # plt.plot(curves['time_points'], curves['offered_load'], label="Offered load")
    plt.legend()
    plt.ylabel("Acceptance rate")
    plt.xlabel("Time slot")
    plt.annotate(str(overall_acc_rate), xy=(curves['acceptance_check_points'][-1], overall_acc_rate))
    plt.axhline(overall_acc_rate, linestyle='--')
    plt.savefig(simulation_file + '_acceptance_rate.png')

    # plot delay requirement stress
    plot_delay_requirement_stress(sim, sorted_event_list, simulation_file)
#########################################


#########################################
def append_values(sim, t, curves, throughput, offered_load_every_slot_list,
                  total_resource_capacities, total_accepted_requests):
    try:
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
        curves['throughput_rate'].append(throughput / offered_load_every_slot_list[t] * 100.0)
        curves['acceptance'].append(total_accepted_requests)
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        print('Simulation::add_event()', datetime.now().time())
        assert False, "append_values"
#########################################


#########################################
if __name__ == "__main__":
    main()
