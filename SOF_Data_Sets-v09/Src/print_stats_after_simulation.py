import copy
import matplotlib.pyplot as plt
from simulation import Simulation
from IO import IO
from datetime import datetime
import os

#########################################

using_existing_result = False

#########################################
def main():
    print('Hello', datetime.now().time())

    if using_existing_result:
        repeat_existing_simulation()
        print("End repeating the existing simulation result")
        # exit(0)
        return 0
    
    else:
        print("Start a new simulation")

    parameters = {'network_file': '../complete_data_sets/ibm_15000_slots_1_con.net',
                  'SFC_file': '../complete_data_sets/sfc_file.sfc',
                  'traffic_file': '../complete_data_sets/reordered_traffic_15000_slots_1_con.tra',
                  'simulation_output_file': 'reordered_traffic_15000_slots_1_con_RCSP_sim.txt'}

    # read graph file
    raw_preprocessed_network_file_name = parameters['network_file']
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name)
    # print(network.edges)

    # read SFC file
    sfc_file_name = parameters['SFC_file']
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)

    # read requests
    traffic_file_name = parameters['traffic_file']
    imported_sfc_requests = IO.read_traffic_from_json_file(traffic_file_name)

    # read simulation's event
    # simulation_file = './RCSP_sim.txt'
    # event_list = IO.read_events_from_simulation(simulation_file)
    # set up parameters
    # parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 1000, '#_time_slots': 10000}

    sim = Simulation(network, sfc_list)

    for sfc_request in imported_sfc_requests:
        event = {"time_point": sfc_request["arrival_time"], "type": 'new', 'sfc_request': sfc_request}
        sim.add_event(event)

    print("Simulation is running ... ", datetime.now().time())
    sim.set_routing_algorithm("RCSP")
    sim.simulation_output_file_name = parameters['simulation_output_file']
    sim.run(save_network_stats=False, save_processed_events=True, print_simulation_to_file=True, save_curves=True)

    print("Nicely done!!!", datetime.now().time())
#########################################


#########################################
def repeat_existing_simulation():
    print('Show the existing simulation result')
    
    # Dataset folder
    DATA_FOLDER = r'C:\Dropbox\SourceCode_WP1\SOF_Data_Sets-v09\test_data_sets'
    
    # read graph file
    # raw_preprocessed_network_file_name = '../complete_data_sets/ibm_15000_slots_1_con.net'
    raw_preprocessed_network_file_name = os.path.join(DATA_FOLDER, 'ibm_15000_slots_1_con.net')
    network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name)
    # print(network.edges)

    # read SFC file
    # sfc_file_name = '../complete_data_sets/sfc_file.sfc'
    sfc_file_name = os.path.join(DATA_FOLDER, 'sfc_file.sfc')
    sfc_file_name = os.path.abspath(sfc_file_name)
    sfc_list = IO.read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc=True)
    sim = Simulation(network, sfc_list)

    # read simulation's event
    # simulation_file = './reordered_traffic_10000_slots_1_con_RCSP_sim_backup.txt'
    # simulation_file = './reordered_traffic_15000_slots_1_con_RCSP_sim.txt'
    simulation_file = os.path.join(DATA_FOLDER, 'reordered_traffic_15000_slots_1_con_RCSP_sim.txt')
    event_list = IO.read_events_from_simulation(simulation_file)
    # set up parameters
    parameters = {'max_concurrent_request_per_time_slot': 1, 'average_duration': 1000, '#_time_slots': 15000}

    curves = {'time_points': [], 'cpu_usage_rate': [], 'ram_usage_rate': [], 'disk_usage_rate': [],
              'bw_usage_rate': [], 'throughput_rate': [], 'offered_load': [], 'throughput': []}
    offered_load_every_slot_list = []

    total_cpu_cap, total_ram_cap, total_disk_cap, total_bw_cap = sim.get_total_resource_capacities()
    total_resource_capacities = {'total_cpu_cap': total_cpu_cap, 'total_ram_cap': total_ram_cap,
                                 'total_disk_cap': total_disk_cap, 'total_bw_cap': total_bw_cap}
    offered_load, through_put = 0, 0

    print('Simulation is running ... ', )  # this simulation run with 'accept' and 'terminate' events only

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

            for i in range(arrival_time, end_time + 1):
                offered_load_every_slot_list[i] += bw

        if event['type'] == 'accept' or event['type'] == 'terminate':
            append_values(sim, t, curves, through_put, offered_load_every_slot_list, total_resource_capacities)

    # plot lines
    plt.rcParams.update({'font.size': 12})
    plt.figure(1)
    plt.plot(curves['time_points'], curves['cpu_usage_rate'], label="CPU usage")
    plt.plot(curves['time_points'], curves['ram_usage_rate'], label="RAM usage")
    plt.plot(curves['time_points'], curves['disk_usage_rate'], label="Storage usage")
    plt.plot(curves['time_points'], curves['bw_usage_rate'], label="Bandwidth usage")
    plt.plot(curves['time_points'], curves['throughput_rate'], label="Throughput/offered load rate")
    plt.legend()
    plt.ylabel("Usage rate (%)")
    plt.xlabel("Time slot")
    plt.savefig(simulation_file + '_main_curves.png')
    plt.show()

    plt.figure(2)
    plt.plot(curves['time_points'], curves['throughput'], label="Throughput")
    plt.plot(curves['time_points'], curves['offered_load'], label="Offered load")
    plt.legend()
    plt.ylabel("Throughput (Gbps)")
    plt.xlabel("Time slot")
    plt.savefig(simulation_file + '_throughput_offered_load.png')
    plt.show()

    #### Thinh: Export data to log file conforming with Thinh's log_file_format
    print('Exporting data to log file for easier to do performance comparison!')
    output_file = os.path.join(DATA_FOLDER, 'RCSP_sim_results.log')
    output_file = os.path.abspath(output_file)
    with open(output_file, 'w') as fp:
        print('Time_slots, Offered_load[bw unit], Real_throughput[bw unit], CPU_usage_rate[%], RAM_usage_rate[%], STO_usage_rate[%], BW_usage_rate[%]', file=fp)
    
    n_time_points = len(curves['time_points'])
    with open(output_file, 'a') as fp:
        for i in range(n_time_points):
            time = curves['time_points'][i]
            off_load = curves['offered_load'][i]
            real_tp = curves['throughput'][i]
            cpu_usage_rate = curves['cpu_usage_rate'][i]
            ram_usage_rate = curves['ram_usage_rate'][i]
            sto_usage_rate = curves['disk_usage_rate'][i]
            bw_usage_rate = curves['bw_usage_rate'][i]
            print(f'{time} {off_load} {real_tp} {cpu_usage_rate} {ram_usage_rate} {sto_usage_rate} {bw_usage_rate}', file=fp)
#########################################


#########################################
def append_values(sim, t, curves, throughput, offered_load_every_slot_list, total_resource_capacities):
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
#########################################


#########################################
if __name__ == "__main__":
    main()
