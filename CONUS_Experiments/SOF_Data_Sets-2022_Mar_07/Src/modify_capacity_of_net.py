import copy
# import sys
import matplotlib.pyplot as plt
from simulation import Simulation
from IO import IO
from datetime import datetime
from data_generator import get_delay_of_layered_graph_path
from data_generator import print_graph_to_file


########################################################################
# modify node/link capacity of a network
def main():
    parameters = {'#_slots': 20000,
                  'original_network_file': '../complete_data_sets/Data_11/non_uniformCONUS36_20000_slots_1_con.net',
                  'modified_network_file': '../complete_data_sets/Data_12/non_uniformCONUS36_20000_slots_1_con.net',
                  # 'sfc_file': '../complete_data_sets/Data_11/CONUS36_sfc_file.sfc',
                  # 'simulation_file': '../complete_data_sets/Data_11'
                  #                    '/non_uniform_CONUS36_reordered_traffic_20000_slots_1_con_RCSP_sim.txt',
                  # 'window_size': 500
                  }
    # read graph file
    # raw_preprocessed_network_file_name = '../complete_data_sets/varied_by_request/CONUS75_15000_slots_1_con.net'
    raw_preprocessed_network_file_name = parameters['original_network_file']
    original_network = IO.read_graph_from_text_file(raw_preprocessed_network_file_name, with_population=True)
    new_network = copy.deepcopy(original_network)

    for u, v in new_network.edges:
        new_network[u][v]['bw'] = original_network[u][v]['bw'] * 0.9
        new_network[u][v]['free_bw'] = new_network[u][v]['bw']

    print_graph_to_file(new_network, parameters['modified_network_file'])
#########################################


#########################################
if __name__ == "__main__":
    main()
