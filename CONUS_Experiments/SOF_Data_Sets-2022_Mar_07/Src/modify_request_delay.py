import copy
# import sys
import matplotlib.pyplot as plt
from simulation import Simulation
from IO import IO
from datetime import datetime
from data_generator import get_delay_of_layered_graph_path
from data_generator import print_traffic_to_file


########################################################################
# modify request delay
def main():
    parameters = {'#_slots': 20000,
                  'original_traffic_file': '../complete_data_sets/Data_11/non_uniform_CONUS36_reordered_traffic_20000_slots_1_con.tra',
                  'modified_traffic_file': '../complete_data_sets/Data_13/non_uniform_CONUS36_reordered_traffic_20000_slots_1_con.tra',
                  # 'sfc_file': '../complete_data_sets/Data_11/CONUS36_sfc_file.sfc',
                  # 'simulation_file': '../complete_data_sets/Data_11'
                  #                    '/non_uniform_CONUS36_reordered_traffic_20000_slots_1_con_RCSP_sim.txt',
                  # 'window_size': 500
                  }
    # read graph file
    # raw_preprocessed_network_file_name = '../complete_data_sets/varied_by_request/CONUS75_15000_slots_1_con.net'
    original_traffic_file_name = parameters['original_traffic_file']
    original_traffic = IO.read_traffic_from_json_file(original_traffic_file_name)
    requests = copy.deepcopy(original_traffic)

    request_map = {}

    for request in requests:
        request['delay_req'] *= 1.2
        request_map[request['id']] = request

    print_traffic_to_file(parameters['modified_traffic_file'], request_map, None)
#########################################


#########################################
if __name__ == "__main__":
    main()
