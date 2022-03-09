import networkx
import copy
import json
#########################################


#########################################
class IO:
    #########################################
    # procedure to read network from text file
    @staticmethod
    def read_graph_from_text_file(network_file_name, with_population):
        file = open(network_file_name, 'r')
        network = networkx.DiGraph()

        # line_list = file.readlines()
        line_iter = iter(file.readlines())
        line = next(line_iter)
        num_nodes = int(line.split()[0])
        next(line_iter)  # 2 lines present format
        line = next(line_iter)
        # print(line)  # format
        for i in range(num_nodes):
            line = next(line_iter).split()
            # print(line)
            node_id, city_name, state, pop, lat, long = int(line[0]), line[1], line[2], int(line[3]), float(line[4]), float(line[5])
            # print(line[2], line[3])
            # state, pop = line[2], int(line[3])
            assert node_id == i, "wrong node id order"
            network.add_node(node_id, city_name=city_name, state=state, population=pop, latitude=lat, longitude=long)
            network.nodes[node_id]["VNF_list"] = []

        line = next(line_iter)
        num_links = int(line.split()[0])
        line = next(line_iter)  # indicative line
        line = next(line_iter)  # format line
        # num_nodes = int(line_list[0].split()[0])
        # num_links = int(line_list[1].split()[0])

        # initialize graph's nodes
        # network.add_nodes_from([x for x in range(num_nodes)])
        # for u in network.nodes:
        #     node_id, city_name, state = line_list

        # read edge list
        for link_id in range(num_links):
            line = next(line_iter)
            # print(line)
            u, v, bw, delay = [float(x) for x in line.split()]  # full-duplex edge
            free_cap = bw
            network.add_edge(int(u), int(v), bw=bw, delay=delay, free_bw=free_cap)  # first direction
            # network.add_edge(v, u, bw=bw, delay=delay, free_bw=free_cap)  # second direction

        # read nodes whose deployable VNFs
        # formation: DC's id, connected router, CPU (#), RAM (GB), storage (GB), # VNFs, VNF list
        next(line_iter)  # format line
        line = next(line_iter)
        # print(line)
        num_data_centers = int(line.split()[0])
        start_line_of_DC_list = 6 + num_links
        for i in range(start_line_of_DC_list, start_line_of_DC_list + num_data_centers):
            line = next(line_iter)
            dc_id, v, cpu, ram, disk, num_vnf = line.split()[0:6]
            num_vnf = int(num_vnf)
            v = int(v)
            network.nodes[v]["DC_ID"] = dc_id
            network.nodes[v]["free_CPU"] = network.nodes[v]["CPU_cap"] = float(cpu)
            network.nodes[v]["free_RAM"] = network.nodes[v]["RAM_cap"] = float(ram)
            network.nodes[v]["free_STO"] = network.nodes[v]["STO_cap"] = float(disk)
            network.nodes[v]["VNF_list"] = line.split()[6:6 + num_vnf]

        file.close()
        return network
    #########################################

    #########################################
    @staticmethod
    def read_SFC_file_from_text_file(sfc_file_name, is_sequent_sfc):
        file = open(sfc_file_name, 'r')
        lines = file.readlines()

        sfc_list = {'vnf_list': {}}
        num_vnfs = int(lines[1].split()[0])
        for i in range(2, num_vnfs + 2):
            # vnf_id, cpu_rate, ram_rate, mem_rate, proc_rate = lines[i].split()[0:5]
            vnf_id, cpu_rate, ram_rate, sto_rate, proc_rate = lines[i].split()[0:5]
            sfc_list['vnf_list'][vnf_id] = {'id': vnf_id, 'CPU_rate': float(cpu_rate), 'RAM_rate': float(ram_rate),
                                            'STO_rate': float(sto_rate), 'PROC_rate': float(proc_rate)}

        num_sfcs = int(lines[num_vnfs + 3].split()[0])
        start_line_of_sfc_list = num_vnfs + 6
        j = start_line_of_sfc_list
        for i in range(num_sfcs):
            sfc_id, num_vnfs_in_sfc = lines[j].split()[0:2]
            j += 1
            num_vnfs_in_sfc = int(num_vnfs_in_sfc)
            sfc_list[sfc_id] = {'SFC_ID': sfc_id, 'relations': {}}
            for k in range(0, num_vnfs_in_sfc):
                pre_vnf, succeeding_vnf = lines[j].split()[0:2]
                j += 1
                sfc_list[sfc_id]['relations'][pre_vnf] = succeeding_vnf

            if is_sequent_sfc:  # if each SFC is a sequence of VNFs, then we build a ordered list
                temp_relations = copy.copy(sfc_list[sfc_id]['relations'])
                first_vnf_marker = [x for x in temp_relations.keys()]
                for pre_vnf, succeeding_vnf in temp_relations.items():
                    if succeeding_vnf in first_vnf_marker:
                        first_vnf_marker.remove(succeeding_vnf)

                assert len(first_vnf_marker) == 1, "relations are inconsistent"

                v = first_vnf_marker[0]
                sfc_list[sfc_id]['sequent_vnfs'] = [v]
                while len(temp_relations) > 0:
                    sfc_list[sfc_id]['sequent_vnfs'].append(temp_relations[v])
                    vv = temp_relations[v]
                    temp_relations.pop(v)
                    v = vv

        file.close()
        return sfc_list
    #########################################

    #########################################
    @staticmethod
    def read_traffic_from_json_file(traffic_file_name):
        file = open(traffic_file_name, 'r')
        data = json.load(file)
        sfc_requests = data['requests']
        # print(sfc_requests)
        file.close()
        return sfc_requests
    #########################################

    #########################################
    @classmethod
    def read_events_from_simulation(cls, simulation_file):
        event_list = []
        my_file = open(simulation_file, "r")
        data = json.load(my_file)
        # print(data)
        for e in data['events']:
            event_list.append(e)

        return event_list
    #########################################
