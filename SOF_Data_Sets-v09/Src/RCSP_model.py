from gurobipy import *


class RCSP_Model:

    ########################################################################
    # initialize an object
    def __init__(self):
        self.model = Model()
        # variable sets
        self.x = {}
        self.delta = {}
        self.phi = {}
        self.a = {}
        self.sfc_request = None
        self.network = None
        # constraint sets
        self.link_aggregation_consts = None
        self.link_usage_indicator = None
        self.layer_0_flow_conservation_consts = None
        self.last_layer_flow_conservation_consts = None
        self.intermediate_layer_flow_conservation_consts = None
        self.unique_location_consts = None
        self.x_delta_consistency_consts = None
        self.node_cpu_capacity_consts = None
        self.node_ram_capacity_consts = None
        self.node_disk_capacity_consts = None
        self.link_capacity_consts = None

    ########################################################################
    # set network
    def set_network(self, network):
        self.network = network

    ########################################################################
    # generate variables and constraints according to a SFC request
    def generate_model(self, sfc_request):
        self.sfc_request = sfc_request
        self.add_variables_to_model()

    ########################################################################
    # add variables to 'model'
    def add_variables_to_model(self):
        # 'x' variables
        for u in self.network.nodes:
            self.x[u] = {}
            for v in self.network[u]:
                self.x[u][v] = self.model.addVar(lb=0, ub=1, obj=0, vtype=GRB.BINARY,
                                                 name='x' + "_" + str(u) + "_" + str(v), column=None)

        # 'delta' variables
        for u in self.network.nodes:
            self.delta[u] = {}
            for v in self.network[u]:
                self.delta[u][v] = self.model.addVar(lb=0, ub=1, obj=0, vtype=GRB.BINARY,
                                                     name='delta' + "_" + str(u) + "_" + str(v), column=None)

        # 'a' variables
        sfc_specification = self.SFC_list[self.sfc_request["sfc_id"]]
        for i in range(sfc_specification['vnf_list']):
            self.a[i] = {}
            for u in self.network.nodes():
                self.a[i][u] = self.model.addVar(lb=0, ub=1, obj=0, vtype=GRB.BINARY,
                                                 name='a_' + str(i) + "_" + str(u), column=None)

        # 'phi' variables
        for i in range(self.sfc_request["vnf_list"] + 1):
            self.phi[i] = {}
            for u, v in self.network.edges():
                self.phi[i][u][v] = self.model.addVar(lb=0, ub=1, obj=0, vtype=GRB.BINARY,
                                                      name='phi_' + str(i) + "_" + str(u) + "_" + str(v), column=None)

    ########################################################################
    # add constraints to 'model'
    def add_constraints_to_model(self):
        num_vnfs = len(self.sfc_request["vnf_list"])
        # Aggregation of link usage
        self.link_aggregation_consts = {}
        for u in self.network.nodes():
            self.link_aggregation_consts[u] = {}
            for v in self.network[u].out_edge():
                exp = LinExpr(self.delta[u][v])
                for i in range(len(self.sfc_request["vnf_list"]) +1):
                    exp -= self.phi[i][u][v]

                self.link_aggregation_consts[u][v] = self.model.addConstr(exp == 0,
                                                                          name='link_aggregation_'+str(u)+"_"+str(v))

        # link usage indicator
        self.link_usage_indicator = {}
        for i in range(num_vnfs + 1):
            self.link_usage_indicator[i] = {}
            for u in self.network.nodes():
                self.link_usage_indicator[i][u] = {}
                for v in self.network[u].out_edge():
                    name = 'link_usage_indicator_' + str(i)+"_"+str(u) + "_" + str(v)
                    self.link_usage_indicator[i][u][v] = self.model.addConstr(self.phi[i][u][v] <= self.x[u][v],
                                                                              name=name)
        # flow conservation constraints
        self.add_flow_conservation_constraints()

        # unique location per VNF constraints
        self.unique_location_consts = {}
        for i in range(num_vnfs):
            exp = LinExpr()
            for v in self.network.nodes:
                exp += self.a[i][v]

            name = "unique_location_consts_" + str(i)
            self.unique_location_consts[i] = self.model.addConstr(exp == 1, name=name)

        # make sure 'x' and 'delta' that are consistent
        self.x_delta_consistency_consts = {}
        for u in self.network.nodes:
            self.x_delta_consistency_consts[u] = {}
            for v in self.network[u].out_edges():
                name = "x_delta_consistency_consts" + str(u) + "_" + str(v)
                self.x_delta_consistency_consts[u][v] = self.model.addConstr(self.x[u][v] <= self.delta[u][v],
                                                                             name=name)

        # node CPU capacity constraints
        self.node_cpu_capacity_consts = {}
        self.node_ram_capacity_consts = {}
        self.node_disk_capacity_consts = {}
        for v in self.network.nodes:
            cpu_exp = LinExpr()
            ram_exp = LinExpr()
            disk_exp = LinExpr()
            for i in range(num_vnfs):
                cpu_exp += self.sfc_request["bw"]*self.sfc_request["vnf_list"][i]["cpu_rate"]*self.a[i][v]
                ram_exp += self.sfc_request["bw"] * self.sfc_request["vnf_list"][i]["ram_rate"] * self.a[i][v]
                disk_exp += self.sfc_request["bw"] * self.sfc_request["vnf_list"][i]["disk_rate"] * self.a[i][v]

            cpu_name = "node_cpu_capacity_consts_" + str(v)
            self.node_cpu_capacity_consts[v] = self.model.addConstr(cpu_exp <= self.network[v]["CPU_cap"],
                                                                    name=cpu_name)

            ram_name = "node_ram_capacity_consts_" + str(v)
            self.node_ram_capacity_consts[v] = self.model.addConstr(ram_exp <= self.network[v]["RAM_cap"],
                                                                    name=ram_name)

            disk_name = "node_disk_capacity_consts_" + str(v)
            self.node_disk_capacity_consts[v] = self.model.addConstr(disk_exp <= self.network[v]["Disk_cap"],
                                                                     name=disk_name)

        # link capacity constraints
        self.link_capacity_consts = {}
        for u in self.network.nodes:
            self.link_capacity_consts[u] = {}
            for v in self.network[u].out_edges():
                name = "link_capacity_consts_" + str(u) + "_" + str(v)
                exp = LinExpr(self.sfc_request["bw"] * self.delta[u][v])
                self.link_capacity_consts[u][v] = self.model.addConstr(exp <= self.network[u][v]["free_cap"],
                                                                       name=name)

    ########################################################################
    # add flow conservation constraints to the model
    def add_flow_conservation_constraints(self):
        # layer 0 flow conservation constraints_1, i.e., layer 0
        self.layer_0_flow_conservation_consts = {}
        for v in self.network.nodes():
            exp = LinExpr(self.a[0][v])
            rhs = 1 if v == self.sfc_request["source"] else 0

            for u in self.network[v].in_edges():    # sum of in links
                exp += self.phi[0][u][v]
            for u in self.network[v].out_edges():   # minus sum of out links
                exp -= self.phi[0][v][u]

            name = "layer_0_flow_conservation_consts_" + str(v)
            self.layer_0_flow_conservation_consts[v] = self.model.addConstr(exp == rhs, name=name)

        # last layer flow conservation constraints_1, i.e., layer len(vnf list)
        self.last_layer_flow_conservation_consts = {}
        k = len(self.sfc_request["vnf_list"])
        for v in self.network.nodes():
            exp = LinExpr(-self.a[k-1][v])
            rhs = -1 if v == self.sfc_request["destination"] else 0

            for u in self.network[v].in_edges():  # sum of in links
                exp += self.phi[k][u][v]
            for u in self.network[v].out_edges():  # minus sum of out links
                exp -= self.phi[k][v][u]

            name = "last_layer_flow_conservation_consts_" + str(v)
            self.last_layer_flow_conservation_consts[v] = self.model.addConstr(exp == rhs, name=name)

        # intermediate layer flow conservation constraints
        self.intermediate_layer_flow_conservation_consts = {}
        for i in range(1, k):
            self.intermediate_layer_flow_conservation_consts[i] = {}
            for v in self.network.nodes():
                exp = LinExpr(self.a[i][v] - self.a[i-1][v])

                for u in self.network[v].in_edges():  # sum of in links
                    exp += self.phi[i][u][v]
                for u in self.network[v].out_edges():  # minus sum of out links
                    exp -= self.phi[i][v][u]

                name = "intermediate_layer_flow_conservation_consts_" + str(v)
                self.intermediate_layer_flow_conservation_consts[i][v] = self.model.addConstr(exp == rhs, name=name)
