import random

import numpy
import feasibility
import heuristic
# import heuristic2phys
import throughput
import time
import math
import json
import matplotlib.pyplot as plt
import os
from settings import Settings
import argparse
import logging
import sys
import exhaustive
import dijkstra
import copy

from deap import base
from deap import creator
from deap import tools

class GAWrapper:
    def __init__(self, settings_file=None):
        self.settings = Settings(settings_file)
        logging.debug("Setting seed to {0}".format(self.settings["seed"]))
        random.seed(self.settings["seed"])

        if "only_50" in self.settings["sf-heuristic"]["testbed_results"] and self.settings["sf-heuristic"]["testbed_results"]["only_50"] == 1:
            del self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]["TSCH_SLOTBONDING_1000_KBPS_PHY"]

        # tsch variables
        self.slot_length = self.settings["simulator"]["slotDuration"]
        self.nr_slots = self.settings["simulator"]["slotframeLength"]
        self.nr_frequencies = self.settings["simulator"]["numChans"]

        # topology variables
        self.nodes = [] # all nodes
        self.nodes_0 = [] # nodes without root
        self.d_parent = {} # maps a node to its parent
        self.distance_to_root = {}
        self.d_parent_closest_to_root = {} # the dictionary that maps a node to its possible parent that is most close to the root
        self.d_parent_closer_to_root = {} # the dictionary that maps a node to its possible parent that is most close to the root
        self.d_interferers = {} # maps a node to its interferers

        self.nodes_d_hst_ix = {}
        self.nodes_d_ix_hst = {}

        # read from modulations file
        self.d_MCS_to_slots = {} # maps MCS to number of slots in one bonded slot
        self.d_MCS_to_max_bonded_slots = {} # maps MCS to max. nr of bonded slots in slot frame
        self.d_MCS_to_index = {} # maps MCS to an index (I think a list is not as clear)
        self.d_index_to_MCS = {} # maps the index to the MCS
        self.d_MCS_to_rate = {} # maps MCS to an index (I think a list is not as clear)
        # read from topology file
        self.d_pdr = {} # maps node to MCS to parent to PDR, read literally from topology file
        # parsed based on GA experiment type
        self.d_node_to_allowed_parents = {} # maps a node to its allowed parents in the individual
        self.d_node_to_allowed_MCSs_per_parent = {} # maps a node to its allowed MCSs per parent in the individual

        self.not_equal_count = 1

        if self.settings["modulations"]["modulations_file"]:
            self.__init_modulations(self.settings["modulations"]["modulations_file"])
        else:
            raise Exception('No modulations file given.')

        if self.settings["topology"]["topology_file"]:
            self.__init_topology(self.settings["topology"]["topology_file"])
        else:
            raise Exception('No topology file given.')

        # initialize the feasibility model
        self.feasibility_model = feasibility.Feasibility(nr_slots=self.nr_slots, nr_frequencies=self.nr_frequencies, slots_per_MCS=self.d_MCS_to_slots, settings_file=settings_file)
        # initialize the heuristic feasibility model
        self.heuristic_model = heuristic.Heuristic(nr_slots=self.nr_slots, nr_frequencies=self.nr_frequencies, slots_per_MCS=self.d_MCS_to_slots, settings_file=settings_file)
        # initialize the throughput model
        self.throughput_model = throughput.Throughput(r_max=self.settings["tsch"]["r_max"], max_queue_length=self.settings["tsch"]["queue_size"], generated_packets_at_node=self.settings["tsch"]["generated_packets"], nr_slots=self.nr_slots, settings_file=settings_file)
        # read all the necessary p_files from pdr_min until pdr 1.0 into memory
        for i in range(int(self.settings["ga"]["min_pdr"]*1000), int(self.settings["ga"]["max_pdr"]*1000) + 1):
            self.throughput_model.set_p(dir_name=self.settings["ga"]["p_files_dir"], pdr=i/1000.0)
            self.print_progress_bar(iteration=i - int(self.settings["ga"]["min_pdr"]*1000), total=int(self.settings["ga"]["max_pdr"]*1000) + 1 - int(self.settings["ga"]["min_pdr"]*1000), prefix='Progress', suffix='of PDR files (PDR {0} - {1}) in memory'.format(self.settings["ga"]["min_pdr"], self.settings["ga"]["max_pdr"]), length=50)

        # stats
        self.valid_tree_time = 0.0
        self.valid_tree_exec = 0.0
        self.feasibility_time = 0.0
        self.feasibility_exec = 0
        self.feasibility_feasible = 0
        self.heuristic_time = 0.0
        self.heuristic_exec = 0
        self.heuristic_feasible = 0
        self.heuristic_false_positives = 0
        self.heuristic_false_negatives = 0
        self.heuristic_true_positives = 0
        self.heuristic_true_negatives = 0
        self.throughput_time = 0.0
        self.throughput_exec = 0
        self.mutation_exec = 0
        self.mutation_exec_total = 0
        self.mutation_time = 0.0
        self.crossover_exec = 0
        self.crossover_exec_total = 0
        self.crossover_time = 0.0
        self.total_time = 0.0

        # GA deap instances
        # log book to keep records of stats
        self.logbook = tools.Logbook()
        # hall of fame to keep best individuals
        self.hof = tools.HallOfFame(self.settings["ga"]["hall_of_fame_size"])

        self.best_individual = None
        self.best_individual_performance = []
        self.unique_individuals = {}
        self.unique_individuals_performance = []
        self.valid_individuals = set()
        self.valid_individuals_performance = []
        self.total_individuals = 0

        self.infeasible_inds_performance = []

        # saves those values here so you do not have to recalculate them always
        self.parent_selection_tournament_size = self.settings["ga"]["parent_selection"]["tournament"]["size"]
        self.survivor_selection_tournament_size = self.settings["ga"]["survivor_selection"]["tournament"]["size"]
        if not (self.settings["ga"]["survivor_selection"]["elitism"]["percentage"] * self.settings["ga"]["pop_size"]).is_integer():
            raise BaseException("Should be integer for elitism survivor selection!")
        self.survivor_selection_elitism_pop_slice = int(self.settings["ga"]["survivor_selection"]["elitism"]["percentage"] * self.settings["ga"]["pop_size"])
        self.survivor_selection_elitism_offspring_slice = int((1.0 - self.settings["ga"]["survivor_selection"]["elitism"]["percentage"]) * self.settings["ga"]["pop_size"])

    def __init_topology(self, json_topology_file):
        if len(self.settings["sf-heuristic"]["testbed_results"]) > 0:
            nodes_hostnames = []
            first = True
            # 1) check if the hostnames over the different files check out
            for phy in self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]:
                with open(self.settings["sf-heuristic"]["testbed_results"]["reliabilities"][phy]) as json_file:
                    data = json.load(json_file)
                    if first:
                        first = False
                        nodes_hostnames = [hstname for hstname in data.keys()]
                    else:
                        tmp_nodes_hostnames = [hstname for hstname in data.keys()]
                        diff = list(set(nodes_hostnames) - set(tmp_nodes_hostnames))
                        if len(diff) > 0:
                            raise Exception('There is difference ({diff}) in node hostnames for the first reliability file and {file}, check it.'.format(diff=diff,file=self.settings["sf-heuristic"]["testbed_results"]["reliabilities"][phy]))
            # 2) order alphabetically
            nodes_hostnames = sorted(nodes_hostnames, reverse=True)
            if (self.settings["sf-heuristic"]["testbed_results"]["root"]) not in nodes_hostnames:
                raise Exception("Root node ({root}) is not in hostnames ({nodes_hostnames}).".format(root=self.settings["sf-heuristic"]["testbed_results"]["root"],nodes_hostnames=nodes_hostnames))
            # 3) get nodes and nodes_0
            # remove the root from the list
            nodes_hostnames.remove(self.settings["sf-heuristic"]["testbed_results"]["root"])
            # substract one node from the total number nodes, so the root can be added later to the list
            nodes_hostnames = random.sample(nodes_hostnames, self.settings['simulator']['numMotes'] - 1)
            # place the root in the front of the list so it can get index 0
            nodes_hostnames = sorted(nodes_hostnames, reverse=True)
            nodes_hostnames.insert(0, self.settings["sf-heuristic"]["testbed_results"]["root"])
            print(nodes_hostnames)
            # print(list(set(nodes_hostnames)))
            # exit()
            if len(list(set(nodes_hostnames))) != self.settings['simulator']['numMotes']:
                raise Exception('Not the correct number of nodes.')
            # get the translating dictionary from hostname to node id
            self.nodes_d_hst_ix = {k: int(v) for v, k in enumerate(nodes_hostnames)}
            self.nodes_d_ix_hst = {int(v): k for v, k in enumerate(nodes_hostnames)}
            # print(nodes_d_ix_hst)
            # get all nodes
            self.nodes = sorted(range(len(self.nodes_d_hst_ix)))
            # only keep all nodes except the root
            self.nodes_0 = [n for n in self.nodes if n != 0]
            # 4) get all reliabilities per node couple
            for phy in self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]:
                print(phy)
                with open(self.settings["sf-heuristic"]["testbed_results"]["reliabilities"][phy]) as json_file:
                    data = json.load(json_file)
                    for n1 in self.nodes:
                        if n1 not in self.d_pdr:
                            self.d_pdr[n1] = {}
                        # if n1 != self.settings["topology"]["root"]:
                        #     self.d_parent[n1] = None
                        if phy not in self.d_pdr[n1]:
                            self.d_pdr[n1][phy] = {}
                        for n2 in self.nodes:
                            if n1 != n2:
                                self.d_pdr[n1][phy][n2] = data[self.nodes_d_ix_hst[n1]][self.nodes_d_ix_hst[n2]]
                        # we are going to include all (other) nodes as interferers
                        self.d_interferers[n1] = copy.copy(self.nodes)
                        self.d_interferers[n1].remove(n1)
        else:
            with open(json_topology_file) as json_file:
                data = json.load(json_file)

                # read out topology
                for node, info in data["simulationTopology"].items():
                    node = int(node) # convert it to an int
                    self.nodes.append(node)
                    if node != self.settings["topology"]["root"]:
                        self.nodes_0.append(node)
                        # save the parents
                        self.d_parent[node] = int(info["parent"])
                    self.d_interferers[node] = info["interferers"]

                    # save the reliabilities
                    self.d_pdr[node] = {}
                    # I am manually reading it out because I want to convert all the nodes to actual integers
                    for mcs, reliabilities in data["simulationTopology"][str(node)]["reliability"].items():
                        if mcs not in self.d_pdr[node]:
                            self.d_pdr[node][mcs] = {}
                        for p, pdr in reliabilities.items():
                            self.d_pdr[node][mcs][int(p)] = pdr

                    self.distance_to_root[node] = info["distance_to_root"]

    def __init_modulations(self, json_modulations_file):
        if len(self.settings["sf-heuristic"]["testbed_results"]) > 0:
            ix_phy = 0
            # if "only_50" in self.settings["sf-heuristic"]["testbed_results"] and self.settings["sf-heuristic"]["testbed_results"]["only_50"] == 1:
            #     del self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]["TSCH_SLOTBONDING_1000_KBPS_PHY"]
            
            for phy in self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]:
                self.d_MCS_to_rate[phy] = self.settings["sf-heuristic"]["testbed_results"]["rates"][phy]
                self.d_MCS_to_slots[phy] = self.settings["sf-heuristic"]["testbed_results"]["slots"][phy]
                self.d_MCS_to_index[phy] = ix_phy
                self.d_index_to_MCS[ix_phy] = phy
                ix_phy += 1

            self.d_MCS_to_max_bonded_slots = {}
            for mcs, slots in self.d_MCS_to_slots.items():
                self.d_MCS_to_max_bonded_slots[mcs] = math.floor(self.nr_slots / slots)
        else:
            with open(json_modulations_file) as json_file:
                data = json.load(json_file)

                self.d_MCS_to_rate = copy.deepcopy(data['modulations']['modulationRates'])

                if self.settings["simulator"]["modulationConfig"] in data["configurations"]:
                    for ix, m in enumerate(data['configurations'][self.settings["simulator"]["modulationConfig"]]['allowedModulations']):
                        self.d_MCS_to_index[m] = ix
                        self.d_index_to_MCS[ix] = m
                    self.d_MCS_to_slots = copy.deepcopy(data['configurations'][self.settings["simulator"]["modulationConfig"]]['modulationSlots'])
                else:
                    raise BaseException("Modulation config {0} not in in modulations file.".format(self.settings["simulator"]["modulationConfig"]))

                self.d_MCS_to_max_bonded_slots = {}
                for mcs, slots in self.d_MCS_to_slots.items():
                    self.d_MCS_to_max_bonded_slots[mcs] = math.floor(self.nr_slots / slots)

        # print(self.d_MCS_to_index)
        # print(self.d_index_to_MCS)
        # print(self.d_MCS_to_slots)
        # print(self.d_MCS_to_max_bonded_slots)
        # exit()

    #### Topology related methods ####

    def calculate_descendants(self, node=0, children=[], l_descendants=[]):
        if node not in l_descendants:
            # self.descendants[node] = 0
            l_descendants[node] = []
        if node in children:
            # self.descendants[node] += len(self.children[node])
            l_descendants[node] += children[node]
            for c in children[node]:
                self.calculate_descendants(node=c, children=children, l_descendants=l_descendants)
                # self.descendants[node] += self.descendants[c]
                l_descendants[node] += l_descendants[c]

    def get_parents(self, ind):
        parents = {}
        tmp_node = 1
        while tmp_node <= len(self.nodes_0):  # calculate all the children
            node_ix = ((tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"])
            parents[tmp_node] = ind[node_ix + 2]
            tmp_node += 1
        return parents

    def reach_root(self, node, par_parents, parents):
        if node not in par_parents:
            return False
        elif par_parents[node] in parents:
            return False
        elif par_parents[node] == self.settings["topology"]["root"]:
            return True
        parents.append(par_parents[node])
        return self.reach_root(par_parents[node], par_parents, parents)

    def valid_individual_topology(self, ind):
        parents = self.get_parents(ind)
        for n in self.nodes:
            if n != 0 and not self.reach_root(n, parents, []):
                return False
        return True

    def fast_feasibility_check(self, individual, dict_children):
        # 1) check if all the children of the root are not sending too much
        nr_slots_children_root = 0
        for c_root in dict_children[self.settings["topology"]["root"]]:
            ix_c_root = (c_root - 1) * self.settings["ga"]["genes_per_node_with_topology"]
            nr_slots_child_root = individual[ix_c_root + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_c_root]]]  # + 1 to get the slot count
            nr_slots_children_root += nr_slots_child_root
        if nr_slots_children_root > self.nr_slots:
            return False
        # 2) check if the total number of slots exceeds the total number of slots in the slotframe
        n = 1
        while n <= len(self.nodes_0):  # calculate all the children, <= because you start at 1
            ix_n = (n - 1) * self.settings["ga"]["genes_per_node_with_topology"]
            # multiply the number of slots with the number of slots of the MCS used
            nr_slots_node = individual[ix_n + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_n]]]
            # logging.debug("Node {0} has a total of {1} slots itself.".format(n, nr_slots_node))
            if nr_slots_node > self.nr_slots:
                return False
            nr_slots_children = 0
            if n in dict_children:
                for c in dict_children[n]:
                    ix_c = (c - 1) * self.settings["ga"]["genes_per_node_with_topology"]
                    nr_slots_child = individual[ix_c + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_c]]] # + 1 to get the slot count
                    nr_slots_children += nr_slots_child
                    # logging.debug("Adding {0} slots for child {1} of node {2}, bringing the total to {3}.".format(nr_slots_child, c, n, nr_slots_children))
            if nr_slots_node + nr_slots_children > self.nr_slots:
                return False
            n += 1
        return True

    def ixs_fast_feasibility_check(self, individual, dict_children):
        ixs = []
        # 1) check if all the children of the root are not sending too much
        nr_slots_children_root = 0
        for c_root in dict_children[self.settings["topology"]["root"]]:
            ix_c_root = (c_root - 1) * self.settings["ga"]["genes_per_node_with_topology"]
            nr_slots_child_root = individual[ix_c_root + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_c_root]]]  # + 1 to get the slot count
            nr_slots_children_root += nr_slots_child_root
            ixs.append(ix_c_root + 1) # add the index
        if nr_slots_children_root > self.nr_slots:
            return ixs
        ixs = []
        # 2) check if the total number of slots exceeds the total number of slots in the slotframe
        n = 1
        while n <= len(self.nodes_0):  # calculate all the children, <= because you start at 1
            ix_n = (n - 1) * self.settings["ga"]["genes_per_node_with_topology"]
            # multiply the number of slots with the number of slots of the MCS used
            nr_slots_node = individual[ix_n + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_n]]]
            # logging.debug("Node {0} has a total of {1} slots itself.".format(n, nr_slots_node))
            if nr_slots_node > self.nr_slots:
                return [ix_n + 1]
            nr_slots_children = 0
            if n in dict_children:
                for c in dict_children[n]:
                    ix_c = (c - 1) * self.settings["ga"]["genes_per_node_with_topology"]
                    nr_slots_child = individual[ix_c + 1] * self.d_MCS_to_slots[self.d_index_to_MCS[individual[ix_c]]] # + 1 to get the slot count
                    nr_slots_children += nr_slots_child
                    ixs.append(ix_c + 1)
                    # logging.debug("Adding {0} slots for child {1} of node {2}, bringing the total to {3}.".format(nr_slots_child, c, n, nr_slots_children))
            if nr_slots_node + nr_slots_children > self.nr_slots:
                ixs.append(ix_n + 1)
                return list(set(ixs))
            n += 1
            ixs = [] # reset
        return []

    # def check_valid_nr_slots(self, individual, dict_children):
    #     tmp_node = 1
    #     total_slots = 0
    #     involved_indices = []
    #     while tmp_node <= len(self.nodes_0):  # calculate all the children, <= because you start at 1
    #         tmp_involved_indices = []
    #         ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
    #         parent_node = individual[ix_tmp_node + 2] # + 2 for the parent node of the tmp node
    #         total_slots = individual[ix_tmp_node + 1]  # + 1 to get the slot count
    #         if individual[ix_tmp_node + 1] > 0: # only append nodes with more than 0 slots
    #             tmp_involved_indices.append(ix_tmp_node + 1)
    #         if tmp_node in dict_children: # only if the node has children
    #             for c in dict_children[tmp_node]:
    #                 ix_c = (c - 1) * self.settings["ga"]["genes_per_node_with_topology"]
    #                 total_slots += individual[ix_c + 1] # + 1 to get the slot count
    #                 if individual[ix_c + 1] > 0: # only append nodes with more than 0 slots
    #                     tmp_involved_indices.append(ix_c + 1)
    #         if parent_node in self.d_interferers: # only if the node has interferers
    #             for ifer in self.d_interferers[parent_node]:
    #                 if ifer != self.settings["topology"]["root"]: # otherwise you will be adding a negative ix_ifer
    #                     ix_ifer = (ifer - 1) * self.settings["ga"]["genes_per_node_with_topology"]
    #                     total_slots += individual[ix_ifer + 1] # + 1 to get the slot count
    #                     if individual[ix_ifer + 1] > 0: # only append nodes with more than 0 slots
    #                         tmp_involved_indices.append(ix_ifer + 1)
    #         if total_slots > self.nr_slots: # if you exceed the slot frame length, all the nodes (with more than 0 slots) are considered involved and should be returned
    #             involved_indices += tmp_involved_indices
    #         tmp_node += 1
    #     return list(set(involved_indices)) # filter all duplicates

    def make_ind_feasible(self, individual):

        # recalculate the children dictionary, after the possible parent change
        dict_children = {}
        tmp_node = 1
        while tmp_node <= len(self.nodes_0):  # calculate all the children
            ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
            ix_tmp_node_parent = ix_tmp_node + 2
            if individual[ix_tmp_node_parent] not in dict_children:  # if the parent is not in the children
                dict_children[individual[ix_tmp_node_parent]] = []  # add the parent
            dict_children[individual[ix_tmp_node_parent]].append(tmp_node)  # add current node as a child
            tmp_node += 1

        # keep adjusting one node its slot count until you have a valid one
        ixs_slot_count = self.ixs_fast_feasibility_check(individual=individual, dict_children=dict_children)
        # valid_nr_slots function returns all the slot count indices in the individual that contribute for a node n (and its children and interferers)
        # to having too many slots (i.e., more slots > slotframe length)
        while len(ixs_slot_count) > 0:
            choice = random.choice(ixs_slot_count)
            while individual[choice] < 1: # do not let it go negative, so 0 is the minimum
                choice = random.choice(ixs_slot_count)
            individual[choice] -= 1
            ixs_slot_count = self.ixs_fast_feasibility_check(individual=individual, dict_children=dict_children)

        return individual

    #### GA related methods ####

    def selElitistAndRestOffspring(self, original_pop, k_elitist, offspring, k_offspring):
        return tools.selBest(original_pop, k_elitist) + tools.selBest(offspring, k_offspring)

    def mutate_with_topology_new(self, individual):
        """Mutate an individual by replacing attributes, with probability *indpb*,
        by a integer uniformly drawn between *low* and *up* inclusively.

        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        # :param low: The lower bound or a :term:`python:sequence` of
        #             of lower bounds of the range from wich to draw the new
        #             integer.
        # :param up: The upper bound or a :term:`python:sequence` of
        #            of upper bounds of the range from wich to draw the new
        #            integer.
        # :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        """
        start_time = time.time()

        # PHASE 1, mutate a branch of the topology
        altered_nodes = []

        tmp_nodes_0 = self.nodes_0.copy()
        # print(self.nodes_0)
        # print(tmp_nodes_0)
        random.shuffle(tmp_nodes_0)
        # print(tmp_nodes_0)

        tmp_n = random.choice(tmp_nodes_0)
        tmp_nodes_0.remove(tmp_n)
        while len(tmp_nodes_0) > 0:
            if random.random() < self.settings["ga"]["mutation_idp_prob"] and len(self.d_node_to_allowed_parents[tmp_n]) > 1:
                # position of the node in the chromosome
                ix_n = (tmp_n - 1) * self.settings["ga"]["genes_per_node_with_topology"]

                # 2) get list for node n of nodes it can connect to
                current_parent = individual[ix_n + 2]
                assert current_parent in self.d_node_to_allowed_parents[tmp_n]

                children = {}
                parents = {}
                descendants = {}
                tmp_node = 1
                while tmp_node <= len(self.nodes_0):  # calculate all the children
                    ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
                    ix_tmp_node_parent = ix_tmp_node + 2
                    if individual[ix_tmp_node_parent] not in children:  # if the parent is not in the children
                        children[individual[ix_tmp_node_parent]] = []  # add the parent
                    children[individual[ix_tmp_node_parent]].append(tmp_node)  # add current node as a child
                    parents[tmp_node] = individual[ix_tmp_node_parent]
                    tmp_node += 1

                # calculate the descendants of node n
                self.calculate_descendants(node=tmp_n, children=children, l_descendants=descendants)

                allowed_parent_list = self.d_node_to_allowed_parents[tmp_n].copy()
                allowed_parent_list.remove(current_parent)
                allowed_parent_list = [p for p in allowed_parent_list if p not in descendants[tmp_n]]  # remove all the descendants because this would create loops
                new_parent = current_parent
                if len(allowed_parent_list) > 0:  # only when there are parents left, pick a new one
                    new_parent = random.choice(allowed_parent_list)  # set the new parent

                if new_parent != current_parent: # now we know for sure this node its parent is altered
                    altered_nodes.append(tmp_n)

                individual[ix_n + 2] = new_parent
                parents[tmp_n] = new_parent
                if not self.reach_root(tmp_n, parents, []):
                    raise BaseException('After mutation, I can not reach the root anymore from this node {0}'.format(tmp_n))

            # continue with the next n
            tmp_n = random.choice(tmp_nodes_0)
            tmp_nodes_0.remove(tmp_n)

        # PHASE 2, mutate the MCSs and the slots
        # also for sure mutate the MCS and the slots of the node n that were adjusted because maybe they are wrong now

        tuple_per_node = list(zip(individual, individual[1:], individual[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            mutated_MCS = False
            if len(self.d_MCS_to_index) > 1 and (ix + 1 in altered_nodes or random.random() < self.settings["ga"]["mutation_idp_prob"]): # ix + 1 is the actual node, if it equals n, you should also mutate its MCS
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0] = random.choice(self.d_node_to_allowed_MCSs_per_parent[ix + 1][parent]) # mutate MCS
                mutated_MCS = True
                # print('Mutate MCS')
            if mutated_MCS or random.random() < self.settings["ga"]["mutation_idp_prob"]:
                tmp_ix_mcs = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0]
                # prev_nr_slots = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1]
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]]) # mutate slots
                # individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]] # mutate slots
                # print('Went from {0} slots to {1} slots for MCS {2}'.format(prev_nr_slots, individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1], self.d_index_to_MCS[tmp_ix_mcs]))
                # exit()
        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
            individual = self.make_ind_feasible(individual)

        self.mutation_time += time.time() - start_time
        self.mutation_exec += 1
        return individual

    def mutate_with_topology_new_variable_slots(self, individual):
        """Mutate an individual by replacing attributes, with probability *indpb*,
        by a integer uniformly drawn between *low* and *up* inclusively.

        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        # :param low: The lower bound or a :term:`python:sequence` of
        #             of lower bounds of the range from wich to draw the new
        #             integer.
        # :param up: The upper bound or a :term:`python:sequence` of
        #            of upper bounds of the range from wich to draw the new
        #            integer.
        # :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        """
        start_time = time.time()

        # PHASE 1, mutate a branch of the topology
        altered_nodes = []

        # tmp_nodes_0 = self.nodes_0.copy()
        # # print(self.nodes_0)
        # # print(tmp_nodes_0)
        # random.shuffle(tmp_nodes_0)
        # # print(tmp_nodes_0)

        # tmp_n = random.choice(tmp_nodes_0)
        # tmp_nodes_0.remove(tmp_n)
        # while len(tmp_nodes_0) > 0:
        #     if random.random() < self.settings["ga"]["mutation_idp_prob"] and len(self.d_node_to_allowed_parents[tmp_n]) > 1:
        #         # position of the node in the chromosome
        #         ix_n = (tmp_n - 1) * self.settings["ga"]["genes_per_node_with_topology"]
        #
        #         # 2) get list for node n of nodes it can connect to
        #         current_parent = individual[ix_n + 2]
        #         assert current_parent in self.d_node_to_allowed_parents[tmp_n]
        #
        #         children = {}
        #         parents = {}
        #         descendants = {}
        #         tmp_node = 1
        #         while tmp_node <= len(self.nodes_0):  # calculate all the children
        #             ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
        #             ix_tmp_node_parent = ix_tmp_node + 2
        #             if individual[ix_tmp_node_parent] not in children:  # if the parent is not in the children
        #                 children[individual[ix_tmp_node_parent]] = []  # add the parent
        #             children[individual[ix_tmp_node_parent]].append(tmp_node)  # add current node as a child
        #             parents[tmp_node] = individual[ix_tmp_node_parent]
        #             tmp_node += 1
        #
        #         # calculate the descendants of node n
        #         self.calculate_descendants(node=tmp_n, children=children, l_descendants=descendants)
        #
        #         allowed_parent_list = self.d_node_to_allowed_parents[tmp_n].copy()
        #         allowed_parent_list.remove(current_parent)
        #         allowed_parent_list = [p for p in allowed_parent_list if p not in descendants[tmp_n]]  # remove all the descendants because this would create loops
        #         new_parent = current_parent
        #         if len(allowed_parent_list) > 0:  # only when there are parents left, pick a new one
        #             new_parent = random.choice(allowed_parent_list)  # set the new parent
        #
        #         if new_parent != current_parent: # now we know for sure this node its parent is altered
        #             altered_nodes.append(tmp_n)
        #
        #         individual[ix_n + 2] = new_parent
        #         parents[tmp_n] = new_parent
        #         if not self.reach_root(tmp_n, parents, []):
        #             raise BaseException('After mutation, I can not reach the root anymore from this node {0}'.format(tmp_n))
        #
        #     # continue with the next n
        #     tmp_n = random.choice(tmp_nodes_0)
        #     tmp_nodes_0.remove(tmp_n)

        # PHASE 2, mutate the MCSs and the slots
        # also for sure mutate the MCS and the slots of the node n that were adjusted because maybe they are wrong now

        tuple_per_node = list(zip(individual, individual[1:], individual[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            # mutated_MCS = False
            # if len(self.d_MCS_to_index) > 1 and (ix + 1 in altered_nodes or random.random() < self.settings["ga"]["mutation_idp_prob"]): # ix + 1 is the actual node, if it equals n, you should also mutate its MCS
            #     individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0] = random.choice(self.d_node_to_allowed_MCSs_per_parent[ix + 1][parent]) # mutate MCS
            #     mutated_MCS = True
            #     # print('Mutate MCS')
            # if mutated_MCS or random.random() < self.settings["ga"]["mutation_idp_prob"]:
            if random.random() < self.settings["ga"]["mutation_idp_prob"]:
                tmp_ix_mcs = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0]
                # prev_nr_slots = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1]
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]]) # mutate slots
                # individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]] # mutate slots
                # print('Went from {0} slots to {1} slots for MCS {2}'.format(prev_nr_slots, individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1], self.d_index_to_MCS[tmp_ix_mcs]))
                # exit()
        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
            individual = self.make_ind_feasible(individual)

        self.mutation_time += time.time() - start_time
        self.mutation_exec += 1
        return individual

    def mutate_with_topology(self, individual):
        """Mutate an individual by replacing attributes, with probability *indpb*,
        by a integer uniformly drawn between *low* and *up* inclusively.

        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        # :param low: The lower bound or a :term:`python:sequence` of
        #             of lower bounds of the range from wich to draw the new
        #             integer.
        # :param up: The upper bound or a :term:`python:sequence` of
        #            of upper bounds of the range from wich to draw the new
        #            integer.
        # :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        """
        start_time = time.time()

        # PHASE 1, mutate a branch of the topology

        # 1) pick a random node in the topology
        n = random.randint(1, len(self.nodes_0))
        # position of the node in the chromosome
        ix_n = (n - 1) * self.settings["ga"]["genes_per_node_with_topology"]

        # 2) get list for node n of nodes it can connect to
        current_parent = individual[ix_n + 2]
        assert current_parent in self.d_node_to_allowed_parents[n]

        # 3) if there are more possible parents than just the current parent, choose another one
        if len(self.d_node_to_allowed_parents[n]) > 1: # it should contain more possible parents than just the current parent to continue
            children = {}
            parents = {}
            descendants = {}
            tmp_node = 1
            while tmp_node <= len(self.nodes_0): # calculate all the children
                ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
                ix_tmp_node_parent = ix_tmp_node + 2
                if individual[ix_tmp_node_parent] not in children: # if the parent is not in the children
                    children[individual[ix_tmp_node_parent]] = [] # add the parent
                children[individual[ix_tmp_node_parent]].append(tmp_node) # add current node as a child
                parents[tmp_node] = individual[ix_tmp_node_parent]
                tmp_node += 1

            # calculate the descendants of node n
            self.calculate_descendants(node=n, children=children, l_descendants=descendants)

            allowed_parent_list = self.d_node_to_allowed_parents[n].copy()
            allowed_parent_list.remove(current_parent)
            allowed_parent_list = [p for p in allowed_parent_list if p not in descendants[n]] # remove all the descendants because this would create loops
            new_parent = current_parent
            if len(allowed_parent_list) > 0: # only when there are parents left, pick a new one
                new_parent = random.choice(allowed_parent_list) # set the new parent
            individual[ix_n + 2] = new_parent
            parents[n] = new_parent
            if not self.reach_root(n, parents, []):
                raise BaseException('After mutation, I can not reach the root anymore from this node {0}'.format(n))
        else:
            pass # leave the individual untouched

        # print("After possibly altering the parent: {0}".format(individual))

        # PHASE 2, mutate the MCSs and the slots
        # also for sure mutate the MCS and the slots of the node n that were adjusted because maybe they are wrong now

        tuple_per_node = list(zip(individual, individual[1:], individual[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            mutated_MCS = False
            if ix + 1 == n or random.random() < self.settings["ga"]["mutation_idp_prob"]: # ix + 1 is the actual node, if it equals n, you should also mutate its MCS
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0] = random.choice(self.d_node_to_allowed_MCSs_per_parent[ix + 1][parent]) # mutate MCS
                mutated_MCS = True
            if mutated_MCS or random.random() < self.settings["ga"]["mutation_idp_prob"]:
                tmp_ix_mcs = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0]
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]]) # mutate slots
                # individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]] # mutate slots

        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
            individual = self.make_ind_feasible(individual)

        self.mutation_time += time.time() - start_time
        self.mutation_exec += 1
        return individual

    def mutate_without_topology(self, individual):
        """Mutate an individual by replacing attributes, with probability *indpb*,
        by a integer uniformly drawn between *low* and *up* inclusively.

        This mutation only mutates MCSs and slots numbers. It does not alter topology.

        """
        start_time = time.time()

        # PHASE 1, only mutate the MCSs and the slots
        # also for sure mutate the MCS and the slots of the node n that were adjusted because maybe they are wrong now

        tuple_per_node = list(zip(individual, individual[1:], individual[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            mutated_MCS = False
            if random.random() < self.settings["ga"]["mutation_idp_prob"]: # ix + 1 is the actual node, if it equals n, you should also mutate its MCS
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0] = \
                    random.choice(self.d_node_to_allowed_MCSs_per_parent[ix + 1][parent]) # mutate MCS
                mutated_MCS = True
            if mutated_MCS or random.random() < self.settings["ga"]["mutation_idp_prob"]:
                tmp_ix_mcs = individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 0]
                individual[(ix * self.settings["ga"]["genes_per_node_with_topology"]) + 1] = \
                    random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_ix_mcs]]) # mutate slots

        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
            individual = self.make_ind_feasible(individual)

        self.mutation_time += time.time() - start_time
        self.mutation_exec += 1
        return individual

    def crossover_twopoint_with_topology(self, ind1, ind2):
        start_time = time.time()
        factor = self.settings["ga"]["genes_per_node_with_topology"]
        cxpoint1 = random.randint(0, len(self.nodes_0))
        cxpoint2 = random.randint(0, len(self.nodes_0) - 1)
        # if cxpoint2 >= cxpoint1:
        if cxpoint2 == cxpoint1:
            cxpoint2 += 1
        # else:  # Swap the two cx points
        elif cxpoint2 < cxpoint1:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        cxpoint1 *= factor
        cxpoint2 *= factor

        ind1_final = None
        ind1_found = False
        ind2_final = None
        ind2_found = False

        cxpoint1_ix = cxpoint1
        while cxpoint1_ix < cxpoint2:
            ind1_tmp = ind1.copy()  # copy for temporary testing
            ind2_tmp = ind2.copy()  # copy for temporary testing

            # do the cross-over
            ind1_tmp[cxpoint1_ix:cxpoint2] = ind2_tmp[cxpoint1_ix:cxpoint2]

            if self.valid_individual_topology(ind1_tmp):
                # save the new ind1
                ind1_final = ind1_tmp.copy()
                ind1_found = True
                break
            else:
                cxpoint1_ix += self.settings["ga"]["genes_per_node_with_topology"]

        cxpoint1_ix = cxpoint1
        while cxpoint1_ix < cxpoint2:
            ind1_tmp = ind1.copy()  # copy for temporary testing
            ind2_tmp = ind2.copy()  # copy for temporary testing

            # do the cross-over
            ind2_tmp[cxpoint1_ix:cxpoint2] = ind1_tmp[cxpoint1_ix:cxpoint2]

            if self.valid_individual_topology(ind2_tmp):
                # save the new ind2
                ind2_final = ind2_tmp.copy()
                ind2_found = True
                break
            else:
                cxpoint1_ix += self.settings["ga"]["genes_per_node_with_topology"]

        if ind1_found:
            ind1[:] = ind1_final # replace inplace (in the actual reference to ind1)
        if ind2_found:
            ind2[:] = ind2_final # replace inplace (in the actual reference to ind1)

        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            for ind in [ind1, ind2]:
                # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
                ind = self.make_ind_feasible(ind)

        self.crossover_time += time.time() - start_time
        self.crossover_exec += 1
        return ind1, ind2

    def crossover_twopoint_with_topology_variable_slots(self, ind1, ind2):
        start_time = time.time()
        factor = self.settings["ga"]["genes_per_node_with_topology"]
        cxpoint1 = random.randint(0, len(self.nodes_0))
        cxpoint2 = random.randint(0, len(self.nodes_0) - 1)
        # if cxpoint2 >= cxpoint1:
        if cxpoint2 == cxpoint1:
            cxpoint2 += 1
        # else:  # Swap the two cx points
        elif cxpoint2 < cxpoint1:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        cxpoint1 *= factor
        cxpoint2 *= factor

        ind1_final = None
        ind1_found = False
        ind2_final = None
        ind2_found = False

        cxpoint1_ix = cxpoint1
        while cxpoint1_ix < cxpoint2:
            ind1_tmp = ind1.copy()  # copy for temporary testing
            ind2_tmp = ind2.copy()  # copy for temporary testing

            # do the cross-over
            ind1_tmp[cxpoint1_ix:cxpoint2] = ind2_tmp[cxpoint1_ix:cxpoint2]

            if self.valid_individual_topology(ind1_tmp):
                # save the new ind1
                ind1_final = ind1_tmp.copy()
                ind1_found = True
                break
            else:
                cxpoint1_ix += self.settings["ga"]["genes_per_node_with_topology"]

        cxpoint1_ix = cxpoint1
        while cxpoint1_ix < cxpoint2:
            ind1_tmp = ind1.copy()  # copy for temporary testing
            ind2_tmp = ind2.copy()  # copy for temporary testing

            # do the cross-over
            ind2_tmp[cxpoint1_ix:cxpoint2] = ind1_tmp[cxpoint1_ix:cxpoint2]

            if self.valid_individual_topology(ind2_tmp):
                # save the new ind2
                ind2_final = ind2_tmp.copy()
                ind2_found = True
                break
            else:
                cxpoint1_ix += self.settings["ga"]["genes_per_node_with_topology"]

        # we only want to change the slots, so iterate over the two individuals and change there slots
        tuple_per_node = list(zip(ind1_final, ind1_final[1:], ind1_final[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            n_id = (ix * self.settings["ga"]["genes_per_node_with_topology"])
            ind1[n_id + 1] = slots
        tuple_per_node = list(zip(ind2_final, ind2_final[1:], ind2_final[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            n_id = (ix * self.settings["ga"]["genes_per_node_with_topology"])
            ind2[n_id + 1] = slots

        # if ind1_found:
        #     ind1[:] = ind1_final # replace inplace (in the actual reference to ind1)
        # if ind2_found:
        #     ind2[:] = ind2_final # replace inplace (in the actual reference to ind1)

        if self.settings["ga"]["type"] == "make-feasible" or \
                self.settings["ga"]["type"] == "make-feasible-new" or \
                self.settings["ga"]["type"] == "emperical-strategy" or \
                self.settings["ga"]["type"] == "es-closertoroot" or \
                self.settings["ga"]["type"] == "es-highreliability":
            for ind in [ind1, ind2]:
                # make the individual feasible by checking if none of the nodes exceeds the slotframe length (with children and interferers)
                ind = self.make_ind_feasible(ind)

        self.crossover_time += time.time() - start_time
        self.crossover_exec += 1
        return ind1, ind2

    def crossover_twopoint_without_topology(self, ind1, ind2):
        '''
        This crossover does not crossover the topology, only the MCS and slots.
        The ASSUMPTION here is that the allocated MCS and slots are always correct, also after crossover, because the
        the topology is not changed.
        All of this results in this just being a regular two-point crossover.
        '''
        start_time = time.time()
        factor = self.settings["ga"]["genes_per_node_with_topology"]
        cxpoint1 = random.randint(0, len(self.nodes_0))
        cxpoint2 = random.randint(0, len(self.nodes_0) - 1)
        # if cxpoint2 >= cxpoint1:
        if cxpoint2 == cxpoint1:
            cxpoint2 += 1
        # else:  # Swap the two cx points
        elif cxpoint2 < cxpoint1:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        cxpoint1 *= factor
        cxpoint2 *= factor

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        for ind in [ind1, ind2]:
            dict_children = {}
            tmp_node = 1
            while tmp_node <= len(self.nodes_0):  # calculate all the children
                ix_tmp_node = (tmp_node - 1) * self.settings["ga"]["genes_per_node_with_topology"]
                ix_tmp_node_parent = ix_tmp_node + 2
                if ind[ix_tmp_node_parent] not in dict_children:  # if the parent is not in the children
                    dict_children[ind[ix_tmp_node_parent]] = []  # add the parent
                dict_children[ind[ix_tmp_node_parent]].append(tmp_node)  # add current node as a child
                tmp_node += 1

            # keep adjusting one node its slot count until you have a valid one
            ixs_slot_count = self.check_valid_nr_slots(individual=ind, dict_children=dict_children)
            # valid_nr_slots function returns all the slot count indices in the individual that contribute for a node n (and its children and interferers)
            # to having too many slots (i.e., more slots > slotframe length)
            while len(ixs_slot_count) > 0:
                ind[random.choice(ixs_slot_count)] -= 1
                ixs_slot_count = self.check_valid_nr_slots(individual=ind, dict_children=dict_children)

        self.crossover_time += time.time() - start_time
        self.crossover_exec += 1
        return ind1, ind2

    def initialize_default(self):
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = self.d_parent[i + 1] # parent that was in the topology file
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(100):
            ind = self.mutate_with_topology(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Default initialization returned an invalid topology...")

        # for i in range(len(self.nodes_0)):
        #     num_slots_ix = i * self.settings["ga"]["genes_per_node_with_topology"] + 1
        #     ind[num_slots_ix] = 1

        return ind

    def find_modulation_per_parent(self, n, delta):
        m_n = dict()
        for p in self.d_node_to_allowed_parents[n]:
            most_reliable = None
            for m in self.d_node_to_allowed_MCSs_per_parent[n][p]: # find the most reliable modulation for this parent
                m = self.d_index_to_MCS[m]
                if most_reliable is None or self.d_pdr[n][m][p] > self.d_pdr[n][most_reliable][p]:
                    most_reliable = m
            m_n[p] = most_reliable
            for m in self.d_node_to_allowed_MCSs_per_parent[n][p]: # find the fastest modulation within a valid interval of the most reliable one
                m = self.d_index_to_MCS[m]
                # if abs(self.d_pdr[n][m][p] - self.d_pdr[n][most_reliable][p]) <= delta and self.d_MCS_to_rate[m] > self.d_MCS_to_rate[m_n[p]]:
                if self.d_pdr[n][most_reliable][p] - self.d_pdr[n][m][p] <= delta and self.d_MCS_to_rate[m] > self.d_MCS_to_rate[m_n[p]]:
                    # print('{0} - {1}: {2}'.format(m, most_reliable, abs(self.d_pdr[n][m][p] - self.d_pdr[n][most_reliable][p])))
                    m_n[p] = m
        return m_n

    def find_fast_modulation_per_parent(self, n):
        # logging.debug('Come in find_fast_modulation_per_parent')
        m_n = dict()
        for p in self.d_node_to_allowed_parents[n]:
            for m in self.d_node_to_allowed_MCSs_per_parent[n][p]: # find the fastest modulation within a valid interval of the most reliable one
                m = self.d_index_to_MCS[m]
                if p not in m_n or self.d_MCS_to_rate[m] > self.d_MCS_to_rate[m_n[p]]:
                    # if p in m_n:
                    #     logging.debug('For node {2} and parent {3}, modulation {0} is faster than previous modulation {1}.'.format(m, m_n[p], n, p))
                    # else:
                    #     logging.debug('For node {2} and parent {3}, modulation {0} is faster than previous modulation {1}.'.format(m, None, n, p))
                    m_n[p] = m
        return m_n

    def assign_parent_and_modulation(self, delta):
        m_n = dict()
        p_n = dict()
        for n in self.nodes:
            if n != 0:
                if self.settings['sf-heuristic']['heuristic'] == 'fast':
                    # print('Come in FAST heuristic.')
                    m_n[n] = self.find_fast_modulation_per_parent(n)
                elif self.settings['sf-heuristic']['heuristic'] == 'heuristic':
                    # print('Come in NORMAL heuristic.')
                    m_n[n] = self.find_modulation_per_parent(n, delta)
                else:
                    raise Exception('No correct heuristic type.')
        x_n = dict({0: 0}) # dictionary with scores per node, score of root is 0
        loops = 0
        changes = 1
        while changes > 0:
            loops += 1
            changes = 0
            for n in self.nodes:
                if n != 0:
                    for p in self.d_node_to_allowed_parents[n]:
                        if p in x_n: # only if possible parent has score already
                            ETX = 1.0 / float(self.d_pdr[n][m_n[n][p]][p])
                            tmp_x_n = x_n[p] + ETX * self.d_MCS_to_slots[m_n[n][p]]
                            if n not in x_n or tmp_x_n < x_n[n]:
                                x_n[n] = tmp_x_n
                                p_n[n] = p
                                changes += 1
            # logging.debug('{0} changes in this iteration.'.format(changes))
        logging.debug('{0} iterations to converge.'.format(loops))

        # check if every node reaches the root
        for n in self.nodes:
            if n != 0 and not self.reach_root(n, p_n, []):
                raise Exception('Could not reach root for node {0}'.format(n))

        return p_n, m_n

    def initialize_default_new_variable_slots(self):

        # assign the modulation and parent
        tmp_p_n, tmp_m_n = self.assign_parent_and_modulation(delta=self.settings['sf-heuristic']['delta'])
        # exit()
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = tmp_p_n[node]
            tmp_mcs = self.d_MCS_to_index[tmp_m_n[node][parent]]
            ind.append(tmp_mcs) # append a MCS
            # ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(0) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(1):
            ind = self.mutate_with_topology_new_variable_slots(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Default initialization returned an invalid topology...")

        # print(ind)
        # exit()

        # for i in range(len(self.nodes_0)):
        #     num_slots_ix = i * self.settings["ga"]["genes_per_node_with_topology"] + 1
        #     ind[num_slots_ix] = 1

        return ind

    def initialize_default_real_testbed_ga(self):
        # start from the topology that was created by the heuristic
        # this topology is mutated so it should not be a big deal

        # assign the modulation and parent
        tmp_p_n, tmp_m_n = self.assign_parent_and_modulation(delta=self.settings['sf-heuristic']['delta'])
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = tmp_p_n[node]
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(100):
            ind = self.mutate_with_topology_new(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Default initialization returned an invalid topology...")

        # for i in range(len(self.nodes_0)):
        #     num_slots_ix = i * self.settings["ga"]["genes_per_node_with_topology"] + 1
        #     ind[num_slots_ix] = 1

        return ind

    def initialize_default_new(self):
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = self.d_parent[i + 1] # parent that was in the topology file
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(100):
            ind = self.mutate_with_topology_new(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Default initialization returned an invalid topology...")

        # for i in range(len(self.nodes_0)):
        #     num_slots_ix = i * self.settings["ga"]["genes_per_node_with_topology"] + 1
        #     ind[num_slots_ix] = 1

        return ind

    def initialize_emperical(self):
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = self.d_parent[i + 1] # parent that was in the topology file
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(100):
            # print("Before mutation {1}: {0}".format(ind, i))
            ind = self.mutate_with_topology(ind)
            # print("after: {0}".format(ind))
            # print("After mutation {1}: {0}".format(ind, i))

        # exit()

        if not self.valid_individual_topology(ind):
            raise BaseException("Default initialization returned an invalid topology...")

        # for i in range(len(self.nodes_0)):
        #     num_slots_ix = i * self.settings["ga"]["genes_per_node_with_topology"] + 1
        #     ind[num_slots_ix] = 1

        return ind


    def initialize_dijkstra(self):
        ind = []
        dijkstra_model = dijkstra.Dijkstra()
        dijkstra_model.calculate(settings_file=self.settings.settings_file, topology_file=self.settings["topology"]["topology_file"])
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = int(dijkstra_model.dijkstra_table[i + 1]['prev']) # parent based on minimal hopcount by Dijkstra
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(30):
            ind = self.mutate_without_topology(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Dijkstra initialization returned an invalid topology...")
        return ind

    def initialize_closest_to_root(self):
        ind = []
        for i in range(len(self.nodes_0)):
            node = i + 1
            parent = self.d_parent_closest_to_root[i + 1] # parent that was in the topology file
            tmp_mcs = random.choice(self.d_node_to_allowed_MCSs_per_parent[node][parent])
            ind.append(tmp_mcs) # append a MCS
            ind.append(random.randint(0, self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[tmp_mcs]])) # append a number of slots
            ind.append(parent) # take the parent that was in the topology file

        # to make it randomized
        for i in range(30):
            ind = self.mutate_without_topology(ind)
        if not self.valid_individual_topology(ind):
            raise BaseException("Closest to the root returned an invalid topology...")
        return ind

    def evaluate_individual(self, individual):
        # print('Evaluating an individual...')
        # packs the individual in a format readable for the feasibility and throughput model, also does a validity test
        # also returns a dict with child -> parents relationship to be to check the validity of the tree
        if str(individual) in self.unique_individuals:
            return (self.unique_individuals[str(individual)][0], self.unique_individuals[str(individual)][1])

        links, parents = self.pack_individual(individual)
        # links2, parents = self.pack_individual(individual)
        if len(links) == 0:
            raise Exception("Individual {0} was mal-formed.".format(individual))

        valid_tree = True
        start_time = time.time()
        for n in self.nodes:
            if n != 0 and not self.reach_root(n, parents, []):
                valid_tree = False
                break
        self.valid_tree_time += (time.time() - start_time)
        self.valid_tree_exec += 1
        if not valid_tree:
            if str(individual) not in self.unique_individuals:
                self.unique_individuals[str(individual)] = (self.settings["ga"]["invalid_tree_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])
            return (self.settings["ga"]["invalid_tree_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])

        dict_children = {}
        # check if the links in the individual were valid
        for child, parent, reliability, slots, mcs, interferers in links:
            if parent not in dict_children:
                dict_children[parent] = []
            dict_children[parent].append(child)
            if child != self.settings["topology"]["root"]:
                if slots > self.d_MCS_to_max_bonded_slots[mcs]:
                    if str(individual) not in self.unique_individuals:
                        self.unique_individuals[str(individual)] = (self.settings["ga"]["invalid_slots_throughput_val"],
                                                                    self.settings["ga"]["infeasible_airtime_val"])
                    return (self.settings["ga"]["invalid_slots_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])

        feasible_heuristic = False
        start_time = time.time()

        self.heuristic_model.set(links)

        if self.settings['heuristic']['manner'] == 'combo':
            all_manners = ['breadth-top',
                           'largest-bonded-length',
                           'most-contention',
                           'dfs',
                           'smallest-bonded-length',
                           'breadth-bottom',
                           'least-contention',
                           'breadth-bottom-mix'
                           ]
            for manner in all_manners:
                # reset it
                self.heuristic_model.set(links)
                # set to other manner
                self.heuristic_model.sort_N_0(manner)
                feasible_heuristic = self.heuristic_model.check()
                if feasible_heuristic:
                    break
        else:
            # keep the sorting based on length
            self.heuristic_model.sort_N_0(self.settings['heuristic']['manner'])
            feasible_heuristic = self.heuristic_model.check()

        self.heuristic_time += (time.time() - start_time)
        self.heuristic_exec += 1

        if feasible_heuristic:
            self.heuristic_feasible += 1

        feasible_ilp = False
        start_time = time.time()

        # enter the individual into the feasibility model and check for feasibility
        # self.feasibility_model.set(links2)
        # feasible_ilp = self.feasibility_model.check()

        self.feasibility_time += (time.time() - start_time)
        self.feasibility_exec += 1
        if feasible_ilp:
            self.feasibility_feasible += 1

        # if feasible_heuristic and not feasible_ilp:
        #     self.heuristic_false_positives += 1
        #     # print(self.d_index_to_MCS)
        #     # print(self.d_MCS_to_slots)
        #     # print(self.d_MCS_to_max_bonded_slots)
        #     # print(self.d_interferers)
        #     # print("Links: {0}".format(links))
        #     # print('Heuristic ({0}) and ILP ({1}) say different feasibilities for individual = {2}'.format(feasible_heuristic, feasible_ilp, individual))
        #     #
        #     # # self.feasibility_model.set(links2, visualize_solution=True)
        #     # # feasible_ilp = self.feasibility_model.check(visualize_solution=True)
        #     #
        #     # feasibility_model_tmp = feasibility.Feasibility(nr_slots=self.nr_slots, nr_frequencies=self.nr_frequencies, slots_per_MCS=self.d_MCS_to_slots, settings_file=settings_file)
        #     # feasibility_model_tmp.set(links2, visualize_solution=True)
        #     # feasible_ilp = feasibility_model_tmp.check(visualize_solution=True)
        #     #
        #     # self.heuristic_model.set(links, visualize_solution=True)
        #     # self.heuristic_model.check(visualize_solution=True)
        #     raise BaseException('Heuristic ({0}) and ILP ({1}) say different feasibilities for individual = {2}'.format(feasible_heuristic, feasible_ilp, individual))
        # elif not feasible_heuristic and feasible_ilp:
        #     self.heuristic_false_negatives += 1
        #     # if self.settings['heuristic']['manner'] == 'combo':
        #     #     print(self.d_index_to_MCS)
        #     #     print(self.d_MCS_to_slots)
        #     #     print(self.d_MCS_to_max_bonded_slots)
        #     #     print(self.d_interferers)
        #     #     print("Links: {0}".format(links))
        #     #     self.feasibility_model.set(links2, visualize_solution=True)
        #     #     feasible_ilp = self.feasibility_model.check(visualize_solution=True)
        #     #     self.heuristic_model.set(links, visualize_solution=True)
        #     #     self.heuristic_model.sort_N_0("most-contention")
        #     #     self.heuristic_model.check(visualize_solution=True)
        #     #     print(self.heuristic_model.sorted_N_0)
        #     #     raise BaseException('Heuristic ({0}) and ILP ({1}) say different feasibilities for individual = {2}'.format(
        #     #         feasible_heuristic, feasible_ilp, individual))
        # elif not feasible_heuristic and not feasible_ilp:
        #     self.heuristic_true_negatives += 1
        # elif feasible_heuristic and feasible_ilp:
        #     self.heuristic_true_positives += 1
        # else:
        #     raise BaseException('Heuristic ({0}) and ILP ({1}) say different feasibilities for individual = {2}'.format(feasible_heuristic, feasible_ilp, individual))

        if not feasible_heuristic:
            if str(individual) not in self.unique_individuals:
                self.unique_individuals[str(individual)] = (self.settings["ga"]["infeasible_ind_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])
            return (self.settings["ga"]["infeasible_ind_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])
        else:

            # calculate total expected throughput
            start_time = time.time()
            self.throughput_model.set(links)
            tput = self.throughput_model.calculate()
            total_airtime = self.throughput_model.calculate_airtime()
            # if individual == [0, 2, 0, 2, 2, 0, 2, 1, 1, 2, 1, 2]:
            #     total_airtime = self.throughput_model.calculate_airtime(verbose=True)
            #     exit()
            self.throughput_time += (time.time() - start_time)
            self.throughput_exec += 1
            # print('tput')
            # print(self.throughput_time)

            if tput > len(self.nodes_0):
                self.unique_individuals[str(individual)] = (self.settings["ga"]["infeasible_ind_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])
                return (self.settings["ga"]["infeasible_ind_throughput_val"], self.settings["ga"]["infeasible_airtime_val"])

            # calculate the airtime for all the slots
            # total_airtime = 0.0
            # total_airtime_normal = 0.0

            # if self.settings["ga"]["airtime_objective"] == "tx_and_rx":
            #     # add tx and rx airtimes
            #     for child, parent, reliability, slots, mcs, interferers in links:
            #         if child != self.settings["topology"]["root"]:
            #             total_airtime += 2 * (slots * self.settings["radio"]["airtimes"][mcs])
            # elif self.settings["ga"]["airtime_objective"] == "average_airtime":

            # print(self.throughput_model.p)
            # expected_packets_per_node = {}
            # for n in self.throughput_model.p:
            #     pkts = 0.0
            #     for packets, prob_packets in enumerate(self.throughput_model.p[n]):
            #         pkts += (packets * prob_packets)
            #     expected_packets_per_node[n] = pkts
            #     print('Expected arriving packets at parent of node {0}: {1}'.format(n, expected_packets_per_node[n]))
            # print(tput)
            # 
            # # add tx and rx airtimes
            # for child, parent, reliability, slots, mcs, interferers in links:
            #     print('Node {0}, parent {1}'.format(child, parent))
            #     if child != self.settings["topology"]["root"]:
            #         total_airtime_normal += (slots * self.settings["radio"]["airtimes"][mcs])
            #         total_airtime += (expected_packets_per_node[child] * self.settings["radio"]["airtimes"][mcs])
            # 
            # print("normal total airtime: {0} ms".format(total_airtime_normal))
            # print("new total airtime: {0} ms".format(total_airtime))
            # 
            # exit()

            # else:
            #     # only add tx airtimes
            # for child, parent, reliability, slots, mcs, interferers in links:
            #     if child != self.settings["topology"]["root"]:
            #         total_airtime += slots * self.settings["radio"]["airtimes"][mcs]

            # print(individual)
            # print(links)
            # print(tput)
            # self.heuristic_model.set(links, visualize_solution=True)
            # self.heuristic_model.check(visualize_solution=True)
            # exit()

            if str(individual) not in self.valid_individuals:
                self.valid_individuals.add(str(individual))

            if str(individual) not in self.unique_individuals:
                self.unique_individuals[str(individual)] = (tput, total_airtime)

            # return both the throughput and airtime
            return tput, total_airtime

    def pack_individual(self, individual):
        '''
        Transform the given individual to a list of (child, parent, reliability, slots, mcs) combination
        Also builds a dictionary of child to parent to check the validity of tree later.
        :param individual: The GA individual solution
        :return: a list of (child, parent, reliability, slots, mcs) combination
        '''
        links = []
        parents = {}
        tuple_per_node = list(zip(individual, individual[1:], individual[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            links.append((ix + 1,  # the first element is the child, but the root does not participate.
                          parent,
                          self.d_pdr[ix + 1][self.d_index_to_MCS[mcs]][parent],
                          slots,
                          self.d_index_to_MCS[mcs],
                          self.d_interferers[ix + 1]))
            parents[ix + 1] = parent

        # also add the root
        links.append((self.settings["topology"]["root"], None, None, None, None, self.d_interferers[self.settings["topology"]["root"]]))
        return links, parents

    def set_allowed_default(self):
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])

        print(self.nodes_d_ix_hst)
        if len(self.settings['sf-heuristic']['testbed_results']) > 0:
            for n, d_mcs in self.d_pdr.items():
                if not self.reach_root_recursive(n, []):
                    raise Exception('Node {n} can not reach root in any way.'.format(n=n))
            # d_pdr_children = {}
            # for n, d_mcs in self.d_pdr.items():
            #     for mcs, d_p in d_mcs.items():
            #         for parent, pdr in d_p.items():
            #             if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
            #                 if parent not in d_pdr_children:
            #                     d_pdr_children[parent] = []
            #                 if n not in d_pdr_children[parent]:
            #                     d_pdr_children[parent].append(n)
            # dijkstra_model = dijkstraheuristic.DijkstraHeuristic()
            # dijkstra_model.calculate(settings_file=self.settings.settings_file,topology_file=self.settings["topology"]["topology_file"],nodes=self.nodes,d_node_to_allowed_parents=d_pdr_children)
            # exit()
        # print(self.d_MCS_to_index)
        # print(self.d_node_to_allowed_parents)
        # print(self.d_node_to_allowed_MCSs_per_parent)
        # print(self.d_pdr)
        # exit()

    def reach_root_recursive(self, n, visited):
        if n == self.settings['topology']['root']:
            return True
        for p in self.d_node_to_allowed_parents[n]:
            if p == self.settings['topology']['root']:
                return True
            else:
                if p not in visited:
                    visited.append(p)
                    if self.reach_root_recursive(p, visited):
                        return True
        return False

    def set_allowed_dijkstra(self):
        dijkstra_model = dijkstra.Dijkstra()
        dijkstra_model.calculate(settings_file=self.settings.settings_file, topology_file=self.settings["topology"]["topology_file"])
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if parent == dijkstra_model.dijkstra_table[n]['prev']: # only allow the minimal hopcount parent, calculated by Dijkstra
                            if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                                logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                                if parent not in self.d_node_to_allowed_parents[n]:
                                    self.d_node_to_allowed_parents[n].append(parent)
                                if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])

    def set_allowed_closest_to_root(self):
        # first collect all possible parents
        possible_parents = {}
        for n, d_mcs in self.d_pdr.items():
            if n not in possible_parents:
                possible_parents[n] = []
            for mcs, d_p in d_mcs.items():
                for parent, pdr in d_p.items():
                    if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                        if parent not in possible_parents[n]:
                            possible_parents[n].append(parent)

        # make the overview of the parents that are closest to the root
        for node in self.nodes_0:
            min_distance = None
            min_parent = None
            for p in possible_parents[node]:
                distance = self.distance_to_root[p]
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_parent = p
            self.d_parent_closest_to_root[node] = min_parent

        # now fill in the allowed parents and MCSs and only allow the closest parent
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if parent == self.d_parent_closest_to_root[n]: # only allow the minimal distance parent, calculated above
                            if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                                logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                                if parent not in self.d_node_to_allowed_parents[n]:
                                    self.d_node_to_allowed_parents[n].append(parent)
                                if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])

    def set_allowed_highest_reliability_MCS(self):
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            if len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 0: # if there is no MCS yet, just take this one
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])
                            elif len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 1: # if there is already an MCS, compare their PDRs and keep the one with the highest MCS
                                tmp_prev_mcs_name = self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]
                                if pdr - self.d_pdr[n][tmp_prev_mcs_name][parent] > self.settings["throughput"]["epsilon"]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                                elif abs(pdr - self.d_pdr[n][tmp_prev_mcs_name][parent]) < self.settings["throughput"]["epsilon"]: # if they are equal, take the one with highest rate
                                    # if the reliabilities are the same
                                    if self.d_MCS_to_rate[mcs] > self.d_MCS_to_rate[self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]]:
                                        # if the rate of the new MCS is larger, pick that one
                                        self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                            else:
                                raise BaseException("Should not be more than two MCS in the set_allowed of highest reliability.")

        print(self.d_MCS_to_index)
        print(self.d_node_to_allowed_parents)
        print(self.d_node_to_allowed_MCSs_per_parent)

    def set_allowed_emperical_strategy(self):
        '''
        *) only allow parents that are closer to the root than you
        *) only the best reliabilities per parent
        *) if there are multiple similar reliabilities, pick the one with the best rate
        :return:
        '''

        # first collect all possible parents
        possible_parents = {}
        for n, d_mcs in self.d_pdr.items():
            if n not in possible_parents:
                possible_parents[n] = []
            for mcs, d_p in d_mcs.items():
                for parent, pdr in d_p.items():
                    if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                        if parent not in possible_parents[n]:
                            possible_parents[n].append(parent)

        d_parent_closer_to_root = {}

        # make the overview of the parents that are closer to the root as the node, if there is no such parent, allow all parents
        for node in self.nodes_0:
            # min_distance = None
            # min_parent = None
            d_parent_closer_to_root[node] = []
            for p in possible_parents[node]:
                distance_of_parent_to_root = self.distance_to_root[p]
                if distance_of_parent_to_root < self.distance_to_root[node]:
                    d_parent_closer_to_root[node].append(p)
                # if min_distance is None or distance_of_parent_to_root < min_distance:
                #     min_distance = distance_of_parent_to_root
                #     min_parent = p

            # if there was no parent closer to the root than the node itself, take all parents
            if len(d_parent_closer_to_root[node]) == 0:
                d_parent_closer_to_root[node] = copy.deepcopy(possible_parents[node])

        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        # if the parent is in the closer to root list and the pdr is valid
                        if parent in d_parent_closer_to_root[n] and self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            if len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 0: # if there is no MCS yet, just take this one
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])
                            elif len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 1: # if there is already an MCS, compare their PDRs and keep the one with the highest MCS
                                tmp_prev_mcs_name = self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]
                                if abs(pdr - self.d_pdr[n][tmp_prev_mcs_name][parent]) < self.settings["throughput"]["epsilon"]: # if they are equal, take the one with highest rate
                                    # if the reliabilities are the same
                                    if self.d_MCS_to_rate[mcs] > self.d_MCS_to_rate[self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]]:
                                        # if the rate of the new MCS is larger, pick that one
                                        self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                                elif pdr > self.d_pdr[n][tmp_prev_mcs_name][parent]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                            else:
                                raise BaseException("Should not be more than two MCS in the set_allowed of highest reliability.")
        # #
        # print(self.d_MCS_to_index)
        # print(self.d_node_to_allowed_parents)
        # print(self.d_node_to_allowed_MCSs_per_parent)
        #
        # exit()

    def set_allowed_es_closertoroot(self):
        '''
        *) only allow parents that are closer to the root than you
        *) only the best reliabilities per parent
        *) if there are multiple similar reliabilities, pick the one with the best rate
        :return:
        '''

        # first collect all possible parents
        possible_parents = {}
        for n, d_mcs in self.d_pdr.items():
            if n not in possible_parents:
                possible_parents[n] = []
            for mcs, d_p in d_mcs.items():
                for parent, pdr in d_p.items():
                    if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                        if parent not in possible_parents[n]:
                            possible_parents[n].append(parent)

        d_parent_closer_to_root = {}

        # make the overview of the parents that are closer to the root as the node, if there is no such parent, allow all parents
        for node in self.nodes_0:
            # min_distance = None
            # min_parent = None
            d_parent_closer_to_root[node] = []
            for p in possible_parents[node]:
                distance_of_parent_to_root = self.distance_to_root[p]
                if distance_of_parent_to_root < self.distance_to_root[node]:
                    d_parent_closer_to_root[node].append(p)
                # if min_distance is None or distance_of_parent_to_root < min_distance:
                #     min_distance = distance_of_parent_to_root
                #     min_parent = p

            # if there was no parent closer to the root than the node itself, take all parents
            if len(d_parent_closer_to_root[node]) == 0:
                d_parent_closer_to_root[node] = copy.deepcopy(possible_parents[node])

        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        # if the parent is in the closer to root list and the pdr is valid
                        if parent in d_parent_closer_to_root[n] and self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            # allow all MCSs
                            self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])

        # print(self.d_MCS_to_index)
        # print(self.d_node_to_allowed_parents)
        # print(self.d_node_to_allowed_MCSs_per_parent)
        # exit()

    def set_allowed_es_highreliability(self):
        '''
        *) only allow parents that are closer to the root than you
        *) only the best reliabilities per parent
        *) if there are multiple similar reliabilities, pick the one with the best rate
        :return:
        '''

        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        # if the pdr is valid
                        # allow all parents
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            if len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 0: # if there is no MCS yet, just take this one
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])
                            elif len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 1: # if there is already an MCS, compare their PDRs and keep the one with the highest MCS
                                tmp_prev_mcs_name = self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]
                                if abs(pdr - self.d_pdr[n][tmp_prev_mcs_name][parent]) < self.settings["throughput"]["epsilon"]: # if they are equal, take the one with highest rate
                                    # if the reliabilities are the same
                                    if self.d_MCS_to_rate[mcs] > self.d_MCS_to_rate[self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]]:
                                        # if the rate of the new MCS is larger, pick that one
                                        self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                                elif pdr > self.d_pdr[n][tmp_prev_mcs_name][parent]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]
                            else:
                                raise BaseException("Should not be more than two MCS in the set_allowed of highest reliability.")

        # print(self.d_MCS_to_index)
        # print(self.d_node_to_allowed_parents)
        # print(self.d_node_to_allowed_MCSs_per_parent)
        # exit()

    def set_allowed_highest_rate_MCS(self):
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            if len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 0: # if there is no MCS yet, just take this one
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])
                            elif len(self.d_node_to_allowed_MCSs_per_parent[n][parent]) == 1: # if there is already an MCS, compare their PDRs and keep the one with the highest MCS
                                tmp_mcs_name = self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[n][parent][0]]
                                if self.d_MCS_to_rate[mcs] > self.d_MCS_to_rate[tmp_mcs_name]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent][0] = self.d_MCS_to_index[mcs]

    def set_allowed_closer_to_root(self):
        # first collect all possible parents
        possible_parents = {}
        for n, d_mcs in self.d_pdr.items():
            if n not in possible_parents:
                possible_parents[n] = []
            for mcs, d_p in d_mcs.items():
                for parent, pdr in d_p.items():
                    if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                        if parent not in possible_parents[n]:
                            possible_parents[n].append(parent)


        # make the overview of the parents that are closest to the root
        for node in self.nodes_0:
            min_distance = None
            min_parent = None
            self.d_parent_closer_to_root[node] = []
            for p in possible_parents[node]:
                distance = self.distance_to_root[p]
                if self.distance_to_root[node] - distance >= self.settings["throughput"]["epsilon"]:
                    self.d_parent_closer_to_root[node].append(p)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_parent = p
            self.d_parent_closest_to_root[node] = min_parent

        # now fill in the allowed parents and MCSs and only allow the closest parent
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if ((len(self.d_parent_closer_to_root[n]) == 0 and parent == self.d_parent_closest_to_root[n]) or
                                (len(self.d_parent_closer_to_root[n]) > 0 and parent in self.d_parent_closer_to_root[n])):
                            # only allow the a parent that is closer to the root than itself, if there is no such parent, take the one with the minimal distance, calculated above
                            if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                                logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                                if parent not in self.d_node_to_allowed_parents[n]:
                                    self.d_node_to_allowed_parents[n].append(parent)
                                if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                    self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                                self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])


        print(possible_parents)
        print(self.d_MCS_to_index)
        print(self.d_node_to_allowed_parents)
        print(self.d_node_to_allowed_MCSs_per_parent)
        print(self.d_parent_closer_to_root)
        print(self.d_parent_closest_to_root)

    def run(self):
        # start a timer to keep the total time
        total_time = time.time()

        # initialize the toolbox
        toolbox = base.Toolbox()

        # define the fitness evaluation function
        toolbox.register("evaluate", self.evaluate_individual)

        # fitness funtion, max. tput, min. airtime
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti, generation_created=0)

        if self.settings["ga"]["type"] == "feasible":
            toolbox.register("mutate", self.mutate_with_topology)
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default)
            self.set_allowed_default()
        elif self.settings["ga"]["type"] == "make-feasible":
            toolbox.register("mutate", self.mutate_with_topology)
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default)
            self.set_allowed_default()
        elif self.settings["ga"]["type"] == "make-feasible-new":
            toolbox.register("mutate", self.mutate_with_topology_new)
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            # toolbox.register("attr_init_ind", self.initialize_default)
            toolbox.register("attr_init_ind", self.initialize_default_new)
            self.set_allowed_default()
        elif self.settings["ga"]["type"] == "make-not-feasible-new":
            toolbox.register("mutate", self.mutate_with_topology_new)
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            # toolbox.register("attr_init_ind", self.initialize_default)
            toolbox.register("attr_init_ind", self.initialize_default_new)
            self.set_allowed_default()
        elif self.settings["ga"]["type"] == "make-not-feasible-new-variable-slots":
            toolbox.register("mutate", self.mutate_with_topology_new_variable_slots)
            toolbox.register("mate", self.crossover_twopoint_with_topology_variable_slots)
            toolbox.register("attr_init_ind", self.initialize_default_new_variable_slots)
            self.set_allowed_default()
            # exit()
        elif self.settings["ga"]["type"] == "make-not-feasible-new-testbed-ga":
            toolbox.register("mutate", self.mutate_with_topology_new)
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default_real_testbed_ga)
            self.set_allowed_default()
            # exit()
        elif self.settings["ga"]["type"] == "minimal-hopcount":
            toolbox.register("mutate", self.mutate_without_topology)
            toolbox.register("mate", self.crossover_twopoint_without_topology)
            toolbox.register("attr_init_ind", self.initialize_dijkstra)
            self.set_allowed_dijkstra()
        elif self.settings["ga"]["type"] == "minimal-distance":
            toolbox.register("mutate", self.mutate_without_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_without_topology)
            toolbox.register("attr_init_ind", self.initialize_closest_to_root)
            self.set_allowed_closest_to_root()
        elif self.settings["ga"]["type"] == "highest-reliability":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default)
            self.set_allowed_highest_reliability_MCS()
        elif self.settings["ga"]["type"] == "highest-rate":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default)
            self.set_allowed_highest_rate_MCS()
        elif self.settings["ga"]["type"] == "closer-to-root":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_closest_to_root)
            self.set_allowed_closer_to_root()
        elif self.settings["ga"]["type"] == "emperical-strategy":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_emperical)
            self.set_allowed_emperical_strategy()
        elif self.settings["ga"]["type"] == "es-closertoroot":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_emperical)
            self.set_allowed_es_closertoroot()
        elif self.settings["ga"]["type"] == "es-highreliability":
            toolbox.register("mutate", self.mutate_with_topology) # we use the same mutate function here, b/c this does not change topology
            toolbox.register("mate", self.crossover_twopoint_with_topology)
            toolbox.register("attr_init_ind", self.initialize_default)
            self.set_allowed_es_highreliability()
        else:
            raise BaseException("No valid GA type selected!")

        # set the parent selection strategy
        if self.settings["ga"]["parent_selection"]["choice"] == "tournament":
            toolbox.register("parent_selection", tools.selTournament)
        else:
            raise BaseException("No valid parent selection strategy {0}".format(self.settings["ga"]["parent_selection"]["choice"]))

        # set the survivor selection strategy
        if self.settings["ga"]["survivor_selection"]["choice"] == "pop_best":
            toolbox.register("survivor_selection", tools.selBest)
        elif self.settings["ga"]["survivor_selection"]["choice"] == "elitism":
            toolbox.register("survivor_selection", self.selElitistAndRestOffspring)
        elif self.settings["ga"]["survivor_selection"]["choice"] == "tournament":
            toolbox.register("survivor_selection", tools.selTournament)
        elif self.settings["ga"]["survivor_selection"]["choice"] == "offspring":
            pass # here you do not need any algorithm from DEAP
        else:
            raise BaseException("No valid survivor selection strategy.")

        # register an individual
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_init_ind)
        # register a population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # generate initial population
        pop = toolbox.population(self.settings["ga"]["pop_size"])

        # stats on tput
        stats_tput = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        # stats on airtime
        stats_airtime = tools.Statistics(key=lambda ind: ind.fitness.values[1])
        # combine stats
        mstats = tools.MultiStatistics(tput=stats_tput, airtime=stats_airtime)
        # metrics
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        # Evaluate the individuals with an invalid fitness for the first time also
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        ######################## START GA ########################
        for g in range(self.settings["ga"]["generations"]):
            if g % self.settings["ga"]["output_x_generations"] == 0:
                logging.info("Generation {0}".format(g))

            if self.settings["ga"]["design"] == "stochastic":
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, pop))

                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    self.crossover_exec_total += 1
                    if random.random() < self.settings["ga"]["crossover_prob"]:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                if g % self.settings["ga"]["output_x_generations"] == 0:
                    logging.debug("Applied all cross-overs.")

                # Apply mutation on the offspring
                for mutant in offspring:
                    self.mutation_exec_total += 1
                    if random.random() < self.settings["ga"]["mutation_prob"]:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                if g % self.settings["ga"]["output_x_generations"] == 0:
                    logging.debug("Applied all mutations.")

                for mutant in offspring:
                    if not mutant.fitness.valid:
                        mutant.generation_created = g

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Do survivor selection
                if self.settings["ga"]["survivor_selection"]["choice"] == "pop_best":
                    pop = toolbox.survivor_selection(pop + offspring, len(pop))
                elif self.settings["ga"]["survivor_selection"]["choice"] == "elitism":
                    pop = toolbox.survivor_selection(pop, self.survivor_selection_elitism_pop_slice, offspring, self.survivor_selection_elitism_offspring_slice)
                elif self.settings["ga"]["survivor_selection"]["choice"] == "tournament":
                    pop = toolbox.survivor_selection(pop + offspring, len(pop), self.survivor_selection_tournament_size)
                else:
                    raise BaseException("No valid survivor selection strategy.")
            elif self.settings["ga"]["design"] == "double-selection":
                # Do parent selection first
                selected = None
                if self.settings["ga"]["parent_selection"]["choice"] == "tournament":
                    selected = toolbox.parent_selection(pop, len(pop), self.parent_selection_tournament_size)
                else:
                    raise BaseException("No valid parent selection strategy.")

                offspring = [toolbox.clone(ind) for ind in selected]

                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    self.crossover_exec_total += 1
                    if random.random() < self.settings["ga"]["crossover_prob"]:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                if g % self.settings["ga"]["output_x_generations"] == 0:
                    logging.debug("Applied all cross-overs.")

                # Apply mutation on the offspring
                for mutant in offspring:
                    self.mutation_exec_total += 1
                    if random.random() < self.settings["ga"]["mutation_prob"]:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                if g % self.settings["ga"]["output_x_generations"] == 0:
                    logging.debug("Applied all mutations.")

                for mutant in offspring:
                    if not mutant.fitness.valid:
                        mutant.generation_created = g

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Do survivor selection
                if self.settings["ga"]["survivor_selection"]["choice"] == "pop_best":
                    pop = toolbox.survivor_selection(pop + offspring, len(pop))
                elif self.settings["ga"]["survivor_selection"]["choice"] == "offspring":
                    pop[:] = offspring
                elif self.settings["ga"]["survivor_selection"]["choice"] == "elitism":
                    pop = toolbox.survivor_selection(pop, self.survivor_selection_elitism_pop_slice, offspring, self.survivor_selection_elitism_offspring_slice)
                elif self.settings["ga"]["survivor_selection"]["choice"] == "tournament":
                    pop = toolbox.survivor_selection(pop + offspring, len(pop), self.survivor_selection_tournament_size)
                else:
                    raise BaseException("No valid survivor selection strategy.")
            else:
                raise BaseException("This GA design ({0}) is not available.".format(self.settings["ga"]["design"]))

            # keep the best individuals in the hall-of-fame
            self.hof.update(pop)

            # keep track of the best individual to put it in the results
            for ind in pop:
                # if self.best_individual is not None and abs(ind.fitness.values[0] - self.best_individual["ind"].fitness.values[0]) < 0.000000001:
                #     print('Near-equal fitnesses: {0}, {1}, {2}'.format(self.best_individual["ind"].fitness.values[0], ind.fitness.values[0], ind))
                #     print('Near-equal fitnesses, with precision: {0:.20f}, {1:.20f}'.format(self.best_individual["ind"].fitness.values[0], ind.fitness.values[0]))
                #     if self.best_individual is not None and ind.fitness.values[0] == self.best_individual["ind"].fitness.values[0]:
                #         print('Also Equal fitnesses: {0}, {1}'.format(self.best_individual["ind"].fitness.values[0], ind.fitness.values[0]))
                #         print('Also Equal fitnesses, with precision: {0:.20f}, {1:.20f}'.format(self.best_individual["ind"].fitness.values[0], ind.fitness.values[0]))

                # if self.best_individual is None or ind.fitness > self.best_individual["ind"].fitness:
                if self.best_individual is None or (ind.fitness.values[0] > (self.best_individual["tput"] + 0.000000001)) or ((abs(ind.fitness.values[0] - self.best_individual["tput"]) < 0.000000001) and (ind.fitness.values[1] + 0.000000001) < self.best_individual["airtime"]):
                    if self.best_individual is None:
                        self.best_individual = {}
                    self.best_individual["ind"] = toolbox.clone(ind)
                    self.best_individual["tput"] = ind.fitness.values[0]
                    self.best_individual["airtime"] = ind.fitness.values[1]
                    self.best_individual["generation"] = g

            # keep track of the best individual its performance over generations
            if self.best_individual is not None:
                if self.best_individual["tput"] is not None and self.best_individual["airtime"] is not None:
                    self.best_individual_performance.append((self.best_individual["tput"], self.best_individual["airtime"]))
                    # print("{0:.15f}, {1:.15f}\n".format(self.best_individual["tput"], self.best_individual["airtime"]))
                    # print("{0:.20f}, {1:.20f}\n".format(self.best_individual["tput"], self.best_individual["airtime"]))
                    # print("{0:.2f}, {1:.2f}\n".format(self.best_individual["tput"], self.best_individual["airtime"]))
                    # print("{0}, {1}\n".format(self.best_individual["tput"], self.best_individual["airtime"]))

            self.unique_individuals_performance.append(len(self.unique_individuals))
            self.valid_individuals_performance.append(len(self.valid_individuals))

            infeasible_inds = 0
            for ind in offspring:
                if ind.fitness.values[0] < 0:
                    infeasible_inds += 1
            self.infeasible_inds_performance.append(infeasible_inds)

            # keep the stats of the current generation
            record = mstats.compile(pop)
            self.logbook.record(gen=g, **record)
            if g % self.settings["ga"]["output_x_generations"] == 0:
                logging.info(record)

        # end the timer for the total time
        self.total_time = time.time() - total_time

        print(self.d_index_to_MCS)

    ##### OUTPUT related methods ####

    def print_progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        logging.debug('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
        # logging.debug('\r%s %s%% %s\r' % (prefix, percent, suffix))
        # Print New Line on Complete
        if iteration == total:
            logging.debug()

    def print_hall_of_fame(self):
        logging.info("Best solution:")
        logging.info("Individual {0} with {1} throughput and {2} ms airtime".format(self.best_individual['ind'], self.best_individual['tput'], self.best_individual['airtime']))

        logging.info("Mutation time: {0} runs (total = {1}), {2} s total time, {3} s on average run".format(self.mutation_exec, self.mutation_exec_total, self.mutation_time, self.mutation_time / float(self.mutation_exec)))
        logging.info("Crossover time: {0} runs (total = {1}), {2} s total time, {3} s on average run".format(self.crossover_exec, self.crossover_exec_total, self.crossover_time, self.crossover_time / float(self.crossover_exec)))
        if self.valid_tree_exec > 0:
            logging.info("Check valid tree stats: {0} runs, {1} s total time, {2} s on average run".format(self.valid_tree_exec, self.valid_tree_time, self.valid_tree_time / float(self.valid_tree_exec)))
        else:
            logging.info("Did not run valid tree check once...")
        if self.heuristic_exec > 0:
            logging.info("Heuristic model stats: {0} runs ({3} feasible solutions), {1} s total time, {2} s on average run".format(self.heuristic_exec, self.heuristic_time, self.heuristic_time / float(self.heuristic_exec), self.heuristic_feasible))
            logging.info("- total setting time: {0}".format(self.heuristic_model.time_setting))
            # logging.info("- total building time: {0}".format(self.feasibility_model.time_building))
            logging.info("- total solving time: {0}".format(self.heuristic_model.time_solving))
            logging.info("- true positives: {0}".format(self.heuristic_true_positives))
            logging.info("- true negatives: {0}".format(self.heuristic_true_negatives))
            logging.info("- false positives: {0}".format(self.heuristic_false_positives))
            logging.info("- false negatives: {0}".format(self.heuristic_false_negatives))
        else:
            logging.info("Did not run heuristic model once...")
        if self.feasibility_exec > 0:
            logging.info("Feasibility model stats: {0} runs ({3} feasible solutions), {1} s total time, {2} s on average run".format(self.feasibility_exec, self.feasibility_time, self.feasibility_time / float(self.feasibility_exec), self.feasibility_feasible))
            logging.info("- total setting time: {0}".format(self.feasibility_model.time_setting))
            logging.info("- total building time: {0}".format(self.feasibility_model.time_building))
            logging.info("- total solving time: {0}".format(self.feasibility_model.time_solving))
        else:
            logging.info("Did not run feasibility model once...")
        if self.throughput_exec > 0:
            logging.info("Throughput calculation stats: {0} runs, {1} s total time, {2} s on average run".format(self.throughput_exec, self.throughput_time, self.throughput_time / float(self.throughput_exec)))
        else:
            logging.info("Did not run throughput calculation once...")

        logging.info("Total number of unique individuals: {0}".format(len(self.unique_individuals)))
        logging.info("Total time: {0} s".format(self.total_time))

    def draw(self):
        gen = self.logbook.select("gen")
        tput_max = self.logbook.chapters["tput"].select("max")
        airtime_min = self.logbook.chapters["airtime"].select("min")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, tput_max, "b-", label="Max. throughput")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Max. throughput", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, airtime_min, "r-", label="Min. airtime")
        ax2.set_ylabel("Min. airtime", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)

        if not os.path.exists(self.settings["ga"]["results_dir"]):
            os.mkdir(self.settings["ga"]["results_dir"])

        plt.savefig("{0}ga-evolution.eps".format(self.settings["ga"]["results_dir"]), format='eps')

    def write_file(self):
        if not os.path.exists(self.settings["ga"]["results_dir"]):
            os.mkdir(self.settings["ga"]["results_dir"])

        file_data = {}
        file_data["results"] = {}
        file_data["results"]["best_ind"] = self.best_individual

        links_tmp, parents_tmp = self.pack_individual(self.best_individual['ind'])
        valid_tree = True
        for n in self.nodes:
            if n != 0 and not self.reach_root(n, parents_tmp, []):
                valid_tree = False
                break
        # make the schedule
        if valid_tree:
            schedule_file = "{0}/ga-schedule.json".format(self.settings["ga"]["results_dir"])

            self.heuristic_model.set(links_tmp)
            feasible = False
            if self.settings['heuristic']['manner'] == 'combo':
                all_manners = ['breadth-top',
                               'largest-bonded-length',
                               'most-contention',
                               'dfs',
                               'smallest-bonded-length',
                               'breadth-bottom',
                               'least-contention',
                               'breadth-bottom-mix'
                               ]
                for manner in all_manners:
                    # reset it
                    self.heuristic_model.set(links_tmp)
                    # set to other manner
                    self.heuristic_model.sort_N_0(manner)
                    feasible = self.heuristic_model.check(solution_file=schedule_file)
                    if feasible:
                        break
            else:
                # keep the sorting based on length
                self.heuristic_model.sort_N_0(self.settings['heuristic']['manner'])
                feasible = self.heuristic_model.check(solution_file=schedule_file)

            # self.feasibility_model.set(links_tmp)
            # feasible = self.feasibility_model.check(solution_file=schedule_file)
            if feasible:
                # add parents
                data = None
                with open(schedule_file) as json_file:
                    data = json.load(json_file)
                data['parents'] = parents_tmp
                data['hostnames'] = self.nodes_d_ix_hst
                data['coordinator'] = self.settings["topology"]["root"] # normally always zero, but okay
                data['slotframe_size'] = self.settings["simulator"]["slotframeLength"]
                data['generated_packets'] = self.settings["tsch"]["generated_packets"]
                data['r_max'] = self.settings["tsch"]["r_max"]
                data['queue_size'] = self.settings["tsch"]["queue_size"]
                with open(schedule_file, 'w') as json_file:
                    json.dump(data, json_file)
            else:
                raise BaseException("The best individual {0} is not deemed feasible by the ILP.".format(self.best_individual['ind']))

        file_data["results"]["best_ind_evolution"] = self.best_individual_performance
        file_data["results"]["unique_individuals_evolution"] = self.unique_individuals_performance
        file_data["results"]["valid_individuals_evolution"] = self.valid_individuals_performance

        file_data["results"]["individuals"] = {}
        file_data["results"]["individuals"]["total"] = 0
        file_data["results"]["individuals"]["filtered"] = 0
        file_data["results"]["individuals"]["unique"] = len(self.unique_individuals)
        file_data["results"]["individuals"]["valid"] = len(self.valid_individuals)
        file_data["results"]["time"] = {}
        file_data["results"]["time"]["total_time"] = self.total_time
        file_data["results"]["time"]["mutation_time"] = self.mutation_time
        file_data["results"]["time"]["crossover_time"] = self.crossover_time
        file_data["results"]["time"]["valid_tree"] = {"total": self.valid_tree_time, "nr": self.valid_tree_exec}
        file_data["results"]["time"]["heuristic"] = {"total": self.heuristic_time, "nr": self.heuristic_exec, "setting": self.heuristic_model.time_setting, "solving": self.heuristic_model.time_solving, "feasible": self.heuristic_feasible, "true_positives": self.heuristic_true_positives, "true_negatives": self.heuristic_true_negatives, "false_positives": self.heuristic_false_positives, "false_negatives": self.heuristic_false_negatives}
        file_data["results"]["time"]["feasibility"] = {"total": self.feasibility_time, "nr": self.feasibility_exec, "setting": self.feasibility_model.time_setting, "building": self.feasibility_model.time_building, "solving": self.feasibility_model.time_solving, "feasible": self.feasibility_feasible}
        file_data["results"]["time"]["throughput"] = {"total": self.throughput_time, "nr": self.throughput_exec}
        file_data["results"]["infeasible_inds"] = self.infeasible_inds_performance
        file_name = self.settings["ga"]["results_dir"] + self.settings["ga"]["results_file_prefix"] + ".json"
        with open(file_name, 'w') as json_file:
            json.dump(file_data, json_file)

def run_ga(input=None, loglevel=None):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=getattr(logging, loglevel.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

    ga = GAWrapper(input)
    ga.run()
    ga.draw()
    ga.print_hall_of_fame()
    ga.write_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--loglevel', '-l', type=str)

    args = parser.parse_args()
    settings_file = str(args.input)
    loglevel = str(args.loglevel)

    tmp_settings = Settings(settings_file)
    if tmp_settings["ga"]["type"] != "exhaustive":
        run_ga(input=settings_file, loglevel=loglevel)
    else:
        exhaustive.run_exhaustive(input=settings_file, loglevel=loglevel)
    #
    # inds = []
    # results = []
    # for i in range(1000):
    #     inds.append([3, random.randint(0, 5), 0, 3, random.randint(0, 5), 0, 4, random.randint(0, 5), 0, 4, random.randint(0, 5), 3, 4, random.randint(0, 5), 2, 3, random.randint(0, 5), 8, 3, random.randint(0, 5), 9, 0, random.randint(0, 5), 0, 4, random.randint(0, 5), 13, 3, random.randint(0, 5), 6, 4, random.randint(0, 5), 2, 3, random.randint(0, 5), 10, 4, random.randint(0, 5), 3, 3, random.randint(0, 5), 5])
    #
    # for ind in inds:
    #     links, parents = ga.pack_individual(individual=ind)
    #     ga.feasibility_model.set(links)
    #     feasible = ga.feasibility_model.check()
    #     results.append(feasible)
    #
    # print("Setting: {0}".format(ga.feasibility_model.time_setting))
    # print("Building: {0}".format(ga.feasibility_model.time_building))
    # print("Solving: {0}".format(ga.feasibility_model.time_solving))
    # # print("eq1: {0}".format(ga.feasibility_model.eq_1))
    # # print("eq2: {0}".format(ga.feasibility_model.eq_2))
    # # print("eq3: {0}".format(ga.feasibility_model.eq_3))
    # # print("eq4: {0}".format(ga.feasibility_model.eq_4))
    # # print("eq5: {0}".format(ga.feasibility_model.eq_5))
    # # print("vars: {0}".format(ga.feasibility_model.vars))
    # print("total eq: {0}".format(ga.feasibility_model.eq_1 + ga.feasibility_model.eq_2 + ga.feasibility_model.eq_3 + ga.feasibility_model.eq_4 + ga.feasibility_model.eq_5 + ga.feasibility_model.vars))
    # print("Total: {0}".format(ga.feasibility_model.time_setting + ga.feasibility_model.time_building + ga.feasibility_model.time_solving))
    # print(len(results))
    # # print(results)
    # # print(ga.evaluate_individual([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    # # print(ga.evaluate_individual([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))
    # # print(ga.MCSs[1])
    # # print(ga.evaluate_individual([0, 0, 4, 2, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0]))
    # # print(ga.MCSs[4])
    # # print(ga.MCSs[0])
    # # print(ga.MCSs[3])
    # # print(ga.MCSs[1])
    # # print(ga.evaluate_individual([0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0]))
    # # print(ga.evaluate_individual([0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0]))