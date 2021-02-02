import random

import math
import json
from settings import Settings
import argparse
import logging
import sys
import copy

from visualize import Visualization
class TopologyHeuristic:
    def __init__(self, settings_file=None):
        self.settings = Settings(settings_file)
        logging.debug("Setting seed to {0}".format(self.settings["seed"]))
        random.seed(self.settings["seed"])

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

        self.node_neighbors = {}

        self.schedule = dict() # initialize empty schedule

        self.write_schedule_tx = {}
        self.schedule_tx = {}
        self.schedule_rx = {}
        self.P = {}
        self.C = {}
        self.mcs_per_node = {} # number of slots per bonded slot
        # maps node to all the nodes it interferers with (so those nodes are in the interference range of the node)
        self.in_interference_range = {}

        if self.settings["modulations"]["modulations_file"]:
            self.__init_modulations(self.settings["modulations"]["modulations_file"])
        else:
            raise Exception('No modulations file given.')

        if self.settings["topology"]["topology_file"]:
            self.__init_topology(self.settings["topology"]["topology_file"])
        else:
            raise Exception('No topology file given.')

        self.set_allowed_default()

    def __init_topology(self, json_topology_file):
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

            logging.info("Topology initialized.")

    def __init_modulations(self, json_modulations_file):
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

            logging.info("Modulations initialized.")

        # print(self.d_MCS_to_index)
        # print(self.d_index_to_MCS)
        # print(self.d_MCS_to_slots)
        # print(self.d_MCS_to_max_bonded_slots)
        # exit()

    def set_allowed_default(self):
        for n, d_mcs in self.d_pdr.items():
            if n != self.settings["topology"]["root"]:
                if n not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[n] = []
                    self.d_node_to_allowed_MCSs_per_parent[n] = {}
                for mcs, d_p in d_mcs.items():
                    for parent, pdr in d_p.items():
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            if n not in self.node_neighbors:
                                self.node_neighbors[n] = []
                            self.node_neighbors[n].append(parent)
                            logging.debug("For node {0} and MCS {1} and parent {2}, PDR = {3}".format(n, mcs, parent, pdr))
                            if parent not in self.d_node_to_allowed_parents[n]:
                                self.d_node_to_allowed_parents[n].append(parent)
                            if parent not in self.d_node_to_allowed_MCSs_per_parent[n]:
                                self.d_node_to_allowed_MCSs_per_parent[n][parent] = []
                            self.d_node_to_allowed_MCSs_per_parent[n][parent].append(self.d_MCS_to_index[mcs])

    # def find_children(self, nodes, possible_parents, mcs, reliability_threshold):
    #     '''
    #     :param nodes:
    #     :param possible_parents:
    #     :param mcs:
    #     :param reliability_threshold:
    #     :return: returns a dictionary mapping the children to their respective parents
    #     '''
    #     dict_children = dict()
    #     slots_to_parent = 0 # placeholder
    #     for n in nodes:
    #         for p in possible_parents:
    #             if n in self.d_pdr and mcs in self.d_pdr[n] and p in self.d_pdr[n][mcs] and self.d_pdr[n][mcs][p] > reliability_threshold:
    #                 dict_children[n] = (p, mcs, slots_to_parent)
    #     return dict_children
    #
    # def make_topology_fast_first(self, nodes_per_level, sorted_nodes_0):
    #     '''
    #     Assigns node a parent by first trying the fastest MCS, then a slower MCS and so on.
    #     '''
    #     MCSs = sorted(self.d_MCS_to_rate.keys(), key=lambda x: self.d_MCS_to_rate[x], reverse=True)  # sort on fastest modulation first
    #     mcs_index = 0
    #     min_reliability_threshold = 0.7
    #     reliability_threshold = 0.9
    #     cpy_nodes = copy.copy(sorted_nodes_0)
    #     level = 0
    #     nodes_per_level.insert(level, {0: (None, None)})  # add the root node at level 0 (index 0)
    #     while len(cpy_nodes) != 0 and mcs_index <= (len(MCSs) - 1):  # keep going as long as there are nodes to distribute or MCSs to try
    #         level = 0
    #         logging.info('Running for MCS {0}'.format(MCSs[mcs_index]))
    #         while len(cpy_nodes) != 0:
    #             # random.shuffle(cpy_nodes)  # randomly shuffle the nodes
    #             dict_children = None
    #             if mcs_index < len(MCSs) - 1:
    #                 dict_children = self.find_children(cpy_nodes, nodes_per_level[level].keys(), MCSs[mcs_index], reliability_threshold)
    #             else:
    #                 dict_children = self.find_children(cpy_nodes, nodes_per_level[level].keys(), MCSs[mcs_index], min_reliability_threshold)
    #             children = dict_children.keys()  # get all the children that are assigned a parent
    #             next_level = level + 1
    #             if len(children) == 0 and next_level > len(nodes_per_level) - 1:
    #                 # if there are no children found for this MCS, and there is no deeper level to try, quit.
    #                 break
    #             if len(children) > 0:
    #                 if next_level < len(nodes_per_level):  # if there is already an entry at this level
    #                     for n, tup in dict_children.items():  # copy all the child to (parent, mcs) at the correct level
    #                         nodes_per_level[next_level][n] = tup  # add the dictionary to the nodes_per_level list, remember the assigned parent and mcs
    #                 else:
    #                     nodes_per_level.insert(next_level, copy.deepcopy(dict_children))
    #                     assert len(nodes_per_level) >= level + 1  # now this should be true
    #                 for n, tup in dict_children.items():
    #                     self.P[n] = tup[0] # store the parent also in a seperate dict
    #                     if tup[0] not in self.C:
    #                         self.C[tup[0]] = []
    #                     self.C[tup[0]].append(n)
    #                     self.mcs_per_node[n] = tup[1]
    #                 cpy_nodes = [n for n in cpy_nodes if not n in children]  # remove the chilren from the list of nodes to iterate
    #             level += 1
    #         mcs_index += 1
    #
    #     logging.info('Nodes left: {0}'.format(cpy_nodes))
    #     logging.info('Nodes per level: {0}'.format(nodes_per_level))
    #
    # def find_children_of_this_level(self, nodes, possible_parents, minimum_threshold):
    #     dict_children = dict()
    #     slots_to_parent = 0 # placeholder
    #     for n in nodes:
    #         for mcs in self.d_MCS_to_rate.keys():
    #             for p in possible_parents:
    #                 if n in self.d_pdr and mcs in self.d_pdr[n] and p in self.d_pdr[n][mcs] and self.d_pdr[n][mcs][p] > minimum_threshold:
    #                     if n not in dict_children: # not yet added, so add it anyway
    #                         dict_children[n] = (p, mcs, slots_to_parent)
    #                     else: # we already had a link, can we improve it?
    #                         if self.d_pdr[n][dict_children[n][1]][dict_children[n][0]] < self.d_pdr[n][mcs][p]: # if the reliability of this new parent and mcs is better, pick this one
    #                             dict_children[n] = (p, mcs, slots_to_parent)
    #                         elif abs(self.d_pdr[n][dict_children[n][1]][dict_children[n][0]] - self.d_pdr[n][mcs][p]) < 0.00000001: # if the reliability is the same
    #                             if self.d_MCS_to_rate[dict_children[n][1]] < self.d_MCS_to_rate[mcs]: # if the new modulation is faster, take this one
    #                                 dict_children[n] = (p, mcs, slots_to_parent)
    #
    #     return dict_children
    #
    # def make_topology_take_most_reliable_mcs(self, nodes_per_level, sorted_nodes_0):
    #     cpy_nodes = copy.copy(sorted_nodes_0)
    #     min_reliability_threshold = 0.7
    #     level = 0
    #     nodes_per_level.insert(level, {0: (None, None)})  # add the root node at level 0 (index 0)
    #     while len(cpy_nodes) != 0:
    #         # random.shuffle(cpy_nodes)  # randomly shuffle the nodes
    #         dict_children = self.find_children_of_this_level(cpy_nodes, nodes_per_level[level].keys(), min_reliability_threshold)
    #         children = dict_children.keys()  # get all the children that are assigned a parent
    #         next_level = level + 1
    #         if len(children) == 0:
    #             # if there are no children found for this MCS, and there is no deeper level to try, quit.
    #             break
    #         elif len(children) > 0:
    #             nodes_per_level.insert(next_level, copy.deepcopy(dict_children))
    #             assert len(nodes_per_level) >= level + 1  # now this should be true
    #             for n, tup in dict_children.items():
    #                 self.P[n] = tup[0] # store the parent also in a seperate dict
    #                 if tup[0] not in self.C:
    #                     self.C[tup[0]] = []
    #                 self.C[tup[0]].append(n)
    #                 self.mcs_per_node[n] = tup[1]
    #             cpy_nodes = [n for n in cpy_nodes if not n in children]  # remove the chilren from the list of nodes to iterate
    #         level += 1
    #
    #     logging.info('Nodes left: {0}'.format(cpy_nodes))
    #     logging.info('Nodes per level: {0}'.format(nodes_per_level))
    #
    # def valid_tx_rx_n(self, node, ts):
    #     for ch in range(0, self.nr_frequencies):
    #         # if the node is transmitting or receiving at that place
    #         if self.schedule_tx[node][ch][ts] or self.schedule_rx[node][ch][ts]:
    #             # the node already sends within the time of this bonded slot
    #             return False
    #     return True
    #
    # def valid_interferers(self, node, ts, ch):
    #     # if an interfering node of my parent is transmitting, return false
    #     for interferer in self.d_interferers[self.P[node]]:
    #         if self.schedule_tx[interferer][ch][ts]:
    #             return False
    #     # if a node in my interfering range is receiving (so node is the interferer) and the other node_in_range is receiving, return false
    #     for node_in_range in self.in_interference_range[node]:
    #         if self.schedule_rx[node_in_range][ch][ts]:
    #             return False
    #     return True
    #
    # def valid_tx_rx_parent(self, node, ts):
    #     for ch in range(0, self.nr_frequencies):
    #         # the parent is already listening to the node's siblings or transmitting to its parents, return false
    #         if self.schedule_tx[self.P[node]][ch][ts] or self.schedule_rx[self.P[node]][ch][ts]:
    #             return False
    #     return True
    #
    # def allocate_bonded_slot(self, node, ts, ch, length_bonded_slot, parent, mcs):
    #     ts_min = ts
    #     ts_max = ts + length_bonded_slot - 1
    #     while ts_min <= ts_max:
    #         self.schedule_tx[node][ch][ts_min] = True
    #         self.schedule_rx[parent][ch][ts_min] = True
    #         ts_min += 1
    #     # write to a dictionary, this will be used for the final result
    #     if ts not in self.write_schedule_tx[node]:
    #         self.write_schedule_tx[node][ts] = dict()
    #     if ch not in self.write_schedule_tx[node][ts]:
    #         self.write_schedule_tx[node][ts][ch] = dict()
    #         self.write_schedule_tx[node][ts][ch]['mcs'] = mcs
    #         self.write_schedule_tx[node][ts][ch]['slots'] = length_bonded_slot
    #         self.write_schedule_tx[node][ts][ch]['parent'] = parent
    #
    # def allocate(self, node, mcs, bonded_slots_to_allocate):
    #     length_bonded_slot = self.d_MCS_to_slots[mcs] # length of the bonded slot for the employed MCS
    #     for ch in range(0, self.nr_frequencies): # make sure you do not exceed the slot frame length
    #         for ts in range(0, ((self.nr_slots - 1) - length_bonded_slot + 1) + 1): # go over all the regular slots in the bonded slot from ts_min to ts_max
    #             ts_min = ts
    #             ts_max = ts + length_bonded_slot - 1
    #             valid_ts = True
    #             while ts_min <= ts_max:
    #                 if not self.valid_tx_rx_n(node, ts_min) \
    #                         or not self.valid_tx_rx_parent(node, ts_min) \
    #                         or not self.valid_interferers(node, ts_min, ch):
    #                     valid_ts = False
    #                     break
    #                 ts_min += 1
    #             if valid_ts:  # allocate the bonded slot in the schedule
    #                 self.allocate_bonded_slot(node, ts, ch, length_bonded_slot, self.P[node], mcs)
    #                 bonded_slots_to_allocate -= 1
    #                 if bonded_slots_to_allocate == 0:
    #                     break  # stop because you have allocated all bonded slots
    #                 else:
    #                     continue  # continue in the next ts with the next bonded slot
    #             else:  # try the next timeslot ts
    #                 continue
    #         if bonded_slots_to_allocate == 0:  # if all bonded slots were allocated, stop
    #             break
    #     if bonded_slots_to_allocate > 0:
    #         return False
    #     return True # if we allocated all bonded slots, return True
    #
    # def assign_slots(self, nodes_per_level):
    #     MCSs = sorted(self.d_MCS_to_rate.keys(), key=lambda x: self.d_MCS_to_rate[x], reverse=True)  # sort on fastest modulation first
    #     levels = len(nodes_per_level)
    #     bonded_slots_to_allocate = 1
    #     for mcs in MCSs:
    #         for level in range(levels):
    #             if level == 0:
    #                 continue # don't do anything for the root
    #             for n, tup in nodes_per_level[level].items():
    #                 if tup[1] != mcs: # not the correct MCS, don't do anything now
    #                     continue # not the correct MCS
    #                 allocation_success = self.allocate(node=n, mcs=mcs, bonded_slots_to_allocate=bonded_slots_to_allocate)
    #                 # if allocation_success:
    #                 #     print('Allocating {0} bonded slots for node {1} succeeded.'.format(bonded_slots_to_allocate, n))
    #                 # else:
    #                 #     print('Allocating {0} bonded slots for node {1} did NOT succeed.'.format(bonded_slots_to_allocate, n))
    #
    # def breadth_first_assign_slots(self, nodes_per_level, bonded_slots_to_allocate = 1):
    #     ''''
    #     Breadth-first assignment of slots, independent of the MCS you are assigning to.
    #     '''
    #     levels = len(nodes_per_level)
    #     allocation_succes = True
    #     while allocation_succes: # keep going as long as there was one allocation success
    #         allocation_succes = False
    #         for level in range(levels):
    #             if level == 0:
    #                 continue # don't do anything for the root
    #             for n, tup in nodes_per_level[level].items():
    #                 if self.allocate(node=n, mcs=tup[1], bonded_slots_to_allocate=bonded_slots_to_allocate):
    #                     allocation_succes = True
    #
    # def depth_first(self, node, bonded_slots_to_allocate=1):
    #     success = False
    #     if node != 0: # root should not allocate
    #         success = self.allocate(node=node, mcs=self.mcs_per_node[node], bonded_slots_to_allocate=bonded_slots_to_allocate)
    #     if node in self.C:
    #         for c in self.C[node]:
    #             if (self.depth_first(c, bonded_slots_to_allocate)):
    #                 success = True
    #     return success
    #
    # def depth_first_assign_slots(self, start_node=0, bonded_slots_to_allocate=1):
    #     ''''
    #     Depth-first assignment of slots, independent of the MCS you are assigning to.
    #     '''
    #     allocation_succes = True
    #     while allocation_succes: # keep going as long as there was one allocation success
    #         allocation_succes = self.depth_first(node=start_node, bonded_slots_to_allocate=bonded_slots_to_allocate)

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
                if abs(self.d_pdr[n][m][p] - self.d_pdr[n][most_reliable][p]) <= delta and self.d_MCS_to_rate[m] > self.d_MCS_to_rate[most_reliable]:
                    m_n[p] = m
        return m_n

    def reach_root(self, n, p_n, parents):
        if n not in p_n:
            return False
        elif p_n[n] in parents:
            return False
        elif p_n[n] == 0:
            return True
        parents.append(p_n[n])
        return self.reach_root(p_n[n], p_n, parents)

    def assign_parent_and_modulation(self, delta):
        m_n = dict()
        p_n = dict()
        for n in self.nodes:
            if n != 0:
                m_n[n] = self.find_modulation_per_parent(n, delta)
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
            logging.debug('{0} changes in this iteration.'.format(changes))
        logging.debug('{0} iterations to converge.'.format(loops))

        # check if every node reaches the root
        for n in self.nodes:
            if n != 0 and not self.reach_root(n, p_n, []):
                raise Exception('Could not reach root for node {0}'.format(n))

        return p_n, m_n

    def run(self):
        logging.info("Heuristic built new topology.")

        # # initialize
        # for n in self.nodes:
        #     if n != self.settings['topology']['root']:
        #         self.write_schedule_tx[n] = {}
        #     self.schedule_tx[n] = []
        #     self.schedule_rx[n] = []
        #     for freq in range(0, self.nr_frequencies):
        #         self.schedule_tx[n].append([False for ts in range(0, self.nr_slots)])
        #         self.schedule_rx[n].append([False for ts in range(0, self.nr_slots)])
        # for n in self.nodes:
        #     if n in self.d_interferers:
        #         for ifer in self.d_interferers[n]:
        #             if ifer not in self.in_interference_range:
        #                 self.in_interference_range[ifer] = []
        #             self.in_interference_range[ifer].append(n)
        #
        # # make the topology
        # nodes_per_level = list()
        # sorted_nodes_0 = copy.copy(self.nodes_0)
        # sorted_nodes_0 = sorted(sorted_nodes_0, key=lambda x: (len(set(self.node_neighbors[x])), x), reverse=True)
        # # print(self.node_neighbors)
        # # for x, lst in self.node_neighbors.items():
        # #     print(set(self.node_neighbors[x]))
        # #     print(len(set(self.node_neighbors[x])))
        # # print(sorted_nodes_0)
        # # sort on the number of possible links, if there all equal numbers, sort on the node's ID number (bigger comes first)
        #
        # if self.settings['heuristic-sf']['topology'] == 'fast':
        #     logging.info("Chosen for a topology with a focus on FAST modulations.")
        #     self.make_topology_fast_first(nodes_per_level, sorted_nodes_0)
        # elif self.settings['heuristic-sf']['topology'] == 'reliable':
        #     logging.info("Chosen for a topology with a focus on RELIABLE modulations.")
        #     self.make_topology_take_most_reliable_mcs(nodes_per_level, sorted_nodes_0)
        #
        # # assign the slots
        # # self.assign_slots(nodes_per_level)
        # if self.settings['heuristic-sf']['slots'] == 'bf':
        #     logging.info("Chosen for BREADTH-FIRST slot assignment.")
        #     self.breadth_first_assign_slots(nodes_per_level=nodes_per_level, bonded_slots_to_allocate=1)
        # elif self.settings['heuristic-sf']['slots'] == 'df':
        #     logging.info("Chosen for DEPTH-FIRST slot assignment.")
        #     self.depth_first_assign_slots(start_node=0, bonded_slots_to_allocate=1)
        #
        # # for node, info in self.write_schedule_tx.items():
        # #     print('Node {0}:'.format(node))
        # #     print(info)

        # for n in self.nodes:
        #     if n != 0:
        #         print('node {0}: {1}'.format(n, self.findModulationPerParent(n, 0.01)))

        print(self.assign_parent_and_modulation(delta=0.01))
        exit()

        # viz = Visualization(self.nr_slots, self.nr_frequencies, self.nodes, self.P, interferers=self.d_interferers)
        # for n in self.nodes_0:
        #     for t in range(self.nr_slots):
        #         for f in range(self.nr_frequencies):
        #             if self.schedule_tx[n][f][t]:
        #                 viz.add_node(t, f, n, self.d_MCS_to_slots[self.mcs_per_node[n]])
        # viz.visualize(suffix='ga', output_dir=self.settings["ga"]["results_dir"])

        # solution_file = "{0}/ga-schedule.json".format(self.settings["ga"]["results_dir"])
        # ga_schedule = {}
        # ga_schedule['schedule'] = self.write_schedule_tx
        # ga_schedule['parents'] = self.P
        # del ga_schedule['parents'][0] # remove the root
        # with open(solution_file, 'w') as outfile:
        #     json.dump(ga_schedule, outfile)

        # print("{0} nodes left to distribute.".format(len(cpy_nodes)))

        #
        # self.schedule = dict()
        # ts = 1  # tmp time slot
        # ch = 1  # tmp channel
        # # for node in self.nodes_0:
        # for node in self.nodes_0:
        #     self.schedule[node] = dict()
        #     if node == 3: # only allocate for node 3
        #         if ts not in self.schedule[node]:
        #             self.schedule[node][ts] = dict()
        #         if ch not in self.schedule[node][ts]:
        #             self.schedule[node][ts][ch] = dict()
        #             self.schedule[node][ts][ch]['mcs'] = self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[int(node)][self.d_parent[int(node)]][0]]
        #             self.schedule[node][ts][ch]['slots'] = self.d_MCS_to_slots[self.d_index_to_MCS[self.d_node_to_allowed_MCSs_per_parent[int(node)][self.d_parent[int(node)]][0]]]
        #             self.schedule[node][ts][ch]['parent'] = int(self.d_parent[int(node)])

    def write_file(self):
        schedule_file = "{0}/ga-schedule.json".format(self.settings["ga"]["results_dir"])
        # add parents
        data = {}
        data['schedule'] = self.write_schedule_tx
        data['parents'] = self.P
        del data['parents'][0]
        with open(schedule_file, 'w') as json_file:
            json.dump(data, json_file)

def run_heuristic(input=None, loglevel=None):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=getattr(logging, loglevel.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

    th = TopologyHeuristic(input)
    th.run()
    th.write_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--loglevel', '-l', type=str)

    args = parser.parse_args()
    settings_file = str(args.input)
    loglevel = str(args.loglevel)

    run_heuristic(input=settings_file, loglevel=loglevel)

