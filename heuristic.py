from gurobipy import *
import settings
import time
import json
from visualize import Visualization

class Heuristic:
    def __init__(self, nr_slots, nr_frequencies, slots_per_MCS, settings_file):
        self.visualization = True
        # self.settings = {}
        # self.settings['topology'] = {'root': 0}
        # self.settings['feasibility'] = {'epsilon': 0.0000001}
        # self.settings['simulator'] = {'noNewInterference': 0}
        self.settings = settings.Settings(settings_file)
        # self.settings = settings.Settings("experiments/input/ga_seed_812_c_ILP10msFR_ss_120_exp_10ms_120ms_.json")
        self.EPSILON = self.settings["feasibility"]["epsilon"]

        # slots in a slot frame
        self.T_MAX = nr_slots - 1 # maximum slot in a slotframe
        self.T = range(0, self.T_MAX + 1)  # goes from 0 to T_MAX
        # frequencies at which a node can transmit
        self.F_MAX = nr_frequencies - 1  # maximal frequency offset
        self.F = range(0, self.F_MAX + 1)  # goes from 0 to F_MAX
        # number of slots an aggregated slot can contain
        # self.S = range(1, self.T_MAX + 2)  # goes from 1 to T_MAX + 1

        self.N = [0, 1, 2]
        self.N_0 = [1, 2]
        self.sorted_N_0 = []

        # length of the bonded slots for that node
        self.Y = dict({
            0: 0,
            1: 3,
            2: 2
        })

        # number of allocated bonded slots for that node
        self.R = dict({
            0: 0,
            1: 3,
            2: 1
        })

        # dictionary map from node to parent of the node
        self.P = dict({
            1: 0,
            2: 1,
            # 3: 2,
            # 4: 3
            # 4: 2
        })

        # dictionary map from node to its children
        self.C = dict({
            0: [1],
            1: [2],
            # 2: [3],
            # 3: [4],
        })

        # maps node to its interferers (so to the nodes of which it is in the interference range)
        self.my_interferers = dict({
            0: [1],
            1: [2],
            # 2: [3],
            # 3: [4],
        })

        # maps node to all the nodes it interferers with (so those nodes are in the interference range of the node)
        self.in_interference_range = dict({})

        self.B = slots_per_MCS

        self.schedule_tx = {}
        self.write_schedule_tx = {}
        self.schedule_rx = {}

        self.time_setting = 0.0
        self.time_solving = 0.0

    def set(self, links, visualize_solution=False):
        start_setting = time.time()
        self.N = []
        self.N_0 = []
        self.sorted_N_0 = []
        self.R = {}
        self.C = {}
        self.P = {}
        self.Y = {}
        # maps node to its interferers (so to the nodes of which it is in the interference range)
        self.my_interferers = {}
        # maps node to all the nodes it interferers with (so those nodes are in the interference range of the node)
        self.in_interference_range = {}
        self.MCS = {}
        self.schedule = []
        self.N_depth = {}
        self.N_depth_length = {}

        for child, parent, reliability, slots, mcs, interferers in links:
            child = int(child)
            if child == self.settings["topology"]["root"]:
                if len(self.settings["sf-heuristic"]["testbed_results"]) == 0:
                    if not self.settings["simulator"]["noNewInterference"]:
                        self.my_interferers[child] = interferers
            else:
                parent = int(parent)
                slots = int(slots)
                reliability = float(reliability)
                # add children
                if child not in self.N:
                    self.N.append(child)
                    self.N_0.append(child)
                if parent not in self.N:
                    self.N.append(parent)
                    if parent != self.settings["topology"]["root"]: # root should not be added to
                        self.N_0.append(parent)
                # add slots
                self.R[child] = slots
                # add to children
                if parent not in self.C:
                    self.C[parent] = []
                self.C[parent].append(child)
                self.P[child] = parent
                # add reliabilities
                self.Y[child] = self.B[mcs]
                self.MCS[child] = mcs
                if not self.settings["simulator"]["noNewInterference"]:
                    if len(self.settings["sf-heuristic"]["testbed_results"]) == 0:
                        self.my_interferers[child] = interferers
                # sort the nodes_0 in desending order based on the length of their data rate slots
                # if len(self.sorted_N_0) == 0:
                #     self.sorted_N_0.append(child)
                # else:
                #     ix = len(self.sorted_N_0) - 1 # set to last index
                #     while ix >= 0: # as long as you did not insert it, keep on going
                #         if self.Y[child] < self.Y[self.sorted_N_0[ix]]:
                #             # child its slot lenth is shorter, so append after ix
                #             self.sorted_N_0.insert(ix + 1, child)
                #             # you have to + 1, otherwise you will insert AT ix instead of behind it
                #             break
                #         elif ix == 0:
                #             # insert at the first place, b/c ix was already 0
                #             self.sorted_N_0.insert(0, child)
                #             break
                #         ix -= 1

        for n in self.N:
            if n != self.settings['topology']['root']:
                self.write_schedule_tx[n] = {}
            self.schedule_tx[n] = []
            self.schedule_rx[n] = []
            for freq in range(0, self.F_MAX + 1):
                self.schedule_tx[n].append([False for ts in range(0, self.T_MAX + 1)])
                self.schedule_rx[n].append([False for ts in range(0, self.T_MAX + 1)])

        self.N_depth[0] = 0
        for n in self.N_0:
            self.N_depth[n] = self.get_depth(n)

        self.sorted_N_0 = []
        for n in self.N_0:
            nr_contenders = 2 # plus two, for the node and the parent
            if n in self.C: # add the children
                nr_contenders += len(self.C[n])
            nr_contenders += len(self.C[self.P[n]]) - 1 # add the number of siblings
            self.sorted_N_0.append((n, self.N_depth[n], self.Y[n]*self.R[n], nr_contenders, self.Y[n]))

        for n in self.N:
            if n in self.my_interferers:
                for ifer in self.my_interferers[n]:
                    if ifer not in self.in_interference_range:
                        self.in_interference_range[ifer] = []
                    self.in_interference_range[ifer].append(n)

        if visualize_solution:
            print("I_p_n: {0}".format(self.my_interferers))
            print("I_of_c: {0}".format(self.in_interference_range))
            print("Y[n]:", self.Y)
            print("R[n]:", self.R)
            print("P[n]:", self.P)
            print("F_MAX:", self.F_MAX)
            print("T_MAX:", self.T_MAX)

        self.time_setting += time.time() - start_setting

    def get_depth(self, n):
        if n in self.N_depth:
            return self.N_depth[n]
        else:
            return 1 + self.get_depth(self.P[n])

    def dfs(self, visited, node):
        if node not in visited:
            visited.append(node)
            if node in self.C:
                for child in self.C[node]:
                    self.dfs(visited, child)

    def valid_tx_rx_n(self, node, ts):
        for ch in range(0, self.F_MAX + 1):
            # if the node is transmitting or receiving at that place
            if self.schedule_tx[node][ch][ts] or self.schedule_rx[node][ch][ts]:
                # the node already sends within the time of this bonded slot
                return False
        return True

    def valid_interferers(self, node, ts, ch, debug=False):
        # if an interfering node of my parent is transmitting, return false
        for interferer in self.my_interferers[self.P[node]]:
            if self.schedule_tx[interferer][ch][ts]:
                return False
        # if a node in my interfering range is receiving (so node is the interferer) and the other node_in_range is receiving, return false
        for node_in_range in self.in_interference_range[node]:
            if self.schedule_rx[node_in_range][ch][ts]:
                return False
        return True

    def valid_tx_rx_parent(self, node, ts):
        for ch in range(0, self.F_MAX + 1):
            # the parent is already listening to the node's siblings or transmitting to its parents, return false
            if self.schedule_tx[self.P[node]][ch][ts] or self.schedule_rx[self.P[node]][ch][ts]:
                return False
        return True

    def allocate_bonded_slot(self, node, ts, ch, length_bonded_slot, solution_file=False):
        ts_min = ts
        ts_max = ts + length_bonded_slot - 1
        while ts_min <= ts_max:
            self.schedule_tx[node][ch][ts_min] = True
            self.schedule_rx[self.P[node]][ch][ts_min] = True
            ts_min += 1
        if solution_file:
            # write to a dictionary, this will be used for the final result
            if ts not in self.write_schedule_tx[node]:
                self.write_schedule_tx[node][ts] = dict()
            if ch not in self.write_schedule_tx[node][ts]:
                self.write_schedule_tx[node][ts][ch] = dict()
                self.write_schedule_tx[node][ts][ch]['mcs'] = self.MCS[node]
                self.write_schedule_tx[node][ts][ch]['slots'] = self.Y[node]
                self.write_schedule_tx[node][ts][ch]['parent'] = self.P[node]

    def sort_N_0(self, manner):
        if manner == 'breadth-top':
            # breadth first from top, additionally sorted on length bonded slot
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (tup[1], -tup[2], -tup[4]))
        elif manner == 'breadth-bottom':
            # breadth first from leaves, additionally sorted on length bonded slot
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (-tup[1], -tup[2], -tup[4]))
        elif manner == 'breadth-bottom-mix':
            # breadth first from leaves, additionally sorted on length bonded slot
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (-tup[1], tup[2], tup[4]))
        elif manner == 'largest-bonded-length':
            # keep the sorting based on length
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (-tup[2], -tup[4]))
        elif manner == 'smallest-bonded-length':
            # keep the sorting based on length
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (tup[2], tup[4]))
        elif manner == 'dfs':
            visited = []
            # sort in DFS fashion
            self.dfs(visited, self.settings['topology']['root'])
            self.sorted_N_0 = []
            for n_tmp in visited:
                if n_tmp != self.settings['topology']['root']:
                    self.sorted_N_0.append((n_tmp, self.N_depth[n_tmp], self.Y[n_tmp], None, None))
        elif manner == 'most-contention':
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (-tup[3], -tup[2], -tup[4]))
        elif manner == 'least-contention':
            self.sorted_N_0 = sorted(self.sorted_N_0, key=lambda tup: (tup[3], -tup[2], -tup[4]))
        else:
            raise BaseException('Wrong heuristic manner!')

    def check(self, visualize_solution=False, solution_file=False):
        feasible = True
        start_solving = time.time()

        # start the greedy bin packing
        for n, depth_tuple, nr_slots, nr_contenders, length_tuple in self.sorted_N_0:
            if visualize_solution:
                print('Fitting node {0}...'.format(n))
                print('Interferers of the parent of the node: {0}'.format(self.my_interferers[self.P[n]]))
            length_bonded_slot = self.Y[n]
            bonded_slots_to_allocate = self.R[n]
            if bonded_slots_to_allocate > 0:
                # go over all frequencies
                for ch in range(0, self.F_MAX + 1):
                    # make sure you do not exceed the slot frame length
                    for ts in range(0, (self.T_MAX - length_bonded_slot + 1) + 1):
                        # go over all the regular slots in the bonded slot from ts_min to ts_max
                        ts_min = ts
                        ts_max = ts + length_bonded_slot - 1
                        valid_ts = True
                        while ts_min <= ts_max:
                            if len(self.settings["sf-heuristic"]["testbed_results"]) == 0:
                                if not self.valid_tx_rx_n(n, ts_min) \
                                        or not self.valid_tx_rx_parent(n, ts_min) \
                                        or not self.valid_interferers(n, ts_min, ch, debug=visualize_solution):
                                    valid_ts = False
                                    break
                            elif len(self.settings["sf-heuristic"]["testbed_results"]) > 0:
                                if not self.valid_tx_rx_n(n, ts_min) \
                                        or not self.valid_tx_rx_parent(n, ts_min):
                                    valid_ts = False
                                break
                            else:
                                raise Exception("Something wrong with the testbed_results of sf-heuristic.")
                            ts_min += 1
                        if valid_ts: # allocate the bonded slot in the schedule
                            self.allocate_bonded_slot(n, ts, ch, length_bonded_slot, solution_file=solution_file)
                            bonded_slots_to_allocate -= 1
                            if bonded_slots_to_allocate == 0:
                                break # stop because you have allocated all bonded slots
                            else:
                                continue # continue in the next ts with the next bonded slot
                        else: # try the next timeslot ts
                            continue
                    if bonded_slots_to_allocate == 0: # if all bonded slots were allocated, stop
                        break
                if bonded_slots_to_allocate > 0:
                    # if visualize_solution:
                    #     print('Breaking here for node {0} at because there are still slots to allocate'.format(n))
                    #     for n in self.N_0:
                    #         print("node {0} (reg. slots in bonded slot = {2}): {1}".format(n, self.schedule_tx[n], length_bonded_slot))
                    #     viz = Visualization(len(self.T), len(self.F), self.N, self.P, interferers=self.my_interferers)
                    #     for n in self.N_0:
                    #         for t in self.T:
                    #             for f in self.F:
                    #                 if self.schedule_tx[n][f][t]:
                    #                     viz.add_node(t, f, n, self.Y[n])
                    #     viz.visualize(suffix='heuristic', output_dir="./")
                    feasible = False
                    break

        self.time_solving += time.time() - start_solving
        # print(self.time_solving)
        if feasible:
            if visualize_solution or solution_file:
                tmp_schedule = dict()
                viz = Visualization(len(self.T), len(self.F), self.N, self.P, interferers=self.my_interferers)
                for n in self.N_0:
                    for t in self.T:
                        for f in self.F:
                            if self.schedule_tx[n][f][t]:
                                viz.add_node(t, f, n, self.Y[n])
                viz.visualize(suffix='ga', output_dir=self.settings["ga"]["results_dir"])

                ga_schedule = {}
                ga_schedule['schedule'] = self.write_schedule_tx
                with open(solution_file, 'w') as outfile:
                    json.dump(ga_schedule, outfile)

                # for n in self.N_0:
                #     print("node {0}: {1}".format(n, self.schedule_tx[n]))

            return True
        else:
            return False

# SLOTS_PER_MCS = dict({ # number of slots per bonded slot per MCS
#     "QAM_16_FEC_3_4": 2,
#     "QAM_16_FEC_1_2": 3,
#     "QPSK_FEC_3_4": 3,
#     "QPSK_FEC_1_2_FR_2": 6,
#     "QPSK_FEC_1_2": 4
# })
# settings_file_par = {}
# settings_file_par["topology"] = {}
# settings_file_par["topology"]["root"] = 0
# settings_file_par["feasibility"] = {}
# settings_file_par["feasibility"]["epsilon"] = 0.000001
# model = Feasibility(visualization=True, nr_slots=24, nr_frequencies=2, slots_per_MCS=SLOTS_PER_MCS, settings_file="settings/settings.json")
# links = [(0, None, None, None, None, []), (1, 0, '1.0', 1, 'QAM_16_FEC_3_4', []), (2, 1, '1.0', 3, 'QPSK_FEC_1_2_FR_2', []), (3, 1, '1.0', 1, 'QPSK_FEC_3_4', []), (4, 2, '1.0', 1, 'QPSK_FEC_1_2', []), (5, 2, '1.0', 1, 'QAM_16_FEC_3_4', [])]
# model.set(links=links)
# start_two = time.time()
# print(model.check())
# print("Everything takes {0} s".format(time.time() - start_two))

# model = Feasibility(visualization=True, nr_slots=7, nr_frequencies=2, slots_per_MCS=SLOTS_PER_MCS, settings_file="settings/settings.json")
# # links = [(0, None, None, None, None, []), (1, 0, '1.0', 2, 'QAM_16_FEC_3_4', []), (2, 1, '1.0', 2, 'QAM_16_FEC_3_4', []), (3, 0, '1.0', 1, 'QAM_16_FEC_3_4', [])]
# links = [(0, None, None, None, None, [2]), (1, 0, '1.0', 2, 'QAM_16_FEC_3_4', [0, 4]), (2, 1, '1.0', 1, 'QPSK_FEC_3_4', [1]), (3, 2, '1.0', 1, 'QAM_16_FEC_3_4', [2]), (4, 0, '1.0', 1, 'QAM_16_FEC_3_4', [0])]
# model.set(links=links)
# start_two = time.time()
# print(model.check())
# print("Everything takes {0} s".format(time.time() - start_two))
#





# SLOTS_PER_MCS = dict({ # number of slots per bonded slot per MCS
#     "QAM_16_FEC_3_4": 2,
#     "QAM_16_FEC_1_2": 3,
#     "QPSK_FEC_3_4": 3,
#     "QPSK_FEC_1_2_FR_2": 6,
#     "QPSK_FEC_1_2": 4
# })
# heuristic = Heuristic(11, 3, SLOTS_PER_MCS, None)
# # links = [(0, None, None, None, None, [2]), (1, 0, '1.0', 1, 'QPSK_FEC_1_2_FR_2', [0, 4, 3]), (2, 1, '1.0', 1, 'QPSK_FEC_3_4', [1]), (3, 0, '1.0', 1, 'QPSK_FEC_1_2_FR_2', [2]), (4, 3, '1.0', 1, 'QAM_16_FEC_3_4', [0])]
# links = [(0, None, None, None, None, [5]), (1, 0, '1.0', 1, 'QPSK_FEC_3_4', [3,4]), (2, 1, '1.0', 1, 'QAM_16_FEC_3_4', [4]), (3, 0, '1.0', 1, 'QPSK_FEC_3_4', []),(4, 0, '1.0', 3, 'QPSK_FEC_3_4', []),(5, 2, '1.0', 1, 'QPSK_FEC_3_4', [])]
# heuristic.set(links=links)
# print(heuristic.check())
# print(heuristic.schedule)


# SLOTS_PER_MCS = dict({ # number of slots per bonded slot per MCS
#     "QPSK_FEC_3_4": 2,
#     "QPSK_FEC_1_2_FR_2": 4,
#     "QPSK_FEC_1_2": 3
# })
# heuristic = Heuristic(4, 1, SLOTS_PER_MCS, None)
# # links = [(0, None, None, None, None, []),
# #          (1, 0, '1.0', 3, 'QPSK_FEC_1_2_FR_2', []),
# #          (2, 1, '1.0', 0, 'QPSK_FEC_1_2', []),
# #          (3, 0, '1.0', 0, 'QPSK_FEC_1_2_FR_2', [])]
# links = [(0, None, None, None, None, []), (1, 0, '1.0', 1, 'QPSK_FEC_3_4', [3]), (2, 0, '1.0', 1, 'QPSK_FEC_3_4', []), (3, 1, '1.0', 1, 'QPSK_FEC_3_4', [1])]
# heuristic.set(links=links)
# print(heuristic.check(visualize_solution=True))
# # print(heuristic.schedule)