from gurobipy import *
import settings
import time
import json
from visualize import Visualization

class Feasibility:
    def __init__(self, nr_slots, nr_frequencies, slots_per_MCS, settings_file, visualization = False):
        self.settings = settings.Settings(settings_file)
        # self.settings = {}
        # self.settings['topology'] = {'root': 0}
        # self.settings['feasibility'] = {'epsilon': 0.0000001}
        # self.settings['simulator'] = {'noNewInterference': 0}
        # self.settings["ga"] = {"results_dir": './'}
        self.visualization = visualization
        self.EPSILON = self.settings["feasibility"]["epsilon"]
        self.gap = 0
        self.threads = 1
        self.timelimit = 0  # we interpret 0 as no time limit

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

        self.my_interferers = dict({
            0: [1],
            1: [2],
            # 2: [3],
            # 3: [4],
        })

        self.B = slots_per_MCS

        self.time_setting = 0.0
        self.time_building = 0.0
        self.time_solving = 0.0

    # def set_slots_per_MCS(self, slots_per_MCS):
    #     self.B = slots_per_MCS

    def set(self, links, visualize_solution=False):
        start_setting = time.time()
        self.N = []
        self.N_0 = []
        self.R = {}
        self.C = {}
        self.P = {}
        self.Y = {}
        self.my_interferers = {}
        self.MCS = {}
        for child, parent, reliability, slots, mcs, interferers in links:
            child = int(child)
            if child == self.settings["topology"]["root"]:
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
                    self.my_interferers[child] = interferers

        # print(self.C)
        # print(self.I_p_n)
        # # make sure you filter out the root node and the children of each node
        # for c, ifs in self.I_p_n.items():
        #     # the root can not be in the interferers list
        #     if self.settings["topology"]["root"] in ifs:
        #         self.I_p_n[c].remove(self.settings["topology"]["root"])
        #     to_keep = []
        #     if c not in self.C:
        #         continue # there are no children so you do not have to filter
        #     for i in ifs:
        #         if i not in self.C[c]:
        #             to_keep.append(i)
        #     self.I_p_n[c] = to_keep
        # print(self.I_p_n)

        if visualize_solution:
            print("feasibility: {0}".format(self.my_interferers))
            print("feasibility Y[n]:", self.Y)
            print("feasibility P[n]:", self.P)
            print("feasibility R[n]:", self.R)
            print("feasibility F_MAX:", self.F_MAX)
            print("feasibility T_MAX:", self.T_MAX)
            print("feasibility F:", self.F)
            print("feasibility T:", self.T)

        # remove the children from a node's set of interferers, otherwise you prevent them from sending to their parent
        # for n, ifs in self.I_p_n.items():
        #     if n in self.C:
        #         for c in self.C[n]:
        #             if c in self.I_p_n[n]:
        #                 self.I_p_n[n].remove(c)
        #         if n in self.P:
        #             if self.P[n] in self.I_p_n[n]:
        #                 self.I_p_n[n].remove(self.P[n])
        self.time_setting += time.time() - start_setting
        # print(self.Y)

    def check(self, solution_file=None, visualize_solution=False):
        start_building = time.time()
        # Create our 'Adaptive TSCH' optimization model
        m = Model('adaptsch')

        ### DECISION VARIABLES

        # binary var that is 1 when node n has s consecutive time slots allocated at time offsets t, t + 1, ..., t + s - 1
        # and frequency offset f in the TSCH schedule, to its parent
        sigma = m.addVars(self.T, self.F, self.N_0, vtype=GRB.BINARY, name='sigma')

        ### CONSTRAINTS

        # equation 1
        m.addConstrs((sigma.sum('*', '*', n) == self.R[n] for n in self.N_0), name='eq_1')

        # equation 2
        for n in self.N_0:
            for t in self.T:
                for f in self.F:
                    m.addConstr((sigma[t, f, n] * (t + self.Y[n] - 1) <= self.T_MAX), name='eq_2[%s,%s,%s]' % (t, f, n))

        # equation 3
        for n in self.N_0:
            for t in self.T:
                constr = LinExpr()
                for f in self.F:
                    for tt in range(max(0, t - self.Y[n] + 1), t + 1):
                        constr += sigma[tt, f, n]  # the constraint for the node itself
                    if n in self.C:  # if it has children
                        for j in self.C[n]:  # account for each child
                            for ttt in range(max(0, t - self.Y[j] + 1), t + 1):
                                constr += sigma[ttt, f, j]
                m.addConstr((constr <= 1), name='eq_3[%s,%s]' % (n, t))

        # equation 4
        for t in self.T:
            constr = LinExpr()
            for f in self.F:
                if self.settings["topology"]["root"] in self.C:  # if it has children
                    for j in self.C[self.settings["topology"]["root"]]:  # account for each child
                        for tt in range(max(0, t - self.Y[j] + 1), t + 1):
                            constr += sigma[tt, f, j]
            m.addConstr((constr <= 1), name='eq_4[%s,%s]' % (self.settings["topology"]["root"], t))

        # # equation 5
        for n in self.N_0:
            for t in self.T:
                for f in self.F:
                    constr = LinExpr()
                    for tt in range(max(0, t - self.Y[n] + 1), t + 1):
                        constr += sigma[tt, f, n]  # the constraint for the node itself
                    sumConstr = LinExpr()
                    if self.P[n] in self.my_interferers:  # if it has interferers
                        for j in self.my_interferers[self.P[n]]:  # account for each node in the interfering range
                            if j != n and j != self.settings["topology"]["root"]:
                            # *** j != n: the node is an interferer of its parent, because it can reach it of course
                            # so exclude the node from the list of interferers here because otherwise
                            # you are stating that node n can not send when (the interferer) node n is sending,
                            # thus prohibiting itself from sending
                            # *** j != self.settings["topology"]["root"]: the root is never a transmitter
                            # so it should be excluded from the list of interferers of a node self.I[n]
                            # if it is not excluded, it actually gives problem with accessing self.Y[0] b/c
                            # there is no entry for 0!
                            # of course, the root itself can still be in the interfering range of other nodes
                            # so it still has an entry in self.I
                                for ttt in range(max(0, t - self.Y[j] + 1), t + 1):
                                    sumConstr += sigma[ttt, f, j]
                    m.addConstr((constr * sumConstr == 0), name='eq_5[%s,%s,%s]' % (n, t, f))

        # if visualize_solution:
        #     m.addConstr((sigma[0, 3, 4] == 1), name='eq_6[%s,%s,%s]' % (4, 0, 3))
        #     m.addConstr((sigma[0, 3, 7] == 1), name='eq_6[%s,%s,%s]' % (7, 0, 3))

        m.setParam('OutputFlag', False)
        if visualize_solution:
            m.setParam('OutputFlag', True)

        # maximize the objective
        m.modelSense = GRB.MAXIMIZE

        # if self.timelimit > 0:
        #     m.setParam('TimeLimit', self.timelimit)
        # if self.gap > 0.0:
        #     m.setParam('MIPGap', self.gap)
        # if self.threads > 1:
        #     m.setParam('Threads', self.threads)
        # else:
        #     m.setParam('Threads', 1)
        m.setParam('Threads', 1)
        m.setParam('Seed', 10)


        # maximize the total number of slots
        # of = LinExpr()
        # # equation 5
        # for n in self.N_0:
        #     for t in self.T:
        #         for f in self.F:
        #             of += (sigma[t, f, n])

        # m.setObjective(of)
        m.setObjective(0)

        self.time_building += time.time() - start_building

        # Save the problem
        if visualize_solution:
            m.write('feasibility.lp')

        start_solving = time.time()
        m.optimize()
        self.time_solving += time.time() - start_solving

        status = m.Status

        feasible = None

        if status == GRB.Status.OPTIMAL:
            solution_sigma = m.getAttr('x', sigma)
            # print('There are %s solutions found.' % m.SOLCOUNT)
            # VISUALIZATION
            if self.visualization or solution_file or visualize_solution:
                viz = Visualization(len(self.T), len(self.F), self.N, self.P, interferers=self.my_interferers)
                # decision variables
                for n in self.N_0:
                    for t in self.T:
                        for f in self.F:
                            if solution_sigma[t, f, n] > self.EPSILON:
                                print("Optimal sigma\'s for n = {0}, t = {1}, f = {2} equals {3}".format(n, t, f, solution_sigma[t, f, n]))
                                viz.add_sigma(t, f, n, self.Y[n])
                viz.visualize(suffix='ga', output_dir=self.settings["ga"]["results_dir"])

            if solution_file:
                schedule = dict()
                for n in self.N_0:
                    if n not in schedule:
                        schedule[n] = dict()
                    for t in self.T:
                        for f in self.F:
                            if solution_sigma[t, f, n] > self.settings["feasibility"]["epsilon"]:
                                if t not in schedule[n]:
                                    schedule[n][t] = dict()
                                if f not in schedule[n][t]:
                                    schedule[n][t][f] = dict()
                                    schedule[n][t][f]['mcs'] = self.MCS[n]
                                    schedule[n][t][f]['slots'] = self.Y[n]
                                    schedule[n][t][f]['parent'] = self.P[n]

                ga_schedule = {}
                ga_schedule['schedule'] = schedule
                with open(solution_file, 'w') as outfile:
                    json.dump(ga_schedule, outfile)

            return True
        else:
            # if status == GRB.Status.INF_OR_UNBD or \
            #         status == GRB.Status.INFEASIBLE or \
            #         status == GRB.Status.UNBOUNDED:
            #     raise RuntimeError('The model cannot be solved because it is infeasible or unbounded')
            #     sys.exit(0)
            return False

# SLOTS_PER_MCS = dict({ # number of slots per bonded slot per MCS
#     "QPSK_FEC_3_4": 2,
#     "QPSK_FEC_1_2_FR_2": 4,
#     "QPSK_FEC_1_2": 3
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

# SLOTS_PER_MCS = dict({ # number of slots per bonded slot per MCS
#     "QPSK_FEC_3_4": 2,
#     "QPSK_FEC_1_2_FR_2": 4,
#     "QPSK_FEC_1_2": 3
# })
# model = Feasibility(visualization=True, nr_slots=4, nr_frequencies=1, slots_per_MCS=SLOTS_PER_MCS, settings_file="settings/settings.json")
# # links = [(0, None, None, None, None, []), (1, 0, '1.0', 2, 'QAM_16_FEC_3_4', []), (2, 1, '1.0', 2, 'QAM_16_FEC_3_4', []), (3, 0, '1.0', 1, 'QAM_16_FEC_3_4', [])]
# links = [(0, None, None, None, None, []), (1, 0, '1.0', 1, 'QPSK_FEC_3_4', [3]), (2, 0, '1.0', 1, 'QPSK_FEC_3_4', []), (3, 1, '1.0', 1, 'QPSK_FEC_3_4', [1])]
# model.set(links=links)
# start_two = time.time()
# print(model.check())
# print("Everything takes {0} s".format(time.time() - start_two))