import heuristic2phys
from settings import Settings
import logging
import math
import copy
import json


class FullHeuristic:

    def __init__(self, settings_file=None):
        self.settings = Settings(settings_file)
        # read from modulations file
        self.d_MCS_to_slots = {} # maps MCS to number of slots in one bonded slot
        self.d_MCS_to_max_bonded_slots = {} # maps MCS to max. nr of bonded slots in slot frame
        self.d_MCS_to_index = {} # maps MCS to an index (I think a list is not as clear)
        self.d_index_to_MCS = {} # maps the index to the MCS
        self.d_MCS_to_rate = {} # maps MCS to an index (I think a list is not as clear)

        # topology variables
        self.nodes = [] # all nodes
        self.nodes_0 = [] # nodes without root
        self.d_parent = {} # maps a node to its parent
        self.d_interferers = {} # maps a node to its interferers

        self.nodes_d_hst_ix = {}
        self.nodes_d_ix_hst = {}

        self.d_pdr = {} # maps node to MCS to parent to PDR, read literally from topology file

        self.d_node_to_allowed_parents = {}  # maps a node to its allowed parents in the individual
        self.d_node_to_allowed_MCSs_per_parent = {}  # maps a node to its allowed MCSs per parent in the individual

        if self.settings["modulations"]["modulations_file"]:
            self.__init_modulations(self.settings["modulations"]["modulations_file"])
        else:
            raise Exception('No modulations file given.')

        if self.settings["topology"]["topology_file"]:
            self.__init_topology(self.settings["topology"]["topology_file"])
        else:
            raise Exception('No topology file given.')

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
            print(nodes_hostnames)
            # remove the root from the list
            nodes_hostnames.remove(self.settings["sf-heuristic"]["testbed_results"]["root"])
            # place the root in the front of the list so it can get index 0
            nodes_hostnames.insert(0, self.settings["sf-heuristic"]["testbed_results"]["root"])
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

    def __init_modulations(self, json_modulations_file):
        ix_phy = 0
        for phy in self.settings["sf-heuristic"]["testbed_results"]["reliabilities"]:
            self.d_MCS_to_rate[phy] = self.settings["sf-heuristic"]["testbed_results"]["rates"][phy]
            self.d_MCS_to_slots[phy] = self.settings["sf-heuristic"]["testbed_results"]["slots"][phy]
            self.d_MCS_to_index[phy] = ix_phy
            self.d_index_to_MCS[ix_phy] = phy
            ix_phy += 1

        self.d_MCS_to_max_bonded_slots = {}
        for mcs, slots in self.d_MCS_to_slots.items():
            self.d_MCS_to_max_bonded_slots[mcs] = math.floor(self.nr_slots / slots)

    def reach_root(self, node, par_parents, parents):
        if node not in par_parents:
            return False
        elif par_parents[node] in parents:
            return False
        elif par_parents[node] == self.settings["topology"]["root"]:
            return True
        parents.append(par_parents[node])
        return self.reach_root(par_parents[node], par_parents, parents)

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
                if abs(self.d_pdr[n][m][p] - self.d_pdr[n][most_reliable][p]) <= delta and self.d_MCS_to_rate[m] > self.d_MCS_to_rate[m_n[p]]:
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