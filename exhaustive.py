from settings import Settings
import logging
import random
import time
import feasibility
import throughput
import argparse
import os
import json
import math
import sys
import numpy
import itertools
import copy

class ExhaustiveSearch:
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

        if self.settings["modulations"]["modulations_file"]:
            self.__init_modulations(self.settings["modulations"]["modulations_file"])
        else:
            raise Exception('No modulations file given.')

        if self.settings["topology"]["topology_file"]:
            self.__init_topology(self.settings["topology"]["topology_file"])
        else:
            raise Exception('No topology file given.')

        # set the allowed parents and MCSs per parent per node
        self.__set_allowed_default()

        # initialize the feasibility model
        self.feasibility_model = feasibility.Feasibility(nr_slots=self.nr_slots, nr_frequencies=self.nr_frequencies, slots_per_MCS=self.d_MCS_to_slots, settings_file=settings_file)
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
        self.throughput_time = 0.0
        self.throughput_exec = 0
        self.total_time = 0.0

        self.best_individual = None
        self.best_individual_performance = []

        self.total_individuals = 0
        self.filtered_individuals = 0
        self.unique_individuals = 0
        self.valid_individuals = 0

    def __init_topology(self, json_topology):
        with open(json_topology) as json_file:
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
        with open(json_modulations_file) as json_file:
            data = json.load(json_file)

            self.d_MCS_to_rate = copy.deepcopy(data['modulations']['modulationRates'])

            if self.settings["simulator"]["modulationConfig"] in data["configurations"]:
                for ix, m in enumerate(data['configurations'][self.settings["simulator"]["modulationConfig"]]['allowedModulations']):
                    self.d_MCS_to_index[m] = ix
                    self.d_index_to_MCS[ix] = m
                    # print(m)
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

    def print_progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
        # Print New Line on Complete
        if iteration == total:
            logging.debug()

    def reach_root(self, node, par_parents, parents):
        if node not in par_parents:
            return False
        elif par_parents[node] in parents:
            return False
        elif par_parents[node] == self.settings["topology"]["root"]:
            return True
        parents.append(par_parents[node])
        return self.reach_root(par_parents[node], par_parents, parents)

    def __set_allowed_default(self):
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

    def evaluate_individual(self, links):
        # calculate total expected throughput
        start_time = time.time()
        self.throughput_model.set(links)
        tput = self.throughput_model.calculate()
        total_airtime = self.throughput_model.calculate_airtime()
        self.throughput_time += (time.time() - start_time)
        self.throughput_exec += 1

        # calculate the airtime for all the slots
        # total_airtime = 0.0
        # for child, parent, reliability, slots, mcs, interferers in links:
        #     if child != self.settings["topology"]["root"]:
        #         total_airtime += slots * self.settings["radio"]["airtimes"][mcs]

        # return both the throughput and airtime
        return tput, total_airtime

    def run(self):
        total_start_time = time.time()
        individual_of_lists = []
        # 1) for each node generate a list of possible parents, MCSs and max. slots
        for i in range(len(self.nodes_0)):
            node = i + 1
            allowed_parents = self.d_node_to_allowed_parents[node]
            allowed_MCSs = []
            for parent in allowed_parents:
                allowed_MCSs += self.d_node_to_allowed_MCSs_per_parent[node][parent]
            allowed_MCSs = list(set(allowed_MCSs)) # filter duplicates
            # TODO still have to check if the correct MCS is used with the correct parent
            max_nr_slots = 0
            for mcs in allowed_MCSs:
                # TODO still have to check if the correct nr of slots is used with the correct MCS
                if self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[mcs]] > max_nr_slots:
                    max_nr_slots = self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[mcs]]
            allowed_slots = list(range(max_nr_slots + 1))
            individual_of_lists.append(allowed_MCSs)
            individual_of_lists.append(allowed_slots)
            individual_of_lists.append(allowed_parents)

        # print(individual_of_lists)
        # exit()
        # all the possibilities are now ready
        total_possibilities = numpy.prod([len(l) for l in individual_of_lists])
        logging.info("The total number of possibilities: {0}".format(total_possibilities))
        ind_count = 1
        # the * makes sure you can give a list of lists to itertools.product()
        # iterator_all_individuals = itertools.product(*individual_of_lists)
        for ind in itertools.product(*individual_of_lists):
            if ind_count % self.settings["ga"]["output_x_generations"] == 0:
                logging.info("Running for individual {1} / {2}, being {0}.".format(ind, ind_count, total_possibilities))
            ind_count += 1
            self.total_individuals += 1
            ind = list(ind)
            individual_valid = True
            # test if the correct MCS is used with the correct parent
            links = []
            parents = {}
            dict_children = {}
            tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
            for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
                tmp_node = ix + 1
                # TEST if the correct mcs is used with the correct parent
                if mcs not in self.d_node_to_allowed_MCSs_per_parent[tmp_node][parent]:
                    individual_valid = False
                    break
                # TEST if the correct number of slots is used with the correct MCS
                if slots > self.d_MCS_to_max_bonded_slots[self.d_index_to_MCS[mcs]]:
                    individual_valid = False
                    break
                links.append((tmp_node,  # the first element is the child, but the root does not participate.
                              parent,
                              self.d_pdr[tmp_node][self.d_index_to_MCS[mcs]][parent],
                              slots,
                              self.d_index_to_MCS[mcs],
                              self.d_interferers[tmp_node]))
                parents[tmp_node] = parent
                if parent not in dict_children:
                    dict_children[parent] = []
                dict_children[parent].append(tmp_node)
            if not individual_valid:
                self.filtered_individuals += 1
                continue

            self.unique_individuals += 1

            # also add the root
            links.append((self.settings["topology"]["root"], None, None, None, None,
                          self.d_interferers[self.settings["topology"]["root"]]))

            # test if the individual is a valid topology.
            valid_tree = True
            valid_tree_start_time = time.time()
            for n in self.nodes:
                if n != 0 and not self.reach_root(n, parents, []):
                    valid_tree = False
                    break # go to next individual
            self.valid_tree_time += (time.time() - valid_tree_start_time)
            self.valid_tree_exec += 1
            if not valid_tree:
                continue

            # run the feasibility test
            feasible = False
            feasible_start_time = time.time()
            # enter the individual into the feasibility model and check for feasibility
            if self.fast_feasibility_check(individual=ind, dict_children=dict_children): # first check with fast feasibility check
                self.feasibility_model.set(links)
                feasible = self.feasibility_model.check()
            self.feasibility_time += (time.time() - feasible_start_time)
            self.feasibility_exec += 1
            if not feasible:
                continue

            # evaluate the individual
            tput, airtime = self.evaluate_individual(links)
            self.valid_individuals += 1

            # if self.best_individual is None or (tput > self.best_individual["tput"]) or (tput == self.best_individual["tput"] and airtime < self.best_individual["airtime"]):
            if self.best_individual is None or (tput > (self.best_individual["tput"] + 0.000000001)) or ((abs(tput - self.best_individual["tput"]) < 0.000000001) and (airtime + 0.000000001) < self.best_individual["airtime"]):
                if self.best_individual is None:
                    self.best_individual = {}
                self.best_individual["ind"] = ind.copy()
                self.best_individual["tput"] = tput
                self.best_individual["airtime"] = airtime
                self.best_individual["generation"] = 0

        self.total_time = time.time() - total_start_time

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

    def write_file(self):
        file_data = {}
        file_data["results"] = {}

        file_data["results"]["best_ind"] = self.best_individual
        file_data["results"]["best_ind_evolution"] = self.best_individual_performance

        file_data["results"]["hof"] = {}
        file_data["results"]["hof"]["0"] = {}
        file_data["results"]["hof"]["0"]["plain"] = self.best_individual["ind"].copy()
        file_data["results"]["hof"]["0"]["tput"] = self.best_individual["tput"]
        file_data["results"]["hof"]["0"]["airtime"] = self.best_individual["airtime"]

        file_data["results"]["individuals"] = {}
        file_data["results"]["individuals"]["total"] = self.total_individuals
        file_data["results"]["individuals"]["filtered"] = self.filtered_individuals
        file_data["results"]["individuals"]["unique"] = self.unique_individuals
        file_data["results"]["individuals"]["valid"] = self.valid_individuals
        file_data["results"]["time"] = {}
        file_data["results"]["time"]["total_time"] = self.total_time
        file_data["results"]["time"]["valid_tree"] = {"total": self.valid_tree_time, "nr": self.valid_tree_exec}
        file_data["results"]["time"]["feasibility"] = {"total": self.feasibility_time, "nr": self.feasibility_exec, "setting": self.feasibility_model.time_setting, "building": self.feasibility_model.time_building, "solving": self.feasibility_model.time_solving}
        file_data["results"]["time"]["throughput"] = {"total": self.throughput_time, "nr": self.throughput_exec}

        # tuple_per_node = list(zip(self.best_individual["ind"], self.best_individual["ind"][1:], self.best_individual["ind"][2:]))[::3]
        # links_tmp = []
        # for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
        #     tmp_node = ix + 1
        #     links_tmp.append((tmp_node,  # the first element is the child, but the root does not participate.
        #                   parent,
        #                   self.d_pdr[tmp_node][self.d_index_to_MCS[mcs]][parent],
        #                   slots,
        #                   self.d_index_to_MCS[mcs],
        #                   self.d_interferers[tmp_node]))
        # # also add the root
        # links_tmp.append((self.settings["topology"]["root"], None, None, None, None,
        #               self.d_interferers[self.settings["topology"]["root"]]))

        links_tmp, parents_tmp = self.pack_individual(self.best_individual['ind'])

        if not os.path.exists(self.settings["ga"]["results_dir"]):
            os.mkdir(self.settings["ga"]["results_dir"])

        self.feasibility_model.set(links_tmp)
        schedule_file = "{0}/ga-schedule.json".format(self.settings["ga"]["results_dir"])
        feasible = self.feasibility_model.check(solution_file=schedule_file)
        if feasible:
            # add parents
            data = None
            with open(schedule_file) as json_file:
                data = json.load(json_file)
            data['parents'] = parents_tmp
            with open(schedule_file, 'w') as json_file:
                json.dump(data, json_file)
        else:
            raise BaseException(
                "The best individual {0} is not deemed feasible by the ILP.".format(self.best_individual['ind']))

        file_name = self.settings["ga"]["results_dir"] + self.settings["ga"]["results_file_prefix"] + ".json"
        # file_name = self.settings["ga"]["results_dir"] + self.settings["ga"]["results_file_prefix"] + "-" + time.strftime("%Y%m%d-%H%M%S") + ".json"
        with open(file_name, 'w') as json_file:
            json.dump(file_data, json_file)

        logging.info("There were {0} invalid individuals out of a total of {1} individuals.".format(self.filtered_individuals, self.total_individuals))
        if self.valid_tree_exec > 0:
            logging.info("Check valid tree stats: {0} runs, {1} s total time, {2} s on average run".format(self.valid_tree_exec, self.valid_tree_time, self.valid_tree_time / float(self.valid_tree_exec)))
        else:
            logging.info("Did not run valid tree check once...")
        if self.feasibility_exec > 0:
            logging.info("Feasibility model stats: {0} runs, {1} s total time, {2} s on average run".format(self.feasibility_exec, self.feasibility_time, self.feasibility_time / float(self.feasibility_exec)))
            logging.info("- total setting time: {0}".format(self.feasibility_model.time_setting))
            logging.info("- total building time: {0}".format(self.feasibility_model.time_building))
            logging.info("- total solving time: {0}".format(self.feasibility_model.time_solving))
        else:
            logging.info("Did not run feasibility model once...")
        if self.throughput_exec > 0:
            logging.info("Throughput calculation stats: {0} runs, {1} s total time, {2} s on average run".format(self.throughput_exec, self.throughput_time, self.throughput_time / float(self.throughput_exec)))
        else:
            logging.info("Did not run throughput calculation once...")

        logging.info("Total time: {0} s".format(self.total_time))

def run_exhaustive(input=None, loglevel=None):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=getattr(logging, loglevel.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

    es = ExhaustiveSearch(input)
    es.run()
    # es.draw()
    # es.print_hall_of_fame()
    es.write_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--loglevel', '-l', type=str)

    args = parser.parse_args()
    settings_file = str(args.input)
    loglevel = str(args.loglevel)

    run_exhaustive(input=settings_file, loglevel=loglevel)