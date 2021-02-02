# import permutations
import time
import itertools
from settings import Settings

def erange(start, end):
    ''' range function including the end in the returned range '''
    return range(start, end + 1)

class Throughput:

    def __init__(self, r_max, max_queue_length, generated_packets_at_node, nr_slots, settings_file):
        self.nodes = []
        self.slots = {}
        self.children = {} # map
        self.parent = {}
        self.descendants = {}
        self.l_descendants = {}
        self.reliabilities = {}
        self.modulations = {}

        self.depth = {}

        self.R_MAX = r_max
        self.MAX_QUEUE_LENGTH = max_queue_length
        self.GENERATED_PACKETS_AT_NODE = generated_packets_at_node
        self.NR_SLOTS = nr_slots

        # self.EPSILON = 0.00000001

        # caching
        self.dict_t_C_p = {}  # cache t_C_p values
        self.cached_calls = 0

        self.B_matrices = {}
        self.absorbing_states_with_j_arrivals = {}

        self.B_calculations = 0
        self.absorbing_states_calculations = 0

        self.p_dict = {}
        self.p = {}

        # permutations
        # self.permutations = permutations.Permutations()

        self.permutations_time = 0.0
        self.permutations_types = []
        self.total_time = 0.0

        self.settings = Settings(settings_file)
  #       self.settings = {'topology': {'root': 0}, 'throughput': {'epsilon': 1e-07}, "radio": {
  #           "airtimes": {
  #               "QPSK_FEC_1_2": {
  #                   "idle": 3.0,
  #                   "txDataNoAck": 12.12,
  #                   "rxDataTxAck": 14.24,
  #                   "txDataRxNack": 14.24,
  #                   "txDataRxAck": 14.24,
  #                   "rxDataTxNack": 14.24
  #               },
  #               "QPSK_FEC_1_2_FR_2": {
  #                   "idle": 3.0,
  #                   "txDataNoAck": 22.28,
  #                   "rxDataTxAck": 26.56,
  #                   "txDataRxNack": 26.56,
  #                   "txDataRxAck": 26.56,
  #                   "rxDataTxNack": 26.56
  #               },
  #               "QPSK_FEC_3_4": {
  #                   "idle": 3.0,
  #                   "txDataNoAck": 8.73,
  #                   "rxDataTxAck": 10.13,
  #                   "txDataRxNack": 10.13,
  #                   "txDataRxAck": 10.13,
  #                   "rxDataTxNack": 10.13
  #               }
  #           }
  # },}

    def hard_reset(self):
        self.nodes = []
        self.children = {}
        self.slots = {}
        self.reliabilities = {}
        self.descendants = {}
        self.absorbing_states_with_j_arrivals = {}
        self.modulations = {}

        self.depth = {}
        self.max_depth = 0

        self.p = {}
        self.dict_t_C_p = {}  # cache t_C_p values

    def set(self, links):
        # reset
        self.hard_reset()
        # fill, do nothing with the interferers
        for child, parent, reliability, slots, mcs, interferers in links:
            if child != self.settings["topology"]["root"]: # the root can also be in the links, but is not interesting for this calculation
                # add children
                if child not in self.nodes:
                    self.nodes.append(child)
                if parent not in self.nodes:
                    self.nodes.append(parent)
                # add slots
                self.slots[child] = int(slots)
                # add to children
                if parent not in self.children:
                    self.children[parent] = []
                self.children[parent].append(child)
                self.parent[child] = parent
                # add reliabilities
                self.reliabilities[child] = float(reliability)
                self.modulations[child] = mcs

        # calculate all descendants from root
        self.calculate_descendants()
        # print(self.descendants)
        # calculate the depths
        self.calcalute_depth()
        # print(self.depth)
        # print(self.children)
        # print(self.reliabilities)
        # print(self.slots)
        # assert False

    # def set_p(self, dir_name, pdr):
    #     file_name = '{0}/p_{1}.csv'.format(dir_name, int(pdr*1000))
    #     pdr = float(pdr)
    #     if pdr not in self.p_dict:
    #         self.p_dict[pdr] = {}
    #     with open(file_name, 'r') as f:
    #         for line in f:
    #             values = line.strip().split(',')
    #             if len(values) == 4: # only correct in this case
    #                 self.p_dict[pdr][(int(values[0]), int(values[1]), int(values[2]))] = float(values[3])

    # def set_p(self, dir_name, pdr):
    #     file_name = '{0}/p_{1}.csv'.format(dir_name, int(pdr*1000))
    #     pdr = float(pdr)
    #     if pdr not in self.p_dict:
    #         self.p_dict[pdr] = {}
    #     with open(file_name, 'r') as f:
    #         for line in f:
    #             values = line.strip().split(',')
    #             if len(values) == 4: # only correct in this case
    #                 self.p_dict[pdr][(int(values[0]), int(values[1]), int(values[2]))] = float(values[3])

    def set_p(self, dir_name, pdr):
        '''
        I did some minor testing with various forms of saving the information in a dictionary and than looking it up.
        This in combination with a lambda lookup turned out to be quite fast with less elements in storage.
        So I do not store the values that are smaller or equal than EPSILON.
        '''
        file_name = '{0}/p_{1}.csv'.format(dir_name, int(pdr * 1000))
        with open(file_name, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                if len(values) == 5 and float(values[3]) > self.settings["throughput"]["epsilon"]:  # only correct in this case
                    self.p_dict[(pdr, int(values[0]), int(values[1]), int(values[2]))] = (float(values[3]), float(values[4]))

    def calculate_descendants(self, node=0):
        if node not in self.descendants:
            self.descendants[node] = 0
            self.l_descendants[node] = []
        if node in self.children:
            self.descendants[node] += len(self.children[node])
            self.l_descendants[node] += self.children[node]
            for c in self.children[node]:
                self.calculate_descendants(node=c)
                self.descendants[node] += self.descendants[c]
                self.l_descendants[node] += self.l_descendants[c]

    def check_tree_validity(self):
        return self.descendants[self.settings["topology"]["root"]] + 1 == len(self.nodes) and len(self.l_descendants) == len(set(self.l_descendants))

    def calcalute_depth(self, node=0, depth=0):
        if depth > self.max_depth:
            self.max_depth = depth
        if depth not in self.depth:
            self.depth[depth] = []
        self.depth[depth].append(node)
        if node in self.children:
            for c in self.children[node]:
                self.calcalute_depth(node=c, depth=(depth + 1))

    # def p_dict_lookup(self, x=None, q=None, s=None, l=None):
    #     # if l in self.p_dict:
    #     #     return self.p_dict[l][(x,q,s)]
    #     # else:
    #     #     print(self.p_dict)
    #     #     print(l)
    #     #     assert False
    #     if l in self.p_dict:
    #         if (x,q,s) in self.p_dict[l]:
    #             return self.p_dict[l][(x,q,s)]
    #         else:
    #             return 0
    #     else:
    #         assert False

    def partition(self, largest, lst, length, *rest):
        '''
        Partitions a number 'largest' into all possible sums adding to largest.
        :param largest: the initial number from which you start and is used for recursive calls
        :param lst: holds the partitions that are of size 'length' or less
        :param length: the length of the partitions that should be hold in lst
        :param rest: represents a tuple to pass to recursive calls
        :return:
        '''
        l = [largest] + list(rest)
        if l and len(l) <= length:
            lst.append(l)
        min = rest[0] if rest else 1
        max = largest // 2
        for n in range(min, max+1):
            self.partition(largest-n, lst, length, n, *rest)

    def get_all_partitions(self, number, length):
        '''
        Returns a list of lists with all partitions of size 'length' of the parameter 'number'.
        Partitions that were shorter are extended with zeroes.
        '''
        lst = []
        self.partition(number, lst, length)
        for l in lst:
            if len(l) < length: # extend with zeroes if size is smaller than length.
                l.extend((length - len(l)) * [0])
        return lst

    # def t_C_n(self, n=None, packets=None):
    #     if (n, packets) in self.dict_t_C_p:
    #         self.cached_calls += 1
    #         return self.dict_t_C_p[(n, packets)]
    #     elif packets == 0:
    #         return 0.0
    #
    #     total_seen = 0
    #
    #     # create a list of the max. number of packets each child could generate, so you can limit the permutations
    #     l_max_packets = [self.descendants[c] + self.GENERATED_PACKETS_AT_NODE for c in self.children[n]]
    #     assert len(l_max_packets) == len(self.children[n])
    #     # the max number of packets that can be generated by one of the children
    #     max_packets = max(l_max_packets)
    #
    #     # the total t_C_n
    #     t_C_n = 0.0
    #
    #     # start the timer
    #     start_time = time.time()
    #
    #     # 1) get all partitions of the number of packets over n nodes
    #     partitions = (self.get_all_partitions(number=packets, length=len(self.children[n])))
    #     # 2) go over all partitions and calculate the permutation for each partition
    #     for partition in partitions:
    #         # if there is any number of packets in the partition that is higher than one of the children can transmit, skip the partition (because the prob. would be zero)
    #         if any(x > max_packets for x in partition):
    #             continue
    #         # 3) go over all permutations of this partition
    #         seen = set()  # keep a history of the current permutations of this partition
    #         for permutation in itertools.permutations(partition): # itertools.permutations returns a iterator
    #             # check two things before you proceed with the calculation of the probability
    #             # - if this is not a duplicate permutation (is possible because 5 1 1 and 5 1 1 can be two different permutations)
    #             # - the number of packets assigned to each node in the permutation should be lower or equal to the allowed number of packets in l_max_packets
    #             if permutation not in seen and any((x >= y) for (x, y) in zip(l_max_packets, permutation)):
    #                 seen.add(permutation)
    #                 # get the probability for this permutation
    #                 prob = 1.0
    #                 for ix, child in enumerate(self.children[n]):
    #                     prob *= self.p[child][permutation[ix]]
    #                 # add to the total t_C_n
    #                 t_C_n += prob
    #         total_seen += len(seen)
    #
    #     # keep the spent time
    #     elapsed = (time.time() - start_time)
    #     self.permutations_time += (time.time() - start_time)
    #     self.permutations_types.append((n, packets, len(self.children[n]), l_max_packets, max_packets, total_seen, elapsed))
    #
    #     # cache it
    #     self.dict_t_C_p[(n, packets)] = t_C_n
    #
    #     return t_C_n

    def t_C_n(self, n=None, packets=None):
        if (n, packets) in self.dict_t_C_p:
            self.cached_calls += 1
            return self.dict_t_C_p[(n, packets)]
        # elif packets == 0:
        #     return 0.0
        # this can't be always zero!

        total_seen = 0

        # create a list of the max. number of packets each child could generate, so you can limit the permutations
        l_max_packets = [((self.descendants[c] * self.GENERATED_PACKETS_AT_NODE) + self.GENERATED_PACKETS_AT_NODE) for c in self.children[n]]
        assert len(l_max_packets) == len(self.children[n])
        # the max number of packets that can be generated by one of the children
        max_packets = max(l_max_packets)
        # the maximum size is limited by the number of packets one node an send
        # max_packets = min(max_packets, self.MAX_QUEUE_LENGTH + self.GENERATED_PACKETS_AT_NODE)
        max_packets = min(max_packets, self.MAX_QUEUE_LENGTH)

        # the total t_C_n
        t_C_n = 0.0

        # start the timer
        start_time = time.time()

        # 1) get all partitions of the number of packets over n nodes
        partitions = (self.get_all_partitions(number=packets, length=len(self.children[n])))
        # 2) go over all partitions and calculate the permutation for each partition
        for partition in partitions:
            # if there is any number of packets in the partition that is higher than one of the children can transmit, skip the partition (because the prob. would be zero)
            if any(x > max_packets for x in partition):
                continue
            # perm = 0
            # 3) go over all permutations of this partition
            seen = set()  # keep a history of the current permutations of this partition
            for permutation in itertools.permutations(partition): # itertools.permutations returns a iterator
                # perm += 1
                # print(perm)
                # check two things before you proceed with the calculation of the probability
                # - if this is not a duplicate permutation (is possible because 5 1 1 and 5 1 1 can be two different permutations)
                # - the number of packets assigned to each node in the permutation should be lower or equal to the allowed number of packets in l_max_packets
                if permutation not in seen and any((x >= y) for (x, y) in zip(l_max_packets, permutation)):
                    seen.add(permutation)
                    # get the probability for this permutation
                    prob = 1.0
                    for ix, child in enumerate(self.children[n]):
                        try:
                            pr = self.p[child][permutation[ix]]
                            prob *= pr
                        except IndexError:
                            # print("child: ", child)
                            # print("permutation: ", permutation)
                            # print("ix: ", ix)
                            raise BaseException('Oow, should NOT happen...')
                    # add to the total t_C_n
                    t_C_n += prob
            total_seen += len(seen)

        # keep the spent time
        elapsed = (time.time() - start_time)
        self.permutations_time += (time.time() - start_time)
        # self.permutations_types.append((n, packets, len(self.children[n]), l_max_packets, max_packets, total_seen, elapsed))

        # cache it
        self.dict_t_C_p[(n, packets)] = t_C_n

        return t_C_n

    def calculate(self):
        '''
        Calculate the expected throughput at the root
        :return: The expected throughput in the current topology
        '''
        if len(self.p_dict) == 0:
            raise Exception('p_dict is empty. No reliabilities loaded in memory.')
        start_time = time.time()
        # MAX_PACKETS = self.MAX_QUEUE_LENGTH + self.GENERATED_PACKETS_AT_NODE
        MAX_PACKETS = self.MAX_QUEUE_LENGTH
        # print(MAX_PACKETS)
        # determine the correct order to go through the tree
        # start at max depth and go to 0
        depth_list = reversed(erange(1, self.max_depth)) # the root should not be used that is why you start at 1
        iteration_order = [n for sublist in [self.depth[d] for d in depth_list] for n in sublist]
        available_n = dict()
        for n in iteration_order: # go over all nodes in correct order
            if n not in self.p: # all probabilities are initially 0
                self.p[n] = [0.0] * (MAX_PACKETS + 1) # add one for 0 packets arriving, so in total you have (MAX_PACKETS + 1) probabilities
            max = min(MAX_PACKETS, self.descendants[n] * self.GENERATED_PACKETS_AT_NODE + self.GENERATED_PACKETS_AT_NODE)
                # assert False
            for x in erange(0, max):
                # for all p values from 0 to max (MAX_PACKETS or limited to self.descendants[n] + self.GENERATED_PACKETS_AT_NODE, because no more packets can arrive, so limit nr of calculations)
                for packets in erange(0, self.descendants[n] * self.GENERATED_PACKETS_AT_NODE): # sum from 0 to number of descendants * self.GENERATED_PACKETS_AT_NODE
                    packets_plus_generated_at_node = min(self.MAX_QUEUE_LENGTH, packets + self.GENERATED_PACKETS_AT_NODE)
                    # packets_plus_generated_at_node = packets + self.GENERATED_PACKETS_AT_NODE
                    # if you would have left packets_plus_generated_at_node = packets + self.GENERATED_PACKETS_AT_NODE
                    # then the lambda would return 0.0 when packets (generated by children) = 10 because there is no entry for packets_plus_generated_at_node = 11
                    # and if there is no entry for 11, the lambda returns 0.0, making the rslt = 0.0 for that node when packets = 10
                    # while the rslt for that node when packets = 10 should not be 0 as still 10 packets out of 11 can arrive (if eg rel = 1.0)
                    # packets_plus_generated_at_node = packets + self.GENERATED_PACKETS_AT_NODE
                    # if the node has no descendants, it will return a list [0]
                    if x <= packets_plus_generated_at_node:
                        # print('n', n)
                        # print('(packets + self.GENERATED_PACKETS_AT_NODE)', (packets + self.GENERATED_PACKETS_AT_NODE))
                        # there can never arrive more packets than the number being transmitted, so leave do not add anything to p[n][x] if x > packets + self.GENERATED_PACKETS_AT_NODE because it will zero anyhow
                        if n in self.children:
                            rel = self.reliabilities[n]
                            sl = self.slots[n]
                            # result_lambda = (lambda t: self.p_dict[t][0] if t in self.p_dict else (1.0 if (t[1] == 0 and (t[0] == 0.0 or t[2] == 0.0 or t[3] == 0.0)) else 0.0))((self.reliabilities[n], x, (packets + self.GENERATED_PACKETS_AT_NODE), self.slots[n]))
                            result_lambda = (lambda t: self.p_dict[t][0] if t in self.p_dict else 0.0)((round(self.reliabilities[n], 3), x, packets_plus_generated_at_node, self.slots[n]))
                            tcn = self.t_C_n(n=n, packets=packets)
                            rslt = tcn * result_lambda # multiply the prob. of "packets" arriving from the children with the probability that x packets will arrive at the parent of this node at the end of this slot frame
                            self.p[n][x] += rslt
                            # print(self.p[n][x])
                        else:
                            # Small note here: in the case if t not in p_dict: in the cases the prob should be 1 (see original lambda), these cases ARE included in the p_dict... so you will never reach this.
                            # So only the else case seems important here. That is why I removed those 1.0 cases. In the calculation of B I seem to have taken care of these special cases where the prob. should be 1.0,
                            # so normally they are all in the p_files where prob. should be 1.0.
                            # self.p[n][x] += (lambda t: self.p_dict[t][0] if t in self.p_dict else (1.0 if (t[1] == 0 and (t[0] == 0.0 or t[2] == 0.0 or t[3] == 0.0)) else 0.0))((self.reliabilities[n], x, (packets + self.GENERATED_PACKETS_AT_NODE), self.slots[n]))
                            self.p[n][x] += (lambda t: self.p_dict[t][0] if t in self.p_dict else 0.0)((round(self.reliabilities[n], 3), x, packets_plus_generated_at_node, self.slots[n]))
                            # print(self.p[n][x])
        # print(self.parent)
        # print(self.descendants)
        # calculate throughput for root
        tput = 0.0
        for packets in erange(0, self.descendants[self.settings["topology"]["root"]] * self.GENERATED_PACKETS_AT_NODE):
            # print(packets)
            tput += self.t_C_n(n=self.settings["topology"]["root"], packets=packets) * packets
            # print('tput', packets, self.t_C_n(n=self.settings["topology"]["root"], packets=packets))

        self.total_time = time.time() - start_time

        # print("Expected throughput at root is {0} packets.".format(tput))
        # print("Number of cached calls t_C_n = {0}".format(self.cached_calls))

        return tput

    def calculate_airtime(self, verbose=False):
        '''
        Calculate the expected airtime of the whole network
        :return: The expected airtime in the current topology
        '''
        if len(self.p_dict) == 0:
            raise Exception('p_dict is empty. No reliabilities loaded in memory.')
        depth_list = reversed(erange(1, self.max_depth)) # the root should not be used that is why you start at 1
        iteration_order = [n for sublist in [self.depth[d] for d in depth_list] for n in sublist]
        available_n = dict()
        airtime = 0.0
        for n in iteration_order: # go over all nodes in correct order
            # calculate the average number of available number of packets at node n for the airtime calculation
            available_n[n] = 0.0
            if n in self.children:
                for packets in erange(0, self.descendants[n] * self.GENERATED_PACKETS_AT_NODE):
                    available_n[n] += (self.t_C_n(n=n, packets=packets) * packets)
            available_n[n] += self.GENERATED_PACKETS_AT_NODE
            available_n[n] = min(self.MAX_QUEUE_LENGTH, round(available_n[n]))

            airtime_n = 0.0
            # you can only transmit as much packets as there are slots, so min(slots, packets)
            for packets in erange(0, min(self.slots[n], available_n[n])):
                if (round(self.reliabilities[n], 3), packets, available_n[n], self.slots[n]) in self.p_dict: # this means we have a non-zero probability
                    probability = self.p_dict[round(self.reliabilities[n], 3), packets, available_n[n], self.slots[n]][0]
                    # print("round(self.reliabilities[n], 3), packets, available_n[n], self.slots[n]: ", round(self.reliabilities[n], 3), packets, available_n[n], self.slots[n])
                    # print("probability: ", probability)
                    average_nr_slots_left = self.p_dict[round(self.reliabilities[n], 3), packets, available_n[n], self.slots[n]][1]
                    # print("average_nr_slots_left: ", average_nr_slots_left)
                    average_nr_slots_used = self.slots[n] - average_nr_slots_left
                    # print("average_nr_slots_used: ", average_nr_slots_used)
                    if average_nr_slots_used < packets:
                        raise BaseException("What the bloody hell is going on?")
                    tmp_airtime_n = ((packets * (self.settings["radio"]["airtimes"][self.modulations[n]]['txDataRxAck'] + self.settings["radio"]["airtimes"][self.modulations[n]]['rxDataTxAck'])) +
                          (average_nr_slots_left * self.settings["radio"]["airtimes"][self.modulations[n]]['idle']) +
                          ((average_nr_slots_used - packets) * (1 - self.reliabilities[n]) * (self.settings["radio"]["airtimes"][self.modulations[n]]['txDataNoAck'] + self.settings["radio"]["airtimes"][self.modulations[n]]['idle'])) +
                          ((average_nr_slots_used - packets) * self.reliabilities[n] * (self.settings["radio"]["airtimes"][self.modulations[n]]['txDataRxNack'] + self.settings["radio"]["airtimes"][self.modulations[n]]['rxDataTxNack'])))
                    if verbose:
                        print("Partial airtime (with probability {3}) for node n {0} and nr packets = {2} is {1} ms (slots used = {4}, slots left = {5})".format(n, tmp_airtime_n, packets, probability, average_nr_slots_used, average_nr_slots_left))
                    airtime_n += (probability * tmp_airtime_n)
                else: # we have zero probability of packets out of available_n[n] packets arriving at the parent of n
                    airtime_n += 0.0
            if verbose:
                print("Airtime for node n {0} with reliability {2}, nr slots = {3} is {1} ms".format(n, airtime_n, self.reliabilities[n], self.slots[n]))
            # add to the total airtime
            airtime += airtime_n

        return airtime


# links = [('1', '0', 1.0, 10, 'MCS0'), ('2', '1', 0.5, 5, 'MCS0'), ('3', '1', 1.0, 1, 'MCS0'), ('4', '2', 1.0, 1, 'MCS0'), ('5', '2', 1.0, 1, 'MCS0')]
# links = [('1', '0', 1.0, 10, 'MCS0', []), ('2', '1', 0.5, 5, 'MCS0', []), ('6', '1', '1.0', 5, 'MCS0', []), ('7', '1', 0.5, 5, 'MCS0', []), ('8', '1', 0.5, 5, 'MCS0', []), ('9', '1', 0.5, 5, 'MCS0', []), ('10', '1', 0.5, 5, 'MCS0', []), ('11', '1', 0.5, 5, 'MCS0', []), ('3', '1', 1.0, 1, 'MCS0', []), ('4', '2', 1.0, 1, 'MCS0', []), ('5', '2', 1.0, 1, 'MCS0', [])]
# links = [(1, 0, 0.75, 5, 'MCS0', []), (2, 0, 0.75, 5, 'MCS0', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 3, 'MCS0', []), (2, 1, 1.0, 2, 'MCS0', []), (3, 2, 1.0, 1, 'MCS0', []), (0, None, None, None, None, [])]
# links = [(1, 0, 0.75, 5, 'MCS0', []), (0, None, None, None, None, [])]
#
# links = [(1, 0, 1.0, 10, 'MCS0', []), (2, 1, 1.0, 5, 'MCS0', []), (3, 1, 1.0, 1, 'MCS0', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 14, 'MCS0', []), (2, 1, 1.0, 13, 'MCS0', []), (3, 2, 1.0, 12, 'MCS0', []), (4, 3, 1.0, 11, 'MCS0', []), (5, 4, 1.0, 10, 'MCS0', []), (6, 5, 1.0, 9, 'MCS0', []), (7, 6, 1.0, 8, 'MCS0', []), (8, 7, 1.0, 7, 'MCS0', []), (9, 8, 1.0, 6, 'MCS0', []), (10, 9, 1.0, 5, 'MCS0', []), (11, 10, 1.0, 4, 'MCS0', []), (12, 11, 1.0, 3, 'MCS0', []), (13, 12, 1.0, 2, 'MCS0', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 14, 'QPSK_FEC_1_2_FR_2', []), (2, 1, 1.0, 13, 'QPSK_FEC_1_2_FR_2', []), (3, 2, 1.0, 12, 'QPSK_FEC_1_2_FR_2', []), (4, 3, 1.0, 11, 'QPSK_FEC_1_2_FR_2', []), (5, 4, 1.0, 10, 'QPSK_FEC_1_2_FR_2', []), (6, 5, 1.0, 9, 'QPSK_FEC_1_2_FR_2', []), (7, 6, 1.0, 8, 'QPSK_FEC_1_2_FR_2', []), (8, 7, 1.0, 7, 'QPSK_FEC_1_2_FR_2', []), (9, 8, 1.0, 6, 'QPSK_FEC_1_2_FR_2', []), (10, 9, 1.0, 5, 'QPSK_FEC_1_2_FR_2', []), (11, 10, 1.0, 4, 'QPSK_FEC_1_2_FR_2', []), (12, 11, 1.0, 3, 'QPSK_FEC_1_2_FR_2', []), (13, 12, 1.0, 2, 'QPSK_FEC_1_2_FR_2', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 14, 'MCS0', []), (2, 1, 1.0, 13, 'MCS0', []), (3, 1, 1.0, 12, 'MCS0', []), (4, 1, 1.0, 11, 'MCS0', []), (5, 1, 1.0, 10, 'MCS0', []), (6, 1, 1.0, 9, 'MCS0', []), (7, 1, 1.0, 8, 'MCS0', []), (8, 1, 1.0, 7, 'MCS0', []), (9, 1, 1.0, 6, 'MCS0', []), (10, 1, 1.0, 5, 'MCS0', []), (11, 1, 1.0, 4, 'MCS0', []), (12, 1, 1.0, 3, 'MCS0', []), (13, 1, 1.0, 2, 'MCS0', []), (0, None, None, None, None, [])]
#
# links = [(1, 0, 1.0, 14, 'MCS0', []), (2, 1, 1.0, 13, 'MCS0', []), (3, 1, 1.0, 12, 'MCS0', []), (4, 1, 1.0, 11, 'MCS0', []), (5, 1, 1.0, 10, 'MCS0', []), (6, 1, 1.0, 9, 'MCS0', []), (7, 1, 1.0, 8, 'MCS0', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 14, 'MCS0', []), (2, 1, 1.0, 13, 'MCS0', []), (3, 2, 1.0, 12, 'MCS0', []), (4, 3, 1.0, 11, 'MCS0', []), (5, 4, 1.0, 10, 'MCS0', []), (6, 5, 1.0, 9, 'MCS0', []), (7, 6, 1.0, 8, 'MCS0', []), (8, 7, 1.0, 7, 'MCS0', []), (0, None, None, None, None, [])]
#
# links = [(1, 0, 1.0, 10, 'QPSK_FEC_1_2_FR_2', []), (2, 1, 1.0, 5, 'QPSK_FEC_1_2_FR_2', []), (0, None, None, None, None, [])]
# links = [(1, 0, 1.0, 2, 'QPSK_FEC_1_2_FR_2', []), (2, 1, 1.0, 1, 'QPSK_FEC_1_2_FR_2', []), (0, None, None, None, None, [])]
#
#
#
#
# tput = Throughput(r_max=4, max_queue_length=10, generated_packets_at_node=1, nr_slots=3, settings_file=None)
# tput.set(links=links)
# # tput.set_p(dir_name='p_files/output_A_11_Q_11_S_11_RMAX_3', pdr=0.0)
# # tput.set_p(dir_name='p_files/output_A_11_Q_11_S_11_RMAX_3', pdr=0.5)
# tput.set_p(dir_name='/Users/gdaneels/Documents/workspace/ga/p_files/output_A_10_Q_10_S_40_RMAX_4', pdr=1.0)
# print("Depth {0}".format(tput.max_depth))
# print("Depths {0}".format(tput.depth))
# print(tput.calculate())
# print(tput.calculate_airtime(verbose=True))
# print("Calculation permutations types = {0}".format(tput.permutations_types))
# print("Calculation permutations took {0} s".format(tput.permutations_time))
# print("Throughput calculation will take {0} s".format(tput.total_time))
