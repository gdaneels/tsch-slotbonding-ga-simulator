import json
import logging
import argparse
import sys
import numpy

from settings import Settings

class Dijkstra:
    def __init__(self, settings_file=None, topology_file=None):
        self.d_node_to_allowed_parents = {}
        self.dijkstra_table = {}
        self.visited = []
        self.unvisited = []
        self.path_cost = 1
        self.settings = None

    def __reset(self):
        self.d_node_to_allowed_parents = {}
        self.dijkstra_table = {}
        self.visited = []
        self.unvisited = []
        self.settings = None

    def __init_topology(self, topology_file=None):
        with open(topology_file) as json_file:
            data = json.load(json_file)
            # read out topology
            for node, info in data["simulationTopology"].items():
                node = int(node) # convert it to an int
                self.unvisited.append(node)
                self.dijkstra_table[node] = {'distance': None, 'prev': None}

                # save the reliabilities
                if node not in self.d_node_to_allowed_parents:
                    self.d_node_to_allowed_parents[node] = []
                # I am manually reading it out because I want to convert all the nodes to actual integers
                for mcs, reliabilities in data["simulationTopology"][str(node)]["reliability"].items():
                    for p, pdr in reliabilities.items():
                        if self.settings["ga"]["max_pdr"] >= pdr >= self.settings["ga"]["min_pdr"]:
                            if int(p) not in self.d_node_to_allowed_parents[node]:
                                self.d_node_to_allowed_parents[node].append(int(p))

            # logging.info("All allowed parents for each node: {0}".format(self.d_node_to_allowed_parents))

    def calculate(self, topology_file, settings_file):
        self.__reset() # reset all the variables
        # load new topology and settings file
        if not settings_file:
            raise Exception('No settings file given.')
        self.settings = Settings(settings_file)
        if not topology_file:
            raise Exception('No topology file given.')
        self.__init_topology(topology_file)

        firstRun = True
        while len(self.unvisited) > 0:
            if firstRun:
                node = self.settings["topology"]["root"]
                firstRun = False
            else:
                shortest_distance = 100000
                for n in self.unvisited:
                    if self.dijkstra_table[n]['distance'] is not None and self.dijkstra_table[n]['distance'] < shortest_distance:
                        node = n
                        shortest_distance = self.dijkstra_table[n]['distance']

            neighbors = self.d_node_to_allowed_parents[node]
            if self.dijkstra_table[node]['distance'] is None and node == self.settings["topology"]["root"]:
                self.dijkstra_table[node]['distance'] = 0
                self.dijkstra_table[node]['prev'] = None
            for neighbor in neighbors:
                new_distance = self.dijkstra_table[node]['distance'] + self.path_cost # when counting hops, the distance is just + 1 (there is no more cost)
                if self.dijkstra_table[neighbor]['distance'] is None or new_distance < self.dijkstra_table[neighbor]['distance']:
                    self.dijkstra_table[neighbor]['distance'] = new_distance
                    self.dijkstra_table[neighbor]['prev'] = node
            self.unvisited.remove(node)

    def calculate_average(self):
        hopcounts = []
        for node, dict_value in self.dijkstra_table.items():
            if node != self.settings['topology']['root']:
                hopcounts.append(self.dijkstra_table[node]['distance'])
        return numpy.mean(hopcounts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--topology', '-t', type=str)
    parser.add_argument('--loglevel', '-l', type=str)

    args = parser.parse_args()
    settings_file = str(args.input)
    topology_file = str(args.topology)
    loglevel = str(args.loglevel)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=getattr(logging, loglevel.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

    dijkstra = Dijkstra()
    dijkstra.calculate(settings_file=settings_file, topology_file=topology_file)
    print(dijkstra.dijkstra_table)
    print(dijkstra.calculate_average())