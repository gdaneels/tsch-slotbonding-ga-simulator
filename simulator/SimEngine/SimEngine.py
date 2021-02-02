#!/usr/bin/python
"""
\brief Discrete-event simulation engine.

\author Thomas Watteyne <watteyne@eecs.berkeley.edu>
\author Kazushi Muraoka <k-muraoka@eecs.berkeley.edu>
\author Nicola Accettura <nicola.accettura@eecs.berkeley.edu>
\author Xavier Vilajosana <xvilajosana@eecs.berkeley.edu>
"""

#============================ logging =========================================

import random
import logging
import json
class NullHandler(logging.Handler):
    def emit(self, record):
        pass
log = logging.getLogger('SimEngine')
log.setLevel(logging.ERROR)
log.addHandler(NullHandler())

#============================ imports =========================================

import threading

import Propagation
import Topology
import Mote
import SimSettings
import Modulation
import numpy as np
import math
import copy

#============================ defines =========================================

#============================ body ============================================

class SimEngine(threading.Thread):

    #===== start singleton
    _instance      = None
    _init          = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SimEngine,cls).__new__(cls, *args, **kwargs)
        return cls._instance
    #===== end singleton

    def __init__(self, cpuID=None, runNum=None, failIfNotInit=False):

        if failIfNotInit and not self._init:
            raise EnvironmentError('SimEngine singleton not initialized.')

        #===== start singleton
        if self._init:
            return
        self._init = True
        #===== end singleton

        # store params
        self.cpuID                          = cpuID
        self.runNum                         = runNum

        self.rect1 = (0.0, 0.9, 3.0, 1.1)
        self.rect2 = (2.0, 1.9, 5.0, 2.1)
        self.rect3 = (0.0, 2.9, 3.0, 3.1)
        self.rect4 = (2.0, 3.9, 5.0, 4.1)

        # self.originX = 2.5 # in km
        # self.originY = 4.7 # in km
        # self.targetX = 4.5 # in km
        # self.targetY = 4.6 # in km

        # first one is the origin
        self.targets = [(4.6, 4.6), (0.9, 3.95), (4.6, 2.95), (0.9, 1.95), (4.6, 0.95)]
        # self.targets = [(1.0, 3.7), (4.0, 2.7), (1.0, 1.7), (4.5, 0.5)]
        # self.targets = [(4.0, 2.7), (1.0, 1.7), (4.5, 0.5)]
        self.targetType = {}

        self.targetRadius = 0.100 # in km
        self.targetPos = {} # dict: mote -> (x, y) relative to workingTargetX
        self.targetIndex = {}
        self.margin = 0.02
        self.goalDistanceFromTarget = 0.01

        # local variables
        self.dataLock                       = threading.RLock()
        self.pauseSem                       = threading.Semaphore(0)
        self.simPaused                      = False
        self.goOn                           = True
        self.asn                            = 0
        self.startCb                        = []
        self.endCb                          = []
        self.events                         = []
        self.settings                       = SimSettings.SimSettings()
        random.seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        self.genMobility = random.Random()
        self.genMobility.seed(self.settings.seed)
        self.propagation                    = Propagation.Propagation()
        self.motes                          = [Mote.Mote(id) for id in range(self.settings.numMotes)]

        self.ilp_topology                   = None

        # self.comehere = {}

        if self.settings.ilpfile is not None:
            if self.settings.json == None:
                assert False # this file should not be None, because we want to use the same parameters as with we made the ILP file
            if self.settings.ilpschedule == None:
                assert False # this file should not be None, because we want to use the same parameters as with we made the ILP file
            # self.topology = Topology.Topology(self.motes)
            # self.topology.createTopology()

            with open(self.settings.ilpfile, 'r') as f:
                ilp_configuration = json.load(f)
            with open(self.settings.ilpschedule, 'r') as f:
                ilp_schedule = json.load(f)

            ilp_parents = copy.deepcopy(ilp_schedule['parents'])
            ilp_schedule = ilp_schedule['schedule']

            if len(ilp_schedule) != (len(self.motes) - 1):
                assert False

            minimal_modulation = Modulation.Modulation().minimalCellModulation[self.settings.modulationConfig]
            MINIMAL_SLOTS_OFFSET = self.settings.nrMinimalCells * Modulation.Modulation().modulationSlots[self.settings.modulationConfig][minimal_modulation]
            # # add it for all motes, also for the root
            # for (ts, ch, modulation) in ilp_configuration['minimal_slots']:
            #     assert modulation in Modulation.Modulation().modulationSlots[self.settings.modulationConfig]
            #     m._tsch_add_ilp_minimal_cell(ts, ch, modulation)

            # create the topology
            # do this before looping the motes, because we need the topology to add the minimal cells (for the myNeighbors function)
            self.ilp_topology = ilp_configuration['simulationTopology']
            self.topology = Topology.Topology(self.motes)
            self.topology.createTopology()

            for m in self.motes:
                # skip the root
                if m.id != 0:
                    parent_id = ilp_parents[str(m.id)]
                    # set the preferred parent, read from the ILP file
                    m.preferredParent = self.getMote(parent_id)

                    chosenMCS = None
                    if str(m.id) not in ilp_schedule:
                        assert False # the m.id should be in the ILP schedule, if it is not the root
                    sigmaTXList = []
                    sigmaRXList = []
                    for timeslot in ilp_schedule[str(m.id)]:
                        for channel in ilp_schedule[str(m.id)][timeslot]:
                            if chosenMCS is None:
                                chosenMCS = ilp_schedule[str(m.id)][timeslot][channel]['mcs']
                            # print 'mcs %s' % ilp_schedule[str(m.id)][timeslot][channel]['mcs']
                            # print 'slots %s' % ilp_schedule[str(m.id)][timeslot][channel]['slots']
                            # print 'parent %s' % m.preferredParent.id
                            sigmaTXList.append((MINIMAL_SLOTS_OFFSET + int(timeslot), int(channel), Mote.DIR_TX,
                                              int(m.preferredParent.id),
                                              ilp_schedule[str(m.id)][timeslot][channel]['mcs'],
                                              int(ilp_schedule[str(m.id)][timeslot][channel]['slots'])))
                            sigmaRXList.append((MINIMAL_SLOTS_OFFSET + int(timeslot), int(channel), Mote.DIR_RX,
                                                int(m.id),
                                                ilp_schedule[str(m.id)][timeslot][channel]['mcs'],
                                                int(ilp_schedule[str(m.id)][timeslot][channel]['slots'])))


                    assert str(m.id) in ilp_configuration['simulationTopology']
                    assert len(sigmaTXList) == len(sigmaRXList)

                    # if there are cells to add
                    if len(sigmaTXList) == len(sigmaRXList) and len(sigmaTXList) > 0:

                        pdr = self.topology._computePDR(m, m.preferredParent, modulation=chosenMCS)
                        m.setPDR(m.preferredParent, pdr)
                        m.preferredParent.setPDR(m, pdr)
                        m.setModulation(m.preferredParent, chosenMCS)
                        m.preferredParent.setModulation(m, chosenMCS)

                        print 'Adding for mote %d, which is %s' % (m.id, str(m))

                        # add the cells to the parent
                        m._ilp_tsch_addCells(sigmaTXList)
                        m.allocatedBondedSlots = len(sigmaTXList)
                        # add the cells to the child
                        m.preferredParent._ilp_tsch_addCells(sigmaRXList)

                        print '%d : %d' % (m.preferredParent.id, m.preferredParent.numCellsFromNeighbors[m])
                        # print '%d : %d' % (m.id, m.numCellsToNeighbors[m.preferredParent])

                        assert m.numCellsToNeighbors[m.preferredParent] == m.preferredParent.numCellsFromNeighbors[m]
                        # add the minimal cells for each mote

        else:
            # for m in self.motes:
            #     m.setModulation()
            self.topology                       = Topology.Topology(self.motes)
            self.topology.createTopology()

        # only do this if the parents are not dictated by the parent files
        if self.settings.ilpfile is None:
            self.dictParent = self.chooseParentInCircularFashion()
            for ix, mote in enumerate(self.motes):
                if not mote.dagRoot:
                    parent = self.dictParent[mote.id][0]
                    for parentIx, parentMote in enumerate(self.motes):
                        if parentMote.id == parent:
                            break
                    self.motes[ix].preferredParent = self.motes[parentIx]

        if self.settings.genTopology == 1:
            exp_file = None
            with open(self.settings.json) as data_file:
                exp_file = json.load(data_file)
                exp_file = exp_file["simulator"]
            # split off the directory in front and the .json behind
            self.extendJSONWithTopology(exp_file, self.settings.json.split('/')[-1].split('.')[0])
            exit()

        # Not valid values. Will be set by the last mote that converged.
        self.asnInitExperiment = 999999999
        self.asnEndExperiment = 999999999

        # for the ILP
        self.ILPTerminationDelay = 999999999

        self.dedicatedCellConvergence = 99999999

        # self.datarates = [50, 100, 150, 200, 250] # kbps
        # self.datarates = [100, 150, 200, 300]
        # if self.settings.slotAggregation == 1:
        #     self.nrOfSlots = {50: 3, 100: 3, 150: 2, 200: 2, 250: 1}
        # if self.settings.slotAggregation == 2:
        #     self.nrOfSlots = {50: 3, 100: 3, 150: 3, 200: 3, 250: 3}
        # if self.settings.slotAggregation == 3:
        #     self.nrOfSlots = {50: 1, 100: 1, 150: 1, 200: 1, 250: 1}
        # if self.settings.slotAggregation == 4:
        #     self.nrOfSlots = {100: 2, 150: 2, 200: 2, 300: 1}

        # boot all motes at once here when loading an experiment from the ILP file
        # if you not do this, you will have problems because the active cells that are directly loaded from the ILP file will need
        # information from the other nodes that is only available after booting the node
        if self.settings.ilpfile is not None:
            for i in range(len(self.motes)):
                self.motes[i].boot()
                self.motes[i].isConverged = True # set all motes to being converged
                self.motes[i].isConvergedASN = self.asn

            self.dedicatedCellConvergence = self.asn

            # experiment time in ASNs
            simTime = self.settings.numCyclesPerRun * self.settings.slotframeLength
            # offset until the end of the current cycle
            offset = self.settings.slotframeLength - (self.asn % self.settings.slotframeLength)
            settlingTime = int((float(self.settings.settlingTime) / float(self.settings.slotDuration)))
            # experiment time + offset
            self.ILPTerminationDelay = simTime + offset + settlingTime
            self.asnInitExperiment = self.asn + offset + settlingTime
            log.info("Start ILP experiment set at ASN {0}, end experiment at ASN {1}.".format(self.asnInitExperiment, self.asnInitExperiment + simTime))

        if self.settings.ilpfile is None:
            self.motes[0].boot()

        # initialize parent class
        threading.Thread.__init__(self)
        self.name                           = 'SimEngine'

    def destroy(self):
        # destroy the propagation singleton
        self.propagation.destroy()

        # destroy my own instance
        self._instance                      = None
        self._init                          = False

    #======================== thread ==========================================

    def getMote(self, id):
        for m in self.motes:
            if m.id == id:
                return m
        assert False

    def extendJSONWithTopology(self, exp_file, name_exp):
        ''' Write the topology to a file '''
        JSON = {'simulationTopology': {}}
        for m in self.motes:
            links = {}
            reliability = {}
            interferers = []
            for mcs in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]: # only use the modulations in modulationSlots
                reliability[mcs] = {}

            for n in self.motes:
                if m.id != n.id:
                    for mcs in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]: # only use the modulations in modulationSlots
                        pdr = self.topology._computePDR(m, n, modulation=mcs)
                        # print(n.id)
                        # print(m.id)
                        # print(pdr)
                        # print(m.getRSSI(n))
                        # print(mcs)
                        # exit()
                        reliability[mcs][n.id] = pdr
                        # if pdr > 0.0 and n.id not in interferers:
                        #     interferers.append(n.id)
                        if Modulation.Modulation()._dBmTomW(m.getRSSI(n)) > Modulation.Modulation().receiverNoise and n.id not in interferers:
                            interferers.append(n.id)
                    links[n.id] = m.getRSSI(n)

            parent_id = None
            if not m.dagRoot:
                parent_id = m.preferredParent.id

            distance_to_root = math.sqrt((self.motes[0].x - m.x)**2 + (self.motes[0].y - m.y)**2)

            JSON['simulationTopology'][m.id] = {
                'x': m.x,
                'y': m.y,
                'links': links,
                'reliability': reliability,
                'parent': parent_id,
                'interferers': interferers,
                'distance_to_root': distance_to_root,
            }
            JSON['modulations'] = Modulation.Modulation().allowedModulations[self.settings.modulationConfig]
            JSON['modulationSlots'] = Modulation.Modulation().modulationSlots[self.settings.modulationConfig]
            JSON['numChans'] = self.settings.numChans

            # JSON['minimal_slots'] = []
            # for c in range(0, self.settings.nrMinimalCells):
            #     modulation = Modulation.Modulation().minimalCellModulation[self.settings.modulationConfig]
            #     parentTs = c * (Modulation.Modulation().modulationSlots[self.settings.modulationConfig][modulation])
            #     JSON['minimal_slots'].append((parentTs, c, modulation))

            modulation = Modulation.Modulation().minimalCellModulation[self.settings.modulationConfig]
            nrMinimalSlots = self.settings.nrMinimalCells * Modulation.Modulation().modulationSlots[self.settings.modulationConfig][modulation]
            JSON['slotframeLength'] = self.settings.slotframeLength - nrMinimalSlots
            JSON['slotLength'] = self.settings.slotDuration * 1000 # convert it to ms
            # set the omega weight for the ILP
            # JSON['omega'] = self.settings.omega
            # set the gap for the ILP
            # JSON['gap'] = self.settings.gap
            # set the threads used for the ILP
            # JSON['threads'] = self.settings.threads
            # set the time limit used for the ILP
            # JSON['timelimit'] = self.settings.timelimit

        # JSON.update(exp_file)
        name = '{0}/simulator-topology.json'.format(self.settings.topologyPath)
        with open(name, 'w') as the_file:
            the_file.write(json.dumps(JSON))

    # def chooseParent(self):
    #     allMotes = range(1, self.settings.numMotes)  # do not count root
    # 
    #     # dictionary that maps node to preferred parent
    #     dictMotes = {0: (None, None, None)}
    # 
    #     while len(allMotes) > 0:
    #         mote = allMotes[0]
    #         del allMotes[0]
    # 
    #         possibleParents = dictMotes.keys()
    # 
    #         while len(possibleParents) > 0:
    #             randomParentIndex = random.randrange(len(possibleParents))
    #             parent = possibleParents[randomParentIndex]
    #             del possibleParents[randomParentIndex]
    # 
    #             if self.getMote(parent) in self.getMote(mote)._myNeighbors():
    #                 parentDepth = dictMotes[parent][1]
    #                 modulationIndex = Modulation.Modulation().modulations.index(self.getMote(mote).getModulation(self.getMote(parent)))
    # 
    #                 depth = None
    #                 if parentDepth is None:
    #                     depth = 1
    #                 else:
    #                     depth = parentDepth + 1
    # 
    #                 # one should prefer modulations with higher indices as they represent higher data rates
    #                 if mote not in dictMotes or dictMotes[mote][2] < modulationIndex:
    #                     dictMotes[mote] = (parent, depth, modulationIndex)
    # 
    #         print 'Picked %d, parent %d, depth %d, modulationIndex %d' % (mote, dictMotes[mote][0], dictMotes[mote][1], dictMotes[mote][2])
    # 
    #     depths = []
    #     for (m, pTuple) in dictMotes.iteritems():
    #         if pTuple[1] is not None:
    #             depths.append(pTuple[1])
    # 
    #     # print sum(depths) / float(len(depths))
    # 
    #     return dictMotes


    def getMostFarFromRoot(self, motes):
        distance = None
        most_far = None
        for m in motes:
            if distance is None or self.motes[0].get_distance(m) > distance:
                distance = self.motes[0].get_distance(m)
                most_far = m
        return most_far, distance

    def getMotesInRadius(self, mote, radius, motes):
        in_circle = []
        ids = []
        for m in motes:
            if math.sqrt((m.x - mote.x) ** 2 + (m.y - mote.y) ** 2) < radius:
                in_circle.append(m)
                ids.append(m.id)
        # print 'Get motes in radius {2} for mote {0}: {1}'.format(mote.id, ids, radius)
        return in_circle

    def getMotesCloserToRoot(self, mote, motes):
        motesCloserToRoot = []
        ids = []
        for m in motes:
            if math.sqrt((mote.x - self.motes[0].x) ** 2 + (mote.y - self.motes[0].y) ** 2) > math.sqrt((m.x - self.motes[0].x) ** 2 + (m.y - self.motes[0].y) ** 2):
                motesCloserToRoot.append(m)
                ids.append(m.id)
        # print 'Get motes closer to the root for mote {0}: {1}'.format(mote.id, ids)
        return motesCloserToRoot

    def getMotesInNeighbors(self, mote, motes):
        inNeighbours = []
        ids = []
        for m in motes:
            rssi = self.topology._computeRSSI(mote, m)
            if self.topology.rssiToPdr(rssi, modulation=Modulation.Modulation().minimalCellModulation[SimSettings.SimSettings().modulationConfig]) > self.settings.stableNeighborPDR:
            # if rssi > Modulation.Modulation().modulationStableRSSI[Modulation.Modulation().minimalCellModulation[SimSettings.SimSettings().modulationConfig]]:
                inNeighbours.append(m)
                ids.append(m.id)
        # print 'Get motes in stable rssi for mote {0}, for neighbours {1}'.format(mote.id, ids)
        return inNeighbours


    def moteLoops(self, p, dictMotes, original_m):
        # print dictMotes
        # print m
        # print original_m
        # print 'p = %d, original_m = %d' % (p, original_m)
        if p == 0 or (p not in dictMotes):
            # print p
            # print dictMotes
            # print 'Come here...'
            return False
        elif dictMotes[p][0] == original_m: # loop!
            # print 'loop!'
            return True
        else:
            return self.moteLoops(dictMotes[p][0], dictMotes, original_m)

    def filterMotesWithLoops(self, start_mote, motes, dictMotes):
        loopingMotes = []

        for p in motes:
            if self.moteLoops(p.id, dictMotes, start_mote.id):
                loopingMotes.append(p)

        for m in loopingMotes:
            motes.remove(m)

        # print 'Filter motes with loops for mote {0}: {1}'.format(start_mote.id, motes)
        return motes

    def getDepth(self, m, dictMotes, depth = 0):
        if m not in dictMotes:
            assert False

        if dictMotes[m][0] == None: # reached the root
            return depth
        else:
            return self.getDepth(dictMotes[m][0], dictMotes, (depth + 1))

    def chooseParentInCircularFashion(self):
        all_motes = [m for m in self.motes]

        # dictionary that maps node to preferred parent
        dictMotes = {0: (None, None)}

        import collections
        all_motes_to_neighbors = collections.OrderedDict()
        for m in all_motes:
            all_motes_except_me = [mote for mote in self.motes]
            all_motes_except_me.remove(m)
            all_motes_to_neighbors[m] = self.getMotesInNeighbors(m, all_motes_except_me)

        ordered_d_all_motes_to_neighbors = collections.OrderedDict(sorted(all_motes_to_neighbors.items(), key=lambda x: len(x[1]), reverse=False))
        del ordered_d_all_motes_to_neighbors[self.motes[0]]

        ids_ordered_d_all_motes_to_neighbors = collections.OrderedDict()
        for m, neigh in ordered_d_all_motes_to_neighbors.iteritems():
            ids = []
            ids_ordered_d_all_motes_to_neighbors[m] = []
            for mo in neigh:
                ids.append(mo.id)
                ids_ordered_d_all_motes_to_neighbors[m].append(mo.id)
            # print '{0}: {1}'.format(m.id, ids)

        motes = ordered_d_all_motes_to_neighbors.keys()
        mote = motes[0]
        while len(motes) > 0:
            # print 'Mote {0}, stable = {1}'.format(mote.id, ids_ordered_d_all_motes_to_neighbors[mote])
            # print mote.id
            # for m, v in ordered_d_all_motes_to_neighbors.iteritems():
            #     print 'm: {0} : {1}'.format(m.id, m)
            motesCloserToRoot = self.getMotesCloserToRoot(mote, ordered_d_all_motes_to_neighbors[mote])

            go = True
            while go:
                randomMote = None
                if len(motesCloserToRoot) > 0:
                    randomMote = random.choice(motesCloserToRoot)
                    # print 'Chosen mote from closer to root: {0} for mote {1}, seed = {2}'.format(randomMote.id, mote.id, self.settings.seed)
                else:
                    randomMote = random.choice(ordered_d_all_motes_to_neighbors[mote])
                    # print 'Chosen mote of all neighbors: {0} for mote {1}, seed = {2}'.format(randomMote.id, mote.id, self.settings.seed)

                #     print self.moteLoops(randomMote.id, dictMotes, mote.id)
                #     print 'dictMotes {0}'.format(dictMotes)
                #     print 'randomMote {0}: {1}'.format(randomMote, randomMote.id)
                #     print 'mote {0}: {1}'.format(mote, mote.id)
                #     print self.moteLoops(randomMote.id, dictMotes, mote.id)
                if not self.moteLoops(randomMote.id, dictMotes, mote.id):
                    dictMotes[mote.id] = (randomMote.id, None)
                    motes.remove(mote) # remove the mote
                    if randomMote.id == 0 or randomMote.id in dictMotes:
                        if len(motes) > 0: # if the len() == 0, it will quit
                            mote = motes[0] # pick a new mote
                    else:
                        mote = randomMote
                    go = False
                else:
                    go = True
                    if len(motesCloserToRoot) > 0:
                        # if you chose a mote from the closerToRoot motes (len(motesCloserToRoot) > 0), but it looped, remove it
                        motesCloserToRoot.remove(randomMote)

        print dictMotes
        # assert False
        # 
        # assert False
        # 
        # start_mote, distance = self.getMostFarFromRoot(all_motes)
        # 
        # print self.settings.seed
        # 
        # # dictionary that maps node to preferred parent
        # dictMotes = {0: (None, None)}
        # 
        # while start_mote.id != 0:
        #     all_motes.remove(start_mote)
        #     start_radius = 0.5 # start with 300 meters
        # 
        #     no_loop_motes = []
        #     in_neighbors_of_mote = []
        #     motes_in_radius_prev = []
        #     while len(no_loop_motes) == 0:
        #         if start_radius > self.settings.squareSide:
        #             print self.settings.seed
        #             print 'error error error'
        #             assert False
        #         motes_in_radius = [m for m in self.getMotesInRadius(start_mote, start_radius, self.motes) if m not in motes_in_radius_prev]
        #         closer_to_root_motes_in_radius = self.getMotesCloserToRoot(start_mote, motes_in_radius)
        #         in_neighbors_of_mote = self.getMotesInNeighbors(start_mote, closer_to_root_motes_in_radius)
        #         no_loop_motes = self.filterMotesWithLoops(start_mote, in_neighbors_of_mote, dictMotes)
        #         motes_in_radius_prev = list(motes_in_radius)
        #         start_radius += (start_radius * 0.1) # add 10 procent to the distance
        # 
        #     assert len(no_loop_motes) > 0 # at least at some point, the root should be in it
        #     # print 'Length of no_loop_motes is %d' % len(no_loop_motes)
        #     parent = random.choice(no_loop_motes)
        #     # modulationIndex = Modulation.Modulation().modulations.index(start_mote.getModulation(parent))
        #     # dictMotes[start_mote.id] = (parent.id, None, modulationIndex)
        #     dictMotes[start_mote.id] = (parent.id, None)
        #     # all_motes.remove(start_mote)
        # 
        #     # if the picked parent is the root, pick again the most far mote as long as other motes are available than the root
        #     if parent.id == 0 or parent not in all_motes:
        #         start_mote, distance = self.getMostFarFromRoot(all_motes)
        #     else:
        #         start_mote = parent

        for (m, pTuple) in dictMotes.iteritems():
            dictMotes[m] = (pTuple[0], self.getDepth(m, dictMotes, 0))

        print dictMotes
        return dictMotes
        # print 'Choosing parent starts with mote %d with %.4f from root.' % (start_mote, distance)


    # def chooseParentInCircularFashion(self):
    #     all_motes = [m for m in self.motes]
    #
    #     start_mote, distance = self.getMostFarFromRoot(all_motes)
    #
    #     print self.settings.seed
    #
    #     # dictionary that maps node to preferred parent
    #     dictMotes = {0: (None, None)}
    #
    #     while start_mote.id != 0:
    #         all_motes.remove(start_mote)
    #         start_radius = 0.5 # start with 300 meters
    #
    #         no_loop_motes = []
    #         in_neighbors_of_mote = []
    #         motes_in_radius_prev = []
    #         while len(no_loop_motes) == 0:
    #             # if start_radius > self.settings.squareSide:
    #             #     print self.settings.seed
    #             #     print 'error error error'
    #             #     assert False
    #             motes_in_radius = [m for m in self.getMotesInRadius(start_mote, start_radius, self.motes) if m not in motes_in_radius_prev]
    #             closer_to_root_motes_in_radius = self.getMotesCloserToRoot(start_mote, motes_in_radius)
    #             in_neighbors_of_mote = self.getMotesInNeighbors(start_mote, closer_to_root_motes_in_radius)
    #             no_loop_motes = self.filterMotesWithLoops(start_mote, in_neighbors_of_mote, dictMotes)
    #             motes_in_radius_prev = list(motes_in_radius)
    #             start_radius += (start_radius * 0.1) # add 10 procent to the distance
    #
    #         assert len(no_loop_motes) > 0 # at least at some point, the root should be in it
    #         # print 'Length of no_loop_motes is %d' % len(no_loop_motes)
    #         parent = random.choice(no_loop_motes)
    #         # modulationIndex = Modulation.Modulation().modulations.index(start_mote.getModulation(parent))
    #         # dictMotes[start_mote.id] = (parent.id, None, modulationIndex)
    #         dictMotes[start_mote.id] = (parent.id, None)
    #         # all_motes.remove(start_mote)
    #
    #         # if the picked parent is the root, pick again the most far mote as long as other motes are available than the root
    #         if parent.id == 0 or parent not in all_motes:
    #             start_mote, distance = self.getMostFarFromRoot(all_motes)
    #         else:
    #             start_mote = parent
    #
    #     for (m, pTuple) in dictMotes.iteritems():
    #         dictMotes[m] = (pTuple[0], self.getDepth(m, dictMotes, 0))
    #
    #     print dictMotes
    #     return dictMotes
    #     # print 'Choosing parent starts with mote %d with %.4f from root.' % (start_mote, distance)

    def getRotatedDegrees(self, omega, delta_x, delta_y):
        if delta_x >= 0.0 and delta_y >= 0.0:
            # in this scenario, add 90
            return 90.0 + omega
        elif delta_x < 0.0 and delta_y >= 0.0:
            # in this scenario, add 180
            return 90.0 - omega + 180.0
        elif delta_x < 0.0 and delta_y < 0.0:
            # in this scenario, add 270
            return 90.0 + omega + 180.0
        elif delta_x >= 0.0 and delta_y < 0.0:
            # in this scenario, add 0
            return 90.0 - omega


    # def chooseParentHalfRadius(self):
    #     allMotes = range(1, self.settings.numMotes)  # do not count root
    #     todoMotes = range(1, self.settings.numMotes)  # do not count root
    #
    #     # dictionary that maps node to preferred parent
    #     dictMotes = {0: (None, None, None)}
    #
    #     rootX = self.motes[0].x
    #     rootY = self.motes[0].y
    #
    #     # only works when root equals (0,0), otherwise I have to calculate the delta's differently
    #     assert rootX == 0.0
    #     assert rootY == 0.0
    #
    #     while len(allMotes) > 0:
    #         mote = todoMotes[0]
    #         del todoMotes[0]
    #
    #         delta_x = mote.x - rootX
    #         delta_y = mote.y - rootY
    #         # get angle
    #         omega_radians = math.atan2(abs(delta_y), abs(delta_x))
    #         # to degrees
    #         omega_degrees = omega_radians * (180.0 / math.pi)
    #         omega_degrees = self.getRotatedDegrees(omega_degrees, delta_x, delta_y)
    #         # to radians
    #         omega_radians = omega_degrees * math.pi / 180.0
    #
    #         # make a line
    #         # calculate the a coefficient of y = ax + b
    #         a = math.tan(omega_radians)
    #         # calculate b of y = ax + b
    #         b = mote.y - a * mote.x
    #
    #         # this is a list of ALL the possible parents
    #         possibleParents = todoMotes
    #         # this is list of all the parents that were evaluated to have a good position
    #         goodPositionedParents = []
    #
    #         # calculate radius from the mote to the root
    #         radius = math.sqrt((rootY - mote.y) ** 2 + (rootX - mote.x) ** 2)
    #         radius += 0.01 # add 50 meters so the root is for sure included
    #         for p in possibleParents:
    #             # the parent should be at the good side of the line
    #             if (rootY < mote.y and p.y < (a * p.x + b)) or (rootY > mote.y and p.y > (a * p.x + b)):
    #                 # however, it should not cross the circle radius
    #                 if math.sqrt((p.y - mote.y) ** 2 + (p.x - mote.x) ** 2) <= radius:
    #                     goodPositionedParents.append(p)
    #
    #         while len(goodPositionedParents) > 0:
    #             randomParentIndex = random.randrange(len(possibleParents))
    #             parent = possibleParents[randomParentIndex]
    #             del possibleParents[randomParentIndex]
    #
    #             if self.getMote(parent) in self.getMote(mote)._myNeighbors():
    #                 modulationIndex = Modulation.Modulation().modulations.index(self.getMote(mote).getModulation(self.getMote(parent)))
    #
    #                 # one should prefer modulations with higher indices as they represent higher data rates
    #                 if mote not in dictMotes:
    #                     dictMotes[mote] = (parent, None, modulationIndex)
    #
    #         print 'Picked %d, parent %d, depth %d, modulationIndex %d' % (mote, dictMotes[mote][0], dictMotes[mote][1], dictMotes[mote][2])
    #
    #     depths = []
    #     for (m, pTuple) in dictMotes.iteritems():
    #         if pTuple[1] is not None:
    #             depths.append(pTuple[1])
    #
    #     # print sum(depths) / float(len(depths))
    #
    #     return dictMotes

    def getTrafficPeriod(self):
        pick = -1.0
        trafficAverage = self.settings.pkPeriod
        trafficStd = trafficAverage / 4
        while pick <= 0.0:
            pick = np.random.normal(trafficAverage, trafficStd, None)
            if pick < trafficAverage:
                pick = math.ceil(pick)
            else:
                pick = math.floor(pick)
        return pick

    def run(self):
        """ event driven simulator, this thread manages the events """

        # log
        log.info("thread {0} starting".format(self.name))

        # schedule the endOfSimulation event if we are not simulating the join process
        if not self.settings.withJoin:
            if not self.settings.convergeFirst:
                self.scheduleAtAsn(
                    asn         = self.settings.slotframeLength*self.settings.numCyclesPerRun,
                    cb          = self._actionEndSim,
                    uniqueTag   = (None,'_actionEndSim'),
                )
            else:
                self.scheduleAtAsn(
                    asn         = self.settings.maxToConverge/self.settings.slotDuration,
                    cb          = self._actionEndSim,
                    uniqueTag   = (None,'_actionEndSim'),
                )

        # set this here, if you do it earlier it is overwritten by the above statements
        if self.settings.ilpfile is not None:
            self.terminateSimulation(self.ILPTerminationDelay)

        if self.settings.trafficGenerator == 'pick':
            periods = None
            # if self.settings.trafficFrequency == 'short':
            #     # periods = [200, 400, 600] # 3s, 6s, 9s
            #     periods = [150, 200, 250]  # 3s, 6s, 9s
            # elif self.settings.trafficFrequency == 'medium':
            #     # periods = [2000, 3000, 4000] # 30s, 45s, 60s
            #     periods = [1500, 2000, 2500]  # 3s, 6s, 9s
            # elif self.settings.trafficFrequency == 'long':
            #     periods = [20000, 30000, 40000] # 5 min (300s), 7.5 min (450s), 10 min (600s)

            # if self.settings.trafficFrequency == '10s':
            #     periods = [10]  # 3s, 6s, 9s
            # elif self.settings.trafficFrequency == '20s':
            #     # periods = [2000, 3000, 4000] # 30s, 45s, 60s
            #     periods = [20]  # 3s, 6s, 9s
            # elif self.settings.trafficFrequency == '30s':
            #     periods = [30] # 5 min (300s), 7.5 min (450s), 10 min (600s)
            # elif self.settings.trafficFrequency == '60s':
            #     periods = [60] # 5 min (300s), 7.5 min (450s), 10 min (600s)

            # periods = [200, 500, 1000, 3000, 6000, 12000]
            # periods = [200, 400, 600, 800, 1000]
            periods = [200]
            # slotduration = 0.015, [3.0, 7.5, 15.0, 45.0, 90.0, 180.0] seconds
            # slotduration = 0.010, [2.0, 5.0, 10.0, 30.0, 60.0, 120.0] seconds
            for m in self.motes:
                if m.id > 0:
                    m.startApp = random.randint(0, 60)
                    m.pkPeriod = periods[random.randint(0, len(periods)-1)]
                    log.info("Mote {0}, theoretical start delay = {1}, period = {2} s.".format(m.id, m.startApp, m.pkPeriod))

            sporadics = [4000, 6000, 8000]
            for m in self.motes:
                if m.id > 0:
                    m.sporadic = sporadics[random.randint(0, len(sporadics)-1)] * float(self.settings.slotDuration)
                    m.sporadicStart = random.randint(0, 2000)
                    log.info("Mote {0}, sporadic sending first = {1}.".format(m.id, m.sporadic))
        elif self.settings.trafficGenerator == 'normal':
            for m in self.motes:
                if m.id > 0:
                    m.startApp = random.randint(0, 6000)
                    m.pkPeriod = self.getTrafficPeriod()
                    log.info("Mote {0},  theoretical start delay = {1}, period = {2}.".format(m.id, m.startApp, m.pkPeriod))
        elif self.settings.trafficGenerator == 'ilp':
            if self.settings.ilpfile is not None:
                for m in self.motes:
                    if m.id > 0:
                        m.startApp = 0
                        # I wan't this to be zero, so all motes send an equal amount of packets
                        # For the ILP you will not risk of oversaturating the network at the same time, because all the reservations are already done
                        m.pkPeriod = self.settings.slotframeLength * self.settings.slotDuration
                        # the period of sending a packet should be every slotframe
                        log.info("Mote {0}, theoretical start delay = {1}, period = {2} s.".format(m.id, m.startApp, m.pkPeriod))
            else:
                assert False # should absolutely be false
        else:
            assert False

        if self.settings.ilpfile is not None:
            self.startSending() # for the ILP you can start sending from the moment you start the experiment


        if self.settings.mobilityModel == 'RPGM':
            for m in range(0, self.settings.numMotes):
                self.targetPos[m] = (
                self.targets[1][0] + self.genMobility.uniform((-1.0 * self.targetRadius) + 0.005, self.targetRadius - 0.005),
                self.targets[1][1] + self.genMobility.uniform((-1.0 * self.targetRadius) + 0.005, self.targetRadius - 0.005))
                self.targetIndex[m] = 1
                self.targetType[m] = 'up'

        # reset the states per modulation
        for m in self.motes:
            for modulation in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]:
                for state, state_dict in m.consumption.iteritems():
                    m.consumption[state][modulation] = 0
                    m.totalConsumption[state][modulation] = 0

        # call the start callbacks
        for cb in self.startCb:
            cb()

        # for m in self.motes:
        #     log.info(
        #         "[topology] shortest mote to {0} is {1}.".format(m.id, m.closestNeighbor.id),
        #     )

        # consume events until self.goOn is False
        while self.goOn:

            with self.dataLock:

                # abort simulation when no more events
                if not self.events:
                    log.info("end of simulation at ASN={0}".format(self.asn))
                    break

                # make sure we are in the future
                (a, b, cb, c) = self.events[0]
                if c[1] != '_actionPauseSim':
                    assert self.events[0][0] >= self.asn

                # update the current ASN
                self.asn = self.events[0][0]

                if self.settings.ilpfile is None:
                    interval = 15
                    newCycle = int(self.getAsn() / self.settings.slotframeLength)
                    index = newCycle / interval
                    if newCycle % interval == 0 and index < len(self.motes) and self.motes[index].isJoined == False:
                        self.motes[index].boot()
                        self.motes[index]._msf_schedule_parent_change()
                        log.info("Booting node {0}".format(index))

                if self.asn % self.settings.slotframeLength == 0:
                    # rdm = self.propagation.print_random()
                    # log.info("topology random={0}".format(rdm))
                    log.info("[6top] ----------- SLOTFRAME BEGIN at ASN %s -----------" % str(self.asn))

                # only start moving when the experiment started, there is a mobility model and do it at the beginning of every cycle
                if self.asn > self.asnInitExperiment and self.settings.mobilityModel != 'none' and self.asn % self.settings.slotframeLength == 0:
                    if self.settings.mobilityModel == 'RWM': # random walk model
                        for m in self.motes:
                            if m.id != 0:
                                m.updateLocation()
                    elif self.settings.mobilityModel == 'RPGM':
                        for m in self.motes:
                            m.updateLocation()
                    self.topology.updateTopology()
                    for m in self.motes:
                        m._tsch_updateMinimalCells() # update the neighbors of the minimal cells

                # call callbacks at this ASN
                while True:
                    if self.events[0][0]!=self.asn:
                        break
                    (_,_,cb,_) = self.events.pop(0)
                    cb()

        # call the end callbacks
        for cb in self.endCb:
            cb()
        #
        # print 'Come here!'
        # print self.comehere

        # log
        log.info("thread {0} ends".format(self.name))

    #======================== public ==========================================

    # called when there is dedicated cell
    def startSending(self):
        # offset until the end of the current cycle
        offset = self.settings.slotframeLength - (self.asn % self.settings.slotframeLength)
        for m in self.motes:
            if m.id > 0:
                delay = ((offset * float(self.settings.slotDuration)) + m.startApp)
                delayASN = delay / self.settings.slotDuration
                log.info("Mote {0}, will start at ASN {1}, in {3} seconds, period of {2}.".format(m.id, self.getAsn()+delayASN, m.pkPeriod, delay))
                # delay *= float(self.settings.slotDuration)
                # schedule the transmission of the first packet
                self.scheduleIn(
                    delay=delay,
                    cb=m._app_action_sendSinglePacket,
                    uniqueTag=(m.id, '_app_action_sendSinglePacket'),
                    priority=2,
                )
                if self.settings.sporadicTraffic == 1:
                    self.scheduleIn(
                        delay=delay + (m.sporadicStart*float(self.settings.slotDuration)),
                        cb=m._app_action_sendSporadicPacket,
                        uniqueTag=(m.id, '_app_action_sendSporadicPacket'),
                        priority=2,
                    )


    def checkValidPosition(self, xcoord, ycoord, countSquare=True, placement=False):
        '''
        Checks if a given postition is valid when moving
        '''

        margin = self.margin
        if placement:
            margin = 0.02

        inSquare = False  # total area
        insideObstacle1 = False  # rectangle 1
        insideObstacle2 = False  # rectangle 2
        insideObstacle3 = False  # rectangle 1
        insideObstacle4 = False  # rectangle 2
        if countSquare:
            if (xcoord < self.settings.squareSide and ycoord < self.settings.squareSide) and (
                    xcoord > 0 and ycoord > 0):
                inSquare = True
        else:
            inSquare = True

        if (xcoord < (self.rect1[2] + margin)) and (ycoord > (self.rect1[1] - margin) and (ycoord < (self.rect1[3] + margin))):
            insideObstacle1 = True
        if (xcoord > (self.rect2[0] - margin)) and (ycoord > (self.rect2[1] - margin) and (ycoord < (self.rect2[3] + margin))):
            insideObstacle2 = True
        if (xcoord < (self.rect3[2] + margin)) and (ycoord > (self.rect3[1] - margin) and (ycoord < (self.rect3[3] + margin))):
            insideObstacle3 = True
        if (xcoord > (self.rect4[0] - margin)) and (ycoord > (self.rect4[1] - margin) and (ycoord < (self.rect4[3] + margin))):
            insideObstacle4 = True

        if inSquare and not insideObstacle1 and not insideObstacle2 and not insideObstacle3 and not insideObstacle4:
            return True
        else:
            return False

    #=== scheduling

    def scheduleAtStart(self,cb):
        with self.dataLock:
            self.startCb    += [cb]

    def scheduleIn(self,delay,cb,uniqueTag=None,priority=0,exceptCurrentASN=True):
        """ used to generate events. Puts an event to the queue """

        with self.dataLock:
            asn = int(self.asn+(float(delay)/float(self.settings.slotDuration)))

            self.scheduleAtAsn(asn,cb,uniqueTag,priority,exceptCurrentASN)

    def scheduleAtAsn(self,asn,cb,uniqueTag=None,priority=0,exceptCurrentASN=True):
        """ schedule an event at specific ASN """

        # make sure we are scheduling in the future
        assert asn>self.asn

        # remove all events with same uniqueTag (the event will be rescheduled)
        if uniqueTag:
            self.removeEvent(uniqueTag,exceptCurrentASN)

        with self.dataLock:

            # find correct index in schedule
            i = 0
            while i<len(self.events) and (self.events[i][0]<asn or (self.events[i][0]==asn and self.events[i][1]<=priority)):
                i +=1

            # add to schedule
            self.events.insert(i,(asn,priority,cb,uniqueTag))

    def removeEvent(self,uniqueTag,exceptCurrentASN=True):
        with self.dataLock:
            i = 0
            while i<len(self.events):
                if self.events[i][3]==uniqueTag and not (exceptCurrentASN and self.events[i][0]==self.asn):
                    self.events.pop(i)
                    if uniqueTag[0] == 3 and uniqueTag[1] == '_msf_action_parent_change_retransmission':
                        self.motes[3]._log(
                            Mote.INFO,
                            '[6top] Actual retransmission event is being removed...',
                        )
                    if uniqueTag[0] == 3 and uniqueTag[1] == '_msf_action_parent_change_removal':
                        self.motes[3]._log(
                            Mote.INFO,
                            '[6top] Actual removal event is being removed...',
                        )
                else:
                    i += 1

    def scheduleAtEnd(self,cb):
        with self.dataLock:
            self.endCb      += [cb]

    # === misc

    #delay in asn
    def terminateSimulation(self,delay):
        self.asnEndExperiment = self.asn + delay
        self.scheduleAtAsn(
                asn         = self.asn+delay + self.settings.cooldown,
                cb          = self._actionEndSim,
                uniqueTag   = (None,'_actionEndSim'),
        )

    #=== play/pause

    def play(self):
        self._actionResumeSim()

    def pauseAtAsn(self,asn):
        if not self.simPaused:
            self.scheduleAtAsn(
                asn         = asn,
                cb          = self._actionPauseSim,
                uniqueTag   = ('SimEngine','_actionPauseSim'),
            )

    #=== getters/setters

    def getAsn(self):
        return self.asn

    #======================== private =========================================

    def _actionPauseSim(self):
        if not self.simPaused:
            self.simPaused = True
            self.pauseSem.acquire()

    def _actionResumeSim(self):
        if self.simPaused:
            self.simPaused = False
            self.pauseSem.release()

    def _actionEndSim(self):
        with self.dataLock:
            self.goOn = False
