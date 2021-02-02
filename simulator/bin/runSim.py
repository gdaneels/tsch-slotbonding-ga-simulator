#!/usr/bin/python
"""
\brief Entry point to the simulator. Starts a batch of simulations concurrently.
\author Thomas Watteyne <watteyne@eecs.berkeley.edu>
\author Malisa Vucinic <malishav@gmail.com>
"""

#============================ adjust path =====================================

import os
import sys
import json

if __name__=='__main__':
    here = sys.path[0]
    sys.path.insert(0, os.path.join(here, '..'))

#============================ logging =========================================

import logging

class NullHandler(logging.Handler):
    def emit(self, record):
        pass
log = logging.getLogger('runSim')
log.setLevel(logging.ERROR)
log.addHandler(NullHandler())

#============================ imports =========================================

import time
import itertools
import logging.config
import threading
import math
import multiprocessing
import argparse

from SimEngine     import SimEngine,   \
                          SimSettings, \
                          SimStats
from SimGui        import SimGui

#============================ helpers =========================================

def parseCliOptions():

    parser = argparse.ArgumentParser()

    # sim
    parser.add_argument('--gui',
                      dest='gui',
                      action='store_true',
                      default=False,
                      help='[sim] Display the GUI.',
                      )
    parser.add_argument('--numCores',
                      dest='numCores',
                      type=int,
                      default=1,
                      help='[sim] Number of CPU cores to use to parallelize the simulation. Pass -1 to run on all available cores.',
                      )
    parser.add_argument('--numRuns',
                      dest='numRuns',
                      type=int,
                      default=1,
                      help='[sim] Minimum number of simulation runs. Parallelized over NUMCORES CPU cores.',
                      )
    parser.add_argument('--numCyclesPerRun',
                      dest='numCyclesPerRun',
                      type=int,
                      default=101,
                      help='[simulation] Duration of a run, in slotframes.',
                      )
    parser.add_argument('--simDataDir',
                      dest='simDataDir',
                      type=str,
                      default='simData',
                      help='[simulation] Simulation log directory.',
                      )
    # topology
    parser.add_argument('--topology',
                      dest='topology',
                      type=str,
                      choices=['random', 'linear', 'grid'],
                      default='grid',
                      help='[topology] Specify a topology creator to be used',
                      )
    parser.add_argument('--numMotes',
                      dest='numMotes',
                      nargs='+',
                      type=int,
                      default=[10],
                      help='[topology] Number of simulated motes.',
                      )
    parser.add_argument('--squareSide',
                      dest='squareSide',
                      type=float,
                      default=5.0,
                      help='[topology] Side of the deployment area (km).',
                      )
    parser.add_argument('--fullyMeshed',
                      dest='fullyMeshed',
                      nargs='+',
                      type=int,
                      default=0,
                      help=' [topology] 1 to enable fully meshed network.',
                      )
    # join process
    parser.add_argument('--withJoin',
                      dest='withJoin',
                      nargs='+',
                      type=int,
                      default=0,
                      help=' [join process] 1 to enable join process.',
                      )
    parser.add_argument('--joinNumExchanges',
                      dest='joinNumExchanges',
                      nargs='+',
                      type=int,
                      default=2,
                      help='[join process] Number of exchanges needed to complete join process.',
                      )
    parser.add_argument('--joinAttemptTimeout',
                      dest='joinAttemptTimeout',
                      type=float,
                      default=60.0,
                      help='[join process] Timeout to attempt join process (s).',
                      )
    # app
    parser.add_argument('--pkPeriod',
                      dest='pkPeriod',
                      nargs='+',
                      type=float,
                      default=10,
                      help='[app] Average period between two data packets (s).',
                      )
    parser.add_argument('--pkPeriodVar',
                      dest='pkPeriodVar',
                      type=float,
                      default=0.05,
                      help='[app] Variability of pkPeriod (0.00-1.00).',
                      )
    parser.add_argument('--burstTimestamp',
                      dest='burstTimestamp',
                      nargs='+',
                      type=float,
                      default=None,
                      help='[app] Timestamp when the burst happens (s).',
                      )
    parser.add_argument('--numPacketsBurst',
                      dest='numPacketsBurst',
                      nargs='+',
                      type=int,
                      default=None,
                      help='[app] Number of packets in a burst, per node.',
                      )
    parser.add_argument('--downwardAcks',
                      dest='downwardAcks',
                      nargs='+',
                      type=int,
                      default=0,
                      help='[app] 1 to enable downward end-to-end ACKs.',
                      )
    parser.add_argument('--numFragments',
                      dest='numFragments',
                      nargs='+',
                      type=int,
                      default=None,
                      help='''[app] Number of fragments for an app packet;
                            a number less than 2 disables fragmentation'''
                      )
    parser.add_argument('--enableFragmentForwarding',
                      dest='enableFragmentForwarding',
                      type=bool,
                      default=False,
                      help='[app] Enable Fragment Forwarding feature'
                      )
    parser.add_argument('--optFragmentForwarding',
                      dest='optFragmentForwarding',
                      nargs='+',
                      type=str,
                      default=None,
                      choices=['kill_entry_by_last', 'kill_entry_by_missing'],
                      help='[app] Specify Fragment Forwarding options',
                      )
    parser.add_argument('--maxVRBEntryNum',
                      dest='enableFragmentForwarding',
                      type=int,
                      help='[app] Maximum number of entries VRBTable can have'
                      )
    parser.add_argument('--numReassQueue',
                      dest='numReassQueue',
                      type=int,
                      default=1,
                      help='[app] Number of reassembly queues; one per packet'
                      )
    # rpl
    parser.add_argument('--dioPeriod',
                      dest='dioPeriod',
                      type=float,
                      default=10.0,
                      help='[rpl] DIO period (s).',
                      )
    parser.add_argument('--daoPeriod',
                      dest='daoPeriod',
                      type=float,
                      default=90.0,
                      help='[rpl] DAO period (s).',
                      )
    # msf
    parser.add_argument('--disableMSF',
                      dest='disableMSF',
                      action='store_true',
                      default=False,
                      help='[msf] Disable MSF.',
                      )
    parser.add_argument('--msfHousekeepingPeriod',
                      dest='msfHousekeepingPeriod',
                      type=float,
                      default=60.0,
                      help='[msf] MSF HOUSEKEEPINGCOLLISION_PERIOD parameter (s).',
                      )
    parser.add_argument('--msfMaxNumCells',
                      dest='msfMaxNumCells',
                      nargs='+',
                      type=int,
                      default=16,
                      help='[msf] MSF MAX_NUMCELLS parameter.',
                      )
    parser.add_argument('--msfLimNumCellsUsedHIGH',
                      dest='msfLimNumCellsUsedHigh',
                      nargs='+',
                      type=int,
                      default=12,
                      help='[msf] MSF LIM_NUMCELLSUSED_HIGH parameter.',
                      )
    parser.add_argument('--msfLimNumCellsUsedLOW',
                      dest='msfLimNumCellsUsedLow',
                      nargs='+',
                      type=int,
                      default=4,
                      help='[msf] MSF LIM_NUMCELLSUSED_LOW parameter.',
                      )
    parser.add_argument('--msfNumCellsToAddOrRemove',
                      dest='msfNumCellsToAddOrRemove',
                      nargs='+',
                      type=int,
                      default=1,
                      help='[msf] MSF number of cells to add/remove when 6P is triggered.',
                      )
    # sixtop
    parser.add_argument('--sixtopMessaging',
                      dest='sixtopMessaging',
                      type=int,
                      default=1,
                      help='[6top] 1 to enable 6top messaging, 0 to enable 6top GOD mode.',
                      )
    parser.add_argument('--sixtopRemoveRandomCell',
                      dest='sixtopRemoveRandomCell',
                      nargs='+',
                      type=int,
                      default=0,
                      help='[6top] 1 to remove random cells, 0 to remove worst cells by PDR.',
                      )
    # tsch
    parser.add_argument('--slotDuration',
                      dest='slotDuration',
                      type=float,
                      default=0.020,
                      help='[tsch] Duration of a timeslot (s).',
                      )
    parser.add_argument('--slotframeLength',
                      dest='slotframeLength',
                      nargs='+',
                      type=int,
                      default=101,
                      help='[tsch] Number of timeslots in a slotframe.',
                      )
    parser.add_argument('--beaconPeriod',
                      dest='beaconPeriod',
                      nargs='+',
                      type=float,
                      default=5.0,
                      help='[tsch] Enhanced Beacon period (s).',
                      )
    # Bayesian broadcast algorithm
    parser.add_argument('--bayesianBroadcast',
                      dest='bayesianBroadcast',
                      type=int,
                      default=0,
                      help='[tsch] Enable Bayesian broadcast algorithm.',
                      )
    parser.add_argument('--beaconProbability',
                      dest='beaconProbability',
                      nargs='+',
                      type=float,
                      default=0.33,
                      help='[tsch] Beacon probability with Bayesian broadcast algorithm.',
                      )
    parser.add_argument('--dioProbability',
                      dest='dioProbability',
                      nargs='+',
                      type=float,
                      default=0.33,
                      help='[tsch] DIO probability with Bayesian broadcast algorithm.',
                      )
    # phy
    parser.add_argument('--numChans',
                      dest='numChans',
                      type=int,
                      default=2,
                      help='[phy] Number of frequency channels.',
                      )
    parser.add_argument('--minRssi',
                      dest='minRssi',
                      type=int,
                      default=-97,
                      help='[phy] Mininum RSSI with positive PDR (dBm).',
                      )
    parser.add_argument('--noInterference',
                      dest='noInterference',
                      nargs='+',
                      type=int,
                      default=0,
                      help='[phy] Disable interference model.',
                      )
    # linear-topology specific
    parser.add_argument('--linearTopologyStaticScheduling',
                      dest='linearTopologyStaticScheduling',
                      type=bool,
                      default=False,
                      help='[topology] Enable a static scheduling in LinearTopology',
                      )
    parser.add_argument('--schedulingMode',
                      dest='schedulingMode',
                      type=str,
                      choices=['static', 'random-pick'],
                      default=None,
                      help='[topology] Specify scheduling mode',
                      )
    parser.add_argument('--seed',
                        dest='seed',
                        nargs='+',
                        type=int,
                        default=0,
                        help='Random seed.',
                        )
    parser.add_argument('--sf',
                        dest='sf',
                        type=str,
                        default='msf',
                        help='Scheduling function [msf/ellsf].',
                        )
    parser.add_argument('--ellsfMode',
                        dest='ellsfMode',
                        type=str,
                        default='all',
                        help='Scheduling function [resticted/halfrestricted/all].',
                        )
    parser.add_argument('--convergeFirst',
                        dest='convergeFirst',
                        type=int,
                        default=1,
                        help='Only start the experiment after convergence of the network.',
                        )
    parser.add_argument('--maxToConverge',
                        dest='maxToConverge',
                        type=int,
                        default=3000,
                        help='Terminate the experiment if the network has not converged (if convergeFirst option is enabled).',
                        )
    parser.add_argument('--json',
                        dest='json',
                        type=str,
                        default=None,
                        help='Location of JSON experiment file.',
                        )

    parser.add_argument('--settlingTime',
                        dest='settlingTime',
                        type=int,
                        default=1000,
                        help='Number of seconds (s) the network gets to settle, before the actual experiments starts.',
                        )
    parser.add_argument('--nrMinimalCells',
                        dest='nrMinimalCells',
                        type=int,
                        default=7,
                        help='Number of minimal cells.',
                        )
    parser.add_argument('--trafficGenerator',
                        dest='trafficGenerator',
                        type=str,
                        default='normal',
                        help='Type traffic generator [normal, pick].',
                        )
    parser.add_argument('--changeParent',
                        dest='changeParent',
                        type=int,
                        default=1,
                        help='Allow changing parents.',
                        )
    parser.add_argument('--mobilityModel',
                        dest='mobilityModel',
                        type=str,
                        default='none',
                        help='The mobility model [none, RWM, RPGM].',
                        )
    parser.add_argument('--mobilitySpeed',
                        dest='mobilitySpeed',
                        type=int,
                        default=8,
                        help='The mobility speed in meters/second.',
                        )
    parser.add_argument('--backoffMaxExp',
                        dest='backoffMaxExp',
                        type=int,
                        default=1,
                        help='The maximum back off exponent.',
                        )
    parser.add_argument('--backoffMinExp',
                        dest='backoffMinExp',
                        type=int,
                        default=1,
                        help='The maximum back off exponent.',
                        )
    parser.add_argument('--minCellsMSF',
                        dest='minCellsMSF',
                        type=int,
                        default=1,
                        help='The default minimum cells in MSF.',
                        )
    # parser.add_argument('--trafficFrequency',
    #                     dest='trafficFrequency',
    #                     type=str,
    #                     default='long',
    #                     help='Different types of possible traffic types [short/medium/long].',
    #                     )
    parser.add_argument('--sporadicTraffic',
                        dest='sporadicTraffic',
                        type=int,
                        default=0,
                        help='Introduce sporadic traffic or not.',
                        )
    # parser.add_argument('--slotAggregation',
    #                     dest='slotAggregation',
    #                     type=int,
    #                     default=0,
    #                     help='Apply slot aggregation.',
    #                     )
    parser.add_argument('--subGHz',
                        dest='subGHz',
                        type=int,
                        default=0,
                        help='Enable the subGHz model.',
                        )
    parser.add_argument('--individualModulations',
                        dest='individualModulations',
                        type=int,
                        default=0,
                        help='There can be individual modulations.',
                        )
    parser.add_argument('--packetSize',
                        dest='packetSize',
                        type=int,
                        default=127,
                        help='The packet size in bytes (including CRC).',
                        )
    parser.add_argument('--acknowledgementSize',
                        dest='acknowledgementSize',
                        type=int,
                        default=27,
                        help='The packet size in bytes (including CRC).',
                        )
    parser.add_argument('--noiseFigure',
                        dest='noiseFigure',
                        type=float,
                        default=4.5,
                        help='Noise figure of the chip.',
                        )
    parser.add_argument('--channelBandwidth',
                        dest='channelBandwidth',
                        type=float,
                        default=156.0,
                        help='Bandwidth of one channel.',
                        )
    parser.add_argument('--gridDistance',
                        dest='gridDistance',
                        type=float,
                        default=0.1,
                        help='Mean grid distance (km) between nodes.',
                        )
    parser.add_argument('--modulationConfig',
                        dest='modulationConfig',
                        type=str,
                        default='default',
                        help='Set the modulation config you want.',
                        )
    parser.add_argument('--modulationFile',
                        dest='modulationFile',
                        type=str,
                        default=None,
                        help='Set the modulation file you want.',
                        )
    parser.add_argument('--noNewInterference',
                        dest='noNewInterference',
                        type=int,
                        default=1,
                        help='Do not disable the new interference model.',
                        )
    parser.add_argument('--noPropagationLoss',
                        dest='noPropagationLoss',
                        type=int,
                        default=1,
                        help='Do not disable the new propagation loss model.',
                        )
    parser.add_argument('--genTopology',
                        dest='genTopology',
                        type=int,
                        default=0,
                        help='Generate a topology file.',
                        )
    parser.add_argument('--topologyPath',
                        dest='topologyPath',
                        type=str,
                        default='./',
                        help='Path where the topology file should be saved.',
                        )
    parser.add_argument('--ilpfile',
                        dest='ilpfile',
                        type=str,
                        default=None,
                        help='The input for the MILP to feed to the simulator.',
                        )
    parser.add_argument('--ilpschedule',
                        dest='ilpschedule',
                        type=str,
                        default=None,
                        help='The schedule output of the MILP to feed to the simulator.',
                        )

    parser.add_argument('--omega',
                        dest='omega',
                        type=float,
                        default=0.0,
                        help='The omega weight for the ILP.',
                        )
    parser.add_argument('--gap',
                        dest='gap',
                        type=float,
                        default=0.0,
                        help='The gap for the ILP.',
                        )
    parser.add_argument('--threads',
                        dest='threads',
                        type=int,
                        default=1,
                        help='The number of threads used for the ILP.',
                        )
    parser.add_argument('--timelimit',
                        dest='timelimit',
                        type=int,
                        default=0,
                        help='The number of seconds before the ILP stops.',
                        )
    parser.add_argument('--cooldown',
                        dest='cooldown',
                        type=int,
                        default=200,
                        help='Number of ASNs to cooldown.',
                        )
    parser.add_argument('--stableNeighbors',
                        dest='stableNeighbors',
                        type=int,
                        default=1,
                        help='Number stable neighbors.',
                        )
    parser.add_argument('--stableNeighborPDR',
                        dest='stableNeighborPDR',
                        type=float,
                        default=0.7,
                        help='The PDR a neighbor should have with the minimal cell modulation to be called stable neighbor.',
                        )
    parser.add_argument('--measuredData',
                        dest='measuredData',
                        type=int,
                        default=0,
                        help='Using the measured OFDM data',
                        )

    options        = parser.parse_args()
    return options.__dict__

def printOrLog(cpuID, output, verbose):
    assert cpuID is not None

    with open('cpu{0}.templog'.format(cpuID),'w') as f:
        f.write(output)

    if verbose:
        print output

# runs simulations sequentially on all combinations of input parameters
def runSimsSequentially(params):

    (cpuID, numRuns, options, verbose) = params

    # record simulation start time
    simStartTime   = time.time()

    # compute all the simulation parameter combinations
    combinationKeys     = sorted([k for (k,v) in options.items() if type(v)==list])
    simParams           = []
    for p in itertools.product(*[options[k] for k in combinationKeys]):
        simParam = {}
        for (k,v) in zip(combinationKeys,p):
            simParam[k] = v
        for (k,v) in options.items():
            if k not in simParam:
                simParam[k] = v
        simParams      += [simParam]

    # run a simulation for each set of simParams
    for (simParamNum,simParam) in enumerate(simParams):

        # record run start time
        runStartTime = time.time()

        # run the simulation runs
        for runNum in xrange(numRuns):

            # print
            output  = 'parameters {0}/{1}, run {2}/{3}'.format(
               simParamNum+1,
               len(simParams),
               runNum+1,
               numRuns
            )
            printOrLog(cpuID, output, verbose)

            # create singletons
            settings         = SimSettings.SimSettings(cpuID=cpuID, runNum=runNum, **simParam)
            settings.setStartTime(runStartTime)
            settings.setCombinationKeys(combinationKeys)
            simengine        = SimEngine.SimEngine(cpuID=cpuID, runNum=runNum)
            simstats         = SimStats.SimStats(cpuID=cpuID, runNum=runNum, verbose=verbose)

            # start simulation run
            simengine.start()

            # wait for simulation run to end
            simengine.join()

            # destroy singletons
            simstats.destroy()
            simengine.destroy()
            settings.destroy()

        # print
        output  = 'simulation ended after {0:.0f}s.'.format(time.time()-simStartTime)
        printOrLog(cpuID, output, verbose)

def printProgress(cpuIDs):
    while True:
        time.sleep(1)
        output     = []
        for cpuID in cpuIDs:
            with open('cpu{0}.templog'.format(cpuID),'r') as f:
                output += ['[cpu {0}] {1}'.format(cpuID,f.read())]
        allDone = True
        for line in output:
            if line.count('ended')==0:
                allDone = False
        output = '\n'.join(output)
        os.system('cls' if os.name == 'nt' else 'clear')
        print output
        if allDone:
            break


def readOptions(jsonf, options):
    # TODO: what if it does not exist
    data = None
    with open(jsonf) as data_file:
        data = json.load(data_file)
        if "seed" not in data:
            assert False
        else:
            options["seed"] = data["seed"]
        data = data["simulator"] # only read the simulator part

    print data

    for key, value in data.iteritems():
        if key not in options:
            print key
            assert False
        options[key] = value

    print options

#============================ main ============================================

def main():
    # initialize logging
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logging.config.fileConfig(os.path.join(dir_path, 'logging.conf'))

    options = parseCliOptions()
    # seed = None
    # if 'seed' in options:
    #     seed = options['seed']

    multiprocessing.freeze_support()
    max_num_cores = multiprocessing.cpu_count()

    if options['numCores'] == -1:
        num_cores_to_use = max_num_cores
    else:
        num_cores_to_use = options['numCores']

    assert num_cores_to_use <= max_num_cores, "NUMCORES to use is larger than the maximum available number of cores found on the system."

    if options['json'] is not None:
        readOptions(options['json'], options)
        # if seed:
        #     options['seed'] = seed

    if options['gui']:
        # create the GUI, single core
        gui        = SimGui.SimGui()

        # run simulations (in separate thread)
        simThread  = threading.Thread(target=runSimsSequentially, args=((0, options['numRuns'], options, True),))
        simThread.start()

        # Glenn, otherwise the GUI tries to draw while the topology is not yet built
        time.sleep(1)

        # start GUI's mainloop (in main thread)
        gui.mainloop()
    elif num_cores_to_use == 1:
        runSimsSequentially((0, options['numRuns'], options, True))
    else:
        # parallelize
        runsPerCore = int(math.ceil(float(options['numRuns']) / float(num_cores_to_use)))
        pool = multiprocessing.Pool(num_cores_to_use)
        pool.map_async(runSimsSequentially,[(i, runsPerCore, options, False) for i in range(num_cores_to_use)])
        printProgress([i for i in range(num_cores_to_use)])
        raw_input("Done. Press Enter to close.")

    for i in range(num_cores_to_use):
        os.remove('cpu{0}.templog'.format(i))

if __name__ == '__main__':
    main()
