import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import datetime
import re
import copy
import math
import seaborn as sns
import json
import sys
sys.path.insert(1, 'simulator/SimEngine/')
import Modulation
from dijkstra import Dijkstra

# ALERT! THIS LAST _ IS VERY IMPORTANT FOR FILTERING THE CORRECT EXPERIMENTS

fileTranslate = {
                 'distance_01_': '10', \
                 'distance_1_': '100', \
                 'distance_2_': '200', \
                 'distance_3_': '300', \
                 'distance_4_': '400', \
                 'distance_5_': '500', \
                 '_config_config_1_': 'Different slot lengths\nper transmission', \
                 '_config_config_2_': '2 slots\nper transmission', \
                'variable20': 'Slot length = 20 ms', \
                'variable10': 'Slot length = 10 ms', \
                'variable5': 'Slot length = 5 ms', \
                'oneslot': 'Slot length = 30 ms', \
                'staticthree': 'Slot length = 3 x 10 ms', \
                'statictwo': 'Slot length = 2 x 10 ms', \
                 '_c_variable20_traffic_10s_distance_1_': '100', \
                 '_c_variable20_traffic_10s_distance_2_': '200', \
                 '_c_variable20_traffic_10s_distance_3_': '300', \
                 '_c_variable20_traffic_10s_distance_4_': '400', \
                 '_c_variable20_traffic_10s_distance_5_': '500', \
                 '_c_config_1_traffic_short_distance_1_': '100 meters\ndiff. lengths', \
                 '_c_config_2_traffic_short_distance_1_': '100 meters\nlength = 2 slots', \
                 '_c_config_1_traffic_short_distance_2_': '200 meters\ndiff. lengths', \
                 '_c_config_2_traffic_short_distance_2_': '200 meters\nlength = 2 slots',
                 '_c_config_1_traffic_short_distance_3_': '300 meters\ndiff. lengths', \
                 '_c_config_2_traffic_short_distance_3_': '300 meters\nlength = 2 slots',
                 '_c_config_1_traffic_short_distance_4_': '400 meters\ndiff. lengths', \
                 '_c_config_2_traffic_short_distance_4_': '400 meters\nlength = 2 slots',
                 '_c_config_1_traffic_short_distance_5_': '500 meters\ndiff. lengths', \
                 '_c_config_2_traffic_short_distance_5_': '500 meters\nlength = 2 slots', \
                '_c_ILP5msFRnotbonded_omega_2_': '5 ms (30 ms) slots - Omega = 0.2', \
                '_c_ILP5msFRnotbonded_omega_4_': '5 ms (30 ms) slots - Omega = 0.4', \
                '_c_ILP5msFRnotbonded_omega_6_': '5 ms (30 ms) slots - Omega = 0.6', \
                '_c_ILP5msFRnotbonded_omega_8_': '5 ms (30 ms) slots - Omega = 0.8', \
                '_c_ILP5msFR_omega_1_': '5 ms bonded, $\omega$ = 0.1', \
                '_c_ILP5msFR_omega_2_': '5 ms bonded, $\omega$ = 0.2', \
                '_c_ILP5msFR_omega_3_': '5 ms bonded, $\omega$ = 0.3',
                '_c_ILP5msFR_omega_4_': '5 ms bonded, $\omega$ = 0.4', \
                '_c_ILP5msFR_omega_5_': '5 ms bonded, $\omega$ = 0.5', \
                '_c_ILP5msFR_omega_6_': '5 ms bonded, $\omega$ = 0.6', \
                '_c_ILP5msFR_omega_7_': '5 ms bonded, $\omega$ = 0.7', \
                '_c_ILP5msFR_omega_8_': '5 ms bonded, $\omega$ = 0.8', \
                '_c_ILP5msFR_omega_9_': '5 ms bonded, $\omega$ = 0.9', \
                '_c_ILP10msFR_omega_1_': '10 ms bonded, $\omega$ = 0.1', \
                '_c_ILP10msFR_omega_125_': '10 ms bonded, $\omega$ = 0.125',\
                '_c_ILP10msFR_omega_15_': '10 ms bonded, $\omega$ = 0.15', \
                '_c_ILP10msFR_omega_175_': '10 ms bonded, $\omega$ = 0.175', \
                '_c_ILP10msFR_omega_2_': '10 ms bonded, $\omega$ = 0.2', \
                '_c_ILP10msFR_omega_3_': '10 ms bonded, $\omega$ = 0.3', \
                '_c_ILP10msFR_omega_4_': '10 ms bonded, $\omega$ = 0.4', \
                '_c_ILP10msFR_omega_5_': '10 ms bonded, $\omega$ = 0.5', \
                '_c_ILP10msFR_omega_6_': '10 ms bonded, $\omega$ = 0.6', \
                '_c_ILP10msFR_omega_7_': '10 ms bonded, $\omega$ = 0.7', \
                '_c_ILP10msFR_omega_8_': '10 ms bonded, $\omega$ = 0.8', \
                '_c_ILP10msFR_omega_9_': '10 ms bonded, $\omega$ = 0.9', \
                '_c_ILP20msFR_omega_2_': '20 ms, $\omega$ = 0.2', \
                '_c_ILP20msFR_omega_4_': '20 ms, $\omega$ = 0.4', \
                '_c_ILP20msFR_omega_6_': '20 ms, $\omega$ = 0.6', \
                '_c_ILP20msFR_omega_8_': '20 ms, $\omega$ = 0.8', \
                '_c_ILP30msFR_omega_1_': '30 ms, $\omega$ = 0.1', \
                '_c_ILP30msFR_omega_2_': '30 ms, $\omega$ = 0.2', \
                '_c_ILP30msFR_omega_3_': '30 ms, $\omega$ = 0.3', \
                '_c_ILP30msFR_omega_4_': '30 ms, $\omega$ = 0.4', \
                '_c_ILP30msFR_omega_5_': '30 ms, $\omega$ = 0.5', \
                '_c_ILP30msFR_omega_6_': '30 ms, $\omega$ = 0.6', \
                '_c_ILP30msFR_omega_7_': '30 ms, $\omega$ = 0.7', \
                '_c_ILP30msFR_omega_8_': '30 ms, $\omega$ = 0.8', \
                '_c_ILP30msFR_omega_9_': '30 ms, $\omega$ = 0.9',
                '_c_ILP10msFR_omega_9_ss_150_': '10 ms, $\omega$ = 0.9, 150 ms slotframe', \
                '_c_ILP30msFR_omega_9_ss_150_': '30 ms, $\omega$ = 0.9, 150 ms slotframe', \
                '_c_ILP10msFR_omega_9_ss_210_': '10 ms, $\omega$ = 0.9, 210 ms slotframe', \
                '_c_ILP30msFR_omega_9_ss_210_': '30 ms, $\omega$ = 0.9, 210 ms slotframe', \
                '_c_ILP10msFR_omega_9_ss_300_': '10 ms, $\omega$ = 0.9, 300 ms slotframe', \
                '_c_ILP30msFR_omega_9_ss_300_': '30 ms, $\omega$ = 0.9, 300 ms slotframe', \
                '_c_ILP5msFR_omega_6_size_150_': '5 ms - O = 0.6 - 150ms', \
                '_c_ILP5msFR_omega_6_size_210_': '5 ms - O = 0.6 - 210ms', \
                '_c_ILP5msFR_omega_6_size_270_': '5 ms - O = 0.6 - 270ms', \
                '_c_ILP5msFR_omega_6_size_330_': '5 ms - O = 0.6 - 330ms', \
                '_c_ILP30msFR_omega_6_size_150_': '30 ms - O = 0.6 - 150ms', \
                '_c_ILP30msFR_omega_6_size_210_': '30 ms - O = 0.6 - 210ms', \
                '_c_ILP30msFR_omega_6_size_270_': '30 ms - O = 0.6 - 270ms', \
                '_c_ILP30msFR_omega_6_size_330_': '30 ms - O = 0.6 - 330ms', \
}

fileTranslateWrite = {
                'distance_01_': '10', \
                'distance_1_': '100', \
                'distance_2_': '200', \
                'distance_3_': '300', \
                'distance_4_': '400', \
                'distance_5_': '500', \
                '_config_config_1_': 'Different slot lengths\nper transmission', \
                '_config_config_2_': '2 slots\nper transmission'
                }
translate = {'lifetime_250mAh': 'Lifetime for 250 mAh battery (days)', \
             'lifetime_500mAh': 'Lifetime for 500 mAh battery (days)', \
             'lifetime_1000mAh': 'Lifetime for 1000 mAh battery (days)', \
             'lifetime_1500mAh': 'Lifetime for 1500 mAh battery (days)', \
             'lifetime_2000mAh_openMoteCC2538': 'Lifetime (OpenMote CC2538)\n2000 mAh battery (days)', \
             'lifetime_2000mAh_openMoteB': 'Lifetime (OpenMote B)\n2000 mAh battery (days)', \
             'kbitsPerJoule_openMoteCC2538': 'kbit / Joule (OpenMote CC2538)', \
             'kbitsPerJoule_openMoteB': 'kbit / Joule (OpenMote B)', \
             'sixtopTxAddReq': '6P ADD Requests', \
             'received': 'Received packets at root', 'DAOMessaging': 'DAO Messaging', \
             'resfConvergence': 'ASN of ReSF convergence', 'latency': 'Latency (s)', 'macDrops': 'MAC drops', 'queueDrops': 'Queue drops',
             'pktGen': 'Generated Packets', 'allDrops': 'Dropped Packets',
             'nrSlots': 'Number of slots used per transmission',
             'modulations': 'Number of nodes',
             'distances': 'Average distance to parent (m)'}
translateMotes = {'100': '100\nMotes', '200': '200\nMotes', '300': '300\nMotes', '14': '14 nodes', '8': '8 nodes', None: 'None'}
translateModulations = {'QAM_16_FEC_3_4': 'MCS 6 (16-QAM, FEC 3/4)', \
                        'QAM_16_FEC_1_2': 'MCS 5 (16-QAM, FEC 1/2)', \
                        'QPSK_FEC_3_4': 'MCS 4 (OQPSK, FEC 3/4)', \
                        'QPSK_FEC_1_2': 'MCS 3 (OQPSK, FEC 1/2)', \
                        'QPSK_FEC_1_2_FR_2': 'MCS 2 (OQPSK, FEC 1/2, FR 2x)'}
translateConfigModulations = {'config_1': 'Different slot lengths\nper transmission', \
                            'config_2': '2 slots\nper transmission',
                              'variable20': 'Slot length = 20 ms', \
                              'variable10': 'Slot length = 10 ms', \
                              'variable5': 'Slot length = 5 ms', \
                              'oneslot': 'Slot length = 30 ms', \
                              'staticthree': 'Slot length = 3 x 10 ms', \
                              'statictwo': 'Slot length = 2 x 20 ms', \
                              'ILP5msFRnotbonded': 'ILP - Slot length = 5 ms (30ms)', \
                              'ILP5msFR': 'ILP - Slot length = 5 ms', \
                              'ILP10msFR': '10 ms bonded slots', \
                              'ILP20msFR': 'ILP - Slot length = 20 ms', \
                              'ILP30msFR': '30 ms slots', \
                              'ILP40msFR': '40 ms slots', \
                              'MCS234s10ms': '10 ms, multi-MCS',
                                'MCS234s5ms': '5 ms, multi-MCS',
                                'MCS234s20ms': '20 ms, multi-MCS',
                                'MCS234s40ms': '40 ms, multi-MCS',
                              'MCS2s10ms': '10 ms, MCS2',
                              'MCS24s40ms': '40 ms, MCSs 2 and 4',
                                'MCS24s10ms': '10 ms, MCSs 2 and 4',
                              None: 'None', \
                              }
translateDistances = {'1': '100', \
                        '2': '200', \
                        '3': '300', \
                        '4': '400', \
                        '5': '500',
                        '01': '10',
                        None: 'None'}
translateOmega = {None:'0', '1': '0.1', '125': '0.125', '15': '0.15', '175': '0.175', '2': '0.2', '3': '0.3', '4': '0.4', '5': '0.5', '6': '0.6', '7': '0.7', '8': '0.8', '9': '0.9'}
metrics = ['bitperjoule', 'lifetime']
colors = ['red', 'green', 'blue', 'orange', 'yellow', 'black']
translateStates = {'nrSleep': 'Sleep', \
          'nrIdle': 'Idle', \
          'nrIdleNotSync': 'Idle, not sync', \
          'nrTxDataRxAck': 'TX Data,\nRX ACK', \
          'nrTxDataRxNack': 'TX Data,\nRX NACK' , \
          'nrTxDataNoAck': 'TX Data,\nNo ACK', \
          'nrTxData': 'TX Data' , \
          'nrRxDataTxAck': 'RX Data,\nTX ACK',\
          'nrRxData': 'RX Data'}

modulationInstance = Modulation.Modulation(parsing_results_call=True, config_file='simulator/SimEngine/modulation_files/modulation_stable_mcs2.json')

VOLTS_RADIO = 3.0
VOLTS_CPU = 1.8
#
# uC_IDLE_CONSUMPTION = 47.54
# uC_IDLE_NOT_SYNC_CONSUMPTION = 47.54
# uC_IDLE_CONSUMPTION = 0.0
# uC_IDLE_NOT_SYNC_CONSUMPTION = 0.0

# simulation values of Accurate energy paper
uC_IDLE_CONSUMPTION = 47.54
uC_IDLE_NOT_SYNC_CONSUMPTION = 47.54
uC_SLEEP_CONSUMPTION = 0.82
uC_TXDATARXACK_CONSUMPTION = 106.45
uC_TXDATA_CONSUMPTION = 83.07
uC_TXDATANOACK_CONSUMPTION = 100.32
uC_RXDATATXACK_CONSUMPTION = 107.66
uC_RXDATA_CONSUMPTION = 82.97

J_IDLE_CONSUMPTION = (uC_IDLE_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_IDLE_NOT_SYNC_CONSUMPTION = (uC_IDLE_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_SLEEP_CONSUMPTION = (uC_SLEEP_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_TXDATARXACK_CONSUMPTION = (uC_TXDATARXACK_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_TXDATA_CONSUMPTION = (uC_TXDATA_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_TXDATANOACK_CONSUMPTION = (uC_TXDATANOACK_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_RXDATATXACK_CONSUMPTION = (uC_RXDATATXACK_CONSUMPTION / 1000000.0) * VOLTS_RADIO
J_RXDATA_CONSUMPTION = (uC_RXDATA_CONSUMPTION / 1000000.0) * VOLTS_RADIO

VOLTS_RADIO_OPENMOTEB = 3.0
VOLTS_CPU_OPENMOTEB = 1.8
#
# uC_IDLE_CONSUMPTION = 47.54
# uC_IDLE_NOT_SYNC_CONSUMPTION = 47.54
# uC_IDLE_CONSUMPTION = 0.0
# uC_IDLE_NOT_SYNC_CONSUMPTION = 0.0

# simulation values of Accurate energy paper
uC_IDLE_CONSUMPTION_OPENMOTEB = 52.17
uC_IDLE_NOT_SYNC_CONSUMPTION_OPENMOTEB = 52.17
uC_SLEEP_CONSUMPTION_OPENMOTEB = 0.02
uC_TXDATARXACK_CONSUMPTION_OPENMOTEB = 134.04
uC_TXDATA_CONSUMPTION_OPENMOTEB = 106.7
uC_TXDATANOACK_CONSUMPTION_OPENMOTEB = 125.89
uC_RXDATATXACK_CONSUMPTION_OPENMOTEB = 137.72
uC_RXDATA_CONSUMPTION_OPENMOTEB = 107.84

J_IDLE_CONSUMPTION_OPENMOTEB = (uC_IDLE_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_IDLE_NOT_SYNC_CONSUMPTION_OPENMOTEB = (uC_IDLE_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_SLEEP_CONSUMPTION_OPENMOTEB = (uC_SLEEP_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_TXDATARXACK_CONSUMPTION_OPENMOTEB = (uC_TXDATARXACK_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_TXDATA_CONSUMPTION_OPENMOTEB = (uC_TXDATA_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_TXDATANOACK_CONSUMPTION_OPENMOTEB = (uC_TXDATANOACK_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_RXDATATXACK_CONSUMPTION_OPENMOTEB = (uC_RXDATATXACK_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
J_RXDATA_CONSUMPTION_OPENMOTEB = (uC_RXDATA_CONSUMPTION_OPENMOTEB / 1000000.0) * VOLTS_RADIO_OPENMOTEB
#
# SLOTDURATION = 0.020 # ms
# SLOTFRAME_LENGTH = 101 # slots
APPLICATION_SIZE_BITS = 104 * 8 # bits

# slotdurationDict = {'variable20': 0.020, 'variable10': 0.010, 'variable5': 0.005, 'statictwo': 0.020, 'staticthree': 0.010, 'oneslot': 0.030, 'ilp_5ms': 0.005}
# slotFrameDict = {'variable20': 199, 'variable10': 397, 'variable5': 797, 'statictwo': 199, 'staticthree': 397, 'oneslot': 131}
# slotFrameDict = {'variable20': 43, 'variable10': 89, 'variable5': 179, 'statictwo': 199, 'staticthree': 397, 'oneslot': 29}
# slotFrameDict = {'variable20': 43, 'variable10': 83, 'variable5': 173, 'statictwo': 199, 'staticthree': 397, 'oneslot': 29}

# slotFrameDict = {'variable20': 45, 'variable10': 90, 'variable5': 180, 'statictwo': 199, 'staticthree': 397, 'oneslot': 30, 'ilp_5ms': 6}


def validate(exp_dir):
    """ Validate the experiment to be really successful."""
    errors = []

    error_log = '%s/error.log' % exp_dir # should be empty
    id_pattern = '*.id.txt' # should only be one file with this id_pattern
    ga_solution = '%s/ga.json' % exp_dir # should be there
    ga_output = '%s/ga_output.txt' % exp_dir # should be there
    simulator_topology = '%s/simulator-topology.json' % exp_dir # should be there
    simulator_topology_output = '%s/simulator_output.txt' % exp_dir # should be there

    import os
    if not os.path.exists(error_log) or os.path.getsize(error_log) > 0:
        raise 'Error log not there or not zero: {0}'.format(exp_dir)
    # if not os.path.exists(ga_solution) or os.path.getsize(ga_solution) == 0:
    #     raise 'There is no GA solution: {0}'.format(exp_dir)
    if not os.path.exists(ga_output) or os.path.getsize(ga_output) == 0:
        raise 'There is no GA output: {0}'.format(exp_dir)
    if not os.path.exists(simulator_topology_output) or os.path.getsize(simulator_topology_output) == 0:
        raise 'There is no simulator topology output: {0}'.format(exp_dir)
    if not os.path.exists(simulator_topology) or os.path.getsize(simulator_topology) == 0:
        raise 'There is no simulator topology: {0}'.format(simulator_topology)
    if "failed" in exp_dir:
        raise 'This is a failed experiment: {0}.'.format(exp_dir)

    import fnmatch
    count_workers = fnmatch.filter(os.listdir(exp_dir), id_pattern)
    if len(count_workers) > 1:
        raise 'Multiple workers worked on this experiment: {0}'.format(exp_dir)

def getNrMotes(datafile):
    with open(datafile) as f:
        for line in f:
            if '## numMotes =' in line:
                return int(line.split('=')[1].strip()) # num motes

def get_set_rgx(experiments, rgx = ''):
    candidates = set()
    for exp in experiments:
        regex_result = re.search(rgx, exp, re.IGNORECASE)
        if regex_result is not None:
            candidates.add(regex_result.group(1))
        # else:
        #     raise 'No %s indicator in experiment dir.' % rgx
    return candidates

def detectInName(search_parameter, exp_dir):
    cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(exp_dir, search_parameter)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    rgx = '[_\/]+%s_([A-Za-z0-9]+)_' % search_parameter
    candidates = get_set_rgx(listFiles, rgx)
    return candidates

def getExperimentName(exp):
    if exp in fileTranslate:
        return fileTranslate[exp]
    return exp

def getLabelName(name):
    if name in translate:
        return translate[name]
    return name

def parseTimeResults(dataDir, parameter):
    # print data
    cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, parameter)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    print "parseTimeResults - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    foundIndices = False
    cycleIndex = None
    allCellsIndex = None
    usedCellsIndex = None
    overlappingCellsIndex = None

    pandaData = {'cycle': [], \
                 'allCells': [], \
                 'usedCells': [], \
                 'overlappingCells': [], \
                 'overlappingRatio': [], \
                 'slotTime': [], \
                 'airTime': [], \
                 'airTimeRatio': [], \
                 'iteration': []}

    for datafile in listFiles:
        validate(os.path.dirname(datafile))

        # get all the data
        with open(datafile, 'r') as inF:
            for line in inF:
                if not foundIndices and 'cycle' in line and 'usedCells' in line:
                    lineList = line.strip().split() # first two are information, drop them
                    foundIndices = True
                    cycleIndex = lineList.index('cycle') - 1
                    usedCellsIndex = lineList.index('usedCells') - 1
                    allCellsIndex = lineList.index('allCells') - 1
                    overlappingCellsIndex = lineList.index('overlappingCells') - 1
                    slotTimeIndex = lineList.index('slotTime') - 1
                    airTimeIndex = lineList.index('airTime') - 1
                elif foundIndices and '#pos' in line:
                    foundIndices = False
                elif foundIndices:
                    lineList = line.strip().split() # first two are information, drop them
                    # print lineList
                    if len(lineList) > usedCellsIndex:
                        pandaData['cycle'].append(float(lineList[cycleIndex]))
                        pandaData['allCells'].append(float(lineList[allCellsIndex]))
                        pandaData['usedCells'].append(float(lineList[usedCellsIndex]))
                        pandaData['overlappingCells'].append(float(lineList[overlappingCellsIndex]))
                        if float(lineList[allCellsIndex]) > 0.0:
                            pandaData['overlappingRatio'].append(float(lineList[overlappingCellsIndex])/float(lineList[allCellsIndex]))
                        else:
                            pandaData['overlappingRatio'].append(0.0)
                        pandaData['slotTime'].append(float(lineList[slotTimeIndex]))
                        pandaData['airTime'].append(float(lineList[airTimeIndex]))
                        if float(lineList[slotTimeIndex]) > 0.0:
                            pandaData['airTimeRatio'].append(float(lineList[airTimeIndex]) / float(lineList[slotTimeIndex]))
                        else:
                            pandaData['airTimeRatio'].append(0.0)
                        pandaData['iteration'].append(datafile)

    df = pd.DataFrame(pandaData)
    return df

def parseILPSolution(dataDir, parameter, slotData, modulationConfig):
    cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, parameter)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations]
    cmd = "find {0} -ipath *{1}*/ilp_schedule.json".format(dataDir, parameter)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "parseILPSolution - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # validate(os.path.dirname(datafile))
        scheduleData = None
        with open(datafile) as json_file:
            scheduleData = json.load(json_file)

        nrCells = 0

        for node, nodeSchedule in scheduleData['schedule'].iteritems():
            for timeslot, timeslotSchedule in nodeSchedule.iteritems():
                for channel, bondedSlot in timeslotSchedule.iteritems():
                    nrCells += bondedSlot['slots']

        slotData = slotData.append([{'iteration': datafile, 'iterationDir': os.path.dirname(datafile),
                                     'ilpCells': nrCells, 'experimentType': getExperimentName(parameter),
                                     'scheduleTime': nrCells * modulationInstance.modulationConfigSlotLength[modulationConfig],
                                     'modulationConfig': modulationConfig}])

    return slotData

def parseLinks(dataDir, parameter, links):
    cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, parameter)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    print "parseLinks - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))
    for datafile in listFiles:
        validate(os.path.dirname(datafile))

        numMotes = getNrMotes(datafile)

        links_datafile = '{0}/ilp.json'.format(os.path.dirname(datafile))
        with open(links_datafile, 'r') as f:
            ilp_json = json.load(f)
            for modu in Modulation.Modulation().modulations:
                if modu not in links:
                    links[modu] = []
                for m in range(numMotes):
                    for mOther in range(numMotes):
                        if m < mOther:
                            # print 'Looking up reliability for modulation {0} from {1} to {2} in datafile {3}: {4}'.format(modu, m, mOther, links_datafile, ilp_json['simulationTopology'][str(m)]['reliability'][str(modu)][str(mOther)])
                            links[modu].append(ilp_json['simulationTopology'][str(m)]['reliability'][str(modu)][str(mOther)])

def calculateILPThroughputPerExperiment(dataDir, exp=None, modulationConfig=None, ss=None, numMotes=None):
    # cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, exp)
    # listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, exp)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, exp)
    listFiles = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]

    print "parseThroughputILP - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))
    results = []
    for datafile in listFiles:
        validate(os.path.dirname(datafile))

        links_datafile = '{0}/ga.json'.format(os.path.dirname(datafile))
        with open(links_datafile, 'r') as f:
            ilp_solution = json.load(f)
            results.append({'exp': getExperimentName(exp), 'ilp_throughput': float(ilp_solution['results']['best_ind']['tput']/(numMotes-1))*100.0, 'iteration': links_datafile, 'iterationDir': os.path.dirname(links_datafile), 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})

    return results

def reach_root(node, par_parents, par_allocatedBondedSlots, parents):
    if par_allocatedBondedSlots[node] == 0:
        return False
    if node not in par_parents:
        return False
    elif par_parents[node] in parents:
        return False
    elif par_parents[node] == 0:
        return True
    parents.append(par_parents[node])
    return reach_root(par_parents[node], par_parents, par_allocatedBondedSlots, parents)


def ga_calculate_minimal_hops(datafile_dir=None):
    cmd = "find {0} -ipath \"*/ga_seed_*/simulator-topology.json\"".format(datafile_dir)
    topology_files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(topology_files) == 0 or len(topology_files) > 1:
        raise BaseException('There should only be one simulator file: {0} > {1}'.format(datafile_dir, topology_files))

    cmd = "find {0} -ipath \"*/ga_seed_*/ga_seed_*.json\"".format(datafile_dir)
    settings_files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(settings_files) == 0 or len(settings_files) > 1:
        raise BaseException('There should only be one settings file: {0} > {1}'.format(datafile_dir, settings_files))

    # look if the GA built topologies or not by reading the settings file
    settings = None
    with open(settings_files[0]) as json_file:
        settings = json.load(json_file)

    dijkstra = Dijkstra()
    dijkstra.calculate(settings_file=settings_files[0], topology_file=topology_files[0])
    return dijkstra

def parseResults(dataDir, parameter, data, consumption):
    resfConvergenceDatafiles = []
    # print data
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, parameter)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/output_cpu0.dat\"".format(dataDir, parameter)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    # cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, parameter)
    # listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    print "parseResults - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))
    for datafile in listFiles:
        validate(os.path.dirname(datafile))
        # print "parseResults: datafile {0}".format(datafile)

        nrIdle = {}
        nrIdleNotSync = {}
        nrSleep = {}
        nrTxDataRxAck = {}
        nrTxData = {}
        nrTxDataNoAck = {}
        nrTxDataRxNack = {}
        nrRxDataTxAck = {}
        nrRxData = {}
        pktReceived = {}
        pktLatencies = {}
        pktDropsMac = {}
        pktDropsQueue = {}
        pktGen = {}
        pktArrivedToGen = {}
        sixtopTxAddReq = {}
        sixtopTxAddResp = {}
        sixtopTxDelReq = {}
        sixtopTxDelResp = {}
        activeDAO = {}
        initiatedDAO = {}
        receivedDAO = {}
        distance = {}
        nrSlots = {}
        modulation = {}
        hopcount = {}
        children = {}
        reliability = {}
        prefParent = {}
        allocatedBondedSlots = {}

        totalPropagationData = {}
        successPropagationData = {}
        interferenceFailures = {}
        interferenceLockFailures = {}
        signalFailures = {}
        allInterferers = {}
        hadInterferers = {}

        resfConvergence = {}
        # get all the data
        with open(datafile, 'r') as inF:
            for line in inF:
                if '#prefParent' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            prefParent[int(mote.split("@")[0])] = int(mote.split("@")[1])
                        else:
                            prefParent[int(mote.split("@")[0])] = None
                if '#allocatedBondedSlots' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            allocatedBondedSlots[int(mote.split("@")[0])] = int(mote.split("@")[1])
                        else:
                            allocatedBondedSlots[int(mote.split("@")[0])] = None
                if '#nrIdle' in line and not '#nrIdleNotSync' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrIdle[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrIdleNotSync' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrIdleNotSync[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrSleep' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrSleep[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrTxDataRxAck' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrTxDataRxAck[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrTxData' in line and not '#nrTxDataNoAck' in line and not '#nrTxDataRxAck' in line and not '#nrTxDataRxNack' in line:
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrTxData[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrTxDataNoAck' in line.strip():
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrTxDataNoAck[int(mote.split("@")[0])] = float(mote.split("@")[1])
                    # print 'Got charge.'
                if '#nrTxDataRxNack' in line.strip():
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        nrTxDataRxNack[int(mote.split("@")[0])] = float(mote.split("@")[1])
                if '#nrRxDataTxAck' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        nrRxDataTxAck[int(mote.split("@")[0])] = float(mote.split("@")[1])
                if '#nrRxData' in line.strip() and not '#nrRxDataTxAck' in line:
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        nrRxData[int(mote.split("@")[0])] = float(mote.split("@")[1])
                if '#PktReceived' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if '@' in mote:
                            if mote.split("@")[1] != 'None':
                                pktReceived[int(mote.split("@")[0])] = float(mote.split("@")[1])
                            else:
                                pktReceived[int(mote.split("@")[0])] = None
                    # print 'Got hopcount.'
                if '#sixtopTxAddReq' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        sixtopTxAddReq[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#sixtopTxAddResp' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        sixtopTxAddResp[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#sixtopTxDelReq' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        sixtopTxDelReq[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#sixtopTxDelResp' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        sixtopTxDelResp[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#activeDAO' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        activeDAO[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#initiatedDAO' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        initiatedDAO[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#receivedDAO' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        receivedDAO[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#PktLatencies' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            pktLatencies[int(mote.split("@")[0])] = float(mote.split("@")[1])
                        else:
                            pktLatencies[int(mote.split("@")[0])] = None
                if '#PktDropsMac' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if '@' in mote:
                            if mote.split("@")[1] != 'None':
                                pktDropsMac[int(mote.split("@")[0])] = float(mote.split("@")[1])
                            else:
                                pktDropsMac[int(mote.split("@")[0])] = None
                if '#PktDropsQueue' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if '@' in mote:
                            if mote.split("@")[1] != 'None':
                                pktDropsQueue[int(mote.split("@")[0])] = float(mote.split("@")[1])
                            else:
                                pktDropsQueue[int(mote.split("@")[0])] = None
                if '#PktGen' in line.strip():
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        if '@' in mote:
                            if mote.split("@")[1] != 'None':
                                pktGen[int(mote.split("@")[0])] = float(mote.split("@")[1])
                            else:
                                pktGen[int(mote.split("@")[0])] = None
                if '#PktArrivedToGen' in line.strip():
                    lineList = line.strip().split(' ')[2:] # first two are information, drop them
                    for mote in lineList:
                        if '@' in mote:
                            if mote.split("@")[1] != 'None':
                                pktArrivedToGen[int(mote.split("@")[0])] = float(mote.split("@")[1])
                            else:
                                pktArrivedToGen[int(mote.split("@")[0])] = None
                if '#distance' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            distance[int(mote.split("@")[0])] = float(mote.split("@")[1])
                        else:
                            distance[int(mote.split("@")[0])] = None
                if '#modulation' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            modulation[int(mote.split("@")[0])] = str(mote.split("@")[1])
                        else:
                            modulation[int(mote.split("@")[0])] = None
                if '#nrSlots' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            nrSlots[int(mote.split("@")[0])] = float(mote.split("@")[1])
                        else:
                            nrSlots[int(mote.split("@")[0])] = None
                if '#totalPropagationData' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        totalPropagationData[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#successPropagationData' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        successPropagationData[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#interferenceFailures' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        interferenceFailures[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#interferenceLockFailures' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        interferenceLockFailures[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#signalFailures' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        signalFailures[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#allInterferers' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        allInterferers[int(mote.split("@")[0])] = float(mote.split("@")[1])
                if '#hadInterferers' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        hadInterferers[int(mote.split("@")[0])] = float(mote.split("@")[1])
                if '#reliability' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        if mote.split("@")[1] != 'None':
                            reliability[int(mote.split("@")[0])] = float(mote.split("@")[1])
                        else:
                            reliability[int(mote.split("@")[0])] = None
                if '#hopcount' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        hopcount[int(mote.split("@")[0])] = int(mote.split("@")[1])
                if '#children' in line.strip():
                    lineList = line.strip().split(' ')[2:]  # first two are information, drop them
                    for mote in lineList:
                        children[int(mote.split("@")[0])] = int(mote.split("@")[1])

        if datafile not in data:
            data[datafile] = [] # make a list of (id, hopcount, consumption) per node per datafile

        numMotes = getNrMotes(datafile)

        reachRoot = {}
        for mote in range(1, numMotes):
            if reach_root(mote, prefParent, allocatedBondedSlots, []):
                reachRoot[mote] = 1
            else:
                reachRoot[mote] = 0
        reachRoot[0] = None

        # calculate the minimal hop counts for this experiment iteration
        dijkstra = ga_calculate_minimal_hops(os.path.dirname(datafile))

        # summarize the data
        for mote in range(0, numMotes):
            if interferenceLockFailures[mote] > 0.0:
                print("Found interference failure")
                print(interferenceFailures[mote])
                print(interferenceLockFailures[mote])
                print(datafile)
                exit()
            regularSlots = 0
            if mote != 0 and reachRoot[mote] == 1:
                regularSlots = allocatedBondedSlots[mote] * nrSlots[mote]
            data[datafile].append({'mote': mote, 'prefParent': prefParent[mote], 'allocatedBondedSlots': allocatedBondedSlots[mote], 'reachRoot': reachRoot[mote],
                                   'allocatedRegularSlots': regularSlots, 'nrIdle': nrIdle[mote],
                                   'nrIdleNotSync': nrIdleNotSync[mote], 'nrSleep': nrSleep[mote],
                                   'nrTxDataRxAck': nrTxDataRxAck[mote], 'nrTxDataNoAck': nrTxDataNoAck[mote],
                                   'nrTxDataRxNack': nrTxDataRxNack[mote],
                                   'nrTxData': nrTxData[mote], 'nrRxDataTxAck': nrRxDataTxAck[mote],
                                   'nrRxData': nrRxData[mote], 'pktReceived': pktReceived[mote], 'pktLatencies': pktLatencies[mote],
                                   'sixtopTxAddReq': sixtopTxAddReq[mote], 'sixtopTxAddResp': sixtopTxAddResp[mote],
                                   'sixtopTxDelReq': sixtopTxDelReq[mote], 'sixtopTxDelResp': sixtopTxDelResp[mote],
                                   'activeDAO': activeDAO[mote], 'initiatedDAO': initiatedDAO[mote], 'receivedDAO': receivedDAO[mote],
                                   'pktDropsMac': pktDropsMac[mote], 'pktDropsQueue': pktDropsQueue[mote], 'pktGen': pktGen[mote],
                                   'pktArrivedToGen': pktArrivedToGen[mote], 'distance': distance[mote], 'modulation': modulation[mote], 'nrSlots': nrSlots[mote],
                                   'totalPropagationData': totalPropagationData[mote], 'successPropagationData': successPropagationData[mote],
                                   'interferenceFailures': interferenceFailures[mote], 'interferenceLockFailures': interferenceLockFailures[mote],
                                   'signalFailures': signalFailures[mote], 'allInterferers': allInterferers[mote], 'hopcount': hopcount[mote], 'minimalHopcount': dijkstra.dijkstra_table[mote]['distance'],
                                   'hadInterferers': hadInterferers[mote], 'reliability': reliability[mote], 'children': children[mote]})

        if data[datafile][0]['mote'] != 0: # should be root node
            assert False

        consumption_datafile = '{0}/consumption.json'.format(os.path.dirname(datafile))
        with open(consumption_datafile, 'r') as f:
            consumption[datafile] = json.load(f)

        # print datafile

    # period = float(list(detectInName('p', listFiles[0]))[0])

    # print listFiles
    # cycles = float(list(detectInName('cycles', listFiles[0]))[0])
    # for df in listFiles:
    #     # p = float(list(detectInName('p', df))[0])
    #     c = float(list(detectInName('cycles', df))[0])
    #     if c != cycles:
    #         # print '%.4f vs %.4f' % (p, period)
    #         print '%.4f vs %.4f' % (c, cycles)
    #         raise 'Different cycles!'

    cycles = 2376
    # cycles = 3037

    # print len(resfConvergenceDatafiles)
    # print resfConvergenceDatafiles

    return cycles

# Calculate the bits per Joule per iteration.
def calculateBitsPerJoulePerIteration(data, moteType, exp):
    results = []
    for iteration in data:
        nrIdle = 0
        nrIdleNotSync = 0
        nrSleep = 0
        nrTxDataRxAck = 0
        nrTxDataNoAck = 0
        nrTxData = 0
        nrRxDataTxAck = 0
        nrRxData = 0
        nrReceived = 0

        for moteList in data[iteration]:
            if moteList['mote'] != 0:
                nrIdle += moteList['nrIdle']
                nrIdleNotSync += moteList['nrIdleNotSync']
                nrSleep += moteList['nrSleep']
                nrTxDataRxAck += moteList['nrTxDataRxAck']
                nrTxData += moteList['nrTxData']
                nrTxDataNoAck += moteList['nrTxDataNoAck']
                nrRxDataTxAck += moteList['nrRxDataTxAck']
                nrRxData += moteList['nrRxData']
            if moteList['mote'] == 0:
                nrReceived += moteList['pktReceived']

        totalConsumption = 0.0
        if moteType == 'OpenMoteCC2538':
            totalConsumption += nrIdle * J_IDLE_CONSUMPTION
            totalConsumption += nrIdleNotSync * J_IDLE_NOT_SYNC_CONSUMPTION
            totalConsumption += nrSleep * J_SLEEP_CONSUMPTION
            totalConsumption += nrTxDataRxAck * J_TXDATARXACK_CONSUMPTION
            totalConsumption += nrTxData * J_TXDATA_CONSUMPTION
            totalConsumption += nrTxDataNoAck * J_TXDATANOACK_CONSUMPTION
            totalConsumption += nrRxDataTxAck * J_RXDATATXACK_CONSUMPTION
            totalConsumption += nrRxData * J_RXDATA_CONSUMPTION
        elif moteType == 'OpenMoteB':
            totalConsumption += nrIdle * J_IDLE_CONSUMPTION_OPENMOTEB
            totalConsumption += nrIdleNotSync * J_IDLE_NOT_SYNC_CONSUMPTION_OPENMOTEB
            totalConsumption += nrSleep * J_SLEEP_CONSUMPTION_OPENMOTEB
            totalConsumption += nrTxDataRxAck * J_TXDATARXACK_CONSUMPTION_OPENMOTEB
            totalConsumption += nrTxData * J_TXDATA_CONSUMPTION_OPENMOTEB
            totalConsumption += nrTxDataNoAck * J_TXDATANOACK_CONSUMPTION_OPENMOTEB
            totalConsumption += nrRxDataTxAck * J_RXDATATXACK_CONSUMPTION_OPENMOTEB
            totalConsumption += nrRxData * J_RXDATA_CONSUMPTION_OPENMOTEB

        # to kbit
        results.append({'exp': getExperimentName(exp), 'val': (((APPLICATION_SIZE_BITS * (nrReceived)) / (float(totalConsumption))) / 1000.0), 'iteration': iteration})

    return results

# Get all the numbers of received packets per iteration.
def calculateReceived(data, exp, nrMotes=None, modulationConfig=None, ss=None, topConfig=None, deltaConfig=None):
    results = []
    for iteration in data:
        result = {'exp': getExperimentName(exp), 'val': None, 'iteration': iteration, 'iterationDir': os.path.dirname(iteration), 'motes': translateMotes[nrMotes], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss, 'topConfig': topConfig, 'deltaConfig': deltaConfig}
        nrReceived = 0
        for moteList in data[iteration]:
            if moteList['mote'] == 0:
                nrReceived += moteList['pktReceived']
                break
        result['val'] = nrReceived
        results.append(copy.deepcopy(result))
    return results

def calculatePktGen(data, exp, nrMotes=None, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        result = {'exp': getExperimentName(exp), 'val': None, 'iteration': iteration, 'motes': translateMotes[nrMotes], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss}
        nrPktGen = 0
        for moteList in data[iteration]:
            if moteList['mote'] != 0 and moteList['pktGen'] is not None:
                nrPktGen += moteList['pktGen']
        result['val'] = nrPktGen
        results.append(copy.deepcopy(result))
    return results

def calculatePktArrivedToGen(data, exp, nrMotes=None, modulationConfig=None, ss=None, topConfig=None, deltaConfig=None):
    results = []
    for iteration in data:
        result = {'exp': getExperimentName(exp), 'val': None, 'iteration': iteration, 'motes': translateMotes[nrMotes], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss, 'topConfig': topConfig, 'deltaConfig': deltaConfig}
        nrPktArrivedToGen = 0
        for moteList in data[iteration]:
            if moteList['mote'] != 0 and moteList['pktArrivedToGen'] is not None:
                nrPktArrivedToGen += moteList['pktArrivedToGen']
        result['val'] = nrPktArrivedToGen
        results.append(copy.deepcopy(result))
    return results

# Get all charges of all motes.
# def calculateChargePerMote(data, moteType, exp, omega=None):
#     results = []
#     for iteration in data:
#         for moteList in data[iteration]:
#             # do not do this for root, only for source nodes
#             if moteList['mote'] != 0:
#                 totalCharge = 0
#                 if moteType == 'OpenMoteCC2538':
#                     totalCharge += moteList['nrIdle'] * uC_IDLE_CONSUMPTION
#                     totalCharge += moteList['nrIdleNotSync'] * uC_IDLE_NOT_SYNC_CONSUMPTION
#                     totalCharge += moteList['nrSleep'] * uC_SLEEP_CONSUMPTION
#                     totalCharge += moteList['nrTxDataRxAck'] * uC_TXDATARXACK_CONSUMPTION
#                     totalCharge += moteList['nrTxData'] * uC_TXDATA_CONSUMPTION
#                     totalCharge += moteList['nrTxDataNoAck'] * uC_TXDATANOACK_CONSUMPTION
#                     totalCharge += moteList['nrRxDataTxAck'] * uC_RXDATATXACK_CONSUMPTION
#                     totalCharge += moteList['nrRxData'] * uC_RXDATA_CONSUMPTION
#                 elif moteType == 'OpenMoteB':
#                     totalCharge += moteList['nrIdle'] * uC_IDLE_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrIdleNotSync'] * uC_IDLE_NOT_SYNC_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrSleep'] * uC_SLEEP_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrTxDataRxAck'] * uC_TXDATARXACK_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrTxData'] * uC_TXDATA_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrTxDataNoAck'] * uC_TXDATANOACK_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrRxDataTxAck'] * uC_RXDATATXACK_CONSUMPTION_OPENMOTEB
#                     totalCharge += moteList['nrRxData'] * uC_RXDATA_CONSUMPTION_OPENMOTEB
#                 # this is the total charge for the whole length of the experiment
#                 results.append({'exp': getExperimentName(exp), 'val': totalCharge, 'iteration': iteration, 'mote': moteList['mote'], 'omega': translateOmega[omega]})
#     return results

# Calculate the lifetimes of all charges of all motes in all iterations.
# def calculateLifetime(chargePerMoteDF, batterySize, cycles, exp, omega=None):
#     results = []
#     for index, row in chargePerMoteDF.iterrows():
#         # total mAh of whole experimetn
#         mAh = (row['val']) / 3600000.0  # uC / 3600000 = mAh
#         # get length experiment:
#         numCycles = cycles
#         # convert numCycles to seconds
#         lengthSeconds = (numCycles) * (modulationInfo['configurations'][modulationConfig]['slotLength'] / 1000.0) * SLOTFRAME_LENGTH
#         # number of seconds you could do with this battery
#         batterySeconds = float(batterySize) / (mAh / float(lengthSeconds))
#         # convert to days
#         days = batterySeconds / 3600.0 / 24.0
#         # days = moteCharge
#         results.append({'exp': getExperimentName(exp), 'val': days, 'iteration': row['iteration'], 'mote': row['mote'], 'omega': translateOmega[omega]})
#     return results


def checkForEqualLengths(consumption, exp, slotLength):
    results = []
    startExperiment = None
    endExperiment = None
    for iteration in consumption:
        if startExperiment is None and endExperiment is None:
            startExperiment = consumption[iteration]['startASN']
            endExperiment = consumption[iteration]['endASN']
        if startExperiment != consumption[iteration]['startASN'] or endExperiment != consumption[iteration]['endASN']:
            assert False # this should be equal

    print (endExperiment - startExperiment) * slotLength
    return (endExperiment - startExperiment) * slotLength

def calculateChargePerMotePerModulation(consumption, exp, slotLength, modulationInstance=None, ss=None):
    results = []
    startExperiment = None
    endExperiment = None
    for iteration in consumption:
        startExperiment = consumption[iteration]['startASN']
        endExperiment = consumption[iteration]['endASN']
        # print 'iteration {0}, start = {1} - end {2}'.format(iteration, startExperiment, endExperiment)
        for moteID, consumptionPerState in consumption[iteration]['totalConsumption'].iteritems():
            # do not do this for root, only for source nodes
            # if moteID == '0':
            #     assert False
            if moteID != '0':
                totalCharge = 0
                totalNrActiveSlots = 0.0

                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrIdle'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('idle', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    slots = (modulationCount)
                    ch = modulationCount * modulationInstance.getCharge('idle', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    # print 'mote %s, modulation %s, slots %d, Idle charge %.7f' % (str(moteID), modulation, slots, ch)
                    # if slots > 1000:
                    #     print iteration
                    #     assert False
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrIdleNotSync'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('idle', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataRxAck'].iteritems():
                    totalNrActiveSlots += (modulationCount - consumptionPerState['nrTxDataRxNack'][modulation]) * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('txDataRxAck', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    slots = (modulationCount - consumptionPerState['nrTxDataRxNack'][modulation])
                    ch = modulationCount * modulationInstance.getCharge('txDataRxAck', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    # print 'mote %s, modulation %s, slots %d, txDataRxAck charge %.7f' % (str(moteID), modulation, slots, ch)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataRxNack'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('txDataRxNack', Modulation.Modulation().modulationSlots[ modulationConfig][modulation] * slotLength, modulation)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxData'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('txData', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataNoAck'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('txDataNoAck', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrRxDataTxAck'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    slots = modulationCount
                    totalCharge += modulationCount * modulationInstance.getCharge('rxDataTxAck', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    ch = modulationCount * modulationInstance.getCharge('rxDataTxAck', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                    # print 'mote %s, modulation %s, slots %d, rxDataTxAck charge %.7f' % (str(moteID), modulation, slots, ch)
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrRxData'].iteritems():
                    totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                    totalCharge += modulationCount * modulationInstance.getCharge('rxData', Modulation.Modulation().modulationSlots[modulationConfig][modulation] * slotLength, modulation)
                # iterate over all modulations for this state

                totalNrSleepSlots = (endExperiment - startExperiment) - totalNrActiveSlots
                # print 'mote %s, totalNrSleepSlots %d, SLEEP charge %.7f' % (str(moteID), totalNrSleepSlots, totalNrSleepSlots * slotLength * Modulation.Modulation().current_mA['sleep'])
                totalCharge += totalNrSleepSlots * slotLength * Modulation.Modulation().current_mA['sleep'] # take the current for sleep directly as this should not have any modulation

                # totalCharge += moteList['nrIdle'] * modulationInstance.getCharge('idle', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrIdleNotSync'] * modulationInstance.getCharge('idle', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrTxDataRxAck'] * modulationInstance.getCharge('txDataRxAck', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrTxDataRxNack'] * modulationInstance.getCharge('txDataRxNack', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrTxData'] * modulationInstance.getCharge('txData', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrTxDataNoAck'] * modulationInstance.getCharge('txDataNoAck', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrRxDataTxAck'] * modulationInstance.getCharge('rxDataTxAck', moteList['nrSlots'] * slotLength, moteList['modulation'])
                # totalCharge += moteList['nrRxData'] * modulationInstance.getCharge('rxData', moteList['nrSlots'] * slotLength, moteList['modulation'])
                #
                # totalCharge += moteList['nrSleep'] * modulationInstance.getCharge('sleep', moteList['nrSlots'] * slotLength, moteList['modulation'])

                # this is the total charge for the whole length of the experiment
                # print 'totalCharge = %.4f' % totalCharge
                results.append({'exp': getExperimentName(exp), 'val': totalCharge, 'iteration': iteration, 'iterationDir': os.path.dirname(iteration), 'mote': moteID, 'ss': ss})

        # if iteration == '../output/results-test-kbit//sa_seed_8658_c_ILP30msFR_omega_8_i_1_p_0_motes_10_top_random-exp/output_cpu0.dat':
        #     assert False

    return results

def calculateAirtimePerModulation(consumption, exp, modulationInstance=None, ss=None):
    results = []
    for iteration in consumption:
        for moteID, consumptionPerState in consumption[iteration]['totalConsumption'].iteritems():
            # do not do this for root, only for source nodes
            # if moteID != '0':
                totalAirtime = 0

                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrIdle'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['idle'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrIdleNotSync'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['idleNotSync'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataRxAck'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataRxNack'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxNack'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxData'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txData'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrTxDataNoAck'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataNoAck'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrRxDataTxAck'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['rxDataTxAck'] # no airtime
                # iterate over all modulations for this state
                for modulation, modulationCount in consumptionPerState['nrRxData'].iteritems():
                    totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['rxData'] # no airtime

                # for modulation, modulationCount in consumptionPerState['nrIdle'].iteritems():
                #     totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime
                # for modulation, modulationCount in consumptionPerState['nrTxDataRxAck'].iteritems():
                #     totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime
                # # iterate over all modulations for this state
                # for modulation, modulationCount in consumptionPerState['nrTxDataRxNack'].iteritems():
                #     totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime
                # # iterate over all modulations for this state
                # for modulation, modulationCount in consumptionPerState['nrTxData'].iteritems():
                #     totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime
                # # iterate over all modulations for this state
                # for modulation, modulationCount in consumptionPerState['nrTxDataNoAck'].iteritems():
                #     totalAirtime += modulationCount * modulationInstance.modulationLengthPerSlotType[modulation]['txDataRxAck'] # no airtime

                # this is the total airtime for the whole length of the experiment
                results.append({'exp': getExperimentName(exp), 'val': totalAirtime, 'iteration': iteration, 'iterationDir': os.path.dirname(iteration), 'mote': moteID, 'ss': ss})

    return results


def calculateStateFrequencyPerModulation(consumption, exp, omega=None, modulationInstance=None):
    results = []
    startExperiment = None
    endExperiment = None
    for iteration in consumption:
        startExperiment = consumption[iteration]['startASN']
        endExperiment = consumption[iteration]['endASN']
        countIdle = 0
        countTxDataRxNoAck = 0
        countTxDataRXAck = 0
        countTxDataRxNack = 0
        countRxDataTxAck = 0
        for moteID, consumptionPerState in consumption[iteration]['totalConsumption'].iteritems():
            # do not do this for root, only for source nodes
            # if moteID != '0':
            totalCharge = 0
            totalNrActiveSlots = 0.0

            # iterate over all modulations for this state
            for modulation, modulationCount in consumptionPerState['nrIdle'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrIdle']})
                countIdle += modulationCount
            for modulation, modulationCount in consumptionPerState['nrIdleNotSync'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrIdleNotSync']})
            for modulation, modulationCount in consumptionPerState['nrTxDataRxAck'].iteritems(): # substract the NACKs!
                totalNrActiveSlots += (modulationCount - consumptionPerState['nrTxDataRxNack'][modulation]) * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': (modulationCount - consumptionPerState['nrTxDataRxNack'][modulation]), 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrTxDataRxAck']})
                countTxDataRXAck += modulationCount - consumptionPerState['nrTxDataRxNack'][modulation]
            for modulation, modulationCount in consumptionPerState['nrTxDataRxNack'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrTxDataRxNack']})
                countTxDataRxNack += modulationCount
            for modulation, modulationCount in consumptionPerState['nrTxData'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrTxData']})
            for modulation, modulationCount in consumptionPerState['nrTxDataNoAck'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrTxDataNoAck']})
                countTxDataRxNoAck += modulationCount
            for modulation, modulationCount in consumptionPerState['nrRxDataTxAck'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrRxDataTxAck']})# iterate over all modulations for this state
                countRxDataTxAck += modulationCount
            for modulation, modulationCount in consumptionPerState['nrRxData'].iteritems():
                totalNrActiveSlots += modulationCount * Modulation.Modulation().modulationSlots[modulationConfig][modulation]
                results.append({'exp': getExperimentName(exp), 'val': modulationCount, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrRxData']})  # iterate over all modulations for this state

            totalNrSleepSlots = (endExperiment - startExperiment) - totalNrActiveSlots
            results.append({'exp': getExperimentName(exp), 'val': totalNrSleepSlots, 'iteration': iteration, 'mote': moteID, 'state': translateStates['nrSleep']})  # iterate over all modulations for this state

        assert countIdle >= countTxDataRxNoAck
        assert countTxDataRXAck + countTxDataRxNack == countRxDataTxAck
        #     print iteration
        #     if iteration != '../output/results-2019-10-07-12-06-18//sa_seed_5421_c_ILP5msFR_omega_4_i_1_p_0_motes_10_top_random-exp/output_cpu0.dat':
        #         assert False
    # assert False

    return results

# Calculate the sum per state of all motes.
def calculateStateFrequency(data, exp):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # if moteList['mote'] != 0:
            totalFrequency = 0
            # do not do this for root, only for source nodes
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrIdle'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrIdle']})
            totalFrequency += moteList['nrIdle']
            # print 'nrIdle %d' % moteList['nrIdle']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrIdleNotSync'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrIdleNotSync']})
            totalFrequency += moteList['nrIdleNotSync']
            # print 'nrIdleNotSync %d' % moteList['nrIdleNotSync']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrSleep'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrSleep']})
            totalFrequency += moteList['nrSleep']
            # print 'nrSleep %d' % moteList['nrSleep']
            results.append({'exp': getExperimentName(exp), 'val': (moteList['nrTxDataRxAck'] - moteList['nrTxDataRxNack']), 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrTxDataRxAck']})
            totalFrequency += (moteList['nrTxDataRxAck'] - moteList['nrTxDataRxNack'])
            # print 'nrTxDataRxAck %d' % (moteList['nrTxDataRxAck'] - moteList['nrTxDataRxNack'])
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrTxDataRxNack'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrTxDataRxNack']})
            totalFrequency += moteList['nrTxDataRxNack']
            # print 'nrTxDataRxNack %d' % moteList['nrTxDataRxNack']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrTxDataNoAck'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrTxDataNoAck']})
            totalFrequency += moteList['nrTxDataNoAck']
            # print 'nrTxDataNoAck %d' % moteList['nrTxDataNoAck']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrRxDataTxAck'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrRxDataTxAck']})
            totalFrequency += moteList['nrRxDataTxAck']
            # print 'nrRxDataTxAck %d' % moteList['nrRxDataTxAck']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrTxData'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrTxData']})
            totalFrequency += moteList['nrTxData']
            # print 'nrTxData %d' % moteList['nrTxData']
            results.append({'exp': getExperimentName(exp), 'val': moteList['nrRxData'], 'iteration': iteration, 'mote': moteList['mote'], 'state': translateStates['nrRxData']})
            totalFrequency += moteList['nrRxData']
            # print 'nrRxData %d' % moteList['nrRxData']
            # print 'Mote %d, total number of time slots: %d' % (moteList['mote'], totalFrequency)

    return results

# Get all charges of all motes.
def calculateLatency(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['pktLatencies'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['pktLatencies'] * (modulationInstance.modulationConfigSlotLength[modulationConfig] / 1000.0), 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculatePktGenPerMote(data, exp, nrMotes=None, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        nrPktGen = 0
        for moteList in data[iteration]:
            if moteList['mote'] != 0 and moteList['pktGen'] is not None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['pktGen'], 'iteration': iteration, 'mote': moteList['mote'], 'motes': translateMotes[nrMotes], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

# Get hopcount of all motes.
def calculateHopCount(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['reachRoot'] == 1:
                results.append({'exp': getExperimentName(exp), 'val': moteList['hopcount'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

# Get hopcount of all motes.
def calculateMinimalHopCount(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['reachRoot'] == 1:
                results.append({'exp': getExperimentName(exp), 'val': moteList['minimalHopcount'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

# Get hopcount of all motes.
def calculateReachRoot(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0:
                results.append({'exp': getExperimentName(exp), 'val': moteList['reachRoot'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateChildren(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0:
                results.append({'exp': getExperimentName(exp), 'val': moteList['children'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateMACDrops(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['pktDropsMac'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['pktDropsMac'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateQueueDrops(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['pktDropsQueue'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['pktDropsQueue'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateAllDrops(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        allDrops = 0
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0:
                qDrops = moteList['pktDropsQueue']
                if qDrops is None:
                    qDrops = 0
                qMAC = moteList['pktDropsMac']
                if qMAC is None:
                    qMAC = 0
                allDrops += (qDrops + qMAC)
        results.append({'exp': getExperimentName(exp), 'val': allDrops, 'iteration': iteration, 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateDAO(data, exp):
    results = []
    for iteration in data:
        # reset for this experiment
        for moteList in data[iteration]:
            if moteList['mote'] != 0:
                # do not do this for root, only for source nodes
                results.append({'exp': getExperimentName(exp), 'val': moteList['activeDAO'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'activeDAO'})
                results.append({'exp': getExperimentName(exp), 'val': moteList['initiatedDAO'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'initiatedDAO'})
    return results

def calculateDAOReceived(data, exp):
    results = []
    for iteration in data:
        # reset for this experiment
        for moteList in data[iteration]:
            if moteList['mote'] == 0:
                results.append({'exp': getExperimentName(exp), 'val': moteList['receivedDAO'], 'iteration': iteration, 'type': 'receivedDAO'})
    return results

def calculateSixTopMessaging(data, exp):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            if moteList['mote'] != 0:
                results.append({'exp': getExperimentName(exp), 'val': moteList['sixtopTxAddReq'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'sixtopTxAddReq'})
                results.append({'exp': getExperimentName(exp), 'val': moteList['sixtopTxAddResp'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'sixtopTxAddResp'})
                results.append({'exp': getExperimentName(exp), 'val': moteList['sixtopTxDelReq'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'sixtopTxDelReq'})
                results.append({'exp': getExperimentName(exp), 'val': moteList['sixtopTxDelResp'], 'iteration': iteration, 'mote': moteList['mote'], 'type': 'sixtopTxDelResp'})
    return results

def calculateDistances(data, exp, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['distance'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['distance'], 'iteration': iteration, 'mote': moteList['mote'], 'ss': ss})
    return results

def calculateModulationPerLink(data, exp, ss=None, modulationConfig=None):
    results = []
    all_modulations = ['QPSK_FEC_3_4', 'QPSK_FEC_1_2', 'QPSK_FEC_1_2_FR_2']
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['reachRoot'] == 1 and moteList['modulation'] is not None:
                results.append({'exp': getExperimentName(exp), 'val': 1, 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[moteList['modulation']], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})
                modulations_cpy = all_modulations[:]
                modulations_cpy.remove(moteList['modulation'])
                for m in modulations_cpy: # add these, so in the end when you add (and average) all values per iterations, you sure have entries for all modulations per iteration
                    results.append({'exp': getExperimentName(exp), 'val': 0, 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[m], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})

    return results

def calculateModulationPerBondedSlot(data, exp, ss=None, modulationConfig=None):
    results = []
    all_modulations = ['QPSK_FEC_3_4', 'QPSK_FEC_1_2', 'QPSK_FEC_1_2_FR_2']
    for iteration in data:
        for moteList in data[iteration]:
            if moteList['mote'] != 0:
                if moteList['modulation'] is not None:
                    results.append({'exp': getExperimentName(exp), 'val': moteList['allocatedBondedSlots'], 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[moteList['modulation']], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})
                modulations_cpy = all_modulations[:]
                if moteList['modulation'] is not None:
                    modulations_cpy.remove(moteList['modulation'])
                for m in modulations_cpy: # add these, so in the end when you add (and average) all values per iterations, you sure have entries for all modulations per iteration
                    results.append({'exp': getExperimentName(exp), 'val': 0, 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[m], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})
    return results

def calculateModulationPerBondedSlotAirtime(data, exp, ss=None, modulationConfig=None):
    results = []
    all_modulations = ['QPSK_FEC_3_4', 'QPSK_FEC_1_2', 'QPSK_FEC_1_2_FR_2']
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0:
                if moteList['modulation'] is not None:
                    results.append({'exp': getExperimentName(exp), 'val': modulationInstance.modulationLengthPerSlotType[moteList['modulation']]['txDataRxAck'] * moteList['allocatedBondedSlots'], 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[moteList['modulation']], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})
                modulations_cpy = all_modulations[:]
                if moteList['modulation'] is not None:
                    modulations_cpy.remove(moteList['modulation'])
                for m in modulations_cpy: # add these, so in the end when you add (and average) all values per iterations, you sure have entries for all modulations per iteration
                    results.append({'exp': getExperimentName(exp), 'val': modulationInstance.modulationLengthPerSlotType[m]['txDataRxAck'] * 0, 'iteration': iteration, 'mote': moteList['mote'], 'modulation': translateModulations[m], 'ss': ss, 'modulationConfig': translateConfigModulations[modulationConfig]})

    return results

def calculateNrSlots(data, exp, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['distance'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['nrSlots'], 'iteration': iteration, 'mote': moteList['mote'], 'ss': ss})
    return results

def calculateTotalPropagationData(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['totalPropagationData'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['totalPropagationData'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateSuccessPropagationData(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['successPropagationData'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['successPropagationData'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig],  'ss': ss})
    return results

def calculateInterferenceFailures(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['interferenceFailures'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['interferenceFailures'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateInterferenceLockFailures(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['interferenceLockFailures'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['interferenceLockFailures'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateSignalFailures(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            # if moteList['mote'] != 0 and moteList['signalFailures'] != None:
            results.append({'exp': getExperimentName(exp), 'val': moteList['signalFailures'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateAllInterferers(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['allInterferers'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['allInterferers'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateHadInterferers(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            if moteList['mote'] != 0 and moteList['hadInterferers'] != None:
                results.append({'exp': getExperimentName(exp), 'val': moteList['hadInterferers'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateReliability(data, exp, modulationConfig=None, ss=None, topConfig=None, deltaConfig=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            # and only do this if there is path from the node to the root and each node on this path has at least 1 allocated slot
            if moteList['mote'] != 0 and moteList['reliability'] != None and moteList['reachRoot'] == 1:
                results.append({'exp': getExperimentName(exp), 'val': moteList['reliability'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss, 'topConfig': topConfig, 'deltaConfig': deltaConfig})
    return results

def calculateAllocatedBondedSlots(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            # and only do this if there is path from the node to the root and each node on this path has at least 1 allocated slot
            if moteList['mote'] != 0 and moteList['allocatedBondedSlots'] != None and moteList['reachRoot'] == 1:
                results.append({'exp': getExperimentName(exp), 'val': moteList['allocatedBondedSlots'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def calculateAllocatedRegularSlots(data, exp, modulationConfig=None, ss=None):
    results = []
    for iteration in data:
        for moteList in data[iteration]:
            # do not do this for root, only for source nodes
            # and only do this if there is path from the node to the root and each node on this path has at least 1 allocated slot
            if moteList['mote'] != 0 and moteList['allocatedRegularSlots'] != None and moteList['reachRoot'] == 1:
                results.append({'exp': getExperimentName(exp), 'val': moteList['allocatedRegularSlots'], 'iteration': iteration, 'mote': moteList['mote'], 'modulationConfig': translateConfigModulations[modulationConfig], 'ss': ss})
    return results

def getPercentileValue(q, data):

    return True

def plotBoxplotSeaborn(metric, data, sorter, outputDir='', xlabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []

    data.exp = data.exp.astype("category")
    data.exp.cat.set_categories(sorter, inplace=True)
    data = data.sort_values(["exp"])

    meanDataframe = data.groupby(['exp'])['val'].mean().reset_index()
    meanDataframe.exp = meanDataframe.exp.astype("category")
    meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    meanDataframe = meanDataframe.sort_values(["exp"])

    # results = []
    # boxplotsMeans = []
    # for dat in boxplots:
    #     iqr = np.percentile(dat, 75) - np.percentile(dat, 25)
    #     datMean = []
    #     for elem in dat:
    #         if elem >= (np.percentile(dat, 25) - iqr * 1.5) and elem <= (np.percentile(dat, 75) + iqr * 1.5):
    #             datMean.append(elem)
    #     boxplotsMeans.append(np.mean(datMean))
    #     # result = {'exp': }


    axBoxplot = sns.boxplot(x='exp', y='val', data=data, palette=pal, width=0.3, showfliers=False, showmeans=False)
    ax = sns.scatterplot(x='exp', y='val', data=meanDataframe, palette=pal, marker='x', size=3, linewidth=2, color='#303030')
    axBoxplot.tick_params(labelsize=5)
    ax.legend_.remove()
    ax.set_xlabel('')
    axBoxplot.set_xlabel('')
    sns.despine()
    name = '{0}/boxplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    if xlabel is not None:
        plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotBoxplotSeabornThroughput(metric, data, sorter, outputDir='', xlabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []

    data.exp_left = data.exp_left.astype("category")
    data.exp_left.cat.set_categories(sorter, inplace=True)
    data = data.sort_values(["exp_left"])

    meanDataframe = data.groupby(['exp_left'])['throughput'].mean().reset_index()
    meanDataframe.exp_left = meanDataframe.exp_left.astype("category")
    meanDataframe.exp_left.cat.set_categories(sorter, inplace=True)
    meanDataframe = meanDataframe.sort_values(["exp_left"])

    # results = []
    # boxplotsMeans = []
    # for dat in boxplots:
    #     iqr = np.percentile(dat, 75) - np.percentile(dat, 25)
    #     datMean = []
    #     for elem in dat:
    #         if elem >= (np.percentile(dat, 25) - iqr * 1.5) and elem <= (np.percentile(dat, 75) + iqr * 1.5):
    #             datMean.append(elem)
    #     boxplotsMeans.append(np.mean(datMean))
    #     # result = {'exp': }


    axBoxplot = sns.boxplot(x='exp_left', y='throughput', data=data, palette=pal, width=0.3, showfliers=False, showmeans=False)
    ax = sns.scatterplot(x='exp_left', y='throughput', data=meanDataframe, palette=pal, marker='x', size=3, linewidth=2, color='#303030')
    axBoxplot.tick_params(labelsize=5)
    ax.legend_.remove()
    ax.set_xlabel('')
    axBoxplot.set_xlabel('')
    sns.despine()
    name = '{0}/boxplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    if xlabel is not None:
        plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plotLineSeaborn(metric, data, sorter, outputDir='', xlabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []

    data.exp = data.exp.astype("category")
    data.exp.cat.set_categories(sorter, inplace=True)
    data = data.sort_values(["exp"])

    meanDataframe = data.groupby(['exp'])['val'].mean().reset_index()
    meanDataframe.exp = meanDataframe.exp.astype("category")
    meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    meanDataframe = meanDataframe.sort_values(["exp"])

    # results = []
    # boxplotsMeans = []
    # for dat in boxplots:
    #     iqr = np.percentile(dat, 75) - np.percentile(dat, 25)
    #     datMean = []
    #     for elem in dat:
    #         if elem >= (np.percentile(dat, 25) - iqr * 1.5) and elem <= (np.percentile(dat, 75) + iqr * 1.5):
    #             datMean.append(elem)
    #     boxplotsMeans.append(np.mean(datMean))
    #     # result = {'exp': }


    axLineplot = sns.lineplot(x='exp', y='val', data=data, err_style="bars", palette=pal)
    axLineplot.tick_params(labelsize=5)
    axLineplot.set_xlabel('')
    sns.despine()
    name = '{0}/lineplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    if xlabel is not None:
        plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotMultipleBoxplotSeaborn(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
    plt.figure(figsize=(6, 3))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []
    pd.set_option('display.max_colwidth', -1)

    newData = pd.DataFrame()

    if percentile is not None:
        expLabels = np.unique(data['exp'].values)
        pcentiles = {}
        for expLabel in expLabels:
            pcentiles[expLabel] = np.percentile(data[data.exp == expLabel]['val'].values, percentile)

        # filter the data
        for expLabel, pcentile in pcentiles.iteritems():
            tmpData = data[data.exp == expLabel]
            tmpData = tmpData[tmpData.val < pcentile]
            newData = newData.append(tmpData)
    else:
        newData = newData.append(data)

    newData.exp = newData.exp.astype("category")
    newData.exp.cat.set_categories(sorter, inplace=True)
    newData = newData.sort_values(["exp"])

    print 'METRIC: {0}'.format(metric)
    meanDataframe = newData.groupby(['exp'])['val'].mean().reset_index()
    stdDataframe = newData.groupby(['exp'])['val'].std().reset_index()
    print 'meanDataframe: {0}'.format(meanDataframe)
    # meanDataframe.exp = meanDataframe.exp.astype("category")
    # meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    # meanDataframe = meanDataframe.sort_values(["exp"])

    output_file = "{0}/results.txt".format(outputDir)
    with open(output_file, "a") as myfile:
        myfile.write("{0} - mean - std\n".format(metric))
        merged_mean_std = pd.merge(meanDataframe, stdDataframe, on='exp', suffixes=('_mean', '_std'))
        myfile.write("{0}\n\n".format(merged_mean_std))

    axBoxplot = sns.boxplot(x=xsplit, y='val', data=newData, palette=pal, width=0.7, linewidth=1, showfliers=True, showmeans=True, hue=hue, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
    # ax = sns.scatterplot(x=xsplit, y='val', data=meanDataframe, palette=pal, marker='x', size=3, linewidth=2, color='#303030', hue='exp')
    axBoxplot.tick_params(labelsize=10)
    # axBoxplot.set(ylim=(0,None))
    # ax.legend_.remove()
    # ax.set_xlabel('')
    axBoxplot.set_xlabel('')
    sns.despine(ax=axBoxplot)
    name = '{0}/multiple-boxplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    axBoxplot.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plotMultipleBoxplotSeabornThroughput(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
    plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []
    pd.set_option('display.max_colwidth', -1)

    newData = pd.DataFrame()

    # if percentile is not None:
    #     expLabels = np.unique(data['exp'].values)
    #     pcentiles = {}
    #     for expLabel in expLabels:
    #         pcentiles[expLabel] = np.percentile(data[data.exp == expLabel]['val'].values, percentile)
    #
    #     # filter the data
    #     for expLabel, pcentile in pcentiles.iteritems():
    #         tmpData = data[data.exp == expLabel]
    #         tmpData = tmpData[tmpData.val < pcentile]
    #         newData = newData.append(tmpData)
    # else:
    newData = newData.append(data)
    newData['throughput'] = newData['throughput'].apply(lambda x: x*100.0)

    newData.exp_left = newData.exp_left.astype("category")
    newData.exp_left.cat.set_categories(sorter, inplace=True)
    newData = newData.sort_values(["exp_left"])

    print metric
    meanDataframe = newData.groupby(['exp_left'])['throughput'].mean().reset_index()
    print meanDataframe
    # meanDataframe.exp = meanDataframe.exp.astype("category")
    # meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    # meanDataframe = meanDataframe.sort_values(["exp"])

    axBoxplot = sns.boxplot(x=xsplit, y='throughput', data=newData, palette=pal, width=0.7, linewidth=1, showfliers=True, showmeans=True, hue=hue, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
    # ax = sns.scatterplot(x=xsplit, y='val', data=meanDataframe, palette=pal, marker='x', size=3, linewidth=2, color='#303030', hue='exp')
    axBoxplot.tick_params(labelsize=10)
    axBoxplot.set(ylim=(0,None))
    # ax.legend_.remove()
    # ax.set_xlabel('')
    axBoxplot.set_xlabel('')
    sns.despine(ax=axBoxplot)
    name = '{0}/multiple-boxplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    axBoxplot.legend(loc='best', frameon=False, fontsize='x-small')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plotMultipleBoxplotSeabornReliability(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
    plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []
    pd.set_option('display.max_colwidth', -1)

    # newData = pd.DataFrame()
    # 
    # newData = newData.append(data)
    # newData['throughput'] = newData['throughput'].apply(lambda x: x*100.0)
    # 
    # newData.exp_left = newData.exp_left.astype("category")
    # newData.exp_left.cat.set_categories(sorter, inplace=True)
    # newData = newData.sort_values(["exp"])

    # print metric
    # meanDataframe = newData.groupby(['exp_left'])['throughput'].mean().reset_index()
    # print meanDataframe

    # meanDataframe.exp = meanDataframe.exp.astype("category")
    # meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    # meanDataframe = meanDataframe.sort_values(["exp"])

    axBoxplot = sns.boxplot(x=xsplit, y='val', data=data, palette=pal, width=0.7, linewidth=1, showfliers=True, showmeans=True, hue=hue, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
    # ax = sns.scatterplot(x=xsplit, y='val', data=meanDataframe, palette=pal, marker='x', size=3, linewidth=2, color='#303030', hue='exp')
    axBoxplot.tick_params(labelsize=10)
    # axBoxplot.set(ylim=(0,None))
    # ax.legend_.remove()
    # ax.set_xlabel('')
    axBoxplot.set_xlabel('')
    sns.despine(ax=axBoxplot)
    name = '{0}/multiple-boxplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    # axBoxplot.legend(loc='best', frameon=False, fontsize='x-small')
    axBoxplot.legend(frameon=False, fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotMultipleLineSeabornThroughput(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
    plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []
    pd.set_option('display.max_colwidth', -1)

    newData = pd.DataFrame()
    newData = newData.append(data)
    # newData['throughput'] = newData['throughput'].apply(lambda x: x*100.0)
    newData.exp_left = newData.exp_left.astype("category")
    newData.exp_left.cat.set_categories(sorter, inplace=True)
    newData = newData.sort_values(["exp_left"])
    meanDataframe = newData.groupby(['exp_left'])['throughput'].mean().reset_index()
    # print metric
    # print meanDataframe
    stdDataframe = newData.groupby(['exp_left'])['throughput'].std().reset_index()
    # print stdDataframe
    mean_std_merge_df = pd.merge(meanDataframe, stdDataframe, on='exp_left', suffixes=('_mean', '_std'))
    mean_std_merge_df.to_csv("{0}/../throughput.txt".format(outputDir), sep='\t')

    axLineplot = sns.lineplot(x=xsplit, y='throughput', data=newData, hue=hue, err_style="bars", style=hue, markers=False, dashes=True, ci='sd')
    axLineplot.tick_params(labelsize=10)
    axLineplot.set(ylim=(0,None))
    # ax.legend_.remove()
    # ax.set_xlabel('')
    axLineplot.set_xlabel('')
    sns.despine(ax=axLineplot)
    name = '{0}/multiple-lineplot-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    plt.xlabel(xlabel)
    # plt.legend(loc='upper left')
    handles, labels = axLineplot.get_legend_handles_labels()
    axLineplot.legend(handles=handles[1:], labels=labels[1:], loc='best', frameon=False, fontsize='small', labelspacing=0.3)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plotScatterParetoFront(metricX=None, xfilemetric=None, xaxis=None, metricY=None, yfilemetric=None, yaxis=None, data=None, hue=None, outputDir=None):
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 2.5))

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    sctter = sns.scatterplot(x=xaxis, y=yaxis, hue=hue, style=hue, data=data, markers=filled_markers, s=80)
    sns.despine()

    # ax.legend()
    # ax.grid(True)
    name = '{0}/scatter-{1}-{2}-{3}.eps'.format(outputDir, xfilemetric, yfilemetric,
                                                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.xlabel(metricX)
    plt.ylabel(metricY)

    handles, labels = sctter.get_legend_handles_labels()
    # print labels
    labels = labels[1:]
    handles = handles[1:]
    labels = [label.split(', ')[1] for label in labels]
    # print labels
    # ax.legend()
    legend = sctter.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), ncol=1, frameon=False, handles=handles, labels=labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotScatterParetoFrontMILPAndSimulationThroughput(metricX=None, xfilemetric=None, xaxis=None, metricY=None, yfilemetric=None, yaxis=None, data=None, hue=None, outputDir=None):
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 2.5))

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    sctter = sns.scatterplot(x=xaxis, y=yaxis, data=data, s=80)
    sctter.set_ylim(0, 1.04)
    sctter.set_xlim(0, 1.02)

    lims = [
        np.min([sctter.get_xlim(), sctter.get_ylim()]),  # min of both axes
        np.max([sctter.get_xlim(), sctter.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    sns.lineplot(lims, lims, alpha=0.4, zorder=0)

    sns.despine()
    # ax.legend()
    # ax.grid(True)
    name = '{0}/scatter-{1}-{2}-{3}.eps'.format(outputDir, xfilemetric, yfilemetric,
                                                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.xlabel(metricX)
    plt.ylabel(metricY)

    handles, labels = sctter.get_legend_handles_labels()
    # print labels
    labels = labels[1:]
    handles = handles[1:]
    labels = [label.split(', ')[1] for label in labels]
    # print labels
    # ax.legend()
    # sctter.get_legend().remove()
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotScatterParetoFrontFullLegend(metricX=None, xfilemetric=None, xaxis=None, metricY=None, yfilemetric=None, yaxis=None, data=None, hue=None, outputDir=None):
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 2.5))

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    sctter = sns.scatterplot(x=xaxis, y=yaxis, hue=hue, style=hue, data=data, markers=filled_markers, s=80)
    sns.despine()

    # ax.legend()
    # ax.grid(True)
    name = '{0}/scatter-full-{1}-{2}-{3}.eps'.format(outputDir, xfilemetric, yfilemetric,
                                                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.xlabel(metricX)
    plt.ylabel(metricY)

    handles, labels = sctter.get_legend_handles_labels()
    # print labels
    labels = labels[1:]
    handles = handles[1:]
    # labels = [label.split(', ')[1] for label in labels]
    # print labels
    # ax.legend()
    legend = sctter.legend(frameon=False, handles=handles, labels=labels, fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotCDFSeaborn(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
    fig, ax = plt.subplots()
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []
    pd.set_option('display.max_colwidth', -1)

    newData = pd.DataFrame()

    expLabels = np.unique(data['exp'].values)
    if percentile is not None:
        pcentiles = {}
        for expLabel in expLabels:
            pcentiles[expLabel] = np.percentile(data[data.exp == expLabel]['val'].values, percentile)

        # filter the data
        for expLabel, pcentile in pcentiles.iteritems():
            tmpData = data[data.exp == expLabel]
            tmpData = tmpData[tmpData.val < pcentile]
            newData = newData.append(tmpData)
    else:
        newData = newData.append(data)

    newData.exp = newData.exp.astype("category")
    newData.exp.cat.set_categories(sorter, inplace=True)
    newData = newData.sort_values(["exp"])

    print metric
    meanDataframe = newData.groupby(['exp'])['val'].mean().reset_index()
    print meanDataframe


    # axBoxplot = sns.boxplot(x=xsplit, y='val', data=newData, palette=pal, width=0.7, linewidth=1, showfliers=True, showmeans=True, hue=hue, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
    for eLabel in expLabels:
        lbl = 'unknown'
        if 'oneslot' in eLabel:
            lbl = '30 ms'
        elif 'variable5' in eLabel:
            lbl = '5 ms'
        elif 'variable10' in eLabel:
            lbl = '10 ms'
        sns.distplot(newData[newData.exp == eLabel]['val'], kde_kws={'cumulative': True}, label=lbl)
    ax.tick_params(labelsize=10)
    # axCDF.set(ylim=(0,None))
    # ax.legend_.remove()
    # ax.set_xlabel('')
    ax.set_xlabel('')
    sns.despine(ax=ax)
    name = '{0}/cdf-{1}-{2}.eps'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.xlabel(getLabelName(metric))
    # plt.legend(loc='upper left')
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plotTimeSeriesSeaborn(metric, data, outputDir='', xlabel=None):


    for experiment, experimentData in data.iteritems():
        plt.figure(figsize=(6, 3))
        sns.set_style('white')
        pal = sns.color_palette('colorblind')
        sns.lineplot(x='cycle', y=metric, hue='iteration', data=experimentData, legend=False)
        name = '{0}/timeseries-{1}-{2}-{3}.eps'.format(outputDir, metric, experiment, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        plt.xlabel(getLabelName(metric))
        plt.tight_layout()
        plt.savefig(name)
        plt.close()

def plotMultipleBoxplotSeaborn2(metric, data, outputDir='', xsplit=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')

    global mapping
    global colors
    boxplots = []

    boxplotsMeans = []
    for dat in boxplots:
        iqr = np.percentile(dat, 75) - np.percentile(dat, 25)
        datMean = []
        for elem in dat:
            if elem >= (np.percentile(dat, 25) - iqr * 1.5) and elem <= (np.percentile(dat, 75) + iqr * 1.5):
                datMean.append(elem)
        boxplotsMeans.append(np.mean(datMean))

    # print boxplotsMeans

    sns.boxplot(x=xsplit, y='val', data=data, palette=pal, hue='exp', width=0.3)
    sns.despine()
    name = '{0}/boxplot-{1}-{2}.pdf'.format(outputDir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(getLabelName(metric))
    # plt.legend(loc='upper left')
    plt.savefig(name)
    plt.close()

def plotBarsSeaborn(yMetric, data, outputDir='', xsplit=None, hue=None, ylabel=None):
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    # assert False
    b = sns.barplot(x=xsplit, y='val', data=data, hue=hue)
    name = '{0}/bars-{1}-{2}.pdf'.format(outputDir, yMetric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    b.tick_params(labelsize=6)

    b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.ylabel(ylabel)

    l = ax.legend()

    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plotBarsSeabornThroughput(outputDir=''):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    #             10 ms           30 ms
    #
    # 150 ms      88.318698 (9.742065)    70.803646 (1.396646)
    # 210 ms      98.558835 (2.442315)    95.061554 (7.312867)
    # 300 ms      99.780577 (0.548090)    98.856501 (3.557211)

    # data = {}
    # data['150'] = {'10ms': {'val': 88.318698, 'std': 9.742065}, '30ms': {'val': 70.803646, 'std': 1.396646}}
    # data['210'] = {'10ms': {'val': 98.558835, 'std': 2.442315}, '30ms': {'val': 95.061554, 'std': 7.312867}}
    # data['300'] = {'10ms': {'val': 99.780577, 'std': 0.548090}, '30ms': {'val': 70.803646, 'std': 3.557211}}

    data = [['150', '10 ms bonded slots', 88.318698, 9.742065], ['150', '30 ms slots', 70.803646, 1.396646],
            ['210', '10 ms bonded slots', 98.558835, 2.442315], ['210', '30 ms slots', 95.061554, 7.312867],
            ['300', '10 ms bonded slots', 99.780577, 0.548090], ['300', '30 ms slots', 98.856501, 3.557211]]

    df = pd.DataFrame(data, columns=['slotframeSize', 'slotLength', 'val', 'std'])

    # assert False
    b = sns.barplot(x='slotframeSize', y='val', data=df, hue='slotLength')
    name = '{0}/bars-{1}-{2}.pdf'.format(outputDir, 'manual-throughputs', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    legend = b.legend(frameon=False, fontsize=9)

    # b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.ylabel('Throughput (%)')
    plt.xlabel('Slot frame size (ms)')

    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plotBarsSeabornMean(yMetric, data, outputDir='', xsplit=None, hue=None, ylabel=None):
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    # assert False
    b = sns.barplot(x=xsplit, y='val', data=data, hue=hue, errwidth=1.0)
    name = '{0}/bars-{1}-{2}.pdf'.format(outputDir, yMetric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    b.tick_params(labelsize=6)

    b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.ylabel(ylabel)

    l = ax.legend()

    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plotBarsSeabornMeanModulations(yMetric, data, outputDir='', xsplit=None, hue=None, ylabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    # assert False
    b = sns.barplot(x=xsplit, y='val', data=data, hue=hue, errwidth=1.0, ci='sd')
    name = '{0}/bars-{1}-{2}.eps'.format(outputDir, yMetric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    b.tick_params(labelsize=20)
    b.set_xticklabels(b.get_xticklabels(), horizontalalignment='right')
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(8, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel('Slotframe length (ms)', fontsize=20)
    # ax.set(ylim=(-1, 10))
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.split(' (')[0] for label in labels]
    l = ax.legend(handles, labels, loc='best', frameon=False, fontsize=20)

    sns.despine()
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plotBarsSeabornMeanModulationsVariableY(yMetric, data, outputDir='', xsplit=None, hue=None, ylabel=None, outputLabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    # assert False
    b = sns.barplot(x=xsplit, y=yMetric, data=data, hue=hue, errwidth=1.0, ci='sd')
    name = '{0}/bars-{1}-{2}.eps'.format(outputDir, outputLabel, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    b.tick_params(labelsize=16)
    b.set_xticklabels(b.get_xticklabels(), horizontalalignment='right')
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(8, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel('Slotframe length (ms)', fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    labels = [label.split(' (')[0] for label in labels]
    l = ax.legend(handles, labels, loc='best', frameon=False, fontsize=14)

    sns.despine()
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plotBarsSeabornMeanPolicies(yMetric, tens_ms, thirty_ms, outputDir='', xsplit=None, hue=None, ylabel=None):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(6, 2.5))

    b = sns.barplot(x=xsplit, y='val', data=tens_ms, hue=hue, errwidth=1.0, ax=axs[0][0], ci=68)
    labels_b = [item.get_text() for item in b.get_xticklabels()]
    # labels_b_new = [label.split('_')[-2] for label in labels_b]
    b.set_xticklabels(labels_b, horizontalalignment='right')
    b.set(xlabel='10 ms bonded slots', ylabel='Links')

    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(17, 0)
    for t in axs[0][0].get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    # print tens_ms

    c = sns.barplot(x=xsplit, y='val', data=thirty_ms, hue=hue, errwidth=1.0, ax=axs[0][1], ci=68)
    labels_c = [item.get_text() for item in c.get_xticklabels()]
    # labels_c_new = [label.split('_')[-2] for label in labels_c]
    c.set_xticklabels(labels_c, horizontalalignment='right')
    c.set(xlabel='40 ms slots', ylabel='')

    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(17, 0)
    for t in axs[0][1].get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    # print '--------------------1'
    # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
    #                        None):  # more options can be specified also
    #     print thirty_ms.to_string()


    axs[0][0].set_ylim(0, 9.0)
    axs[0][1].set_ylim(0, 9.0)

    handles, labels = axs[0][0].get_legend_handles_labels()
    labels = [label.split(' (')[0] for label in labels]

    axs[0][0].get_legend().remove()
    axs[0][1].legend(handles, labels, loc='best', bbox_to_anchor=(0.67, 0.62), frameon=False, fontsize=6.5)

    name = '{0}/bars-mcs-policies-{1}-{2}.eps'.format(outputDir, yMetric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    sns.despine()
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

    # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
    #                        None):  # more options can be specified also
    #     print data_one.to_string()
    # print '--------------------'
    # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
    #                        None):  # more options can be specified also
    #     print data_two.to_string()

    # assert False

def plotHistogram(xMetric, data, outputDir=''):
    global mapping
    global colors

    # print data

    fig, ax = plt.subplots()

    # assert False
    b = sns.countplot(data);
    name = '{0}/histogram-{1}-{2}.pdf'.format(outputDir, xMetric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    b.tick_params(labelsize=4)

    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def writeData(metric, data):
    print translate[metric]
    for exp, dat in sorted(data.items(), key=lambda x: x[0]):
        if exp in fileTranslateWrite:
            print '{exp}: {data}'.format(exp=fileTranslateWrite[exp],data=np.mean(dat))
        else:
            print '{exp}: {data}'.format(exp=exp, data=np.mean(dat))

def getMetricY(metric):
    if metric == 'all':
        return metrics
    else:
        return [metric]

if __name__ == '__main__':
    data = collections.OrderedDict()
    consumption = collections.OrderedDict()
    links = collections.OrderedDict()
    throughputILP = collections.OrderedDict()
    timeData = collections.OrderedDict()
    resultType = str(sys.argv[1])
    resultTypes = ['paper', 'paper-heuristic', 'paper-modulation-analysis', 'paper-ilp-throughput', 'paper-airtime', 'distance', 'modulation-distance', 'modulation-omega', 'modulation-slotframe', 'cdf', 'time', 'other', 'manualplot', 'modulation-ss', 'modulation-ilp', 'modulation-analysis']
    if resultType not in resultTypes:
        assert False

    outputDir = 'plots-%s' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outputDirAll = '%s/individual' % outputDir
    outputDirPerIteration = '%s/network' % outputDir
    try:
        os.makedirs(outputDir)
        os.makedirs(outputDirAll)
        os.makedirs(outputDirPerIteration)
    except OSError:
        if not os.path.isdir(outputDir):
            raise

    # dataDir = str(sys.argv[2])
    dataDirs = str(sys.argv[2]).split(',')
    experimentTypes = str(sys.argv[3]).split(',')

    # aggregated over all motes
    lifetime2000OpenMoteCC2538 = pd.DataFrame()
    lifetime2000OpenMoteB = pd.DataFrame()
    stateFrequency = pd.DataFrame()
    stateFrequencySum = pd.DataFrame()
    sixtopMessaging = pd.DataFrame()
    DAOMessaging = pd.DataFrame()
    DAOReceived = pd.DataFrame()
    airtimePerMotePerModulation = pd.DataFrame()
    chargePerMotePerModulation = pd.DataFrame()
    chargePerMoteOpenMoteCC2538 = pd.DataFrame()
    chargePerMoteOpenMoteB = pd.DataFrame()
    latency = pd.DataFrame()
    macDrops = pd.DataFrame()
    queueDrops = pd.DataFrame()
    pktGenerated = pd.DataFrame()
    pktGeneratedPerMote = pd.DataFrame()
    pktArrivedToGenerated = pd.DataFrame()
    allDrops = pd.DataFrame()
    distances = pd.DataFrame()
    modulations = pd.DataFrame()
    modulationsPerBondedSlot = pd.DataFrame()
    modulationsPerBondedSlotAirtime = pd.DataFrame()
    allNrSlots = pd.DataFrame()
    hopcount = pd.DataFrame()
    minimalHopcount = pd.DataFrame()
    children = pd.DataFrame()
    reliability = pd.DataFrame()
    allocatedBondedSlots = pd.DataFrame()
    allocatedRegularSlots = pd.DataFrame()
    nrReachRoot = pd.DataFrame()

    totalPropagationData = pd.DataFrame()
    successPropagationData = pd.DataFrame()
    interferenceFailures = pd.DataFrame()
    interferenceLockFailures = pd.DataFrame()
    signalFailures = pd.DataFrame()
    allInterferers = pd.DataFrame()
    hadInterferers = pd.DataFrame()

    ilp_throughput = pd.DataFrame()

    # per iteration
    received = pd.DataFrame()
    bitsPerJouleOpenMoteCC2538 = pd.DataFrame()
    bitsPerJouleOpenMoteB = pd.DataFrame()

    slotData = pd.DataFrame()

    sorter = []
    sorterMode = []

    links = {}
    parsedLinks = False

    # for ix in range(3, len(sys.argv)):
    for experimentType in experimentTypes:
        nrMotes = None
        distance = None
        omega = None
        slotframeSize = None
        modulationConfig = None
        topologyConfig = None
        deltaConfig = None
        # slotframeSize = 100
        # modulationConfig = 'ILP10msFR'

        if resultType == 'distance':
            rgx = '%s_([A-Za-z0-9]+)_' % 'distance'
            distance = list(get_set_rgx([experimentType], rgx))
            if len(distance) == 0:
                distance = None
                print 'Variable \'distance\' is None.'
            else:
                distance = list(get_set_rgx([experimentType], rgx))[0]
                print distance
            rgx = '^_%s_([A-Za-z0-9]+)_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'modulation-distance' or resultType == 'modulation-omega':
            # rgx = '%s_([A-Za-z0-9]+)_' % 'distance'
            # distance = list(get_set_rgx([experimentType], rgx))
            # if len(distance) == 0:
            #     distance = None
            #     print 'Variable \'distance\' is None.'
            # else:
            #     distance = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '^_%s_([A-Za-z0-9_]+)_omega' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                assert False
            rgx = '%s_([A-Za-z0-9]+)_' % 'omega'
            omega = list(get_set_rgx([experimentType], rgx))
            if len(omega) == 0:
                omega = None
                print 'Variable \'omega\' is None.'
            else:
                omega = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'other' or resultType == 'modulation-ss' or resultType == 'modulation-ilp' or resultType == 'modulation-analysis':
            rgx = '^_%s_([A-Za-z0-9_]+)_ss_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                assert False
            rgx = '%s_([A-Za-z0-9]+)_' % 'omega'
            omega = list(get_set_rgx([experimentType], rgx))
            if len(omega) == 0:
                omega = None
                print 'Variable \'omega\' is None.'
            else:
                omega = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '%s_([A-Za-z0-9]+)_exp_' % 'ss'
            slotframeSize = list(get_set_rgx([experimentType], rgx))
            if len(slotframeSize) == 0:
                slotframeSize = None
                print 'Variable \'slotframeSize\' is None.'
            else:
                slotframeSize = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'modulation-slotframe':
            rgx = '^_%s_([A-Za-z0-9_]+)_omega' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                assert False
            rgx = '%s_([A-Za-z0-9]+)_' % 'omega'
            omega = list(get_set_rgx([experimentType], rgx))
            if len(omega) == 0:
                omega = None
                print 'Variable \'omega\' is None.'
            else:
                omega = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '%s_([A-Za-z0-9]+)_' % 'size'
            slotframeSize = list(get_set_rgx([experimentType], rgx))
            if len(slotframeSize) == 0:
                slotframeSize = None
                print 'Variable \'slotframeSize\' is None.'
            else:
                slotframeSize = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'cdf':
            rgx = '%s_([A-Za-z0-9]+)_' % 'distance'
            distance = list(get_set_rgx([experimentType], rgx))
            if len(distance) == 0:
                distance = None
                print 'Variable \'distance\' is None.'
            else:
                distance = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '^_%s_([A-Za-z0-9]+)_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'time':
            rgx = '%s_([A-Za-z0-9]+)_' % 'distance'
            distance = list(get_set_rgx([experimentType], rgx))
            if len(distance) == 0:
                distance = None
                print 'Variable \'distance\' is None.'
            else:
                distance = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '^_%s_([A-Za-z0-9_]+)_ss_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print 'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                assert False
            rgx = '%s_([A-Za-z0-9]+)_' % 'omega'
            omega = list(get_set_rgx([experimentType], rgx))
            if len(omega) == 0:
                omega = None
                print 'Variable \'omega\' is None.'
            else:
                omega = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'paper' or resultType == 'paper-modulation-analysis' or resultType == 'paper-ilp-throughput' or resultType == 'paper-airtime':
            rgx = '^_%s_([A-Za-z0-9_]+)_ss_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print
                'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                assert False
            rgx = '%s_([A-Za-z0-9]+)_exp_' % 'ss'
            slotframeSize = list(get_set_rgx([experimentType], rgx))
            if len(slotframeSize) == 0:
                slotframeSize = None
                print
                'Variable \'slotframeSize\' is None.'
            else:
                slotframeSize = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '%s_([A-Za-z0-9]+)_' % 'motes'
            nrMotes = list(get_set_rgx([experimentType], rgx))
            if len(nrMotes) == 0:
                nrMotes = None
                print 'Variable \'motes\' is None.'
            else:
                nrMotes = list(get_set_rgx([experimentType], rgx))[0]
        elif resultType == 'paper-heuristic':
            rgx = '^_%s_([A-Za-z0-9_]+)_ss_' % 'c'
            modulationConfig = list(get_set_rgx([experimentType], rgx))
            if len(modulationConfig) == 0:
                modulationConfig = None
                print
                'Variable \'modulationConfig\' is None.'
            else:
                modulationConfig = list(get_set_rgx([experimentType], rgx))[0]
            if modulationConfig not in modulationInstance.configs:
                print experimentType
                print modulationConfig
                assert False
            rgx = '%s_([A-Za-z0-9]+)_exp_' % 'ss'
            slotframeSize = list(get_set_rgx([experimentType], rgx))
            if len(slotframeSize) == 0:
                slotframeSize = None
                print
                'Variable \'slotframeSize\' is None.'
            else:
                slotframeSize = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '%s_([A-Za-z0-9]+)_' % 'motes'
            nrMotes = list(get_set_rgx([experimentType], rgx))
            if len(nrMotes) == 0:
                nrMotes = None
                print 'Variable \'motes\' is None.'
            else:
                nrMotes = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '_%s_([A-Za-z0-9_]+)_delta_' % 'top'
            topologyConfig = list(get_set_rgx([experimentType], rgx))
            if len(topologyConfig) == 0:
                topologyConfig = None
                print 'Variable \'topologyConfig\' is None.'
            else:
                topologyConfig = list(get_set_rgx([experimentType], rgx))[0]
            rgx = '_%s_([A-Za-z0-9_]+)_' % 'delta'
            deltaConfig = list(get_set_rgx([experimentType], rgx))
            if len(deltaConfig) == 0:
                deltaConfig = None
                print
                'Variable \'deltaConfig\' is None.'
            else:
                deltaConfig = list(get_set_rgx([experimentType], rgx))[0]
            # print(topologyConfig)
            # print(slotsConfig)
            # exit()
        else:
            pass

        sorter.append(getExperimentName(experimentType))
        data[experimentType] = {}
        consumption[experimentType] = {}
        timeData[experimentType] = {}
        throughputILP[experimentType] = []
        print 'Parsing results for %s.' % str(experimentType)

        equalLengths = {}

        for dataDir in dataDirs:
            print dataDir
            # timeData[experimentType] = copy.deepcopy(parseTimeResults(dataDir, experimentType))
            
            parseResults(dataDir, experimentType, data[experimentType], consumption[experimentType])
            allLinkData = []
            # if not parsedLinks:
            #     parseLinks(dataDir, experimentType, links)
            #     for modulationOfLink, linkData in links.iteritems():
            #         plotHistogram(modulationOfLink, links[modulationOfLink], outputDir=outputDirPerIteration)
            #         links[modulationOfLink] = list(filter(lambda a: a != 0.0, links[modulationOfLink]))
            #         allLinkData += links[modulationOfLink]
            #     plotHistogram('AllLinks', allLinkData, outputDir=outputDirPerIteration)
            #     parsedLinks = True
            # from collections import Counter
            # with open('histogram-reliability.log', 'a') as f:
            #     f.write(str(Counter(allLinkData)))
            
            # slotData = parseILPSolution(dataDir, experimentType, slotData, modulationConfig)
            
            # # aggregated over all motes
            # chargePerMoteOpenMoteCC2538 = chargePerMoteOpenMoteCC2538.append((calculateChargePerMote(data[experimentType], 'OpenMoteCC2538', experimentType, omega=omega)))
            # lifetime2000OpenMoteCC2538 = lifetime2000OpenMoteCC2538.append(calculateLifetime(chargePerMoteOpenMoteCC2538, 2000, cycles, experimentType, omega=omega))
            # chargePerMoteOpenMoteB = chargePerMoteOpenMoteB.append((calculateChargePerMote(data[experimentType], 'OpenMoteB', experimentType, omega=omega)))
            # lifetime2000OpenMoteB = lifetime2000OpenMoteB.append(calculateLifetime(chargePerMoteOpenMoteB, 2000, cycles, experimentType, omega=omega))

            #### These few line are the fix to make plotting out of multiple data dirs work.
            #I disabled them again to be sure I did not screw something up with the normal working (when enabling them). Uncomment them back if you want to use it.
            cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(dataDir, experimentType)
            listFiles = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it

            # if data[experimentType]:
            #     print("ExperimentType: {0}, datadir {1}, data[experimenttype] = true, len = {2}".format(experimentType, dataDir, len(data[experimentType])))
            # else:
            #     print("ExperimentType: {0}, datadir {1}, data[experimenttype] = false, len = {2}".format(experimentType, dataDir, len(data[experimentType])))

            # if data[experimentType] and len(listFiles) > 0: # only if there is data for this experiment type in the data of this datadir

            # if data[experimentType]: # only if there is data for this experiment type in the data of this datadir
                # # ! This should be on for the ILP/MILP comparison.
            # ilp_throughput = ilp_throughput.append(calculateILPThroughputPerExperiment(dataDir, exp=experimentType, modulationConfig=modulationConfig, ss=slotframeSize, numMotes=int(nrMotes)))
            # print throughputILP[experimentType]
            # exit()
        print modulationConfig
        equalLengths[experimentType] = checkForEqualLengths(consumption[experimentType], experimentType, modulationInstance.modulationConfigSlotLength[modulationConfig])
        # stateFrequency = stateFrequency.append(calculateStateFrequency(data[experimentType], experimentType))
        stateFrequency = stateFrequency.append(calculateStateFrequencyPerModulation(consumption[experimentType], experimentType, modulationInstance=modulationInstance))
        airtimePerMotePerModulation = airtimePerMotePerModulation.append((calculateAirtimePerModulation(consumption[experimentType], experimentType, modulationInstance=modulationInstance, ss=slotframeSize)))
        chargePerMotePerModulation = chargePerMotePerModulation.append((calculateChargePerMotePerModulation(consumption[experimentType], experimentType, modulationInstance.modulationConfigSlotLength[modulationConfig], modulationInstance=modulationInstance, ss=slotframeSize)))
        sixtopMessaging = sixtopMessaging.append(calculateSixTopMessaging(data[experimentType], experimentType))
        DAOMessaging = DAOMessaging.append(calculateDAO(data[experimentType], experimentType))
        # print 'distance: {0}'.format(distance)
        # print 'modulationConfig: {0}'.format(modulationConfig)
        # print 'omega: {0}'.format(omega)
        latency = latency.append(calculateLatency(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        macDrops = macDrops.append(calculateMACDrops(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        queueDrops = queueDrops.append(calculateQueueDrops(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        allDrops = allDrops.append(calculateAllDrops(data[experimentType], experimentType,  modulationConfig=modulationConfig, ss=slotframeSize))
        distances = distances.append(calculateDistances(data[experimentType], experimentType, ss=slotframeSize))
        modulations = modulations.append(calculateModulationPerLink(data[experimentType], experimentType, ss=slotframeSize, modulationConfig=modulationConfig))
        modulationsPerBondedSlot = modulationsPerBondedSlot.append(calculateModulationPerBondedSlot(data[experimentType], experimentType, ss=slotframeSize,modulationConfig=modulationConfig))
        modulationsPerBondedSlotAirtime = modulationsPerBondedSlotAirtime.append(calculateModulationPerBondedSlotAirtime(data[experimentType], experimentType, ss=slotframeSize,modulationConfig=modulationConfig))

        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print modulations.to_string()
        allNrSlots = allNrSlots.append(calculateNrSlots(data[experimentType], experimentType, ss=slotframeSize))
        hopcount = hopcount.append(calculateHopCount(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        minimalHopcount = minimalHopcount.append(calculateMinimalHopCount(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))

        children = children.append(calculateChildren(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        reliability = reliability.append(calculateReliability(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize, topConfig=topologyConfig, deltaConfig=deltaConfig))
        allocatedBondedSlots = allocatedBondedSlots.append(calculateAllocatedBondedSlots(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        allocatedRegularSlots = allocatedRegularSlots.append(calculateAllocatedRegularSlots(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        nrReachRoot = nrReachRoot.append(calculateReachRoot(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        # aggregated per network
        # bitsPerJouleOpenMoteCC2538 = bitsPerJouleOpenMoteCC2538.append(calculateBitsPerJoulePerIteration(data[experimentType], 'OpenMoteCC2538', experimentType))
        # bitsPerJouleOpenMoteB = bitsPerJouleOpenMoteB.append(calculateBitsPerJoulePerIteration(data[experimentType], 'OpenMoteB', experimentType))
        received = received.append(calculateReceived(data[experimentType], experimentType, nrMotes=nrMotes, modulationConfig=modulationConfig,ss=slotframeSize, topConfig=topologyConfig, deltaConfig=deltaConfig))
        pktArrivedToGenerated = pktArrivedToGenerated.append(calculatePktArrivedToGen(data[experimentType], experimentType, nrMotes=nrMotes, modulationConfig=modulationConfig, ss=slotframeSize, topConfig=topologyConfig, deltaConfig=deltaConfig))
        pktGenerated = pktGenerated.append(calculatePktGen(data[experimentType], experimentType, nrMotes=nrMotes, modulationConfig=modulationConfig, ss=slotframeSize))
        pktGeneratedPerMote = pktGeneratedPerMote.append(calculatePktGenPerMote(data[experimentType], experimentType, nrMotes=nrMotes, modulationConfig=modulationConfig, ss=slotframeSize))
        DAOReceived = DAOReceived.append(calculateDAOReceived(data[experimentType], experimentType))

        totalPropagationData = totalPropagationData.append(calculateTotalPropagationData(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        successPropagationData = successPropagationData.append(calculateSuccessPropagationData(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        interferenceFailures = interferenceFailures.append(calculateInterferenceFailures(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        interferenceLockFailures = interferenceLockFailures.append(calculateInterferenceLockFailures(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        signalFailures = signalFailures.append(calculateSignalFailures(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        allInterferers = allInterferers.append(calculateAllInterferers(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))
        hadInterferers = hadInterferers.append(calculateHadInterferers(data[experimentType], experimentType, modulationConfig=modulationConfig, ss=slotframeSize))

    # check if all the calculated lengths are equal.
    lengthFirst = None
    for eT, l in equalLengths.iteritems():
        if lengthFirst is None:
            lengthFirst = l
        if l != lengthFirst:
            assert False

    # aggregated over all motes
    # plotBoxplotSeaborn('lifetime_2000mAh_openMoteCC2538', lifetime2000OpenMoteCC2538, sorter, outputDir=outputDirAll)
    # plotBoxplotSeaborn('lifetime_2000mAh_openMoteB', lifetime2000OpenMoteB, sorter, outputDir=outputDirAll)
    print sorter
    plotBoxplotSeaborn('sixtopMessaging', sixtopMessaging, sorter, outputDir=outputDirAll)
    stateFrequencySum = stateFrequency.groupby(['exp', 'state'])['val'].sum().reset_index() # sum all the same states per experiment
    stateFrequencySum = stateFrequencySum.loc[stateFrequencySum.state != 'Idle'] # remove the idle states
    stateFrequencySum = stateFrequencySum.loc[stateFrequencySum.state != 'Sleep'] # remove the sleep states
    # with pd.option_context('display.max_colwidth', 500, 'display.max_columns', None):  # more options can be specified also
    #     print stateFrequencySum.head().to_string()
    plotBarsSeaborn('StateFrequency', stateFrequencySum, outputDir=outputDirAll, xsplit='state', hue='exp')
    # plotMultipleBoxplotSeaborn('StateFrequency', stateFrequencySum, outputDir=outputDirAll, xsplit='state')

    sixtopMessagingMean = sixtopMessaging.groupby(['exp', 'type'])['val'].sum().reset_index() # sum all the same states per experiment
    plotBarsSeaborn('SixTopMessaging', sixtopMessagingMean, outputDir=outputDirAll, xsplit='type')
    plotBoxplotSeaborn('latency', latency, sorter, outputDir=outputDirAll)
    #
    # # aggregatd per iteration
    plotBoxplotSeaborn('DAOMessagingReceived', DAOReceived, sorter, outputDir=outputDirPerIteration)
    DAOMessaging = DAOMessaging.groupby(['exp', 'iteration', 'type'])['val'].mean().reset_index() # WATCH OUT: you can not just do the mean here because of the received DAOs
    # plotMultipleBoxplotSeaborn('DAOMessaging', DAOMessaging, outputDir=outputDirPerIteration, xsplit='type')
    # plotBoxplotSeaborn('kbitsPerJoule_openMoteCC2538', bitsPerJouleOpenMoteCC2538, sorter, outputDir=outputDirPerIteration)
    # plotBoxplotSeaborn('kbitsPerJoule_openMoteB', bitsPerJouleOpenMoteB, sorter, outputDir=outputDirPerIteration)
    plotBoxplotSeaborn('received', received, sorter, outputDir=outputDirPerIteration)
    plotBoxplotSeaborn('pktGen', pktGenerated, sorter, outputDir=outputDirPerIteration)
    plotBoxplotSeaborn('allDrops', allDrops, sorter, outputDir=outputDirPerIteration)

    plotBoxplotSeaborn('distances', distances, sorter, outputDir=outputDirPerIteration, xlabel='Distance (m)')
    plotBoxplotSeaborn('modulations', modulations, sorter, outputDir=outputDirPerIteration, xlabel='Omega')
    plotLineSeaborn('nrSlots', allNrSlots, sorter, outputDir=outputDirPerIteration, xlabel='Distance (m)')

    # lifetime2000OpenMoteCC2538PerIteration = lifetime2000OpenMoteCC2538.groupby(['exp', 'iteration'])['val'].mean().reset_index() # take the mean of all motes in the same iteration
    # plotBoxplotSeaborn('lifetime_2000mAh_openMoteCC2538', lifetime2000OpenMoteCC2538PerIteration, sorter, outputDir=outputDirPerIteration)
    # lifetime2000OpenMoteBPerIteration = lifetime2000OpenMoteB.groupby(['exp', 'iteration'])['val'].mean().reset_index() # take the mean of all motes in the same iteration
    # plotBoxplotSeaborn('lifetime_2000mAh_openMoteB', lifetime2000OpenMoteBPerIteration, sorter, outputDir=outputDirPerIteration)
    macDropsPerIteration = macDrops.groupby(['exp', 'iteration'])['val'].sum().reset_index()
    plotBoxplotSeaborn('macDrops', macDropsPerIteration, sorter, outputDir=outputDirPerIteration)
    queueDropsPerIteration = queueDrops.groupby(['exp', 'iteration'])['val'].sum().reset_index()
    plotBoxplotSeaborn('queueDrops', queueDropsPerIteration, sorter, outputDir=outputDirPerIteration)

    # plotMultipleBoxplotSeaborn('latency', latencyPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='freq', hue='modeC')
    # plotMultipleBoxplotSeaborn('received', received, sorter, outputDir=outputDirPerIteration, xsplit='motes', hue='sfmode')
    # plotBars('DAOMessaging', DAOMessaging, None, outputDir=outputDir)

    if resultType == 'distance':
        sortBoxPlots = ['MCS 6 (QAM_16_FEC_3_4)', 'MCS 5 (QAM_16_FEC_1_2)', 'MCS 4 (QPSK_FEC_3_4)', 'MCS 3 (QPSK_FEC_1_2)', 'MCS 2 (QPSK_FEC_1_2, FR)']
        # sortBoxPlots = ['MCS 6 (QAM_16_FEC_3_4)', 'MCS 5 (QAM_16_FEC_1_2)', 'MCS 4 (QPSK_FEC_3_4)', 'MCS 3 (QPSK_FEC_1_2)']

        modulationsPerIteration = modulations.groupby(['exp', 'modulation', 'iteration'])['val'].sum().reset_index()
        modulationsPerIteration.modulation = modulationsPerIteration.modulation.astype("category")
        modulationsPerIteration.modulation.cat.set_categories(sortBoxPlots, inplace=True)
        plotMultipleBoxplotSeaborn('modulations', modulationsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', xlabel='Distance (meters)')
    elif resultType == 'modulation-distance':
        latencyPerIteration = latency.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('latency', latencyPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)', percentile=95)
        hopcountPerIteration = hopcount.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hopcount', hopcountPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        pktGenPerIteration = pktGenerated.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktGen', pktGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        pktArrivedToGenPerIteration = pktArrivedToGenerated.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktArrivedToGen', pktArrivedToGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        receivedPerIteration = received.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('received', receivedPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        allDropsPerIteration = allDrops.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('allDrops', allDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        queueDropsPerIteration = queueDrops.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('queueDrops', queueDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        macDropsPerIteration = macDrops.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('macDrops', macDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        # totalPropagationDataPerIteration = totalPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('totalPropagationData', totalPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        # successPropagationDataPerIteration = successPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('successPropagationData', successPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        interferenceFailuresDataPerIteration = interferenceFailures.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceFailures', interferenceFailuresDataPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        interferenceLockFailuresPerIteration = interferenceLockFailures.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceLockFailures', interferenceLockFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        signalFailuresPerIteration = signalFailures.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('signalFailures', signalFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        allInterferersPerIteration = allInterferers.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('allInterferers', allInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        hadInterferersPerIteration = hadInterferers.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hadInterferers', hadInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        reliabilityPerIteration = reliability.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
    elif resultType == 'modulation-analysis':
        # sortBoxPlots = ['MCS 6 (16-QAM, FEC 3/4)', 'MCS 5 (16-QAM, FEC 1/2)', 'MCS 4 (OQPSK, FEC 3/4)',
        #                 'MCS 3 (OQPSK, FEC 1/2)', 'MCS 2 (OQPSK, FEC 1/2, FR 2x)']

        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',None):  # more options can be specified also
            print modulations.to_string()
        # assert False
        modulationsPerIteration = modulations.groupby(['exp', 'omega', 'modulation', 'iteration'])['val'].sum().reset_index()
        # modulationsPerIteration.modulation = modulationsPerIteration.modulation.astype("category")
        # modulationsPerIteration.modulation.cat.set_categories(sortBoxPlots, inplace=True)
        plotMultipleBoxplotSeaborn('modulations', modulationsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', xlabel='Omega')

        modulationsPerIterationSummed = modulationsPerIteration.groupby(['exp', 'omega', 'modulation'])['val'].sum().reset_index()
        plotBarsSeaborn('modulations', modulationsPerIterationSummed, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Total count per experiment')

        modulationsPerIterationSummedALL = modulationsPerIteration.groupby(['exp', 'omega', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMean('modulationsperiterationALL', modulationsPerIterationSummedALL,outputDir=outputDirPerIteration, xsplit='exp', ylabel='Average count per iteration')
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns', None):  # more options can be specified also
            print modulationsPerIterationSummedALL.to_string()
        modulationsPerIterationSummedALLMEAN = modulationsPerIterationSummedALL.groupby(['exp', 'omega'])['val'].mean().reset_index()
        print 'The modulationsPerIterationSummedALLMEAN'
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',None):  # more options can be specified also
            print modulationsPerIterationSummedALLMEAN.to_string()

        modulationsPerIterationSummed = modulationsPerIteration.groupby(['exp', 'omega', 'modulation', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMean('modulationsperiteration', modulationsPerIterationSummed, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Average count per iteration')
        print sorter

        # # # ENABLE THIS IF YOU WANT TO COMPARE 10 MS TO 30 MS!!!! For the modulation config results
        # modulation comparison 0.1, 0.5, 0.9 for 10 and 30 ms

        # ten_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # ten_ms = ten_ms.reset_index(drop=True)
        # # ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_120_exp_30ms_120ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP40msFR_ss_120_exp_40ms_120ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP40msFR_ss_200_exp_40ms_200ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP40msFR_ss_280_exp_40ms_280ms_']
        # # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_120_exp_30ms_120ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP40msFR_ss_120_exp_40ms_120ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP40msFR_ss_200_exp_40ms_200ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP40msFR_ss_280_exp_40ms_280ms_'])
        #
        # thirty_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # thirty_ms = thirty_ms.reset_index(drop=True)
        # # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_120_exp_10ms_120ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_120_exp_10ms_120ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_200_exp_10ms_200ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_280_exp_10ms_280ms_']
        # # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_120_exp_10ms_120ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_120_exp_10ms_120ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_200_exp_10ms_200ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_280_exp_10ms_280ms_'])


        # ten_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # ten_ms = ten_ms.reset_index(drop=True)
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_120_exp_30ms_120ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_150_exp_30ms_150ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_210_exp_30ms_210ms_']
        # ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_300_exp_30ms_300ms_']
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_120_exp_30ms_120ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_150_exp_30ms_150ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_210_exp_30ms_210ms_'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_300_exp_30ms_300ms_'])
        #
        # thirty_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # thirty_ms = thirty_ms.reset_index(drop=True)
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_120_exp_10ms_120ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_150_exp_10ms_150ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_210_exp_10ms_210ms_']
        # thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_300_exp_10ms_300ms_']
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_120_exp_10ms_120ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_150_exp_10ms_150ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_210_exp_10ms_210ms_'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_300_exp_10ms_300ms_'])

        ten_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        ten_ms = ten_ms.reset_index(drop=True)
        ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_150_exp_30ms_150ms_']
        ten_ms = ten_ms[ten_ms.exp != '_c_ILP30msFR_ss_210_exp_30ms_210ms_']
        ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_150_exp_30ms_150ms_'])
        ten_ms.exp = ten_ms.exp.cat.remove_categories(['_c_ILP30msFR_ss_210_exp_30ms_210ms_'])

        thirty_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        thirty_ms = thirty_ms.reset_index(drop=True)
        thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_150_exp_10ms_150ms_']
        thirty_ms = thirty_ms[thirty_ms.exp != '_c_ILP10msFR_ss_210_exp_10ms_210ms_']
        thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_150_exp_10ms_150ms_'])
        thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['_c_ILP10msFR_ss_210_exp_10ms_210ms_'])

        plotBarsSeabornMeanPolicies('modulationsperiteration', ten_ms, thirty_ms, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Average count per iteration')
        plotBarsSeabornThroughput(outputDir=outputDirPerIteration)

        modulationsPerIterationSummedMEAN = modulationsPerIterationSummed.groupby(['exp', 'omega', 'modulation'])['val'].mean().reset_index()
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print modulationsPerIterationSummedMEAN.to_string()

    elif resultType == 'modulation-ss':

        ### charge vs average throughput

        # calculate the charge per iteration, summing all the charge of all motes
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print throughput_merged_dataframe.to_string()
        # assert False
        # plotBoxplotSeabornThroughput('throughput', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration)
        plotMultipleBoxplotSeabornThroughput('PDR (%)', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='modulationConfig_left', xlabel='Slot frame length (ms)')


        # merging charge and throughput
        charge_merged_dataframe = pd.merge(chargePerMotePerModulationPerIteration, throughput_merged_dataframe, on='iterationDir')

        # plotting ALL the charge/throughput points
        plotScatterParetoFront(metricX='Throughput', xfilemetric='throughput', xaxis='throughput', metricY='Charge ($\mu$C)', yfilemetric='charge', yaxis='val', data=charge_merged_dataframe, hue='exp', outputDir=outputDirPerIteration)

        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        # # # ! This should be on for the ILP/MILP comparison.
        # merging charge and throughput
        # ilp_throughput_merged_dataframe = pd.merge(ilp_throughput, throughput_merged_dataframe, on='iterationDir')
        # # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        # #                        None):  # more options can be specified also
        # #     print ilp_throughput_merged_dataframe.to_string()
        # # plotting ALL the charge/throughput points
        # plotScatterParetoFront(metricX='Throughput', xfilemetric='throughput', xaxis='throughput', metricY='ILP Throughput', yfilemetric='ilp_throughput', yaxis='ilp_throughput', data=ilp_throughput_merged_dataframe, hue='exp', outputDir=outputDirPerIteration)

        # average the throughput over all iterations
        average_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].mean().reset_index()
        # average the charges over all iterations
        average_charge_dataframe = charge_merged_dataframe.groupby(['exp'])['val'].mean().reset_index()
        # merge the charge and throughput average
        averages_merged_charge_throughput_dataframe = pd.merge(average_charge_dataframe, average_throughput_dataframe, left_on='exp', right_on='exp_left')
        # plotting ALL the average charge/throughput points
        plotScatterParetoFront(metricX='Average throughput', xfilemetric='average-throughput', xaxis='throughput', metricY='Average charge ($\mu$C)', yfilemetric='average-charge', yaxis='val', hue='exp', data=averages_merged_charge_throughput_dataframe, outputDir=outputDirPerIteration)

        ### bits per Joule vs average throughput

        # calculate the charge per iteration, summing all the charge of all motes
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: (row.val_left / float(row.val_right))*100.0, axis=1)
        # merging charge and throughput so we can calculate bits per joule
        bitsperjoule_merged_dataframe = pd.merge(chargePerMotePerModulationPerIteration, throughput_merged_dataframe, on='iterationDir')
        # calculate the bits per joule
        bitsperjoule_merged_dataframe['bitsperjoule'] = bitsperjoule_merged_dataframe.apply(lambda row: ((row.val_left * APPLICATION_SIZE_BITS) / 1000.0) / ((float(row.val) / 1000000.0) * Modulation.Modulation().VOLTS_RADIO), axis=1)
        # plotting ALL the charge/throughput points
        plotScatterParetoFront(metricX='Throughput', xfilemetric='throughput', xaxis='throughput', metricY='kbits/Joule', yfilemetric='kbitsjoule', yaxis='bitsperjoule', data=bitsperjoule_merged_dataframe, hue='exp', outputDir=outputDirPerIteration)

        # average the throughput over all iterations
        average_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].mean().reset_index()

        # average the charges over all iterations
        average_bitsperjoule_dataframe = bitsperjoule_merged_dataframe.groupby(['exp'])['bitsperjoule'].mean().reset_index()
        # merge the bits per joule and throughput average
        averages_merged_bitsperjoule_throughput_dataframe = pd.merge(average_bitsperjoule_dataframe, average_throughput_dataframe, left_on='exp', right_on='exp_left')
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print averages_merged_bitsperjoule_throughput_dataframe.to_string()
        std_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].std().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print std_throughput_dataframe.to_string()
        # plotting ALL the average bitsperjoule/throughput points
        plotScatterParetoFront(metricX='Throughput (%)', xfilemetric='average-throughput', xaxis='throughput', metricY='Energy efficiency (kbit/Joule)', yaxis='bitsperjoule', yfilemetric='kbitsjoule', hue='exp', data=averages_merged_bitsperjoule_throughput_dataframe, outputDir=outputDirPerIteration)
        plotScatterParetoFrontFullLegend(metricX='Throughput (%)', xfilemetric='average-throughput', xaxis='throughput', metricY='Energy efficiency (kbit/Joule)', yaxis='bitsperjoule', yfilemetric='kbitsjoule', hue='exp', data=averages_merged_bitsperjoule_throughput_dataframe, outputDir=outputDirPerIteration)
        ### average schedule time vs average throughput

        hopcountPerIteration = hopcount.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hopcount', hopcountPerIteration, sorter, outputDir=outputDirPerIteration,xsplit='omega', hue='modulationConfig', xlabel='Weight')
        receivedPerIteration = received.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('received', receivedPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        allDropsPerIteration = allDrops.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('allDrops', allDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        queueDropsPerIteration = queueDrops.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('queueDrops', queueDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        macDropsPerIteration = macDrops.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('macDrops', macDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        # totalPropagationDataPerIteration = totalPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('totalPropagationData', totalPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        # successPropagationDataPerIteration = successPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('successPropagationData', successPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        interferenceFailuresDataPerIteration = interferenceFailures.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceFailures', interferenceFailuresDataPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        interferenceLockFailuresPerIteration = interferenceLockFailures.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceLockFailures', interferenceLockFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        signalFailuresPerIteration = signalFailures.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('signalFailures', signalFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        allInterferersPerIteration = allInterferers.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('allInterferers', allInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        hadInterferersPerIteration = hadInterferers.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hadInterferers', hadInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')
        reliabilityPerIteration = reliability.groupby(['exp', 'ss', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Weight')

    elif resultType == 'modulation-ilp':
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration',suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right) * 100.0, axis=1)
        # print(throughput_merged_dataframe)
        # print('waaait')
        # print(ilp_throughput)
        # merging charge and throughput
        ilp_throughput_merged_dataframe = pd.merge(ilp_throughput, throughput_merged_dataframe, on='iterationDir')
        print(ilp_throughput_merged_dataframe)
        # plotting ALL the charge/throughput points
        plotScatterParetoFrontMILPAndSimulationThroughput(metricX='Simulation PDR (%)', xfilemetric='throughput',
                                                          xaxis='throughput', metricY='GA PDR (%)',
                                                          yfilemetric='ilp_throughput', yaxis='ilp_throughput',
                                                          data=ilp_throughput_merged_dataframe,
                                                          outputDir=outputDirPerIteration)
        print 'The RMSE is: %.4f' % (((ilp_throughput_merged_dataframe.throughput - ilp_throughput_merged_dataframe.ilp_throughput) ** 2).mean() ** .5)
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print ilp_throughput_merged_dataframe.to_string()
        # assert False

    elif resultType == 'modulation-omega':

        ### charge vs average throughput

        # calculate the charge per iteration, summing all the charge of all motes
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)

        plotBoxplotSeabornThroughput('throughput', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration)

        # merging charge and throughput
        charge_merged_dataframe = pd.merge(chargePerMotePerModulationPerIteration, throughput_merged_dataframe, on='iterationDir')
        # plotting ALL the charge/throughput points


        # plotScatterParetoFront(metricX='Throughput', xfilemetric='throughput', xaxis='throughput', metricY='Charge ($\mu$C)', yfilemetric='charge', yaxis='val', data=charge_merged_dataframe, hue='exp', outputDir=outputDirPerIteration)

        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right)*100.0, axis=1)

        # # # ! This should be on for the ILP/MILP comparison.
        # merging charge and throughput
        # ilp_throughput_merged_dataframe = pd.merge(ilp_throughput, throughput_merged_dataframe, on='iterationDir')
        # plotting ALL the charge/throughput points
        # plotScatterParetoFrontMILPAndSimulationThroughput(metricX='Simulation throughput (%)', xfilemetric='throughput', xaxis='throughput', metricY='MILP throughput (%)', yfilemetric='ilp_throughput', yaxis='ilp_throughput', data=ilp_throughput_merged_dataframe, outputDir=outputDirPerIteration)
        # print 'The RMSE is: %.4f' % (((ilp_throughput_merged_dataframe.throughput - ilp_throughput_merged_dataframe.ilp_throughput) ** 2).mean() ** .5)
        # assert False

        # plotScatterParetoFront(metricX='Simulation throughput (%)', xfilemetric='throughput', xaxis='throughput', metricY='MILP Throughput (%)', yfilemetric='ilp_throughput', yaxis='ilp_throughput', data=ilp_throughput_merged_dataframe, outputDir=outputDirPerIteration)
        # # ilp_throughput_merged_dataframe['rm']
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print ilp_throughput_merged_dataframe.to_string()
        # print 'The RMSE is: %.4f' % (((ilp_throughput_merged_dataframe.throughput - ilp_throughput_merged_dataframe.ilp_throughput) ** 2).mean() ** .5)
        # assert False

        # assert False

        # average the throughput over all iterations
        average_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].mean().reset_index()
        # average the charges over all iterations
        average_charge_dataframe = charge_merged_dataframe.groupby(['exp'])['val'].mean().reset_index()
        # merge the charge and throughput average
        averages_merged_charge_throughput_dataframe = pd.merge(average_charge_dataframe, average_throughput_dataframe, left_on='exp', right_on='exp_left')
        # plotting ALL the average charge/throughput points
        plotScatterParetoFront(metricX='Average throughput', xfilemetric='average-throughput', xaxis='throughput', metricY='Average charge ($\mu$C)', yfilemetric='average-charge', yaxis='val', hue='exp', data=averages_merged_charge_throughput_dataframe, outputDir=outputDirPerIteration)

        ### bits per Joule vs average throughput

        # calculate the charge per iteration, summing all the charge of all motes
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: (row.val_left / float(row.val_right))*100.0, axis=1)
        # merging charge and throughput so we can calculate bits per joule
        bitsperjoule_merged_dataframe = pd.merge(chargePerMotePerModulationPerIteration, throughput_merged_dataframe, on='iterationDir')
        print 'charge'
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print bitsperjoule_merged_dataframe.to_string()
        # calculate the bits per joule
        bitsperjoule_merged_dataframe['bitsperjoule'] = bitsperjoule_merged_dataframe.apply(lambda row: ((row.val_left * APPLICATION_SIZE_BITS) / 1000.0) / ((float(row.val) / 1000000.0) * Modulation.Modulation().VOLTS_RADIO), axis=1)
        # plotting ALL the charge/throughput points
        plotScatterParetoFront(metricX='Throughput', xfilemetric='throughput', xaxis='throughput', metricY='kbits/Joule', yfilemetric='kbitsjoule', yaxis='bitsperjoule', data=bitsperjoule_merged_dataframe, hue='exp', outputDir=outputDirPerIteration)

        # average the throughput over all iterations
        average_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].mean().reset_index()

        # average the charges over all iterations
        average_bitsperjoule_dataframe = bitsperjoule_merged_dataframe.groupby(['exp'])['bitsperjoule'].mean().reset_index()
        # merge the bits per joule and throughput average
        averages_merged_bitsperjoule_throughput_dataframe = pd.merge(average_bitsperjoule_dataframe, average_throughput_dataframe, left_on='exp', right_on='exp_left')
        print 'Throughput and bitsperjoule:'
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print averages_merged_bitsperjoule_throughput_dataframe.to_string()
        std_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].std().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print std_throughput_dataframe.to_string()
        # assert False
        # plotting ALL the average bitsperjoule/throughput points
        plotScatterParetoFront(metricX='PDR (%)', xfilemetric='average-throughput', xaxis='throughput', metricY='Energy efficiency (kbit/Joule)', yaxis='bitsperjoule', yfilemetric='kbitsjoule', hue='exp', data=averages_merged_bitsperjoule_throughput_dataframe, outputDir=outputDirPerIteration)
        plotScatterParetoFrontFullLegend(metricX='Throughput (%)', xfilemetric='average-throughput', xaxis='throughput', metricY='kbit/Joule', yaxis='bitsperjoule', yfilemetric='kbitsjoule', hue='exp', data=averages_merged_bitsperjoule_throughput_dataframe, outputDir=outputDirPerIteration)
        ### average schedule time vs average throughput

        latencyPerIteration = latency.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('latency', latencyPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight', percentile=95)
        hopcountPerIteration = hopcount.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hopcount', hopcountPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        pktGenPerIteration = pktGenerated.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktGen', pktGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        pktArrivedToGenPerIteration = pktArrivedToGenerated.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktArrivedToGen', pktArrivedToGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        receivedPerIteration = received.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('received', receivedPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        allDropsPerIteration = allDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('allDrops', allDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        queueDropsPerIteration = queueDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('queueDrops', queueDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        macDropsPerIteration = macDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('macDrops', macDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        # totalPropagationDataPerIteration = totalPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('totalPropagationData', totalPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        # successPropagationDataPerIteration = successPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('successPropagationData', successPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        interferenceFailuresDataPerIteration = interferenceFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceFailures', interferenceFailuresDataPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        interferenceLockFailuresPerIteration = interferenceLockFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceLockFailures', interferenceLockFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        signalFailuresPerIteration = signalFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('signalFailures', signalFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        allInterferersPerIteration = allInterferers.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('allInterferers', allInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        hadInterferersPerIteration = hadInterferers.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hadInterferers', hadInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        reliabilityPerIteration = reliability.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')

        sortBoxPlots = ['MCS 6 (16-QAM, FEC 3/4)', 'MCS 5 (16-QAM, FEC 1/2)', 'MCS 4 (OQPSK, FEC 3/4)', 'MCS 3 (OQPSK, FEC 1/2)', 'MCS 2 (OQPSK, FEC 1/2, FR 2x)']

        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print modulations.to_string()
        # assert False
        modulationsPerIteration = modulations.groupby(['exp', 'omega', 'modulation', 'iteration'])['val'].sum().reset_index()
        modulationsPerIteration.modulation = modulationsPerIteration.modulation.astype("category")
        modulationsPerIteration.modulation.cat.set_categories(sortBoxPlots, inplace=True)
        plotMultipleBoxplotSeaborn('modulations', modulationsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', xlabel='Omega')

        modulationsPerIterationSummed = modulationsPerIteration.groupby(['exp', 'omega', 'modulation'])['val'].sum().reset_index()
        plotBarsSeaborn('modulations', modulationsPerIterationSummed, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Total count per experiment')

        modulationsPerIterationSummedALL = modulationsPerIteration.groupby(['exp', 'omega', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMean('modulationsperiterationALL', modulationsPerIterationSummedALL, outputDir=outputDirPerIteration, xsplit='exp', ylabel='Average count per iteration')
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print modulationsPerIterationSummedALL.to_string()
        modulationsPerIterationSummedALLMEAN = modulationsPerIterationSummedALL.groupby(['exp', 'omega'])['val'].mean().reset_index()
        print 'The modulationsPerIterationSummedALLMEAN'
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print modulationsPerIterationSummedALLMEAN.to_string()

        modulationsPerIterationSummed = modulationsPerIteration.groupby(['exp', 'omega', 'modulation', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMean('modulationsperiteration', modulationsPerIterationSummed, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Average count per iteration')

        # # # ENABLE THIS IF YOU WANT TO COMPARE 10 MS TO 30 MS!!!! For the modulation config results
        # modulation comparison 0.1, 0.5, 0.9 for 10 and 30 ms

        # ten_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # ten_ms = ten_ms.reset_index(drop=True)
        # ten_ms = ten_ms[ten_ms.exp != '30 ms, $\omega$ = 0.1']
        # ten_ms = ten_ms[ten_ms.exp != '30 ms, $\omega$ = 0.5']
        # ten_ms = ten_ms[ten_ms.exp != '30 ms, $\omega$ = 0.9']
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['30 ms, $\omega$ = 0.1'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['30 ms, $\omega$ = 0.5'])
        # ten_ms.exp = ten_ms.exp.cat.remove_categories(['30 ms, $\omega$ = 0.9'])
        #
        # thirty_ms = pd.DataFrame(data=modulationsPerIterationSummed)
        # thirty_ms = thirty_ms.reset_index(drop=True)
        # thirty_ms = thirty_ms[thirty_ms.exp != '10 ms bonded, $\omega$ = 0.1']
        # thirty_ms = thirty_ms[thirty_ms.exp != '10 ms bonded, $\omega$ = 0.5']
        # thirty_ms = thirty_ms[thirty_ms.exp != '10 ms bonded, $\omega$ = 0.9']
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['10 ms bonded, $\omega$ = 0.1'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['10 ms bonded, $\omega$ = 0.5'])
        # thirty_ms.exp = thirty_ms.exp.cat.remove_categories(['10 ms bonded, $\omega$ = 0.9'])
        #
        # plotBarsSeabornMeanPolicies('modulationsperiteration', ten_ms, thirty_ms, outputDir=outputDirPerIteration, xsplit='exp', hue='modulation', ylabel='Average count per iteration')
        # plotBarsSeabornThroughput(outputDir=outputDirPerIteration)
        #
        # modulationsPerIterationSummedMEAN = modulationsPerIterationSummed.groupby(['exp', 'omega', 'modulation'])['val'].mean().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print modulationsPerIterationSummedMEAN.to_string()

    elif resultType == 'modulation-slotframe':
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print chargePerMotePerModulationPerIteration.to_string()

        # assert False

        merge_throughput = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        merge_throughput['throughput'] = merge_throughput.apply(lambda row: row.val_left / float(row.val_right), axis=1)

        mergeCharge = pd.merge(chargePerMotePerModulationPerIteration, merge_throughput, on='iterationDir')
        plotScatterParetoFront(metricX='Throughput ratio', xaxis='throughput', metricY='Total Charge ($\mu$C)', yaxis='val', data=mergeCharge, hue='exp', outputDir=outputDirPerIteration)

        avg_throughput = merge_throughput.groupby(['exp_left'])['throughput'].mean().reset_index()
        avg_charge = mergeCharge.groupby(['exp'])['val'].mean().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print avg_charge.to_string()
        merge_charge_avgs = pd.merge(avg_charge, avg_throughput, left_on='exp', right_on='exp_left')
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print merge_charge_avgs.to_string()
        plotScatterParetoFront(metricX='Average throughput', xaxis='throughput', metricY='Average charge ($\mu$C)', yaxis='val', hue='exp', data=merge_charge_avgs, outputDir=outputDirPerIteration)
    elif resultType == 'cdf':
        # plot CDF for latency per mote
        latencyPerMote = latency.groupby(['exp', 'distance', 'modulationConfig', 'iteration', 'mote'])['val'].mean().reset_index()
        plotCDFSeaborn('latency', latencyPerMote, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)', percentile=95)

        pktGenPerMote = pktGeneratedPerMote.groupby(['exp', 'distance', 'modulationConfig', 'iteration', 'mote'])['val'].mean().reset_index()
        plotCDFSeaborn('pktGen', pktGenPerMote, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')

        # # plot CDF for latency per iteration
        # latencyPerIteration = latency.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        # plotCDFSeaborn('latency', latencyPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)', percentile=95)
        #
        # pktGenPerIteration = pktGenerated.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        # plotCDFSeaborn('pktGen', pktGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')
        # pktArrivedToGenPerIteration = pktArrivedToGenerated.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        # plotCDFSeaborn('pktArrivedToGen', pktArrivedToGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='distance', hue='modulationConfig', xlabel='Distance (meters)')

    elif resultType == 'time':
        plotTimeSeriesSeaborn('allCells', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('overlappingCells', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('usedCells', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('overlappingRatio', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('slotTime', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('airTime', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')
        plotTimeSeriesSeaborn('airTimeRatio', data=timeData, outputDir=outputDirPerIteration, xlabel='Time (cycles)')

    elif resultType == 'other':
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print throughput_merged_dataframe.to_string()
        # exit()
        # assert False
        # plotBoxplotSeabornThroughput('throughput', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration)
        plotMultipleBoxplotSeabornThroughput('PDR (%)', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='modulationConfig_left', xlabel='Slot frame length (ms)')
        plotMultipleLineSeabornThroughput('PDR (%)', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='modulationConfig_left', xlabel='Slot frame length (ms)')
        chargePerMotePerModulationPerIteration = chargePerMotePerModulation.groupby(['exp', 'iterationDir', 'iteration'])['val'].sum().reset_index()

        mergeCharge = pd.merge(chargePerMotePerModulationPerIteration, throughput_merged_dataframe, on='iterationDir')
        plotScatterParetoFront(metricX='Throughput ratio', xaxis='throughput', metricY='Total Charge ($\mu$C)', yaxis='val', data=mergeCharge, hue='exp', outputDir=outputDirPerIteration)


        # average the throughput over all iterations
        average_throughput_dataframe = throughput_merged_dataframe.groupby(['exp_left'])['throughput'].mean().reset_index()
        # average the charges over all iterations
        average_charge_dataframe = mergeCharge.groupby(['exp'])['val'].mean().reset_index()
        # merge the charge and throughput average
        averages_merged_charge_throughput_dataframe = pd.merge(average_charge_dataframe, average_throughput_dataframe, left_on='exp', right_on='exp_left')
        # plotting ALL the average charge/throughput points
        plotScatterParetoFront(metricX='Average throughput', xfilemetric='average-throughput', xaxis='throughput', metricY='Average charge ($\mu$C)', yfilemetric='average-charge', yaxis='val', hue='exp', data=averages_merged_charge_throughput_dataframe, outputDir=outputDirPerIteration)


        latencyPerIteration = latency.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('latency', latencyPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight', percentile=95)
        hopcountPerIteration = hopcount.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hopcount', hopcountPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        pktGenPerIteration = pktGenerated.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktGen', pktGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        pktArrivedToGenPerIteration = pktArrivedToGenerated.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('pktArrivedToGen', pktArrivedToGenPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        receivedPerIteration = received.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('received', receivedPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        allDropsPerIteration = allDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('allDrops', allDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        queueDropsPerIteration = queueDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('queueDrops', queueDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        macDropsPerIteration = macDrops.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('macDrops', macDropsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        # totalPropagationDataPerIteration = totalPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('totalPropagationData', totalPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        # successPropagationDataPerIteration = successPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('successPropagationData', successPropagationData, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        interferenceFailuresDataPerIteration = interferenceFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceFailures', interferenceFailuresDataPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        interferenceLockFailuresPerIteration = interferenceLockFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('interferenceLockFailures', interferenceLockFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        signalFailuresPerIteration = signalFailures.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        plotMultipleBoxplotSeaborn('signalFailures', signalFailuresPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        allInterferersPerIteration = allInterferers.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('allInterferers', allInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        hadInterferersPerIteration = hadInterferers.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('hadInterferers', hadInterferersPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')
        reliabilityPerIteration = reliability.groupby(['exp', 'omega', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='omega', hue='modulationConfig', xlabel='Weight')

    elif resultType == 'paper':
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print throughput_merged_dataframe.to_string()
        # exit()
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        throughput_merged_dataframe['motes_and_modulationConfig'] = throughput_merged_dataframe.apply(lambda row: "{0}, {1}".format(row.motes_left, row.modulationConfig_left) , axis=1)
        plotMultipleBoxplotSeabornThroughput('PDR (%)', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='motes_and_modulationConfig', xlabel='Slotframe length (ms)')
        plotMultipleLineSeabornThroughput('PDR', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='motes_and_modulationConfig', xlabel='Slotframe length (ms)')

        # calculating all drops percentage over the number of packets that arrived to generate
        # all drops is counted per iteration
        alldrops_merged_df = pd.merge(allDrops, pktArrivedToGenerated, on='iteration', suffixes=('_allDrops', '_generated'))
        alldrops_merged_df['alldrops_percentage'] = alldrops_merged_df.apply(lambda row: 100 * row.val_allDrops / float(row.val_generated), axis=1)
        mean_alldrops_percentage_per_exp = alldrops_merged_df.groupby(['exp_allDrops'])['alldrops_percentage'].mean().reset_index()
        std_alldrops_percentage_per_exp = alldrops_merged_df.groupby(['exp_allDrops'])['alldrops_percentage'].std().reset_index()
        final_alldrops_merged_df = pd.merge(mean_alldrops_percentage_per_exp, std_alldrops_percentage_per_exp, on='exp_allDrops', suffixes=('_mean', '_std'))
        final_alldrops_merged_df.to_csv("{0}/alldrops.txt".format(outputDir), sep='\t')

        # calculating all queue dropped packets percentage over the number of packets that arrived to generate
        # queue drops is counted per mote
        queueDropsPerIteration = queueDrops.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        queuedrops_merged_df = pd.merge(queueDropsPerIteration, pktArrivedToGenerated, on='iteration', suffixes=('_queueDrops', '_generated'))
        queuedrops_merged_df['queuedrops_percentage'] = queuedrops_merged_df.apply(lambda row: 100 * row.val_queueDrops / float(row.val_generated), axis=1)
        mean_queuedrops_percentage_per_exp = queuedrops_merged_df.groupby(['exp_queueDrops'])['queuedrops_percentage'].mean().reset_index()
        std_queuedrops_percentage_per_exp = queuedrops_merged_df.groupby(['exp_queueDrops'])['queuedrops_percentage'].std().reset_index()
        final_queuedrops_merged_df = pd.merge(mean_queuedrops_percentage_per_exp, std_queuedrops_percentage_per_exp, on='exp_queueDrops', suffixes=('_mean', '_std'))
        final_queuedrops_merged_df.to_csv("{0}/queuedrops.txt".format(outputDir), sep='\t')

        # calculating all mac dropped packets percentage over the number of packets that arrived to generate
        # mac drops is counted per mote
        macDropsPerIteration = macDrops.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        macdrops_merged_df = pd.merge(macDropsPerIteration, pktArrivedToGenerated, on='iteration', suffixes=('_macDrops', '_generated'))
        macdrops_merged_df['macdrops_percentage'] = macdrops_merged_df.apply(lambda row: 100 * row.val_macDrops / float(row.val_generated), axis=1)
        mean_macdrops_percentage_per_exp = macdrops_merged_df.groupby(['exp_macDrops'])['macdrops_percentage'].mean().reset_index()
        std_macdrops_percentage_per_exp = macdrops_merged_df.groupby(['exp_macDrops'])['macdrops_percentage'].std().reset_index()
        final_macdrops_merged_df = pd.merge(mean_macdrops_percentage_per_exp, std_macdrops_percentage_per_exp, on='exp_macDrops', suffixes=('_mean', '_std'))
        final_macdrops_merged_df.to_csv("{0}/macdrops.txt".format(outputDir), sep='\t')

        # the reliability is defined per link
        mean_reliabilityPerIteration = reliability.groupby(['exp'])['val'].mean().reset_index()
        std_reliabilityPerIteration = reliability.groupby(['exp'])['val'].std().reset_index()
        final_reliability_merged_df = pd.merge(mean_reliabilityPerIteration, std_reliabilityPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_reliability_merged_df.to_csv("{0}/reliability.txt".format(outputDir), sep='\t')

        # hopcount
        mean_hopcountPerIteration = hopcount.groupby(['exp'])['val'].mean().reset_index()
        std_hopcountPerIteration = hopcount.groupby(['exp'])['val'].std().reset_index()
        final_hopcount_merged_df = pd.merge(mean_hopcountPerIteration, std_hopcountPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_hopcount_merged_df.to_csv("{0}/hopcount.txt".format(outputDir), sep='\t')

        mean_minimalHopcountPerIteration = minimalHopcount.groupby(['exp'])['val'].mean().reset_index()
        std_minimalHopcountPerIteration = minimalHopcount.groupby(['exp'])['val'].std().reset_index()
        final_minimalHopcount_merged_df = pd.merge(mean_minimalHopcountPerIteration, std_minimalHopcountPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_minimalHopcount_merged_df.to_csv("{0}/minimalhopcount.txt".format(outputDir), sep='\t')

        # allocated bonded slots
        mean_allocatedBondedSlotsPerIteration = allocatedBondedSlots.groupby(['exp'])['val'].mean().reset_index()
        std_allocatedBondedSlotsPerIteration = allocatedBondedSlots.groupby(['exp'])['val'].std().reset_index()
        final_allocatedBondedSlots_merged_df = pd.merge(mean_allocatedBondedSlotsPerIteration, std_allocatedBondedSlotsPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_allocatedBondedSlots_merged_df.to_csv("{0}/allocatedBondedSlots.txt".format(outputDir), sep='\t')

        summedReachRootPerIteration = nrReachRoot.groupby(['exp', 'iteration'])['val'].sum().reset_index()
        # with pd.option_context('display.max_colwidth', 20, 'display.max_columns', None):
        #     print summedReachRootPerIteration.to_csv("{0}/reachRootPerIteration.txt".format(outputDir), sep='\t')
        # exit()
        mean_reachRootPerIteration = summedReachRootPerIteration.groupby(['exp'])['val'].mean().reset_index()
        std_reachRootPerIteration = summedReachRootPerIteration.groupby(['exp'])['val'].std().reset_index()
        final_reachRootPerIteration_merged_df = pd.merge(mean_reachRootPerIteration, std_reachRootPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_reachRootPerIteration_merged_df.to_csv("{0}/reachRootPerIteration.txt".format(outputDir), sep='\t')

        # I am not sure that the following plots are correct in terms of per iteration or per mote
        # reliabilityPerIteration = reliability.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        # plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Slot frame length (ms)')
        # allocatedBondedSlotsPerIteration = allocatedBondedSlots.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('allocatedBondedSlots', allocatedBondedSlotsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Slot frame length (ms)')
        # allocatedRegularSlotsPerIteration = allocatedRegularSlots.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('allocatedRegularSlots', allocatedRegularSlotsPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Slot frame length (ms)')
        # hopcountPerIteration = hopcount.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].mean().reset_index()
        # plotMultipleBoxplotSeaborn('hopcount', hopcountPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig', xlabel='Slot frame length (ms)')

        # plotMultipleBoxplotSeaborn('totalPropagationData', totalPropagationData, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # # successPropagationDataPerIteration = successPropagationData.groupby(['exp', 'distance', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('successPropagationData', successPropagationData, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # interferenceFailuresDataPerIteration = \
        # interferenceFailures.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])['val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('interferenceFailures', interferenceFailuresDataPerIteration, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # interferenceLockFailuresPerIteration = \
        # interferenceLockFailures.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])[
        #     'val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('interferenceLockFailures', interferenceLockFailuresPerIteration, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # signalFailuresPerIteration = signalFailures.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])[
        #     'val'].sum().reset_index()
        # plotMultipleBoxplotSeaborn('signalFailures', signalFailuresPerIteration, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # allInterferersPerIteration = allInterferers.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])[
        #     'val'].mean().reset_index()
        # plotMultipleBoxplotSeaborn('allInterferers', allInterferersPerIteration, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')
        # hadInterferersPerIteration = hadInterferers.groupby(['exp', 'ss', 'modulationConfig', 'iteration'])[
        #     'val'].mean().reset_index()
        # plotMultipleBoxplotSeaborn('hadInterferers', hadInterferersPerIteration, sorter,
        #                            outputDir=outputDirPerIteration, xsplit='ss', hue='modulationConfig',
        #                            xlabel='Distance (meters)')


    elif resultType == 'paper-heuristic':
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration', suffixes=('_left', '_right'))
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print throughput_merged_dataframe.to_string()
        # exit()
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print throughput_merged_dataframe.to_string()
        def replace_name(name):
            if name == 'heuristic, 001':
                return "$\delta$ = 0.01"
            elif name == 'heuristic, 002':
                return "$\delta$ = 0.02"
            elif name == 'heuristic, 005':
                return "$\delta$ = 0.05"
            elif name == 'heuristic, 010':
                return "$\delta$ = 0.1"
            elif name == 'heuristic, 020':
                return "$\delta$ = 0.2"
            elif name == 'heuristic, 030':
                return "$\delta$ = 0.3"
            elif name == 'heuristic, 040':
                return "$\delta$ = 0.4"
            elif name == 'heuristic, 060':
                return "$\delta$ = 0.6"
            elif name == 'heuristic, 080':
                return "$\delta$ = 0.8"
            elif name == 'heuristic, 100':
                return "$\delta$ = 1"
            elif name == 'ga, 000':
                return "GA"

        throughput_merged_dataframe['topdelta'] = throughput_merged_dataframe.apply(lambda row: replace_name("{0}, {1}".format(row.topConfig_left, row.deltaConfig_left)), axis=1)
        plotMultipleBoxplotSeabornThroughput('PDR (%)', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='topdelta', xlabel='Slotframe length (ms)')
        # plotMultipleLineSeabornThroughput('PDR', throughput_merged_dataframe, sorter, outputDir=outputDirPerIteration, xsplit='ss_left', hue='topdelta', xlabel='Slotframe length (ms)')

        # calculating all drops percentage over the number of packets that arrived to generate
        # all drops is counted per iteration
        alldrops_merged_df = pd.merge(allDrops, pktArrivedToGenerated, on='iteration', suffixes=('_allDrops', '_generated'))
        alldrops_merged_df['alldrops_percentage'] = alldrops_merged_df.apply(lambda row: 100 * row.val_allDrops / float(row.val_generated), axis=1)
        mean_alldrops_percentage_per_exp = alldrops_merged_df.groupby(['exp_allDrops'])['alldrops_percentage'].mean().reset_index()
        std_alldrops_percentage_per_exp = alldrops_merged_df.groupby(['exp_allDrops'])['alldrops_percentage'].std().reset_index()
        final_alldrops_merged_df = pd.merge(mean_alldrops_percentage_per_exp, std_alldrops_percentage_per_exp, on='exp_allDrops', suffixes=('_mean', '_std'))
        final_alldrops_merged_df.to_csv("{0}/alldrops.txt".format(outputDir), sep='\t')

        # calculating all queue dropped packets percentage over the number of packets that arrived to generate
        # queue drops is counted per mote
        queueDropsPerIteration = queueDrops.groupby(['exp', 'ss', 'iteration'])['val'].sum().reset_index()
        queuedrops_merged_df = pd.merge(queueDropsPerIteration, pktArrivedToGenerated, on='iteration', suffixes=('_queueDrops', '_generated'))
        queuedrops_merged_df['queuedrops_percentage'] = queuedrops_merged_df.apply(lambda row: 100 * row.val_queueDrops / float(row.val_generated), axis=1)
        mean_queuedrops_percentage_per_exp = queuedrops_merged_df.groupby(['exp_queueDrops'])['queuedrops_percentage'].mean().reset_index()
        std_queuedrops_percentage_per_exp = queuedrops_merged_df.groupby(['exp_queueDrops'])['queuedrops_percentage'].std().reset_index()
        final_queuedrops_merged_df = pd.merge(mean_queuedrops_percentage_per_exp, std_queuedrops_percentage_per_exp, on='exp_queueDrops', suffixes=('_mean', '_std'))
        final_queuedrops_merged_df.to_csv("{0}/queuedrops.txt".format(outputDir), sep='\t')

        # calculating all mac dropped packets percentage over the number of packets that arrived to generate
        # mac drops is counted per mote
        macDropsPerIteration = macDrops.groupby(['exp', 'ss', 'iteration'])['val'].sum().reset_index()
        macdrops_merged_df = pd.merge(macDropsPerIteration, pktArrivedToGenerated, on='iteration', suffixes=('_macDrops', '_generated'))
        macdrops_merged_df['macdrops_percentage'] = macdrops_merged_df.apply(lambda row: 100 * row.val_macDrops / float(row.val_generated), axis=1)
        mean_macdrops_percentage_per_exp = macdrops_merged_df.groupby(['exp_macDrops'])['macdrops_percentage'].mean().reset_index()
        std_macdrops_percentage_per_exp = macdrops_merged_df.groupby(['exp_macDrops'])['macdrops_percentage'].std().reset_index()
        final_macdrops_merged_df = pd.merge(mean_macdrops_percentage_per_exp, std_macdrops_percentage_per_exp, on='exp_macDrops', suffixes=('_mean', '_std'))
        final_macdrops_merged_df.to_csv("{0}/macdrops.txt".format(outputDir), sep='\t')
        #paper-heuristic
        # # the reliability is defined per link
        mean_reliabilityPerIteration = reliability.groupby(['exp'])['val'].mean().reset_index()
        std_reliabilityPerIteration = reliability.groupby(['exp'])['val'].std().reset_index()
        final_reliability_merged_df = pd.merge(mean_reliabilityPerIteration, std_reliabilityPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_reliability_merged_df.to_csv("{0}/reliability.txt".format(outputDir), sep='\t')


        # reliability['topdelta'] = throughput_merged_dataframe.apply(lambda row: replace_name("{0}, {1}".format(row.topConfig_left, row.deltaConfig_left)), axis=1)
        reliability['topdelta'] = reliability.apply(lambda row: replace_name("{0}, {1}".format(row.topConfig, row.deltaConfig)), axis=1)

        # reliabilityPerIteration = reliability.groupby(['exp', 'ss'])['val'].mean().reset_index()
        # plotMultipleBoxplotSeaborn('reliability', reliabilityPerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='exp', xlabel='Weight')
        plotMultipleBoxplotSeabornReliability('Link Reliability', reliability, sorter, outputDir=outputDirPerIteration, xsplit='ss', hue='topdelta', xlabel='Slotframe length (ms)')

        #
        # # hopcount
        mean_hopcountPerIteration = hopcount.groupby(['exp'])['val'].mean().reset_index()
        std_hopcountPerIteration = hopcount.groupby(['exp'])['val'].std().reset_index()
        final_hopcount_merged_df = pd.merge(mean_hopcountPerIteration, std_hopcountPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_hopcount_merged_df.to_csv("{0}/hopcount.txt".format(outputDir), sep='\t')

        # mean_minimalHopcountPerIteration = minimalHopcount.groupby(['exp'])['val'].mean().reset_index()
        # std_minimalHopcountPerIteration = minimalHopcount.groupby(['exp'])['val'].std().reset_index()
        # final_minimalHopcount_merged_df = pd.merge(mean_minimalHopcountPerIteration, std_minimalHopcountPerIteration, on='exp', suffixes=('_mean', '_std'))
        # final_minimalHopcount_merged_df.to_csv("{0}/minimalhopcount.txt".format(outputDir), sep='\t')
        # 
        # allocated bonded slots
        mean_allocatedBondedSlotsPerIteration = allocatedBondedSlots.groupby(['exp'])['val'].mean().reset_index()
        std_allocatedBondedSlotsPerIteration = allocatedBondedSlots.groupby(['exp'])['val'].std().reset_index()
        final_allocatedBondedSlots_merged_df = pd.merge(mean_allocatedBondedSlotsPerIteration, std_allocatedBondedSlotsPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_allocatedBondedSlots_merged_df.to_csv("{0}/allocatedBondedSlots.txt".format(outputDir), sep='\t')

        mean_allocatedRegularSlotsPerIteration = allocatedRegularSlots.groupby(['exp'])['val'].mean().reset_index()
        std_allocatedRegularSlotsPerIteration = allocatedRegularSlots.groupby(['exp'])['val'].std().reset_index()
        final_allocatedRegularSlots_merged_df = pd.merge(mean_allocatedRegularSlotsPerIteration, std_allocatedRegularSlotsPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_allocatedRegularSlots_merged_df.to_csv("{0}/allocatedRegularSlots.txt".format(outputDir), sep='\t')


        summedReachRootPerIteration = nrReachRoot.groupby(['exp', 'iteration'])['val'].sum().reset_index()
        # with pd.option_context('display.max_colwidth', 20, 'display.max_columns', None):
        #     print summedReachRootPerIteration.to_csv("{0}/reachRootPerIteration.txt".format(outputDir), sep='\t')
        # exit()
        mean_reachRootPerIteration = summedReachRootPerIteration.groupby(['exp'])['val'].mean().reset_index()
        std_reachRootPerIteration = summedReachRootPerIteration.groupby(['exp'])['val'].std().reset_index()
        final_reachRootPerIteration_merged_df = pd.merge(mean_reachRootPerIteration, std_reachRootPerIteration, on='exp', suffixes=('_mean', '_std'))
        final_reachRootPerIteration_merged_df.to_csv("{0}/reachRootPerIteration.txt".format(outputDir), sep='\t')


    elif resultType == 'paper-modulation-analysis':
        hopcount_merge_modulations = pd.merge(hopcount, modulationsPerBondedSlot, on='iteration', suffixes=('_left', '_right'))
        # with pd.option_context('display.max_colwidth', 20, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print hopcount_merge_modulations.head().to_string()

        plotBarsSeabornMeanModulationsVariableY('val_left', hopcount_merge_modulations, outputDir=outputDirPerIteration, xsplit='ss_left', hue='modulation', ylabel='Hop count', outputLabel='hopcount_merge_modulations')

        children_merge_modulations = pd.merge(children, modulations, on='iteration', suffixes=('_left', '_right'))
        # with pd.option_context('display.max_colwidth', 20, 'display.max_columns',
        #                        None):  # more options can be sp
        # ecified also
        #     print hopcount_merge_modulations.head().to_string()
        children_merge_modulations = children_merge_modulations.loc[children_merge_modulations['val_left'] > 0]

        plotBarsSeabornMeanModulationsVariableY('val_left', children_merge_modulations, outputDir=outputDirPerIteration, xsplit='ss_left', hue='modulation', ylabel='Children', outputLabel='children_merge_modulations')

        modulationsPerIteration = modulations.groupby(['exp', 'modulationConfig', 'ss', 'modulation', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMeanModulations('modulationsperiteration', modulationsPerIteration, outputDir=outputDirPerIteration, xsplit='ss', hue='modulation', ylabel='Links')
        print modulationsPerIteration.to_string()
        modulationsPerBondedSlotPerIteration = modulationsPerBondedSlot.groupby(['exp', 'modulationConfig', 'ss', 'modulation', 'iteration'])['val'].sum().reset_index()
        plotBarsSeabornMeanModulations('modulationsPerBondedSlotPerIteration', modulationsPerBondedSlotPerIteration, outputDir=outputDirPerIteration, xsplit='ss', hue='modulation', ylabel='Bonded slots')

    elif resultType == 'paper-ilp-throughput':
        # merging the number of generated and received packets
        throughput_merged_dataframe = pd.merge(received, pktArrivedToGenerated, on='iteration',suffixes=('_left', '_right'))
        # calculating the throughput as received/generated
        throughput_merged_dataframe['throughput'] = throughput_merged_dataframe.apply(lambda row: row.val_left / float(row.val_right), axis=1)
        # print(throughput_merged_dataframe)
        # print(ilp_throughput)
        # merging charge and throughput
        ilp_throughput['ilp_throughput'] = ilp_throughput.apply(lambda row: row.ilp_throughput / 100.0, axis = 1)
        ilp_throughput_merged_dataframe = pd.merge(ilp_throughput, throughput_merged_dataframe, on='iterationDir')
        # print(ilp_throughput_merged_dataframe)
        # plotting ALL the charge/throughput points
        plotScatterParetoFrontMILPAndSimulationThroughput(metricX='Simulation PDR', xfilemetric='throughput',
                                                          xaxis='throughput', metricY='GA PDR',
                                                          yfilemetric='ilp_throughput', yaxis='ilp_throughput',
                                                          data=ilp_throughput_merged_dataframe,
                                                          outputDir=outputDirPerIteration)
        print 'The RMSE is: %.4f' % (((ilp_throughput_merged_dataframe.throughput - ilp_throughput_merged_dataframe.ilp_throughput) ** 2).mean() ** .5)
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print ilp_throughput_merged_dataframe.head().to_string()
        # assert False

    elif resultType == 'paper-airtime':
        airtimePerMotePerModulation = airtimePerMotePerModulation.groupby(['exp', 'ss', 'iterationDir', 'iteration'])['val'].sum().reset_index()
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print airtimePerMotePerModulation.to_string()
        plotMultipleBoxplotSeaborn('Airtime (ms)', airtimePerMotePerModulation, sorter, outputDir=outputDirPerIteration, xsplit='ss', xlabel='Slot frame length (ms)')
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print modulationsPerBondedSlotAirtime.to_string()
        modulationsPerBondedSlotAirtimePerIteration = modulationsPerBondedSlotAirtime.groupby(['exp', 'ss', 'iteration'])['val'].sum().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print modulationsPerBondedSlotAirtimePerIteration.to_string()
        # plotBarsSeabornMeanModulations('modulationsPerBondedSlotAirtimePerIteration', modulationsPerBondedSlotAirtimePerIteration, outputDir=outputDirPerIteration, xsplit='ss', hue='modulation', ylabel='Airtime')
        plotMultipleBoxplotSeaborn('Airtime bonded slots (ms)', modulationsPerBondedSlotAirtimePerIteration, sorter, outputDir=outputDirPerIteration, xsplit='ss', xlabel='Slot frame length (ms)')

        pass