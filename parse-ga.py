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
                '_c_ILP10msFR_ss_150_exp_10ms_150ms_': '150 ms', \
                '_c_ILP10msFR_ss_210_exp_10ms_210ms_': '210 ms',
                '_c_ILP10msFR_ss_120_exp_sel_tourn2_repl_elitism_mx_06_cx_10_': 'Tourn. 2, elitism',
                '_c_ILP10msFR_ss_120_exp_sel_tourn2_repl_elitism_mx_06_cx_10_': 'Tourn. 2, elitism',
                '_c_ILP10msFR_ss_120_exp_sel_tourn2_repl_offspring_mx_06_cx_10_': 'Tourn. 2, offspring',
                '_c_ILP10msFR_ss_120_exp_sel_tourn2_repl_tourn2_mx_06_cx_10_': 'Tourn. 2, tourn. 2',
                '_c_ILP10msFR_ss_120_exp_sel_tourn5_repl_elitism_mx_06_cx_10_': 'Tourn. 5, elitism',
                '_c_ILP10msFR_ss_120_exp_sel_tourn5_repl_offspring_mx_06_cx_10_': 'Tourn. 5, offspring',
                '_c_ILP10msFR_ss_120_exp_sel_tourn5_repl_tourn5_mx_06_cx_10_': 'Tourn. 5, tourn. 5',
                '_c_ILP10msFR_ss_120_exp_sel_tourn10_repl_elitism_mx_06_cx_10_': 'Tourn. 10, elitism',
                '_c_ILP10msFR_ss_120_exp_sel_tourn10_repl_offspring_mx_06_cx_10_': 'Tourn. 10, offspring',
                '_c_ILP10msFR_ss_120_exp_sel_tourn10_repl_tourn10_mx_06_cx_10_': 'Tourn. 10, tourn. 10',
                '_c_ILP10msFR_ss_280_exp_sel_tourn2_repl_elitism_mx_06_cx_10_': 'Tourn. 2, elitism',
                '_c_ILP10msFR_ss_280_exp_sel_tourn2_repl_elitism_mx_06_cx_10_': 'Tourn. 2, elitism',
                '_c_ILP10msFR_ss_280_exp_sel_tourn2_repl_offspring_mx_06_cx_10_': 'Tourn. 2, offspring',
                '_c_ILP10msFR_ss_280_exp_sel_tourn2_repl_tourn2_mx_06_cx_10_': 'Tourn. 2, tourn. 2',
                '_c_ILP10msFR_ss_280_exp_sel_tourn5_repl_elitism_mx_06_cx_10_': 'Tourn. 5, elitism',
                '_c_ILP10msFR_ss_280_exp_sel_tourn5_repl_offspring_mx_06_cx_10_': 'Tourn. 5, offspring',
                '_c_ILP10msFR_ss_280_exp_sel_tourn5_repl_tourn5_mx_06_cx_10_': 'Tourn. 5, tourn. 5',
                '_c_ILP10msFR_ss_280_exp_sel_tourn10_repl_elitism_mx_06_cx_10_': 'Tourn. 10, elitism',
                '_c_ILP10msFR_ss_280_exp_sel_tourn10_repl_offspring_mx_06_cx_10_': 'Tourn. 10, offspring',
                '_c_ILP10msFR_ss_280_exp_sel_tourn10_repl_tourn10_mx_06_cx_10_': 'Tourn. 10, tourn. 10',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_010_f_34_': 'Heuristic, $\delta$ = 0.1',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_020_f_34_': 'Heuristic, $\delta$ = 0.2',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_030_f_34_': 'Heuristic, $\delta$ = 0.3',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_040_f_34_': 'Heuristic, $\delta$ = 0.4',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_050_f_34_': 'Heuristic, $\delta$ = 0.5',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_060_f_34_': 'Heuristic, $\delta$ = 0.6',
                '_c_MCS234s10ms_ss_990_exp_sfl_99ms_motes_12_top_heuristic_delta_070_f_34_': 'Heuristic, $\delta$ = 0.7',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_010_f_34_': 'Heuristic\n$\delta$ = 0.1',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_020_f_34_': 'Heuristic\n$\delta$ = 0.2',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_030_f_34_': 'Heuristic\n$\delta$ = 0.3',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_040_f_34_': 'Heuristic\n$\delta$ = 0.4',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_050_f_34_': 'Heuristic\n$\delta$ = 0.5',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_060_f_34_': 'Heuristic\n$\delta$ = 0.6',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_070_f_34_': 'Heuristic\n$\delta$ = 0.7',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_080_f_34_': 'Heuristic\n$\delta$ = 0.8',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_heuristic_delta_100_f_34_': 'Heuristic\n$\delta$ = 1',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_010_f_34_': 'Heuristic\n$\delta$ = 0.1',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_020_f_34_': 'Heuristic\n$\delta$ = 0.2',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_030_f_34_': 'Heuristic\n$\delta$ = 0.3',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_040_f_34_': 'Heuristic\n$\delta$ = 0.4',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_050_f_34_': 'Heuristic\n$\delta$ = 0.5',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_060_f_34_': 'Heuristic\n$\delta$ = 0.6',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_070_f_34_': 'Heuristic\n$\delta$ = 0.7',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_080_f_34_': 'Heuristic\n$\delta$ = 0.8',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_heuristic_delta_100_f_34_': 'Heuristic\n$\delta$ = 1',
                '_c_MCS234s10ms_ss_1530_exp_sfl_153ms_motes_12_top_gaheuristic_delta_000_f_34_': 'GA',
                '_c_MCS234s10ms_ss_3240_exp_sfl_324ms_motes_12_top_gaheuristic_delta_000_f_34_': 'GA',
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
             'distances': 'Average distance to parent (m)', 'tput': 'Throughput per slotframe'}
translateMotes = {'100': '100\nMotes', '200': '200\nMotes', '300': '300\nMotes', None: 'None'}
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
                              None: 'None', \
                              }
translateDistances = {'1': '100', \
                        '2': '200', \
                        '3': '300', \
                        '4': '400', \
                        '5': '500',
                        '01': '10',
                        None: 'None'}
translateOmega = {'1': '0.1', '125': '0.125', '15': '0.15', '175': '0.175', '2': '0.2', '3': '0.3', '4': '0.4', '5': '0.5', '6': '0.6', '7': '0.7', '8': '0.8', '9': '0.9'}
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
    if not os.path.exists(ga_solution) or os.path.getsize(ga_solution) == 0:
        raise 'There is no GA solution: {0}'.format(exp_dir)
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

#
# def getNrMotes(datafile):
#     with open(datafile) as f:
#         for line in f:
#             if '## numMotes =' in line:
#                 return int(line.split('=')[1].strip()) # num motes
#
def get_set_rgx(experiments, rgx = ''):
    candidates = set()
    for exp in experiments:
        regex_result = re.search(rgx, exp, re.IGNORECASE)
        if regex_result is not None:
            candidates.add(regex_result.group(1))
        # else:
        #     raise 'No %s indicator in experiment dir.' % rgx
    return candidates
#
# def detectInName(search_parameter, exp_dir):
#     cmd = "find {0} -ipath *{1}*/output_cpu0.dat".format(exp_dir, search_parameter)
#     listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
#     rgx = '[_\/]+%s_([A-Za-z0-9]+)_' % search_parameter
#     candidates = get_set_rgx(listFiles, rgx)
#     return candidates
#
def getExperimentName(exp):
    if exp in fileTranslate:
        return fileTranslate[exp]
    return exp

def getLabelName(name):
    if name in translate:
        return translate[name]
    return name

def ga_hop_count(n, parents, count):
    if n == 0:
        return 0
    elif n in parents and int(parents[n]) == 0:
        return 1
    else:
        return ga_hop_count(int(parents[n]), parents, count) + 1

def ga_calculate_hops(ind=None, datafile_dir=None):
    cmd = "find {0} -ipath \"*/ga_seed_*/ga_seed_*.json\"".format(datafile_dir)
    files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(files) == 0 or len(files) > 1:
        raise BaseException('There should only be one ga_seed settings file: {0} > {1}'.format(datafile_dir, files))

    # look if the GA built topologies or not by reading the settings file
    settings = None
    with open(files[0]) as json_file:
        settings = json.load(json_file)

    parents = {}
    # read in the parents in the correct manner, depending on the experiment
    # if settings["ga"]["building_topologies"] == 0:
    #     # get the simulator topology file
    #     simulator_topology_path = "{0}/simulator-topology.json".format(datafile_dir)
    #     simulator_topology = None
    #     with open(simulator_topology_path) as json_file:
    #         simulator_topology = json.load(json_file)
    #     for n in simulator_topology["simulationTopology"]:
    #         if n != "0":
    #             parents[int(n)] = int(simulator_topology["simulationTopology"][n]["parent"])
    # else:
    tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
    for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
        parents[ix + 1] = parent

    # calculate hop count per node
    hops = []
    for n, p in parents.iteritems():
        hops.append(ga_hop_count(n, parents, 0))
    return np.mean(hops)

def ga_calculate_minimal_hops(datafile_dir=None):
    cmd = "find {0} -ipath \"*/ga_seed_*/simulator-topology.json\"".format(datafile_dir)
    topology_files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(topology_files) == 0 or len(topology_files) > 1:
        raise BaseException('There should only be one simulator file: {0} > {1}'.format(datafile_dir, topology_files))

    cmd = "find {0} -ipath \"*/ga_seed_*/ga_seed_*.json\"".format(datafile_dir)
    settings_files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(settings_files) == 0 or len(settings_files) > 1:
        raise BaseException('There should only be one settings file: {0} > {1}'.format(datafile_dir, settings_files))

    # print(datafile_dir)
    # print(topology_files)
    # print(settings_files)
    # exit()

    # look if the GA built topologies or not by reading the settings file
    settings = None
    with open(settings_files[0]) as json_file:
        settings = json.load(json_file)

    parents = {}
    avg_minimal_hopcount = None
    # read in the parents in the correct manner, depending on the experiment
    # if settings["ga"]["building_topologies"] == 0:
    #     # get the simulator topology file
    #     # simulator_topology_path = "{0}/simulator-topology.json".format(datafile_dir)
    #     # simulator_topology = None
    #     # with open(simulator_topology_path) as json_file:
    #     #     simulator_topology = json.load(json_file)
    #     # for n in simulator_topology["simulationTopology"]:
    #     #     if n != "0":
    #     #         parents[int(n)] = int(simulator_topology["simulationTopology"][n]["parent"])
    #     raise BaseException("This calculation can not be done for building topologies equaling zero.")
    # else:
    dijkstra = Dijkstra()
    dijkstra.calculate(settings_file=settings_files[0], topology_file=topology_files[0])
    avg_minimal_hopcount = dijkstra.calculate_average()

    return avg_minimal_hopcount

def ga_calculate_modulation_count(ind, datafile_dir):
    cmd = "find {0} -ipath \"*/ga_seed_*/ga_seed_*.json\"".format(datafile_dir)
    files = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    if len(files) == 0 or len(files) > 1:
        raise BaseException('There should only be one ga_seed settings file: {0} > {1}'.format(datafile_dir, files))

    # look if the GA built topologies or not by reading the settings file
    # settings = None
    # with open(files[0]) as json_file:
    #     settings = json.load(json_file)

    prefParent = {}
    children = {}
    allocatedBondedSlots = {}
    numMotes = 1 # root
    tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
    for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
        if parent not in children:
            children[parent] = []
        children[parent].append(ix + 1)
        prefParent[ix + 1] = parent
        allocatedBondedSlots[ix + 1] = slots
        numMotes += 1
    allocatedBondedSlots[0] = 0

    reachRoot = {}
    for mote in range(1, numMotes):
        if reach_root(mote, prefParent, allocatedBondedSlots, []):
            reachRoot[mote] = 1
        else:
            reachRoot[mote] = 0
    reachRoot[0] = None

    PHYS = {'TSCH_SLOTBONDING_50_KBPS_PHY': 0, 'TSCH_SLOTBONDING_1000_KBPS_PHY': 0}

    # if settings["ga"]["building_topologies"] == 0:
    #     tuple_per_node = list(zip(ind, ind[1:]))[::2]
    #     for ix, (mcs, slots) in enumerate(tuple_per_node):
    #         if slots > 0: # only add when there were actual allocations
    #             dict_MCSs[MCSs[mcs]] += 1
    # else:
    tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
    for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
        if reachRoot[ix + 1] == 1:  # only add when there were actual allocations
            if mcs == 0:
                PHYS["TSCH_SLOTBONDING_50_KBPS_PHY"] += 1
            elif mcs == 1:
                PHYS["TSCH_SLOTBONDING_1000_KBPS_PHY"] += 1

    return PHYS

def ga_parse_evolution(dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA evolution - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    evolution_data = []
    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        generation = 0
        tput_prev = -1
        airtime_prev = -1
        tput_rel = 0.0
        airtime_rel = 0.0
        for tput, airtime in data["results"]["best_ind_evolution"]:
            if tput == -50 or tput_prev <= 0:
                if tput == -50:
                    tput = -1
                tput_rel = 0.0
            else:
                tput_rel = (tput/float(tput_prev)) - 1.0
            if airtime == 10000 or airtime_prev <= 0:
                if airtime == 10000:
                    airtime = -1
                airtime_rel = 0.0
            else:
                airtime_rel = 1.0 - (airtime/float(airtime_prev))
            evolution_data.append({'experiment_type': experimentType, 'dataDir': dataDir, 'datafile': datafile, 'generation': generation, 'tput': tput, 'tput_rel': tput_rel,'airtime': airtime, 'airtime_rel': airtime_rel})
            tput_prev = tput
            airtime_prev = airtime
            generation += 1

    return evolution_data

def ga_parse_unique_inds(dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA unique individuals - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    unique_inds = []
    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        generation = 0
        for inds in data["results"]["unique_individuals_evolution"]:
            unique_inds.append({'experiment_type': experimentType, 'dataDir': dataDir, 'datafile': datafile, 'generation': generation, 'unique_inds': inds})
            generation += 1

    return unique_inds

def reach_root_tmp(node, par_parents, par_allocatedBondedSlots, parents):
    node = str(node)
    if str(par_parents[node]) == '0':
        return True
    elif str(par_parents[node]) != '0' and par_allocatedBondedSlots[node] == 0:
        return False
    elif node not in par_parents:
        return False
    elif str(par_parents[node]) in parents:
        return False
    parents.append(str(par_parents[node]))
    return reach_root_tmp(str(par_parents[node]), par_parents, par_allocatedBondedSlots, parents)

def ga_parse_nr_reach_root(dataDir, experiment_type, nr_reach_root):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga-schedule.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA nr reach root - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        parents_tmp = copy.deepcopy(data["parents"])
        nr_slots_tmp = copy.deepcopy(data["nr_slots"])
        reach_root_count = 0
        for mote, nr_slots_mote in data["nr_slots"].iteritems():
            if int(mote) != 0 and reach_root_tmp(str(mote), parents_tmp, nr_slots_tmp, []):
                reach_root_count += 1

        nr_reach_root = nr_reach_root.append({'experiment_type': experiment_type, 'dataDir': dataDir, 'datafile': datafile, 'nr_reach_root': reach_root_count}, ignore_index=True)

    return nr_reach_root

def ga_parse_nr_slots(dataDir, experiment_type, nr_slots_df):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga-schedule.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1]  # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA nr slots - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        parents_tmp = copy.deepcopy(data["parents"])
        nr_slots_tmp = copy.deepcopy(data["nr_slots"])

        for mote, nr_slots_mote in data["nr_slots"].iteritems():
            if int(mote) != 0 and reach_root_tmp(str(mote), parents_tmp, nr_slots_tmp, []):
                # print("come here", mote, reliability)
                nr_slots_df = nr_slots_df.append({'experiment_type': experiment_type, 'dataDir': dataDir, 'datafile': datafile, 'mote': int(mote), 'nr_slots': nr_slots_mote}, ignore_index=True)
            elif int(mote) != 0 and not reach_root_tmp(str(mote), parents_tmp, nr_slots_tmp, []):
                print('could not reach root')
    return nr_slots_df

def ga_parse_reliabilities(dataDir, experiment_type, reliabilities):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga-schedule.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA reliabilities - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        parents_tmp = copy.deepcopy(data["parents"])
        nr_slots_tmp = copy.deepcopy(data["nr_slots"])
        
        for mote, reliability in data["reliabilities"].iteritems():
            if int(mote) != 0 and reach_root_tmp(str(mote), parents_tmp, nr_slots_tmp, []):
                # print("come here", mote, reliability)
                reliabilities = reliabilities.append({'experiment_type': experiment_type, 'dataDir': dataDir, 'datafile': datafile, 'mote': int(mote), 'reliability': reliability}, ignore_index=True)
            elif int(mote) != 0 and not reach_root_tmp(str(mote), parents_tmp, nr_slots_tmp, []):
                print('could not reach root')
    return reliabilities

def ga_parse_valid_inds(dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA valid individuals - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    valid_inds = []
    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        generation = 0
        for inds in data["results"]["valid_individuals_evolution"]:
            valid_inds.append({'experiment_type': experimentType, 'dataDir': dataDir, 'datafile': datafile, 'generation': generation, 'valid_inds': inds})
            generation += 1

    return valid_inds

def ga_parse_infeasible_inds(dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA infeasible individuals - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    infeasible_inds = []
    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        generation = 0
        for inds in data["results"]["infeasible_inds"]:
            infeasible_inds.append({'experiment_type': experimentType, 'dataDir': dataDir, 'datafile': datafile, 'generation': generation, 'infeasible_inds': inds})
            generation += 1

    return infeasible_inds

def ga_parse_solutions(dataDir, experiment_type, ga_data):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA solution - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        # calculate the minimal number of hops possible in the given topology
        avg_minimal_hopcount = ga_calculate_minimal_hops(datafile_dir=os.path.dirname(datafile))

        # simulator_topology_path = "{0}/simulator-topology.json".format(os.path.dirname(datafile))
        # calculate the hop count of the best individual
        avg_hopcount = ga_calculate_hops(ind=data['results']['best_ind']['ind'], datafile_dir=os.path.dirname(datafile))

        normalized_airtime = None
        if data['results']['best_ind']['tput'] > 0.0:
            normalized_airtime = data['results']['best_ind']['airtime'] / float(data['results']['best_ind']['tput'])

        # a dictionary mapping the modulation to the number of count is has been used
        dict_modulations = ga_calculate_modulation_count(ind=data['results']['best_ind']['ind'], datafile_dir=os.path.dirname(datafile))

        # enable the following lines to calculate the confusion matrix values
        # also enable the correct lines in the dict_results further below to also store the confusion matrix values
        true_positive_rate = None
        false_negative_rate = None
        true_negative_rate = None
        false_positive_rate = None
        if data['results']['time']['heuristic']['true_positives'] + data['results']['time']['heuristic']['false_negatives'] > 0:
            true_positive_rate = data['results']['time']['heuristic']['true_positives'] / float(data['results']['time']['heuristic']['true_positives'] + data['results']['time']['heuristic']['false_negatives'])
        else:
            true_positive_rate = np.NaN
        if data['results']['time']['heuristic']['true_positives'] + data['results']['time']['heuristic']['false_negatives'] > 0:
            false_negative_rate = data['results']['time']['heuristic']['false_negatives'] / float(data['results']['time']['heuristic']['true_positives'] + data['results']['time']['heuristic']['false_negatives'])
        else:
            false_negative_rate = np.NaN
        if data['results']['time']['heuristic']['true_negatives'] + data['results']['time']['heuristic']['false_positives'] > 0:
            true_negative_rate = data['results']['time']['heuristic']['true_negatives'] / float(data['results']['time']['heuristic']['true_negatives'] + data['results']['time']['heuristic']['false_positives'])
        else:
            true_negative_rate = np.NaN
        if data['results']['time']['heuristic']['true_negatives'] + data['results']['time']['heuristic']['false_positives'] > 0:
            false_positive_rate = data['results']['time']['heuristic']['false_positives'] / float(data['results']['time']['heuristic']['true_negatives'] + data['results']['time']['heuristic']['false_positives'])
        else:
            false_positive_rate = np.NaN

        # print data['results']['time']['heuristic']['true_positives']
        # print data['results']['time']['heuristic']['true_negatives']
        # print data['results']['time']['heuristic']['false_positives']
        # print data['results']['time']['heuristic']['false_negatives']


        def replace_slotframe_lengths(s_size):
            if s_size == '153':
                return '261'
            elif s_size == '324':
                return '423'
            else:
                raise Exception('Wrong slot frame size.')

        slotframe_size = None
        delta = None
        if True:
            slotframe_size = replace_slotframe_lengths(str(int(float(datafile.split('_ss')[1].split('_')[1]) / 10)))
            delta = datafile.split('_delta')[1].split('_')[1]
            topConfig = datafile.split('_top')[1].split('_')[1]
        #     print(slotframe_size)
        #     print(delta)
        # exit()

        dict_results = {'experiment_type': experiment_type,
                        'ss': slotframe_size,
                        'delta': delta,
                        'topConfig': topConfig,
                        'path': datafile,
                        'seed': int(datafile.split('seed')[1].split('_')[1]),
                        'ind': data['results']['best_ind']['ind'],
                        'tput': data['results']['best_ind']['tput'],
                        'airtime': data['results']['best_ind']['airtime'],
                        'normalized_airtime': normalized_airtime,
                        'avg_hopcount': avg_hopcount,
                        'avg_minimal_hopcount': avg_minimal_hopcount,
                        'gen_best_ind': int(data['results']['best_ind']['generation']),
                        'total_time': data['results']['time']['total_time'],
                        'total_ind': data['results']['individuals']['total'],
                        'filtered_ind': data['results']['individuals']['filtered'],
                        'unique_ind': data['results']['individuals']['unique'],
                        'valid_ind': data['results']['individuals']['valid'],
                        'best_ind': str(data['results']['best_ind']['ind']),
                        'ilp_invalids': 0,
                        'fast_invalids': 0,
                        'diff_percentage_invalids': 0,
                        # 'ilp_invalids': data["results"]["time"]["feasibility"]["ilp_invalids"],
                        # 'fast_invalids': data["results"]["time"]["feasibility"]["fast_invalids"],
                        # 'diff_percentage_invalids': data["results"]["time"]["feasibility"]["fast_invalids"]/float(data["results"]["time"]["feasibility"]["ilp_invalids"]),
                        'total_feasibility_checks': data["results"]["time"]["feasibility"]["nr"],
                        'total_heuristic_checks': data["results"]["time"]["heuristic"]["nr"],
                        # 'diff_percentage_feasible': data["results"]["time"]["feasibility"]["feasible"]/float(data["results"]["time"]["feasibility"]["nr"]),
                        # 'diff_percentage_heuristic': data["results"]["time"]["heuristic"]["feasible"]/float(data["results"]["time"]["heuristic"]["nr"]),
                        'total_time_heuristic': data['results']['time']['heuristic']['total'], \
                        'total_time_feasibility': data['results']['time']['feasibility']['total'],
                        'total_time_percentage': data['results']['time']['heuristic']['total'] / float(data['results']['time']['feasibility']['total']),
                        'true_negatives': data['results']['time']['heuristic']['true_negatives'],
                        'true_negative_rate': true_negative_rate,
                        'false_positives': data['results']['time']['heuristic']['false_positives'],
                        'false_positive_rate': false_positive_rate,
                        'true_positives': data['results']['time']['heuristic']['true_positives'],
                        'true_positive_rate': true_positive_rate,
                        'false_negatives': data['results']['time']['heuristic']['false_negatives'],
                        'false_negative_rate': false_negative_rate}

        # 'ix_hof': int(data['results']['best_ind']['index']),}

        for modulation, cnt in dict_modulations.iteritems():
            dict_results[modulation] = cnt

        ga_data = ga_data.append(dict_results, ignore_index=True)

    return ga_data

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

def calculate_descendants(node=0, children=[], l_descendants=[]):
    if node not in l_descendants:
        l_descendants[node] = []
    if node in children:
        l_descendants[node] += children[node]
        for c in children[node]:
            calculate_descendants(node=c, children=children, l_descendants=l_descendants)
            l_descendants[node] += l_descendants[c]

def ga_parse_deep_analysis(minPDR, dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing deep analysis - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    if len(listFiles) == 0:
        return {}

    picked_best_reliability_overall = 0
    picked_not_best_reliability_overall = 0
    picked_best_reliability_of_parent = 0
    picked_not_best_reliability_of_parent = 0

    picked_best_rate_overall = 0
    picked_not_best_rate_overall = 0
    picked_best_rate_of_parent = 0
    picked_not_best_rate_of_parent = 0

    best_rate_picked_best_reliability_overall = 0
    best_rate_picked_not_best_reliability_overall = 0
    best_rate_picked_best_reliability_of_parent = 0
    best_rate_picked_not_best_reliability_of_parent = 0
    best_reliability_picked_best_rate_overall = 0
    best_reliability_picked_not_best_rate_overall = 0
    best_reliability_picked_best_rate_of_parent = 0
    best_reliability_picked_not_best_rate_of_parent = 0

    overall_found_equal_reliabilities_with_higher_rate = 0
    overall_found_equal_reliabilities_with_lower_or_equal_rate = 0
    parent_found_equal_reliabilities_with_higher_rate = 0
    parent_found_equal_reliabilities_with_lower_or_equal_rate = 0

    parent_closer_to_root = 0
    parent_more_away_from_root = 0
    diff_distances_to_root = []

    lsts_possible_parents = []
    lst_of_all_rates = []

    diff_with_best_reliability_overall = []
    diff_with_best_reliability_of_parent = []
    diff_rates_with_higher_reliabilities_overall = []
    diff_rates_with_higher_reliabilities_parent = []

    num_min_distance_to_root_of_possible_parents = 0
    num_max_distance_to_root_of_possible_parents = 0
    num_min_distance_to_node = 0
    num_max_distance_to_node = 0
    num_parent_with_min_hopcount = 0
    num_parent_with_max_hopcount = 0
    num_parent_with_min_descendants = 0
    num_parent_with_max_descendants = 0
    num_parent_better_reliability = 0
    num_parent_equal_reliability = 0
    num_parent_less_reliability = 0
    num_parent_better_rate = 0
    num_parent_equal_rate = 0
    num_parent_less_rate = 0
    num_parent_equal_reliability_with_lessparents = 0
    num_parent_better_reliability_with_lessparents = 0
    num_parent_less_reliability_with_lessparents = 0
    num_parent_equal_rate_with_lessparents = 0
    num_parent_better_rate_with_lessparents = 0
    num_parent_less_rate_with_lessparents = 0
    num_choose_root_while_possible = 0
    num_choose_root_not_while_possible = 0

    total_num_motes = 0
    total_num_motes_without_root = 0
    total_has_parent_with_less_reliability = 0
    total_has_parent_with_less_rate = 0

    dict_rates = {
      "QPSK_FEC_1_2_FR_2": 50,
      "QPSK_FEC_1_2": 100,
      "QPSK_FEC_3_4": 150,
      "QAM_16_FEC_1_2": 200,
      "QAM_16_FEC_3_4": 300
    }

    results = {}

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        simulator_topology_path = "{0}/simulator-topology.json".format(os.path.dirname(datafile))
        simulator_topology = None
        with open(simulator_topology_path) as json_file:
            simulator_topology = json.load(json_file)

        MCSs = simulator_topology["modulations"]
        if MCSs == [u'QPSK_FEC_1_2_FR_2', u'QPSK_FEC_3_4']:
            MCSs = [u'QPSK_FEC_1_2_FR_2', u'QPSK_FEC_1_2', u'QPSK_FEC_3_4']

        prefParent = {}
        children = {}
        allocatedBondedSlots = {}
        numMotes = 1 # root
        ind = data['results']['best_ind']['ind']
        tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            if parent not in children:
                children[parent] = []
            children[parent].append(ix + 1)
            prefParent[ix + 1] = parent
            allocatedBondedSlots[ix + 1] = slots
            numMotes += 1
        allocatedBondedSlots[0] = 0

        reachRoot = {}
        for mote in range(1, numMotes):
            if reach_root(mote, prefParent, allocatedBondedSlots, []):
                reachRoot[mote] = 1
            else:
                reachRoot[mote] = 0
        reachRoot[0] = None

        nodes = []
        reliabilities = {}
        rates = {}
        parents = {}
        # 1) find all the reliabilities of all the nodes
        ind = data['results']['best_ind']['ind']
        tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            # if slots > 0:
            if reachRoot[ix + 1] == 1:
                mcs_name = MCSs[mcs]
                rel = simulator_topology["simulationTopology"][str(ix + 1)]["reliability"][mcs_name][str(parent)]
                distance_node_to_root = float(simulator_topology["simulationTopology"][str(ix + 1)]["distance_to_root"])
                if rel > minPDR:
                    parents[ix + 1] = parent
                    reliabilities[ix + 1] = rel
                    rates[ix + 1] = dict_rates[mcs_name]
                    lst_of_all_rates.append(dict_rates[mcs_name])
                    distance_parent_to_root = float(simulator_topology["simulationTopology"][str(parent)]["distance_to_root"])
                    nodes.append(ix+1)
                    if distance_parent_to_root < distance_node_to_root:
                        parent_closer_to_root += 1
                    else:
                        parent_more_away_from_root += 1
                    diff_distances_to_root.append(distance_node_to_root - distance_parent_to_root)

        # exit()
        # filter nodes
        # filter out all the descendants that can not reach the root
        # descendants = {}
        # calculate_descendants(node=0, children=children, l_descendants=descendants)
        # filtered_descendants = {}
        # average_nr_descendants = []
        # for n in nodes:
        #     filtered_descendants[n] = [d for d in descendants[n] if reachRoot[d] == 1]
        #     average_nr_descendants.append(len(filtered_descendants[n]))
        # nodes = [n for n in nodes if len(filtered_descendants[n]) > 2]
        

        nodes = [n for n in nodes if rates[n] == 50]
        # print datafile
        # print nodes

        # print descendants
        # print filtered_descendants
        # print nodes
        # exit()
        # 2) find all possible reliabilities per node
        all_possible_reliabilities = {}
        all_possible_rates = {}
        parent_possible_reliabilities = {}
        parent_possible_rates = {}
        for n in nodes:
            all_possible_reliabilities[n] = []
            all_possible_rates[n] = []
            parent_possible_reliabilities[n] = []
            parent_possible_rates[n] = []
            lst_possible_parents = []
            for mcs, p_rel in simulator_topology["simulationTopology"][str(n)]["reliability"].iteritems():
                for p, rel in p_rel.iteritems():
                    if rel > minPDR:
                        if p not in lst_possible_parents:
                            lst_possible_parents.append(p)
                        all_possible_reliabilities[n].append(rel)
                        all_possible_rates[n].append(dict_rates[mcs])
                        if rel > reliabilities[n]:
                            diff_rates_with_higher_reliabilities_overall.append(dict_rates[mcs] - rates[n])

                        if abs(reliabilities[n] - rel) < 0.0000001:
                            if rates[n] < dict_rates[mcs]:
                                # BE AWARE! if you find one of these, this is interesting!
                                overall_found_equal_reliabilities_with_higher_rate += 1
                            else:
                                overall_found_equal_reliabilities_with_lower_or_equal_rate += 1

                        if p == str(parents[n]):
                            parent_possible_reliabilities[n].append(rel)
                            parent_possible_rates[n].append(dict_rates[mcs])
                            if rel > reliabilities[n]:
                                diff_rates_with_higher_reliabilities_parent.append(dict_rates[mcs] - rates[n])
                            if abs(reliabilities[n] - rel) < 0.0000001:
                                if rates[n] < dict_rates[mcs]:
                                    parent_found_equal_reliabilities_with_higher_rate += 1
                                else:
                                    # print('For node {0} (parent {6}) , picked rel {1} with rate {2}, now found rel {3} with rate {4} ({5}).'.format(n, reliabilities[n], rates[n], rel, dict_rates[mcs], mcs, p))
                                    parent_found_equal_reliabilities_with_lower_or_equal_rate += 1

            # for p in lst_possible_parents:
            # print("For node {0}, these are the possible parents: {1}".format(n, lst_possible_parents))
            distances_to_root_of_possible_parents = []
            x = float(simulator_topology["simulationTopology"][str(n)]['x'])
            y = float(simulator_topology["simulationTopology"][str(n)]['y'])
            distances_to_node = []
            hopcounts_of_parents = []
            filtered_lst_possible_parents = lst_possible_parents[:]
            filtered_lst_possible_parents = [p for p in filtered_lst_possible_parents if (reachRoot[int(p)] == 1 or int(p) == 0)]
            for p in filtered_lst_possible_parents:
                distances_to_root_of_possible_parents.append(float(simulator_topology["simulationTopology"][p]['distance_to_root']))
                x_parent = float(simulator_topology["simulationTopology"][p]['x'])
                y_parent = float(simulator_topology["simulationTopology"][p]['y'])
                distances_to_node.append(math.sqrt((x_parent - x)**2 + (y_parent - y)**2))

            # get the distance of the parent to the root
            distance_to_root_of_parent = float(simulator_topology["simulationTopology"][str(parents[n])]['distance_to_root'])
            # the distance to the root from the parent is the minimum distance available
            if (abs(distance_to_root_of_parent - min(distances_to_root_of_possible_parents)) < 0.0000001):
                num_min_distance_to_root_of_possible_parents += 1
            # the distance to the root from the parent is the maximum distance available
            if (abs(distance_to_root_of_parent - max(distances_to_root_of_possible_parents)) < 0.0000001):
                num_max_distance_to_root_of_possible_parents += 1

            x_parent = float(simulator_topology["simulationTopology"][str(parents[n])]['x'])
            y_parent = float(simulator_topology["simulationTopology"][str(parents[n])]['y'])
            # get the distance of the parent to the root
            distance_to_node_of_parent = float(math.sqrt((x_parent - x)**2 + (y_parent - y)**2))
            # the distance to the root from the parent is the minimum distance available
            if (abs(distance_to_node_of_parent - min(distances_to_node)) < 0.0000001):
                num_min_distance_to_node += 1
            # the distance to the root from the parent is the maximum distance available
            if (abs(distance_to_node_of_parent - max(distances_to_node)) < 0.0000001):
                num_max_distance_to_node += 1

            # filter out all the parents that can not reach the root
            descendants = {}
            calculate_descendants(node=0, children=children, l_descendants=descendants)

            has_parent_with_less_reliability = False
            has_parent_with_less_rate = False
            descendants_of_parents = []
            hopcounts_of_parents = []
            rates_of_p = []
            reliabilities_of_p = []
            filtered_lst_possible_parents = lst_possible_parents[:]
            filtered_lst_possible_parents = [p for p in filtered_lst_possible_parents if (reachRoot[int(p)] == 1 or int(p) == 0)]
            for p in filtered_lst_possible_parents:
                hopcounts_of_parents.append(ga_hop_count(int(p), parents, 0))
                # print("parent {0} its descendants {1}".format(p, descendants[int(p)]))
                descendants_of_parents.append(len(descendants[int(p)]))
                if int(parents[n]) != 0 and int(p) != 0:
                    rates_of_p.append(rates[int(p)])
                    reliabilities_of_p.append(reliabilities[int(p)])
                    if reliabilities[int(p)] < reliabilities[int(parents[n])]:
                        has_parent_with_less_reliability = True
                    if rates[int(p)] < rates[int(parents[n])]:
                        has_parent_with_less_rate = True

            hopcount_parent = ga_hop_count(int(parents[n]), parents, 0)
            # the distance to the root from the parent is the minimum distance available
            if hopcount_parent == min(hopcounts_of_parents):
                num_parent_with_min_hopcount += 1
            # the distance to the root from the parent is the maximum distance available
            if hopcount_parent == max(hopcounts_of_parents):
                num_parent_with_max_hopcount += 1

            descendants_parent = len(descendants[int(parents[n])])
            # the distance to the root from the parent is the minimum distance available
            if descendants_parent == min(descendants_of_parents):
                num_parent_with_min_descendants += 1
            # the distance to the root from the parent is the maximum distance available
            if descendants_parent == max(descendants_of_parents):
                num_parent_with_max_descendants += 1

            # if 100 not in all_possible_rates[n] and 150 not in all_possible_rates[n]:
            if int(parents[n]) != 0:
                if has_parent_with_less_reliability:
                    total_has_parent_with_less_reliability += 1
                if has_parent_with_less_rate:
                    total_has_parent_with_less_rate += 1
                total_num_motes_without_root += 1
                if abs(reliabilities[n] - reliabilities[int(parents[n])]) < 0.0000001:
                    num_parent_equal_reliability += 1
                    if has_parent_with_less_reliability:
                        num_parent_equal_reliability_with_lessparents += 1
                if reliabilities[int(parents[n])] > reliabilities[n]:
                    num_parent_better_reliability += 1
                    if has_parent_with_less_reliability:
                        num_parent_better_reliability_with_lessparents += 1
                if reliabilities[int(parents[n])] < reliabilities[n]:
                    num_parent_less_reliability += 1
                    if has_parent_with_less_reliability:
                        num_parent_less_reliability_with_lessparents += 1
                if abs(rates[n] - rates[int(parents[n])]) < 0.0000001:
                    num_parent_equal_rate += 1
                    if has_parent_with_less_rate:
                        num_parent_equal_rate_with_lessparents += 1
                # print(rates[int(parents[n])], rates[n])
                if rates[int(parents[n])] > rates[n]:
                    num_parent_better_rate += 1
                    if has_parent_with_less_rate:
                        num_parent_better_rate_with_lessparents += 1
                if rates[int(parents[n])] < rates[n]:
                    num_parent_less_rate += 1
                    if has_parent_with_less_rate:
                        num_parent_less_rate_with_lessparents += 1

            if str(0) in lst_possible_parents:
                if int(parents[n]) == 0:
                    num_choose_root_while_possible += 1
                elif int(parents[n]) != 0:
                    num_choose_root_not_while_possible += 1
                else:
                    assert False

            # calculate the descendants of node n
            # descendants = {}
            # calculate_descendants(node=(ix + 1), children=children, l_descendants=descendants)
            # print(prefParent)
            # print(reachRoot)
            # print("Descendants of node {0}: {1}".format((ix + 1), descendants))

            # print("For node {0} distance of parent {2} to root: {1}".format(n, distance_to_root_of_parent, parents[n]))
            # print("For node {0} distance to parent {2}: {1}".format(n, math.sqrt((x_parent - x)**2 + (y_parent - y)**2), parents[n]))

            total_num_motes += 1

            lsts_possible_parents.append(len(lst_possible_parents))

        # 3) determine if it took the best of all reliabilities or of all parents
        for n in nodes:
            if abs(max(all_possible_reliabilities[n]) - reliabilities[n]) < 0.00000001:
                picked_best_reliability_overall += 1
                # check if it also picked the best rate
                if max(all_possible_rates[n]) == rates[n]:
                    best_reliability_picked_best_rate_overall += 1
                else:
                    best_reliability_picked_not_best_rate_overall += 1
            else:
                picked_not_best_reliability_overall += 1
                diff_with_best_reliability_overall.append(max(all_possible_reliabilities[n]) - reliabilities[n])
                # print(datafile)
                # print(max(all_possible_reliabilities[n]) - reliabilities[n])
                # print('For node {0}, picked rel {1} with rate {2}, while max is {4}.'.format(n, reliabilities[n], rates[n], p, max(all_possible_reliabilities[n])))
                # exit()
                
            if abs(max(parent_possible_reliabilities[n]) - reliabilities[n]) < 0.00000001:
                picked_best_reliability_of_parent += 1
                # check if it also picked the best rate of the parent
                if max(parent_possible_rates[n]) == rates[n]:
                    best_reliability_picked_best_rate_of_parent += 1
                else:
                    best_reliability_picked_not_best_rate_of_parent += 1
            else:
                picked_not_best_reliability_of_parent += 1
                diff_with_best_reliability_of_parent.append(max(parent_possible_reliabilities[n]) - reliabilities[n])

            if max(all_possible_rates[n]) == rates[n]:
                picked_best_rate_overall += 1
                # check if it picked also the best reliability overall
                if abs(max(all_possible_reliabilities[n]) - reliabilities[n]) < 0.00000001:
                    best_rate_picked_best_reliability_overall += 1
                else:
                    best_rate_picked_not_best_reliability_overall += 1
            else:
                picked_not_best_rate_overall += 1

            if max(parent_possible_rates[n]) == rates[n]:
                picked_best_rate_of_parent += 1
                # check if it also picked the best reliability of the parent
                if abs(max(parent_possible_reliabilities[n]) - reliabilities[n]) < 0.00000001:
                    best_rate_picked_best_reliability_of_parent += 1
                else:
                    best_rate_picked_not_best_reliability_of_parent += 1
            else:
                picked_not_best_rate_of_parent += 1

        # print(reliabilities)
        # print(all_possible_reliabilities)
        # print(parent_possible_reliabilities)
        # print(rates)
        # print(all_possible_rates)
        # print(parent_possible_rates)

    # if best_rate_picked_best_reliability_overall + best_rate_picked_not_best_reliability_overall == 0:
    #     print nodes
    #     print datafile
    #     assert False

    # print('reliabilities')
    # print(picked_best_reliability_overall)
    results['picked_best_reliability_overall'] = picked_best_reliability_overall
    # print(picked_best_reliability_of_parent)
    results['picked_not_best_reliability_overall'] = picked_not_best_reliability_overall
    # print(picked_not_best_reliability_overall)
    results['picked_best_reliability_of_parent'] = picked_best_reliability_of_parent
    # print(picked_not_best_reliability_of_parent)
    results['picked_not_best_reliability_of_parent'] = picked_not_best_reliability_of_parent

    # print('rates')
    # print(picked_best_rate_overall)
    results['picked_best_rate_overall'] = picked_best_rate_overall
    # print(picked_best_rate_of_parent)
    results['picked_best_rate_of_parent'] = picked_best_rate_of_parent
    # print(picked_not_best_rate_overall)
    results['picked_not_best_rate_overall'] = picked_not_best_rate_overall
    # print(picked_not_best_rate_of_parent)
    results['picked_not_best_rate_of_parent'] = picked_not_best_rate_of_parent

    # print('combined')
    # print(best_rate_picked_best_reliability_overall)
    results['best_rate_picked_best_reliability_overall'] = best_rate_picked_best_reliability_overall
    # print(best_rate_picked_not_best_reliability_overall)
    results['best_rate_picked_not_best_reliability_overall'] = best_rate_picked_not_best_reliability_overall
    # print(best_rate_picked_best_reliability_of_parent)
    results['best_rate_picked_best_reliability_of_parent'] = best_rate_picked_best_reliability_of_parent
    # print(best_rate_picked_not_best_reliability_of_parent)
    results['best_rate_picked_not_best_reliability_of_parent'] = best_rate_picked_not_best_reliability_of_parent
    # print(best_reliability_picked_best_rate_overall)
    results['best_reliability_picked_best_rate_overall'] = best_reliability_picked_best_rate_overall
    # print(best_reliability_picked_not_best_rate_overall)
    results['best_reliability_picked_not_best_rate_overall'] = best_reliability_picked_not_best_rate_overall
    # print(best_reliability_picked_best_rate_of_parent)
    results['best_reliability_picked_best_rate_of_parent'] = best_reliability_picked_best_rate_of_parent
    # print(best_reliability_picked_not_best_rate_of_parent)
    results['best_reliability_picked_not_best_rate_of_parent'] = best_reliability_picked_not_best_rate_of_parent

    # print('best rate or not')
    # print(overall_found_equal_reliabilities_with_higher_rate)
    results['overall_found_equal_reliabilities_with_higher_rate'] = overall_found_equal_reliabilities_with_higher_rate
    # print(overall_found_equal_reliabilities_with_lower_or_equal_rate)
    results['overall_found_equal_reliabilities_with_lower_or_equal_rate'] = overall_found_equal_reliabilities_with_lower_or_equal_rate
    # print(parent_found_equal_reliabilities_with_higher_rate)
    results['parent_found_equal_reliabilities_with_higher_rate'] = parent_found_equal_reliabilities_with_higher_rate
    # print(parent_found_equal_reliabilities_with_lower_or_equal_rate)
    results['parent_found_equal_reliabilities_with_lower_or_equal_rate'] = parent_found_equal_reliabilities_with_lower_or_equal_rate

    # print("distances")
    # print(parent_closer_to_root)
    results['parent_closer_to_root'] = parent_closer_to_root
    # print(parent_more_away_from_root)
    results['parent_more_away_from_root'] = parent_more_away_from_root
    # print(diff_distances_to_root)
    results['diff_distances_to_root'] = copy.deepcopy(diff_distances_to_root)
    results['lsts_possible_parents'] = lsts_possible_parents
    results['lst_of_all_rates'] = lst_of_all_rates

    results['num_min_distance_to_root_of_possible_parents'] = num_min_distance_to_root_of_possible_parents
    results['num_max_distance_to_root_of_possible_parents'] = num_max_distance_to_root_of_possible_parents
    results['num_min_distance_to_node'] = num_min_distance_to_node
    results['num_max_distance_to_node'] = num_max_distance_to_node
    results['num_parent_with_min_hopcount'] = num_parent_with_min_hopcount
    results['num_parent_with_max_hopcount'] = num_parent_with_max_hopcount
    results['num_parent_with_min_descendants'] = num_parent_with_min_descendants
    results['num_parent_with_max_descendants'] = num_parent_with_max_descendants
    results['num_parent_better_reliability'] = num_parent_better_reliability
    results['num_parent_equal_reliability'] = num_parent_equal_reliability
    results['num_parent_less_reliability'] = num_parent_less_reliability
    results['num_parent_better_rate'] = num_parent_better_rate
    results['num_parent_equal_rate'] = num_parent_equal_rate
    results['num_parent_less_rate'] = num_parent_less_rate
    results['num_parent_better_reliability_with_lessparents'] = num_parent_better_reliability_with_lessparents
    results['num_parent_equal_reliability_with_lessparents'] = num_parent_equal_reliability_with_lessparents
    results['num_parent_less_reliability_with_lessparents'] = num_parent_less_reliability_with_lessparents
    results['num_parent_better_rate_with_lessparents'] = num_parent_better_rate_with_lessparents
    results['num_parent_equal_rate_with_lessparents'] = num_parent_equal_rate_with_lessparents
    results['num_parent_less_rate_with_lessparents'] = num_parent_less_rate_with_lessparents

    results['num_choose_root_while_possible'] = num_choose_root_while_possible
    results['num_choose_root_not_while_possible'] = num_choose_root_not_while_possible
    results['total_num_motes_without_root'] = total_num_motes_without_root
    results['total_num_motes'] = total_num_motes
    results['total_has_parent_with_less_reliability'] = total_has_parent_with_less_reliability
    results['total_has_parent_with_less_rate'] = total_has_parent_with_less_rate

    results['diff_with_best_reliability_overall'] = copy.deepcopy(diff_with_best_reliability_overall)
    results['diff_with_best_reliability_of_parent'] = copy.deepcopy(diff_with_best_reliability_of_parent)
    results['diff_rates_with_higher_reliabilities_overall'] = copy.deepcopy(diff_rates_with_higher_reliabilities_overall)
    results['diff_rates_with_higher_reliabilities_parent'] = copy.deepcopy(diff_rates_with_higher_reliabilities_parent)

    assert best_reliability_picked_best_rate_overall + best_reliability_picked_not_best_rate_overall == picked_best_reliability_overall
    assert best_reliability_picked_best_rate_of_parent + best_reliability_picked_not_best_rate_of_parent == picked_best_reliability_of_parent
    assert best_rate_picked_best_reliability_overall + best_rate_picked_not_best_reliability_overall == picked_best_rate_overall
    assert best_rate_picked_best_reliability_of_parent + best_rate_picked_not_best_reliability_of_parent == picked_best_rate_of_parent

    return results

def ga_parse_individual_reliabilities(minPDR, dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/ga.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing individual reliabilities - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    for datafile in listFiles:
        # print datafile
        validate(os.path.dirname(datafile))
        data = None
        with open(datafile) as json_file:
            data = json.load(json_file)

        simulator_topology_path = "{0}/simulator-topology.json".format(os.path.dirname(datafile))
        simulator_topology = None
        with open(simulator_topology_path) as json_file:
            simulator_topology = json.load(json_file)

        MCSs = simulator_topology["modulations"]
        if MCSs == [u'QPSK_FEC_1_2_FR_2', u'QPSK_FEC_3_4']:
            MCSs = [u'QPSK_FEC_1_2_FR_2', u'QPSK_FEC_1_2', u'QPSK_FEC_3_4']

        ind = data['results']['best_ind']['ind']
        tuple_per_node = list(zip(ind, ind[1:], ind[2:]))[::3]
        for ix, (mcs, slots, parent) in enumerate(tuple_per_node):
            if slots > 0:
                mcs_name = MCSs[mcs]
                rel = simulator_topology["simulationTopology"][str(ix + 1)]["reliability"][mcs_name][str(parent)]
                rel = round(rel, 3)
                if rel > minPDR:
                    reliabilities.append(rel)
                else:
                    assert False

    return reliabilities

def ga_parse_all_reliabilities(minPDR, dataDir, experiment_type):
    cmd = "find {0} -ipath \"*{1}*/error.log\"".format(dataDir, experiment_type)
    listWithAllowedIterations = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listWithAllowedIterationDirs = [os.path.dirname(x) for x in listWithAllowedIterations if '_failed' not in os.path.dirname(x)]
    cmd = "find {0} -ipath \"*{1}*/simulator-topology.json\"".format(dataDir, experiment_type)
    listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it
    listFiles = [x for x in listFiles if os.path.dirname(x) in listWithAllowedIterationDirs]
    print "Parsing GA link reliabilities - Processing %d file(s) in %s." % (len(listFiles), str(dataDir))

    reliabilities = []

    for datafile in listFiles:
        links = set()
        # print datafile
        validate(os.path.dirname(datafile))
        simulator_topology = None
        with open(datafile) as json_file:
            simulator_topology = json.load(json_file)

        for node, node_info in simulator_topology["simulationTopology"].iteritems():
            for mcs, mcs_info in node_info["reliability"].iteritems():
                for linked_node, reliability in mcs_info.iteritems():
                    if (node, linked_node) not in links and (linked_node, node) not in links:
                        reliability = round(reliability, 3)
                        if reliability > minPDR:
                            reliabilities.append(reliability)
                            links.add((node, linked_node))

    return reliabilities

def plotMultipleBoxplotSeabornThroughput(metric, data, sorter, outputDir='', xsplit=None, hue=None, xlabel=None, percentile=None):
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
    # newData['tput'] = newData['tput'].apply(lambda x: x*100.0)
    # 
    # newData.exp_left = newData.exp_left.astype("category")
    # newData.exp_left.cat.set_categories(sorter, inplace=True)
    # newData = newData.sort_values(["delta"])
    # 
    # print metric
    # meanDataframe = newData.groupby(['delta'])['tput'].mean().reset_index()
    # print meanDataframe
    

    # meanDataframe.exp = meanDataframe.exp.astype("category")
    # meanDataframe.exp.cat.set_categories(sorter, inplace=True)
    # meanDataframe = meanDataframe.sort_values(["exp"])

    axBoxplot = sns.boxplot(x=xsplit, y='pdr', data=data, palette=pal, width=0.7, linewidth=1, showfliers=True, showmeans=True, hue=hue, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
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
    axBoxplot.legend(frameon=False, fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_boxplot(data, metric, x_label, y_label, output_dir):
    # plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    ax_boxplot = sns.boxplot(x='experiment_type', y=metric, data=data, palette=pal, width=0.3, showfliers=False, showmeans=True, meanprops=dict(marker='x', markersize=3, linewidth=2, markeredgecolor="#303030"))
    ax_boxplot.tick_params(labelsize=6)
    tick_labels = [getExperimentName(item.get_text()) for item in ax_boxplot.get_xticklabels()]
    ax_boxplot.set_xticklabels(tick_labels)
    ax_boxplot.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')
    # ax_boxplot.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')

    # ax_boxplot.set(ylim=(0,None))
    ax_boxplot.set_xlabel('')
    sns.despine(ax=ax_boxplot)
    name = '{0}/boxplot-{1}-{2}.png'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_bars(data, metric, x_label, y_label, hue, output_dir, name_suffix=''):
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    ax_barplot = None
    if hue == None:
        ax_barplot = sns.barplot(x='experiment_type', y=metric, data=data, palette=pal, errwidth=1.0)
    else:
        ax_barplot = sns.barplot(x='experiment_type', y=metric, hue=hue, data=data, palette=pal, errwidth=1.0)

    ax_barplot.tick_params(labelsize=4, axis='x')
    tick_labels = [getExperimentName(item.get_text()) for item in ax_barplot.get_xticklabels()]
    ax_barplot.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')

    # ax_boxplot.set(ylim=(0,None))
    ax_barplot.set_xlabel('')
    sns.despine(ax=ax_barplot)
    name = '{0}/bars-{1}-{2}-{3}.png'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    sns.despine()
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plot_bars_modulations(data, metric, x_label, y_label, hue, output_dir, name_suffix=''):
    plt.figure(figsize=(7, 5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors
    # exit()
    ax_barplot = sns.barplot(x='experiment_type', y=metric, hue=hue, data=data, palette=pal, errwidth=1.0, ci='sd')

    # ax_barplot.tick_params(labelsize=4, axis='x')
    tick_labels = [getExperimentName(item.get_text()) for item in ax_barplot.get_xticklabels()]
    ax_barplot.set_xticklabels(tick_labels)
    ax_barplot.tick_params(labelsize=14)
    ax_barplot.set(ylim=(-1,14))
    ax_barplot.set_xlabel('')
    sns.despine(ax=ax_barplot)
    name = '{0}/bars-{1}-{2}-{3}.eps'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    plt.ylabel(y_label, fontsize=22)
    plt.xlabel(x_label, fontsize=22)

    labels_dict = {'TSCH_SLOTBONDING_50_KBPS_PHY': '50 kbps PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY': '1000 kbps PHY'}

    handles, labels = ax_barplot.get_legend_handles_labels()
    labels = [labels_dict[label] for label in labels]
    l = ax_barplot.legend(handles, labels, loc='upper left', frameon=False, fontsize=16)

    sns.despine()
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plot_histogram_bars(data, x_metric, y_metric, x_label, y_label, hue, output_dir, name_suffix=''):
    # plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    ax_barplot = None
    ax_barplot = sns.barplot(x=x_metric, y=y_metric, hue=hue, data=data, palette=pal, errwidth=1.0)

    ax_barplot.tick_params(labelsize=10)
    ax_barplot.set_xticklabels(ax_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')

    # ax_boxplot.set(ylim=(0,None))
    ax_barplot.set_xlabel('')
    ax_barplot.tick_params(labelsize=6)
    sns.despine(ax=ax_barplot)
    name = '{0}/histogram-bars-{1}-{2}-{3}.pdf'.format(output_dir, x_metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_catplot(data, metric, x_label, y_label, output_dir):
    # fig = plt.figure(figsize=(10, 10))
    # fig.subplots_adjust(bottom=0)
    # fig.subplots_adjust(top=1)
    # fig.subplots_adjust(right=1)
    # fig.subplots_adjust(left=0)
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    meanDataframe = data.groupby(['experiment_type'])[metric].mean().reset_index()
    stdDataframe = data.groupby(['experiment_type'])[metric].std().reset_index()
    output_file = "{0}/{1}.txt".format(output_dir, metric)
    # with open(output_file, "a") as myfile:
    #     myfile.write("{0} - mean - std\n".format(metric))
    merged_mean_std = pd.merge(meanDataframe, stdDataframe, on='experiment_type', suffixes=('_mean', '_std'))
    merged_mean_std.to_csv("{0}/{1}.txt".format(output_dir, metric), sep='\t')
    # ax_catplot = sns.catplot(x='experiment_type', y=metric, data=data, palette=pal, jitter=False)
    # for ax in ax_catplot.axes.flat:
    #     tick_labels = [getExperimentName(item.get_text()) for item in ax.get_xticklabels()]
    #     # ax.set_xticklabels(tick_labels)
    #     ax.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right', fontsize=6)
    # hardware-heuristic-delta-diff-ss
    # # sns.despine(ax=ax_catplot)
    # name = '{0}/catplot-{1}-{2}.png'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    # plt.ylabel(y_label)
    # plt.xlabel(x_label)
    # 
    # plt.tight_layout()
    # plt.savefig(name)
    # plt.close()

    data_file_name = '{0}/data-{1}-{2}.csv'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    data[['experiment_type', 'seed', metric]].to_csv(data_file_name, sep='\t', encoding='utf-8')

def write_accuracy_heuristic(data, output_dir):
    output = ''
    # data.to_csv('before.csv', sep='\t')
    # # drop all the rows where true negative rate is None b/c otherwise you can not take the average or std
    data['true_negative_rate'].to_csv('column-before.csv', sep='\t')
    data = data.dropna(axis=0, subset=['true_negative_rate'])
    data['true_negative_rate'].to_csv('column.csv', sep='\t')
    df_mean = data.mean()
    df_std = data.std()
    output += 'heuristic mean time: {0}, {1}\n'.format(df_mean['total_time_heuristic'], df_std['total_time_heuristic'])
    output += 'feasibility mean time: {0}, {1}\n'.format(df_mean['total_time_feasibility'], df_std['total_time_feasibility'])
    output += 'total time percentage: {0}, {1}\n'.format(df_mean['total_time_percentage'],df_std['total_time_percentage'])
    output += 'mean total heuristic checks: {0}, {1}\n'.format(df_mean['total_heuristic_checks'], df_std['total_heuristic_checks'])
    output += 'true_positive_rate: {0}, {1}\n'.format(df_mean['true_positive_rate'], df_std['true_positive_rate'])
    output += 'true_negative_rate: {0}, {1}\n'.format(df_mean['true_negative_rate'], df_std['true_negative_rate'])
    output += 'false_positive_rate: {0}, {1}\n'.format(df_mean['false_positive_rate'], df_std['false_positive_rate'])
    output += 'false_negative_rate: {0}, {1}\n'.format(df_mean['false_negative_rate'], df_std['false_negative_rate'])

    # output += 'feasibility mean time: {0}'.format()
    # output += 'heuristic nr: {0}'.format()
    # output += 'feasibility nr: {0}'.format()
    # output += 'heuristic mean time: {0}'.format()
    # output += 'heuristic mean time: {0}'.format()

    print(output)

def plot_lineplot(data, metric, x_label, y_label, hue, output_dir, name_suffix):
    # plt.figure(figsize=(6, 2.5))
    sns.set_style('white')
    pal = sns.color_palette('colorblind')
    global mapping
    global colors

    ax_lineplot = None
    if hue:
        ax_lineplot = sns.lineplot(x='generation', y=metric, hue=hue, data=data, color='red', legend=False)
    else:
        ax_lineplot = sns.lineplot(x='generation', y=metric, data=data, color='red', legend=False)
    # ax_catplot.set_xticklabels(rotation=45, horizontalalignment='right')
    name = '{0}/lineplot-{1}-{2}-{3}.png'.format(output_dir, metric, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    plt.title(name_suffix)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_histogram(data, x_label, output_dir, name_suffix):
    global mapping
    global colors

    # print data
    fig, ax = plt.subplots()

    # assert False
    b = sns.countplot(data)
    b.set_xticklabels(b.get_xticklabels(), rotation=45)

    name = '{0}/histogram-{1}-{2}-{3}.pdf'.format(output_dir, x_label, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    b.tick_params(labelsize=4)
    plt.xlabel(x_label)
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def plot_double_histogram(data_first, x_label_first, data_second, x_label_second, output_dir, name_suffix):
    global mapping
    global colors

    # fig, ax = plt.subplots(1, 2)
    # countplot1 = sns.countplot(data_first, ax=ax[0])
    # countplot2 = sns.countplot(data_second, ax=ax[1])
    # 
    # countplot1.set_xticklabels(countplot1.get_xticklabels(), rotation=45)
    # countplot2.set_xticklabels(countplot2.get_xticklabels(), rotation=45)
    # 
    # name = '{0}/double-histogram-{1}-{2}-{3}.pdf'.format(output_dir, x_label_first, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    # countplot1.tick_params(labelsize=3)
    # countplot2.tick_params(labelsize=3)
    # 
    # # plt.xlabel(x_label)

    plt.style.use('seaborn-deep')

    x = data_first
    y = data_second
    # bins = np.linspace(-10, 10, 30)

    plt.hist([x, y], label=['x', 'y'])
    plt.legend(loc='upper right')
    plt.tight_layout()
    name = '{0}/double-histogram-{1}-{2}-{3}.pdf'.format(output_dir, x_label_first, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name_suffix)
    plt.savefig(name)
    plt.close()


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

if __name__ == '__main__':
    result_type = str(sys.argv[1])
    result_types = ['ga-slotbonding', 'ga-evolution', 'exhaustive-compare', 'heuristic-compare', 'modulation-analysis', 'hardware-heuristic-delta', 'hardware-heuristic-delta-diff-ss', 'hardware-heuristic-modulation-diff-ss', 'hardware-ga', 'hardware-phys']
    if result_type not in result_types:
        assert False

    plots_dir = 'plots'
    try:
        os.makedirs(plots_dir)
    except OSError:
        if not os.path.isdir(plots_dir):
            raise

    output_dir = '{0}/plots-{1}'.format(plots_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    try:
        os.makedirs(output_dir)
    except OSError:
        if not os.path.isdir(output_dir):
            raise

    # dataDir = str(sys.argv[2])
    dataDirs = str(sys.argv[2]).split(',')
    experimentTypes = str(sys.argv[3]).split(',')

    # aggregated over all motes
    ga_throughput = pd.DataFrame()

    sorter = []
    sorterMode = []

    # make the initial data columns
    columns_dataframe = ['experiment_type', 'path', 'seed', 'ind', 'tput', 'airtime', 'normalized_airtime', 'avg_hopcount', 'avg_minimal_hopcount', 'gen_best_ind', 'total_ind', 'filtered_ind', 'unique_ind', 'valid_ind', 'best_ind', 'ilp_invalids', 'fast_invalids', 'diff_percentage_invalids', 'total_feasibility_checks', 'total_heuristic_checks', 'diff_percentage_heuristic', 'diff_percentage_feasibility', 'ix_hof', 'total_time_feasibility', 'total_time_heuristic', 'true_positives', 'true_positive_rate', 'true_negatives', 'true_negative_rate', 'false_negatives', 'false_negative_rate', 'false_positives', 'false_positive_rate', 'total_time_percentage']
    # add the modulations
    MCSs = modulationInstance.modulations
    columns_dataframe += MCSs
    ga_data = pd.DataFrame(columns = columns_dataframe, dtype=float)
    # reliabilities = pd.DataFrame(columns = ['experiment_type', 'dataDir', 'datafile', 'mote', 'reliability'])
    reliabilities = pd.DataFrame()
    nr_slots_df = pd.DataFrame()
    nr_reach_root_df = pd.DataFrame()
    columns_evolution_dataframe = ['experiment_type', 'dataDir', 'datafile', 'tput', 'tput_rel', 'airtime', 'airtime_rel']
    evolution_data = pd.DataFrame(columns_evolution_dataframe)
    list_evolution_data = []
    list_unique_inds_data = []
    list_valid_inds_data = []
    list_infeasible_inds_data = []
    exhaustive_experiment_type = None
    ga_experiment_types = []
    all_reliabilities = {}
    individual_reliabilities = {}
    deep_analysis = {}
    for experimentType in experimentTypes:
        print 'Parsing for experiment {0}...'.format(experimentType)
        if result_type == 'ga-slotbonding' or result_type == 'ga-exhaustive' or result_type == 'exhaustive-compare' or result_type == 'heuristic-compare' or result_type == 'hardware-ga':
            rgx = '%s_([A-Za-z0-9_]+)$' % 'exp'
            exp_value = list(get_set_rgx([experimentType], rgx))
            if len(exp_value) == 0:
                exp_value = None
                print("Variable \'exp\' is None.")
            else:
                exp_value = list(get_set_rgx([experimentType], rgx))[0]
        elif result_type == 'hardware-heuristic-delta' or result_type == 'hardware-heuristic-delta-diff-ss' or result_type == 'hardware-heuristic-modulation-diff-ss':
            rgx = '%s_([A-Za-z0-9_]+)$' % 'delta'
            delta_value = list(get_set_rgx([experimentType], rgx))
            if len(delta_value) == 0:
                delta_value = None
                print("Variable \'delta\' is None.")
            else:
                delta_value = list(get_set_rgx([experimentType], rgx))[0]
        else:
            pass

        sorter.append(getExperimentName(experimentType))

        for dataDir in dataDirs:
            # print 'For experiment {0}, parsing results in {1}.'.format(experimentType, dataDir)
            ga_data = ga_parse_solutions(dataDir, experimentType, ga_data)

        if result_type == 'hardware-heuristic-delta':
            for dataDir in dataDirs:
                reliabilities = ga_parse_reliabilities(dataDir, experimentType, reliabilities)
                nr_slots_df = ga_parse_nr_slots(dataDir, experimentType, nr_slots_df)
                nr_reach_root_df = ga_parse_nr_reach_root(dataDir, experimentType, nr_reach_root_df)
                # print(len(reliabilities))

        if result_type == 'exhaustive-compare':
            if 'exhaustive' in experimentType:
                exhaustive_experiment_type = experimentType
            else:
                ga_experiment_types.append(experimentType)

        if result_type == 'heuristic-compare':
            if 'ilp' in experimentType:
                exhaustive_experiment_type = experimentType
            else:
                ga_experiment_types.append(experimentType)

        if result_type == 'ga-evolution':
            for dataDir in dataDirs:
                # print 'For experiment {0}, parsing evolutions in {1}.'.format(experimentType, dataDir)
                list_evolution_data += ga_parse_evolution(dataDir=dataDir, experiment_type=experimentType)
                list_unique_inds_data += ga_parse_unique_inds(dataDir=dataDir, experiment_type=experimentType)
                list_valid_inds_data += ga_parse_valid_inds(dataDir=dataDir, experiment_type=experimentType)
                list_infeasible_inds_data += ga_parse_infeasible_inds(dataDir=dataDir, experiment_type=experimentType)

        minPDR = 0.7
        if result_type == 'modulation-analysis':
            deep_analysis[experimentType] = {}
            all_reliabilities[experimentType] = []
            individual_reliabilities[experimentType] = []
            for dataDir in dataDirs:
                deep_analysis_results = copy.deepcopy(ga_parse_deep_analysis(minPDR=minPDR, dataDir=dataDir, experiment_type=experimentType))
                if len(deep_analysis_results) > 0 and experimentType is deep_analysis and len(deep_analysis[experimentType]) > 0:
                    raise BaseException("Found two different sets of results for this experiment type.")
                else:
                    if len(deep_analysis_results) > 0:
                        deep_analysis[experimentType] = deep_analysis_results
                all_reliabilities[experimentType] += copy.deepcopy(ga_parse_all_reliabilities(minPDR=minPDR, dataDir=dataDir, experiment_type=experimentType))
                individual_reliabilities[experimentType] += copy.deepcopy(ga_parse_individual_reliabilities(minPDR=minPDR, dataDir=dataDir, experiment_type=experimentType))

            all_reliabilities[experimentType] = [x for x in all_reliabilities[experimentType] if minPDR <= x <= 1.0]
            individual_reliabilities[experimentType] = [x for x in individual_reliabilities[experimentType] if minPDR <= x <= 1.0]

    evolution_data = pd.DataFrame(list_evolution_data)
    unique_inds_data = pd.DataFrame(list_unique_inds_data)
    valid_inds_data = pd.DataFrame(list_valid_inds_data)
    infeasible_inds_data = pd.DataFrame(list_infeasible_inds_data)

    if result_type == 'hardware-phys':
        PHYS = ['TSCH_SLOTBONDING_50_KBPS_PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY']
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + PHYS
        df_modulations = ga_data[columns_to_keep]
        sum_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        melt_df_modulations = pd.melt(sum_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="sum_modulation")
        # plot the different modulations per experiment
        plot_bars(melt_df_modulations, 'sum_modulation', x_label='Experiment type', y_label='modulations', hue="modulation", output_dir=output_dir)
    elif result_type == "hardware-heuristic-modulation-diff-ss":
        PHYS = ['TSCH_SLOTBONDING_50_KBPS_PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY']
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + PHYS
        df_modulations = ga_data[columns_to_keep]
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print(df_modulations.to_string())
        df_modulations.to_csv("{0}/phy-usage.txt".format(output_dir), sep='\t')
        # exit()
        mean_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        std_df_modulations = df_modulations.groupby('experiment_type').std().reset_index()
        melt_mean_df_modulations = pd.melt(mean_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="mean_modulation")
        melt_std_df_modulations = pd.melt(std_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="std_modulation")
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print(melt_mean_df_modulations.to_string())
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print(melt_std_df_modulations.to_string())
        final_mean_std_merged_df = pd.merge(melt_mean_df_modulations, melt_std_df_modulations, on=['experiment_type', 'modulation'], suffixes=('_mean', '_std'))
        final_mean_std_merged_df.to_csv("{0}/modulations.txt".format(output_dir), sep='\t')
        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print(final_mean_std_merged_df.to_string())
        melt_pivot_df_modulations = pd.melt(df_modulations, id_vars="experiment_type", var_name="modulation",
                                            value_name="value")
        plot_bars_modulations(melt_pivot_df_modulations, 'value', x_label='Experiment type', y_label='Links', hue="modulation", output_dir=output_dir)
    elif result_type == "hardware-heuristic-delta-diff-ss":

        ga_data['pdr'] = ga_data.apply(lambda row: (float(row.tput) / 11.0), axis=1)
        # plot_boxplot(ga_data, 'tput', x_label='Delta', y_label='Throughput', output_dir=output_dir)
        def return_name(topConfig, delta):
            if topConfig == 'heuristic' and delta == '000':
                return "Heuristic, $\delta$ = 0.0"
            elif topConfig == 'heuristic' and delta == '010':
                return "Heuristic, $\delta$ = 0.1"
            elif topConfig == 'heuristic' and delta == '020':
                return "Heuristic, $\delta$ = 0.2"
            elif topConfig == 'heuristic' and delta == '030':
                return "Heuristic, $\delta$ = 0.3"
            elif topConfig == 'heuristic' and delta == '040':
                return "Heuristic, $\delta$ = 0.4"
            elif topConfig == 'heuristic' and delta == '050':
                return "Heuristic, $\delta$ = 0.5"
            elif topConfig == 'heuristic' and delta == '060':
                return "Heuristic, $\delta$ = 0.6"
            elif topConfig == 'heuristic' and delta == '080':
                return "Heuristic, $\delta$ = 0.8"
            elif topConfig == 'heuristic' and delta == '100':
                return "Heuristic, $\delta$ = 1.0"
            elif topConfig == 'gaheuristic' and delta == '000':
                return "GA scheduler"
            elif topConfig == 'gaheuristiconly50' and delta == '000':
                return "GA, 50 kbps PHY"

        ga_data['new_name'] = ga_data.apply(lambda row: return_name(row.topConfig, row.delta), axis=1)
        plotMultipleBoxplotSeabornThroughput('PDR', ga_data, sorter, outputDir=output_dir, xsplit='ss', hue='new_name', xlabel='Slot frame length (ms)')

        # hopcount
        mean_pdr_pd = ga_data.groupby(['experiment_type'])['pdr'].mean().reset_index()
        std_pdr_pd = ga_data.groupby(['experiment_type'])['pdr'].std().reset_index()
        final_pdr_merged_df = pd.merge(mean_pdr_pd, std_pdr_pd, on='experiment_type', suffixes=('_mean', '_std'))
        final_pdr_merged_df.to_csv("{0}/pdr.txt".format(output_dir), sep='\t')

        columns_to_keep = ['experiment_type', 'seed', 'pdr']
        df_pdrs = ga_data[columns_to_keep]
        df_pdrs.to_csv("{0}/pdrs.txt".format(output_dir), sep='\t')

        PHYS = ['TSCH_SLOTBONDING_50_KBPS_PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY']
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + PHYS
        df_modulations = ga_data[columns_to_keep]
        sum_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        melt_df_modulations = pd.melt(sum_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="sum_modulation")
        # plot the different modulations per experiment
        plot_bars(melt_df_modulations, 'sum_modulation', x_label='Experiment type', y_label='modulations', hue="modulation", output_dir=output_dir)
    elif result_type == "hardware-heuristic-delta":

        columns_to_keep = ['experiment_type', 'reliability']
        df_reliabilities = reliabilities[columns_to_keep]
        mean_df_reliabilities = df_reliabilities.groupby('experiment_type').mean().reset_index()
        path_to_output_file = "{0}/reliabilities.txt".format(output_dir)
        mean_df_reliabilities.to_csv(path_to_output_file, sep=' ', mode='a')
        columns_to_keep = ['experiment_type', 'nr_slots']
        df_nr_slots_only = nr_slots_df[columns_to_keep]
        mean_df_nr_slots = df_nr_slots_only.groupby('experiment_type').mean().reset_index()
        path_to_output_file = "{0}/nr_slots.txt".format(output_dir)
        mean_df_nr_slots.to_csv(path_to_output_file, sep=' ', mode='a')
        columns_to_keep = ['experiment_type', 'nr_reach_root']
        df_nr_reach_root_only = nr_reach_root_df[columns_to_keep]
        mean_df_nr_reach_root = df_nr_reach_root_only.groupby('experiment_type').mean().reset_index()
        path_to_output_file = "{0}/nr_reach_root.txt".format(output_dir)
        mean_df_nr_reach_root.to_csv(path_to_output_file, sep=' ', mode='a')
        # output_file = open(path_to_output_file, "w")
        # output_file.writelines(mean_df_reliabilities)
        # output_file.close()
        plot_boxplot(ga_data, 'tput', x_label='Delta', y_label='Throughput', output_dir=output_dir)
        # plot_catplot(ga_data, 'tput', x_label='Delta', y_label='Throughput', output_dir=output_dir)
        plot_boxplot(ga_data, 'airtime', x_label='Delta', y_label='Radio on time', output_dir=output_dir)
        plot_catplot(ga_data, 'airtime', x_label='Delta', y_label='Radio on time', output_dir=output_dir)
        plot_boxplot(ga_data, 'avg_hopcount', x_label='Delta', y_label='Average hop count',
                     output_dir=output_dir)
        PHYS = ['TSCH_SLOTBONDING_50_KBPS_PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY']
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + PHYS
        df_modulations = ga_data[columns_to_keep]
        sum_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        melt_df_modulations = pd.melt(sum_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="sum_modulation")
        # plot the different modulations per experiment
        plot_bars(melt_df_modulations, 'sum_modulation', x_label='Experiment type', y_label='modulations', hue="modulation", output_dir=output_dir)
    elif result_type == "hardware-ga":
        plot_boxplot(ga_data, 'tput', x_label='Slotframe', y_label='Throughput', output_dir=output_dir)
        # plot_catplot(ga_data, 'tput', x_label='Slotframe', y_label='Throughput', output_dir=output_dir)
        plot_boxplot(ga_data, 'airtime', x_label='Slotframe', y_label='Radio on time', output_dir=output_dir)
        # plot_catplot(ga_data, 'airtime', x_label='Slotframe', y_label='Radio on time', output_dir=output_dir)
        plot_boxplot(ga_data, 'avg_hopcount', x_label='Slotframe', y_label='Average hop count', output_dir=output_dir)
        PHYS = ['TSCH_SLOTBONDING_50_KBPS_PHY', 'TSCH_SLOTBONDING_1000_KBPS_PHY']
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + PHYS
        df_modulations = ga_data[columns_to_keep]
        sum_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        melt_df_modulations = pd.melt(sum_df_modulations, id_vars="experiment_type", var_name="modulation", value_name="sum_modulation")
        # plot the different modulations per experiment
        plot_bars(melt_df_modulations, 'sum_modulation', x_label='Experiment type', y_label='modulations', hue="modulation", output_dir=output_dir)
    elif result_type == "ga-slotbonding":
        # plot the throughput
        plot_boxplot(ga_data, 'tput', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        plot_catplot(ga_data, 'tput', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        # plot the airtime
        plot_boxplot(ga_data, 'airtime', x_label='Experiment type', y_label='Radio on time', output_dir=output_dir)
        plot_catplot(ga_data, 'airtime', x_label='Experiment type', y_label='Radio on time', output_dir=output_dir)
        # plot the airtime
        # plot_boxplot(ga_data, 'normalized_airtime', x_label='Experiment type', y_label='Normalized radio on time',
        #              output_dir=output_dir)
        plot_catplot(ga_data, 'normalized_airtime', x_label='Experiment type', y_label='Normalized radio on time',
                     output_dir=output_dir)
        # plot the avg_hopcount
        plot_boxplot(ga_data, 'avg_hopcount', x_label='Experiment type', y_label='Average hop count',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'avg_minimal_hopcount', x_label='Experiment type',
                     y_label='Average minimal hop count', output_dir=output_dir)

        plot_catplot(ga_data, 'avg_hopcount', x_label='Experiment type', y_label='Average hop count',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'avg_minimal_hopcount', x_label='Experiment type',
                     y_label='Average minimal hop count', output_dir=output_dir)

        plot_boxplot(ga_data, 'fast_invalids', x_label='Experiment type', y_label='Infeasible by fast feasibility check',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'ilp_invalids', x_label='Experiment type', y_label='Infeasible by ILP',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'total_feasibility_checks', x_label='Experiment type', y_label='Total checked for ILP infeasibility',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'total_heuristic_checks', x_label='Experiment type', y_label='Total checked for heuristic infeasibility',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'diff_percentage_invalids', x_label='Experiment type', y_label='Found infeasible by fast feasibility check (%)',
                     output_dir=output_dir)
        # plot_boxplot(ga_data, 'diff_percentage_feasible', x_label='Experiment type', y_label='Found feasible by ILP check (%)',
        #              output_dir=output_dir)
        plot_boxplot(ga_data, 'diff_percentage_heuristic', x_label='Experiment type', y_label='Found feasible by heuristic check (%)',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'gen_best_ind', x_label='Experiment type', y_label='Generation with best ind.',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'ix_hof', x_label='Experiment type', y_label='Index in hall-of-fame',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'false_negatives', x_label='Experiment type', y_label='Heuristic false negatives',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'false_negative_rate', x_label='Experiment type', y_label='Heuristic false negatives (% of total)',
                     output_dir=output_dir)
        # prepare the modulations data frame
        columns_to_keep = ['experiment_type'] + MCSs
        df_modulations = ga_data[columns_to_keep]
        sum_df_modulations = df_modulations.groupby('experiment_type').mean().reset_index()
        melt_df_modulations = pd.melt(sum_df_modulations, id_vars="experiment_type", var_name="modulation",
                                      value_name="sum_modulation")
        # plot the different modulations per experiment
        plot_bars(melt_df_modulations, 'sum_modulation', x_label='Experiment type', y_label='modulations',
                  hue="modulation", output_dir=output_dir)

        plot_catplot(ga_data, 'gen_best_ind', x_label='Experiment type', y_label='Generation with best ind.',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'ix_hof', x_label='Experiment type', y_label='Index in hall-of-fame',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'total_time', x_label='Experiment type', y_label='Total experiment time',
                     output_dir=output_dir)
        plot_boxplot(ga_data, 'total_time', x_label='Experiment type', y_label='Total experiment time',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'total_time_feasibility', x_label='Experiment type', y_label='Total ILP feasibility time',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'total_time_heuristic', x_label='Experiment type', y_label='Total heuristic feasibility time',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'total_ind', x_label='Experiment type', y_label='Possible individuals',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'filtered_ind', x_label='Experiment type', y_label='Filtered individuals',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'unique_ind', x_label='Experiment type', y_label='Unique individuals',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'valid_ind', x_label='Experiment type', y_label='Valid individuals',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'false_negatives', x_label='Experiment type', y_label='Heuristic false negatives',
                     output_dir=output_dir)
        plot_catplot(ga_data, 'false_negative_rate', x_label='Experiment type', y_label='Heuristic false negatives (% of total)',
                     output_dir=output_dir)
        write_accuracy_heuristic(ga_data, output_dir=output_dir)
    elif result_type == "ga-evolution":

        for exp_type in experimentTypes:
            exp_unique_inds_data = unique_inds_data.loc[unique_inds_data['experiment_type'] == exp_type]
            plot_lineplot(data=exp_unique_inds_data, metric='unique_inds', x_label='Generation', y_label='Unique individuals', hue='datafile', output_dir=output_dir, name_suffix=exp_type)
            exp_valid_inds_data = valid_inds_data.loc[valid_inds_data['experiment_type'] == exp_type]
            plot_lineplot(data=exp_valid_inds_data, metric='valid_inds', x_label='Generation', y_label='Valid individuals', hue='datafile', output_dir=output_dir, name_suffix=exp_type)
            exp_infeasible_inds_data = infeasible_inds_data.loc[infeasible_inds_data['experiment_type'] == exp_type]
            plot_lineplot(data=exp_infeasible_inds_data, metric='infeasible_inds', x_label='Generation', y_label='Infeasible individuals in one population', hue='datafile', output_dir=output_dir, name_suffix=exp_type)


        for exp_type in experimentTypes:
            exp_evolution_data = evolution_data.loc[evolution_data['experiment_type'] == exp_type]
            plot_lineplot(data=exp_evolution_data, metric='tput', x_label='Generation', y_label='Throughput', hue='datafile', output_dir=output_dir, name_suffix=exp_type)
            plot_lineplot(data=exp_evolution_data, metric='airtime', x_label='Generation', y_label='Airtime', hue='datafile', output_dir=output_dir, name_suffix=exp_type)

        for exp_type in experimentTypes:
            # tput_rel and airtime_rel per iteration
            exp_evolution_data = evolution_data.loc[evolution_data['experiment_type'] == exp_type]
            plot_lineplot(data=exp_evolution_data, metric='tput_rel', x_label='Generation', y_label='Throughput improvement (%)', hue='datafile', output_dir=output_dir, name_suffix=exp_type)
            plot_lineplot(data=exp_evolution_data, metric='airtime_rel', x_label='Generation', y_label='Airtime improvement (%)', hue='datafile', output_dir=output_dir, name_suffix=exp_type)

            # mean of tput_rel
            exp_evolution_data = evolution_data.loc[evolution_data['experiment_type'] == exp_type]
            mean_evolution_data_tput_rel = exp_evolution_data.groupby(['experiment_type', 'generation'])['tput_rel'].mean().reset_index()
            plot_lineplot(data=mean_evolution_data_tput_rel, metric='tput_rel', x_label='Generation', y_label='Throughput improvement (%)', hue=None, output_dir=output_dir, name_suffix="mean-{0}".format(exp_type))
            # mean of airtime_rel
            mean_evolution_data_airtime_rel = exp_evolution_data.groupby(['experiment_type', 'generation'])['airtime_rel'].mean().reset_index()
            plot_lineplot(data=mean_evolution_data_airtime_rel, metric='airtime_rel', x_label='Generation', y_label='Airtime improvement (%)', hue=None, output_dir=output_dir, name_suffix="mean-{0}".format(exp_type))

    elif result_type == "exhaustive-compare" or result_type == "heuristic-compare":
        columns_to_keep = ['experiment_type', 'seed', 'tput', 'airtime', 'best_ind']
        filtered_df = ga_data[columns_to_keep]
        exhaustive_data = filtered_df.loc[filtered_df['experiment_type'] == exhaustive_experiment_type]
        remainder_ga_data = filtered_df.loc[filtered_df['experiment_type'] != exhaustive_experiment_type]
        merge_df = pd.merge(exhaustive_data, remainder_ga_data, on='seed', suffixes=('_exhaustive', ''))
        merge_df['tput_diff'] = merge_df.apply(lambda row: (row.tput_exhaustive - row.tput), axis=1)
        merge_df['tput_optimality'] = merge_df.apply(lambda row: (row.tput / float(row.tput_exhaustive)), axis=1)
        merge_df['airtime_diff'] = merge_df.apply(lambda row: abs(row.airtime - row.airtime_exhaustive), axis=1)
        # merge_df['airtime_optimality'] = merge_df.apply(lambda row: (abs(row.airtime) - row.airtime_exhaustive)), axis=1)
        merge_df['diff_ind'] = merge_df.apply(lambda row: 1 if (row.best_ind != row.best_ind_exhaustive) else 0, axis=1)
        merge_df['same_ind'] = merge_df.apply(lambda row: 1 if (row.best_ind == row.best_ind_exhaustive) else 0, axis=1)

        with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
                               None):  # more options can be specified also
            print(merge_df.to_string())

        plot_boxplot(merge_df, 'tput_diff', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        plot_boxplot(merge_df, 'tput_optimality', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        plot_boxplot(merge_df, 'airtime_diff', x_label='Experiment type', y_label='Air time', output_dir=output_dir)
        plot_catplot(merge_df, 'tput_diff', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        plot_catplot(merge_df, 'tput_optimality', x_label='Experiment type', y_label='Throughput', output_dir=output_dir)
        plot_catplot(merge_df, 'airtime_diff', x_label='Experiment type', y_label='Air time', output_dir=output_dir)
        diff_ind_df = merge_df.groupby('experiment_type').sum().reset_index()
        plot_bars(diff_ind_df, 'diff_ind', x_label='Experiment type', y_label='Different ind. from optimal one', hue=None, output_dir=output_dir)
        same_ind_df = merge_df.groupby('experiment_type').sum().reset_index()
        # with pd.option_context('display.max_colwidth', 500, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print(same_ind_df.to_string())
        # exit()
        plot_bars(same_ind_df, 'same_ind', x_label='Experiment type', y_label='Same ind. as optimal one', hue=None, output_dir=output_dir)

        average_tput_diff = merge_df.groupby(['experiment_type'])['tput_diff'].mean().reset_index()
        write_output = 'Average throughput difference:\n{0}\n'.format(average_tput_diff)
        std_tput_diff = merge_df.groupby(['experiment_type'])['tput_diff'].std().reset_index()
        write_output += 'Std throughput difference:\n{0}\n'.format(std_tput_diff)
        average_tput_optimality_diff = merge_df.groupby(['experiment_type'])['tput_optimality'].mean().reset_index()
        write_output += 'Average throughput optimality difference:\n{0}\n'.format(average_tput_optimality_diff)
        std_tput_optimality_diff = merge_df.groupby(['experiment_type'])['tput_optimality'].std().reset_index()
        write_output += 'Std throughput optimality difference:\n{0}\n'.format(std_tput_optimality_diff)
        average_airtime_diff = merge_df.groupby(['experiment_type'])['airtime_diff'].mean().reset_index()
        write_output += 'Average airtime difference:\n{0}\n'.format(average_airtime_diff)
        std_airtime_diff = merge_df.groupby(['experiment_type'])['airtime_diff'].std().reset_index()
        write_output += 'Std airtime difference:\n{0}\n'.format(std_airtime_diff)
        same_ind_df_percentage = same_ind_df.apply(lambda row: float(row.same_ind) / float(row.diff_ind + row.same_ind), axis=1)
        write_output += 'Same individuals:\n{0}\n'.format(same_ind_df_percentage)

        path_to_output_file = "{0}/exhaustive_compare_data.txt".format(output_dir)
        output_file = open(path_to_output_file, "w")
        output_file.writelines(write_output)
        output_file.close()

    elif result_type == "modulation-analysis":
        # for exp_type in experimentTypes:
        #
        #     plot_histogram(data=all_reliabilities[exp_type], x_label='all-reliabilities', output_dir=output_dir, name_suffix=exp_type)
        #     plot_histogram(data=individual_reliabilities[exp_type], x_label='ind-reliabilities', output_dir=output_dir, name_suffix=exp_type)
        #
        #     # count all the reliabilities
        #     unique_all = {rel : all_reliabilities[exp_type].count(rel) for rel in set(all_reliabilities[exp_type])}
        #     unique_individual = {rel : individual_reliabilities[exp_type].count(rel) for rel in set(individual_reliabilities[exp_type])}
        #     column_names = ["experimentType", "Type", "reliability", "count"]
        #     df_count = pd.DataFrame(columns=column_names)
        #     for rel, count in unique_all.items():
        #         df_count = df_count.append({"experimentType": exp_type, "Type": "available", "reliability": rel, "count": count}, ignore_index=True)
        #     for rel, count in unique_individual.items():
        #         df_count = df_count.append({"experimentType": exp_type, "Type": "used", "reliability": rel, "count": count}, ignore_index=True)
        #     # plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="type", output_dir=output_dir, name_suffix=exp_type)
        #     plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="Type", output_dir=output_dir, name_suffix="all-{0}".format(exp_type))
        #     df_count = df_count.loc[df_count["reliability"] != 1.0]
        #     plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="Type", output_dir=output_dir, name_suffix=exp_type)

        write_output = ''
        percentages_output = ''
        # picked_best_reliability_overall
        column_names = ["experiment_type", "type", "count"]
        df_picked_best_reliability_overall = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_picked_best_reliability_overall = df_picked_best_reliability_overall.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["picked_best_reliability_overall"]}, ignore_index=True)
            df_picked_best_reliability_overall = df_picked_best_reliability_overall.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["picked_best_reliability_overall"] + deep_analysis[exp_type]["picked_not_best_reliability_overall"]}, ignore_index=True)
            percentages_output += '{1} -> diff_with_best_reliability_overall: {0}%\n'.format(round(deep_analysis[exp_type]["picked_best_reliability_overall"]/float(deep_analysis[exp_type]["picked_best_reliability_overall"] + deep_analysis[exp_type]["picked_not_best_reliability_overall"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["picked_best_reliability_overall"]/float(deep_analysis[exp_type]["picked_best_reliability_overall"] + deep_analysis[exp_type]["picked_not_best_reliability_overall"]),3)*100.0)
            write_output += '{2} -> diff_with_best_reliability_overall: average = {0}, values = {1}\n'.format(np.mean(deep_analysis[exp_type]["diff_with_best_reliability_overall"]), deep_analysis[exp_type]["diff_with_best_reliability_overall"], exp_type)
            write_output += '{2} -> diff_rates_with_higher_reliabilities_overall: average = {0}, values = {1}\n'.format(np.mean(deep_analysis[exp_type]["diff_rates_with_higher_reliabilities_overall"]), deep_analysis[exp_type]["diff_rates_with_higher_reliabilities_overall"], exp_type)
        plot_histogram_bars(df_picked_best_reliability_overall, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="picked_best_reliability_overall")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # picked_best_reliability_of_parent
        column_names = ["experiment_type", "type", "count"]
        df_picked_best_reliability_of_parent = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_picked_best_reliability_of_parent = df_picked_best_reliability_of_parent.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["picked_best_reliability_of_parent"]}, ignore_index=True)
            df_picked_best_reliability_of_parent = df_picked_best_reliability_of_parent.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["picked_best_reliability_of_parent"] + deep_analysis[exp_type]["picked_not_best_reliability_of_parent"]}, ignore_index=True)
            percentages_output += '{1} -> diff_with_best_reliability_of_parent: {0}%\n'.format(round(deep_analysis[exp_type]["picked_best_reliability_of_parent"]/float(deep_analysis[exp_type]["picked_best_reliability_of_parent"] + deep_analysis[exp_type]["picked_not_best_reliability_of_parent"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["picked_best_reliability_of_parent"]/float(deep_analysis[exp_type]["picked_best_reliability_of_parent"] + deep_analysis[exp_type]["picked_not_best_reliability_of_parent"]),3)*100.0)
            write_output += '{2} -> diff_with_best_reliability_of_parent: average = {0}, values = {1}\n'.format(np.mean(deep_analysis[exp_type]["diff_with_best_reliability_of_parent"]), deep_analysis[exp_type]["diff_with_best_reliability_of_parent"], exp_type)
            write_output += '{2} -> diff_rates_with_higher_reliabilities_parent: average = {0}, values = {1}\n'.format(np.mean(deep_analysis[exp_type]["diff_rates_with_higher_reliabilities_parent"]), deep_analysis[exp_type]["diff_rates_with_higher_reliabilities_parent"], exp_type)
        plot_histogram_bars(df_picked_best_reliability_of_parent, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="picked_best_reliability_of_parent")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # picked_best_rate_overall
        column_names = ["experiment_type", "type", "count"]
        df_picked_best_rate_overall = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_picked_best_rate_overall = df_picked_best_rate_overall.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["picked_best_rate_overall"]}, ignore_index=True)
            df_picked_best_rate_overall = df_picked_best_rate_overall.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["picked_best_rate_overall"] + deep_analysis[exp_type]["picked_not_best_rate_overall"]}, ignore_index=True)
            percentages_output += '{1} -> df_picked_best_rate_overall: {0}%\n'.format(round(deep_analysis[exp_type]["picked_best_rate_overall"]/float(deep_analysis[exp_type]["picked_best_rate_overall"] + deep_analysis[exp_type]["picked_not_best_rate_overall"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["picked_best_rate_overall"]/float(deep_analysis[exp_type]["picked_best_rate_overall"] + deep_analysis[exp_type]["picked_not_best_rate_overall"]),3)*100.0)
        plot_histogram_bars(df_picked_best_rate_overall, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="picked_best_rate_overall")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # picked_best_rate_of_parent
        column_names = ["experiment_type", "type", "count"]
        df_picked_best_rate_of_parent = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_picked_best_rate_of_parent = df_picked_best_rate_of_parent.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["picked_best_rate_of_parent"]}, ignore_index=True)
            df_picked_best_rate_of_parent = df_picked_best_rate_of_parent.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["picked_best_rate_of_parent"] + deep_analysis[exp_type]["picked_not_best_rate_of_parent"]}, ignore_index=True)
            percentages_output += '{1} -> df_picked_best_rate_of_parent: {0}%\n'.format(round(deep_analysis[exp_type]["picked_best_rate_of_parent"]/float(deep_analysis[exp_type]["picked_best_rate_of_parent"] + deep_analysis[exp_type]["picked_not_best_rate_of_parent"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["picked_best_rate_of_parent"]/float(deep_analysis[exp_type]["picked_best_rate_of_parent"] + deep_analysis[exp_type]["picked_not_best_rate_of_parent"]),3)*100.0)
        plot_histogram_bars(df_picked_best_rate_of_parent, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="picked_best_rate_of_parent")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # best_reliability_picked_best_rate_overall
        # column_names = ["experiment_type", "type", "count"]
        # df_best_reliability_picked_best_rate_overall = pd.DataFrame(columns=column_names)
        # for exp_type in experimentTypes:
        #     df_best_reliability_picked_best_rate_overall = df_best_reliability_picked_best_rate_overall.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["best_reliability_picked_best_rate_overall"]}, ignore_index=True)
        #     df_best_reliability_picked_best_rate_overall = df_best_reliability_picked_best_rate_overall.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["best_reliability_picked_best_rate_overall"] + deep_analysis[exp_type]["best_reliability_picked_not_best_rate_overall"]}, ignore_index=True)
        #     percentages_output += '{1} -> df_best_reliability_picked_best_rate_overall: {0}%\n'.format(round(deep_analysis[exp_type]["best_reliability_picked_best_rate_overall"]/float(deep_analysis[exp_type]["best_reliability_picked_best_rate_overall"] + deep_analysis[exp_type]["best_reliability_picked_not_best_rate_overall"]),3)*100.0, exp_type)
        # plot_histogram_bars(df_best_reliability_picked_best_rate_overall, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="best_reliability_picked_best_rate_overall")

        # best_reliability_picked_best_rate_of_parent
        # column_names = ["experiment_type", "type", "count"]
        # df_best_reliability_picked_best_rate_of_parent = pd.DataFrame(columns=column_names)
        # for exp_type in experimentTypes:
        #     df_best_reliability_picked_best_rate_of_parent = df_best_reliability_picked_best_rate_of_parent.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["best_reliability_picked_best_rate_of_parent"]}, ignore_index=True)
        #     df_best_reliability_picked_best_rate_of_parent = df_best_reliability_picked_best_rate_of_parent.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["best_reliability_picked_best_rate_of_parent"] + deep_analysis[exp_type]["best_reliability_picked_not_best_rate_of_parent"]}, ignore_index=True)
        #     percentages_output += '{1} -> best_reliability_picked_best_rate_of_parent: {0}%\n'.format(round(deep_analysis[exp_type]["best_reliability_picked_best_rate_of_parent"]/float(deep_analysis[exp_type]["best_reliability_picked_best_rate_of_parent"] + deep_analysis[exp_type]["best_reliability_picked_not_best_rate_of_parent"]),3)*100.0, exp_type)
        # plot_histogram_bars(df_best_reliability_picked_best_rate_of_parent, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="best_reliability_picked_best_rate_of_parent")

        # best_rate_picked_best_reliability_overall
        # column_names = ["experiment_type", "type", "count"]
        # df_best_rate_picked_best_reliability_overall = pd.DataFrame(columns=column_names)
        # for exp_type in experimentTypes:
        #     df_best_rate_picked_best_reliability_overall = df_best_rate_picked_best_reliability_overall.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["best_rate_picked_best_reliability_overall"]}, ignore_index=True)
        #     df_best_rate_picked_best_reliability_overall = df_best_rate_picked_best_reliability_overall.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["best_rate_picked_best_reliability_overall"] + deep_analysis[exp_type]["best_rate_picked_not_best_reliability_overall"]}, ignore_index=True)
        #     percentages_output += '{1} -> df_best_rate_picked_best_reliability_overall: {0}%\n'.format(round(deep_analysis[exp_type]["best_rate_picked_best_reliability_overall"]/float(deep_analysis[exp_type]["best_rate_picked_best_reliability_overall"] + deep_analysis[exp_type]["best_rate_picked_not_best_reliability_overall"]),3)*100.0, exp_type)
        # plot_histogram_bars(df_best_rate_picked_best_reliability_overall, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="best_rate_picked_best_reliability_overall")

        # best_rate_picked_best_reliability_of_parent
        # column_names = ["experiment_type", "type", "count"]
        # df_best_rate_picked_best_reliability_of_parent = pd.DataFrame(columns=column_names)
        # for exp_type in experimentTypes:
        #     df_best_rate_picked_best_reliability_of_parent = df_best_rate_picked_best_reliability_of_parent.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["best_rate_picked_best_reliability_of_parent"]}, ignore_index=True)
        #     df_best_rate_picked_best_reliability_of_parent = df_best_rate_picked_best_reliability_of_parent.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["best_rate_picked_best_reliability_of_parent"] + deep_analysis[exp_type]["best_rate_picked_not_best_reliability_of_parent"]}, ignore_index=True)
        #     percentages_output += '{1} -> df_best_rate_picked_best_reliability_of_parent: {0}%\n'.format(round(deep_analysis[exp_type]["best_rate_picked_best_reliability_of_parent"]/float(deep_analysis[exp_type]["best_rate_picked_best_reliability_of_parent"] + deep_analysis[exp_type]["best_rate_picked_not_best_reliability_of_parent"]),3)*100.0, exp_type)
        # plot_histogram_bars(df_best_rate_picked_best_reliability_of_parent, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="best_rate_picked_best_reliability_of_parent")

        # overall_found_equal_reliabilities_with_higher_rate
        column_names = ["experiment_type", "type", "count"]
        df_overall_found_equal_reliabilities_with_higher_rate = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_overall_found_equal_reliabilities_with_higher_rate = df_overall_found_equal_reliabilities_with_higher_rate.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"]}, ignore_index=True)
            df_overall_found_equal_reliabilities_with_higher_rate = df_overall_found_equal_reliabilities_with_higher_rate.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["overall_found_equal_reliabilities_with_lower_or_equal_rate"]}, ignore_index=True)
            percentages_output += '{1} -> df_overall_found_equal_reliabilities_with_higher_rate: {0}%\n'.format(round(deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"]/float(deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["overall_found_equal_reliabilities_with_lower_or_equal_rate"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"]/float(deep_analysis[exp_type]["overall_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["overall_found_equal_reliabilities_with_lower_or_equal_rate"]),3)*100.0)
        plot_histogram_bars(df_overall_found_equal_reliabilities_with_higher_rate, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="overall_found_equal_reliabilities_with_higher_rate")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # parent_found_equal_reliabilities_with_higher_rate
        column_names = ["experiment_type", "type", "count"]
        df_parent_found_equal_reliabilities_with_higher_rate = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_found_equal_reliabilities_with_higher_rate = df_parent_found_equal_reliabilities_with_higher_rate.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"]}, ignore_index=True)
            df_parent_found_equal_reliabilities_with_higher_rate = df_parent_found_equal_reliabilities_with_higher_rate.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["parent_found_equal_reliabilities_with_lower_or_equal_rate"]}, ignore_index=True)
            percentages_output += '{1} -> df_parent_found_equal_reliabilities_with_higher_rate: {0}%\n'.format(round(deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"]/float(deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["parent_found_equal_reliabilities_with_lower_or_equal_rate"]),3)*100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"]/float(deep_analysis[exp_type]["parent_found_equal_reliabilities_with_higher_rate"] + deep_analysis[exp_type]["parent_found_equal_reliabilities_with_lower_or_equal_rate"]),3)*100.0)
        plot_histogram_bars(df_parent_found_equal_reliabilities_with_higher_rate, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="parent_found_equal_reliabilities_with_higher_rate")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # parent_closer_to_root
        column_names = ["experiment_type", "type", "count"]
        df_parent_closer_to_root = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_closer_to_root = df_parent_closer_to_root.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["parent_closer_to_root"]}, ignore_index=True)
            df_parent_closer_to_root = df_parent_closer_to_root.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["parent_closer_to_root"] + deep_analysis[exp_type]["parent_more_away_from_root"]}, ignore_index=True)
            write_output += 'diff_distances_to_root: average = {0}, values = {1}\n'.format(np.mean(deep_analysis[exp_type]["diff_distances_to_root"]),deep_analysis[exp_type]["diff_distances_to_root"], exp_type)
            percentages_output += '{1} -> * df_parent_closer_to_root: {0}%\n'.format(round(deep_analysis[exp_type]["parent_closer_to_root"] / float(deep_analysis[exp_type]["parent_closer_to_root"] + deep_analysis[exp_type]["parent_more_away_from_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["parent_closer_to_root"] / float(deep_analysis[exp_type]["parent_closer_to_root"] + deep_analysis[exp_type]["parent_more_away_from_root"]), 3) * 100.0)
        plot_histogram_bars(df_parent_closer_to_root, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="parent_closer_to_root")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # min distance to root of all possible parents
        column_names = ["experiment_type", "type", "count"]
        df_parent_min_distance_to_root = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_min_distance_to_root = df_parent_min_distance_to_root.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_min_distance_to_root_of_possible_parents"]}, ignore_index=True)
            df_parent_min_distance_to_root = df_parent_min_distance_to_root.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_parent_min_distance_to_root: {0}%\n'.format(round(deep_analysis[exp_type]["num_min_distance_to_root_of_possible_parents"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_min_distance_to_root_of_possible_parents"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_parent_min_distance_to_root, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="df_parent_min_distance_to_root")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # max distance to root of all possible parents
        column_names = ["experiment_type", "type", "count"]
        df_parent_max_distance_to_root = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_max_distance_to_root = df_parent_max_distance_to_root.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_max_distance_to_root_of_possible_parents"]}, ignore_index=True)
            df_parent_max_distance_to_root = df_parent_max_distance_to_root.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_parent_max_distance_to_root: {0}%\n'.format(round(deep_analysis[exp_type]["num_max_distance_to_root_of_possible_parents"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_max_distance_to_root_of_possible_parents"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_parent_max_distance_to_root, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="df_parent_max_distance_to_root")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # min distance to node of all possible parents
        column_names = ["experiment_type", "type", "count"]
        df_parent_min_distance_to_node = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_min_distance_to_node = df_parent_min_distance_to_node.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_min_distance_to_node"]}, ignore_index=True)
            df_parent_min_distance_to_node = df_parent_min_distance_to_node.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_parent_min_distance_to_node: {0}%\n'.format(round(deep_analysis[exp_type]["num_min_distance_to_node"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_min_distance_to_node"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_parent_min_distance_to_node, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="df_parent_min_distance_to_node")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # max distance to root of all possible parents
        column_names = ["experiment_type", "type", "count"]
        df_parent_max_distance_to_node = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_parent_max_distance_to_node = df_parent_max_distance_to_node.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_max_distance_to_node"]}, ignore_index=True)
            df_parent_max_distance_to_node = df_parent_max_distance_to_node.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_parent_max_distance_to_node: {0}%\n'.format(round(deep_analysis[exp_type]["num_max_distance_to_node"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_max_distance_to_node"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_parent_max_distance_to_node, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="df_parent_max_distance_to_node")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # chosen parent with min hop count
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_with_min_hopcount = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_with_min_hopcount = df_num_parent_with_min_hopcount.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_with_min_hopcount"]}, ignore_index=True)
            df_num_parent_with_min_hopcount = df_num_parent_with_min_hopcount.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_with_min_hopcount: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_with_min_hopcount"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_with_min_hopcount"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_with_min_hopcount, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_with_min_hopcount")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # chosen parent with max hop count
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_with_max_hopcount = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_with_max_hopcount = df_num_parent_with_max_hopcount.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_with_max_hopcount"]}, ignore_index=True)
            df_num_parent_with_max_hopcount = df_num_parent_with_max_hopcount.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_with_max_hopcount: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_with_max_hopcount"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_with_max_hopcount"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_with_max_hopcount, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_with_max_hopcount")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # chosen parent with min num of descendants
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_with_min_descendants = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_with_min_descendants = df_num_parent_with_min_descendants.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_with_min_descendants"]}, ignore_index=True)
            df_num_parent_with_min_descendants = df_num_parent_with_min_descendants.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_with_min_descendants: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_with_min_descendants"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_with_min_descendants"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_with_min_descendants, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_with_min_descendants")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # chosen parent with max num of descendants
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_with_max_descendants = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_with_max_descendants = df_num_parent_with_max_hopcount.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_with_max_descendants"]}, ignore_index=True)
            df_num_parent_with_max_descendants = df_num_parent_with_max_hopcount.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_with_max_descendants: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_with_max_descendants"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_with_max_descendants"] / float(deep_analysis[exp_type]["total_num_motes"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_with_max_descendants, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_with_max_descendants")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # reliability parent
        # better reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_better_reliability = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_better_reliability = df_num_parent_better_reliability.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_better_reliability"]}, ignore_index=True)
            df_num_parent_better_reliability = df_num_parent_better_reliability.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_better_reliability: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_better_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_better_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_better_reliability, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_better_reliability")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # equal reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_equal_reliability = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_equal_reliability = df_num_parent_equal_reliability.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_equal_reliability"]}, ignore_index=True)
            df_num_parent_equal_reliability = df_num_parent_equal_reliability.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_equal_reliability: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_equal_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_equal_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_equal_reliability, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_equal_reliability")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # less reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_less_reliability = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_less_reliability = df_num_parent_less_reliability.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_less_reliability"]}, ignore_index=True)
            df_num_parent_less_reliability = df_num_parent_less_reliability.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_less_reliability: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_less_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_less_reliability"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_less_reliability, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_less_reliability")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # reliability parent _with_lessparents
        # better reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_better_reliability_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_better_reliability_with_lessparents = df_num_parent_better_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_better_reliability_with_lessparents"]}, ignore_index=True)
            df_num_parent_better_reliability_with_lessparents = df_num_parent_better_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_reliability"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_better_reliability_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_better_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_better_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_better_reliability_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_better_reliability_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # equal reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_equal_reliability_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_equal_reliability_with_lessparents = df_num_parent_equal_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_equal_reliability_with_lessparents"]}, ignore_index=True)
            df_num_parent_equal_reliability_with_lessparents = df_num_parent_equal_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_reliability"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_equal_reliability_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_equal_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_equal_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_equal_reliability_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_equal_reliability_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # less reliability
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_less_reliability_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_less_reliability_with_lessparents = df_num_parent_less_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_less_reliability_with_lessparents"]}, ignore_index=True)
            df_num_parent_less_reliability_with_lessparents = df_num_parent_less_reliability_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_reliability"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_less_reliability_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_less_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_less_reliability_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_reliability"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_less_reliability_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_less_reliability_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # rate parent
        # better rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_better_rate = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_better_rate = df_num_parent_better_rate.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_better_rate"]}, ignore_index=True)
            df_num_parent_better_rate = df_num_parent_better_rate.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_better_rate: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_better_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_better_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_better_rate, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_better_rate")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # equal rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_equal_rate = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_equal_rate = df_num_parent_equal_rate.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_equal_rate"]}, ignore_index=True)
            df_num_parent_equal_rate = df_num_parent_equal_rate.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_equal_rate: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_equal_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_equal_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_equal_rate, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_equal_rate")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # less rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_less_rate = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_less_rate = df_num_parent_less_rate.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_less_rate"]}, ignore_index=True)
            df_num_parent_less_rate = df_num_parent_less_rate.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_num_motes_without_root"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_less_rate: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_less_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_less_rate"] / float(deep_analysis[exp_type]["total_num_motes_without_root"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_less_rate, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_less_rate")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # rate parent _with_lessparents
        # better rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_better_rate_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_better_rate_with_lessparents = df_num_parent_better_rate_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_better_rate_with_lessparents"]}, ignore_index=True)
            df_num_parent_better_rate_with_lessparents = df_num_parent_better_rate_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_rate"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_better_rate_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_better_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_better_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_better_rate_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_better_rate_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # equal rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_equal_rate_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_equal_rate_with_lessparents = df_num_parent_equal_rate_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_equal_rate_with_lessparents"]}, ignore_index=True)
            df_num_parent_equal_rate_with_lessparents = df_num_parent_equal_rate_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_rate"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_equal_rate_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_equal_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_equal_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_equal_rate_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_equal_rate_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # less rate
        column_names = ["experiment_type", "type", "count"]
        df_num_parent_less_rate_with_lessparents = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_num_parent_less_rate_with_lessparents = df_num_parent_less_rate_with_lessparents.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_parent_less_rate_with_lessparents"]}, ignore_index=True)
            df_num_parent_less_rate_with_lessparents = df_num_parent_less_rate_with_lessparents.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["total_has_parent_with_less_rate"]}, ignore_index=True)
            percentages_output += '{1} -> * df_num_parent_less_rate_with_lessparents: {0}%\n'.format(round(deep_analysis[exp_type]["num_parent_less_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_parent_less_rate_with_lessparents"] / float(deep_analysis[exp_type]["total_has_parent_with_less_rate"]), 3) * 100.0)
        plot_histogram_bars(df_num_parent_less_rate_with_lessparents, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_parent_less_rate_with_lessparents")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # choose root as parent if it was possible
        column_names = ["experiment_type", "type", "count"]
        df_choose_root_while_possible = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_choose_root_while_possible = df_choose_root_while_possible.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_choose_root_while_possible"]}, ignore_index=True)
            df_choose_root_while_possible = df_choose_root_while_possible.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]}, ignore_index=True)
            percentages_output += '{1} -> * df_choose_root_while_possible: {0}%\n'.format(round(deep_analysis[exp_type]["num_choose_root_while_possible"] / float(deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_choose_root_while_possible"] / float(deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]), 3) * 100.0)
        plot_histogram_bars(df_choose_root_while_possible, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_choose_root_while_possible")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # did not choose root as parent while it was possible
        column_names = ["experiment_type", "type", "count"]
        df_choose_root_not_while_possible = pd.DataFrame(columns=column_names)
        vals = []
        for exp_type in experimentTypes:
            df_choose_root_not_while_possible = df_choose_root_not_while_possible.append({"experiment_type": exp_type, "type": "picked_best", "count": deep_analysis[exp_type]["num_choose_root_not_while_possible"]}, ignore_index=True)
            df_choose_root_not_while_possible = df_choose_root_not_while_possible.append({"experiment_type": exp_type, "type": "total", "count": deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]}, ignore_index=True)
            percentages_output += '{1} -> * df_choose_root_not_while_possible: {0}%\n'.format(round(deep_analysis[exp_type]["num_choose_root_not_while_possible"] / float(deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]), 3) * 100.0, exp_type)
            vals.append(round(deep_analysis[exp_type]["num_choose_root_not_while_possible"] / float(deep_analysis[exp_type]["num_choose_root_while_possible"] + deep_analysis[exp_type]["num_choose_root_not_while_possible"]), 3) * 100.0)
        plot_histogram_bars(df_choose_root_not_while_possible, x_metric='experiment_type', y_metric='count', x_label='Experiment', y_label='Count', hue="type", output_dir=output_dir, name_suffix="num_choose_root_not_while_possible")
        percentages_output += 'avg = {0}, std dev = {1}\n'.format(np.mean(vals), np.std(vals, ddof=1))

        # average parents to choose out
        for exp_type in experimentTypes:
            percentages_output += '{1} -> average_parent_to_choose_out: {0}%\n'.format(np.mean(deep_analysis[exp_type]["lsts_possible_parents"]), exp_type)
        # average rates to parent
        for exp_type in experimentTypes:
            percentages_output += '{1} -> lst_of_all_rates: {0}%\n'.format(np.mean(deep_analysis[exp_type]["lst_of_all_rates"]), exp_type)

        path_to_output_file = "{0}/output_data.txt".format(output_dir)
        output_file = open(path_to_output_file, "w")
        output_file.writelines(write_output)
        output_file.close()

        path_to_output_file = "{0}/percentage_output_data.txt".format(output_dir)
        output_file = open(path_to_output_file, "w")
        output_file.writelines(percentages_output)
        output_file.close()

        for exp_type in experimentTypes:

            plot_histogram(data=all_reliabilities[exp_type], x_label='all-reliabilities', output_dir=output_dir, name_suffix=exp_type)
            plot_histogram(data=individual_reliabilities[exp_type], x_label='ind-reliabilities', output_dir=output_dir, name_suffix=exp_type)

            # count all the reliabilities
            unique_all = {rel : all_reliabilities[exp_type].count(rel) for rel in set(all_reliabilities[exp_type])}
            unique_individual = {rel : individual_reliabilities[exp_type].count(rel) for rel in set(individual_reliabilities[exp_type])}
            column_names = ["experimentType", "Type", "reliability", "count"]
            df_count = pd.DataFrame(columns=column_names)
            for rel, count in unique_all.items():
                df_count = df_count.append({"experimentType": exp_type, "Type": "available", "reliability": rel, "count": count}, ignore_index=True)
            for rel, count in unique_individual.items():
                df_count = df_count.append({"experimentType": exp_type, "Type": "used", "reliability": rel, "count": count}, ignore_index=True)
            # plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="type", output_dir=output_dir, name_suffix=exp_type)
            plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="Type", output_dir=output_dir, name_suffix="all-{0}".format(exp_type))
            df_count = df_count.loc[df_count["reliability"] != 1.0]
            plot_histogram_bars(df_count, x_metric='reliability', y_metric='count', x_label='Reliability', y_label='Count', hue="Type", output_dir=output_dir, name_suffix=exp_type)
