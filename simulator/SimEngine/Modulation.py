import SimSettings
import scipy.special as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import fnmatch
import math
import json
import numpy as np
import copy
import pandas as pd
import os

T1 = 2 # 2 ms

class Modulation(object):

    #===== start singleton
    _instance      = None
    _init          = False

    modulations = {}
    modulationRates = {}
    modulationMinRSSI = {}
    modulationStableRSSI = {}
    modulationLengthPerSlotType = {}
    modulationChargePerSlotType = {}

    configs = {}

    # depends on config
    allowedModulations = {}
    modulationSlots = {}
    modulationConfigSlotLength = {}
    minimalCellModulation = {}

    receiverNoise = 0.0
    packetSize = 127
    acknowledgementSize = 27

    # current_mA = {'tx': 36, 'rx': 23.5, 'listening': 23.5, 'sleep': 0.00012, 'idle': 1.5, 'cpu': 0.0}  # mA
    # current_mA = {'tx': 62, 'rx': 28, 'listening': 6.28, 'sleep': 0.00003, 'idle': 1.5, 'cpu': 0.0}  # mA
    current_mA = {'tx': 62, 'rx': 28, 'listening': 6.28, 'sleep': 0.00003, 'idle': 1.5, 'cpu': 0.0}  # mA


    VOLTS_RADIO = 3.0

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Modulation,cls).__new__(cls, *args, **kwargs)
        return cls._instance
    #===== end singleton


    def __init__(self, failIfNotInit=False, parsing_results_call=False, config_file=None):

        if failIfNotInit and not self._init:
            raise EnvironmentError('SimEngine singleton not initialized.')

        #===== start singleton
        if self._init:
            return
        self._init = True
        #===== end singleton

        if parsing_results_call:
            # self.readConfig('modulation.json')
            # self.readConfig('simulator/SimEngine/modulation.json')
            # self.readConfig('simulator/SimEngine/modulation-mcs2-mcs3-mcs4.json')
            self.readConfig(config_file)
            # self.readConfig('simulator/SimEngine/modulation_files/modulation_stable_mcs2.json')
            # self.readConfig('modulation_files/modulation_stable_mcs2.json')

            # self.readConfig('modulation-mcs2-mcs3-mcs4.json')
        else:
            modulationFilePath = '../SimEngine/modulation_files/{0}'.format(SimSettings.SimSettings().modulationFile)
            self.readConfig(modulationFilePath)

        if hasattr(SimSettings.SimSettings(), 'modulationConfig'):
            self.packetSize = SimSettings.SimSettings().packetSize
            self.acknowledgementSize = SimSettings.SimSettings().acknowledgementSize
            # the receiver's noise in milliwatt
            self.receiverNoise = self.calculateChannelNoise(bandwidth=SimSettings.SimSettings().channelBandwidth, receiverNF=SimSettings.SimSettings().noiseFigure)
        else:
            self.receiverNoise = self.calculateChannelNoise(bandwidth=156.0, receiverNF=4.5)
            # print(self._mWTodBm(self.receiverNoise))

        self.determineChargePerSlotType()
        self.mcs_to_modulation = {'O4 MCS2': 'QPSK_FEC_1_2_FR_2', 'O4 MCS3': 'QPSK_FEC_1_2', 'O4 MCS4': 'QPSK_FEC_3_4'}
        if not parsing_results_call:
            self.model = {'QPSK_FEC_1_2': LinearRegression(), 'QPSK_FEC_1_2_FR_2': LinearRegression(), 'QPSK_FEC_3_4': LinearRegression()}
            self.regression_polynomial_degree = 3
            self.regression_payload_size = 127
            self.initializeModel()

    def initializeModel(self):
        results_dir = './../../results-measurements-original'
        # results_dir = './results-measurements-original'
        csv_list = fnmatch.filter(os.listdir(results_dir), "TX_%dB.csv" % self.regression_payload_size)
        tx_raw = pd.DataFrame()
        for filename in csv_list:
            if tx_raw.empty:
                tx_raw = pd.read_csv('{0}/{1}'.format(results_dir, filename), header=None)
            else:
                tx_raw = pd.concat([tx_raw, pd.read_csv('{0}/{1}'.format(results_dir, filename), header=None)])
        tx_raw.columns = ["PHY", "PRR", "RSSI"]
        tx_raw.replace({"SUN-OFDM 863-870MHz ": ""}, regex=True, inplace=True)
        tx_raw = tx_raw[tx_raw["PRR"] > 0.05]
        tx_raw = tx_raw[tx_raw["PHY"] != "O4 MCS5"]
        tx_raw.drop_duplicates(inplace=True)
        tx_raw.reset_index(drop=True, inplace=True)

        phy_list = tx_raw["PHY"].unique()
        for index, phy in enumerate(phy_list):
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -103.50]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -103.00]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -102.00]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -101.00]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -100.00]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -99.00]
            tx_raw.loc[len(tx_raw)] = [phy, 1.000, -98.00]
            x = np.array(tx_raw[tx_raw["PHY"] == phy]["RSSI"]).reshape((-1, 1))
            y = np.array(tx_raw[tx_raw["PHY"] == phy]["PRR"])

            transformer = PolynomialFeatures(degree=self.regression_polynomial_degree, include_bias=False)
            transformer.fit(x)
            x_ = transformer.transform(x)
            self.model[self.mcs_to_modulation[phy]].fit(x_, y)
            print('Initialized model for PHY = {0}'.format(phy))

    def predictPRR(self, mcs, rssi):
        # # these are absolute boundaries read from the regression model bounds
        if rssi > -107:
            return 1.0
        elif rssi < -118:
            return 0.0
        if mcs not in self.model:
            raise BaseException('Wrong modulation for prediction of PRR.')
        x_range_array = np.array(rssi).reshape((-1, 1))
        transformer2 = PolynomialFeatures(degree=self.regression_polynomial_degree, include_bias=False)
        transformer2.fit(x_range_array)
        x_range_array_ = transformer2.transform(x_range_array)
        y_pred = self.model[mcs].predict(x_range_array_)
        if 0.0 <= y_pred <= 1.0:
            return y_pred[0]
        elif y_pred < 0.0:
            return 0.0
        elif y_pred > 1.0:
            return 1.0

    def determineChargePerSlotType(self):
        CPUDuration = 0 # ms
        slotTypes = {'idle': 0.0, \
                     'idleNotSync': 0.0, \
                     'sleep': 0.0, \
                     'txDataRxAck': 0.0, \
                     'txDataNoAck': 0.0, \
                     'txData': 0.0, \
                     'rxDataTxAck': 0.0, \
                     'rxData': 0.0}

        # for m in self.modulations:
        #     self.modulationLengthPerSlotType[m] = copy.deepcopy(slotTypes)
        #     self.modulationLengthPerSlotType[m]['idle'] =  3.0
        #     self.modulationLengthPerSlotType[m]['idleNotSync'] = 3.0
        #     self.modulationLengthPerSlotType[m]['sleep'] = 0.0
        #     self.modulationLengthPerSlotType[m]['txDataRxAck'] = self.calculateTXLength(self.packetSize, m) + self.calculateTXLength(self.acknowledgementSize, m) + CPUDuration
        #     self.modulationLengthPerSlotType[m]['txDataNoAck'] = self.calculateTXLength(self.packetSize, m) + CPUDuration
        #     self.modulationLengthPerSlotType[m]['txData'] = self.calculateTXLength(self.packetSize, m) + CPUDuration
        #     self.modulationLengthPerSlotType[m]['rxDataTxAck'] = self.calculateTXLength(self.packetSize, m) + self.calculateTXLength(self.acknowledgementSize, m) + CPUDuration
        #     self.modulationLengthPerSlotType[m]['rxData'] = self.calculateTXLength(self.packetSize, m) + CPUDuration
        #
        #     # uC = mAs = 1000 mAms
        #     self.modulationChargePerSlotType[m] = copy.deepcopy(slotTypes)
        #     self.modulationChargePerSlotType[m]['idle'] = 3.0 * self.current_mA['listening']
        #     self.modulationChargePerSlotType[m]['idleNotSync'] = self.modulationChargePerSlotType[m]['idle']
        #     self.modulationChargePerSlotType[m]['sleep'] = 0.0
        #     self.modulationChargePerSlotType[m]['txDataRxAck'] = (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + (self.calculateTXLength(self.acknowledgementSize, m)) * self.current_mA['rx'] + (CPUDuration) * self.current_mA['cpu']
        #     self.modulationChargePerSlotType[m]['txDataNoAck'] = (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + (CPUDuration) * self.current_mA['cpu']
        #     self.modulationChargePerSlotType[m]['txData'] = (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + (CPUDuration) * self.current_mA['cpu']
        #     self.modulationChargePerSlotType[m]['rxDataTxAck'] = (self.calculateTXLength(self.packetSize, m)) * self.current_mA['rx'] + (self.calculateTXLength(self.acknowledgementSize, m)) * self.current_mA['tx'] + (CPUDuration) * self.current_mA['cpu']
        #     self.modulationChargePerSlotType[m]['rxData'] = (self.calculateTXLength(self.packetSize, m)) * self.current_mA['rx'] + (CPUDuration) * self.current_mA['cpu']

        IDLE_LISTENING = 3.0 # 3 milliseconds for idle data listening
        IDLE_ACK_LISTENING = 1.0 # 1 millisecond for idle listening

        # add 6 bytes for SHR and PHR
        for m in self.modulations:
            self.modulationLengthPerSlotType[m] = copy.deepcopy(slotTypes)
            self.modulationLengthPerSlotType[m]['sleep'] = 0.0 # you can put this to 0, b/c the rest will be 'filled' with the time for sleep and that is what we want in this state
            self.modulationLengthPerSlotType[m]['idle'] = IDLE_LISTENING
            self.modulationLengthPerSlotType[m]['idleNotSync'] = self.modulationLengthPerSlotType[m]['idle']
            self.modulationLengthPerSlotType[m]['txDataRxAck'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m) + self.calculateTXLength(self.acknowledgementSize, m) + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2')
            self.modulationLengthPerSlotType[m]['txDataRxNack'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m) + self.calculateTXLength(self.acknowledgementSize, m) + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2')
            self.modulationLengthPerSlotType[m]['txDataNoAck'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m) + IDLE_ACK_LISTENING
            self.modulationLengthPerSlotType[m]['txData'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m)
            self.modulationLengthPerSlotType[m]['rxDataTxAck'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m) + self.calculateTXLength(self.acknowledgementSize, m) + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2')
            self.modulationLengthPerSlotType[m]['rxData'] = self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') + self.calculateTXLength(self.packetSize, m)
            # print 'modulation %s' % m
            # print '%.6f' % self.calculateTXLength(self.packetSize, m)
            # print '%.6f' % self.calculateTXLength(self.acknowledgementSize, m)
            # print '%.6f' % self.modulationLengthPerSlotType[m]['idle']


            # add 6 bytes for SHR and PHR
            # uC = mAs = 1000 mAms
            self.modulationChargePerSlotType[m] = copy.deepcopy(slotTypes)
            self.modulationChargePerSlotType[m]['sleep'] = 0.0 # you can put this to 0, b/c the rest will be 'filled' with the time for sleep and that is what we want in this state
            self.modulationChargePerSlotType[m]['idle'] = IDLE_LISTENING * self.current_mA['listening']
            self.modulationChargePerSlotType[m]['idleNotSync'] = self.modulationChargePerSlotType[m]['idle']
            self.modulationChargePerSlotType[m]['txDataRxAck'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['tx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + (self.calculateTXLength(self.acknowledgementSize, m)) * self.current_mA['rx'] + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['rx']
            self.modulationChargePerSlotType[m]['txDataRxNack'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['tx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + (self.calculateTXLength(self.acknowledgementSize, m)) * self.current_mA['rx'] + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['rx']
            self.modulationChargePerSlotType[m]['txDataNoAck'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['tx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx'] + IDLE_ACK_LISTENING * self.current_mA['listening']
            self.modulationChargePerSlotType[m]['txData'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['tx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['tx']
            self.modulationChargePerSlotType[m]['rxDataTxAck'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['rx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['rx'] + (self.calculateTXLength(self.acknowledgementSize, m)) * self.current_mA['tx'] + self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['tx']
            self.modulationChargePerSlotType[m]['rxData'] = (self.calculateTXLength(6, 'QPSK_FEC_1_2_FR_2') * self.current_mA['rx']) + (self.calculateTXLength(self.packetSize, m)) * self.current_mA['rx']

    def getCharge(self, slotType, totalSlotLength, modulation):
        assert totalSlotLength > self.modulationLengthPerSlotType[modulation][slotType]

        remainderLength = totalSlotLength - self.modulationLengthPerSlotType[modulation][slotType] # ms
        remainderCharge = remainderLength * self.current_mA['sleep'] # uC

        totalCharge = self.modulationChargePerSlotType[modulation][slotType] + remainderCharge # uC
        # print 'getCharge: slotType{4} --> totalLength {0}, remainderlength {1}, remainderCharge {2}, totalCharge {3}'.format(totalSlotLength, remainderLength, remainderCharge, totalCharge, slotType)
        return totalCharge

    def readConfig(self, configFile):
        with open(configFile) as f:
            data = json.load(f)

            self.modulations = copy.deepcopy(data['modulations']['modulations'])
            self.modulationRates = copy.deepcopy(data['modulations']['modulationRates'])
            self.modulationMinRSSI = copy.deepcopy(data['modulations']['modulationMinRSSI'])
            self.modulationStableRSSI = copy.deepcopy(data['modulations']['modulationStableRSSI'])

            self.configs = copy.deepcopy(data['configurations'].keys())

            for config, contents in data['configurations'].iteritems():
                self.allowedModulations[config] = copy.deepcopy(data['configurations'][config]['allowedModulations'])
                self.modulationSlots[config] = copy.deepcopy(data['configurations'][config]['modulationSlots'])
                self.minimalCellModulation[config] = copy.deepcopy(data['configurations'][config]['minimalCellModulation'])
                self.modulationConfigSlotLength[config] = copy.deepcopy(data['configurations'][config]['slotLength'])

    def _dBmTomW(self, dBm):
        """ translate dBm to mW """
        return math.pow(10.0, dBm / 10.0)

    def _mWTodBm(self, mW):
        """ translate dBm to mW """
        return 10 * math.log10(mW)

    def _toPDR(self, ber, packetSize = 127):
        """ packetSize in bytes"""
        return round(math.pow((1 - ber), (packetSize * 8)), 3)

    def calculateTXLength(self, packetSize, modulation=None):
        ''' Returns the length of the TX/RX data in ms. '''

        assert modulation in self.modulations
        assert modulation in self.modulationRates

        return ((packetSize * 8) / float(float(self.modulationRates[modulation])* 1000.0)) * 1000.0

    def calculateTimingTX(self, receiver, interferers, slotLength):
        # determine the minimum start slot of all ongoing transmissions
        minStartSlot = receiver['startSlot']
        for interferer in interferers:
            if interferer['startSlot'] < minStartSlot:
                minStartSlot = interferer['startSlot']

        # determine all start and end timings of each ongoing transmission
        for interferer in interferers:
            # the timing of the slot in which this interfering TX starts, compared to the minStartSlot start
            interferer['startTX'] = (interferer['startSlot'] - minStartSlot) * slotLength
            # add the time between the start of the aggregated slot and the actual TX
            interferer['startTX'] += T1
            # the end of the TX is the start summed with the duraton of the transmission
            interferer['endTX'] = interferer['startTX'] + self.calculateTXLength(interferer['packetSize'], interferer['modulation'])

        # determine the start and end timings of the ongoing receiver transmission
        receiver['startTX'] = (receiver['startSlot'] - minStartSlot) * slotLength
        receiver['startTX'] += T1
        receiver['endTX'] = receiver['startTX'] + self.calculateTXLength(receiver['packetSize'], receiver['modulation'])

    def calculateStartTX(self, transmission, slotLength):
        transmission['aggregatedInfo']['startTX'] = transmission['aggregatedInfo']['startSlot'] * slotLength
        transmission['aggregatedInfo']['startTX'] += T1
        transmission['aggregatedInfo']['startTX'] = round(transmission['aggregatedInfo']['startTX'], 1)

    def calculateEndTX(self, transmission):
        transmission['aggregatedInfo']['endTX'] = round(transmission['aggregatedInfo']['startTX'] + self.calculateTXLength(transmission['aggregatedInfo']['packetSize'], transmission['aggregatedInfo']['modulation']), 1)

    def getNICChunks(self, receivers, interferers):
        # the min and max bound of the possible interference range
        min = receivers['aggregatedInfo']['startTX']
        max = receivers['aggregatedInfo']['endTX']

        # add all the network interface changes in the startTX and endTX range of the receiver
        NICs = []
        for interferer in interferers:
            if min < interferer['aggregatedInfo']['startTX'] < max:
                NICs.append(interferer['aggregatedInfo']['startTX'])
            if min < interferer['aggregatedInfo']['endTX'] < max:
                NICs.append(interferer['aggregatedInfo']['endTX'])

        NICs += [min, max] # add the startTX and endTX from the receiver
        NICs = sorted(list(set(NICs))) # get unique, sorted list

        # calculate the different chunks from the network interface changes
        chunks = []
        index = 0
        while index < (len(NICs) - 1):
            chunks.append((NICs[index], NICs[index+1]))
            index += 1

        return chunks

    def calculateInterferersPerChunk(self, chunk, interferers):
        start = chunk[0]
        end = chunk[1]

        chunkInterferers = []
        for interferer in interferers:
            if start < interferer['aggregatedInfo']['startTX'] < end:
                assert False
            if start < interferer['aggregatedInfo']['endTX'] < end:
                assert False
            if interferer['aggregatedInfo']['startTX'] <= start and end <= interferer['aggregatedInfo']['endTX']:
                chunkInterferers.append({'mote': interferer['mote'], \
                                         'aggragedInfo': {'startSlot': interferer['aggregatedInfo']['startSlot'], \
                                                          'endSlot': interferer['aggregatedInfo']['endSlot'], \
                                                          'modulation': interferer['aggregatedInfo']['modulation'], \
                                                          'packetSize': interferer['aggregatedInfo']['packetSize'], \
                                                          'startTX': interferer['aggregatedInfo']['startTX'], \
                                                          'endTX': interferer['aggregatedInfo']['endTX']
                                                          }
                                         })

        return chunkInterferers

    # def getBER(self, snr, modulation, packetSize):
    #     if modulation == 'QAM_16_FEC_1_2':
    #         return self.getBerFecModulationNist(lambda x: self.getBer16QamNist(x), snr, packetSize, 1)
    #     elif modulation == 'QAM_16_FEC_3_4':
    #         return self.getBerFecModulationNist(lambda x: self.getBer16QamNist(x), snr, packetSize, 3)
    #     elif modulation == 'QPSK_FEC_1_2_50':
    #         return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x, dummy50=True), snr, packetSize, 1)
    #     elif modulation == 'QPSK_FEC_1_2':
    #         return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x), snr, packetSize, 1)
    #     elif modulation == 'QPSK_FEC_1_2_FR_2':
    #         return self.getBerFecModulationNist(lambda y: self.getBerFrequencyRepetition(lambda x: self.getBerQpskNist(x), y, repetition=2), snr, packetSize, 1)
    #     elif modulation == 'QPSK_FEC_3_4':
    #         return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x), snr, packetSize, 3)
    #     else:
    #         assert False

    def calculateChannelNoise(self, bandwidth, receiverNF):
        ''' Returns the noise floor in milliwatts, N = kTB * 1000
            receiverNF is in dBm
        '''
        # Boltzman's constant
        k = 1.3803e-23
        # bandwidth of the channel (Hz)
        B = bandwidth * 1e03
        # sytem temperature, assumed to be 290K
        T = 290

        noise = k * T * B # watts
        noise *= 1000 # milliwats
        return self._dBmTomW(self._mWTodBm(noise) + receiverNF)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     modulationInstance = Modulation(parsing_results_call=True)
#     # self.mcs_to_modulation = {'O4 MCS2': 'QPSK_FEC_1_2_FR_2', 'O4 MCS3': 'QPSK_FEC_1_2', 'O4 MCS4': 'QPSK_FEC_3_4'}
#
#     # import time
#     # start_time = time.time()
#     # print(modulationInstance.predictPRR('QPSK_FEC_1_2_FR_2', -116))
#     # end_time = time.time()
#     # print('Prediction took {0}'.format(end_time - start_time))
#     # exit()
#
#     # subGHz model
#     antennaGain = 0.0
#     txPower = 10.0
#     distances = []
#     distances.extend(range(1, 1500))
#     rssis = []
#     for distance in distances:
#         # This is the AH macro model.
#         # You add the 21.0 * math.log10(868000000.0 / 900000000.0) for the correct frequency.
#         pathlossDb = 8 + 10.0 * 3.67 * math.log10(distance / 1.0) + 21.0 * math.log10(868000000.0 / 900000000.0)
#         # Calculate the signal
#         rssis += [txPower + antennaGain + antennaGain - pathlossDb]
#
#     m = 16
#     signalSpread = 200000.0
#     phyRate = 200000.0
#
#     EbNos = []
#     EbNos.extend(range(1, 20))
#     BerFecQpskNist12_fr2 = []
#     PdrFecQpskNist12_fr2 = []
#     BerFecQpskNist12 = []
#     PdrFecQpskNist12 = []
#     BerFecQpskNist34 = []
#     PdrFecQpskNist34 = []
#
#     dBmSignals = []
#     dBmSignals.extend(np.arange(-120, -95, 0.2))
#     mWSignals = [modulationInstance._dBmTomW(signal) for signal in dBmSignals]
#     noise = modulationInstance.receiverNoise
#     dBSnrs = [modulationInstance._mWTodBm(signal/float(noise)) for signal in mWSignals]
#     mWSnrs = [modulationInstance._dBmTomW(snr) for snr in dBSnrs]
#     mWEbNos = [(snr / float(phyRate) * signalSpread) for snr in mWSnrs]
#     dbEbNos = [modulationInstance._mWTodBm(EbNo) for EbNo in mWEbNos]
#
#     dbSNRs = []
#
#     csv = 'signal,QPSK_12_FR_2,QPSK_12,QPSK_34\r\n'
#
#     for dbmSignal in dBmSignals:
#         mwSignal = modulationInstance._dBmTomW(dbmSignal)
#         dbSnr = modulationInstance._mWTodBm(mwSignal/float(noise))
#         dbSNRs.append(dbSnr)
#         snr = modulationInstance._dBmTomW(dbSnr)
#         # snr = _dBmTomW(e) * phyRate / float(signalSpread)
#         PdrFecQpskNist12_fr2 += [modulationInstance.predictPRR('QPSK_FEC_1_2_FR_2', dbmSignal)]
#         PdrFecQpskNist12 += [modulationInstance.predictPRR('QPSK_FEC_1_2', dbmSignal)]
#         PdrFecQpskNist34 += [modulationInstance.predictPRR('QPSK_FEC_3_4', dbmSignal)]
#         csv += '{0},{1},{2},{3}\r\n'.format(dbmSignal, PdrFecQpskNist12_fr2[-1], PdrFecQpskNist12[-1], PdrFecQpskNist34[-1])
#
#     with open('signal-to-pdr.csv', 'a') as the_file:
#         the_file.write(csv)
#
#     # plot the subghz model
#     plt.plot(distances, rssis)
#     plt.ylabel('RSSI (dBm)')
#     plt.xlabel('Distance (m)')
#     plt.show()
#
#     plt.plot(dBmSignals, dBSnrs)
#     plt.ylabel('SNR (db)')
#     plt.xlabel('Signals (dBm)')
#     plt.show()
#
#     # plt.plot(dbEbNos, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
#     # plt.plot(dbEbNos, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
#     # plt.plot(dbEbNos, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
#     # plt.legend()
#     # plt.ylabel('PDR')
#     # plt.xlabel('Eb/No (dB)')
#     # plt.show()
#
#     plt.plot(dBmSignals, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
#     plt.plot(dBmSignals, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
#     plt.plot(dBmSignals, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
#     plt.legend(loc='best')
#     plt.ylabel('PDR')
#     plt.xlabel('Signal (dBm)')
#     plt.ylim(0, 1.0)
#     plt.show()
#     print(PdrFecQpskNist34)
#
#     plt.plot(dbSNRs, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
#     plt.plot(dbSNRs, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
#     plt.plot(dbSNRs, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
#     plt.legend(loc='best')
#     plt.ylabel('PDR')
#     plt.xlabel('SNR (dBm)')
#     plt.ylim(0, 1.0)
#     plt.show()
