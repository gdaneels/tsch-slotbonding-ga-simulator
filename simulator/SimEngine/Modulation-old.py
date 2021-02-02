import SimSettings
import scipy.special as sp
import math
import json
import numpy as np
import copy

# QAM_16_FEC_1_2 = 'QAM_16_FEC_1_2'
# QAM_16_FEC_3_4 = 'QAM_16_FEC_3_4'
# QPSK_FEC_1_2 = 'QPSK_FEC_1_2'
# QPSK_FEC_3_4 = 'QPSK_FEC_3_4'
#
# # modulations, BE AWARE: MUST be sorted from slowest to fastest!
# modulations = [QPSK_FEC_1_2, QPSK_FEC_3_4, QAM_16_FEC_1_2, QAM_16_FEC_3_4]
#
# # See "The IEEE 802.15.4g Standard for Smart Metering Utility Networks" paper
# modulationRates = {QPSK_FEC_1_2: 100, \
#                     QPSK_FEC_3_4: 150, \
#                     QAM_16_FEC_1_2: 200, \
#                     QAM_16_FEC_3_4: 300}
#
# # see Google Sheet
# modulationSlots = {QPSK_FEC_1_2: 2, \
#                     QPSK_FEC_3_4: 2, \
#                     QAM_16_FEC_1_2: 2, \
#                     QAM_16_FEC_3_4: 1}
#
# modulationMinRSSI = {QPSK_FEC_1_2: -111.4, \
#                     QPSK_FEC_3_4: -108.6, \
#                     QAM_16_FEC_1_2: -105.1, \
#                     QAM_16_FEC_3_4: -102}
#
# modulationStableRSSI = {QPSK_FEC_1_2: -110.5, \
#                         QPSK_FEC_3_4: -107.65, \
#                         QAM_16_FEC_1_2: -104.05, \
#                         QAM_16_FEC_3_4: -101.0}
#
# # stableRSSI = -110.5 # ~ 75% PDR at minimalCellModulation = QPSK_FEC_1_2
# minimalCellModulation = QPSK_FEC_1_2

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


    def __init__(self, failIfNotInit=False, parsing_results_call=False):

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
            # self.readConfig('simulator/SimEngine/modulation_files/modulation_stable_mcs2.json')
            self.readConfig('modulation_files/modulation_stable_mcs2.json')

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
            self.receiverNoise = self.calculateChannelNoise(bandwidth=200.0, receiverNF=4.5)

        self.determineChargePerSlotType()

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

    def getBer16QamYans(self, snr, m, signalSpread, phyRate):
        # double EbNo = snr * signalSpread / phyRate;
        EbNo = snr * signalSpread / float(phyRate)
        # double z = std::sqrt((1.5 * log2(m) * EbNo) / (m - 1.0));
        z = math.sqrt((1.5 * math.log(m, 2) * EbNo) / float(m - 1.0))
        # double z1 = ((1.0 - 1.0 / std::sqrt (m)) * erfc(z));
        z1 = ((1.0 - 1.0 / float(math.sqrt(m))) * sp.erfc(z))
        # double z2 = 1 - std::pow((1 - z1), 2);
        z2 = 1 - math.pow((1 - z1), 2)
        # double ber = z2 / log2(m);
        ber = z2 / float(math.log(m, 2))
        return ber

    def getBer16QamNist(self, snr):
        # this is a bug fix for the FEC calculation that goes wrong after SNR of 38 dB
        if snr > self._dBmTomW(38.0):
            snr = self._dBmTomW(35.0)
        # double z = std::sqrt(snr / (5.0 * 2.0));
        z = math.sqrt(snr / float(5.0 * 2.0))
        # double ber = 0.75 * 0.5 * erfc(z);
        ber = 0.75 * 0.5 * sp.erfc(z)
        return ber

    def getBerQpskNist(self, snr, dummy50=False, repetition=0):
        # this is a bug fix for the FEC calculation that goes wrong after SNR of 30 dB
        # if snr > self._dBmTomW(30.0):
        #     snr = self._dBmTomW(30.0)
        # 24/09/2019: this is a bug fix for the FEC calculation that goes wrong after SNR of 28 dB, changed this to 28 when I added the frequency repetition
        if snr > self._dBmTomW(28.0):
            snr = self._dBmTomW(28.0)
        # double z = std::sqrt(snr / 2.0);
        if not dummy50:
            z = math.sqrt(snr / 2.0)
        else: # HACKIETIEHACK
            if snr > self._dBmTomW(25.0):
                snr = self._dBmTomW(25.0)
            z = math.sqrt(snr / 0.9)
        # double ber = 0.5 * erfc(z);
        ber = 0.5 * sp.erfc(z)

        if repetition == 2:
            ber *= ber

        return ber

    def getBerFecModulationNist(self, modulation, snr, nbits, bValue):
        # double ber = GetQpskBer(snr);
        ber = modulation(snr)
        # if (ber == 0.0) {
        #     return 1.0;
        #     }

        if ber == 0.0:
            return 1.0
        # double pe = CalculatePe(ber, bValue);
        pe = self.calculatePe(ber, bValue)
        # pe = std::min(pe, 1.0);
        pe = min(pe, 1.0)
        # double pms = std::pow(1 - pe, nbits);
        # pms = math.pow((1 - pe), nbits)

        return pe

    def calculatePe(self, p, bValue):
        # double D = std::sqrt(4.0 * p * (1.0 - p));
        D = math.sqrt(4.0 * p * (1.0 - p))
        # double pe = 1.0;
        pe = 1.0
        if bValue == 1:
            pe = 0.5 * (36.0 * math.pow(D, 10) \
                        + 211.0 * math.pow(D, 12) \
                        + 1404.0 * math.pow(D, 14) \
                        + 11633.0 * math.pow(D, 16) \
                        + 77433.0 * math.pow(D, 18) \
                        + 502690.0 * math.pow(D, 20) \
                        + 3322763.0 * math.pow(D, 22) \
                        + 21292910.0 * math.pow(D, 24) \
                        + 134365911.0 * math.pow(D, 26))
        elif bValue == 3:
            pe = 1.0 / float(2.0 * bValue) * \
            (42.0 * math.pow(D, 5) \
             + 201.0 * math.pow(D, 6) \
             + 1492.0 * math.pow(D, 7) \
             + 10469.0 * math.pow(D, 8) \
             + 62935.0 * math.pow(D, 9) \
             + 379644.0 * math.pow(D, 10) \
             + 2253373.0 * math.pow(D, 11) \
             + 13073811.0 * math.pow(D, 12) \
             + 75152755.0 * math.pow(D, 13) \
             + 428005675.0 * math.pow(D, 14))

        # if bValue == 3 and pe < p:
        #     print 'D = %.15f' % D
        #     print 'pe = %.15f' % pe
        #     print 'p =  %.15f' % p
        #     print '42.0 * math.pow(D, 5) = %.15f' % (42.0 * math.pow(D, 5))
        #     print 'math.pow(D, 14) = %.15f' % (math.pow(D, 14))
        #     print '428005675.0 * math.pow(D, 14) = %.15f' % (428005675.0 * math.pow(D, 14))

        return pe

    def getBerFrequencyRepetition(self, modulation, snr, repetition=0):
        newBer = None
        if repetition == 2:
            newBer = modulation(snr) * modulation(snr)
        elif repetition == 4:
            newBer = modulation(snr) * modulation(snr) * modulation(snr) * modulation(snr)
        else:
            assert False

        return newBer

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

    # def getMinStartSlot(self, transmissions):
    #     print transmissions
    #     minimum = None
    #     for t in transmissions:
    #         if minimum is None or t['aggregatedInfo']['startSlot'] < minimum:
    #             minimum = t['aggregatedInfo']['startSlot']
    #     return minimum

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

    def getBER(self, snr, modulation, packetSize):
        if modulation == 'QAM_16_FEC_1_2':
            return self.getBerFecModulationNist(lambda x: self.getBer16QamNist(x), snr, packetSize, 1)
        elif modulation == 'QAM_16_FEC_3_4':
            return self.getBerFecModulationNist(lambda x: self.getBer16QamNist(x), snr, packetSize, 3)
        elif modulation == 'QPSK_FEC_1_2_50':
            return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x, dummy50=True), snr, packetSize, 1)
        elif modulation == 'QPSK_FEC_1_2':
            return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x), snr, packetSize, 1)
        elif modulation == 'QPSK_FEC_1_2_FR_2':
            return self.getBerFecModulationNist(lambda y: self.getBerFrequencyRepetition(lambda x: self.getBerQpskNist(x), y, repetition=2), snr, packetSize, 1)
        elif modulation == 'QPSK_FEC_3_4':
            return self.getBerFecModulationNist(lambda x: self.getBerQpskNist(x), snr, packetSize, 3)
        else:
            assert False

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    modulationInstance = Modulation(parsing_results_call=True)

    # subGHz model
    antennaGain = 0.0
    txPower = 0.0
    distances = []
    distances.extend(range(1, 1500))
    rssis = []
    for distance in distances:
        # This is the AH macro model.
        # You add the 21.0 * math.log10(868000000.0 / 900000000.0) for the correct frequency.
        pathlossDb = 8 + 10.0 * 3.67 * math.log10(distance / 1.0) + 21.0 * math.log10(868000000.0 / 900000000.0)
        # Calculate the signal
        rssis += [txPower + antennaGain + antennaGain - pathlossDb]

    m = 16
    signalSpread = 200000.0
    phyRate = 200000.0

    EbNos = []
    EbNos.extend(range(1, 20))
    Ber16QamYans = []
    Pdr16QamYans = []
    Ber16QamNist = []
    Pdr16QamNist = []
    BerFec16QamNist12 = []
    PdrFec16QamNist12 = []
    BerFec16QamNist34 = []
    PdrFec16QamNist34 = []
    BerQpskNist = []
    PdrQpskNist = []
    BerFecQpskNist12_fr2 = []
    PdrFecQpskNist12_fr2 = []
    BerFecQpskNist12_fr22 = []
    PdrFecQpskNist12_fr22 = []
    # BerFecQpskNist12_50 = []
    # PdrFecQpskNist12_50 = []
    BerFecQpskNist12 = []
    PdrFecQpskNist12 = []
    BerFecQpskNist34 = []
    PdrFecQpskNist34 = []

    dBmSignals = []
    dBmSignals.extend(np.arange(-140, -60, 0.2))
    mWSignals = [modulationInstance._dBmTomW(signal) for signal in dBmSignals]
    noise = modulationInstance.receiverNoise
    dBSnrs = [modulationInstance._mWTodBm(signal/float(noise)) for signal in mWSignals]
    mWSnrs = [modulationInstance._dBmTomW(snr) for snr in dBSnrs]
    mWEbNos = [(snr / float(phyRate) * signalSpread) for snr in mWSnrs]
    dbEbNos = [modulationInstance._mWTodBm(EbNo) for EbNo in mWEbNos]

    dbSNRs = []

    csv = 'signal,QPSK_12_FR_2,QPSK_12,QPSK_34,QAM_12,QAM_34\r\n'

    # for snr in mWSnrs:

    for dbmSignal in dBmSignals:
        mwSignal = modulationInstance._dBmTomW(dbmSignal)
        dbSnr = modulationInstance._mWTodBm(mwSignal/float(noise))
        dbSNRs.append(dbSnr)
        snr = modulationInstance._dBmTomW(dbSnr)
        # snr = _dBmTomW(e) * phyRate / float(signalSpread)
        Ber16QamYans += [modulationInstance.getBer16QamYans(snr, m, signalSpread, phyRate)]
        Pdr16QamYans += [modulationInstance._toPDR(Ber16QamYans[-1])]
        Ber16QamNist += [modulationInstance.getBer16QamNist(snr)]
        Pdr16QamNist += [modulationInstance._toPDR(Ber16QamNist[-1])]
        BerFec16QamNist12 += [modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), snr, 127, 1)]
        PdrFec16QamNist12 += [modulationInstance._toPDR(BerFec16QamNist12[-1])]
        BerFec16QamNist34 += [modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), snr, 127, 3)]
        PdrFec16QamNist34 += [modulationInstance._toPDR(BerFec16QamNist34[-1])]
        BerQpskNist += [modulationInstance.getBerQpskNist(snr)]
        PdrQpskNist += [modulationInstance._toPDR(BerQpskNist[-1])]
        BerFecQpskNist12_fr2 += [modulationInstance.getBerFecModulationNist(lambda y: modulationInstance.getBerFrequencyRepetition(lambda x: modulationInstance.getBerQpskNist(x), y, repetition=2), snr, 127, 1)]
        PdrFecQpskNist12_fr2 += [modulationInstance._toPDR(BerFecQpskNist12_fr2[-1])]
        BerFecQpskNist12_fr22 += [modulationInstance.getBerFrequencyRepetition(lambda y: modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), y, 127, 1), snr, repetition=2)]
        PdrFecQpskNist12_fr22 += [modulationInstance._toPDR(BerFecQpskNist12_fr22[-1])]
        # BerFecQpskNist12_50 += [modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x, dummy50=True), snr, 127, 1)]
        # PdrFecQpskNist12_50 += [modulationInstance._toPDR(BerFecQpskNist12_50[-1])]
        BerFecQpskNist12 += [modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), snr, 127, 1)]
        PdrFecQpskNist12 += [modulationInstance._toPDR(BerFecQpskNist12[-1])]
        BerFecQpskNist34 += [modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), snr, 127, 3)]
        PdrFecQpskNist34 += [modulationInstance._toPDR(BerFecQpskNist34[-1])]
        csv += '{0},{1},{2},{3},{4},{5}\r\n'.format(dbmSignal, PdrFecQpskNist12_fr2[-1], PdrFecQpskNist12[-1], PdrFecQpskNist34[-1], PdrFec16QamNist12[-1], PdrFec16QamNist34[-1])

    with open('signal-to-pdr.csv', 'a') as the_file:
        the_file.write(csv)

    # print EbNos
    # print BERsYANS
    # print BERsNIST

    # plot the subghz model
    plt.plot(distances, rssis)
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Distance (m)')
    plt.show()

    plt.plot(dBmSignals, dBSnrs)
    plt.ylabel('SNR (db)')
    plt.xlabel('Signals (dBm)')
    plt.show()

    # plt.semilogy(EbNos, Ber16QamYans, label='16QAM Yans')
    # plt.semilogy(dbEbNos, Ber16QamNist, label='16QAM Nist')
    plt.semilogy(dbEbNos, BerFec16QamNist12, label='16QAM Nist 1/2')
    plt.semilogy(dbEbNos, BerFec16QamNist34, label='16QAM Nist 3/4')
    # plt.semilogy(dbEbNos, BerQpskNist, label='QPSK Nist')
    # plt.semilogy(dbEbNos, BerFecQpskNist12_50, label='QPSK Nist, 50 kbits, FEC 1/2')
    plt.semilogy(dbEbNos, BerFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
    # plt.semilogy(dbEbNos, BerFecQpskNist12_fr22, label='QPSK Nist, FR = 2, FEC 1/2')
    plt.semilogy(dbEbNos, BerFecQpskNist12, label='QPSK Nist, FEC 1/2')
    plt.semilogy(dbEbNos, BerFecQpskNist34, label='QPSK Nist, FEC 3/4')
    plt.legend()
    plt.ylabel('BER')
    plt.xlabel('Eb/No (dB)')
    plt.show()

    # plt.semilogy(EbNos, Ber16QamYans, label='16QAM Yans')
    # plt.semilogy(dBmSignals, Ber16QamNist, label='16QAM Nist')
    plt.semilogy(dBmSignals, BerFec16QamNist12, label='16QAM Nist 1/2')
    plt.semilogy(dBmSignals, BerFec16QamNist34, label='16QAM Nist 3/4')
    # plt.semilogy(dBmSignals, BerQpskNist, label='QPSK Nist')
    # plt.semilogy(dBmSignals, BerFecQpskNist12_50, label='QPSK Nist, 50 kbits, FEC 1/2')
    plt.semilogy(dBmSignals, BerFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
    # plt.semilogy(dBmSignals, BerFecQpskNist12_fr22, label='QPSK Nist, FR = 2, FEC 1/2')
    plt.semilogy(dBmSignals, BerFecQpskNist12, label='QPSK Nist, FEC 1/2')
    plt.semilogy(dBmSignals, BerFecQpskNist34, label='QPSK Nist, FEC 3/4')
    plt.legend()
    plt.ylabel('BER')
    plt.xlabel('Signal (dBm)')
    plt.show()

    # plt.plot(dbEbNos, Pdr16QamNist, label='16QAM Nist')
    plt.plot(dbEbNos, PdrFec16QamNist12, label='16QAM Nist 1/2')
    plt.plot(dbEbNos, PdrFec16QamNist34, label='16QAM Nist 3/4')
    # plt.plot(dbEbNos, PdrQpskNist, label='QPSK Nist')
    # plt.plot(dbEbNos, PdrFecQpskNist12_50, label='QPSK Nist, 50 kbits, FEC 1/2')
    plt.plot(dbEbNos, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
    # plt.plot(dbEbNos, PdrFecQpskNist12_fr22, label='QPSK Nist, FR = 2, FEC 1/2')
    plt.plot(dbEbNos, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
    plt.plot(dbEbNos, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
    plt.legend()
    plt.ylabel('PDR')
    plt.xlabel('Eb/No (dB)')
    plt.show()

    # plt.plot(dBmSignals, Pdr16QamNist, label='16QAM Nist')
    plt.plot(dBmSignals, PdrFec16QamNist12, label='16QAM Nist 1/2')
    plt.plot(dBmSignals, PdrFec16QamNist34, label='16QAM Nist 3/4')
    # plt.plot(dBmSignals, PdrQpskNist, label='QPSK Nist')
    # plt.plot(dBmSignals, PdrFecQpskNist12_50, label='QPSK Nist, 50 kbits, FEC 1/2')
    plt.plot(dBmSignals, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
    # plt.plot(dBmSignals, PdrFecQpskNist12_fr22, label='QPSK Nist, FR = 2, FEC 1/2')
    plt.plot(dBmSignals, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
    plt.plot(dBmSignals, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
    plt.legend()
    plt.ylabel('PDR')
    plt.xlabel('Signal (dBm)')
    plt.show()


    # plt.plot(dBmSignals, Pdr16QamNist, label='16QAM Nist')
    plt.plot(dbSNRs, PdrFec16QamNist12, label='16QAM Nist 1/2')
    plt.plot(dbSNRs, PdrFec16QamNist34, label='16QAM Nist 3/4')
    # plt.plot(dBmSignals, PdrQpskNist, label='QPSK Nist')
    # plt.plot(dBmSignals, PdrFecQpskNist12_50, label='QPSK Nist, 50 kbits, FEC 1/2')
    plt.plot(dbSNRs, PdrFecQpskNist12_fr2, label='QPSK Nist, FEC 1/2, FR = 2')
    # plt.plot(dBmSignals, PdrFecQpskNist12_fr22, label='QPSK Nist, FR = 2, FEC 1/2')
    plt.plot(dbSNRs, PdrFecQpskNist12, label='QPSK Nist, FEC 1/2')
    plt.plot(dbSNRs, PdrFecQpskNist34, label='QPSK Nist, FEC 3/4')
    plt.legend()
    plt.ylabel('PDR')
    plt.xlabel('SNR (dBm)')
    plt.show()

    # modulationCharges = {}
    # for m in modulationInstance.modulations:
    #     for length in arange(10.0, 30.0, 0.1):
    #     plt.plot(dBmSignals, PdrFec16QamNist12, label=m)
    # plt.legend()
    # plt.ylabel('uC')
    # plt.xlabel('Total slot length (ms)')
    # plt.show()

    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), modulationInstance._dBmTomW(-111.4)/float(noise), 127, 1))
    # print _toPDR(getBerFecModulationNist(lambda x: getBerQpskNist(x), _dBmTomW(-110.5)/float(noise), 127, 1))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), modulationInstance._dBmTomW(-108.6)/float(noise), 127, 3))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), modulationInstance._dBmTomW(-105.1)/float(noise), 127, 1))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), modulationInstance._dBmTomW(-102)/float(noise), 127, 3))

    print 'Signal to PDRs:'
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), modulationInstance._dBmTomW(-110.5)/float(noise), 127, 1))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBerQpskNist(x), modulationInstance._dBmTomW(-107.65)/float(noise), 127, 3))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), modulationInstance._dBmTomW(-104.05)/float(noise), 127, 1))
    print modulationInstance._toPDR(modulationInstance.getBerFecModulationNist(lambda x: modulationInstance.getBer16QamNist(x), modulationInstance._dBmTomW(-101.0)/float(noise), 127, 3))

    print dBmSignals

    #### INTERFERENCE TEST CODE ####

    # receiver = {'aggregatedInfo': {}}
    # receiver['aggregatedInfo']['startSlot'] = 5
    # receiver['aggregatedInfo']['endSlot'] = 7
    # receiver['aggregatedInfo']['modulation'] = 'QPSK_FEC_1_2'
    # receiver['aggregatedInfo']['packetSize'] = 54
    # interferers = []
    # interferers.append({'aggregatedInfo': {'startSlot': 5, 'endSlot': 6, 'modulation': 'QAM_16_FEC_3_4', 'packetSize': 54}})
    # interferers.append({'aggregatedInfo': {'startSlot': 7, 'endSlot': 9, 'modulation': 'QPSK_FEC_3_4', 'packetSize': 127}})
    # interferers.append({'aggregatedInfo': {'startSlot': 3, 'endSlot': 5, 'modulation': 'QPSK_FEC_1_2', 'packetSize': 23}})
    # interferers.append({'aggregatedInfo': {'startSlot': 6, 'endSlot': 8, 'modulation': 'QPSK_FEC_1_2', 'packetSize': 31}})
    # interferers.append({'aggregatedInfo': {'startSlot': 5, 'endSlot': 7, 'modulation': 'QPSK_FEC_1_2', 'packetSize': 64}})
    #
    # SLOTLENGTH = 10
    # # minStartSlot = modulationInstance.getMinStartSlot(interferers + [receiver])
    #
    # modulationInstance.calculateStartTX(receiver, SLOTLENGTH)
    # modulationInstance.calculateEndTX(receiver)
    # print 'Receiver: %s' % receiver
    #
    # for interferer in interferers:
    #     modulationInstance.calculateStartTX(interferer, SLOTLENGTH)
    #     modulationInstance.calculateEndTX(interferer)
    #     print interferer
    #
    # chunks = modulationInstance.getNICChunks(receiver, interferers)
    # print chunks
    # for chunk in chunks:
    #     print 'Chunk %s = %s' % (chunk, modulationInstance.calculateInterferersPerChunk(chunk, interferers))
