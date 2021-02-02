#!/usr/bin/python
"""
\brief Wireless propagation model.

\author Thomas Watteyne <watteyne@eecs.berkeley.edu>
\author Kazushi Muraoka <k-muraoka@eecs.berkeley.edu>
\author Nicola Accettura <nicola.accettura@eecs.berkeley.edu>
\author Xavier Vilajosana <xvilajosana@eecs.berkeley.edu>
"""

#============================ logging =========================================

import logging
class NullHandler(logging.Handler):
    def emit(self, record):
        pass
log = logging.getLogger('Propagation')
log.setLevel(logging.DEBUG)
log.addHandler(NullHandler())

#============================ imports =========================================

import threading
import random
import math
import copy
from abc import ABCMeta, abstractmethod

import Topology
import SimSettings
import SimEngine
import Mote
import Modulation

#============================ defines =========================================

#============================ functions =======================================

def _dBmTomW(dBm):
    """ translate dBm to mW """
    return math.pow(10.0, dBm / 10.0)


def _mWTodBm(mW):
    """ translate dBm to mW """
    return 10 * math.log10(mW)

#============================ classes =========================================

class Propagation(object):

    def __new__(cls, *args, **kwargs):
        """
        This method instantiates the proper `Propagate` class given the simulator settings.
        :return: a Propagate class depending on the settings
        :rtype: PropagationFromModel | PropagationFormTrace
        """
        settings = SimSettings.SimSettings()
        if hasattr(settings, "scenario"):
            return PropagationFromTrace()
        else:
            return PropagationFromModel()


class PropagationCreator(object):
    """
    This class is a meta class, it is not mean to be instantiated.
    """

    __metaclass__ = ABCMeta

    #===== start singleton
    _instance      = None
    _init          = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PropagationCreator,cls).__new__(cls, *args, **kwargs)
        return cls._instance
    #===== end singleton

    def __init__(self):
        #===== start singleton
        # don't re-initialize an instance (needed because singleton)
        if self._init:
            return
        self._init = True
        #===== end singleton

        # store params
        self.settings                  = SimSettings.SimSettings()
        self.engine                    = SimEngine.SimEngine()

        # random.seed(self.settings.seed)

        self.genPropagation = random.Random()
        self.genPropagation.seed(self.settings.seed)

        # variables
        self.dataLock                  = threading.Lock()
        self.receivers                 = [] # motes with radios currently listening
        self.transmissions             = [] # ongoing transmissions

        self.storedReceivers           = []
        self.storedTransmissions       = []

        # schedule propagation task
        self._schedule_propagate()

    def destroy(self):
        self._instance                 = None
        self._init                     = False

    #======================== public ==========================================

    #===== communication

    def print_random(self):
        return self.genPropagation.random()

    def startRx(self,mote,channel, aggregatedInfo=None):
        """ add a mote as listener on a channel"""
        with self.dataLock:
            self.receivers += [{
                'mote':                mote,
                'channel':             channel,
                'aggregatedInfo':      aggregatedInfo,
            }]

    def startTx(self,channel,type,code,smac,dmac,srcIp,dstIp,srcRoute, payload, aggregatedInfo=None):
        """ add a mote as using a channel for tx"""
        with self.dataLock:
            self.transmissions  += [{
                'channel':             channel,
                'type':                type,
                'code':                code,
                'smac':                smac,
                'dmac':                dmac,
                'srcIp':               srcIp,
                'dstIp':               dstIp,
                'sourceRoute':         srcRoute,
                'payload':             payload,
                'aggregatedInfo':      aggregatedInfo
            }]

    @abstractmethod
    def propagate(self):
        """ Simulate the propagation of pkts in a slot. """
        raise NotImplementedError

    #======================== private =========================================

    def _schedule_propagate(self):
        with self.dataLock:
            self.engine.scheduleAtAsn(
                asn         = self.engine.getAsn()+1,# so propagation happens in next slot
                cb          = self.propagate,
                uniqueTag   = (None,'propagation'),
                priority    = 1,
            )

# ==================== Propagation From Model =================================

class PropagationFromModel(PropagationCreator):
    def propagate(self):
        """ Simulate the propagation of pkts in a slot. """

        with self.dataLock:

            asn   = self.engine.getAsn()
            ts    = asn % self.settings.slotframeLength

            for receiver in self.storedReceivers:
                self.receivers += [{
                    'mote':                receiver['mote'], \
                    'channel':             receiver['channel'], \
                    'aggregatedInfo':      receiver['aggregatedInfo']
                }]

            for transmitter in self.storedTransmissions:
                self.transmissions += [{
                    'channel': transmitter['channel'],
                    'type': transmitter['type'],
                    'code': transmitter['code'],
                    'smac': transmitter['smac'],
                    'dmac': transmitter['dmac'],
                    'srcIp': transmitter['srcIp'],
                    'dstIp': transmitter['dstIp'],
                    'sourceRoute': transmitter['sourceRoute'],
                    'payload': transmitter['payload'],
                    'aggregatedInfo': transmitter['aggregatedInfo']
                }]

            self.storedReceivers = [] # clear the stored receivers
            self.storedTransmissions = [] # clear the stored transmissions

            arrivalTime = {}

            # store arrival times of transmitted packets
            for transmission in self.transmissions:
                arrivalTime[transmission['smac']] = transmission['smac'].clock_getOffsetToDagRoot()

            for transmission in self.transmissions:

                i           = 0 # index of a receiver
                isACKed     = False
                isNACKed    = False

                while i<len(self.receivers):

                    if self.receivers[i]['channel']==transmission['channel']:
                        # this receiver is listening on the right channel

                        if self.receivers[i]['mote'] in transmission['dmac']:
                            # this packet is destined for this mote

                            assert self.receivers[i]['aggregatedInfo']['startSlot'] == transmission['aggregatedInfo']['startSlot']
                            assert self.receivers[i]['aggregatedInfo']['endSlot'] == transmission['aggregatedInfo']['endSlot']

                            if not self.settings.noInterference:

                                #================ with interference ===========

                                # other transmissions on the same channel?
                                interferers = []
                                if not self.settings.noNewInterference:
                                    # interferers = [{'mote': t['smac'], \
                                    #                 'aggregatedInfo': {'modulation': t['aggregatedInfo']['modulation'], \
                                    #                                    'packetSize': t['aggregatedInfo']['packetSize'], \
                                    #                                    'startSlot': t['aggregatedInfo']['startSlot'], \
                                    #                                    'endSlot': t['aggregatedInfo']['endSlot']}} \
                                    #                for t in self.transmissions if (t!=transmission) and (t['channel']==transmission['channel']) and Modulation.Modulation()._dBmTomW(t['smac'].getRSSI(self.receivers[i]['mote'])) > Modulation.Modulation().receiverNoise]
                                     interferers = [{'mote': t['smac'], \
                                                    'aggregatedInfo': {'modulation': t['aggregatedInfo']['modulation'], \
                                                                       'packetSize': t['aggregatedInfo']['packetSize'], \
                                                                       'startSlot': t['aggregatedInfo']['startSlot'], \
                                                                       'endSlot': t['aggregatedInfo']['endSlot']}} \
                                                   for t in self.transmissions if (t!=transmission) and (t['channel']==transmission['channel'])]

                                lockOn = transmission['smac']
                                # for itfr in interferers:
                                #     if self.settings.individualModulations == 1:
                                #         # This minRSSI should be the minimum of the modulation. We should make a difference between the minimum and the 75%
                                #         # we only allow an interferer to "take" the slot if it is the first slot
                                #         if arrivalTime[itfr['mote']] < arrivalTime[lockOn] and \
                                #                 self.receivers[i]['mote'].getRSSI(itfr['mote'])>Modulation.Modulation().modulationMinRSSI[self.receivers[i]['aggregatedInfo']['modulation']] and \
                                #                 ts == self.receivers[i]['aggregatedInfo']['startSlot']:
                                #             # lock on interference
                                #             lockOn = itfr['mote']
                                #             # assert False

                                if lockOn == transmission['smac']:
                                    # mote locked in the current signal

                                    # transmission['smac'].schedule[ts]['debug_lockInterference'] += [0] # debug only

                                    assert transmission['aggregatedInfo'] is not None
                                    modulation = transmission['aggregatedInfo']['modulation']

                                    outputFile = ''

                                    allInterferers = []
                                    allTXInterferers = []
                                    hadInterferers = False
                                    pdrWithInterference = 1.0
                                    pdrNoInterference = 1.0
                                    # if we have individual modulations, take all interferers and recalculate the SINR
                                    if ts == transmission['aggregatedInfo']['endSlot']:

                                        allInterferers = interferers[:]
                                        for iferer in self.receivers[i]['aggregatedInfo']['interferers']:
                                            if iferer not in allInterferers:
                                                allInterferers += [iferer]

                                        # if len(allInterferers) > 0:
                                        #     print 'ALL INTEFERERS AT TIMESLOT %d: %s' % (ts, str(allInterferers))

                                        Modulation.Modulation().calculateStartTX(self.receivers[i], self.settings.slotDuration * 1000.0)
                                        Modulation.Modulation().calculateEndTX(self.receivers[i])

                                        outputFile += '------- Transmission at ASN %d: \n' % self.engine.asn
                                        outputFile += '* Mote %d to mote %d\n' % (transmission['smac'].id, self.receivers[i]['mote'].id)
                                        outputFile += '** Start TX (ts %d): %.4f\n' % (self.receivers[i]['aggregatedInfo']['startSlot'], self.receivers[i]['aggregatedInfo']['startTX'])
                                        outputFile += '** End TX (ts %d): %.4f\n' % (self.receivers[i]['aggregatedInfo']['endSlot'], self.receivers[i]['aggregatedInfo']['endTX'])
                                        outputFile += '** TX Duration: %.4f\n' % (self.receivers[i]['aggregatedInfo']['endTX'] - self.receivers[i]['aggregatedInfo']['startTX'])

                                        # print 'Receiver: %s' % self.receivers[i]

                                        for iferer in allInterferers:
                                            Modulation.Modulation().calculateStartTX(iferer, self.settings.slotDuration * 1000.0)
                                            Modulation.Modulation().calculateEndTX(iferer)

                                        if len(allInterferers) > 0:
                                            outputFile += '***** Interferers:\n'
                                        for iferer in allInterferers:
                                            outputFile += '* Interferer mote %d\n' % iferer['mote'].id
                                            outputFile += '** Start TX (ts %d): %.4f\n' % (iferer['aggregatedInfo']['startSlot'], iferer['aggregatedInfo']['startTX'])
                                            outputFile += '** End TX (ts %d): %.4f\n' % (iferer['aggregatedInfo']['endSlot'], iferer['aggregatedInfo']['endTX'])
                                            outputFile += '** TX Duration: %.4f\n' % (iferer['aggregatedInfo']['endTX'] - iferer['aggregatedInfo']['startTX'])

                                        ### calculate the chunks of the receivers WITH the interferers

                                        outputFile += '***** PDR WITH interference\n'

                                        chunksWithInterference = Modulation.Modulation().getNICChunks(self.receivers[i], allInterferers)
                                        outputFile += '*** Chunks (# %d): %s \n' % (len(chunksWithInterference), str(chunksWithInterference))
                                        # print chunksWithInterference

                                        # pdrWithInterference = 1.0
                                        chunkInterferers = {}
                                        for chunk in chunksWithInterference:
                                            chunkInterferers[chunk] = Modulation.Modulation().calculateInterferersPerChunk(chunk, allInterferers)
                                            if len(chunkInterferers[chunk]) > 0:
                                                hadInterferers = True
                                            for ifer in chunkInterferers[chunk]:
                                                if ifer not in allTXInterferers:
                                                    allTXInterferers.append(ifer)
                                            # print 'Chunk %s = %s' % (chunk, chunkInterferers[chunk])
                                            # overwrite the SINR with the correct interferers
                                            sinrWithInterference = self._computeSINR(transmission['smac'], self.receivers[i]['mote'], chunkInterferers[chunk])
                                            outputFile += '*** SINR chunk %s interference = %.10f \n' % (str(chunk), sinrWithInterference)

                                            chunkSize = chunk[1] - chunk[0] # duration in ms
                                            chunkSize /= 1000.0 # duration in seconds
                                            chunkSize *= Modulation.Modulation().modulationRates[modulation] # chunk size in kbit
                                            chunkSize *= 1000.0 # chunk size in bits
                                            chunkSize = chunkSize / 8.0 # chunk size in bytes
                                            outputFile += '*** Chunk size bytes: %.10f \n' % (chunkSize)

                                            # print 'Chunk size: %.4f' % chunkSize
                                            # print 'Old pdrWithInterference: %.4f' % pdrWithInterference

                                            # calculate pdr, including interference
                                            tmpPdrWithInterference = self._computePdrFromSINR(sinrWithInterference, self.receivers[i]['mote'], modulation=modulation, chunkSize=chunkSize)
                                            # it may be better NOT to round because this rounded results will screw up the successive multiplication of tmpPdrWithInterference in pdrWithInterference
                                            # then pdrWithInterference will not be equal to pdrNoInterference later in the code, this is WRONG behaviour
                                            tmptmpPdrWithInterference = self._computePdrFromSINRWithoutCut(sinrWithInterference, self.receivers[i]['mote'], modulation=modulation, chunkSize=chunkSize)
                                            outputFile += '*** Chunk %s has Rounded PDR = %.10f \n' % (str(chunk), tmpPdrWithInterference)
                                            outputFile += '*** Chunk %s has PDR = %.10f \n' % (str(chunk), tmptmpPdrWithInterference)
                                            pdrWithInterference *= tmptmpPdrWithInterference
                                            # print 'New pdrWithInterference: %.4f' % pdrWithInterference

                                        outputFile += '*** Total PDR = %.20f\n' % (pdrWithInterference)

                                        ### calculate the chunks of the receivers WITHOUT the interferers

                                        chunksNoInterference = Modulation.Modulation().getNICChunks(self.receivers[i], [])
                                        assert len(chunksNoInterference) == 1
                                        # print chunksNoInterference

                                        outputFile += '***** PDR WITHOUT interference\n'

                                        chunkNoInterferers = {}
                                        for chunk in chunksNoInterference:
                                            chunkNoInterferers[chunk] = Modulation.Modulation().calculateInterferersPerChunk(chunk, [])
                                            # overwrite the SINR with the correct interferers
                                            sinrNoInterference = self._computeSINR(transmission['smac'], self.receivers[i]['mote'], chunkNoInterferers[chunk])
                                            outputFile += '*** SINR no interference = %.10f \n' % (sinrNoInterference)

                                            chunkSize = chunk[1] - chunk[0] # duration in ms
                                            chunkSize /= 1000.0 # duration in seconds
                                            chunkSize *= Modulation.Modulation().modulationRates[modulation] # chunk size in kbit
                                            chunkSize *= 1000.0 # chunk size in bits
                                            chunkSize = chunkSize / 8.0 # chunk size in bytes
                                            outputFile += '*** Chunk size bytes: %.10f \n' % (chunkSize)

                                            # calculate pdr, including interference
                                            tmpPdrNoInterference = self._computePdrFromSINR(sinrNoInterference, self.receivers[i]['mote'], modulation=modulation, chunkSize=chunkSize)
                                            tmptmpPdrNoInterference = self._computePdrFromSINRWithoutCut(sinrNoInterference, self.receivers[i]['mote'], modulation=modulation, chunkSize=chunkSize)
                                            outputFile += '*** Chunk %s has Rounded PDR = %.10f \n' % (str(chunk), tmpPdrNoInterference)
                                            outputFile += '*** Chunk %s has PDR = %.10f \n' % (str(chunk), tmptmpPdrNoInterference)
                                            pdrNoInterference *= tmptmpPdrNoInterference

                                        outputFile += '*** Total PDR = %.20f\n' % (pdrNoInterference)

                                        if abs(pdrWithInterference - pdrNoInterference) < 1*10**(-5):
                                            outputFile += '***** Equal PDRs\n'
                                        elif pdrWithInterference > pdrNoInterference:
                                            outputFile += '***** pdrWithInterference > pdrNoInterference\n'
                                            # assert False
                                        elif pdrWithInterference < pdrNoInterference:
                                            outputFile += '***** pdrWithInterference < pdrNoInterference\n'

                                        # with open('propagation.log', 'a') as the_file:
                                        #     the_file.write(outputFile)
                                        # assert False


                                        # # overwrite the SINR with the correct interferers
                                        # sinrSigalOnly = self._computeSINR(transmission['smac'], self.receivers[i]['mote'], [])
                                        # sinr = self._computeSINR(transmission['smac'], self.receivers[i]['mote'], allInterferers)
                                        #
                                        # # calculate pdr, including interference
                                        # pdrSignalOnly = self._computePdrFromSINR(sinrSigalOnly, self.receivers[i]['mote'], modulation)
                                        # pdr = self._computePdrFromSINR(sinr, self.receivers[i]['mote'], modulation)

                                    if self.receivers[i]['aggregatedInfo']['endSlot'] == ts and self.receivers[i]['aggregatedInfo']['success']:

                                        if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment and transmission['type'] == Mote.APP_TYPE_DATA:
                                            self.receivers[i]['mote'].totalPropagationData += 1

                                        if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment and transmission['type'] == Mote.APP_TYPE_DATA:
                                            if hadInterferers:
                                                self.receivers[i]['mote'].hadInterferers += 1
                                            if len(allTXInterferers) > 0:
                                                self.receivers[i]['mote'].allInterferers.append(len(allTXInterferers))

                                        # pick a random number
                                        failure = self.genPropagation.random()
                                        if self.settings.noPropagationLoss == 1:
                                            failure = 0.0
                                        if self.settings.measuredData == 1:
                                            # it is not really a pdr calculated with interference taking into account, but we just use this variable to be conform with the already existing code
                                            # we just calculate the PDR based on the received RSSI at the receiver and the used modulation
                                            # and we assume that there were not interferers whose rssi > noise floor, so nobody could interfere
                                            # if there would have been an interferer with rssi > noise, than there should have been a base exception in computeSINR
                                            pdrWithInterference = Modulation.Modulation().predictPRR(modulation, transmission['smac'].getRSSI(self.receivers[i]['mote']))
                                            # if 0.85 >= pdrWithInterference >= 0.7:
                                            #     pdrWithInterference = 0.75
                                            # elif 1.0 >= pdrWithInterference > 0.85:
                                            #     pdrWithInterference = 0.71
                                            # else:
                                            #     pdrWithInterference = 0.0
                                        if pdrWithInterference >= failure:
                                            # packet is received correctly
                                            isACKed, isNACKed = self.receivers[i]['mote'].radio_rxDone(
                                                type=transmission['type'],
                                                code=transmission['code'],
                                                smac=transmission['smac'],
                                                dmac=transmission['dmac'],
                                                srcIp=transmission['srcIp'],
                                                dstIp=transmission['dstIp'],
                                                srcRoute=transmission['sourceRoute'],
                                                payload=transmission['payload']
                                            )

                                            if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment and transmission['type'] == Mote.APP_TYPE_DATA:
                                                self.receivers[i]['mote'].successPropagationData += 1

                                            # this mote stops listening
                                            del self.receivers[i]

                                        else:
                                            # packet is NOT received correctly
                                            self.receivers[i]['mote'].radio_rxDone()

                                            if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment and transmission['type'] == Mote.APP_TYPE_DATA:
                                                if len(allInterferers) == 0 and pdrNoInterference < failure and pdrWithInterference < failure:
                                                    # when the propagation would have failed due to the propagation model and there are no interferers, this is signal loss
                                                    self.receivers[i]['mote'].signalFailures += 1
                                                elif len(allInterferers) > 0 and pdrNoInterference < failure and pdrWithInterference < failure:
                                                    # when the propagation would have failed due to the propagation model independent of the interferers, this is signal loss
                                                    self.receivers[i]['mote'].signalFailures += 1
                                                elif len(allInterferers) > 0 and pdrNoInterference >= failure and pdrWithInterference < failure:
                                                    # when there is no propagation loss, but there is due to the interference, this is interference loss
                                                    # print("There is an interference failure.")
                                                    # for iferer in allInterferers:
                                                    #     print("iferer: {0}".format(iferer['mote'].id))
                                                    # print("failure: {0}".format(failure))
                                                    # print("pdrNoInterference: {0}".format(pdrNoInterference))
                                                    # print("pdrWithInterference: {0}".format(pdrWithInterference))
                                                    self.receivers[i]['mote'].interferenceFailures += 1
                                                    # exit()

                                            del self.receivers[i]

                                    elif self.receivers[i]['aggregatedInfo']['endSlot'] == ts and self.receivers[i]['aggregatedInfo']['success'] is False:
                                        # the node was locked in an interfering signal in the first slot of the transmission slots
                                        self.receivers[i]['mote'].radio_rxDone()

                                        if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment and transmission['type'] == Mote.APP_TYPE_DATA:
                                            self.receivers[i]['mote'].interferenceLockFailures += 1

                                        del self.receivers[i]

                                    elif ts < self.receivers[i]['aggregatedInfo']['endSlot']:
                                        # keep the interferers for the end slot
                                        for iferer in interferers:
                                            if iferer not in self.receivers[i]['aggregatedInfo']['interferers']:
                                                self.receivers[i]['aggregatedInfo']['interferers'] += [iferer]

                                        self.storedReceivers += [{
                                            'mote': self.receivers[i]['mote'], \
                                            'channel': self.receivers[i]['channel'], \
                                            'aggregatedInfo': self.receivers[i]['aggregatedInfo'], \
                                            }]
                                        # this mote stops listening
                                        del self.receivers[i]
                                else:
                                    # mote locked in an interfering signal

                                    # # for debug
                                    # transmission['smac'].schedule[ts]['debug_lockInterference'] += [1]
                                    #
                                    # # receive the interference as if it's a desired packet
                                    # interferers.remove(lockOn)
                                    # pseudo_interferers = interferers + [transmission['smac']]
                                    #
                                    # modulation = None
                                    # if self.settings.individualModulations == 1:
                                    #     assert transmission['aggregatedInfo'] is not None
                                    #     modulation = transmission['aggregatedInfo']['modulation']
                                    #
                                    # # calculate SINR where locked interference and other signals are considered S and I+N respectively
                                    # pseudo_sinr  = self._computeSINR(lockOn,self.receivers[i]['mote'],pseudo_interferers)
                                    # pseudo_pdr   = self._computePdrFromSINR(pseudo_sinr, self.receivers[i]['mote'], modulation)
                                    #
                                    # # pick a random number
                                    # failure = self.genPropagation.random()
                                    # if pseudo_pdr>=failure and self.receivers[i]['mote'].radio_isSync():
                                    #     # success to receive the interference and realize collision
                                    #     self.receivers[i]['mote'].schedule[ts]['rxDetectedCollision'] = True

                                    # desired packet is not received
                                    # in this case, where the transmission length is only 1 slot
                                    if ts == self.receivers[i]['aggregatedInfo']['startSlot'] and ts == self.receivers[i]['aggregatedInfo']['endSlot']:
                                        self.receivers[i]['mote'].radio_rxDone()
                                    elif ts == self.receivers[i]['aggregatedInfo']['startSlot']:
                                        # store this for the end of the transmission
                                        self.receivers[i]['aggregatedInfo']['success'] = False
                                        self.storedReceivers += [{
                                            'mote': self.receivers[i]['mote'], \
                                            'channel': self.receivers[i]['channel'], \
                                            'aggregatedInfo': self.receivers[i]['aggregatedInfo'], \
                                            }]
                                    del self.receivers[i]
                        else:
                            # this packet is NOT destined for this mote

                            # move to the next receiver
                            i += 1

                    else:
                        # this receiver is NOT listening on the right channel

                        # move to the next receiver
                        i += 1

                # indicate to source packet was sent
                assert transmission['aggregatedInfo'] is not None
                if ts == transmission['aggregatedInfo']['endSlot']:
                    transmission['smac'].radio_txDone(isACKed, isNACKed)

            # remaining receivers that does not receive a desired packet
            for r in self.receivers:
                # desired packet is not received
                assert r['aggregatedInfo'] is not None
                if ts == r['aggregatedInfo']['endSlot']:
                    r['mote'].radio_rxDone()
                if ts < r['aggregatedInfo']['endSlot']:
                    self.storedReceivers += [{
                            'mote': r['mote'], \
                            'channel': r['channel'], \
                            'aggregatedInfo': r['aggregatedInfo'], \
                        }]

            # in case of slot aggregation, get all transmissions that are not done
            for transmission in self.transmissions:
                assert transmission['aggregatedInfo'] is not None
                if ts < transmission['aggregatedInfo']['endSlot']:
                    self.storedTransmissions += [{
                        'channel': transmission['channel'],
                        'type': transmission['type'],
                        'code': transmission['code'],
                        'smac': transmission['smac'],
                        'dmac': transmission['dmac'],
                        'srcIp': transmission['srcIp'],
                        'dstIp': transmission['dstIp'],
                        'sourceRoute': transmission['sourceRoute'],
                        'payload': transmission['payload'],
                        'aggregatedInfo': transmission['aggregatedInfo']
                    }]

            # clear all outstanding transmissions
            self.transmissions              = []
            self.receivers                  = []


        self._schedule_propagate()

    # ======================== static =========================================

    @staticmethod
    def _computeSINR(source, destination, interferers):
        """ compute SINR  """

        # if SimSettings.SimSettings().individualModulations == 0:
        #
        #     noise = _dBmTomW(destination.noisepower)
        #     # S = RSSI - N
        #     signal = _dBmTomW(source.getRSSI(destination)) - noise
        #     if signal < 0.0:
        #         # RSSI has not to be below noise level. If this happens, return very low SINR (-10.0dB)
        #         return -10.0
        #
        #     totalInterference = 0.0
        #     for interferer in interferers:
        #         # I = RSSI - N
        #         interference = _dBmTomW(interferer.getRSSI(destination)) - noise
        #         if interference < 0.0:
        #             # RSSI has not to be below noise level. If this happens, set interference 0.0
        #             interference = 0.0
        #         totalInterference += interference
        #
        #     sinr = signal / (totalInterference + noise)
        #
        #     return _mWTodBm(sinr)
        #
        # elif SimSettings.SimSettings().individualModulations == 1:

        noise = Modulation.Modulation().receiverNoise
        signal = _dBmTomW(source.getRSSI(destination))
        if signal < noise:
            # RSSI has not to be below noise level. If this happens, return very low SINR (-10.0dB)
            return -10.0

        totalInterference = 0.0
        interfererCount = 0
        for interferer in interferers:
            # print 'Interferer %d: %.15f dBm' % (interfererCount, interferer.getRSSI(destination))
            interference = _dBmTomW(interferer['mote'].getRSSI(destination))
            # print(interferers)
            # print('interference from interferer {0} to {1} = {2}'.format(interferer['mote'].id, destination.id, interferer['mote'].getRSSI(destination)))
            # print(interference)
            # print(signal)
            # print(noise)
            # print(_mWTodBm(signal))
            # print(_mWTodBm(noise))
            # print('snr:', _mWTodBm(signal/noise))
            # print('self calculated sinr', _mWTodBm(signal/(interference + noise)))
            if interference < noise:
                # RSSI has not to be below noise level. If this happens, set interference 0.0
                interference = 0.0
            else:
                if SimSettings.SimSettings().measuredData == 1:
                    # we do not want any interferers > noise floor, because we do not know what to do with the SINR, Robbe's measurements are not SNR to PRR, but RSSI to PRR
                    raise BaseException('With the measured data model we want to be sure there is absolutely no interferer that has RSSI higher than the noise floor')
            totalInterference += interference
            interfererCount += 1

        sinr = signal / (totalInterference + noise)
        # print('sinr', sinr, interferers)
        # print '* SINR = %.15f dBm (%.15f mW)' % (_mWTodBm(sinr), sinr)
        # print '* Signal = %.15f dBm (%.15f mW)' % (_mWTodBm(signal), signal)
        # if len(interferers) > 0 and totalInterference == 0.0:
        #     print '* Totalinterference = 0.0 mW'
        # elif len(interferers) > 0:
        #     print '* Totalinterference = %.15f dBm (%.15f mW)' % (_mWTodBm(totalInterference), totalInterference)
        # else:
        #     print '* No interferers.'
        # print '* Noise = %.15f dBm (%.15f mW)' % (_mWTodBm(noise), noise)
        if sinr < 1.0:
            return -10.0

        return _mWTodBm(sinr)

    @staticmethod
    def _computePdrFromSINRWithoutCut(sinr, destination=None, modulation=None, chunkSize=None):
        """ compute PDR from SINR """

        # if SimSettings.SimSettings().individualModulations == 0:
        #
        #     equivalentRSSI = _mWTodBm(
        #         _dBmTomW(sinr + destination.noisepower) +
        #         _dBmTomW(destination.noisepower)
        #     )
        #
        #     pdr = Topology.Topology.rssiToPdr(equivalentRSSI)
        #
        #     return pdr
        #
        # elif SimSettings.SimSettings().individualModulations == 1:
        assert modulation is not None

        if SimSettings.SimSettings().measuredData == 1:
            # with the measured data model, we can not calculate a PDR from S(I)NR, so just return 0.0
            return 0.0
        else:
            if sinr < 0.0: # SINR comes in in dBm, 0.0 is 1.0 in mW. This means that when signal / noise in mW is < 1.0 (so signal is smaller than noise), in dBm this will be < 0.0.
                return 0.0

            # has to be in mW
            sinr = _dBmTomW(sinr)

            # BER
            # with theoretical models
            ber = Modulation.Modulation().getBER(sinr, modulation, SimSettings.SimSettings().packetSize)

            pdr = math.pow((1 - ber), (chunkSize * 8))
            # pdr = round(math.pow((1 - ber), (chunkSize * 8)), 3)

            # with measured data
            # pdr = Modulation.Modulation().predictPRR(modulation, sinr)

            return pdr

    @staticmethod
    def _computePdrFromSINR(sinr, destination=None, modulation=None, chunkSize=None):
        """ compute PDR from SINR """

        # if SimSettings.SimSettings().individualModulations == 0:
        #
        #     equivalentRSSI = _mWTodBm(
        #         _dBmTomW(sinr + destination.noisepower) +
        #         _dBmTomW(destination.noisepower)
        #     )
        #
        #     pdr = Topology.Topology.rssiToPdr(equivalentRSSI)
        #
        #     return pdr
        #
        # elif SimSettings.SimSettings().individualModulations == 1:
        assert modulation is not None
        # print 'come in _computePdrFromSINR'
        # print sinr

        if SimSettings.SimSettings().measuredData == 1:
            # with the measured data model, we can not calculate a PDR from S(I)NR, so just return 0.0
            return 0.0
        else:
            if sinr < 0.0: # SINR comes in in dBm, 0.0 is 1.0 in mW. This means that when signal / noise in mW is < 1.0 (so signal is smaller than noise), in dBm this will be < 0.0.
                return 0.0

            # has to be in mW
            sinr = _dBmTomW(sinr)
            # print sinr
            # BER
            # with theoretical models
            ber = Modulation.Modulation().getBER(sinr, modulation, SimSettings.SimSettings().packetSize)

            pdr = Modulation.Modulation()._toPDR(ber, packetSize=chunkSize)
            # pdr = round(math.pow((1 - ber), (chunkSize * 8)), 3)

            # with measured data
            # pdr = round(Modulation.Modulation().predictPRR(modulation, sinr), 3)

            return pdr

# ==================== Propagation From Trace =================================

class PropagationFromTrace(PropagationCreator):
    def propagate(self):
        """ Simulate the propagation of pkts in a slot. """
        raise NotImplementedError
