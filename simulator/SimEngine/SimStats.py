#!/usr/bin/python
"""
\brief Collects and logs statistics about the ongoing simulation.

\author Thomas Watteyne <watteyne@eecs.berkeley.edu>
\author Kazushi Muraoka <k-muraoka@eecs.berkeley.edu>
\author Nicola Accettura <nicola.accettura@eecs.berkeley.edu>
\author Xavier Vilajosana <xvilajosana@eecs.berkeley.edu>
"""

#============================ logging =========================================

import logging
import math
class NullHandler(logging.Handler):
    def emit(self, record):
        pass
log = logging.getLogger('SimStats')
log.setLevel(logging.ERROR)
log.addHandler(NullHandler())

#============================ imports =========================================

import SimEngine
import SimSettings
import Mote
import Modulation

#============================ defines =========================================

#============================ body ============================================

class SimStats(object):

    #===== start singleton
    _instance      = None
    _init          = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SimStats,cls).__new__(cls, *args, **kwargs)
        return cls._instance
    #===== end singleton

    def __init__(self, cpuID, runNum, verbose):

        #===== start singleton
        if self._init:
            return
        self._init = True
        #===== end singleton

        # store params
        self.cpuID                          = cpuID
        self.runNum                         = runNum
        self.verbose                        = verbose

        # local variables
        self.engine                         = SimEngine.SimEngine()
        self.settings                       = SimSettings.SimSettings()

        # stats
        self.stats                          = {}
        self.columnNames                    = []
        self.numCycles                      = 0

        # schedule bootstrap complete
        self.scheduleBootstrapped           = False

        # start file
        if self.runNum==0:
            self._fileWriteHeader()

        # schedule actions
        self.engine.scheduleAtStart(
            cb          = self._actionStart,
        )
        self.engine.scheduleAtAsn(
            asn         = self.engine.getAsn()+self.settings.slotframeLength-1,
            cb          = self._actionEndCycle,
            uniqueTag   = (None,'_actionEndCycle'),
            priority    = 10,
        )
        self.engine.scheduleAtEnd(
            cb          = self._actionEnd,
        )

        self.usedSchedule = None
        self.slotTime = 0.0
        self.airTime = 0.0

    def destroy(self):
        # destroy my own instance
        self._instance                      = None
        self._init                          = False

    #======================== private =========================================

    def _actionStart(self):
        """Called once at beginning of the simulation."""
        pass

    def _actionEndCycle(self):
        """Called at each end of cycle."""

        self.usedSchedule = [0] * self.settings.slotframeLength * self.settings.numChans
        self.slotTime = 0.0
        self.airTime = 0.0

        # add the number of SLEEP slots
        for mote in self.engine.motes:
            if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn < self.engine.asnEndExperiment: # == self.engine.asnEndExperiment will be handled in EndCycle

                if not mote.dagRoot:
                    timeUsed = 0.0
                    maxMinimalCell = (self.settings.nrMinimalCells * (Modulation.Modulation().modulationSlots[self.settings.modulationConfig][Modulation.Modulation().minimalCellModulation[self.settings.modulationConfig]])) - 1
                    for (ts, cell) in mote.schedule.items():
                        # calculate the schedule usage
                        if maxMinimalCell < ts and (cell['dir'] == Mote.DIR_TX or (cell['dir'] == Mote.DIR_TXRX_SHARED and cell['neighbor'] == mote.preferredParent)):
                            listIndex = ts + self.settings.slotframeLength * cell['ch']
                            self.usedSchedule[listIndex] += 1

                        # calculate the total usage in slottime
                        # do not count minimal cells and only look at TX or SHARED to preferred parent
                        # ! this can start introducing problems when there are parent changes and there are more shared cells to old parents fore examples: in that case it won't be exactly accurate
                        if maxMinimalCell < ts and (cell['dir'] == Mote.DIR_TX or (cell['dir'] == Mote.DIR_TXRX_SHARED and cell['neighbor'] == mote.preferredParent)):
                            self.slotTime += (self.settings.slotDuration * 1000.0)

                        # calculate the total airtime in slottime
                        # do not count minimal cells and only look at TX or SHARED
                        # only look for ts and make your calculation based on the number of parentTSs and the modulation type
                        if maxMinimalCell < ts and cell['parentTs'] == ts and (cell['dir'] == Mote.DIR_TX or (cell['dir'] == Mote.DIR_TXRX_SHARED and cell['neighbor'] == mote.preferredParent)):
                            # returns in milliseconds
                            self.airTime += Modulation.Modulation().calculateTXLength(self.settings.packetSize, cell['modulation'])

                # print 'End Cycle'
                # for (ts, cell) in mote.schedule.iteritems():
                #     tss.append(ts)
                # print 'self.engine.asnInitExperiment %d --> %d' % (self.engine.asnInitExperiment, self.engine.asnInitExperiment % self.settings.slotframeLength)
                # print 'self.engine.asnEndExperiment %d --> %d' % (self.engine.asnEndExperiment, self.engine.asnEndExperiment % self.settings.slotframeLength)
                # print 'diff %d' % (self.engine.asnEndExperiment - self.engine.asnInitExperiment)
                # print 'tss %s' % tss
                # print 'ASN begin frame %d' % (self.engine.asn - (self.settings.slotframeLength - 1))
                # print 'ASN %d' % self.engine.asn
                # print 'ASN TS %d' % (self.engine.asn % self.settings.slotframeLength)

                ASNBeginFrame = (self.engine.asn - (self.settings.slotframeLength - 1))
                if ASNBeginFrame <= self.engine.asnInitExperiment <= self.engine.asn:
                    # print '----- MOTE %d -----' % mote.id
                    # print 'ASN %d' % self.engine.asn
                    # print 'ASN TS %d' % (self.engine.asn % self.settings.slotframeLength)
                    # print 'ASNBeginFrame %d' % ASNBeginFrame
                    # print 'ASNBeginFrame TS %d' % (ASNBeginFrame % self.settings.slotframeLength)
                    tsBegin = self.engine.asnInitExperiment % self.settings.slotframeLength # 100
                    # tssUnfiltered = []
                    # tss = []
                    # for (ts, cell) in mote.schedule.iteritems(): # get all the ASNs
                    #     tssUnfiltered.append(ts)
                    #     if ts >= tsBegin:
                    #         tss.append(ts)
                    # print 'self.engine.asnInitExperiment %d' % self.engine.asnInitExperiment
                    # print 'TS of begin experiment: %d' % tsBegin
                    # print 'Used slots (# = %d) = %s' % (len(tssUnfiltered), str(tssUnfiltered))
                    # print 'Remainder used slots (# = %d) = %s' % (len(tss), str(tss))
                    # nrSleepSlots = (self.settings.slotframeLength - tsBegin) # get the remainder of slots in this frame
                    # nrSleepSlots -= len(tss)
                    # mote.nrSleep += nrSleepSlots
                    # for ts in mote.nrNoTxDataRxAck: # get all the ASNs
                    #     if ts >= tsBegin:
                    #         mote.nrSleep += 1
                    # mote.nrNoTxDataRxAck = []
                    # mote.nrSleep -= mote.nrRemovedCells # you should not count this cell for the sleep slots
                    # mote.nrRemovedCells = 0
                    # print '# sleep slots = %d' % nrSleepSlots

                    totalActive = mote.nrIdle + mote.nrIdleNotSync + (mote.nrTxDataRxAck - mote.nrTxDataRxNack) + mote.nrTxDataRxNack + mote.nrTxData + mote.nrTxDataNoAck + mote.nrRxDataTxAck + mote.nrRxData
                    assert totalActive <= (self.settings.slotframeLength - tsBegin)
                    mote.totalSleep += ((self.settings.slotframeLength - tsBegin) - totalActive)
                    mote.totalIdle += mote.nrIdle
                    mote.totalIdleNotSync += mote.nrIdleNotSync
                    mote.totalTxDataRxAck += mote.nrTxDataRxAck
                    mote.totalTxDataRxNack += mote.nrTxDataRxNack
                    mote.totalTxData += mote.nrTxData
                    mote.totalTxDataNoAck += mote.nrTxDataNoAck
                    mote.totalRxDataTxAck += mote.nrRxDataTxAck
                    mote.totalRxData += mote.nrRxData

                    # assert False
                else:
                    mote.nrSleep += self.settings.slotframeLength - len(mote.schedule)
                    mote.nrSleep += len(mote.nrNoTxDataRxAck)
                    mote.nrNoTxDataRxAck = []
                    mote.nrSleep -= mote.nrRemovedCells # you should not count this cell for the sleep slots
                    mote.nrRemovedCells = 0

                    totalActive = mote.nrIdle + mote.nrIdleNotSync + (mote.nrTxDataRxAck - mote.nrTxDataRxNack) + mote.nrTxDataRxNack + mote.nrTxData + mote.nrTxDataNoAck + mote.nrRxDataTxAck + mote.nrRxData
                    assert totalActive <= self.settings.slotframeLength
                    mote.totalSleep += (self.settings.slotframeLength - totalActive)
                    mote.totalIdle += mote.nrIdle
                    mote.totalIdleNotSync += mote.nrIdleNotSync
                    mote.totalTxDataRxAck += mote.nrTxDataRxAck
                    mote.totalTxDataRxNack += mote.nrTxDataRxNack
                    mote.totalTxData += mote.nrTxData
                    mote.totalTxDataNoAck += mote.nrTxDataNoAck
                    mote.totalRxDataTxAck += mote.nrRxDataTxAck
                    mote.totalRxData += mote.nrRxData

                mote.nrIdle = 0
                mote.nrIdleNotSync = 0
                mote.nrTxDataRxAck = 0
                mote.nrTxDataRxNack = 0
                mote.nrTxData = 0
                mote.nrTxDataNoAck = 0
                mote.nrRxDataTxAck = 0
                mote.nrRxData = 0
            elif not self.settings.convergeFirst:
                mote.nrSleep += self.settings.slotframeLength - len(mote.schedule)

            # copy all the modulation states in the total consumption states
            for modulation in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]:
                for state, state_dict in mote.consumption.iteritems():
                    mote.totalConsumption[state][modulation] += mote.consumption[state][modulation]

            # reset the states per modulation
            for modulation in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]:
                for state, state_dict in mote.consumption.iteritems():
                    mote.consumption[state][modulation] = 0

            # print 'Mote %d, %d/101. SLEEP consumption %.4f' % (mote.id, len(mote.schedule), consumption)
        cycle = int(self.engine.getAsn()/self.settings.slotframeLength)

        # print
        if self.verbose:
            print('   cycle: {0}/{1}'.format(cycle,self.settings.numCyclesPerRun-1))

        usedCells = len([i for i in self.usedSchedule if i > 0])
        # calculate all the reserved cells
        allCells = sum(self.usedSchedule)

        # calculated all the overlapping cells
        # xxxxx
        #   xxxx
        # This I count as 3 overlapping cells.
        # xxxxx
        #   xxxx
        #   x
        # This I count as 4 overlapping cells.
        overlappingCells = allCells - usedCells

        # write statistics to output file
        self._fileWriteStats(
            dict(
                {
                    'runNum':              self.runNum,
                    'cycle':               cycle,
                }.items() +
                self._collectSumMoteStats().items()  +
                self._collectScheduleStats().items() +
                {
                    'allCells':           allCells,
                    'usedCells':          usedCells,
                    'overlappingCells':   overlappingCells,
                    'slotTime':           self.slotTime,
                    'airTime':            self.airTime,
                }.items()
            )
        )

        # schedule next statistics collection
        self.engine.scheduleAtAsn(
            asn         = self.engine.getAsn()+self.settings.slotframeLength,
            cb          = self._actionEndCycle,
            uniqueTag   = (None,'_actionEndCycle'),
            priority    = 10,
        )

    def _actionEnd(self):
        """Called once at end of the simulation."""
        assert self.engine.asn == self.engine.asnEndExperiment + self.settings.cooldown  # should be the same
        for mote in self.engine.motes:
            if self.settings.convergeFirst and self.engine.asn >= self.engine.asnInitExperiment and self.engine.asn <= self.engine.asnEndExperiment:
                tsASN = self.engine.asn % self.settings.slotframeLength
                ASNBeginFrame = self.engine.asn - tsASN
                # print 'tsASN %d' % tsASN
                # print 'ASNBeginFrame %d' % ASNBeginFrame
                # print 'self.engine.asnEndExperiment %d' % self.engine.asnEndExperiment
                # print 'self.engine.asn %d' % self.engine.asn
                if ASNBeginFrame <= self.engine.asnEndExperiment <= self.engine.asn:
                    # print 'ASN %d' % self.engine.asn
                    # print 'ASN TS %d' % (self.engine.asn % self.settings.slotframeLength)
                    # print 'ASNBeginFrame %d' % ASNBeginFrame
                    # print 'ASNBeginFrame TS %d' % (ASNBeginFrame % self.settings.slotframeLength)
                    # tssUnfiltered = []
                    # tss = []
                    # for (ts, cell) in mote.schedule.iteritems(): # get all the ASNs
                    #     tssUnfiltered.append(ts)
                    #     if ts <= tsASN:
                    #         tss.append(ts)
                    # # print 'TS of end experiment: %d' % tsASN
                    # # print 'Used slots (# = %d) = %s' % (len(tssUnfiltered), str(tssUnfiltered))
                    # # print 'Remainder used slots (# = %d) = %s' % (len(tss), str(tss))
                    # nrSleepSlots = (tsASN + 1) # + 1 for the 0
                    # nrSleepSlots -= len(tss)
                    # mote.nrSleep += nrSleepSlots
                    # # print '# sleep slots = %d' % nrSleepSlots
                    # for ts in mote.nrNoTxDataRxAck: # get all the ASNs
                    #     if ts <= tsASN:
                    #         mote.nrSleep += 1
                    # mote.nrNoTxDataRxAck = []
                    # mote.nrSleep -= mote.nrRemovedCells # you should not count this cell for the sleep slots
                    # mote.nrRemovedCells = 0

                    totalActive = mote.nrIdle + mote.nrIdleNotSync + (mote.nrTxDataRxAck - mote.nrTxDataRxNack) + mote.nrTxDataRxNack + mote.nrTxData + mote.nrTxDataNoAck + mote.nrRxDataTxAck + mote.nrRxData
                    assert totalActive <= (tsASN + 1)
                    mote.totalSleep += ((tsASN + 1) - totalActive)
                    mote.totalIdle += mote.nrIdle
                    mote.totalIdleNotSync += mote.nrIdleNotSync
                    mote.totalTxDataRxAck += mote.nrTxDataRxAck
                    mote.totalTxDataRxNack += mote.nrTxDataRxNack
                    mote.totalTxData += mote.nrTxData
                    mote.totalTxDataNoAck += mote.nrTxDataNoAck
                    mote.totalRxDataTxAck += mote.nrRxDataTxAck
                    mote.totalRxData += mote.nrRxData

                    totalActiveTotal = mote.totalIdle + mote.totalIdleNotSync + (mote.totalTxDataRxAck - mote.totalTxDataRxNack) + mote.totalTxDataRxNack + mote.totalTxData + mote.totalTxDataNoAck + mote.totalRxDataTxAck + mote.totalRxData
                    print 'Start experiment ASN: {0}'.format(self.engine.asnInitExperiment)
                    print 'End experiment ASN {0}'.format(self.engine.asnEndExperiment)
                    print 'Total number of ASNs {0}'.format(self.engine.asnEndExperiment - self.engine.asnInitExperiment + 1)
                    print 'Total active: {0}'.format(totalActiveTotal)
                    print 'Mote {0}'.format(mote.id)
                    print 'Sleep: {0}'.format(mote.totalSleep)
                    print 'Idle: {0}'.format(mote.totalIdle)
                    print 'nrIdleNotSync: {0}'.format(mote.totalIdleNotSync)
                    print 'nrTxDataRxAck: {0}'.format(mote.totalTxDataRxAck)
                    print 'nrTxDataRxNack: {0}'.format(mote.totalTxDataRxNack)
                    print 'nrTxData: {0}'.format(mote.totalTxData)
                    print 'nrTxDataNoAck: {0}'.format(mote.totalTxDataNoAck)
                    print 'nrRxDataTxAck: {0}'.format(mote.totalRxDataTxAck)
                    print 'nrRxData: {0}'.format(mote.totalRxData)
                    assert (self.engine.asnEndExperiment - self.engine.asnInitExperiment + 1) == (totalActiveTotal + mote.totalSleep)
                # else:
                #     mote.nrSleep += self.settings.slotframeLength - len(mote.schedule)
            elif not self.settings.convergeFirst:
                mote.nrSleep += self.settings.slotframeLength - len(mote.schedule)

            # copy all the modulation states in the total consumption states
            for modulation in Modulation.Modulation().allowedModulations[self.settings.modulationConfig]:
                for state, state_dict in mote.consumption.iteritems():
                    mote.totalConsumption[state][modulation] += mote.consumption[state][modulation]

        self.numCycles = int(self.engine.getAsn()/self.settings.slotframeLength)
        self._fileWriteTopology()
        # write the consumption per modulation to a file
        self._fileWriteConsumption()

    #=== collecting statistics

    def _fileWriteConsumption(self):
        import copy
        import json

        consumptionFile = 'consumption.json'
        consumption = {'totalConsumption': {}, 'startASN': self.engine.asnInitExperiment, 'endASN': self.engine.asnEndExperiment}
        for m in self.engine.motes:
            consumption['totalConsumption'][m.id] = copy.deepcopy(m.totalConsumption)

        with open(consumptionFile, 'w') as the_file:
            the_file.write(json.dumps(consumption))

    def _collectSumMoteStats(self):
        returnVal = {}

        for mote in self.engine.motes:
            moteStats        = mote.getMoteStats()
            if not returnVal:
                returnVal    = moteStats
            else:
                for k in returnVal.keys():
                    returnVal[k] += moteStats[k]

        return returnVal

    def _collectScheduleStats(self):

        returnVal = {}

        # count number of motes with at least one TX cell in their schedule
        numBootstrappedMotes = 0
        dagRoot = None
        for mote in self.engine.motes:
            if len(mote.getTxCells()) > 0:
                numBootstrappedMotes += 1
            if mote.dagRoot is True:
                dagRoot = mote

        if numBootstrappedMotes == len(self.engine.motes) - 1 and self.scheduleBootstrapped is False:
            # dagRoot._log(dagRoot.INFO, "[bootstrap] complete, all motes have at least one TX cell.")
            self.scheduleBootstrapped = True

        # compute the number of schedule collisions

        # Note that this cannot count past schedule collisions which have been relocated by 6top
        # as this is called at the end of cycle
        scheduleCollisions = 0
        txCells = []
        for mote in self.engine.motes:
            for (ts,cell) in mote.schedule.items():
                (ts,ch) = (ts,cell['ch'])
                if cell['dir'] == Mote.DIR_TX:
                    if (ts,ch) in txCells:
                        scheduleCollisions += 1
                    else:
                        txCells += [(ts,ch)]

        # collect collided links
        txLinks = {}
        for mote in self.engine.motes:
            for (ts,cell) in mote.schedule.items():
                if cell['dir'] == Mote.DIR_TX:
                    (ts,ch) = (ts,cell['ch'])
                    (tx,rx) = (mote,cell['neighbor'])
                    if (ts,ch) in txLinks:
                        txLinks[(ts,ch)] += [(tx,rx)]
                    else:
                        txLinks[(ts,ch)]  = [(tx,rx)]

        collidedLinks = [txLinks[(ts,ch)] for (ts,ch) in txLinks if len(txLinks[(ts,ch)])>=2]

        # compute the number of Tx in schedule collision cells
        collidedTxs = 0
        for links in collidedLinks:
            collidedTxs += len(links)

        # compute the number of effective collided Tx
        effectiveCollidedTxs = 0
        insufficientLength   = 0
        for links in collidedLinks:
            for (tx1,rx1) in links:
                for (tx2,rx2) in links:
                    if tx1!=tx2 and rx1!=rx2:
                        # check whether interference from tx1 to rx2 is effective
                        if tx1.getRSSI(rx2) > rx2.minRssi:
                            effectiveCollidedTxs += 1

        # collect shared cell stats for each individual shared cell (by default there is only one)
        for mote in self.engine.motes:
            sharedCellStats        = mote.getSharedCellStats()
            if not returnVal:
                returnVal    = sharedCellStats
            else:
                for k in returnVal.keys():
                    if k in sharedCellStats.keys():
                        returnVal[k] += sharedCellStats[k]
                    else:
                        returnVal[k] += 0

        returnVal.update({'scheduleCollisions':scheduleCollisions, 'collidedTxs': collidedTxs, 'effectiveCollidedTxs': effectiveCollidedTxs , 'numBootstrappedMotes': numBootstrappedMotes})

        return returnVal

    #=== writing to file

    def _fileWriteHeader(self):
        output          = []
        output         += ['## {0} = {1}'.format(k,v) for (k,v) in self.settings.__dict__.items() if not k.startswith('_')]
        output         += ['\n']
        output          = '\n'.join(output)

        with open(self.settings.getOutputFile(),'w') as f:
            f.write(output)

    def _fileWriteStats(self,stats):
        output          = []

        # columnNames
        if not self.columnNames:
            self.columnNames = sorted(stats.keys())
            output     += ['\n# '+' '.join(self.columnNames)]

        # dataline
        formatString    = ' '.join(['{{{0}:>{1}}}'.format(i,len(k)) for (i,k) in enumerate(self.columnNames)])
        formatString   += '\n'

        vals = []
        for k in self.columnNames:
            if type(stats[k])==float:
                vals += ['{0:.3f}'.format(stats[k])]
            else:
                vals += [stats[k]]

        output += ['  '+formatString.format(*tuple(vals))]

        # write to file
        with open(self.settings.getOutputFile(),'a') as f:
            f.write('\n'.join(output))

    def _computeDistance(self, mote, neighbor):
        """
        mote.x and mote.y are in km. This function returns the distance in m.
        """

        return 1000*math.sqrt((mote.x - neighbor.x)**2 +
                              (mote.y - neighbor.y)**2)

    def _fileWriteTopology(self):
        output  = []
        output += [
            '#pos runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@({1:.5f},{2:.5f})@{3}'.format(mote.id,mote.x,mote.y,mote.rank) for mote in self.engine.motes])
            )
        ]
        links = {}
        for m in self.engine.motes:
            for n in self.engine.motes:
                if m==n:
                    continue
                if (n,m) in links:
                    continue
                try:
                    links[(m,n)] = (m.getRSSI(n),m.getPDR(n))
                except KeyError:
                    pass
        output += [
            '#links runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}-{1}@{2:.0f}dBm@{3:.3f}'.format(moteA.id,moteB.id,rssi,pdr) for ((moteA,moteB),(rssi,pdr)) in links.items()])
            )
        ]
        cycles = self.numCycles
        if self.settings.convergeFirst:
            cycles = int((self.engine.asnEndExperiment - self.engine.asnInitExperiment) / self.settings.slotframeLength)
        output += [
            '#aveChargePerCycle runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1:.2f}'.format(mote.id,mote.getMoteStats()['chargeConsumed']/float(cycles)) for mote in self.engine.motes])
            )
        ]

        hopcnt = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                hopcnt[mote.id] = 0
                continue
            hopcnt[mote.id] = 1
            m = mote
            while m.preferredParent.id != 0:
                hopcnt[mote.id] += 1
                m = m.preferredParent
        output += [
            '#hopcount runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id,hopcnt[mote.id]) for mote in self.engine.motes])
            )
        ]

        pp = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pp[mote.id] = None
                continue
            pp[mote.id] = mote.preferredParent.id
        output += [
            '#prefParent runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id,pp[mote.id]) for mote in self.engine.motes])
            )
        ]

        children = {}
        for mote in self.engine.motes:
            children[mote.id] = 0
        for mote in self.engine.motes:
            if mote.preferredParent != None:
                children[mote.preferredParent.id] += 1
        output += [
            '#children runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, children[mote.id]) for mote in self.engine.motes])
            )
        ]

        parrivedToGen = 0
        for mote in self.engine.motes:
            parrivedToGen = parrivedToGen + mote.getMoteStats()['arrivedToGen']
        parrivedToGenDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                parrivedToGenDict[mote] = None
            else:
                parrivedToGenDict[mote] = mote.getMoteStats()['arrivedToGen']
        output += [
            '#PktArrivedToGen runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, parrivedToGenDict[mote]) for mote in self.engine.motes]),parrivedToGen
            )
        ]

        pnotGenerated = 0
        for mote in self.engine.motes:
            pnotGenerated = pnotGenerated + mote.getMoteStats()['notGenerated']
        pnotGeneratedDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pnotGeneratedDict[mote] = None
            else:
                pnotGeneratedDict[mote] = mote.getMoteStats()['notGenerated']
        output += [
            '#PktNotGenerated runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, pnotGeneratedDict[mote]) for mote in self.engine.motes]),pnotGenerated
            )
        ]

        pgen = 0
        for mote in self.engine.motes:
            pgen = pgen + mote.getMoteStats()['pktGen']
        pgenDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pgenDict[mote] = None
            else:
                pgenDict[mote] = mote.getMoteStats()['pktGen']
        output += [
            '#PktGen runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, pgenDict[mote]) for mote in self.engine.motes]),pgen
            )
        ]

        prec = 0
        for mote in self.engine.motes:
            prec = prec + mote.getMoteStats()['pktReceived']
        precDict = {}
        for mote in self.engine.motes:
            if mote.id != 0:
                precDict[mote] = None
            else:
                precDict[mote] = mote.getMoteStats()['pktReceived']
        output += [
            '#PktReceived runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, precDict[mote]) for mote in self.engine.motes]),prec
            )
        ]

        pqueued = 0
        for mote in self.engine.motes:
            pqueued = pqueued + mote.getMoteStats()['dataQueueFill']
        pqueuedDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pqueuedDict[mote] = None
            else:
                pqueuedDict[mote] = mote.getMoteStats()['dataQueueFill']
        output += [
            '#PktInQueue runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, pqueuedDict[mote]) for mote in self.engine.motes]),pqueued
            )
        ]

        pdropqueue = 0
        for mote in self.engine.motes:
            pdropqueue = pdropqueue + mote.getMoteStats()['pktDropQueue']
        pdropqueueDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pdropqueueDict[mote] = None
            else:
                pdropqueueDict[mote] = mote.getMoteStats()['pktDropQueue']
        output += [
            '#PktDropsQueue runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, pdropqueueDict[mote]) for mote in self.engine.motes]),pdropqueue
            )
        ]
        pdropmac = 0
        for mote in self.engine.motes:
            pdropmac = pdropmac + mote.getMoteStats()['pktDropMac']
        pdropmacDict = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                pdropmacDict[mote] = None
            else:
                pdropmacDict[mote] = mote.getMoteStats()['pktDropMac']
        output += [
            '#PktDropsMac runNum={0} {1} {2}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, pdropmacDict[mote]) for mote in self.engine.motes]),pdropmac
            )
        ]

        print 'parrived'
        print parrivedToGen
        print 'pgen'
        print pgen
        print 'prec'
        print prec
        print 'pqueued'
        print pqueued
        print 'pdropqueue'
        print pdropqueue
        print 'pdropmac'
        print pdropmac
        print 'pnotgenerated'
        print pnotGenerated

        print 'total (prec + pqueued + pdropqueue + pdropmac) %d' % (prec + pqueued + pdropqueue + pdropmac)
        print 'total (prec + pqueued + pdropqueue + pdropmac + pnotGenerated) %d' % (prec + pqueued + pdropqueue + pdropmac + pnotGenerated)

        print 'startExperiment'
        print self.engine.asnInitExperiment
        print 'endExperiment'
        print self.engine.asnEndExperiment

        assert pgen == prec + pqueued + pdropqueue + pdropmac
        assert parrivedToGen == prec + pqueued + pdropqueue + pdropmac + pnotGenerated

        avgLatencies = {}
        for mote in self.engine.motes:
            if mote.id in self.engine.motes[0].pktLatencies.keys():
                d = self.engine.motes[0].pktLatencies[mote.id]
                avgLatencies[mote.id] = float(sum(d)) / float(len(d)) if len(d) > 0 else None
            else:
                avgLatencies[mote.id] = None
        # for mote in self.engine.motes[0].pktLatencies.keys():
        #     d = self.engine.motes[0].pktLatencies[mote]
        #     avgLatencies[mote] = float(sum(d)) / float(len(d)) if len(d) > 0 else None
        # avgLatencies[0] = None # set it to zero for stats
        output += [
            '#PktLatencies runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote, avgLatencies[mote]) for mote in avgLatencies])
            )
        ]

        output += [
            '#nrSleep runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalSleep) for mote in self.engine.motes])
            )
        ]

        output += [
            '#nrIdle runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalIdle) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrIdleNotSync runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalIdleNotSync) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrTxDataRxAck runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalTxDataRxAck) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrTxDataRxNack runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalTxDataRxNack) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrTxData runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalTxData) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrTxDataNoAck runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalTxDataNoAck) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrRxDataTxAck runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalRxDataTxAck) for mote in self.engine.motes])
            )
        ]
        output += [
            '#nrRxData runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalRxData) for mote in self.engine.motes])
            )
        ]

        output += [
            '#pkPeriod runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.pkPeriod) for mote in self.engine.motes])
            )
        ]

        output += [
            '#activeDAO runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.activeDAO) for mote in self.engine.motes])
            )
        ]

        output += [
            '#initiatedDAO runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.initiatedDAO) for mote in self.engine.motes])
            )
        ]

        output += [
            '#receivedDAO runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.receivedDAO) for mote in self.engine.motes])
            )
        ]

        if not all(mote.isConverged == True for mote in self.engine.motes):
            converged = len([mote.id for mote in self.engine.motes if mote.isConverged == True])
            total = len(self.engine.motes)
            msg = 'Not all nodes have a dedicated cell converged. Only %d out of %d nodes have a cell.' % (converged, total)
            raise ValueError(msg)

        output += [
            '#dedicatedCellConvergence runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.isConvergedASN) for mote in self.engine.motes])
            )
        ]

        output += [
            '#rplPrefParentChurn runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.rplPrefParentChurns) for mote in self.engine.motes])
            )
        ]

        numberActualParentChanges = {}
        avgDurationParentChange = {}
        for mote in self.engine.motes:
            numberActualParentChanges[mote.id] = len(mote.rplPrefParentASNDiffs)
            avgDurationParentChange[mote.id] = float(sum(mote.rplPrefParentASNDiffs)) / float(len(mote.rplPrefParentASNDiffs)) if len(mote.rplPrefParentASNDiffs) > 0 else None

        output += [
            '#numberActualParentChanges runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, numberActualParentChanges[mote.id]) for mote in self.engine.motes])
            )
        ]
        output += [
            '#avgDurationParentChange runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, avgDurationParentChange[mote.id]) for mote in self.engine.motes])
            )
        ]

        output += [
            '#oldPrefParentRemoval runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.oldPrefParentRemoval) for mote in self.engine.motes])
            )
        ]

        # sixtop stats
        output += [
            '#sixtopTxAddReq runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.sixtopTxAddReq) for mote in self.engine.motes])
            )
        ]
        output += [
            '#sixtopTxAddResp runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.sixtopTxAddResp) for mote in self.engine.motes])
            )
        ]
        output += [
            '#sixtopTxDelReq runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.sixtopTxDelReq) for mote in self.engine.motes])
            )
        ]
        output += [
            '#sixtopTxDelResp runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.sixtopTxDelResp) for mote in self.engine.motes])
            )
        ]
        # output += [
        #     '#slotLength runNum={0} {1}'.format(
        #         self.runNum,
        #         ' '.join(['{0}@{1}'.format(mote.id, Modulation[mote.datarate]) for mote in self.engine.motes])
        #     )
        # ]
        distances = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                distances[mote] = None
            else:
                distances[mote] = self._computeDistance(mote, mote.preferredParent)
        output += [
            '#distance runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, distances[mote]) for mote in self.engine.motes])
            )
        ]

        modulations = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                modulations[mote] = None
            else:
                # in the case of the ILP, this can be None.
                modulations[mote] = mote.getModulation(mote.preferredParent)
        output += [
            '#modulation runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, modulations[mote]) for mote in self.engine.motes])
            )
        ]
        signals = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                signals[mote] = None
            else:
                signals[mote] = mote.getRSSI(mote.preferredParent)
        output += [
            '#signals runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, signals[mote]) for mote in self.engine.motes])
            )
        ]
        reliability = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                reliability[mote] = None
            else:
                modulation = mote.getModulation(mote.preferredParent)
                if mote.getModulation(mote.preferredParent) is None and self.settings.ilpfile is None:
                    assert False # not normal
                elif mote.getModulation(mote.preferredParent) is None and self.settings.ilpfile is not None:
                    reliability[mote] = None # should be 0, because no cell was reserved
                else:
                    reliability[mote] = self.engine.topology._computePDR(mote, mote.preferredParent, modulation=modulation)

        output += [
            '#reliability runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, reliability[mote]) for mote in self.engine.motes])
            )
        ]

        nrSlots = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                nrSlots[mote] = None
            else:
                modulation = mote.getModulation(mote.preferredParent)
                if mote.getModulation(mote.preferredParent) is None and self.settings.ilpfile is None:
                    assert False # not normal
                elif mote.getModulation(mote.preferredParent) is None and self.settings.ilpfile is not None:
                    nrSlots[mote] = None # should be 0, because no cell was reserved
                else:
                    nrSlots[mote] = Modulation.Modulation().modulationSlots[self.settings.modulationConfig][modulation]
        output += [
            '#nrSlots runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, nrSlots[mote]) for mote in self.engine.motes])
            )
        ]

        allocatedBondedSlots = {}
        for mote in self.engine.motes:
            if mote.id == 0:
                allocatedBondedSlots[mote] = None
            else:
                allocatedBondedSlots[mote] = mote.allocatedBondedSlots
        output += [
            '#allocatedBondedSlots runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, allocatedBondedSlots[mote]) for mote in self.engine.motes])
            )
        ]

        output += [
            '#totalPropagationData runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.totalPropagationData) for mote in self.engine.motes])
            )
        ]

        output += [
            '#successPropagationData runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.successPropagationData) for mote in self.engine.motes])
            )
        ]

        output += [
            '#interferenceFailures runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.interferenceFailures) for mote in self.engine.motes])
            )
        ]

        output += [
            '#interferenceLockFailures runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.interferenceLockFailures) for mote in self.engine.motes])
            )
        ]

        output += [
            '#signalFailures runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.signalFailures) for mote in self.engine.motes])
            )
        ]

        output += [
            '#allInterferers runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, sum(mote.allInterferers)) for mote in self.engine.motes])
            )
        ]

        output += [
            '#hadInterferers runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.hadInterferers) for mote in self.engine.motes])
            )
        ]

        output += [
            '#startApp runNum={0} {1}'.format(
                self.runNum,
                ' '.join(['{0}@{1}'.format(mote.id, mote.startApp) for mote in self.engine.motes])
            )
        ]

        if self.settings.withJoin:
            output += [
                '#join runNum={0} {1}'.format(
                    self.runNum,
                    ' '.join(['{0}@{1}'.format(mote.id, mote.joinAsn) for mote in self.engine.motes])
                )
            ]
            output += [
                '#firstBeacon runNum={0} {1}'.format(
                    self.runNum,
                    ' '.join(['{0}@{1}'.format(mote.id, mote.firstBeaconAsn) for mote in self.engine.motes])
                )
            ]
        output  = '\n'.join(output)

        with open(self.settings.getOutputFile(),'a') as f:
            f.write(output)

        # outputSchedule = ''
        # for m in self.engine.motes:
        #     dictCells = {}
        #     for (ts, cell) in m.schedule.iteritems():
        #         if not type(cell['neighbor']) == list:
        #             if cell['neighbor'].id not in dictCells:
        #                 dictCells[cell['neighbor'].id] = []
        #             dictCells[cell['neighbor'].id].append(ts)
        #         else:
        #             if 'broadcast' not in dictCells:
        #                 dictCells['broadcast'] = []
        #             dictCells['broadcast'].append(ts)
        #     outputSchedule += '# mote %d, schedule: ' % m.id
        #     for (neighbor, cellList) in dictCells.iteritems():
        #         outputSchedule += '### neighbor %s, schedule: %s\n' % (str(neighbor), str(cellList))
        #     outputSchedule += '\n\n'
        #
        # with open('schedule.txt','a') as f:
        #     f.write(outputSchedule)
        #
