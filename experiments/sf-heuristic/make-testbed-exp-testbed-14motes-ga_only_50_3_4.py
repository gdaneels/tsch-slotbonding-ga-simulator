import random
import json
import os
import sys
sys.path.insert(1, '../../simulator/SimEngine/')

random.seed(100)

ITERATIONS = 14
INPUT_DIR = '../input'
MAX_SEED = 10000
random_seeds = []

##### SIMULATOR SPECIFIC SETTINGS #####

expLength = 300  # seconds
cooldownLength = 60  # seconds

simulator_config = {}
simulator_config['numRuns'] = 1
simulator_config['convergeFirst'] = 1
simulator_config['numChans'] = 3  # take only two channels to limit the total number of slots
simulator_config['numMotes'] = 14  # number of motes
simulator_config['topology'] = 'random'  # we only consider a random topology
simulator_config[
    'nrMinimalCells'] = 0  # we only want one 1 minimal cell, however it won't really be used in the ILP case
simulator_config['trafficGenerator'] = 'ilp'  # the ilp setting deals with the correct ILP traffic
simulator_config['sporadicTraffic'] = 0  # to be sure, set it to 0. we do not want sporadic traffic.
simulator_config['changeParent'] = 0  # we do not change parents
simulator_config[
    'backoffMinExp'] = 1  # we do not have SHARED cells anymore (except for the minimal ones), so we do not care
simulator_config[
    'backoffMaxExp'] = 1  # we do not have SHARED cells anymore (except for the minimal ones), so we do not care
simulator_config['sf'] = 'ilp'  # we only want to use the ILP SF.
simulator_config['minCellsMSF'] = 1  # is ignored.
simulator_config['packetSize'] = 127  # maximum packet size
simulator_config['subGHz'] = 1  # yes, go subGHz
simulator_config['individualModulations'] = 1  # we have modulations per link
simulator_config['pkPeriodVar'] = 0  # important to set to 0 to have a nice equally sent packet distribution

simulator_config['modulationConfig'] = 'MCS234s10ms'
simulator_config['slotDuration'] = 0.010  # will be replaced
simulator_config['slotframeLength'] = 12 # will be replaced
simulator_config['numCyclesPerRun'] = int(expLength / float(simulator_config['slotDuration'] * simulator_config['slotframeLength']))  # will be replaced
simulator_config['cooldown'] = cooldownLength / float(simulator_config['slotDuration'])  # in ASNs

simulator_config['noNewInterference'] = 0  # will be replaced
simulator_config['noPropagationLoss'] = 0  # will be replaced

simulator_config['settlingTime'] = 120  # in seconds
simulator_config['maxToConverge'] = 6060  # in seconds

simulator_config['stableNeighbors'] = 1
simulator_config['measuredData'] = 1

# TODO SEED STILL HAS TO BE TAKEN FROM THE GLOBAL SETTINGS FILE NOW

##### GA SPECIFIC SETTINGS #####

DEFAULT_SETTINGS = '../final-settings/settings-testbed-makenotfeasible-testbed-ga.json'
config = {}
with open(DEFAULT_SETTINGS) as json_file:
    config = json.load(json_file)
    config["simulator"] = simulator_config
    config["simulator"]["modulationFile"] = os.path.basename(config["modulations"]["modulations_file"])

##### CONFIGURATIONS #####

configurations = {
'sfl_153ms_motes_12_top_gaheuristiconly50_delta_000_f_34_': {
    "slotDuration": 0.09,
    "slotframeLength": 17,
    "delta": 0.05
},
'sfl_324ms_motes_12_top_gaheuristiconly50_delta_000_f_34_': {
    "slotDuration": 0.09,
    "slotframeLength": 36,
    "delta": 0.05
}
}

##### SETTINGS FILE #####

def getUniqueSeed():
    seed = random.randint(0, MAX_SEED)
    while seed in random_seeds:
        seed = random.randint(0, MAX_SEED)
    random_seeds.append(seed)
    return seed

if __name__ == "__main__":
    seeds = [getUniqueSeed() for x in range(ITERATIONS)]
    # NODES = ['nuc9-3', 'nuc9-5', 'nuc9-6', 'nuc9-14', 'nuc9-33', 'nuc9-34', 'nuc9-35', 'nuc9-37', 'nuc10-1', 'nuc10-5',
    #          'nuc10-18', 'nuc10-20', 'nuc10-33', 'nuc10-34', 'nuc10-35']
    # NODES = ['nuc10-17', 'nuc10-19', 'nuc10-26', 'nuc10-35', 'nuc9-14', 'nuc9-22', 'nuc9-24', 'nuc9-29', 'nuc9-6']
    NODES = ['nuc9-3', 'nuc9-6', 'nuc9-11', 'nuc9-14', 'nuc9-22', 'nuc9-24', 'nuc9-29', 'nuc9-33', 'nuc9-37', 'nuc10-19', 'nuc10-21', 'nuc10-26', 'nuc10-31', 'nuc10-35']
    if not os.path.exists(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    for c_name, c in configurations.items():
        cnt = 0
        # set config
        config["sf-heuristic"]["delta"] = c['delta']
        config["sf-heuristic"]["testbed_results"]["only_50"] = 1
        config["simulator"]['slotDuration'] = c['slotDuration']
        config["simulator"]['slotframeLength'] = c['slotframeLength']
        config["simulator"]['numCyclesPerRun'] = int(expLength / float(simulator_config['slotDuration'] * config["simulator"]['slotframeLength']))  # will be replaced
        config["simulator"]['cooldown'] = cooldownLength / float(simulator_config['slotDuration'])  # in ASNs
        while cnt < ITERATIONS:
            config["seed"] = seeds[cnt]
            config["name"] = "seed-{0}-exp-{1}".format(config["seed"], c_name)
            config["sf-heuristic"]["testbed_results"]["root"] = NODES[cnt % len(NODES)]
            nameFile = '{input_dir}/ga_seed_{seed}_c_{config_modulation}_ss_{slotframe_size}_exp_{file_suffix}.json'.format(
                input_dir=INPUT_DIR,
                seed=config["seed"],
                config_modulation=config['simulator']['modulationConfig'],
                slotframe_size=int(1000*config['simulator']['slotframeLength']*config['simulator']['slotDuration']),
                file_suffix=c_name)
            with open(nameFile, 'w') as outfile:
                json.dump(config, outfile)
            cnt += 1
