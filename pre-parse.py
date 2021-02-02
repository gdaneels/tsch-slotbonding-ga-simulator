import sys
import os
import re
import json
import copy

ERROR_ERROR_LOG = 'ERROR_ERROR_LOG'
ERROR_MULTIPLE_WORKERS = 'ERROR_MULTIPLE_WORKERS'
ERROR_GA_SOLUTION = 'ERROR_GA_SOLUTION'
ERROR_GA_OUTPUT = 'ERROR_GA_OUTPUT'
ERROR_SIMULATOR_TOPOLOGY = 'ERROR_SIMULATOR_TOPOLOGY'
ERROR_SIMULATOR_TOPOLOGY_OUTPUT = 'ERROR_SIMULATOR_TOPOLOGY_OUTPUT'
ERROR_VISUALIZATION_FILE = 'ERROR_VISUALIZATION_FILE'
ERROR_GA_SCHEDULE = 'ERROR_GA_SCHEDULE'
ERROR_GA_INVALID_SOLUTION = 'ERROR_GA_INVALID_SOLUTION'

ERROR_FAILED_EXPERIMENT = 'ERROR_FAILED_EXPERIMENT'

# these type of errors won't prevent the script from being able to parse the results
allowed_errors = [ERROR_VISUALIZATION_FILE]
# these errors will prevent the parser from parsing the results and thus the iteration should be excluded from the results
# illegal_errors = [ERROR_ILP_TIMELIMIT, ERROR_ILP_GAP, ERROR_ERROR_LOG, ERROR_RUNSIM_LOG, ERROR_OUTPUT_CPU, ERROR_MULTIPLE_WORKERS, ERROR_ILP_JSON, ERROR_ILP_SOLUTION, ERROR_ILP_ERRORS, ERROR_ILP_SOLUTION_COUNT, ERROR_ILP_SCHEDULE]
illegal_errors = [ERROR_GA_SCHEDULE, ERROR_GA_SOLUTION, ERROR_ERROR_LOG, ERROR_MULTIPLE_WORKERS, ERROR_GA_OUTPUT, ERROR_SIMULATOR_TOPOLOGY, ERROR_SIMULATOR_TOPOLOGY_OUTPUT, ERROR_FAILED_EXPERIMENT, ERROR_GA_INVALID_SOLUTION]

def invalid_solution(ga_solution):
    data = None
    # print(ga_solution)
    try:
        with open(ga_solution) as json_file:
            data = json.load(json_file)
        if data['results']['best_ind']['tput'] < 0.0:
            return True
        return False
    except:
        return True

def validate_dir(exp_dir):
    """ Validate the experiment to be really successful."""
    errors = []

    error_log = '%s/error.log' % exp_dir # should be empty
    id_pattern = '*.id.txt' # should only be one file with this id_pattern
    ga_solution = '%s/ga.json' % exp_dir # should be there
    ga_output = '%s/ga_output.txt' % exp_dir # should be there
    simulator_topology = '%s/simulator-topology.json' % exp_dir # should be there
    simulator_topology_output = '%s/simulator_output.txt' % exp_dir # should be there
    visualization_output = '%s/visualization-ga-TSCH_SLOTBONDING_50_KBPS_PHY.html' % exp_dir  # should be there
    ga_schedule = '%s/ga-schedule.json' % exp_dir  # should be there

    import os
    if not os.path.exists(error_log) or os.path.getsize(error_log) > 0:
        errors.append(ERROR_ERROR_LOG)
    if not os.path.exists(ga_solution) or os.path.getsize(ga_solution) == 0:
        errors.append(ERROR_GA_SOLUTION)
    if not os.path.exists(ga_output) or os.path.getsize(ga_output) == 0:
        errors.append(ERROR_GA_OUTPUT)
    if not os.path.exists(simulator_topology_output) or os.path.getsize(simulator_topology_output) == 0:
        errors.append(ERROR_SIMULATOR_TOPOLOGY_OUTPUT)
    if not os.path.exists(simulator_topology) or os.path.getsize(simulator_topology) == 0:
        errors.append(ERROR_SIMULATOR_TOPOLOGY)
    if not os.path.exists(visualization_output) or os.path.getsize(visualization_output) == 0:
        errors.append(ERROR_VISUALIZATION_FILE)
    if not os.path.exists(ga_schedule) or os.path.getsize(ga_schedule) == 0:
        errors.append(ERROR_GA_SCHEDULE)
    if os.path.exists(ga_solution) and invalid_solution(ga_solution):
        errors.append(ERROR_GA_INVALID_SOLUTION)
    if "failed" in exp_dir:
        errors.append(ERROR_FAILED_EXPERIMENT)

    import fnmatch
    count_workers = fnmatch.filter(os.listdir(exp_dir), id_pattern)
    if len(count_workers) > 1:
        errors.append(ERROR_MULTIPLE_WORKERS)

    return errors

# def validate_ilp(exp_data):
#     """ Validate the experiment to be really successful."""
#
#     errors = []
#
#     if exp_data['hitTimeLimit']:
#         errors.append(ERROR_ILP_TIMELIMIT)
#     if int(exp_data['nrSolutions']) < 1:
#         errors.append(ERROR_ILP_SOLUTION_COUNT)
#     if float(exp_data['MIPGap']) > float(exp_data['setMIPGap']) + 0.0001:
#         errors.append(ERROR_ILP_GAP)
#     if len(exp_data['errors']) > 0:
#         errors.append(ERROR_ILP_ERRORS)
#
#     return errors

# def disable_exp(errors_dir, errors_ilp):

def get_set_rgx(exp, rgx = ''):
    candidates = set()
    regex_result = re.search(rgx, exp, re.IGNORECASE)
    if regex_result is not None:
        candidates.add(regex_result.group(1))
    else:
        raise 'No {0} indicator in experiment dir.'.format(rgx)
    return candidates

def enable_exp(exp_dir):
    try:
        os.rename("{0}".format(exp_dir), "{0}".format(exp_dir.split("_failed")[0]))
        print("* Enabled experiment: {0}.".format(exp_dir))
        return True
    except:
        print("* Could not enable {0}.".format(exp_dir))
        return False

def disable_exp(exp_dir):
    try:
        os.rename("{0}".format(exp_dir), "{0}_failed".format(exp_dir))
        print("* Disabled experiment: {0}.".format(exp_dir))
    except:
        print("* Could not disable {0}.".format(exp_dir))

if __name__ == '__main__':
    # 'enable' enables all failed experiments again
    # 'disable' disables only those experiment that have illegal_errors
    # 'disable-all' disables all experiments with the same seed as the failed specific experiment
    # 'nothing' does nothing
    modes = ['enable', 'disable', 'disable-all', 'nothing']
    enable_disable = str(sys.argv[1])
    if enable_disable not in modes:
        raise("Enable/disable/disable-all/nothing are the only correct modes.")

    dataDirs = str(sys.argv[2]).split(',')
    print("The different data directories: {0}".format(dataDirs))
    experimentTypes = str(sys.argv[3]).split(',')
    print("The different experiment types: {0}".format(experimentTypes))

    failedSeeds = []
    data = {}
    errors_dir = {}
    failed_exp = []
    for experimentType in experimentTypes:
        data[experimentType] = {}
        errors_dir[experimentType] = {}
        for dataDir in dataDirs:
            cmd = "find {0} -ipath *ga_seed_*{1}*/error.log".format(dataDir, experimentType)
            listFiles = os.popen(cmd).read().split("\n")[:-1] # for some reason, there is a trailing whitespace in the list, remove it

            print("Processing {0} file(s) for experiment {1} in data directory {2}".format(len(listFiles), str(experimentType), dataDir))
            for datafile in listFiles:
                # get this iteration's name
                name_iteration = datafile.split('/')[-2]
                # check and save the wrong this for this iteration's directory
                errors_dir[experimentType][name_iteration] = list(validate_dir(os.path.dirname(datafile)))
                if enable_disable == 'enable' and ERROR_FAILED_EXPERIMENT in errors_dir[experimentType][name_iteration]:
                    resultEnable = enable_exp(os.path.dirname(datafile))
                    if resultEnable:
                        errors_dir[experimentType][name_iteration].remove(ERROR_FAILED_EXPERIMENT)

                # count for the average
                if ERROR_FAILED_EXPERIMENT not in errors_dir[experimentType][name_iteration]:
                    if not all(elem not in illegal_errors for elem in errors_dir[experimentType][name_iteration]):
                        failed_exp.append(datafile)

                    if len(errors_dir[experimentType][name_iteration]) > 0:
                        print("For iteration {0}, there are following directory errors: {1}".format(name_iteration, errors_dir[experimentType][name_iteration]))
                        # pass
                    else:
                        # pass
                        print("For iteration {0}, there are no errors".format(name_iteration))
                else:
                    failed_exp.append(datafile)

        # if results[experimentType]['count'] > 0:
        #     results[experimentType]['avg_gap'] /= float(results[experimentType]['count'])
        #     results[experimentType]['avg_objVal'] /= float(results[experimentType]['count'])
        #     results[experimentType]['avg_runtime'] /= float(results[experimentType]['count'])

    if enable_disable == 'disable':
        print("Executing mode \'disable\'")
        for fail_exp in failed_exp:
            disable_exp(os.path.dirname(fail_exp))
    elif enable_disable == 'disable-all':
        print("Executing mode \'disable-all\'")

        failedSeeds = []
        for fail in failed_exp:
            # print fail
            rgx = '[_\/]+%s_([A-Za-z0-9]+)_' % 'seed'
            candidates = get_set_rgx(fail, rgx)
            failedSeeds += candidates
        failedSeeds = list(set(failedSeeds)) # make unique
        print("* Failed seeds: {0}".format(failedSeeds))

        countToRename = 0
        filesToRename = []
        for dataDir in dataDirs:
            for s in failedSeeds:
                s = 'seed_%s_' % s
                cmd = "find {0} -ipath *ga_*{1}*/error.log".format(dataDir, s)
                listFiles = os.popen(cmd).read().split("\n")[:-1]
                countToRename += len(listFiles)
                filesToRename += listFiles

        for f in filesToRename:
            print(f)
            disable_exp(os.path.dirname(f))