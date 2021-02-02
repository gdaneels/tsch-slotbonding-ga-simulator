import numpy as np
import time
import math

def erange(start, end):
    ''' range function including the end in the returned range '''
    return range(start, end + 1)

def expected_nr_packets(B, ix_start_state, absorbing_states):
    row = B[ix_start_state,:] # get entire row for this starting state
    expected = 0.0 # number of expected packets
    ix = 0
    for column in row.T:
        expected += column * absorbing_states[ix][3] # index 3 is the number of arrived packets
        ix += 1
    return expected

def total_probability_with_j_packets(j, B, ix_start_state, absorbing_states):
    # print 'absorbing states:'
    number_of_slots_leftover = 0
    row = B[ix_start_state,:] # get entire row for this starting state
    expected = 0.0 # number of expected packets
    ix = 0
    for column in row.T:
        if absorbing_states[ix][3] == j:  # index 3 is the number of arrived packets
            expected += column
        ix += 1
    if expected > 0.0:
        ix = 0
        for column in row.T:
            if absorbing_states[ix][3] == j:  # index 3 is the number of arrived packets
                # print("abs. state with j: ", absorbing_states[ix], " prob: ", column)
                number_of_slots_leftover += (column/float(expected)) * absorbing_states[ix][1]
            ix += 1
    else:
        number_of_slots_leftover = float("nan")
    return expected, number_of_slots_leftover

def get_B_matrix(A, Q, S, r_max, p_succ):
    start_time = time.time()
    dim = (1 + Q) * (S + 1) * r_max * (1 + Q) # dimension of P matrix

    p_fail = 1 - p_succ

    # P = np.zeros(shape=(dim, dim), dtype=float)

    # add all transient states
    transient_states = list()
    for q in erange(1, Q):
        for f in erange(1, S):
            for r in erange(1, r_max):
                for t in erange(0, Q):
                    transient_states.append((q, f, r, t))

    # number of transient states
    len_transient = len(transient_states)

    absorbing_states = list()
    # add the first class of absorbing states (0, f_s, r_s, t_s)
    for f in erange(1, S):
        for r in erange(1, r_max):
            for t in erange(0, Q):
                absorbing_states.append((0, f, r, t))

    # add the second class of absorbing states (q_s, 0, r_s, t_s)
    for q in erange(1, Q):
        for r in erange(1, r_max):
            for t in erange(0, Q):
                absorbing_states.append((q, 0, r, t))

    # add the absorbing states where both q_s and f_s are 0 (belong to both absorbing classes above)
    for r in erange(1, r_max):
        for t in erange(0, Q):
            absorbing_states.append((0, 0, r, t))

    len_absorbing = len(absorbing_states)
    states = transient_states + absorbing_states
    len_states = len(states)

    import pandas as pd
    df = pd.DataFrame(np.zeros(shape=(dim, dim), dtype=float), index=states, columns=states)
    # print(df.index.dtype)
    # fill in the P matrix
    # print(df)
    # exit()
    for q in erange(1, Q):
        for f in erange(1, S):
            for r in erange(2, r_max):
                for t in erange(0, Q - 1):
                    # print(df.loc[(q, f, r, t), (q - 1, f - 1, r_max, t + 1)])
                    # # exit()
                    # print('come here')
                    # print(p_succ)
                    # print((q, f, r, t))
                    # print((q - 1, f - 1, r_max, t + 1))
                    df.loc[(q, f, r, t), (q - 1, f - 1, r_max, t + 1)] = p_succ
                    # print(df)
                    # exit()
                    df.loc[(q, f, r, t), (q, f - 1, r - 1, t)] = p_fail
            # for t in erange(0, Q - 1):
            for t in erange(0, Q - 1): # todo Should this extra loop be here?
                df.loc[(q, f, 1, t), (q - 1, f - 1, r_max, t + 1)] = p_succ
                df.loc[(q, f, 1, t), (q - 1, f - 1, r_max, t)] = p_fail
    #
    # for f in erange(0, S):
    #     for r in erange(1, r_max):
    #         for t in erange(0, Q-1):
    #             print("(q = 0, f = {f}, r = {r}, t = {t}) -> (q = 0, f = {f2}, r = {r2}, t = {t2}) = 1.0".format(f=f, r=r, t=t, f2=f, r2=r, t2=t))
    #             df.loc[(0, f, r, t), (0, f, r, t)] = 1.0
    #
    for f in erange(1, S):
        for r in erange(1, r_max):
            for t in erange(0, Q-1):
                # print("(q = 0, f = {f}, r = {r}, t = {t}) -> (q = 0, f = {f2}, r = {r2}, t = {t2}) = 1.0".format(f=f, r=r, t=t, f2=f, r2=r, t2=t))
                df.loc[(0, f, r, t), (0, f, r, t)] = 1.0

    # exit()

    # for q in erange(0, Q):
    #     for r in erange(1, r_max):
    #         for t in erange(0, Q-1):
    #             print("(q = {q}, f = 0, r = {r}, t = {t}) -> (q = {q2}, f = 0, r = {r2}, t = {t2}) = 1.0".format(q=q, q2=q, r=r, t=t, f2=f, r2=r, t2=t))
    #             df.loc[(q, 0, r, t), (q, 0, r, t)] = 1.0

    for q in erange(0, Q):
        for r in erange(1, r_max):
            for t in erange(0, Q-1):
                # print("(q = {q}, f = {f}, r = {r}, t = {t}) -> (q = {q2}, f = {f2}, r = {r2}, t = {t2}) = 1.0".format(f=0, r=r, t=t, f2=0, r2=r, t2=t, q=q, q2=q))
                df.loc[(q, 0, r, t), (q, 0, r, t)] = 1.0

    # print(df)
    # exit()

    from sys import getsizeof
    matrix = df.values
    del df
    T = matrix[0:len_transient, 0:len_transient]
    R = matrix[0:len_transient, len_transient:len_states]
    del matrix
    # print("Matrix bytes: {0}".format(matrix.nbytes))

    # P = [ T R ; 0 I ]
    # T = matrix[0:len_transient, 0:len_transient]
    # print("T bytes: {0}".format(T.nbytes))
    # R = matrix[0:len_transient, len_transient:len_states]
    # print("R bytes: {0}".format(R.nbytes))
    # I_test = matrix[len_transient:len_states, len_transient:len_states]
    # print("I bytes: {0}".format(I_test.nbytes))
    # check if this is an identity matrix

    # assert (I_test.shape[0] == I_test.shape[1]) and np.allclose(I_test, np.eye(I_test.shape[0]))
    elapsed_time = time.time() - start_time
    # print 'Calculating everthing before B took: {0}'.format(elapsed_time)

    start_time = time.time()
    B = np.matmul(np.linalg.inv(np.subtract(np.eye(len_transient), T)), R)
    del T
    del R

    # print(B)

    elapsed_time = time.time() - start_time
    # print 'Calculating B took: {0}'.format(elapsed_time)

    # dict_q_s_to_packets = {}
    # dict_q_s_to_packets[(0, 0, 1)] = total_probability_with_j_packets(0, B, ix_start_state=transient_states.index((0, 1, r_max, 0)), absorbing_states=absorbing_states)

    dict_q_s_to_packets = {}
    for j in erange(0, A):
        for q in erange(0, Q):
            for s in erange(0, S):
                # TODO has to be included in the model?
                if j == 0 and (q == 0 or s == 0):
                    if q == 0 and s > 0:
                        # if there are no packets queued, no slots will be used
                        dict_q_s_to_packets[(j, q, s)] = (1.0, float(s))
                    elif q > 0 and s == 0:
                        dict_q_s_to_packets[(j, q, s)] = (1.0, 0.0)
                    elif q == 0 and s == 0:
                        dict_q_s_to_packets[(j, q, s)] = (1.0, 0.0)
                    else:
                        raise BaseException("Should not happen, all cases should be covered.")
                elif j > 0 and (q == 0 or s == 0):
                    # set the number of slots left over to NaN, because if there is no probability of ending in these states, slots being left over is out of the question
                    dict_q_s_to_packets[(j, q, s)] = (0.0, float("nan"))
                else:
                    total_probability, number_of_slots_left_over = total_probability_with_j_packets(j, B, ix_start_state=transient_states.index((q, s, r_max, 0)), absorbing_states=absorbing_states)
                    # if (q, s, r_max, 0) == (1, 2, 2, 0) and j == 1:
                        # print (q, s, r_max, 0)
                        # print j
                        # print total_probability
                        # print number_of_slots_left_over
                        # exit()
                    if number_of_slots_left_over > s:
                        raise BaseException("This should DEFINITELY not happen: number_of_slots_left_over = {0} while s = {1}", number_of_slots_left_over, s)
                    dict_q_s_to_packets[(j, q, s)] = (total_probability, number_of_slots_left_over)

    # print(A, Q, S)
    # dict_q_s_to_packets[(A, Q, S)] = total_probability_with_j_packets(A, B, ix_start_state=transient_states.index((Q, S, r_max, 0)), absorbing_states=absorbing_states)
    return dict_q_s_to_packets

def main(A, Q, S, r_max, p_succ):
    # Q = 9 # 10 packets in the queue, 0 -> 9
    # S = 40 # number of slots in a slot frame
    # r_max = 3 # maximum number of MAC retransmissions per packet
    # p_succ = 0.5

    return get_B_matrix(A=A, Q=Q, S=S, r_max=r_max, p_succ=p_succ)

    #
    # start_time = time.time()
    # dict_q_s_to_packets = {}
    # for q in erange(1, Q):
    #     for s in erange(1, S):
    #         dict_q_s_to_packets[(q, s)] = expected_nr_packets(B, ix_start_state=transient_states.index((q, s, 2, 0)), absorbing_states=absorbing_states)
    # elapsed_time = time.time() - start_time
    # print 'Calculating expected number of packets: {0}'.format(elapsed_time)
    #
    # start_time = time.time()
    # test = 0.0
    # for q in erange(1, Q):
    #     for s in erange(1, S):
    #         test += dict_q_s_to_packets[(q, s)]
    # elapsed_time = time.time() - start_time
    # print 'Calculating expected number of packets: {0}'.format(elapsed_time)
    # # print P


if __name__ == "__main__":
    import sys
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--arrived', '-a', type=int)
    parser.add_argument('--queued', '-q', type=int)
    parser.add_argument('--slots', '-s', type=int)
    parser.add_argument('--r_max', '-r', type=int)
    parser.add_argument('--p_succ', '-p', type=float)
    parser.add_argument('--prettify', '-y', type=bool, default=False)

    args = parser.parse_args()
    A = int(args.arrived)
    Q = int(args.queued)
    S = int(args.slots)
    r_max = int(args.r_max)
    p_succ = float(args.p_succ) / 1000.0
    prettify = bool(args.prettify)
    d = main(A=A, Q=Q, S=S, r_max=r_max, p_succ=p_succ)
    output = ''
    for key, value in d.items():
        if output != '':
            output += '\r\n'
        if not prettify:
            output += '{0},{1},{2},{3},{4}'.format(key[0], key[1], key[2], value[0], value[1])
        else:
            output += 'arrived = {0}, queued = {1}, slots = {2}, r_max = {3}, p_succ = {4}, total probability = {5}, slots left over = {6}'.format(key[0], key[1], key[2], r_max, p_succ, value[0], value[1])
    print(output)
