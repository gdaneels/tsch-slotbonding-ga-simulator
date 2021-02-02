#!/bin/bash -l

A=11
Q=11
S=11
R_MAX=3

OUTPUT_DIR="output_A_${A}_Q_${Q}_S_${S}_RMAX_${R_MAX}"
mkdir -p $OUTPUT_DIR # make if not exists

for P_SUCC in $(seq 0 500 1000)
do
    sem -j+0 python mc-probability.py -a ${A} -q ${Q} -s ${S} -r ${R_MAX} -p ${P_SUCC} > "${OUTPUT_DIR}/p_${P_SUCC}.csv"
done
sem --wait