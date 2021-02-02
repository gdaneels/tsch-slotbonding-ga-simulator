#!/bin/bash -l

A=3
Q=3
S=3
R_MAX=3

OUTPUT_DIR="output_A_${A}_Q_${Q}_S_${S}_RMAX_${R_MAX}"
mkdir -p $OUTPUT_DIR # make if not exists

for P_SUCC in $(seq 1000 1 1000)
do
    echo "python mc-probability.py -a ${A} -q ${Q} -s ${S} -r ${R_MAX} -p ${P_SUCC} > \"${OUTPUT_DIR}/p_${P_SUCC}.csv\"" >> jobs.txt
done
parallel --jobs 2 < jobs.txt