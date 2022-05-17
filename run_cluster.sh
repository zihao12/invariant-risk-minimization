#!/bin/bash

for method in erm irm
do
    maxiter=50
    if [ "$method" = "irm" ]
    then
        let maxiter=100
    fi
    for i in {1..100}
    do
        out_result="./output/${method}_${i}.pkl"
        out_model="./output/${method}_${i}.pt"
        out_print="./output/${method}_${i}.out"
        sbatch run_script.sbatch  $i $method $maxiter $out_result $out_model $out_print
    done
done