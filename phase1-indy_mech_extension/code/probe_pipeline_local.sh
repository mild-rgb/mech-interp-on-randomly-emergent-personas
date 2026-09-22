#!/bin/bash
cd "/home/me/Documents/eleuther-soar/mech interp on randomly emergent personas/phase1-indy_mech_extension"
echo "=== STAGE 1 massmean grid $(date)"
python3 -u train_probes.py --probes massmean --n_perm 10 --jobs 4 --out results_massmean.json
echo "=== STAGE 2 trigger probes $(date)"
python3 -u train_trigger_probes.py --probes massmean,logistic --C 0.1 --jobs 4 --out results_trigger.json --dirs_out trigger_directions.npz
echo "=== STAGE 3 logistic $(date)"
python3 -u train_probes.py --probes massmean,logistic --fixedC 0.1 --layers 8,16,20,24,32 --n_perm 0 --jobs 4 --configs assistant_A,broken_A,assistant_A_fluentonly,broken_A_notassistant --out results_logistic.json
echo "=== STAGE 4 phase19 scoring $(date)"
while [ ! -f phase19/DOWNLOAD_DONE ]; do sleep 20; done
python3 -u apply_probes_p19.py --results results_massmean.json --p19dir phase19 --out p19_probe_scores.json
echo "=== ALL DONE $(date)"
