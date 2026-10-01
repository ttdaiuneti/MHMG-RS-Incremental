#!/bin/zsh
# v3 full re-run after the dataset-loading fix (datasets_v3). Sequential on purpose:
# timings are reported, so jobs must not compete for CPU.
cd "$(dirname "$0")"
LOG=run_v3_all.log
step() { echo "\n##### $(date '+%F %T') START $*" >> $LOG; "$@" >> $LOG 2>&1; echo "##### $(date '+%F %T') END rc=$? $*" >> $LOG; }
: > $LOG
step python3 e1_pipeline.py --only D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D12 --prefix stageA_
step python3 e1_pipeline.py --only D11 --prefix stageB_
step python3 baseline_xu2025b/run_baseline_comparison.py
step python3 baseline_xu2025b/measure_ce_reevaluation_cost.py
step python3 baseline_li2026/run_baseline_li2026.py --exclude D11
step python3 baseline_deng2026/run_baseline_deng2026.py --exclude D11 --t 2
step python3 baseline_deng2026/run_baseline_deng2026.py --exclude D11 --t 1 --out baseline_deng2026_t1_results.csv
step python3 baseline_deng2026/run_baseline_deng2026.py --exclude D11 --t 3 --out baseline_deng2026_t3_results.csv
step python3 e0_measurement2_acc_gap_trajectory.py
step python3 e0_measurement6_gradual_new_classes.py
step python3 e0_measurement9_li2026_protocol_pilot.py
step python3 test_rounding_confound_musk_urban.py
step python3 ../theory/e0_incremental_energy_check.py
step python3 ../theory/e0_batch_insertion_check.py
step python3 ../theory/e0_batch_size_sweep.py
step python3 ../theory/e0_new_class_synthetic_test.py
echo "\n##### ALL DONE $(date '+%F %T')" >> $LOG
