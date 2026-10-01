#!/bin/zsh
# Clean re-measurement of reported timings; run alone on an idle machine.
cd "$(dirname "$0")"
LOG=run_timing_clean.log
rm -f timing_clean_mhmg.csv timing_clean_xu.csv
: > $LOG
step() { echo "\n##### $(date '+%F %T') START $*" >> $LOG; "$@" >> $LOG 2>&1; echo "##### $(date '+%F %T') END rc=$? $*" >> $LOG; }
step python3 timing_clean.py --method mhmg --only D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D12
step python3 timing_clean.py --method mhmg --only D11 --reuse-init
step python3 timing_clean.py --method xu
step python3 timing_clean.py --method mhmg --stream-only
step python3 timing_clean.py --method mhmg --eval-only
echo "\n##### ALL DONE $(date '+%F %T')" >> $LOG
