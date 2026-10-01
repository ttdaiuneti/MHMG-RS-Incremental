# Incremental MHMG-RS under insertion-only streams: code and data

Code, datasets and raw results for the manuscript

> T.D. Tran, *Exact Incremental Maintenance of Maximal Homogeneous Metric
> Granules for Neighborhood Rough Set Attribute Reduction*, submitted to Knowledge-Based
> Systems.

The batch model being maintained is MHMG-RS:

> N.T.T. Tram, N.L. Giang, N.M. Hung, T.D. Tran, *Attribute Reduction via
> Knowledge Energy and Maximal Homogeneous Metric Granules in Neighborhood
> Rough Sets*, International Journal of Approximate Reasoning, 2026.
> doi:[10.1016/j.ijar.2026.109787](https://doi.org/10.1016/j.ijar.2026.109787)
> (code: <https://github.com/ttdaiuneti/MHMG-RS>).

## Layout

The scripts locate the batch reducer and the original data through the
relative path `../IJAR-MHMG-Published/`, so the two folders must stay side by side.

```
IJAR-MHMG-Published/IJAR-Code/
    HMMG_Reducer.py            batch MHMG-RS reducer, unchanged from the published code
    12-datasets/               benchmark files as distributed with the published code
                               (input of datasets_v3/build_datasets_v3.py only; see "Data")
MHMG-RS-Incremental/
    datasets_v3/               the twelve datasets used here, all objects, with header row
    experiments/               incremental pipeline, baselines, checks, raw result CSVs, run logs
        e1_pipeline.py         main experiment: batch references + incremental stream
        multi_seed_staleness.py  staleness over further random 80/20 splits
        timing_clean.py        timing re-measurement on an otherwise idle machine
        baseline_xu2025b/      fixed-radius local NRS + composite entropy
        baseline_li2026/       fuzzy neighborhood decision system
        baseline_deng2026/     overlap-function fuzzy rough sets
        test_incremental_property.py  randomized check of every insertion against recomputation
        check_reducer_distance.py     distances/gamma of the batch reducer vs. exact definition
    theory/                    numerical checks of Theorems 1-5 and the batch-size sweep
    paper/
        make_results_tex.py    writes every number and table of Section 5 (paper/generated/)
        generate_*.py          figures
```

## Requirements

Python 3.9 with the packages in `requirements.txt` (versions used for the
reported results):

```
pip install -r requirements.txt
```

## Reproducing the results

Commands are run from `MHMG-RS-Incremental/experiments/` unless noted.
The main split uses seed 42 (80% initial objects, 20% stream); accuracy is
5-fold stratified CV with 3-NN and SVM-RBF (scikit-learn defaults) on the
selected attributes.

**Regenerate the manuscript's numbers and tables from the shipped CSVs**
(seconds, no experiment is re-run):

```
cd ../paper && python make_results_tex.py
```

This writes `paper/generated/numbers.tex` (one LaTeX macro per number quoted in
Section 5, the abstract and the conclusion) and `paper/generated/tab_*.tex`;
the manuscript includes these files, so no number in it is typed by hand. All
statistical tests are computed from unrounded accuracies in the per-method
result files.

**Re-run the experiments.** `e1_pipeline.py`, `multi_seed_staleness.py` and
`timing_clean.py` append to their output files (so that interrupted runs can be
resumed); delete their outputs before a fresh run. The other scripts overwrite
their outputs.

| Step | Command | Output |
|---|---|---|
| All single-split experiments, baselines and theorem checks, in sequence | `zsh run_v3_all.sh` | files below, log `run_v3_all.log` |
| Main experiment (exactness, speedup, staleness at seed 42) | `python e1_pipeline.py --only D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D12 --prefix stageA_` and `python e1_pipeline.py --only D11 --prefix stageB_` | `stage{A,B}_*.csv` |
| Baselines | `python baseline_xu2025b/run_baseline_comparison.py`, `python baseline_xu2025b/measure_ce_reevaluation_cost.py`, `python baseline_li2026/run_baseline_li2026.py --exclude D11`, `python baseline_deng2026/run_baseline_deng2026.py --exclude D11 --t 2` (and `--t 1`, `--t 3` with `--out`) | `baseline_*_results.csv`, `ce_reevaluation_cost.csv` |
| Merged comparison tables | `python build_comparison_tables.py` | `comparison_accuracy_size.csv`, `deng2026_t_sweep.csv`, `total_reduction_time_*.csv`, `baseline_comparison_4way.csv` |
| Staleness over further splits (seeds 0-8; D11 excluded) | `python multi_seed_staleness.py --seeds 0 1 2 --only D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D12 --out multi_seed_w1.csv` (likewise seeds 3-5 and 6-8 into `_w2`, `_w3`) | `multi_seed_w{1,2,3}.csv` |
| Timings reported in the manuscript, on an otherwise idle machine | `zsh run_timing_clean.sh` | `timing_clean_{mhmg,xu}.csv`, `timing_stream_mhmg.csv`, `timing_eval_mhmg.csv`, log `run_timing_clean.log` |
| Batch-reducer distance check | `python check_reducer_distance.py` | `reducer_distance_check.csv` |
| Randomized property test of the updates | `python test_incremental_property.py`; `python baseline_deng2026/test_vec_equivalence.py` | printed |
| Figures | `python generate_figures.py` (speedup figure, into `experiments/figs/`), then in `../paper/`: `python generate_analysis_figures.py`, `python generate_synthesis_figures.py`, `python generate_baseline_pareto.py`, `python generate_mechanism_figure.py` | `figs/` |

`generate_figures.py` and `generate_synthesis_figures.py` read
`speedup_final.csv` and `total_reduction_time_comparison_final.csv`, which
`make_results_tex.py` writes; run it first.

`experiments/_archive_v2_pre_header_fix/stageA_accuracy_checkpoints.csv` is
kept from the earlier run on the files that omitted one object per dataset;
`make_results_tex.py` reads it only for the comparison with that run quoted in
the Discussion.

**Timings.** Wall-clock times are measured on an Apple M4 Pro (14 cores,
48 GB), Python 3.9.6. The main run (`run_v3_all.sh`) shared the machine with
other jobs, so the timings in the manuscript come from `run_timing_clean.sh`,
run alone: one batch search on 80% and one on 100% of the data, and the stream
repeated five times (median reported). For MHMG-RS the stream and state
initialization are measured in a process that runs no batch search
(`timing_clean.py --stream-only`, output `timing_stream_mhmg.csv`). The batch evaluation of
gamma(B_init) on all objects, the reference for the per-insertion ratio, is
measured the same way (`timing_clean.py --eval-only`, output `timing_eval_mhmg.csv`). `make_results_tex.py` checks that the
reducts selected in this re-measurement equal those of the main run.
Absolute times will differ on other hardware.

## Data

The twelve datasets are from the UCI Machine Learning Repository
(<https://archive.ics.uci.edu>): numerical condition attributes,
unnormalized, followed by the decision attribute in the last column.
Min–max normalization is applied in code, once per dataset over all objects,
before the initial/stream split.

`datasets_v3/` is built by `datasets_v3/build_datasets_v3.py`. The files
distributed with the published MHMG-RS code have no header row but were read
there with pandas' default `header=0`, which drops the first object of every
dataset; the script re-reads them with `header=None`. Files 4 (Dermatology)
and 8 (Arrhythmia), which contain missing values, are rebuilt from the
original UCI files with `KNNImputer(n_neighbors=5)` on the attributes over all
objects, the imputation used for the published files.

| File | Dataset | n | d | classes |
|---|---|---|---|---|
| 1.csv | Wine | 178 | 13 | 3 |
| 2.csv | Breast Cancer Wisconsin (Diagnostic), WDBC | 569 | 30 | 2 |
| 3.csv | Ionosphere | 351 | 34 | 2 |
| 4.csv | Dermatology | 366 | 34 | 6 |
| 5.csv | Sonar (Connectionist Bench, Mines vs. Rocks) | 208 | 60 | 2 |
| 6.csv | Urban Land Cover | 675 | 147 | 9 |
| 7.csv | Musk (Version 1) | 476 | 166 | 2 |
| 8.csv | Arrhythmia | 452 | 279 | 13 |
| 9.csv | Spambase | 4601 | 57 | 2 |
| 10.csv | Parkinsons | 195 | 22 | 2 |
| 11.csv | ISOLET | 7797 | 617 | 26 |
| 12.csv | Letter Recognition | 20000 | 16 | 26 |

UCI datasets are distributed under their repository licences (most under
CC BY 4.0); please cite the original donors when reusing them.

## Batch reducer

`HMMG_Reducer.py` is used unchanged. Two of its properties are described in
the manuscript (Section 5.1) and quantified by `check_reducer_distance.py`:
its forward search compares the dependency rounded to two decimals and stops
when the best candidate does not increase it; and it computes distances with the Gram identity, which differs
from direct computation by floating-point error. The incremental updates and
all exactness checks use direct distances.

## Licence

Code: MIT (see `LICENSE`). Result CSVs: CC BY 4.0.

## Contact

Thanh Dai Tran, University of Economics and Technology for Industries (UNETI),
Hanoi, Vietnam. ttdaiuneti@gmail.com, ORCID
[0009-0003-4527-9754](https://orcid.org/0009-0003-4527-9754).
