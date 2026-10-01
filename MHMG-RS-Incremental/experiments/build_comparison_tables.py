"""Build the merged comparison tables from the raw per-method outputs.

Replaces the earlier inline merging scripts. All values keep full precision;
rounding happens only when a table is printed in the manuscript.

Inputs (this directory):
  stage{A,B}_batch_summary.csv, stage{A,B}_accuracy_checkpoints.csv  (MHMG-RS)
  baseline_xu2025b_results.csv, baseline_li2026_results.csv,
  baseline_deng2026_results.csv (t=2), baseline_deng2026_t{1,3}_results.csv
Outputs:
  comparison_accuracy_size.csv      one row per dataset, reduct sizes and B_full
                                    accuracies (percent) of the four methods
  baseline_comparison_4way.csv      the same sizes/accuracies, columns read by figure scripts
  total_reduction_time_detail.csv   MHMG-RS vs Xu2025b total pipeline time, per component
  total_reduction_time_comparison.csv  the totals, columns read by figure scripts
  deng2026_t_sweep.csv              Deng2026 reduct size / accuracy for t=1,2,3
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def csv(name):
    return pd.read_csv(os.path.join(HERE, name))


def mhmg_table():
    summ = pd.concat([csv(f"stage{s}_batch_summary.csv") for s in "AB"])
    acc = pd.concat([csv(f"stage{s}_accuracy_checkpoints.csv") for s in "AB"])
    full = acc[(acc.stream_type == "iid") & (acc.reduct_used == "B_full")]
    full = full.drop_duplicates("dataset_name", keep="last").set_index("dataset_name")
    summ = summ.drop_duplicates("dataset_name", keep="last").set_index("dataset_name")
    return pd.DataFrame({
        "n": summ.n_total, "p": summ.n_features,
        "ours_size": summ.reduct_full_size,
        "ours_3nn": 100 * full.acc_3nn_mean, "ours_svm": 100 * full.acc_svm_mean,
        "ours_batch_init_sec": summ.time_batch_init_sec,
        "ours_state_init_sec": summ.time_state_init_sec,
        "ours_stream_sec": summ.time_stream_update_iid_sec,
    })


def baseline(name, prefix):
    b = csv(name).set_index("dataset_name")
    return pd.DataFrame({
        f"{prefix}_size": b.reduct_full_size,
        f"{prefix}_3nn": 100 * b.acc_3nn_full, f"{prefix}_svm": 100 * b.acc_svm_full,
    })


def main():
    ours = mhmg_table()
    xu = csv("baseline_xu2025b_results.csv").set_index("dataset_name")
    table = ours.join(baseline("baseline_xu2025b_results.csv", "xu"), how="inner")
    table = table.join(baseline("baseline_li2026_results.csv", "li"), how="inner")
    table = table.join(baseline("baseline_deng2026_results.csv", "deng"), how="inner")
    table.to_csv(os.path.join(HERE, "comparison_accuracy_size.csv"))
    # same content under the file name and columns the figure scripts read
    four = table[[c for c in table.columns if c.split("_")[-1] in ("size", "3nn", "svm")]]
    four.rename_axis("dataset").to_csv(os.path.join(HERE, "baseline_comparison_4way.csv"))

    tt = ours.join(xu[["time_batch_init_sec", "time_state_init_sec", "stream_time_sec"]]
                   .add_prefix("xu_"), how="inner")
    tt["ours_total_sec"] = tt.ours_batch_init_sec + tt.ours_state_init_sec + tt.ours_stream_sec
    tt["xu_total_sec"] = tt.xu_time_batch_init_sec + tt.xu_time_state_init_sec + tt.xu_stream_time_sec
    tt["xu_over_ours"] = tt.xu_total_sec / tt.ours_total_sec
    tt = tt.sort_values("p")
    tt.to_csv(os.path.join(HERE, "total_reduction_time_detail.csv"))
    pd.DataFrame({"dataset_name": tt.index, "n_total": tt.n.values, "n_features": tt.p.values,
                  "mhmg_total_sec": tt.ours_total_sec.values, "xu_total_sec": tt.xu_total_sec.values,
                  "mhmg_faster_by": tt.xu_over_ours.values}).to_csv(
        os.path.join(HERE, "total_reduction_time_comparison.csv"), index=False)

    rows = []
    for t, f in [(1, "baseline_deng2026_t1_results.csv"), (2, "baseline_deng2026_results.csv"),
                 (3, "baseline_deng2026_t3_results.csv")]:
        if os.path.exists(os.path.join(HERE, f)):
            d = csv(f).set_index("dataset_name")
            for ds, r in d.iterrows():
                rows.append({"t": t, "dataset_name": ds, "size": r.reduct_full_size,
                             "acc_3nn": 100 * r.acc_3nn_full, "acc_svm": 100 * r.acc_svm_full})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "deng2026_t_sweep.csv"), index=False)
    print("written: comparison_accuracy_size.csv, total_reduction_time_comparison.csv, deng2026_t_sweep.csv")


if __name__ == "__main__":
    main()
