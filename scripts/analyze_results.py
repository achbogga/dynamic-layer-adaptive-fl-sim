#!/usr/bin/env python3
"""Summarize simulation results and plot variability."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Analyze simulation CSV outputs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-table", default="docs/sim_summary.csv")
    p.add_argument("--out-plot", default="docs/images/sim_variability.png")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        cfg = os.path.splitext(os.path.basename(df["config_name"][0]))[0]
        total_rounds = int(df["total_rounds"].iloc[0])
        final_avg = df["final_avg_param"].iloc[0]
        mean_abs = (df["avg_param"] - final_avg).abs().mean()
        std = df["avg_param"].std()
        rows.append({
            "Scenario": cfg,
            "Total Rounds": total_rounds,
            "Final Avg Param": final_avg,
            "Mean |Δ|": mean_abs,
            "Std Param": std,
        })
    summary = pd.DataFrame(rows).sort_values("Scenario")
    summary.to_csv(args.out_table, index=False)

    summary.plot(x="Scenario", y="Std Param", kind="bar", legend=False)
    plt.ylabel("Std of Param Mean")
    plt.tight_layout()
    plt.savefig(args.out_plot)
    print(summary)


if __name__ == "__main__":
    main()

