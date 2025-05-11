#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
import pandas as pd
from dla_fl.simulation import SimulationRunner
from dla_fl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description='Run DLA-AI sim and export results')
    p.add_argument('-c', '--config', default='configs/config.yaml')
    p.add_argument('-o', '--output_dir', default='results/')
    p.add_argument('-r', '--rounds', type=int, help='Override rounds')
    return p.parse_args()


def save_results(out, met, hist, ts, config_name):
    os.makedirs(out, exist_ok=True)
    # Flatten metrics and history for DataFrame
    df_rows = []
    # Assume history is a list of dicts, metrics is a dict
    for h in hist:
        row = {}
        row.update(h)
        row.update(met)
        row['timestamp'] = ts
        row['config_name'] = config_name
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    df.to_csv(f'{out}/results_{ts}.csv', index=False)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.rounds:
        cfg['simulation']['rounds'] = args.rounds
    metrics, history = SimulationRunner(cfg).run()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_name = os.path.basename(args.config)
    save_results(args.output_dir, metrics, history, ts, config_name)
    print(f'Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()
