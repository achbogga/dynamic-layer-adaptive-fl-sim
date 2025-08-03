#!/usr/bin/env bash
set -euo pipefail

# List of config files (must live under configs/)
configs=(
  config.yaml
  partition_equal.yaml
  partition_3_2.yaml
  devices_5.yaml
  devices_10.yaml
  threshold_01.yaml
  threshold_05.yaml
)

# Base output folder
OUT_BASE="results"

echo "Starting batch of DLA‑AI experiments..."

for cfg in "${configs[@]}"; do
  name="${cfg%.yaml}"                     # strip .yaml for directory name
  out_dir="$OUT_BASE"                     # save all results in results/
  echo -e "\n>>> Experiment: $name"
  echo "Config: configs/$cfg"
  echo "Output: $out_dir"
  mkdir -p "$out_dir"
  # Run simulation; remove -r if you want to use config's own rounds
  uv run run_sim.py -c "configs/$cfg" -o "$out_dir"
done

echo -e "\n>>> Real-data experiment: MNIST"
uv run src/dla_fl/end_to_end_mnist.py

echo -e "\nAll experiments complete. Results are under $OUT_BASE/."
