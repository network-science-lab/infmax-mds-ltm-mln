#! /bin/zsh
set -euo pipefail

data_stub="data/raw_results"

for file in $(dvc list . "$data_stub" --recursive | grep -E '\.csv$|\.yaml$' | awk '{print $1}'); do
    dvc pull "$data_stub"/"$file"
done
