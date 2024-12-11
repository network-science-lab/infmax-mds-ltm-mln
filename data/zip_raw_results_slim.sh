#! /bin/zsh
set -euo pipefail

find data/raw_results -type f \( -name "*.csv" -o -name "*.yaml" \) ! -path "*.zip*" | tar -cvzf raw_results.tar.gz -T -
