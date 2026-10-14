#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/path/to/AMASS}"
OUT_DIR="${OUT_DIR:-dataset/limmt_no_foot_compliance_full}"
ALLOWLIST="${ALLOWLIST:-scripts/data_process/allowlist_limmt_no_foot_compliance.json}"
PYTHON="${PYTHON:-python}"

if [[ "$DATASET_ROOT" == "/path/to/AMASS" ]]; then
  echo "error: set DATASET_ROOT to your G1-retargeted AMASS directory (the level containing ACCAD/, CMU/, ...)," >&2
  echo "       e.g. DATASET_ROOT=/data/amass_g1_retargeted $0" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "error: DATASET_ROOT=$DATASET_ROOT is not a directory" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/data_process/generate_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --allowlist "$ALLOWLIST" \
  --mem-path "$OUT_DIR"
