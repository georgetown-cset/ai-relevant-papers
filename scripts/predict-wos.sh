#!/usr/bin/env bash
# Run a deployed model on full arXiv data

set -euo pipefail

model_name=$1
# Note expected location of test data (below we confirm it exists)
data_path="$GCS_DATA_PREFIX/wos/cleaned-wos-outputs-for-beam-20200205-*"
now=$(date +"%Y%m%d-%H%M%S")

# dataflow job names must be in [-a-z0-9]
job_name=wos-prediction-scibert-"$model_name"-"$now"
job_name=${job_name//_/-}
job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')

# Note expected location of the deployed model
model_prefix=$GCS_MODEL_PREFIX
output_path="$GCS_PREDICTION_PREFIX/$model_name/test/$job_name"

if ! gsutil -q stat "$model_prefix"/"$model_name"/weights.th; then
  echo Did not find a model "$model_name" at "$model_prefix"
  exit 1
fi

python3 predict_wos.py \
  "$data_path" \
  "$model_name" \
  "$output_path" \
  --project "$GCP_PROJECT" \
  --runner DataflowRunner \
  --setup_file ./setup.py \
  --disk_size_gb 50 \
  --job_name "$job_name" \
  --save_main_session \
  --region us-east1 \
  --temp_location "$GCS_PREDICTION_PREFIX"/tmp-"$job_name" \
  --machine_type n1-highmem-2

