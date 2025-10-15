#!/usr/bin/env bash
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end acceptance-rate profiling pipeline.
# 1. Sample prompts from benchmarks (train/test splits).
# 2. Run drafter→verifier profiling on training prompts.
# 3. Train VIDUR-style acceptance regressors.
# 4. Profile held-out prompts and evaluate the regressor.
#
# Adjust dataset sizes/columns or model identifiers below as needed.

set -euo pipefail

# Force unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CONDA_RUN="${HOME}/miniconda3/bin/conda run -n llama2spec"

# For interactive commands with real-time output, source conda directly
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="llama2spec"

PROMPT_DIR="${PROJECT_ROOT}/prompts"
RESULTS_DIR="${PROJECT_ROOT}/results"
ACCEPTANCE_DIR="${PROJECT_ROOT}/acceptance"
mkdir -p "${PROMPT_DIR}" "${RESULTS_DIR}" "${ACCEPTANCE_DIR}"

SPEC_TOKENS=4
MAX_TOKENS=160
MAX_PROMPT_TOKENS=128
DRAFTER_MODEL="meta-llama/Llama-3.2-1B-Instruct"
VERIFIER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
RUN_LABEL=""

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --drafter-model MODEL       Drafter (speculative) model identifier
  --verifier-model MODEL      Verifier (target) model identifier
  --spec-tokens N             Number of speculative tokens per iteration (default: ${SPEC_TOKENS})
  --max-tokens N              Maximum tokens to generate per prompt (default: ${MAX_TOKENS})
  --max-prompt-tokens N       Maximum prompt tokens (default: ${MAX_PROMPT_TOKENS})
  --run-label LABEL           Optional label for output directories/files
  -h, --help                  Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drafter-model)
      DRAFTER_MODEL="$2"
      shift 2
      ;;
    --verifier-model)
      VERIFIER_MODEL="$2"
      shift 2
      ;;
    --spec-tokens)
      SPEC_TOKENS="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --max-prompt-tokens)
      MAX_PROMPT_TOKENS="$2"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

sanitize_label() {
  echo "$1" | sed 's|.*/||' | sed 's/[^A-Za-z0-9._-]/_/g'
}

if [[ -n "${RUN_LABEL}" ]]; then
  MODEL_TAG="${RUN_LABEL}"
else
  D_TAG=$(sanitize_label "${DRAFTER_MODEL}")
  V_TAG=$(sanitize_label "${VERIFIER_MODEL}")
  MODEL_TAG="${D_TAG}_vs_${V_TAG}"
fi

RUN_RESULTS_DIR="${RESULTS_DIR}/${MODEL_TAG}"
mkdir -p "${RUN_RESULTS_DIR}"

RUN_ACCEPTANCE_MODEL="${ACCEPTANCE_DIR}/${MODEL_TAG}.joblib"
TRAIN_DETAILS_COMBINED="${RUN_RESULTS_DIR}/train_details.jsonl"
TEST_DETAILS_COMBINED="${RUN_RESULTS_DIR}/test_details.jsonl"
TEST_METRICS_JSON="${RUN_RESULTS_DIR}/test_regressor_metrics.json"

echo ">>> Configuration:"
echo "    Drafter : ${DRAFTER_MODEL}"
echo "    Verifier: ${VERIFIER_MODEL}"
echo "    Tag     : ${MODEL_TAG}"

declare -a DATASETS=(
  "cnndm simulator/thirdparty/benchmarks/cnn_dailymail train article 500 100"
  "gsm8k simulator/thirdparty/benchmarks/gsm8k train question 500 100"
  "humaneval simulator/thirdparty/benchmarks/humaneval test prompt 131 33"
)

profile_split() {
  local split_name="$1"
  local prompts_file="$2"
  local metrics_file="$3"
  local details_file="$4"

  mkdir -p "$(dirname "${metrics_file}")"
  mkdir -p "$(dirname "${details_file}")"

  if [[ -f "${details_file}" ]]; then
    echo ">>> Profiling ${split_name} prompts (${prompts_file})"
    echo "    cached results detected (${details_file}); skipping (delete file to regenerate)"
    return
  fi

  echo ">>> Profiling ${split_name} prompts (${prompts_file})"
  # Source conda directly for real-time output (conda run buffers aggressively)
  bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh' && \
    conda activate '${CONDA_ENV}' && \
    exec python -u '${SCRIPT_DIR}/speculative.py' \
      --drafter-model '${DRAFTER_MODEL}' \
      --verifier-model '${VERIFIER_MODEL}' \
      --spec-tokens '${SPEC_TOKENS}' \
      --max-tokens '${MAX_TOKENS}' \
      --max-prompt-tokens '${MAX_PROMPT_TOKENS}' \
      --prompts-file '${prompts_file}' \
      --metrics-jsonl '${metrics_file}' \
      --details-jsonl '${details_file}'
  "
}

echo ">>> Step 1: Export prompts"
for entry in "${DATASETS[@]}"; do
  read -r tag dataset_path split text_column train_size test_size <<<"${entry}"
  train_out="${PROMPT_DIR}/${tag}_train.jsonl"
  test_out="${PROMPT_DIR}/${tag}_test.jsonl"
  if [[ -f "${train_out}" && -f "${test_out}" ]]; then
    echo "  - ${tag}: cached prompts detected (delete files to regenerate)"
    continue
  fi
  echo "  - ${tag}: ${train_size} train / ${test_size} test"
  ${CONDA_RUN} python "${SCRIPT_DIR}/prepare_prompts.py" \
    --dataset-path "${dataset_path}" \
    --split "${split}" \
    --text-column "${text_column}" \
    --train-size "${train_size}" \
    --test-size "${test_size}" \
    --train-output "${train_out}" \
    --test-output "${test_out}"
done

echo ">>> Step 2: Profile training prompts"
train_detail_paths=()
for entry in "${DATASETS[@]}"; do
  read -r tag _ _ _ _ _ <<<"${entry}"
  prompts_file="${PROMPT_DIR}/${tag}_train.jsonl"
  metrics_file="${RUN_RESULTS_DIR}/${tag}_train_metrics.jsonl"
  details_file="${RUN_RESULTS_DIR}/${tag}_train_details.jsonl"
  profile_split "${tag} train" "${prompts_file}" "${metrics_file}" "${details_file}"
  train_detail_paths+=("${details_file}")
done
cat "${train_detail_paths[@]}" > "${TRAIN_DETAILS_COMBINED}"

echo ">>> Step 3: Train regressor"
${CONDA_RUN} python "${SCRIPT_DIR}/regressor.py" \
  --details-jsonl "${TRAIN_DETAILS_COMBINED}" \
  --spec-tokens "${SPEC_TOKENS}" \
  --output-model "${RUN_ACCEPTANCE_MODEL}" \
  --metadata "drafter=${DRAFTER_MODEL}" \
  --metadata "verifier=${VERIFIER_MODEL}" \
  --print-report

echo ">>> Step 4: Profile test prompts"
test_detail_paths=()
for entry in "${DATASETS[@]}"; do
  read -r tag _ _ _ _ _ <<<"${entry}"
  prompts_file="${PROMPT_DIR}/${tag}_test.jsonl"
  metrics_file="${RUN_RESULTS_DIR}/${tag}_test_metrics.jsonl"
  details_file="${RUN_RESULTS_DIR}/${tag}_test_details.jsonl"
  profile_split "${tag} test" "${prompts_file}" "${metrics_file}" "${details_file}"
  test_detail_paths+=("${details_file}")
done
cat "${test_detail_paths[@]}" > "${TEST_DETAILS_COMBINED}"

echo ">>> Step 5: Evaluate regressor"
${CONDA_RUN} python "${SCRIPT_DIR}/evaluate_regressor.py" \
  --model "${RUN_ACCEPTANCE_MODEL}" \
  --details-jsonl "${TEST_DETAILS_COMBINED}" \
  --print-report \
  --metrics-json "${TEST_METRICS_JSON}"

echo ">>> Pipeline completed successfully."
