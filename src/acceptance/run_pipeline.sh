#!/usr/bin/env bash
# End-to-end acceptance profiling pipeline (prompt export → profiling → training → eval).

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="llama2spec"
SCRIPT_DIR="${PROJECT_ROOT}/src/acceptance"
PROMPT_DIR="${PROJECT_ROOT}/prompts"
RESULTS_ROOT="${PROJECT_ROOT}/results"
ACCEPTANCE_DIR="${PROJECT_ROOT}/acceptance"

PREP_SCRIPT="${PROJECT_ROOT}/src/experiments/speculative/prepare_prompts.py"
PROFILE_SCRIPT="${SCRIPT_DIR}/speculative_profiler.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_regressor.py"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_regressor.py"

mkdir -p "${PROMPT_DIR}" "${RESULTS_ROOT}" "${ACCEPTANCE_DIR}"

SPEC_TOKENS=4
MAX_TOKENS=160
MAX_PROMPT_TOKENS=128
DRAFTER_MODEL="meta-llama/Llama-2-7b-hf"
VERIFIER_MODEL="meta-llama/Llama-2-70b-hf"
RUN_LABEL=""

usage() {
  cat <<USAGE
Usage: $0 [options]
  --drafter-model MODEL       Drafter (speculative) model identifier
  --verifier-model MODEL      Verifier (target) model identifier
  --spec-tokens N             Speculative chunk size (default ${SPEC_TOKENS})
  --max-tokens N              Max tokens to generate per prompt (default ${MAX_TOKENS})
  --max-prompt-tokens N       Max prompt tokens (default ${MAX_PROMPT_TOKENS})
  --run-label LABEL           Optional label for outputs (default derived from models)
  -h, --help                  Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drafter-model) DRAFTER_MODEL="$2"; shift 2 ;;
    --verifier-model) VERIFIER_MODEL="$2"; shift 2 ;;
    --spec-tokens) SPEC_TOKENS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

sanitize() {
  echo "$1" | sed 's|.*/||' | sed 's/[^A-Za-z0-9._-]/_/g'
}

if [[ -z "${RUN_LABEL}" ]]; then
  RUN_LABEL="$(sanitize "${DRAFTER_MODEL}")_vs_$(sanitize "${VERIFIER_MODEL}")"
fi

RUN_RESULTS_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
mkdir -p "${RUN_RESULTS_DIR}"
MODEL_BUNDLE="${ACCEPTANCE_DIR}/${RUN_LABEL}.joblib"

declare -a DATASETS=(
  "cnndm src/thirdparty/benchmarks/cnn_dailymail train article 5000 100 100"
  "gsm8k src/thirdparty/benchmarks/gsm8k train question 5000 100 100"
  "humaneval src/thirdparty/benchmarks/humaneval test prompt 131 0 33"
)

conda_exec() {
  # Source conda for interactive output
  bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh' && \
    conda activate '${CONDA_ENV}' && \
    exec \"$@\"
  "
}

echo ">>> Exporting prompts"
for entry in "${DATASETS[@]}"; do
  read -r tag dataset_path split text_col train_sz val_sz test_sz <<<"${entry}"
  train_out="${PROMPT_DIR}/${tag}_train.jsonl"
  val_out="${PROMPT_DIR}/${tag}_val.jsonl"
  test_out="${PROMPT_DIR}/${tag}_test.jsonl"
  if [[ -f "${train_out}" && -f "${test_out}" && ( "${val_sz}" -eq 0 || -f "${val_out}" ) ]]; then
    echo "  - ${tag}: cached prompts detected"
    continue
  fi
  cmd=(python "${PREP_SCRIPT}"
       --dataset-path "${dataset_path}"
       --split "${split}"
       --text-column "${text_col}"
       --train-size "${train_sz}"
       --test-size "${test_sz}"
       --train-output "${train_out}"
       --test-output "${test_out}")
  if [[ "${val_sz}" -gt 0 ]]; then
    cmd+=(--val-size "${val_sz}" --val-output "${val_out}")
  fi
  conda_exec "${cmd[@]}"
done

profile_split() {
  local label="$1"
  local prompts="$2"
  local metrics="$3"
  local details="$4"
  mkdir -p "$(dirname "${metrics}")" "$(dirname "${details}")"
  if [[ -f "${details}" ]]; then
    echo "  - ${label}: cached profile detected"
    return
  fi
  echo "  - Profiling ${label}"
  conda_exec python "${PROFILE_SCRIPT}" \
    --drafter-model "${DRAFTER_MODEL}" \
    --verifier-model "${VERIFIER_MODEL}" \
    --spec-tokens "${SPEC_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
    --prompts-file "${prompts}" \
    --metrics-jsonl "${metrics}" \
    --details-jsonl "${details}"
}

echo ">>> Profiling training prompts"
train_details=()
for entry in "${DATASETS[@]}"; do
  read -r tag _ _ _ _ _ _ <<<"${entry}"
  prompts="${PROMPT_DIR}/${tag}_train.jsonl"
  metrics="${RUN_RESULTS_DIR}/${tag}_train_metrics.jsonl"
  details="${RUN_RESULTS_DIR}/${tag}_train_details.jsonl"
  profile_split "${tag} train" "${prompts}" "${metrics}" "${details}"
  train_details+=("${details}")
done
cat "${train_details[@]}" > "${RUN_RESULTS_DIR}/train_details.jsonl"

echo ">>> Training acceptance regressor"
conda_exec python "${TRAIN_SCRIPT}" \
  --details-jsonl "${RUN_RESULTS_DIR}/train_details.jsonl" \
  --spec-tokens "${SPEC_TOKENS}" \
  --output-model "${MODEL_BUNDLE}" \
  --metadata "drafter=${DRAFTER_MODEL}" \
  --metadata "verifier=${VERIFIER_MODEL}" \
  --print-report

echo ">>> Profiling validation/test prompts"
val_details=()
test_details=()
for entry in "${DATASETS[@]}"; do
  read -r tag _ _ _ _ val_sz _ <<<"${entry}"
  val_prompts="${PROMPT_DIR}/${tag}_val.jsonl"
  test_prompts="${PROMPT_DIR}/${tag}_test.jsonl"
  if [[ "${val_sz}" -gt 0 ]]; then
    profile_split "${tag} val" "${val_prompts}" \
      "${RUN_RESULTS_DIR}/${tag}_val_metrics.jsonl" \
      "${RUN_RESULTS_DIR}/${tag}_val_details.jsonl"
    val_details+=("${RUN_RESULTS_DIR}/${tag}_val_details.jsonl")
  fi
  profile_split "${tag} test" "${test_prompts}" \
    "${RUN_RESULTS_DIR}/${tag}_test_metrics.jsonl" \
    "${RUN_RESULTS_DIR}/${tag}_test_details.jsonl"
  test_details+=("${RUN_RESULTS_DIR}/${tag}_test_details.jsonl")
done
[[ ${#val_details[@]} -gt 0 ]] && cat "${val_details[@]}" > "${RUN_RESULTS_DIR}/val_details.jsonl"
cat "${test_details[@]}" > "${RUN_RESULTS_DIR}/test_details.jsonl"

echo ">>> Evaluating regressor"
conda_exec python "${EVAL_SCRIPT}" \
  --model "${MODEL_BUNDLE}" \
  --details-jsonl "${RUN_RESULTS_DIR}/test_details.jsonl" \
  --print-report \
  --metrics-json "${RUN_RESULTS_DIR}/test_regressor_metrics.json"

echo ">>> Pipeline complete. Artifacts in ${RUN_RESULTS_DIR}"
