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

SPEC_TOKENS=10
MAX_TOKENS=160
MAX_PROMPT_TOKENS=128
DRAFTER_MODEL="meta-llama/Llama-2-7b-hf"
VERIFIER_MODEL="meta-llama/Llama-2-70b-hf"
RUN_LABEL=""
GAMMA_MIN=2
GAMMA_MAX=10
GAMMA_MEAN=6
GAMMA_STD=2
USE_PROGRESS_BAR=1
PIPELINE_PROGRESS=1
PROGRESS_REFRESH_SECS=5

usage() {
  cat <<USAGE
Usage: $0 [options]
  --drafter-model MODEL       Drafter (speculative) model identifier
  --verifier-model MODEL      Verifier (target) model identifier
  --spec-tokens N             Speculative chunk size (default ${SPEC_TOKENS})
  --max-tokens N              Max tokens to generate per prompt (default ${MAX_TOKENS})
  --max-prompt-tokens N       Max prompt tokens (default ${MAX_PROMPT_TOKENS})
  --gamma-min N               Minimum speculative tokens (default ${GAMMA_MIN})
  --gamma-max N               Maximum speculative tokens (default ${GAMMA_MAX})
  --gamma-mean X              Mean for Gaussian sampler (default ${GAMMA_MEAN})
  --gamma-std X               Std dev for Gaussian sampler (default ${GAMMA_STD})
  --no-progress-bar           Disable tqdm progress bars during profiling
  --run-label LABEL           Optional label for outputs (default derived from models)
  --pipeline-progress 0|1     Show shell-side progress tracker (default ${PIPELINE_PROGRESS})
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
    --gamma-min) GAMMA_MIN="$2"; shift 2 ;;
    --gamma-max) GAMMA_MAX="$2"; shift 2 ;;
    --gamma-mean) GAMMA_MEAN="$2"; shift 2 ;;
    --gamma-std) GAMMA_STD="$2"; shift 2 ;;
    --no-progress-bar) USE_PROGRESS_BAR=0; shift ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --pipeline-progress) PIPELINE_PROGRESS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

sanitize() {
  echo "$1" | sed 's|.*/||' | sed 's/[^A-Za-z0-9._-]/_/g'
}

monitor_prompt_progress() {
  local label="$1"
  local total="$2"
  local details_path="$3"
  local target_pid="$4"
  local last_printed=-1

  while kill -0 "${target_pid}" 2>/dev/null; do
    local processed=0
    if [[ -f "${details_path}" ]]; then
      processed=$(wc -l < "${details_path}")
    fi
    if [[ "${processed}" -ne "${last_printed}" ]]; then
      printf "\r    [%s] %d/%d prompts complete" "${label}" "${processed}" "${total}"
      last_printed=${processed}
    fi
    sleep "${PROGRESS_REFRESH_SECS}"
  done

  local processed=0
  if [[ -f "${details_path}" ]]; then
    processed=$(wc -l < "${details_path}")
  fi
  printf "\r    [%s] %d/%d prompts complete\n" "${label}" "${processed}" "${total}"
}

if [[ -z "${RUN_LABEL}" ]]; then
  RUN_LABEL="$(sanitize "${DRAFTER_MODEL}")_vs_$(sanitize "${VERIFIER_MODEL}")"
fi

RUN_RESULTS_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
mkdir -p "${RUN_RESULTS_DIR}"
MODEL_BUNDLE="${ACCEPTANCE_DIR}/${RUN_LABEL}.joblib"

declare -a DATASETS=(
  "cnndm src/thirdparty/benchmarks/cnn_dailymail train article 500 100 100 cnn_dailymail 3.0.0"
  "gsm8k src/thirdparty/benchmarks/gsm8k train question 500 100 100 gsm8k main"
  "humaneval src/thirdparty/benchmarks/humaneval test prompt 131 0 33 openai_humaneval default"
)

conda_exec() {
  "${CONDA_BASE}/bin/conda" run -n "${CONDA_ENV}" "$@"
}

echo ">>> Exporting prompts"
for entry in "${DATASETS[@]}"; do
  read -r tag dataset_path split text_col train_sz val_sz test_sz hf_name hf_config <<<"${entry}"
  train_out="${PROMPT_DIR}/${tag}_train.jsonl"
  val_out="${PROMPT_DIR}/${tag}_val.jsonl"
  test_out="${PROMPT_DIR}/${tag}_test.jsonl"
  needs_export=0
  if [[ ! -f "${train_out}" ]]; then
    needs_export=1
  elif [[ "${train_sz}" -gt 0 ]]; then
    train_lines=$(wc -l < "${train_out}")
    if [[ "${train_lines}" -ne "${train_sz}" ]]; then
      needs_export=1
    fi
  fi

  if [[ "${val_sz}" -gt 0 ]]; then
    if [[ ! -f "${val_out}" ]]; then
      needs_export=1
    else
      val_lines=$(wc -l < "${val_out}")
      if [[ "${val_lines}" -ne "${val_sz}" ]]; then
        needs_export=1
      fi
    fi
  fi

  if [[ ! -f "${test_out}" ]]; then
    needs_export=1
  elif [[ "${test_sz}" -gt 0 ]]; then
    test_lines=$(wc -l < "${test_out}")
    if [[ "${test_lines}" -ne "${test_sz}" ]]; then
      needs_export=1
    fi
  fi

  if [[ "${needs_export}" -eq 0 ]]; then
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
       --test-output "${test_out}"
       --dataset-name "${tag}"
       --hf-dataset "${hf_name}"
  )
  if [[ "${hf_config}" != "default" ]]; then
    cmd+=(--hf-config "${hf_config}")
  fi
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

  local prompt_total=0
  if [[ -f "${prompts}" ]]; then
    prompt_total=$(wc -l < "${prompts}")
  fi
  if [[ "${prompt_total}" -le 0 ]]; then
    echo "  - ${label}: no prompts found (skipping)"
    return
  fi

  local processed=0
  if [[ -f "${details}" ]]; then
    processed=$(wc -l < "${details}")
  fi
  if [[ "${processed}" -ge "${prompt_total}" ]]; then
    echo "  - ${label}: cached profile detected"
    return
  fi

  if [[ "${processed}" -gt 0 ]]; then
    echo "  - Profiling ${label} (resuming at ${processed}/${prompt_total})"
  else
    echo "  - Profiling ${label}"
  fi

  cmd=(env PYTHONUNBUFFERED=1 python "${PROFILE_SCRIPT}"
    --drafter-model "${DRAFTER_MODEL}"
    --verifier-model "${VERIFIER_MODEL}"
    --spec-tokens "${SPEC_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
    --gamma-min "${GAMMA_MIN}"
    --gamma-max "${GAMMA_MAX}"
    --gamma-mean "${GAMMA_MEAN}"
    --gamma-std "${GAMMA_STD}"
    --prompts-file "${prompts}"
    --metrics-jsonl "${metrics}"
    --details-jsonl "${details}"
    --prompt-offset "${processed}")
  if [[ "${USE_PROGRESS_BAR}" -eq 1 ]]; then
    cmd+=(--progress-bar)
  fi

  if [[ -f "${metrics}" && "${processed}" -gt 0 ]]; then
    rm -f "${metrics}"
  fi

  conda_exec "${cmd[@]}" &
  local profiler_pid=$!
  local monitor_pid=
  if [[ "${PIPELINE_PROGRESS}" -ne 0 ]]; then
    monitor_prompt_progress "${label}" "${prompt_total}" "${details}" "${profiler_pid}" &
    monitor_pid=$!
  fi
  wait "${profiler_pid}"
  local profiler_status=$?
  if [[ -n "${monitor_pid}" ]]; then
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  return "${profiler_status}"
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
