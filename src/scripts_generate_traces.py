import json
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = Path("traces")
SAMPLES = {
    "gsm8k": 2000,
    "cnndm": 2000,
    "humaneval": None,
}

DATASETS = {
    "gsm8k": {
        "path": "src/thirdparty/benchmarks/gsm8k",
        "split": "train",
        "prompt_field": "question",
        "answer_field": "answer",
    },
    "cnndm": {
        "path": "src/thirdparty/benchmarks/cnn_dailymail",
        "split": "train",
        "prompt_field": "article",
        "answer_field": "highlights",
    },
    "humaneval": {
        "path": "src/thirdparty/benchmarks/humaneval",
        "split": "test",
        "prompt_field": "prompt",
        "answer_field": "canonical_solution",
    },
}

DEVICE_TIER = "default"
INTERARRIVAL_MS = 50.0

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
print("Tokenizer ready", flush=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Writing traces to {OUTPUT_DIR.resolve()}", flush=True)


def iter_rows(name: str, spec: dict) -> Iterable[dict]:
    dataset = load_from_disk(spec["path"])[spec["split"]]
    limit = SAMPLES[name]
    total = len(dataset)
    count = total if limit is None else min(limit, total)
    print(f"Dataset {name}: using {count} rows out of {total}", flush=True)
    for idx in range(count):
        yield {
            "conversation_index": idx,
            "prompt": dataset[idx][spec["prompt_field"]],
            "answer": dataset[idx][spec["answer_field"]] or "",
        }


def write_trace(dataset_name: str, spec: dict) -> None:
    rows = list(iter_rows(dataset_name, spec))
    out_path = OUTPUT_DIR / f"{dataset_name}_trace.jsonl"
    arrival_ms = 0.0
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            prompt = row["prompt"] or ""
            answer = row["answer"] or ""
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            target_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
            target_tokens = max(1, target_tokens)
            record = {
                "request_id": f"{dataset_name}_{row['conversation_index']:05d}",
                "arrival_ms": arrival_ms,
                "prompt_tokens": prompt_tokens,
                "target_tokens": target_tokens,
                "device_tier": DEVICE_TIER,
                "metadata": {
                    "dataset": dataset_name,
                    "client_id": f"{dataset_name}_{row['conversation_index']:05d}",
                },
            }
            out.write(json.dumps(record))
            out.write("\n")
            arrival_ms += INTERARRIVAL_MS
    print(f"Wrote {len(rows)} records to {out_path}", flush=True)


for name, spec in DATASETS.items():
    write_trace(name, spec)

print("All traces generated.", flush=True)
