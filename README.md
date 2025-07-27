# Distributed and Fused Speculative Decoding on AWS Trainium

This repository supports two architectures for **speculative decoding** on AWS Trainium, using **Meta LLaMA 3.2-1B** (draft) and **LLaMA 3.1-8B** (target) models:

1. **Distributed Architecture**: Client-side draft model with multiple round-trips
2. **Fused Architecture**: Server-side fused model with single round-trip

## Project Structure

Below is an overview of the repository structure and how the modules relate to each other:

```
choral-spec-internal/
├── main.py                      # CLI entry point; parses args and launches roles (draft, target, fused_target, fused_client, verify)
├── inference/                   # Package for model loading, speculative decoding, and verification logic
│   ├── model_loader.py          # Utilities to load/compile models, includes load_fused_speculative_model()
│   ├── draft_worker.py          # Distributed: Draft client with speculative decoding
│   ├── target_worker.py         # Distributed: Target server (token-by-token verification)
│   ├── fused_draft_worker.py    # Fused: Simple client that sends requests
│   ├── fused_target_worker.py   # Fused: Server with FusedSpeculativeDecoder
│   ├── speculative.py           # Distributed speculative decoding algorithm
│   └── verify.py                # Verification utilities for standalone model testing
├── grpc_comm/                   # gRPC definitions and generated code
│   ├── inference.proto          # Protocol definitions for both architectures
│   ├── inference_pb2.py         # Generated Python classes
│   └── inference_pb2_grpc.py    # Generated gRPC client/server code
├── compare_architectures.py     # Automated performance comparison script
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
└── CLAUDE.md                    # Instructions for future Claude instances

```

## Dpendencies

Create a Trainium instance with AWS Neuron SDK using EC2 with the following settings:

1. 1. **Name:** Your Name
   2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
   3. **Instance type:** trn1.2xlarge
   4. **Key pair (login):** create a new key pair
   5. **Metadata version [under “Advanced details”]:** V2 only (otherwise, you will encounter a not authorized error)
   6. **When connecting to these instances via SSH, use the username of *ubuntu***
2. Activate the Neuron virtual environment to run inference by running `source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate`.

Install dependencies

```
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
pip install --upgrade transformers-neuronx
```

## Setup

1. **Clone Repo & Install**:

   ```
   git clone https://github.com/yfzzzyyls/choral-spec
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```
3. **Optinal: Generate new grpc files**

   ```
   cd grpc_comm
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
   ```

   Notice: in the newly generated inference_pb2_grpc.py, if you have the following code:

   ```
   import inference_pb2 as inference__pb2
   ```

   replace it with:

   ```
   from . import inference_pb2 as inference__pb2
   ```

## **Usage:**

### **Optional:**

Clean cache before compile:

```
rm -r /var/tmp/neuron-compile-cache
```

### **Distributed Architecture**

#### **1. Start the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.1-8b/ --port 50051 --sequence_length 128 --batch 2 --profile --top_p 1.0
```

#### **2. Run the Draft Model Client**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host 18.189.128.139 --port 50051 --prompt_text prompt.txt --sequence_length 128 --profile --top_p 1.0 --temperature 0.9
```

### **Fused Architecture**

#### **1. Start the Fused Speculative Server**

```
python main.py --role fused_target --draft_model /home/ubuntu/models/llama-3.2-1b --target_model /home/ubuntu/models/llama-3.1-8b --port 50051 --sequence_length 128 --batch 2 --tp_degree 2 --speculation_length 5
```

#### **2. Run the Fused Client**

```
python main.py --role fused_client --target_host 18.189.128.139 --port 50051 --prompt_text prompt.txt --temperature 0.9 --top_p 1.0 --speculation_length 5 --profile
```

### **Performance Comparison**

Run automated comparison between both architectures:

```
python compare_architectures.py --prompt-file prompt.txt --num-runs 3
```

Note: By default, both architectures will generate tokens up to the maximum sequence length (128) for fair comparison.

### **Example Output**

#### Distributed Architecture:
```
2025-07-26 23:04:15,748 INFO inference.draft_worker: [Thread 0] completed in 1.41s, tokens=48, throughput=34.15 t/s

=== Final Outputs (CONCURRENT) ===
[Prompt 0 Output]:
The history of AI can be traced back to 1960s, when the AI pioneers started to develop symbolic AI, also named as the von Neumann program. The early programs developed in the 1950s, including the Logic Theorist, a program that can prove various
```

#### Fused Architecture:
```
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Server generation time: 2582.84ms
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Tokens per second: 19.36
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Acceptance rate: 71.07%

[Prompt 0 Output]:
The history of AI can be traced back to the 1950s with the advent of the computer. The first application of AI was in the field of mathematics, where it was used to solve complex equations. Since then, AI has been used in a variety of fields, including finance, healthcare
```

## **Performance Profiling Stats**

```
INFO:inference.verify:Performance metrics saved to performance_target_only_20250408_013547.csv and performance_target_only_20250408_013547.json
```

Performance stats are saved to .cvs and .json files

## **Run a Single Model for Verification**

You can also run either the draft or target model **standalone** (without speculative decoding) to verify its generation output token-by-token. This is useful for debugging and sanity checks to ensure each model behaves as expected given a prompt.

To run the **target model** by itself on a prompt:

```
python main.py --role verify_target --model /home/ubuntu/models/llama-3.1-8b --prompt_text prompt.txt --sequence_length 128 --profile
```

This will load the 8B target model and generate tokens up to the sequence length, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt_text prompt.txt --sequence_length 128 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128`** is used; ensure you use the same sequence length that the model was compiled with (or specify `--sequence_length` accordingly) to avoid recompilation.

## **Key Parameters**

### Common Parameters:
- `--sequence_length`: Maximum sequence length (default: 128)
- `--max_new_tokens`: Maximum number of tokens to generate
- `--temperature`: Temperature for sampling (default: 1.0)
- `--top_p`: Top-p for nucleus sampling (default: 0.9)
- `--profile`: Enable performance profiling

### Distributed-Specific:
- `--gamma`: Number of draft tokens per verification step (default: 4)
- `--batch`: Batch size for model compilation (default: 2)

### Fused-Specific:
- `--speculation_length`: Number of tokens to speculate at once (default: 5)
- `--tp_degree`: Tensor parallelism degree (default: 2 for trn1.2xlarge)

## **Performance Notes**

- **Distributed architecture**: ~34 tokens/second throughput
- **Fused architecture**: ~19 tokens/second throughput
- Fused models always use `batch_size=1` internally
- Both architectures can use the same port (just not simultaneously)

