# Distributed Speculative Decoding on AWS Trainium

This repository has been adapted for **multi-device** AWS Trainium usage with **speculative decoding**, using **Meta LLaMA 3.2** (1B draft + 3B target) in **bfloat16**.

## Project Structure

Below is an overview of the repository structure and how the modules relate to each other:

```
Choral-Spec/
├── main.py                  # CLI entry point; parses args and launches roles (draft, target, compile, verify)
├── inference/               # Package for model loading, speculative decoding, and verification logic
│   ├── model_loader.py      # Utilities to load or compile LLaMA models on AWS Neuron, provides `load_model` and `compile_model`
│   ├── draft_worker.py      # Draft client process: performs speculative decoding, communicates with target server via gRPC
│   ├── target_worker.py     # Target server process: serves the target model over gRPC (one token at a time)
│   ├── speculative.py       # Implements the speculative decoding algorithm (combines draft model predictions with target verification)
│   └── verify.py            # Verification utilities: can run a model standalone for debugging, and compare draft vs target outputs
├── grpc_comm/               # gRPC definitions and generated code for inter-process communication
│   ├── grpc_client.py          # Definition of SpeculativeService (gRPC 
│   ├── inference.proto          # Definition of SpeculativeService (gRPC service for generation and verification)
│   ├── inference_pb2.py         # Generated Python classes from the proto definitions
│   └── inference_pb2_grpc.py    # Generated gRPC client/server code based on the proto
├── requirements.txt         # Python dependencies for the project
└── README.md                # Documentation and usage instructions

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

### **Compile & Run the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.1-8b/ --port 50051 --sequence_length 128 --batch 2 --profile --top_p 1.0
```

### **Compile & Run the Draft Model server**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host 52.15.111.1 --port 50051 --prompt_text prompt.txt --max_new_tokens 100 --sequence_length 128 --profile --top_p 1.0 --temperature 0.9
```

### **Example Output**

```
2025-04-25 03:36:14,234 INFO inference.draft_worker: [BATCH] Decoding prompt 0: What is the difference between llama and alpaca?
2025-04-25 03:36:22,733 INFO inference.speculative: Latency: 8.49 seconds
2025-04-25 03:36:22,733 INFO inference.speculative: Speculative decoding match rate: 9.38% (Draft accepted: 6, Target generated: 58)
2025-04-25 03:36:22,733 INFO inference.draft_worker: Batched decode completed in 8.50s.

=== Final Outputs (BATCH approach) ===
[Prompt 0 Output]:
What is the difference between llama and alpaca? Alpacas are native to South America Peru, Bolivia, Chile have grey, brown, white, rose grey and fawn coloured coats while llamas have been domestic garded for over 4,000 years, are usually brown in colour, have thick woolly coats and longer legs and have been used far quieter and
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
python main.py --role verify_target --model /home/ubuntu/models/llama-3.1-8b --prompt_text prompt.txt --sequence_length 128 --max_new_tokens 100 --profile
```

This will load the 8B target model and generate 64 tokens continuing the prompt, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_target --model /home/ubuntu/models/llama-3.2-1b --prompt_text prompt.txt --sequence_length 128 --max_new_tokens 100 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128` is used; ensure you use the same sequence length that the model was compiled with (or specify** **`--sequence_length` accordingly) to avoid recompilation. The** `--max_tokens` option controls how many new tokens to generate for the prompt.

## **Supported features**

