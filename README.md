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
   cd Choral-Spec/grpc_comm
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
python main.py --role target --model /home/ubuntu/models/llama-3.1-8b/ --port 50051 --sequence_length 128 --top_p 1.0
```

### **Compile & Run the Draft Model server**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host 18.188.214.41 --port 50051 --prompt_text prompt.txt --max_new_tokens 64 --gamma 4 --sequence_length 128 --profile --top_p 1.0 --temperature 0.9
```

### **Example Output**

```
INFO:inference.draft_worker:[Thread-2] Starting speculative decoding with session_id=104464132
INFO:inference.draft_worker:[Thread-0] Starting speculative decoding with session_id=3780988024
INFO:inference.draft_worker:[Thread-1] Starting speculative decoding with session_id=1574770097
INFO:inference.speculative:Speculative decoding match rate: 37.50% (Draft accepted: 48, Target generated: 80)
INFO:inference.speculative:Speculative decoding match rate: 22.48% (Draft accepted: 29, Target generated: 100)
INFO:inference.speculative:Speculative decoding match rate: 17.83% (Draft accepted: 23, Target generated: 106)

=== Final Batched Outputs ===

[Prompt 0 Output]:
Once upon a time, there there
......
[Prompt 1 Output]:
......
[Prompt 2 Output]:
......
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
python main.py --role verify_target --model /home/ubuntu/models/llama-3.1-8b --prompt "What is the difference between llama and alpaca?" --sequence_length 128 --max_new_tokens 64 --profile
```

This will load the 8B target model and generate 64 tokens continuing the prompt, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt "Hi, how are you? Tell me about the difference between llama and alpaca." --sequence_length 640 --max_new_tokens 128 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128` is used; ensure you use the same sequence length that the model was compiled with (or specify** **`--sequence_length` accordingly) to avoid recompilation. The** `--max_tokens` option controls how many new tokens to generate for the prompt.

## **Supported features**
