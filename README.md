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
│   ├── inference.proto          # Definition of SpeculativeService (gRPC service for generation and verification)
│   ├── inference_pb2.py         # Generated Python classes from the proto definitions
│   └── inference_pb2_grpc.py    # Generated gRPC client/server code based on the proto
├── evaluate_test.py         # Script to evaluate performance of speculative decoding vs. baseline
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
   git clone https://github.com/yfzzzyyls/Choral-Spec
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

export compiler flag

```
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_RT_NUM_CORES=2
```

### **Optional:**

Clean cache before compile:

```
rm -rf /var/tmp/neuron-compile-cache
```

### **Compile & Run the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.2-3b/ --port 50051 --sequence_length 128
```

### **Compile & Run the Draft Model server**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_model llama-3.2-3b-compiled-128/ --target_host 18.222.253.234 --port 50051 --prompt "Once upon a time," --max_new_tokens 100 --gamma 4 --profile
```

### **Example Output**

```
......
INFO:inference.speculative:Draft chunk of 4 tokens accepted (all matched).
INFO:inference.speculative:Draft chunk of 4 tokens accepted (all matched).
INFO:inference.speculative:Draft predicted 2 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 3 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 1 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 1 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 2 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 1 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 1 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 2 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Draft predicted 0 tokens correctly, then diverged. Replaced mismatch token with target's token.
INFO:inference.speculative:Generation finished (target_finished=False, draft_finished=False, tokens_generated=100).
INFO:inference.speculative:Speculative decoding completed in 48.79 seconds, avg 0.4879s per token.
INFO:inference.speculative:Tokens generated: 100, Throughput: 2.05 tokens/sec, Match rate: 0.32
INFO:inference.draft_worker:Speculative decoding completed.

=== Final Output ===
Once upon a time, when I was in college, I loved to watch old movies on my computer. As a matter of fact, I still love to watch old movies on my computer. However, if I use my computer to watch one of these movies, I must take some actions to optimize the video performance in advance, usually because I must have good video quality. So, what are the main factors affecting the ability of a notebook pc to display video?
1, The display of the laptop is dependent on the screen of
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
python main.py --role verify_target --model /home/ubuntu/models/llama-3.2-3b/ --prompt "Once upon a time," --max_new_tokens 100 --sequence_length 128 --profile
```

This will load the 3B target model and generate 100 tokens continuing the prompt, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt "Once upon a time," --max_new_tokens 20 --sequence_length 128 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128` is used; ensure you use the same sequence length that the model was compiled with (or specify** **`--sequence_length` accordingly) to avoid recompilation. The** `--max_tokens` option controls how many new tokens to generate for the prompt.

## **Advanced Tips**

* **NEURON_RT_VISIBLE_CORES**: If your instance has multiple NeuronCores, you can dedicate certain cores to the draft or server processes:

```
#In terminal 1 (server):export NEURON_RT_VISIBLE_CORES=4-15
python model_service.py ...#In terminal 2 (draft):export NEURON_RT_VISIBLE_CORES=0-3
python draft_client.py ...
```

This can allow parallel execution, improving throughput.

* **Larger Models**: If using LLaMA 7B or bigger, you might need to distribute the model across multiple Neuron cores. That requires advanced compilation with **neuronx-distributed** or optimum-neuron. The approach is similar; just ensure the code references the sharded model.
* **Modifying the Speculative Mechanism**: The draft code uses a simple loop with **use_cache=True**. If you want to do partial or multi-token steps differently, you can adapt the logic in **draft_client.py**
