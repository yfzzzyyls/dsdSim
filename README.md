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

install dependencies

```
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
pip install --upgrade transformers-neuronx
```

## Setup

1. **Clone Repo & Install**:

   ```bash
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

### **Compile & Run the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.2-3b --port 50051 --sequence_length 128 --profile
```

### **Compile & Run the Draft Model server**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b --target_host 3.16.109.246 --port 50051 --prompt "Once upon a time," --target_model /home/ubuntu/models/llama-3.2-3b
```

### **Example Output**

```
[Draft] Final output: Once upon a time, ...
[Draft] Time: 2.35 s, tokens: 20, speed: 8.51 tokens/s
```

## **Run a Single Model for Verification**

You can also run either the draft or target model **standalone** (without speculative decoding) to verify its generation output token-by-token. This is useful for debugging and sanity checks to ensure each model behaves as expected given a prompt.

For example, to run the **target model** by itself on a prompt:

```
python main.py --role verify_target --model /home/ubuntu/models/llama-3.2-3b --prompt "Once upon a time," --max_new_tokens 20 --sequence_length 128 --profile
```

This will load the 3B target model and generate 20 tokens continuing the prompt, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt "Once upon a time," --max_tokens 20
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default,** **`--sequence_length 128` is used; ensure you use the same sequence length that the model was compiled with (or specify** **`--sequence_length` accordingly) to avoid recompilation. The** **`--max_tokens` option controls how many new tokens to generate for the prompt.

## **Performance Testing**

Run the **evaluate_test.py** script to compare speculative decoding vs. target-only:

```
Speculative decoding result:
Once upon a time, ...
Spec time: 2.12s, tokens=40, throughput=18.87 t/sBaseline target-only result:
Once upon a time, ...
Baseline time: 3.95s, tokens=40, throughput=10.12 t/s
```

This shows ~1.8x speedup from speculative decoding.

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
