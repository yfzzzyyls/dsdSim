# Choral-Spec

Choral Spec is a project aims to optimize the performance of distributed inference for speculative decoding on AWS Trainium

# Speculative LLM Decoding on AWS Trainium (LLaMA3.2 Example)

This repository implements a **scalable, distributed speculative decoding system** for large language model (LLM) inference using two AWS Trainium instances (e.g., `trn1.2xlarge`). It uses a small **draft model** to generate tokens speculatively and a large **target model** to validate those tokens. The system is built with Hugging Face Transformers and Optimum Neuron for model compilation, and uses gRPC for communication between the draft and target nodes.

This repository implements a **scalable, distributed speculative decoding system** for large language model (LLM) inference on **AWS Trainium** (e.g., **trn1.2xlarge**) **without** using **optimum-neuron**. Instead, we directly use **Hugging Face Transformers** plus **torch-neuronx.trace** to compile models for Trainium. The system involves:

* A **draft model** (smaller, faster) generating tokens speculatively.
* A **target model** (larger, more accurate) validating those tokens.
* **gRPC** for communication between the draft and target nodes.

## Project Structure

```text
spec-decoding-llama3/
├── README.md
├── requirements.txt
├── run_node.py                  # Main entry point for either draft or target role
├── grpc_comm/
│   ├── proto/
│   │   ├── inference.proto      # gRPC service definitions
│   │   └── (generated code)     # Inference gRPC stub/server code (after protoc)
│   ├── grpc_server.py           # gRPC server implementation (target node)
│   └── grpc_client.py           # gRPC client helper (draft node)
├── inference/
│   ├── model_loader.py          # Neuron model compilation & loading utilities
│   ├── draft_worker.py          # Draft node logic (speculative token generation)
│   ├── target_worker.py         # Target node logic (token verification & generation)
│   ├── speculative.py           # Speculative decoding pipeline coordination
│   └── verify.py                # Functional and performance verification routines
└── scripts/
    ├── compile_model.py         # Helper to compile a model with Optimum Neuron
    └── launch_example.sh        # Example commands to launch draft/target nodes
```


## **2. Setup and Installation**

Below are the **step-by-step** instructions to get this running on **AWS Trainium** (**trn1** instance). We assume you have an **Ubuntu 22.04** AMI (or the official AWS Deep Learning AMI for Trainium). Adjust accordingly for other OS versions.

1. **Provision your trn1 instances** (at least two if you want to split draft & target, or you can run both roles on the same instance in separate terminals).
2. **Clone or copy** this repository onto **each** Trainium node. For instance:

   ```
   git clone https://github.com/yfzzzyyls/Choral-Spec.git
   cd Choral-Spec
   ```


3. **Install Python dependencies**
   Make sure your **pip** uses the Neuron repository (if needed) and then install:
   ```
   python -m pip install --upgrade \
       "grpcio>=1.50.0" \
       "grpcio-tools>=1.50.0" \
       "sentencepiece>=0.1.99"
   ```
