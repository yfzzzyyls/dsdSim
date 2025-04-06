# Distributed Inference with Speculative Decoding on AWS Trainium

This repository has been adapted for **single-device** AWS Trainium usage with **speculative decoding** by default, using **Meta LLaMA 3.2** (1B draft + 3B target) in **bfloat16**. We assume you have an **AWS DLAMI** with Neuron SDK installed.

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
   cd Choral-Spec
   # Ensure torch-neuronx, transformers[neuron], grpcio, etc. are installed
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```
3. **Optinal: Generate new grpc files**

   ```
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. model_service.proto
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. service.proto
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. batch.proto
   ```

   Notice: if you encounter import failure issue:

   replace:

   ```
   import model_service_pb2 as model__service__pb2
   ```

   to:

   ```
   from . import model_service_pb2 as model__service__pb2
   ```

## **Usage:**

export correct compiler flag

```
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_RT_NUM_CORES=2
```

### **Compile the Target and Draft Model Server**

```
python main.py --role compile --model /home/ubuntu/models/llama-3.2-3b/ --sequence_length 128
python main.py --role compile --model /home/ubuntu/models/llama-3.2-3b/ --sequence_length 128
```

### **Run the target server on target instance**

```
python main.py --role target --model /home/ubuntu/models/llama-3.2-3b/ --port 50051 --sequence_length 128
```

### Run the draft server on draft instance

```
# Replace <TARGET_IP> with your target machine’s IP address.
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host <TARGET_IP> --port 50051 --prompt "Once upon a time," --target_model /home/ubuntu/models/llama-3.2-3b/ --sequence_length 128 --max_new_tokens 50
```

### **Example Output**

```
[Draft] Final output: Once upon a time, ...
[Draft] Time: 2.35 s, tokens: 20, speed: 8.51 tokens/s
```

## **Performance Testing**

Run the **evaluate_test.py** script to compare speculative decoding vs. target-only:

1. **Ensure server is running** with the 3B model.
2. Launch:

```
python evaluate_test.py
  --host localhost --port 50051
  --draft_model models/llama-3.2-1b
  --compiled_draft_model models/llama1b_neuron.pt
  --compile
  --max_tokens 128 --gamma 4
  --prompt "Once upon a time,"
  --target_model_service
```

You’ll see something like:

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
