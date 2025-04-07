# Distributed Speculative Decoding on AWS Trainium

This repository has been adapted for **multi-device** AWS Trainium usage with **speculative decoding**, using **Meta LLaMA 3.2** (1B draft + 3B target) in **bfloat16**.

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

   Notice: in inference_pb2_grpc.py, if you have the following code:

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
python main.py --role target --model /home/ubuntu/models/llama-3.2-3b --port 50051 --sequence_length 128
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
