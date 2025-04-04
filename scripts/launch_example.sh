#!/bin/bash
# Example launch script for speculative decoding (2 nodes required).

# ** Target node setup **
# On the target machine, run:
# python run_node.py --role target --model-id ./compiled-llama3-8b --port 50051

# ** Draft node setup **
# On the draft machine, edit the TARGET_IP below, then run:
TARGET_IP="xx.xx.xx.xx"   # <- Replace with the actual target node IP

python run_node.py --role draft --peer-ip $TARGET_IP --model-id ./compiled-llama3-1b \
    --port 50051 --prompt "Once upon a time," --max-tokens 50 --verify --perf-test

# This will start the draft process, connect to the target, and run speculative decoding.
# The outputs and performance comparison will be printed on the draft node console.