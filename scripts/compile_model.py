import argparse
from inference import model_loader

def main():
    parser = argparse.ArgumentParser(description="Compile a Hugging Face model for AWS Trainium (Neuron).")
    parser.add_argument("--model", type=str, required=True, help="Model ID or path to compile (e.g., meta-llama/Meta-Llama-3-8B).")
    parser.add_argument("--output", type=str, default=None, help="Directory to save the compiled model.")
    parser.add_argument("--num-cores", type=int, default=2, help="Number of Neuron cores to use for compilation.")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16", help="Precision for compilation (FP16 or BF16).")
    args = parser.parse_args()

    model_id = args.model
    output_dir = args.output
    if output_dir is None:
        # Default output directory name based on model name
        base_name = model_id.strip("/").split("/")[-1]
        output_dir = f"{base_name}-neuron"
    num_cores = args.num_cores
    dtype = args.dtype

    print(f"Compiling model '{model_id}' for Neuron with {num_cores} cores at {dtype} precision...")
    model, tokenizer = model_loader.load_model(model_id, num_cores=num_cores, dtype=dtype)
    # Save the compiled model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Compiled model saved to '{output_dir}'. You can use this directory with --model-id for inference.")

if __name__ == "__main__":
    main()