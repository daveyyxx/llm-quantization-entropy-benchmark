import argparse
import time
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def compute_entropy(logits):
    """Calculates per-token Shannon entropy in bits from unnormalized logits."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()

def evaluate(model_id, precision, max_samples=50):
    print(f"Loading {model_id} with precision: {precision}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantization configurations
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif precision == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif precision == "awq":
        # Assumes an AWQ pre-quantized model tag or AWQ-compatible weights
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        raise ValueError("Precision must be one of: fp16, nf4, awq")

    model.eval()

    # Peak VRAM checkpoint
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load dataset sample
    dataset = load_dataset("gsm8k", "main", split=f"test[:{max_samples}]")

    correct = 0
    total_entropy = []
    start_time = time.time()
    total_tokens = 0

    print("Running diagnostic evaluation...")
    for idx, item in enumerate(dataset):
        prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy argmax decoding
                return_dict_in_generate=True,
                output_scores=True
            )

        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[-1]:]
        total_tokens += len(gen_tokens)

        # Calculate average token entropy across generated step tokens
        for step_logits in outputs.scores:
            h_step = compute_entropy(step_logits[0])
            total_entropy.append(h_step)

    elapsed_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
    mean_entropy = np.mean(total_entropy) if total_entropy else 0.0

    print("\n--- Diagnostic Results ---")
    print(f"Model: {model_id} | Precision: {precision}")
    print(f"Mean Token Entropy: {mean_entropy:.4f} bits")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Throughput: {throughput:.2f} tokens/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM reasoning degradation and token entropy.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--precision", type=str, default="nf4", choices=["fp16", "nf4", "awq"])
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    evaluate(args.model, args.precision, args.samples)
