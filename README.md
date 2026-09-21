# Quantization-Induced Degradation in Multi-Step Symbolic Reasoning: An Empirical Analysis of 4-Bit Weight Formats and Token Entropy

[![Paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](https://doi.org10.5281/zenodo.22871125)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

This repository provides the official implementation, evaluation testbeds, and diagnostic logging scripts for the paper:  
**"Quantization-Induced Degradation in Multi-Step Symbolic Reasoning: An Empirical Analysis of 4-Bit Weight Formats and Token Entropy"**  
*Author: Ajayi David (2026)*

---

## 📌 Abstract & Key Findings

Post-Training Quantization (PTQ) allows resource-constrained local environments to run modern open-weight LLMs, but standard perplexity metrics often mask severe degradation on strict multi-step reasoning. 

This repository contains diagnostic pipelines measuring exact-match (EM) chain-of-thought accuracy, GPU memory footprints, latency, and step-wise Shannon token entropy ($H_t$) across `Qwen2.5-1.5B-Instruct` and `Llama-3.2-1B-Instruct` under FP16 baselines and 4-bit quantization regimes (NormalFloat4 [NF4] and Activation-Aware Weight Quantization [AWQ]).

### Core Findings:
* **Quantization Penalty:** 4-bit NF4 quantization reduces chain-of-thought mathematical reasoning accuracy by up to **13.8 percentage points** on GSM8K.
* **Entropy Spikes over Semantic Drift:** Degradation is not driven by continuous cumulative error. Instead, it is concentrated in **sharp token-entropy spikes ($H_t > 2.80$ bits)** at operational transitions and arithmetic operators.
* **Format Superiority:** AWQ protects salient activation channels, outperforming uniform NF4 by **7.4 percentage points** on reasoning benchmarks while sustaining identical memory constraints and higher decoding throughput.

---

## 📊 Benchmark Results

Evaluated on a 250-problem evaluation split of GSM8K using greedy argmax decoding ($T = 0.0$) on an NVIDIA A100-SXM4 GPU:

| Model / Configuration | Precision | Acc (EM %) | Mean $H_t$ (bits) | Peak VRAM | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B-Instruct (Baseline)** | FP16 | **61.2%** | 0.42 ± 0.12 | 3.42 GB | 58.2 tok/s |
| **Qwen2.5-1.5B-Instruct (AWQ)** | INT4 | 54.8% | 0.58 ± 0.18 | 1.48 GB | **82.4 tok/s** |
| **Qwen2.5-1.5B-Instruct (NF4)** | 4-bit | 47.4% | 0.74 ± 0.24 | **1.24 GB** | 44.1 tok/s |
| **Llama-3.2-1B-Instruct (Baseline)** | FP16 | **44.8%** | 0.51 ± 0.14 | 2.68 GB | 64.0 tok/s |
| **Llama-3.2-1B-Instruct (AWQ)** | INT4 | 38.0% | 0.69 ± 0.19 | 1.15 GB | **88.7 tok/s** |
| **Llama-3.2-1B-Instruct (NF4)** | 4-bit | 31.6% | 0.89 ± 0.29 | **0.98 GB** | 49.6 tok/s |

---

## 🛠️ Repository Structure

```text
├── data/
│   └── gsm8k_sample.jsonl         # Curated multi-step reasoning evaluation subset
├── scripts/
│   ├── eval_reasoning.py          # Primary evaluation pipeline (FP16, NF4, AWQ)
│   ├── entropy_tracker.py         # Step-wise Softmax Shannon entropy extraction
│   └── plot_diagnostics.py        # Entropy vs. token index visualization generator
├── results/
│   ├── raw_metrics.csv            # Run metrics, exact match scores, and VRAM stats
│   └── entropy_distribution.png   # Generated paper visualization
├── requirements.txt
└── README.md
