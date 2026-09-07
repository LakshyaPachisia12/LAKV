# LAKV
### Adaptive KV-Cache Compression for Multi-Agent LLM Systems

> A training-free, layer-aware framework for reducing KV-cache memory while preserving the information needed for downstream reasoning.

Large language models accumulate a Key-Value (KV) cache during inference. For long contexts and multi-agent pipelines, this cache can become a major memory bottleneck.

LAKV explores whether we can compress that cache **selectively** instead of treating every transformer layer and every token equally.

---

## Overview

LAKV is a training-free KV-cache compression framework built around **layer-aware importance estimation**.

Instead of applying a single compression strategy uniformly across the model, LAKV profiles each transformer layer and assigns it an appropriate compression tier.

The current system combines three signals:

- **Attention Importance**
- **Offset Variance**
- **Effective Rank**

These signals are used to estimate how sensitive each layer is to KV-cache compression.

The resulting layer profiles determine whether a layer should use:

- FP16 / uncompressed KV
- INT8 quantization
- INT4 quantization
- Token dropping

The goal is simple:

> **Compress aggressively where the model can tolerate it, and preserve information where compression is likely to hurt reasoning.**

---

## Why KV-Cache Compression?

During autoregressive generation, transformers store previously computed keys and values so they do not need to be recomputed at every step.

For a long context, this cache can consume a significant amount of GPU memory.

This becomes even more important in multi-agent systems, where intermediate reasoning states and context may be passed between sequential model calls.

A naive approach is to apply the same compression strategy uniformly.

LAKV instead asks:

**Which layers can be compressed heavily, and which layers need more information preserved?**

---

## Architecture

```text
                         Input Model
                              │
                              ▼
                   ┌────────────────────┐
                   │  Layer Profiling   │
                   └─────────┬──────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
      Attention         Offset          Effective
      Importance        Variance           Rank
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                   ┌────────────────────┐
                   │  Layer Selection   │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Compression Tier   │
                   └─────────┬──────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
              INT8         INT4      Token Dropping
                │            │            │
                └────────────┼────────────┘
                             ▼
                      Compressed KV
                             │
                             ▼
                       LLM Inference
```

---

## Current Results

LAKV currently achieves:

### **2.81× KV-cache compression**

on **Qwen2.5-7B-Instruct** using a training-free compression pipeline.

The system evaluates layers individually and assigns compression tiers based on their measured characteristics rather than applying a uniform policy.

### Compression vs. Reasoning

Aggressive compression exposed an important failure mode.

Under an INT4 configuration, per-tensor outlier clipping caused GSM8K accuracy to collapse to **0%** in the evaluated setup.

This led to the development of an **offset-correction mechanism** designed to recover information lost during KV quantization.

This remains an active area of investigation.

---

## Design

### 1. Layer Profiling

Each transformer layer is evaluated using multiple signals describing the information contained in its KV representations.

#### Attention Importance

Estimates how strongly the layer's cached representations contribute to attention.

#### Offset Variance

Measures distributional characteristics that become important when applying low-bit quantization.

#### Effective Rank

Estimates the dimensional structure of the representations and provides a signal for how much information can potentially be compressed.

---

### 2. Layer Selection

The profiling signals are combined to determine how aggressively each layer should be compressed.

This produces a layer-specific compression profile rather than a single global compression ratio.

---

### 3. Tiered Compression

LAKV explores multiple compression mechanisms:

| Tier                | Strategy                   |
| ------------------- | -------------------------- |
| High preservation   | FP16 / minimal compression |
| Moderate            | INT8                       |
| Aggressive          | INT4                       |
| Maximum compression | Token dropping             |

The objective is to allocate the available KV-cache budget where it matters most.

---

### 4. Offset Correction

Low-bit quantization introduces errors when KV distributions contain significant offsets or outliers.

LAKV includes an offset-correction mechanism to investigate whether these errors can be reduced without giving up the memory savings of aggressive quantization.

---

## Multi-Agent Setting

LAKV is designed with **multi-agent LLM systems** in mind.

In sequential agent pipelines, intermediate reasoning states can become part of the context consumed by later model calls.

This creates an important trade-off:

```text
More context
     │
     ▼
Better information retention
     │
     ▼
Larger KV cache
     │
     ▼
Higher memory cost
```

LAKV explores the opposite direction:

```text
Layer-aware compression
          │
          ▼
     Smaller KV cache
          │
          ▼
     Lower memory pressure
          │
          ▼
Preserve the information
that downstream agents actually need
```

---

## Repository Structure

```text
LAKV/
│
├── profiles/
│   └── Layer profiling / compression profiles
│
├── results/
│   └── Experimental results
│
├── calibration_profiler.py
├── evaluator.py
├── kv_compressor.py
├── layer_selector.py
├── offset_corrector.py
├── pipeline.py
└── run.py
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/LakshyaPachisia12/LAKV.git
cd LAKV
```

Run the pipeline:

```bash
python run.py
```

> The project is currently under active development. The exact evaluation configuration and model setup may change as the compression pipeline evolves.

---

## Evaluation

The project evaluates compression with two competing objectives:

### Memory Efficiency

How much the KV cache can be reduced.

### Model Fidelity

Whether the compressed cache preserves downstream model performance.

The central question is therefore not:

> "How much can we compress?"

but:

> **"How much can we compress before the information we care about disappears?"**

Current experiments use reasoning benchmarks including **GSM8K** to study the accuracy/compression trade-off.

---

## Current Status

LAKV is an ongoing research project.

Current work focuses on:

* improving low-bit KV quantization
* reducing accuracy degradation under aggressive compression
* evaluating offset correction
* improving layer-level compression policies
* extending evaluation across inference configurations
* understanding how compressed KV representations affect downstream multi-agent reasoning

---

## Motivation

KV-cache compression sits at an interesting intersection of:

**LLM inference × memory efficiency × model representations**

LAKV started from a simple question:

> **Do all transformer layers really need the same amount of KV-cache precision?**

The current results suggest they do not.

The rest of the project is about understanding how far that idea can be pushed without sacrificing the capabilities that make the model useful.

---
