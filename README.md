# Train on What You Deploy: Fixing the Train–Test Mismatch in Early Crisis Detection

Anonymous submission to NeurIPS 2026

---

## Overview

Standard sequential models are trained on complete sessions but deployed on partial prefixes — a systematic mismatch that causes catastrophic performance collapse at early turns. This repository provides code to reproduce all experiments from the paper.

**Key finding:** A 3B LLM trained on complete counseling sessions collapses to AUC = 0.602 at five turns — worse than a simple GRU (0.684). The fix is a single line: sample `t ~ Uniform(1, T)` at each training step.

---

## Repository Structure

```
tct-crisis-detection/
├── experiments/
│   ├── 01_gru_experiment.py        # TCT-GRU, Full-GRU, TCT-TRAM, Mean-pool, Last-turn
│   ├── 02_bert_experiment.py       # TCT-BERT, Full-BERT (klue/roberta-base)
│   ├── 03_lora_experiment.py       # TCT-LoRA, Full-LoRA (r=8, ~0.5% params)
│   ├── 04_llm_experiment.py        # TCT-LLM, Full-LLM (Qwen2.5-3B, QLoRA)
│   ├── 05_scheduled_sampling.py    # Scheduled Sampling comparison (Appendix E)
│   ├── 06_curriculum_fixed.py      # Curriculum-GRU, Fixed-T5-GRU (Appendix C)
│   └── 07_qwen_05b_fp16.py         # Qwen2.5-0.5B fp16 ablation (Appendix D)
├── data/
│   └── README.md                   # Data download instructions
├── requirements.txt
└── README.md
```

---

## Method

**Truncated Context Training (TCT):**

```python
# Full-context training (standard — broken)
loss = criterion(model(x_1_to_T), y)

# TCT (one-line fix)
t = random.randint(1, T)
loss = criterion(model(x_1_to_t), y)
```

At each training step, we sample a random prefix length `t ~ Uniform(1, T)` and train on the truncated sequence. This aligns the training distribution with deployment conditions where only partial context is available.

---

## Data

The experiments use the **AI Hub Korean Child Counseling Corpus** (3,236 sessions, 360,816 turns, ages 7–13).

The dataset is publicly available at:
```
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71680
```

See `data/README.md` for preprocessing instructions.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- transformers 4.35+
- peft 0.6+
- sentence-transformers 2.2.2
- scikit-learn 1.3.0
- bitsandbytes 0.41+ (for QLoRA experiments)

---

## Reproducing Main Results (Table 1 & 2)

### Step 1: GRU experiments (fastest, ~2 hours on A100)

```bash
python experiments/01_gru_experiment.py
```

Reproduces: TCT-GRU, Full-GRU, TCT-TRAM, Mean-pool, Last-turn across 5 seeds.

Output: `results_gru.pkl`

### Step 2: BERT experiments (~4 hours on A100)

```bash
python experiments/02_bert_experiment.py
```

Reproduces: TCT-BERT, Full-BERT (klue/roberta-base, full fine-tuning).

Output: `results_bert.pkl`

### Step 3: LoRA experiments (~3 hours on A100)

```bash
python experiments/03_lora_experiment.py
```

Reproduces: TCT-LoRA, Full-LoRA (r=8, α=16, ~0.5% trainable parameters).

Output: `results_lora.pkl`

### Step 4: LLM experiments (~8 hours on A100)

```bash
python experiments/04_llm_experiment.py
```

Reproduces: TCT-LLM, Full-LLM (Qwen2.5-3B via QLoRA, 4-bit NF4).

Output: `results_llm.pkl`

---

## Reproducing Appendix Results

### Appendix C: Training Scheme Ablation

```bash
python experiments/06_curriculum_fixed.py
```

Reproduces: Curriculum-GRU (deterministic difficulty schedule) and Fixed-T5-GRU (trains exclusively on T=5 prefixes).

### Appendix D: Quantization Ablation

```bash
python experiments/07_qwen_05b_fp16.py
```

Reproduces: Qwen2.5-0.5B in fp16 (no quantization) to confirm collapse is not a 4-bit artifact.

### Appendix E: Scheduled Sampling Comparison

```bash
python experiments/05_scheduled_sampling.py
```

Reproduces: SS-Linear (linear ε decay) and SS-Exp (exponential ε decay) vs TCT-GRU and Full-GRU.

---

### Appendix F: Multi-prefix Comparison

```bash
python experiments/08_multi_prefix_experiment.py
```

Reproduces Table 8: Multi-prefix vs. TCT-GRU 
vs. Full-GRU. Note: ~50× slower than TCT 
per update step.


## Expected Results (Table 1, Normal vs Emergency)

| T  | TCT-GRU       | Full-GRU      | Mean-pool | Δ      |
|----|---------------|---------------|-----------|--------|
| 5  | .860 ± .034   | .684 ± .007   | .873      | +.176  |
| 10 | .918 ± .018   | .830 ± .007   | .913      | +.087  |
| 15 | .942 ± .013   | .906 ± .006   | .936      | +.037  |
| 20 | .951 ± .013   | .908 ± .002   | .957      | +.043  |
| 30 | .974 ± .005   | .970 ± .001   | .966      | +.004  |
| 50 | .985 ± .001   | .993 ± .000   | .990      | -.008  |

---

## Random Seeds

All experiments use 5 seeds: `{42, 123, 2026, 7, 777}`.

Results are reported as mean ± std AUC across seeds.

---

## Hardware

All experiments were run on a single NVIDIA A100 GPU (40GB).

Approximate runtimes per seed:
- GRU: ~25 minutes
- BERT: ~50 minutes
- LoRA: ~40 minutes
- LLM (Qwen2.5-3B QLoRA): ~90 minutes

---

## Citation

```
Anonymous submission. Under review at NeurIPS 2026.
```

---

## License

MIT License (code only). Data subject to AI Hub terms of use.
