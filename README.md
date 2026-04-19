# Train on What You Deploy: Fixing the Train–Test Mismatch in Early Crisis Detection

Anonymous submission to NeurIPS 2026

## Overview

Standard sequential models are trained on
complete sessions but deployed on partial
prefixes — a systematic mismatch that causes
catastrophic performance collapse at early
turns. This repository provides code to
reproduce all experiments from the paper.

**Key finding:** A 3B LLM trained on
complete counseling sessions collapses to
AUC = 0.602 at five turns — worse than a
simple GRU (0.684). The fix is a single
line: sample `t ~ Uniform(1, T)` at each
training step. TCT's efficiency relative
to exhaustive prefix augmentation is
architecture-dependent: for sequential
encoders (GRU), TCT recovers 97% of the
DA-GRU upper bound at 1/47 the data cost;
for attention-based encoders (BERT),
exhaustive coverage provides substantially
larger gains (DA-BERT: 0.912 vs.
TCT-BERT: 0.737 at T=5).

---

## Repository Structure

```
tct-crisis-detection/
├── experiments/
│   ├── 01_gru_experiment.py           # TCT-GRU, Full-GRU, TCT-TRAM, Mean-pool, Last-turn
│   ├── 02_bert_experiment.py          # TCT-BERT, Full-BERT (klue/roberta-base)
│   ├── 03_lora_experiment.py          # TCT-LoRA, Full-LoRA (r=8, ~0.5% params)
│   ├── 04_llm_experiment.py           # TCT-LLM, Full-LLM (Qwen2.5-3B, QLoRA)
│   ├── 05_scheduled_sampling.py       # Scheduled Sampling comparison (Appendix E)
│   ├── 06_curriculum_fixed.py         # Curriculum-GRU, Fixed-T5-GRU (Appendix C)
│   ├── 07_qwen_05b_fp16.py            # Qwen2.5-0.5B fp16 ablation (Appendix D)
│   ├── 08_multi_prefix_experiment.py  # Multi-prefix baseline (Appendix F)
│   └── 09_da_gru_experiment.py        # DA-GRU oracle baseline (Appendix G)
    └── 10_da_bert_experiment.py       # DA-BERT oracle (Appendix H)
├── data/
│   └── README.md                      # Data download instructions
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

At each training step, we sample a random prefix length `t ~ Uniform(1, T)`
and train on the truncated sequence. This aligns the training distribution
with deployment conditions where only partial context is available.

**DA-GRU and DA-BERT (prefix-exhaustive baselines):**
```python
# DA-GRU/DA-BERT: expand each session into T
# independent prefix examples
for t in range(min_t, T + 1):
    loss = criterion(model(x_1_to_t), y)
    loss.backward()   # independent update per prefix
```
DA-GRU and DA-BERT deterministically cover all
prefix lengths, establishing short-prefix upper
bounds ($T{\leq}20$). Training is step-matched
to their respective TCT variants for fair
comparison. TCT's efficiency relative to these
baselines is architecture-dependent: for
sequential encoders (GRU), TCT recovers 97%
of DA-GRU's short-horizon gain at 1/47 the
data cost; for attention-based encoders (BERT),
exhaustive coverage provides substantially
larger gains.

---

## Data

The experiments use the **AI Hub Korean Child Counseling Corpus**
(3,236 sessions, 360,816 turns, ages 7–13).

The dataset is publicly available at:
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71680

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

## Reproducing Main Results (Tables 1 & 2)

### Step 1: GRU experiments (~2 hours on A100)
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
Reproduces: Curriculum-GRU (deterministic difficulty schedule) and
Fixed-T5-GRU (trains exclusively on T=5 prefixes).

### Appendix D: Quantization Ablation
```bash
python experiments/07_qwen_05b_fp16.py
```
Reproduces: Qwen2.5-0.5B in fp16 (no quantization) to confirm collapse
is not a 4-bit artifact.

### Appendix E: Scheduled Sampling Comparison
```bash
python experiments/05_scheduled_sampling.py
```
Reproduces: SS-Linear (linear ε decay) and SS-Exp (exponential ε decay)
vs TCT-GRU and Full-GRU.

### Appendix F: Multi-prefix Baseline
```bash
python experiments/08_multi_prefix_experiment.py
```
Reproduces: Multi-prefix simultaneous supervision vs TCT-GRU vs Full-GRU
(5 seeds).

**Note:** Multi-prefix requires ~50× more forward passes per update than TCT.
Runtime per seed is approximately 3–4 hours on A100.

### Appendix G: DA-GRU Oracle Baseline
```bash
python experiments/09_da_gru_experiment.py
```
Reproduces: DA-GRU (step-matched, ~835 gradient updates, 47× data expansion)
establishing an empirical upper bound on prefix-level exposure under the GRU
architecture.

### Appendix H: DA-BERT Oracle Baseline
```bash
python experiments/10_da_bert_experiment.py
```
Reproduces: DA-BERT (step-matched,
~1,336 gradient updates, 47× data expansion).
Reveals that attention-based encoders
require denser prefix coverage than
stochastic sampling provides — TCT's
efficiency advantage is
architecture-dependent (strongest for
sequential encoders).

**Key result:** TCT recovers 97% of the DA-GRU upper
bound at 1/47 the data cost for sequential
encoders; efficiency is architecture- dependent 
(DA-BERT shows a larger gap to TCT-BERT).

**Note:** Training is step-matched to TCT for fair comparison. DA-GRU expands
each session into T independent prefix examples; one pass through the expanded
dataset (~49,778 examples) corresponds to ~47× the original session count.

---

## Expected Results

### Main experiment (Table 2, Normal vs. Emergency)

| T  | TCT-GRU      | Full-GRU     | DA-GRU       | Multi-prefix | Δ      |
|----|--------------|--------------|--------------|--------------|--------|
| 5  | .860 ± .034  | .684 ± .007  | .904 ± .006  | .861 ± .019  | +.176  |
| 10 | .918 ± .018  | .830 ± .007  | .944 ± .002  | .917 ± .013  | +.087  |
| 15 | .942 ± .013  | .906 ± .006  | .956 ± .001  | .941 ± .011  | +.037  |
| 20 | .951 ± .013  | .908 ± .002  | .961 ± .002  | .950 ± .010  | +.043  |
| 30 | .974 ± .005  | .970 ± .001  | .975 ± .002  | .972 ± .004  | +.004  |
| 50 | .985 ± .001  | .993 ± .000  | .976 ± .003  | .985 ± .001  | -.008  |

DA-GRU = oracle upper bound (step-matched, 47× data).
Δ = TCT-GRU − Full-GRU.

---

## Random Seeds

All experiments use 5 seeds: `{42, 123, 2026, 7, 777}`.
Results are reported as mean ± std AUC across seeds.

---

## Hardware

All experiments were run on a single NVIDIA A100 GPU (40GB).

Approximate runtimes per seed:

| Model          | Runtime/seed |
|----------------|-------------|
| GRU            | ~25 min     |
| BERT           | ~50 min     |
| LoRA           | ~40 min     |
| LLM (QLoRA)    | ~90 min     |
| DA-GRU         | ~30 min     |
| Multi-prefix   | ~180 min    |
| DA-BERT        | ~25 min     |
---

## Citation

Anonymous submission. Under review at NeurIPS 2026.

---

## License

MIT License (code only). Data subject to AI Hub terms of use.
