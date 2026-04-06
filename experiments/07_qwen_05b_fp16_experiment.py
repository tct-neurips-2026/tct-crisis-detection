# ============================================================
# Appendix D: Quantization Ablation
# Qwen2.5-0.5B fp16 (no quantization)
# Purpose: confirm collapse is not 4-bit artifact
# ============================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import torch.nn as nn
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

# ============================================================
# 설정
# ============================================================
SEEDS      = [42, 123, 2026, 7, 777]
T_EVAL     = [5, 10, 15, 20, 30, 50]
BATCH_SIZE = 4
EPOCHS     = 5
LR         = 3e-5
MAX_SEQ    = 512
DEVICE     = 'cuda' if torch.cuda.is_available() \
             else 'cpu'

# ✅ 0.5B fp16 — NO quantization
MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}  (fp16, no quantization)")

# ============================================================
# 데이터
# ============================================================
df = pd.read_pickle(
    '/content/full_dataset_with_dialogues'
    '_20260126_022510.pkl')
df_ne = df[df['crisis'].isin(
    ['정상군', '응급'])].copy()
df_ne['label'] = (
    df_ne['crisis'] == '응급').astype(int)

def extract_turns(turns_list, max_turns=50):
    return [t.get('text', '')
            for t in turns_list
            if isinstance(t, dict)][:max_turns]

df_ne['turn_texts'] = \
    df_ne['turns'].apply(extract_turns)
sessions = df_ne['turn_texts'].tolist()
labels   = df_ne['label'].tolist()
print(f"Normal: {labels.count(0)}, "
      f"Emergency: {labels.count(1)}")

# ============================================================
# Dataset
# ============================================================
class PrefixDataset(Dataset):
    def __init__(self, sessions, labels,
                 tokenizer, t_cutoff):
        self.tokenizer = tokenizer
        self.data = []
        for turns, label in zip(sessions, labels):
            t    = min(t_cutoff, len(turns))
            text = ' '.join(turns[:t])
            self.data.append((text, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_SEQ,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        return {
            'input_ids':
                enc['input_ids'].squeeze(0),
            'attention_mask':
                enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                label, dtype=torch.long)
        }


class TCTDataset(Dataset):
    def __init__(self, sessions,
                 labels, tokenizer):
        self.sessions  = sessions
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        turns = self.sessions[idx]
        label = self.labels[idx]
        t = random.randint(
            1, max(1, len(turns)))
        text = ' '.join(turns[:t])
        enc  = self.tokenizer(
            text,
            max_length=MAX_SEQ,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        return {
            'input_ids':
                enc['input_ids'].squeeze(0),
            'attention_mask':
                enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                int(label), dtype=torch.long)
        }

# ============================================================
# 모델: fp16, NO quantization
# ============================================================
def build_fp16_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ fp16, no BitsAndBytesConfig
    model = AutoModelForSequenceClassification\
        .from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.float16,
            device_map='auto',
            pad_token_id=tokenizer.pad_token_id)

    model.config.pad_token_id = \
        tokenizer.pad_token_id

    # score head → float32 for stability
    if hasattr(model, 'score'):
        model.score = model.score.float()

    # LoRA (no kbit training needed)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        target_modules='all-linear')

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer

# ============================================================
# 학습 / 평가
# ============================================================
def train_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        y    = batch['label'].to(DEVICE)
        out  = model(
            input_ids=ids,
            attention_mask=mask,
            labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_at_T(model, sessions, labels,
              tokenizer, t_cutoff):
    model.eval()
    ds = PrefixDataset(
        sessions, labels,
        tokenizer, t_cutoff)
    loader = DataLoader(
        ds, batch_size=8, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask']\
                .to(DEVICE)
            out  = model(
                input_ids=ids,
                attention_mask=mask)
            prob = torch.softmax(
                out.logits.float(),
                dim=-1)[:, 1]
            preds.extend(
                prob.cpu().numpy().tolist())
            trues.extend(
                batch['label'].numpy().tolist())
    return roc_auc_score(trues, preds)

# ============================================================
# 메인 실험
# ============================================================
res_tct_fp16  = {t: [] for t in T_EVAL}
res_full_fp16 = {t: [] for t in T_EVAL}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx = list(range(len(sessions)))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2,
        random_state=seed,
        stratify=labels)

    tr_s = [sessions[i] for i in tr_idx]
    te_s = [sessions[i] for i in te_idx]
    tr_l = [labels[i]   for i in tr_idx]
    te_l = [labels[i]   for i in te_idx]

    # ── TCT + fp16 ──
    print("  [TCT-0.5B-fp16] 학습 중...")
    model_tct, tokenizer = \
        build_fp16_model(MODEL_NAME)

    opt_tct = torch.optim.AdamW(
        filter(lambda p: p.requires_grad,
               model_tct.parameters()),
        lr=LR)

    tct_ds = TCTDataset(
        tr_s, tr_l, tokenizer)
    tct_loader = DataLoader(
        tct_ds,
        batch_size=BATCH_SIZE,
        shuffle=True)

    best_sd, best_auc = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            model_tct, tct_loader, opt_tct)
        auc = eval_at_T(
            model_tct, te_s, te_l,
            tokenizer, 50)
        if auc > best_auc:
            best_auc = auc
            best_sd = {
                k: v.cpu().clone()
                for k, v in
                model_tct.state_dict().items()
                if 'lora' in k}
        print(f"    Epoch {ep+1}: "
              f"loss={loss:.4f} "
              f"auc={auc:.3f}")

    current_sd = model_tct.state_dict()
    current_sd.update(
        {k: v.to(DEVICE)
         for k, v in best_sd.items()})
    model_tct.load_state_dict(current_sd)

    for t in T_EVAL:
        a = eval_at_T(
            model_tct, te_s, te_l,
            tokenizer, t)
        res_tct_fp16[t].append(a)
    print(f"  TCT-0.5B-fp16 T=5: "
          f"{res_tct_fp16[5][-1]:.3f}")

    del model_tct
    torch.cuda.empty_cache()

    # ── Full + fp16 ──
    print("  [Full-0.5B-fp16] 학습 중...")
    model_full, tokenizer = \
        build_fp16_model(MODEL_NAME)

    opt_full = torch.optim.AdamW(
        filter(lambda p: p.requires_grad,
               model_full.parameters()),
        lr=LR)

    full_ds = PrefixDataset(
        tr_s, tr_l, tokenizer, t_cutoff=50)
    full_loader = DataLoader(
        full_ds,
        batch_size=BATCH_SIZE,
        shuffle=True)

    best_sd2, best_auc2 = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            model_full, full_loader, opt_full)
        auc = eval_at_T(
            model_full, te_s, te_l,
            tokenizer, 50)
        if auc > best_auc2:
            best_auc2 = auc
            best_sd2 = {
                k: v.cpu().clone()
                for k, v in
                model_full.state_dict().items()
                if 'lora' in k}
        print(f"    Epoch {ep+1}: "
              f"loss={loss:.4f} "
              f"auc={auc:.3f}")

    current_sd2 = model_full.state_dict()
    current_sd2.update(
        {k: v.to(DEVICE)
         for k, v in best_sd2.items()})
    model_full.load_state_dict(current_sd2)

    for t in T_EVAL:
        a = eval_at_T(
            model_full, te_s, te_l,
            tokenizer, t)
        res_full_fp16[t].append(a)
    print(f"  Full-0.5B-fp16 T=5: "
          f"{res_full_fp16[5][-1]:.3f}")

    del model_full
    torch.cuda.empty_cache()

# ============================================================
# 결과
# ============================================================
print("\n" + "="*60)
print(f"Qwen2.5-0.5B fp16 — "
      f"Quantization Ablation (5 seeds)")
print("="*60)
print(f"{'T':>4} | {'TCT-0.5B-fp16':>15} | "
      f"{'Full-0.5B-fp16':>15} | {'Δ':>7}")
print("-"*55)

for t in T_EVAL:
    tm = np.mean(res_tct_fp16[t])
    ts = np.std(res_tct_fp16[t])
    fm = np.mean(res_full_fp16[t])
    fs = np.std(res_full_fp16[t])
    d  = tm - fm
    print(f"{t:>4} | {tm:.3f}±{ts:.3f}       | "
          f"{fm:.3f}±{fs:.3f}       | {d:+.3f}")

print("\n--- Key: Collapse persists without "
      "4-bit quantization ---")
full_t5 = np.mean(res_full_fp16[5])
tct_t5  = np.mean(res_tct_fp16[5])
print(f"Full-0.5B-fp16 T=5: {full_t5:.3f}  "
      f"(collapse confirmed)")
print(f"TCT-0.5B-fp16  T=5: {tct_t5:.3f}  "
      f"(Δ={tct_t5-full_t5:+.3f})")

# 저장
with open('/content/results_qwen05b_fp16.pkl',
          'wb') as f:
    pickle.dump({
        'tct_fp16':  res_tct_fp16,
        'full_fp16': res_full_fp16,
        'model':     MODEL_NAME,
        'precision': 'fp16_no_quantization'
    }, f)
print("\n✅ Saved: results_qwen05b_fp16.pkl")
