# ============================================================
# 03_lora_experiment.py
# TCT-LoRA vs Full-LoRA (klue/roberta-base + LoRA r=8)
# 5 seeds, prefix-AUC evaluation
# ============================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import pickle

SEEDS      = [42, 123, 2026, 7, 777]
T_EVAL     = [5, 10, 15, 20, 30, 50]
BATCH_SIZE = 8
EPOCHS     = 10
LR         = 2e-4
MAX_SEQ    = 512
DEVICE     = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
MODEL_NAME    = 'klue/roberta-base'
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
LORA_TARGETS  = ['query', 'value']

print(f"Device: {DEVICE}")

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
            text, max_length=MAX_SEQ,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        return {
            'input_ids':
                enc['input_ids'].squeeze(0),
            'attention_mask':
                enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                label, dtype=torch.float32)}


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
        t    = random.randint(
            1, max(1, len(turns)))
        text = ' '.join(turns[:t])
        enc  = self.tokenizer(
            text, max_length=MAX_SEQ,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        return {
            'input_ids':
                enc['input_ids'].squeeze(0),
            'attention_mask':
                enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                int(label), dtype=torch.float32)}

# ============================================================
# 모델
# ============================================================
class LoRAClassifier(nn.Module):
    def __init__(self, model_name,
                 lora_r=8, lora_alpha=16,
                 lora_dropout=0.1,
                 target_modules=None):
        super().__init__()
        if target_modules is None:
            target_modules = ['query', 'value']
        base = AutoModel.from_pretrained(
            model_name, add_pooling_layer=False)
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none')
        self.bert = get_peft_model(base, lora_cfg)
        hidden = base.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        h      = out.last_hidden_state
        mask   = attention_mask\
            .unsqueeze(-1).float()
        pooled = ((h * mask).sum(1)
                  / mask.sum(1).clamp(min=1e-9))
        return self.classifier(pooled).squeeze(-1)

    def print_trainable(self):
        self.bert.print_trainable_parameters()

# ============================================================
# 학습 / 평가
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        y    = batch['label'].to(DEVICE)
        pred = model(ids, mask)
        loss = criterion(pred, y)
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
        sessions, labels, tokenizer, t_cutoff)
    loader = DataLoader(
        ds, batch_size=16, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            pred = model(ids, mask).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(
                batch['label'].numpy().tolist())
    return roc_auc_score(trues, preds)

# ============================================================
# 메인 실험
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n=== LoRA trainable parameters ===")
_m = LoRAClassifier(MODEL_NAME).to(DEVICE)
_m.print_trainable()
del _m
torch.cuda.empty_cache()

res_tct  = {t: [] for t in T_EVAL}
res_full = {t: [] for t in T_EVAL}

for seed in SEEDS:
    print(f"\nSeed {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx = list(range(len(sessions)))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2,
        random_state=seed, stratify=labels)
    tr_s = [sessions[i] for i in tr_idx]
    te_s = [sessions[i] for i in te_idx]
    tr_l = [labels[i]   for i in tr_idx]
    te_l = [labels[i]   for i in te_idx]
    criterion = nn.BCELoss()

    # TCT-LoRA
    print("  TCT-LoRA...")
    m_tct = LoRAClassifier(
        MODEL_NAME, lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS).to(DEVICE)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad,
               m_tct.parameters()), lr=LR)
    loader = DataLoader(
        TCTDataset(tr_s, tr_l, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True)
    best_sd, best_auc = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            m_tct, loader, opt, criterion)
        auc = eval_at_T(
            m_tct, te_s, te_l, tokenizer, 50)
        if auc > best_auc:
            best_auc = auc
            best_sd  = {k: v.cpu().clone()
                        for k, v in
                        m_tct.state_dict().items()}
        if (ep+1) % 2 == 0:
            print(f"    Ep{ep+1}: "
                  f"loss={loss:.4f} auc={auc:.3f}")
    m_tct.load_state_dict(
        {k: v.to(DEVICE) for k, v in best_sd.items()})
    for t in T_EVAL:
        res_tct[t].append(
            eval_at_T(m_tct, te_s, te_l, tokenizer, t))
    print(f"  TCT-LoRA T=5: {res_tct[5][-1]:.3f}")
    del m_tct; torch.cuda.empty_cache()

    # Full-LoRA
    print("  Full-LoRA...")
    m_full = LoRAClassifier(
        MODEL_NAME, lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS).to(DEVICE)
    opt2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad,
               m_full.parameters()), lr=LR)
    loader2 = DataLoader(
        PrefixDataset(tr_s, tr_l, tokenizer, 50),
        batch_size=BATCH_SIZE, shuffle=True)
    best_sd2, best_auc2 = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            m_full, loader2, opt2, criterion)
        auc = eval_at_T(
            m_full, te_s, te_l, tokenizer, 50)
        if auc > best_auc2:
            best_auc2 = auc
            best_sd2  = {k: v.cpu().clone()
                         for k, v in
                         m_full.state_dict().items()}
        if (ep+1) % 2 == 0:
            print(f"    Ep{ep+1}: "
                  f"loss={loss:.4f} auc={auc:.3f}")
    m_full.load_state_dict(
        {k: v.to(DEVICE)
         for k, v in best_sd2.items()})
    for t in T_EVAL:
        res_full[t].append(
            eval_at_T(m_full, te_s, te_l, tokenizer, t))
    print(f"  Full-LoRA T=5: {res_full[5][-1]:.3f}")
    del m_full; torch.cuda.empty_cache()

# ============================================================
# 결과
# ============================================================
print("\n" + "="*55)
print("TCT-LoRA vs Full-LoRA (5 seeds)")
print("="*55)
print(f"{'T':>4} | {'TCT-LoRA':>13} | "
      f"{'Full-LoRA':>13} | {'Δ':>7}")
print("-"*50)
for t in T_EVAL:
    tm = np.mean(res_tct[t])
    ts = np.std(res_tct[t])
    fm = np.mean(res_full[t])
    fs = np.std(res_full[t])
    print(f"{t:>4} | {tm:.3f}±{ts:.3f}  | "
          f"{fm:.3f}±{fs:.3f}  | {tm-fm:+.3f}")

with open('/content/results_lora.pkl', 'wb') as f:
    pickle.dump(
        {'tct': res_tct, 'full': res_full}, f)
print("\n✅ Saved: results_lora.pkl")
