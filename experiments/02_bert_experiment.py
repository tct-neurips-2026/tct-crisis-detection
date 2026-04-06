# ============================================================
# 02_bert_experiment.py
# TCT-BERT vs Full-BERT (klue/roberta-base)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import pickle

SEEDS      = [42, 123, 2026, 7, 777]
T_EVAL     = [5, 10, 15, 20, 30, 50]
BATCH_SIZE = 8
EPOCHS     = 10
LR         = 2e-5
MAX_SEQ    = 512
DEVICE     = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
MODEL_NAME = 'klue/roberta-base'
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
                label, dtype=torch.float32)
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
        t    = random.randint(
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
                int(label), dtype=torch.float32)
        }

# ============================================================
# 모델
# ============================================================
class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids,
                attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        h    = out.last_hidden_state
        mask = attention_mask\
            .unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / \
            mask.sum(1).clamp(min=1e-9)
        return self.classifier(
            pooled).squeeze(-1)

# ============================================================
# 학습 / 평가
# ============================================================
def train_epoch(model, loader,
                optimizer, criterion):
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
            mask = batch['attention_mask']\
                .to(DEVICE)
            pred = model(ids, mask)\
                .cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(
                batch['label']
                .numpy().tolist())
    return roc_auc_score(trues, preds)

# ============================================================
# 메인 실험
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME)

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
        random_state=seed,
        stratify=labels)

    tr_s = [sessions[i] for i in tr_idx]
    te_s = [sessions[i] for i in te_idx]
    tr_l = [labels[i]   for i in tr_idx]
    te_l = [labels[i]   for i in te_idx]

    criterion = nn.BCELoss()

    # TCT-BERT
    print("  TCT-BERT...")
    model_tct = BERTClassifier(
        MODEL_NAME).to(DEVICE)
    opt = torch.optim.AdamW(
        model_tct.parameters(), lr=LR)
    tct_ds = TCTDataset(
        tr_s, tr_l, tokenizer)
    tct_loader = DataLoader(
        tct_ds, batch_size=BATCH_SIZE,
        shuffle=True)
    best_sd, best_auc = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            model_tct, tct_loader,
            opt, criterion)
        auc = eval_at_T(
            model_tct, te_s, te_l,
            tokenizer, 50)
        if auc > best_auc:
            best_auc = auc
            best_sd  = {
                k: v.cpu().clone()
                for k, v in
                model_tct.state_dict().items()}
        if (ep + 1) % 2 == 0:
            print(f"    Epoch {ep+1}: "
                  f"loss={loss:.4f} "
                  f"auc={auc:.3f}")

    model_tct.load_state_dict(
        {k: v.to(DEVICE)
         for k, v in best_sd.items()})
    for t in T_EVAL:
        res_tct[t].append(
            eval_at_T(model_tct, te_s, te_l,
                      tokenizer, t))
    print(f"  TCT-BERT T=5: "
          f"{res_tct[5][-1]:.3f}")

    # Full-BERT
    print("  Full-BERT...")
    model_full = BERTClassifier(
        MODEL_NAME).to(DEVICE)
    opt2 = torch.optim.AdamW(
        model_full.parameters(), lr=LR)
    full_ds = PrefixDataset(
        tr_s, tr_l, tokenizer, t_cutoff=50)
    full_loader = DataLoader(
        full_ds, batch_size=BATCH_SIZE,
        shuffle=True)
    best_sd2, best_auc2 = None, 0
    for ep in range(EPOCHS):
        loss = train_epoch(
            model_full, full_loader,
            opt2, criterion)
        auc = eval_at_T(
            model_full, te_s, te_l,
            tokenizer, 50)
        if auc > best_auc2:
            best_auc2 = auc
            best_sd2  = {
                k: v.cpu().clone()
                for k, v in
                model_full.state_dict().items()}
        if (ep + 1) % 2 == 0:
            print(f"    Epoch {ep+1}: "
                  f"loss={loss:.4f} "
                  f"auc={auc:.3f}")

    model_full.load_state_dict(
        {k: v.to(DEVICE)
         for k, v in best_sd2.items()})
    for t in T_EVAL:
        res_full[t].append(
            eval_at_T(model_full, te_s, te_l,
                      tokenizer, t))
    print(f"  Full-BERT T=5: "
          f"{res_full[5][-1]:.3f}")

# ============================================================
# 결과
# ============================================================
print("\n" + "="*55)
print("TCT-BERT vs Full-BERT (5 seeds)")
print("="*55)
print(f"{'T':>4} | {'TCT-BERT':>13} | "
      f"{'Full-BERT':>13} | {'Δ':>7}")
print("-"*50)
for t in T_EVAL:
    tm = np.mean(res_tct[t])
    ts = np.std(res_tct[t])
    fm = np.mean(res_full[t])
    fs = np.std(res_full[t])
    d  = tm - fm
    print(f"{t:>4} | {tm:.3f}±{ts:.3f}  | "
          f"{fm:.3f}±{fs:.3f}  | {d:+.3f}")

with open('/content/results_bert.pkl', 'wb') as f:
    pickle.dump(
        {'tct': res_tct, 'full': res_full}, f)
print("\n✅ Saved: results_bert.pkl")
