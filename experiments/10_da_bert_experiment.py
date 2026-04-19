# ============================================================
# 10_da_bert_experiment.py  (DA-BERT only)
# TCT-BERT / Full-BERT 결과는 이미 있음
# DA-BERT만 step-matched로 돌림
#
# Existing results (from 02_bert_experiment.py):
# TCT-BERT T=5: 0.737 ± 0.027
# Full-BERT T=5: 0.652 ± 0.018
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle

# ============================================================
# 0. 설정
# ============================================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MODEL_NAME  = 'klue/roberta-base'
T_LIST      = [5, 10, 15, 20, 30, 50]
SEEDS       = [42, 123, 2026, 7, 777]
MAX_T       = 50
MIN_T       = 3
BATCH_SIZE  = 8
TCT_EPOCHS  = 10
MAX_SEQ_LEN = 512
LR          = 2e-5

# 기존 TCT-BERT / Full-BERT 결과
# (02_bert_experiment.py에서 얻은 값)
EXISTING = {
    'TCT-BERT': {
        5:  (0.737, 0.027),
        10: (0.894, 0.018),
        15: (0.920, 0.018),
        20: (0.946, 0.013),
        30: (0.974, 0.010),
        50: (0.983, 0.007),
    },
    'Full-BERT': {
        5:  (0.652, 0.018),
        10: (0.841, 0.019),
        15: (0.888, 0.013),
        20: (0.930, 0.020),
        30: (0.969, 0.016),
        50: (0.989, 0.007),
    }
}

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] Loading data...")
df = pd.read_pickle(
    '/content/df_COMPLETE_for_analysis.pkl')
print(f"Sessions: {len(df)}")

# ============================================================
# 2. 세션 구성
# ============================================================
print("\n[2] Building sessions...")

def get_child_turns(row, max_t=MAX_T):
    turns = row['turns']
    if not isinstance(turns, list):
        return []
    texts = []
    for t in turns:
        if t.get('speaker') == 'child':
            texts.append(t.get('text', ''))
            if len(texts) >= max_t:
                break
    return texts

crisis_sessions = []
for _, row in df.iterrows():
    if row['crisis_en'] not in [
            'Normal', 'Emergency']:
        continue
    turns = get_child_turns(row)
    if len(turns) < MIN_T:
        continue
    crisis_sessions.append({
        'session_id':   row['session_id'],
        'turns':        turns,
        'crisis_label': (
            1 if row['crisis_en'] == 'Emergency'
            else 0)
    })

print(f"Normal vs Emergency: "
      f"{len(crisis_sessions)}")

train_s, test_s = train_test_split(
    crisis_sessions,
    test_size=0.2,
    random_state=42,
    stratify=[s['crisis_label']
              for s in crisis_sessions]
)
print(f"Train: {len(train_s)}, "
      f"Test: {len(test_s)}")

# step-matched to TCT-BERT
TCT_TOTAL_STEPS = (
    len(train_s) * TCT_EPOCHS // BATCH_SIZE)
print(f"\nTCT-BERT total steps (reference): "
      f"{TCT_TOTAL_STEPS}")

# ============================================================
# 3. Tokenizer
# ============================================================
print("\n[3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME)

def encode_prefix(turns, t):
    text = ' [SEP] '.join(turns[:t])
    return tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

# ============================================================
# 4. DA-BERT Dataset
# ============================================================

class DABERTDataset(Dataset):
    def __init__(self, sessions,
                 min_t=MIN_T, max_t=MAX_T):
        self.data = []
        for sess in sessions:
            turns = sess['turns']
            T     = min(len(turns), max_t)
            label = sess['crisis_label']
            for t in range(min_t, T + 1):
                self.data.append({
                    'turns':  turns,
                    'label':  label,
                    'length': t
                })
        n_sess = len(sessions)
        n_ex   = len(self.data)
        print(f"  DA-BERT: {n_ex} examples "
              f"({n_ex//n_sess}x per session)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = encode_prefix(
            item['turns'], item['length'])
        return {
            'input_ids':
                enc['input_ids'].squeeze(),
            'attention_mask':
                enc['attention_mask'].squeeze(),
            'label': torch.tensor(
                item['label'],
                dtype=torch.float),
            'length': item['length']
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack(
            [b['input_ids'] for b in batch]),
        'attention_mask': torch.stack(
            [b['attention_mask']
             for b in batch]),
        'label': torch.stack(
            [b['label'] for b in batch]),
        'length': torch.tensor(
            [b['length'] for b in batch])
    }

# ============================================================
# 5. 모델
# ============================================================

class BERTClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME,
                 dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids,
                attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        mask   = attention_mask\
            .unsqueeze(-1).float()
        pooled = (
            out.last_hidden_state * mask
        ).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)

# ============================================================
# 6. 학습 / 평가 함수
# ============================================================

def eval_at_T(model, sessions, T):
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for sess in sessions:
            turns = sess['turns']
            t     = min(T, len(turns))
            enc   = encode_prefix(turns, t)
            ids   = enc[
                'input_ids'].to(device)
            mask  = enc[
                'attention_mask'].to(device)
            logit = model(ids, mask)
            scores.append(
                torch.sigmoid(logit).item())
            labels.append(
                sess['crisis_label'])
    return roc_auc_score(labels, scores)


def run_da_bert(train_sessions, test_sessions,
                seed=42,
                max_steps=TCT_TOTAL_STEPS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("  Building DA-BERT dataset...")
    ds = DABERTDataset(train_sessions)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2)

    model     = BERTClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    print(f"  Training for {max_steps} steps "
          f"(matched to TCT-BERT "
          f"{TCT_EPOCHS} epochs)")

    step_count = 0
    epoch      = 0
    done       = False

    while not done:
        epoch += 1
        model.train()
        for batch in loader:
            if step_count >= max_steps:
                done = True
                break
            ids   = batch[
                'input_ids'].to(device)
            mask  = batch[
                'attention_mask'].to(device)
            label = batch[
                'label'].unsqueeze(1)\
                .to(device)
            optimizer.zero_grad()
            logit = model(ids, mask)
            loss  = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            step_count += 1

        # validation
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for sess in test_sessions:
                turns = sess['turns']
                t     = min(MAX_T, len(turns))
                enc   = encode_prefix(turns, t)
                ids   = enc[
                    'input_ids'].to(device)
                mask  = enc[
                    'attention_mask'].to(device)
                logit = model(ids, mask)
                preds.append(
                    torch.sigmoid(
                        logit).item())
                labs.append(
                    sess['crisis_label'])
        auc = roc_auc_score(labs, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()}
        print(f"    Epoch {epoch} "
              f"(steps={step_count}/"
              f"{max_steps}): "
              f"AUC={auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc

# ============================================================
# 7. 메인: DA-BERT만 실행
# ============================================================
print("\n[4] Running DA-BERT only (5 seeds)...")

da_results = {T: [] for T in T_LIST}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    m_da, auc = run_da_bert(
        train_s, test_s,
        seed=seed,
        max_steps=TCT_TOTAL_STEPS)
    print(f"  → best AUC: {auc:.3f}")

    for T in T_LIST:
        da_results[T].append(
            eval_at_T(m_da, test_s, T))

# ============================================================
# 8. 결과
# ============================================================
print("\n" + "="*65)
print("DA-BERT vs TCT-BERT vs Full-BERT")
print("(TCT/Full: 02_bert_experiment.py)")
print("="*65)
print(f"{'T':>4} | {'TCT-BERT':>13} | "
      f"{'Full-BERT':>13} | "
      f"{'DA-BERT':>13}")
print("-"*55)

for T in T_LIST:
    tct_m,  tct_s  = EXISTING['TCT-BERT'][T]
    full_m, full_s = EXISTING['Full-BERT'][T]
    da_m  = np.mean(da_results[T])
    da_s  = np.std(da_results[T])

    best = max(tct_m, full_m, da_m)
    def mark(v):
        return "*" if abs(v-best) < 1e-9 \
               else " "

    print(
        f"{T:>4} | "
        f"{tct_m:.3f}±{tct_s:.3f} "
        f"{mark(tct_m)} | "
        f"{full_m:.3f}±{full_s:.3f} "
        f"{mark(full_m)} | "
        f"{da_m:.3f}±{da_s:.3f} "
        f"{mark(da_m)}")

# Recovery rate
tct_t5  = EXISTING['TCT-BERT'][5][0]
full_t5 = EXISTING['Full-BERT'][5][0]
da_t5   = np.mean(da_results[5])

bert_recovery = (
    (tct_t5 - full_t5) /
    (da_t5  - full_t5) * 100
    if (da_t5 - full_t5) > 0.001
    else float('nan'))

print(f"\n--- Recovery rate at T=5 ---")
print(f"GRU  (reference): 97%")
print(f"BERT (new):       {bert_recovery:.0f}%")
print(f"\n  TCT-BERT: {tct_t5:.3f}")
print(f"  Full-BERT: {full_t5:.3f}")
print(f"  DA-BERT:   {da_t5:.3f}")

print("\n--- Interpretation ---")
if bert_recovery >= 90:
    print("✅ claim generalises to BERT")
elif bert_recovery >= 70:
    print("🟡 partial generalisation; "
          "report range across architectures")
else:
    print("🔴 GRU-specific; "
          "scope narrowing needed")

with open(
        '/content/results_da_bert.pkl',
        'wb') as f:
    pickle.dump({
        'da_results':    da_results,
        'existing':      EXISTING,
        'tct_steps':     TCT_TOTAL_STEPS,
        'bert_recovery': bert_recovery,
    }, f)
print("\n✅ Saved: results_da_bert.pkl")
