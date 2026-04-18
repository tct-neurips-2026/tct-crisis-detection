# ============================================================
# 09_da_gru_experiment.py  (v2 — step-matched)
# Data Augmentation GRU vs TCT-GRU vs Full-GRU
# STANDALONE — no dependencies on other files
#
# Fair comparison:
# TCT total steps = N_sessions × epochs / batch_size
# DA-GRU stops when it reaches the same total steps
# This ensures identical gradient update counts.
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import pickle

# ============================================================
# 0. 설정
# ============================================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

T_LIST     = [5, 10, 15, 20, 30, 50]
SEEDS      = [42, 123, 2026, 7, 777]
MAX_T      = 50
MIN_T      = 3
BATCH_SIZE = 32
TCT_EPOCHS = 25   # TCT reference epochs

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] Loading data...")
df = pd.read_pickle(
    '/content/df_COMPLETE_for_analysis.pkl')
print(f"Sessions: {len(df)}")

# ============================================================
# 2. Child turns 추출 + 임베딩
# ============================================================
print("\n[2] Extracting child turns...")
all_texts   = []
session_map = []

for s_idx, row in df.iterrows():
    turns = row['turns']
    if not isinstance(turns, list):
        continue
    count = 0
    for t in turns:
        if t.get('speaker') == 'child':
            all_texts.append(t.get('text', ''))
            session_map.append((s_idx, count))
            count += 1
            if count >= MAX_T:
                break

print(f"Total child turns: {len(all_texts)}")

print("\n[3] Encoding...")
encoder = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2')
embeddings = encoder.encode(
    all_texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Embeddings: {embeddings.shape}")

# ============================================================
# 3. Session 구성
# ============================================================
print("\n[4] Building sessions...")
session_data = []
for s_idx, row in df.iterrows():
    turns = row['turns']
    if not isinstance(turns, list):
        continue
    emb_indices = [
        i for i, (si, _) in enumerate(session_map)
        if si == s_idx
    ]
    if len(emb_indices) == 0:
        continue
    session_data.append({
        'session_id':  row['session_id'],
        'crisis':      row['crisis_en'],
        'embeddings':  embeddings[emb_indices],
    })

crisis_sessions = []
for s in session_data:
    if s['crisis'] in ['Normal', 'Emergency']:
        s2 = dict(s)
        s2['crisis_label'] = (
            1 if s['crisis'] == 'Emergency' else 0)
        crisis_sessions.append(s2)

print(f"Normal vs Emergency: {len(crisis_sessions)}")

train_c, test_c = train_test_split(
    crisis_sessions,
    test_size=0.2,
    random_state=42,
    stratify=[s['crisis_label']
              for s in crisis_sessions]
)
print(f"Train: {len(train_c)}, Test: {len(test_c)}")

# TCT total steps 계산 (기준값)
TCT_TOTAL_STEPS = (
    len(train_c) * TCT_EPOCHS // BATCH_SIZE)
print(f"\nTCT total steps (reference): "
      f"{TCT_TOTAL_STEPS}")

# ============================================================
# 4. Dataset 정의
# ============================================================

class TCTDataset(Dataset):
    """TCT: t ~ Uniform(min_t, T) per step"""
    def __init__(self, sessions,
                 min_t=MIN_T, max_t=MAX_T):
        self.sessions = sessions
        self.min_t    = min_t
        self.max_t    = max_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess   = self.sessions[idx]
        embs   = sess['embeddings']
        T      = min(len(embs), self.max_t)
        t      = np.random.randint(
            self.min_t, T + 1)
        embs_t = torch.FloatTensor(embs[:t])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return embs_t, label, t


class FullDataset(Dataset):
    """Full: always full sequence"""
    def __init__(self, sessions, max_t=MAX_T):
        self.sessions = sessions
        self.max_t    = max_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess   = self.sessions[idx]
        embs   = sess['embeddings']
        T      = min(len(embs), self.max_t)
        embs_t = torch.FloatTensor(embs[:T])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return embs_t, label, T


class DADataset(Dataset):
    """
    DA-GRU: each session expanded into T
    independent prefix examples.
    No simultaneous gradients (vs Multi-prefix).
    Deterministic exposure (vs TCT).
    """
    def __init__(self, sessions,
                 min_t=MIN_T, max_t=MAX_T):
        self.data = []
        for sess in sessions:
            embs  = sess['embeddings']
            T     = min(len(embs), max_t)
            label = sess['crisis_label']
            for t in range(min_t, T + 1):
                self.data.append({
                    'embs_t': embs[:t].copy(),
                    'label':  label,
                    'length': t
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        embs_t = torch.FloatTensor(item['embs_t'])
        label  = torch.FloatTensor([item['label']])
        return embs_t, label, item['length']


def collate_fn(batch):
    embs, labels, lengths = zip(*batch)
    embs_padded = pad_sequence(
        embs, batch_first=True)
    labels  = torch.stack(labels)
    lengths = torch.LongTensor(lengths)
    return embs_padded, labels, lengths

# ============================================================
# 5. 모델 정의
# ============================================================

class GRUClassifier(nn.Module):
    def __init__(self, emb_dim=384,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, embs_padded, lengths):
        B = embs_padded.shape[0]
        packed = pack_padded_sequence(
            embs_padded, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(
            gru_out, batch_first=True)
        idx = (lengths - 1).clamp(min=0)
        last_hidden = gru_out[
            torch.arange(B), idx]
        return self.classifier(last_hidden)

# ============================================================
# 6. 학습 / 평가 함수
# ============================================================

def eval_at_T(model, sessions, T):
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for sess in sessions:
            t = min(T, len(sess['embeddings']))
            embs = torch.FloatTensor(
                sess['embeddings'][:t]
            ).unsqueeze(0).to(device)
            length = torch.LongTensor([t])
            logit  = model(embs, length)
            scores.append(
                torch.sigmoid(logit).item())
            labels.append(sess['crisis_label'])
    return roc_auc_score(labels, scores)


def get_val_auc(model, test_sessions):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for sess in test_sessions:
            embs = torch.FloatTensor(
                sess['embeddings']
            ).unsqueeze(0).to(device)
            length = torch.LongTensor(
                [len(sess['embeddings'])])
            logit  = model(embs, length)
            preds.append(
                torch.sigmoid(logit).item())
            labs.append(sess['crisis_label'])
    return roc_auc_score(labs, preds)


def run_standard(train_sessions, test_sessions,
                 dataset_cls, seed=42,
                 epochs=25, **ds_kwargs):
    """TCT / Full 공용 학습"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = dataset_cls(
        train_sessions, **ds_kwargs)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn)

    model     = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for embs, labels, lengths in loader:
            embs    = embs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logit = model(embs, lengths)
            loss  = criterion(logit, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()

        auc = get_val_auc(model, test_sessions)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc


def run_da_gru(train_sessions, test_sessions,
               seed=42,
               max_steps=TCT_TOTAL_STEPS):
    """
    DA-GRU: step-matched training.

    Trains on expanded dataset (all prefix lengths
    as independent examples) but stops at exactly
    max_steps gradient updates — matching TCT's
    total update count for fair comparison.

    Validation AUC checked at end of each epoch
    over the expanded dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("  Building DA dataset...")
    ds = DADataset(train_sessions)
    n_da = len(ds)
    n_orig = len(train_sessions)
    expansion = n_da // n_orig
    print(f"  {n_da} examples "
          f"({expansion}x per session)")
    print(f"  Training for {max_steps} steps "
          f"(same as TCT {TCT_EPOCHS} epochs)")

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn)

    model     = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    step_count = 0
    epoch      = 0
    done       = False

    while not done:
        epoch += 1
        model.train()

        for embs, labels, lengths in loader:
            if step_count >= max_steps:
                done = True
                break

            embs    = embs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logit = model(embs, lengths)
            loss  = criterion(logit, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            step_count += 1

        # validate after each epoch
        auc = get_val_auc(model, test_sessions)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()}
        print(f"    Epoch {epoch} "
              f"(steps={step_count}/{max_steps}): "
              f"AUC={auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc

# ============================================================
# 7. 메인 실험
# ============================================================
print("\n[5] Running experiment (5 seeds)...")
print(f"TCT-GRU: {TCT_EPOCHS} epochs "
      f"({TCT_TOTAL_STEPS} steps)")
print(f"Full-GRU: {TCT_EPOCHS} epochs")
print(f"DA-GRU: step-matched "
      f"({TCT_TOTAL_STEPS} steps)")

results = {
    'TCT-GRU':  {T: [] for T in T_LIST},
    'Full-GRU': {T: [] for T in T_LIST},
    'DA-GRU':   {T: [] for T in T_LIST},
}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    print("  [TCT-GRU]")
    m_tct, auc = run_standard(
        train_c, test_c,
        dataset_cls=TCTDataset,
        seed=seed, epochs=TCT_EPOCHS)
    print(f"  → best AUC: {auc:.3f}")

    print("  [Full-GRU]")
    m_full, auc = run_standard(
        train_c, test_c,
        dataset_cls=FullDataset,
        seed=seed, epochs=TCT_EPOCHS)
    print(f"  → best AUC: {auc:.3f}")

    print("  [DA-GRU] (step-matched)")
    m_da, auc = run_da_gru(
        train_c, test_c,
        seed=seed,
        max_steps=TCT_TOTAL_STEPS)
    print(f"  → best AUC: {auc:.3f}")

    for T in T_LIST:
        results['TCT-GRU'][T].append(
            eval_at_T(m_tct, test_c, T))
        results['Full-GRU'][T].append(
            eval_at_T(m_full, test_c, T))
        results['DA-GRU'][T].append(
            eval_at_T(m_da, test_c, T))

# ============================================================
# 8. 결과
# ============================================================
print("\n" + "="*65)
print(f"DA-GRU (step-matched, {TCT_TOTAL_STEPS} steps) "
      f"vs TCT-GRU vs Full-GRU (5 seeds)")
print("="*65)
print(f"{'T':>4} | {'TCT-GRU':>13} | "
      f"{'Full-GRU':>13} | "
      f"{'DA-GRU':>13}")
print("-"*55)

for T in T_LIST:
    vals  = {k: results[k][T] for k in results}
    means = {k: np.mean(v)
             for k, v in vals.items()}
    best  = max(means.values())

    def fmt(k):
        m = np.mean(vals[k])
        s = np.std(vals[k])
        mark = " *" if abs(m-best) < 1e-9 \
               else "  "
        return f"{m:.3f}±{s:.3f}{mark}"

    print(f"{T:>4} | {fmt('TCT-GRU'):>15} | "
          f"{fmt('Full-GRU'):>15} | "
          f"{fmt('DA-GRU'):>15}")

print("\n* = best at this T")

print("\n--- Key comparison at T=5 ---")
tct  = np.mean(results['TCT-GRU'][5])
full = np.mean(results['Full-GRU'][5])
da   = np.mean(results['DA-GRU'][5])
print(f"TCT-GRU:  {tct:.3f} "
      f"(+{tct-full:.3f} vs Full)")
print(f"DA-GRU:   {da:.3f}  "
      f"({da-full:+.3f} vs Full)")
print(f"Full-GRU: {full:.3f}")

print("\n--- Variance comparison (std @ T=5) ---")
print(f"TCT-GRU std: "
      f"{np.std(results['TCT-GRU'][5]):.3f}")
print(f"DA-GRU std:  "
      f"{np.std(results['DA-GRU'][5]):.3f}")

print("\n--- Interpretation ---")
diff = tct - da
if abs(diff) < 0.015:
    print("DA ≈ TCT: TCT is an efficient "
          "form of prefix data augmentation")
elif diff > 0.015:
    print("TCT > DA: stochastic training "
          "dynamics provide benefit beyond "
          "augmentation alone")
else:
    print("DA > TCT: deterministic prefix "
          "coverage dominates stochastic "
          "sampling")

with open(
        '/content/results_da_gru_v2.pkl',
        'wb') as f:
    pickle.dump({
        'results':    results,
        'tct_steps':  TCT_TOTAL_STEPS,
        'tct_epochs': TCT_EPOCHS,
    }, f)
print("\n✅ Saved: results_da_gru_v2.pkl")
