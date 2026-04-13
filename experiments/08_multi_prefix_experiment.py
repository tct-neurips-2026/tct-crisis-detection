# ============================================================
# 08_multi_prefix_experiment.py
# Multi-prefix Loss Baseline vs TCT-GRU vs Full-GRU
# STANDALONE — no dependencies on other files
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

T_LIST = [5, 10, 15, 20, 30, 50]
SEEDS  = [42, 123, 2026, 7, 777]
MAX_T  = 50

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
        'session_id':      row['session_id'],
        'crisis':          row['crisis_en'],
        'embeddings':      embeddings[emb_indices],
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

# ============================================================
# 4. Dataset 정의
# ============================================================

class TCTDataset(Dataset):
    """TCT: t ~ Uniform(min_t, T)"""
    def __init__(self, sessions,
                 min_t=3, max_t=MAX_T):
        self.sessions = sessions
        self.min_t    = min_t
        self.max_t    = max_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions[idx]
        embs = sess['embeddings']
        T    = min(len(embs), self.max_t)
        t    = np.random.randint(self.min_t, T + 1)
        embs_t = torch.FloatTensor(embs[:t])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return embs_t, label, t


class FullDataset(Dataset):
    """Full: 항상 전체 시퀀스"""
    def __init__(self, sessions, max_t=MAX_T):
        self.sessions = sessions
        self.max_t    = max_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions[idx]
        embs = sess['embeddings']
        T    = min(len(embs), self.max_t)
        embs_t = torch.FloatTensor(embs[:T])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return embs_t, label, T


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
# 6. 학습 함수
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


def run_standard(train_sessions, test_sessions,
                 dataset_cls, seed=42,
                 epochs=25, **ds_kwargs):
    """TCT / Full 공용 학습"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = dataset_cls(
        train_sessions, **ds_kwargs)
    loader = DataLoader(
        ds, batch_size=32,
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

        auc = roc_auc_score(labs, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc


def run_multi_prefix(train_sessions, test_sessions,
                     seed=42, epochs=25):
    """
    Multi-prefix baseline:
    L = (1/T) * Σ_{t=1}^{T} ℓ(f(x₁:t), y)

    핵심 차이:
    - TCT: 매 스텝 t 하나 샘플링
    - Multi-prefix: 모든 t에 동시 supervision
      → 훨씬 비싸고 over-constrained
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(
            len(train_sessions))

        # 배치 단위 처리
        for i in range(
                0, len(indices), 32):
            batch_idx = indices[i:i+32]
            total_loss = torch.zeros(
                1, requires_grad=False
            ).to(device)
            count = 0

            for idx in batch_idx:
                sess  = train_sessions[idx]
                embs  = sess['embeddings']
                T     = min(
                    len(embs), MAX_T)
                label = torch.FloatTensor(
                    [sess['crisis_label']]
                ).to(device)

                # 모든 prefix t=1..T에 대해
                sess_loss = torch.zeros(
                    1).to(device)
                for t in range(1, T + 1):
                    emb_t = torch.FloatTensor(
                        embs[:t]
                    ).unsqueeze(0).to(device)
                    length = torch.LongTensor(
                        [t])
                    logit  = model(
                        emb_t, length).squeeze()
                    sess_loss = sess_loss + \
                        criterion(
                            logit.unsqueeze(0),
                            label)

                total_loss = total_loss + \
                    sess_loss / T
                count += 1

            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                optimizer.step()

        # 검증
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

        auc = roc_auc_score(labs, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:2d}: "
                  f"AUC={auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc

# ============================================================
# 7. 메인 실험
# ============================================================
print("\n[5] Running experiment (5 seeds)...")
print("Models: TCT-GRU | Full-GRU | Multi-prefix")

results = {
    'TCT-GRU':      {T: [] for T in T_LIST},
    'Full-GRU':     {T: [] for T in T_LIST},
    'Multi-prefix': {T: [] for T in T_LIST},
}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    # TCT-GRU
    print("  [TCT-GRU]")
    m_tct, auc = run_standard(
        train_c, test_c,
        dataset_cls=TCTDataset,
        seed=seed, epochs=25)
    print(f"  → best AUC: {auc:.3f}")

    # Full-GRU
    print("  [Full-GRU]")
    m_full, auc = run_standard(
        train_c, test_c,
        dataset_cls=FullDataset,
        seed=seed, epochs=25)
    print(f"  → best AUC: {auc:.3f}")

    # Multi-prefix
    print("  [Multi-prefix]")
    m_multi, auc = run_multi_prefix(
        train_c, test_c,
        seed=seed, epochs=25)
    print(f"  → best AUC: {auc:.3f}")

    for T in T_LIST:
        results['TCT-GRU'][T].append(
            eval_at_T(m_tct, test_c, T))
        results['Full-GRU'][T].append(
            eval_at_T(m_full, test_c, T))
        results['Multi-prefix'][T].append(
            eval_at_T(m_multi, test_c, T))

# ============================================================
# 8. 결과
# ============================================================
print("\n" + "="*70)
print("Multi-prefix vs TCT-GRU vs Full-GRU (5 seeds)")
print("="*70)
print(f"{'T':>4} | {'TCT-GRU':>13} | "
      f"{'Full-GRU':>13} | "
      f"{'Multi-prefix':>13}")
print("-"*60)

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
          f"{fmt('Multi-prefix'):>15}")

print("\n* = best at this T")

# 핵심 비교
print("\n--- Key: T=5 ---")
tct   = np.mean(results['TCT-GRU'][5])
full  = np.mean(results['Full-GRU'][5])
multi = np.mean(results['Multi-prefix'][5])
print(f"TCT-GRU:      {tct:.3f}  (+{tct-full:.3f} vs Full)")
print(f"Multi-prefix: {multi:.3f}  ({multi-full:+.3f} vs Full)")
print(f"Full-GRU:     {full:.3f}")

print("\n--- Variance comparison (std @ T=5) ---")
print(f"TCT-GRU std:      {np.std(results['TCT-GRU'][5]):.3f}")
print(f"Multi-prefix std: {np.std(results['Multi-prefix'][5]):.3f}")
print(f"Full-GRU std:     {np.std(results['Full-GRU'][5]):.3f}")

with open(
        '/content/results_multi_prefix.pkl',
        'wb') as f:
    pickle.dump(results, f)
print("\n✅ Saved: results_multi_prefix.pkl")
