# ============================================================
# 01_gru_experiment.py
# Main experiment: TCT-GRU, Full-GRU, TCT-TRAM,
#                  Mean-pool, Last-turn
# 5 seeds, prefix-AUC evaluation
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

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] Loading data...")
df = pd.read_pickle(
    '/content/df_COMPLETE_for_analysis.pkl')
print(f"Sessions: {len(df)}, "
      f"columns: {df.columns.tolist()}")

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
            if count >= 50:
                break

print(f"Total child turns: {len(all_texts)}")

print("\n[3] Encoding utterances...")
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
# 3. Session data 구성
# ============================================================
print("\n[4] Building session data...")

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
    sess_embs = embeddings[emb_indices]
    session_data.append({
        'session_id':      row['session_id'],
        'crisis':          row['crisis_en'],
        'has_risk':        int(row['has_risk']),
        'first_risk_turn': row['first_risk_turn'],
        'embeddings':      sess_embs,
    })

print(f"Total sessions: {len(session_data)}")

crisis_sessions = []
for s in session_data:
    if s['crisis'] in ['Normal', 'Emergency']:
        s2 = dict(s)
        s2['crisis_label'] = (
            1 if s['crisis'] == 'Emergency' else 0)
        crisis_sessions.append(s2)

print(f"Normal vs Emergency: {len(crisis_sessions)}")
print(f"  Normal:    "
      f"{sum(1 for s in crisis_sessions if s['crisis']=='Normal')}")
print(f"  Emergency: "
      f"{sum(1 for s in crisis_sessions if s['crisis']=='Emergency')}")

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

class TruncatedCrisisDataset(Dataset):
    """TCT: t ~ Uniform(min_t, T)"""
    def __init__(self, sessions,
                 random_trunc=True,
                 min_t=3, max_t=50):
        self.sessions     = sessions
        self.random_trunc = random_trunc
        self.min_t        = min_t
        self.max_t        = max_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions[idx]
        embs = sess['embeddings']
        T    = min(len(embs), self.max_t)
        t    = (np.random.randint(self.min_t, T + 1)
                if self.random_trunc else T)
        embs_t = torch.FloatTensor(embs[:t])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return (embs_t, label, t,
                sess.get('first_risk_turn') or 0)


def collate_fn(batch):
    embs, labels, lengths, risks = zip(*batch)
    embs_padded = pad_sequence(
        embs, batch_first=True)
    labels  = torch.stack(labels)
    lengths = torch.LongTensor(lengths)
    return embs_padded, labels, lengths, risks

# ============================================================
# 5. 모델 정의
# ============================================================

class GRUClassifier(nn.Module):
    """TCT-GRU / Full-GRU"""
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


class TRAMModel(nn.Module):
    """TCT-TRAM: explicit risk accumulation"""
    def __init__(self, emb_dim=384,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.log_alpha  = nn.Parameter(
            torch.tensor(0.0))
        self.classifier = nn.Linear(1, 1)

    def forward(self, embs_padded, lengths):
        B, T_max, _ = embs_padded.shape
        packed = pack_padded_sequence(
            embs_padded, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(
            gru_out, batch_first=True)
        r_t   = self.risk_scorer(
            gru_out).squeeze(-1)
        alpha = torch.sigmoid(self.log_alpha)
        s_t   = torch.zeros(
            B, device=embs_padded.device)
        for t in range(T_max):
            s_t = alpha * s_t + r_t[:, t]
        return self.classifier(s_t.unsqueeze(1))

# ============================================================
# 6. 학습 / 평가 함수
# ============================================================

def run_experiment(train_sessions, test_sessions,
                   model_type='gru',
                   truncated=True,
                   seed=42, epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        TruncatedCrisisDataset(
            train_sessions,
            random_trunc=truncated),
        batch_size=32, shuffle=True,
        collate_fn=collate_fn)

    model = (GRUClassifier().to(device)
             if model_type == 'gru'
             else TRAMModel().to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for embs, labels, lengths, _ in loader:
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


def eval_mean_pool(train_sessions,
                   test_sessions, T, seed=42):
    X_tr = np.array([
        s['embeddings'][
            :min(T, len(s['embeddings']))
        ].mean(axis=0)
        for s in train_sessions])
    y_tr = np.array([
        s['crisis_label']
        for s in train_sessions])
    X_te = np.array([
        s['embeddings'][
            :min(T, len(s['embeddings']))
        ].mean(axis=0)
        for s in test_sessions])
    y_te = np.array([
        s['crisis_label']
        for s in test_sessions])
    scaler = StandardScaler()
    lr = LogisticRegression(
        max_iter=500,
        random_state=seed,
        class_weight='balanced')
    lr.fit(scaler.fit_transform(X_tr), y_tr)
    sc = lr.predict_proba(
        scaler.transform(X_te))[:, 1]
    return roc_auc_score(y_te, sc)


def eval_last_turn(train_sessions,
                   test_sessions, T, seed=42):
    X_tr = np.array([
        s['embeddings'][
            min(T, len(s['embeddings'])) - 1]
        for s in train_sessions])
    y_tr = np.array([
        s['crisis_label']
        for s in train_sessions])
    X_te = np.array([
        s['embeddings'][
            min(T, len(s['embeddings'])) - 1]
        for s in test_sessions])
    y_te = np.array([
        s['crisis_label']
        for s in test_sessions])
    scaler = StandardScaler()
    lr = LogisticRegression(
        max_iter=500,
        random_state=seed,
        class_weight='balanced')
    lr.fit(scaler.fit_transform(X_tr), y_tr)
    sc = lr.predict_proba(
        scaler.transform(X_te))[:, 1]
    return roc_auc_score(y_te, sc)

# ============================================================
# 7. 메인 실험
# ============================================================
print("\n[5] Running main experiment "
      "(5 models × 5 seeds)...")

results = {
    'TCT-GRU':   {T: [] for T in T_LIST},
    'TCT-TRAM':  {T: [] for T in T_LIST},
    'Full-GRU':  {T: [] for T in T_LIST},
    'Mean-pool': {T: [] for T in T_LIST},
    'Last-turn': {T: [] for T in T_LIST},
}

for seed in SEEDS:
    print(f"\nSeed {seed}...")

    print("  TCT-GRU...")
    m_tct_gru, auc = run_experiment(
        train_c, test_c,
        model_type='gru',
        truncated=True,
        seed=seed, epochs=25)
    print(f"    best AUC: {auc:.3f}")

    print("  TCT-TRAM...")
    m_tct_tram, auc = run_experiment(
        train_c, test_c,
        model_type='tram',
        truncated=True,
        seed=seed, epochs=25)
    print(f"    best AUC: {auc:.3f}")

    print("  Full-GRU...")
    m_full_gru, auc = run_experiment(
        train_c, test_c,
        model_type='gru',
        truncated=False,
        seed=seed, epochs=25)
    print(f"    best AUC: {auc:.3f}")

    for T in T_LIST:
        results['TCT-GRU'][T].append(
            eval_at_T(m_tct_gru, test_c, T))
        results['TCT-TRAM'][T].append(
            eval_at_T(m_tct_tram, test_c, T))
        results['Full-GRU'][T].append(
            eval_at_T(m_full_gru, test_c, T))
        results['Mean-pool'][T].append(
            eval_mean_pool(
                train_c, test_c, T, seed))
        results['Last-turn'][T].append(
            eval_last_turn(
                train_c, test_c, T, seed))

# ============================================================
# 8. 결과
# ============================================================
print("\n" + "="*90)
print("Table 1: AUC (mean±std) — "
      "Normal vs Emergency, 5 seeds")
print("="*90)
print(f"{'T':>5} | {'TCT-GRU':>14} | "
      f"{'Full-GRU':>14} | "
      f"{'Mean-pool':>14} | "
      f"{'Last-turn':>14} | "
      f"{'Δ_TCT':>7} | "
      f"{'TCT-TRAM':>14}")
print("-"*90)

for T in T_LIST:
    vals  = {k: results[k][T] for k in results}
    means = {k: np.mean(v)
             for k, v in vals.items()}
    best  = max(means.values())

    def fmt(k):
        v = vals[k]
        s = f"{np.mean(v):.3f}±{np.std(v):.3f}"
        return s + "*" \
            if abs(np.mean(v) - best) < 1e-9 \
            else s

    delta = means['TCT-GRU'] - means['Full-GRU']
    tram_mark = "†" \
        if means['TCT-TRAM'] < 0.6 else ""

    print(f"{T:>5} | {fmt('TCT-GRU'):>14} | "
          f"{fmt('Full-GRU'):>14} | "
          f"{fmt('Mean-pool'):>14} | "
          f"{fmt('Last-turn'):>14} | "
          f"{delta:>+.3f}  | "
          f"{fmt('TCT-TRAM'):>14}"
          f"{tram_mark}")

print("\n* = best at this T")
print("† = degenerate solution")

with open('/content/results_gru.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n✅ Saved: results_gru.pkl")
