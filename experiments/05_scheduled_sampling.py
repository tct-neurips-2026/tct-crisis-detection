# ============================================================
# Scheduled Sampling vs TCT-GRU vs Full-GRU
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
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
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

# Normal vs Emergency only
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

class TCTDataset(Dataset):
    """TCT: t ~ Uniform(min_t, T) 랜덤 샘플링"""
    def __init__(self, sessions,
                 min_t=3, max_t=50):
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
        return (embs_t, label, t,
                sess.get('first_risk_turn') or 0)


class FullDataset(Dataset):
    """Full: 항상 전체 시퀀스"""
    def __init__(self, sessions, max_t=50):
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
        return (embs_t, label, T,
                sess.get('first_risk_turn') or 0)


class ScheduledSamplingDataset(Dataset):
    """
    Scheduled Sampling:
    epsilon=1.0 → TCT처럼 완전 랜덤
    epsilon=0.0 → Full처럼 항상 전체
    epsilon은 epoch마다 set_epsilon()으로 업데이트
    """
    def __init__(self, sessions,
                 epsilon=1.0,
                 min_t=3, max_t=50):
        self.sessions = sessions
        self.epsilon  = epsilon
        self.min_t    = min_t
        self.max_t    = max_t

    def set_epsilon(self, epsilon):
        self.epsilon = max(0.0, min(1.0, epsilon))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions[idx]
        embs = sess['embeddings']
        T    = min(len(embs), self.max_t)

        # epsilon 확률로 랜덤, 나머지는 전체
        if np.random.random() < self.epsilon:
            t = np.random.randint(self.min_t, T + 1)
        else:
            t = T

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
    """TCT-GRU / Full-GRU 공용"""
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
        packed     = pack_padded_sequence(
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

def train_standard(train_sessions, test_sessions,
                   dataset_cls, seed=42,
                   epochs=25, **ds_kwargs):
    """TCT / Full 공용 학습"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = dataset_cls(train_sessions, **ds_kwargs)
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

        # 검증 (full sequence)
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for sess in test_sessions:
                embs = torch.FloatTensor(
                    sess['embeddings']
                ).unsqueeze(0).to(device)
                length = torch.LongTensor(
                    [len(sess['embeddings'])])
                logit = model(embs, length)
                preds.append(
                    torch.sigmoid(logit).item())
                labs.append(sess['crisis_label'])

        auc = roc_auc_score(labs, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, best_auc


def train_scheduled_sampling(
        train_sessions, test_sessions,
        seed=42, epochs=25,
        decay='linear'):
    """
    Scheduled Sampling 학습.
    decay 옵션:
      'linear'           : ε = 1 - (ep-1)/(E-1)
      'exponential'      : ε = 0.99^(ep-1)
      'inverse_sigmoid'  : ε = k/(k+exp(ep/k))
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ScheduledSamplingDataset(
        train_sessions, epsilon=1.0)
    loader  = DataLoader(
        dataset, batch_size=32,
        shuffle=True, collate_fn=collate_fn)

    model     = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        # ε 스케줄 계산
        if decay == 'linear':
            epsilon = max(
                0.0,
                1.0 - (epoch - 1) / (epochs - 1))
        elif decay == 'exponential':
            epsilon = 0.99 ** (epoch - 1)
        elif decay == 'inverse_sigmoid':
            k = 10
            epsilon = k / (
                k + np.exp(epoch / k))
        else:
            epsilon = 1.0

        dataset.set_epsilon(epsilon)

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
                logit = model(embs, length)
                preds.append(
                    torch.sigmoid(logit).item())
                labs.append(sess['crisis_label'])

        auc = roc_auc_score(labs, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {
                k: v.clone()
                for k, v in
                model.state_dict().items()
            }

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:2d}: "
                  f"ε={epsilon:.3f}  "
                  f"AUC={auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc

# ============================================================
# 7. 평가 함수
# ============================================================

def eval_at_T(model, sessions, T):
    """T턴까지만 보고 prefix-AUC 반환"""
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
    """Mean pooling baseline"""
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

# ============================================================
# 8. 메인 실험
# ============================================================
print("\n[5] Running experiment...")
print("Models: TCT-GRU | Full-GRU | "
      "SchedSamp-Linear | SchedSamp-Exp | "
      "Mean-pool")

results = {
    'TCT-GRU':          {T: [] for T in T_LIST},
    'Full-GRU':         {T: [] for T in T_LIST},
    'SchedSamp-Linear': {T: [] for T in T_LIST},
    'SchedSamp-Exp':    {T: [] for T in T_LIST},
    'Mean-pool':        {T: [] for T in T_LIST},
}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    # ── TCT-GRU ──
    print("  [TCT-GRU] 학습 중...")
    m_tct, auc = train_standard(
        train_c, test_c,
        dataset_cls=TCTDataset,
        seed=seed, epochs=25)
    print(f"    Best AUC (full): {auc:.3f}")

    # ── Full-GRU ──
    print("  [Full-GRU] 학습 중...")
    m_full, auc = train_standard(
        train_c, test_c,
        dataset_cls=FullDataset,
        seed=seed, epochs=25)
    print(f"    Best AUC (full): {auc:.3f}")

    # ── SchedSamp-Linear ──
    print("  [SchedSamp-Linear] 학습 중...")
    m_ss_lin, auc = train_scheduled_sampling(
        train_c, test_c,
        seed=seed, epochs=25,
        decay='linear')
    print(f"    Best AUC (full): {auc:.3f}")

    # ── SchedSamp-Exponential ──
    print("  [SchedSamp-Exp] 학습 중...")
    m_ss_exp, auc = train_scheduled_sampling(
        train_c, test_c,
        seed=seed, epochs=25,
        decay='exponential')
    print(f"    Best AUC (full): {auc:.3f}")

    # ── prefix-AUC 평가 ──
    for T in T_LIST:
        results['TCT-GRU'][T].append(
            eval_at_T(m_tct, test_c, T))
        results['Full-GRU'][T].append(
            eval_at_T(m_full, test_c, T))
        results['SchedSamp-Linear'][T].append(
            eval_at_T(m_ss_lin, test_c, T))
        results['SchedSamp-Exp'][T].append(
            eval_at_T(m_ss_exp, test_c, T))
        results['Mean-pool'][T].append(
            eval_mean_pool(
                train_c, test_c, T, seed))

# ============================================================
# 9. 결과 출력
# ============================================================
print("\n" + "="*90)
print("Scheduled Sampling vs TCT — "
      "prefix-AUC (mean±std, 5 seeds)")
print("="*90)
print(f"{'T':>4} | {'TCT-GRU':>13} | "
      f"{'Full-GRU':>13} | "
      f"{'SS-Linear':>13} | "
      f"{'SS-Exp':>13} | "
      f"{'Mean-pool':>13}")
print("-"*90)

for T in T_LIST:
    vals  = {k: results[k][T] for k in results}
    means = {k: np.mean(v)
             for k, v in vals.items()}
    best  = max(means.values())

    def fmt(k):
        v = vals[k]
        m = np.mean(v)
        s = np.std(v)
        marker = " *" if abs(m - best) < 1e-9 \
                 else "  "
        return f"{m:.3f}±{s:.3f}{marker}"

    print(f"{T:>4} | {fmt('TCT-GRU'):>15} | "
          f"{fmt('Full-GRU'):>15} | "
          f"{fmt('SchedSamp-Linear'):>15} | "
          f"{fmt('SchedSamp-Exp'):>15} | "
          f"{fmt('Mean-pool'):>15}")

print("\n* = best at this T")

# 핵심 비교
print("\n--- Key Comparison at T=5, T=10 ---")
for T in [5, 10]:
    tct = np.mean(results['TCT-GRU'][T])
    ful = np.mean(results['Full-GRU'][T])
    ssl = np.mean(results['SchedSamp-Linear'][T])
    sse = np.mean(results['SchedSamp-Exp'][T])
    print(f"\nT={T}:")
    print(f"  TCT-GRU:      {tct:.3f}  "
          f"(Δ vs Full = {tct-ful:+.3f})")
    print(f"  SS-Linear:    {ssl:.3f}  "
          f"(Δ vs Full = {ssl-ful:+.3f})")
    print(f"  SS-Exp:       {sse:.3f}  "
          f"(Δ vs Full = {sse-ful:+.3f})")
    print(f"  Full-GRU:     {ful:.3f}")

# 저장
with open('/content/results_scheduled_sampling.pkl',
          'wb') as f:
    pickle.dump(results, f)
print("\n저장 완료! "
      "→ /content/results_scheduled_sampling.pkl")
