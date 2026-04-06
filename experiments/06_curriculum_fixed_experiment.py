# ============================================================
# STANDALONE: Curriculum-GRU & Fixed-T5-GRU
# Appendix C — Training Scheme Ablation
# 필요 파일: df_COMPLETE_for_analysis.pkl
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

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
            if count >= 50:
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
# 3. Session 구성 + Normal/Emergency 필터
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
        'has_risk':        int(row['has_risk']),
        'first_risk_turn': row['first_risk_turn'],
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

# 기존과 동일한 split (random_state=42 고정)
train_c, test_c = train_test_split(
    crisis_sessions,
    test_size=0.2,
    random_state=42,
    stratify=[s['crisis_label']
              for s in crisis_sessions]
)
print(f"Train: {len(train_c)}, Test: {len(test_c)}")

# ============================================================
# 4. 모델 정의
# ============================================================

def collate_fn(batch):
    embs, labels, lengths, risks = zip(*batch)
    embs_padded = pad_sequence(embs, batch_first=True)
    labels  = torch.stack(labels)
    lengths = torch.LongTensor(lengths)
    return embs_padded, labels, lengths, risks


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
        last_hidden = gru_out[torch.arange(B), idx]
        return self.classifier(last_hidden)

# ============================================================
# 5. Dataset 정의
# ============================================================

class CurriculumDataset(Dataset):
    """
    Curriculum-GRU:
    epoch 초반 → 짧은 prefix
    epoch 후반 → 긴 prefix 허용
    선형 스케줄 (deterministic difficulty schedule)
    """
    def __init__(self, sessions,
                 current_epoch, total_epochs,
                 min_t=3, max_t=50):
        self.sessions = sessions
        self.min_t    = min_t
        progress      = (current_epoch - 1) / max(
            total_epochs - 1, 1)
        self.curr_max = int(
            min_t + progress * (max_t - min_t))
        self.curr_max = max(self.curr_max, min_t)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess   = self.sessions[idx]
        embs   = sess['embeddings']
        T      = min(len(embs), self.curr_max)
        t      = np.random.randint(self.min_t, T + 1)
        embs_t = torch.FloatTensor(embs[:t])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return (embs_t, label, t,
                sess.get('first_risk_turn') or 0)


class FixedTDataset(Dataset):
    """
    Fixed-T5-GRU:
    항상 exactly fixed_t 턴만 사용
    single-point specialisation 검증용
    """
    def __init__(self, sessions, fixed_t=5):
        self.sessions = sessions
        self.fixed_t  = fixed_t

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess   = self.sessions[idx]
        embs   = sess['embeddings']
        t      = min(self.fixed_t, len(embs))
        embs_t = torch.FloatTensor(embs[:t])
        label  = torch.FloatTensor(
            [sess['crisis_label']])
        return (embs_t, label, t,
                sess.get('first_risk_turn') or 0)

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


def run_curriculum(train_sessions, test_sessions,
                   seed=42, epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        # 매 epoch 데이터셋 갱신 (핵심)
        loader = DataLoader(
            CurriculumDataset(
                train_sessions,
                current_epoch=epoch,
                total_epochs=epochs),
            batch_size=32, shuffle=True,
            collate_fn=collate_fn)

        model.train()
        for embs, labels, lengths, _ in loader:
            embs    = embs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(embs, lengths), labels)
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
            curr_max = int(
                3 + (epoch-1)/(epochs-1) * 47)
            print(f"    Epoch {epoch:2d} | "
                  f"curr_max={curr_max:2d} | "
                  f"AUC={auc:.3f}")

    model.load_state_dict(best_state)
    return model, best_auc


def run_fixed_t(train_sessions, test_sessions,
                fixed_t=5, seed=42, epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        FixedTDataset(train_sessions,
                      fixed_t=fixed_t),
        batch_size=32, shuffle=True,
        collate_fn=collate_fn)

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
            loss = criterion(
                model(embs, lengths), labels)
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

# ============================================================
# 7. 메인 실험
# ============================================================
print("\n[5] Running Curriculum & Fixed-T5 "
      "(5 seeds)...")

results = {
    'Curriculum-GRU': {T: [] for T in T_LIST},
    'Fixed-T5-GRU':   {T: [] for T in T_LIST},
}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    # Curriculum-GRU
    print("  [Curriculum-GRU]")
    m_curr, auc = run_curriculum(
        train_c, test_c,
        seed=seed, epochs=25)
    print(f"  → best val AUC: {auc:.3f}")

    # Fixed-T5-GRU
    print("  [Fixed-T5-GRU]")
    m_fixed, auc = run_fixed_t(
        train_c, test_c,
        fixed_t=5, seed=seed, epochs=25)
    print(f"  → best val AUC: {auc:.3f}")

    # prefix-AUC 평가
    for T in T_LIST:
        results['Curriculum-GRU'][T].append(
            eval_at_T(m_curr, test_c, T))
        results['Fixed-T5-GRU'][T].append(
            eval_at_T(m_fixed, test_c, T))

# ============================================================
# 8. 저장
# ============================================================
with open('/content/results_curriculum_fixed.pkl',
          'wb') as f:
    pickle.dump(results, f)
print("\n✅ Saved: results_curriculum_fixed.pkl")

# ============================================================
# 9. 결과 테이블
# ============================================================
print("\n" + "="*60)
print("Curriculum-GRU vs Fixed-T5-GRU (5 seeds)")
print("="*60)
print(f"{'T':>5} | {'Curriculum-GRU':>16} | "
      f"{'Fixed-T5-GRU':>16}")
print("-"*45)

for T in T_LIST:
    curr  = results['Curriculum-GRU'][T]
    fixed = results['Fixed-T5-GRU'][T]
    print(f"{T:>5} | "
          f"{np.mean(curr):.3f}±"
          f"{np.std(curr):.3f}       | "
          f"{np.mean(fixed):.3f}±"
          f"{np.std(fixed):.3f}")

print("\n--- Key: Fixed-T5 collapse at T>=10 ---")
fixed_5  = np.mean(results['Fixed-T5-GRU'][5])
fixed_10 = np.mean(results['Fixed-T5-GRU'][10])
print(f"Fixed-T5: T=5={fixed_5:.3f} → "
      f"T=10={fixed_10:.3f} "
      f"(Δ={fixed_10-fixed_5:+.3f})")
print("Done!")
