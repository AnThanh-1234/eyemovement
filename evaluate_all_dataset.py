import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from hmmlearn.hmm import GMMHMM
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

FIX, PUR, SAC = 0, 1, 2
NAMES = {FIX: "fixation", PUR: "smooth pursuit", SAC: "saccade"}

CFG = dict(
    fs=90,
    seed=42,
    K_MAX=9,
    KMEANS_MAX_ITER=100,
    KMEANS_RESTARTS=10,
    L1_FEATURES=['x', 'y'],
    L2_FEATURES=['v'],
    N_MIX=2, COV_TYPE='full', N_ITER=50, TOL=1e-5,
    N_RESTARTS=5,
    MIN_SEG_LEN=15,
    STANDARDIZE=True
)

def kmeans_sse(X, k, max_iter, restarts, rng):
    best = np.inf
    for _ in range(restarts):
        mu = X[rng.choice(len(X), k, replace=False)]
        for _ in range(max_iter):
            d = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
            c = d.argmin(1)
            new = np.array([X[c == i].mean(0) if (c == i).any() else mu[i] for i in range(k)])
            if np.allclose(new, mu): break
            mu = new
        sse = sum(((X[c == i] - mu[i]) ** 2).sum() for i in range(k))
        best = min(best, sse)
    return best

def find_elbow(ks, sse):
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (sse - sse.min()) / (sse.max() - sse.min() + 1e-12)
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[d.argmax()])

def step0_optimal_k(df, cfg):
    rng = np.random.default_rng(cfg['seed'])
    X = df[['x', 'y']].values
    ks = np.arange(1, min(cfg['K_MAX'], len(X) - 1) + 1)
    sse = np.array([kmeans_sse(X, k, cfg['KMEANS_MAX_ITER'], cfg['KMEANS_RESTARTS'], rng) for k in ks])
    return find_elbow(ks, sse), ks, sse

def fit_gmmhmm(X, n_states, cfg, seed=0):
    for n_mix in dict.fromkeys([cfg['N_MIX'], 1]):
        best = None
        for r in range(cfg['N_RESTARTS']):
            try:
                m = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type=cfg['COV_TYPE'],
                           n_iter=cfg['N_ITER'], tol=cfg['TOL'], min_covar=1e-3, random_state=seed + r)
                m.fit(X)
                s = m.score(X)
                if np.isfinite(s) and (best is None or s > best[1]): best = (m, s)
            except Exception:
                continue
        if best is not None:
            return best[0]
    return None

def viterbi_states(model, X):
    _, states = model.decode(X, algorithm="viterbi")
    return states

def _scale(X, on):
    return StandardScaler().fit_transform(X) if on else X

def hierarchical_gmm_hmm(df, cfg, k=None):
    N = len(df)
    if k is None:
        k, ks_, sse_ = step0_optimal_k(df, cfg)
    X1 = _scale(df[cfg['L1_FEATURES']].values, cfg['STANDARDIZE'])
    m1 = fit_gmmhmm(X1, k, cfg, cfg['seed'])
    seg_id = viterbi_states(m1, X1)
    
    v = df['v'].values
    local = np.full(N, -1)
    seg_stats = []
    tiny = []
    for s in np.unique(seg_id):
        idx = np.where(seg_id == s)[0]
        if len(idx) < cfg['MIN_SEG_LEN']:
            tiny.append(idx); continue
        X2 = _scale(df.loc[idx, cfg['L2_FEATURES']].values, cfg['STANDARDIZE'])
        m2 = fit_gmmhmm(X2, 3, cfg, cfg['seed'] + 100)
        if m2 is None:
            tiny.append(idx); continue
        st = viterbi_states(m2, X2)
        local[idx] = st
        for c in np.unique(st):
            seg_stats.append((s, c, v[idx][st == c].mean()))
            
    lv = np.log10(np.array([m for _, _, m in seg_stats]) + 1e-6).reshape(-1, 1)
    km = KMeans(3, n_init=10, random_state=cfg['seed']).fit(lv)
    order = np.argsort(km.cluster_centers_.ravel())
    rank = {int(c): r for r, c in enumerate(order)}
    centers = km.cluster_centers_.ravel()[order]
    state2label = {(s, c): rank[int(g)] for (s, c, _), g in zip(seg_stats, km.labels_)}
    
    pred = np.full(N, -1)
    for i in range(N):
        if local[i] >= 0:
            pred[i] = state2label[(seg_id[i], local[i])]
    for idx in tiny:
        d = np.abs(np.log10(v[idx] + 1e-6)[:, None] - centers[None])
        pred[idx] = d.argmin(1)
    return dict(k=k, seg_id=seg_id, pred=pred, centers=10 ** centers, n_tiny=len(tiny))

# Run on all 87 files in dataset/testdataset and dataset/results
txt_files = sorted(glob.glob(r'D:\cac_mon_hoc\eyemovement\dataset\testdataset\*.txt'))
res_dir = r'D:\cac_mon_hoc\eyemovement\dataset\results'

results = []

for txt_path in txt_files:
    fname = os.path.basename(txt_path)
    base_name = os.path.splitext(fname)[0]
    csv_path = os.path.join(res_dir, f"{base_name}_results.csv")
    
    if not os.path.exists(csv_path):
        continue
        
    arr = np.loadtxt(txt_path)
    csv_df = pd.read_csv(csv_path)
    
    df = pd.DataFrame(dict(t=np.arange(len(arr))/CFG['fs'], x=arr[:,0], y=arr[:,1], v=arr[:,2]))
    y_true = csv_df['manually results'].values.astype(int)
    
    try:
        out = hierarchical_gmm_hmm(df, CFG)
        y_pred = out['pred']
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[FIX, PUR, SAC], zero_division=0)
        
        results.append({
            'file': fname,
            'k': out['k'],
            'acc': acc,
            'p_fix': p[0], 'r_fix': r[0], 'f_fix': f[0],
            'p_pur': p[1], 'r_pur': r[1], 'f_pur': f[1],
            'p_sac': p[2], 'r_sac': r[2], 'f_sac': f[2],
            'p_macro': p.mean(), 'r_macro': r.mean(), 'f_macro': f.mean()
        })
    except Exception as e:
        print(f"Error on {fname}: {e}")

res_df = pd.DataFrame(results)
print(f"\n--- ĐÁNH GIÁ TRÊN TỔNG CỘNG {len(res_df)} TRIAL CỦA TOÀN BỘ DATASET ---")
print(f"Accuracy trung bình = {res_df['acc'].mean():.4f} +/- {res_df['acc'].std():.4f}")
print(f"Precision trung bình (Macro) = {res_df['p_macro'].mean():.4f} +/- {res_df['p_macro'].std():.4f}")
print(f"Recall trung bình (Macro) = {res_df['r_macro'].mean():.4f} +/- {res_df['r_macro'].std():.4f}")
print(f"F1-Score trung bình (Macro) = {res_df['f_macro'].mean():.4f} +/- {res_df['f_macro'].std():.4f}")

print("\n--- CHI TIẾT TỪNG LỚP ---")
print(f"Fixation: P={res_df['p_fix'].mean():.4f}, R={res_df['r_fix'].mean():.4f}, F1={res_df['f_fix'].mean():.4f}")
print(f"Smooth Pursuit: P={res_df['p_pur'].mean():.4f}, R={res_df['r_pur'].mean():.4f}, F1={res_df['f_pur'].mean():.4f}")
print(f"Saccade: P={res_df['p_sac'].mean():.4f}, R={res_df['r_sac'].mean():.4f}, F1={res_df['f_sac'].mean():.4f}")
