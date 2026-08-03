import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# Experiment 3.5: blend-margin mediation
# (Phase-2 artifact test design, Section 3.5)
#
# The deterministic part of the merged composite is the blend
#   S_blend = 0.5*(S_i + S_j),  with margin  m_blend = lambda_min(S_blend),
# which is the stability margin BEFORE the cross-noise is added. If the
# structural signal (esp. the negative sym_alignment coefficient) is really a
# proxy for m_blend, then adding m_blend to the model should absorb it.
#
# Models compared on per-attempt data:
#   M1: margin features only              [min, mean, diff]
#   M2: margin + structural (no blend)    [the 13-feature model of Exp 2/3]
#   M3: blend_margin alone
#   M4: full, all features + blend_margin
#
# Also: element-wise train/test split (210/90 elements) to check whether the
# signal generalizes to unseen elements or is element-specific.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
NUM_ATTEMPTS = 8000
CROSS_TERM_STRENGTH = 1.4
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
DATASET_SEED = 42
N_PERM = 200
TEST_FRAC = 0.3

MARGIN_FEATURES = ["min_margin", "mean_margin", "margin_diff"]


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def attempt_merge(Gi, Gj, cross_strength, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * cross_strength  # canonical: symmetric-only interference
    G_merged = 0.5 * (Gi + Gj) + cross
    return lambda_min_S(G_merged)


def blend_margin(Gi, Gj):
    """Deterministic margin of the composite BEFORE cross noise:
    lambda_min(0.5*(S_i+S_j)) = lambda_min(0.25*(G_i+G_i^T+G_j+G_j^T))."""
    S_blend = 0.25 * (Gi + Gi.T + Gj + Gj.T)
    return lambda_min_S(S_blend)


def pair_features(Gi, Gj):
    Si = 0.5 * (Gi + Gi.T)
    Sj = 0.5 * (Gj + Gj.T)
    Ai = 0.5 * (Gi - Gi.T)
    Aj = 0.5 * (Gj - Gj.T)
    vi = np.linalg.eigvalsh(Si)
    vj = np.linalg.eigvalsh(Sj)
    mi, mj = vi[0], vj[0]
    sym_i_norm = np.linalg.norm(Si, "fro")
    sym_j_norm = np.linalg.norm(Sj, "fro")
    anti_i_norm = np.linalg.norm(Ai, "fro")
    anti_j_norm = np.linalg.norm(Aj, "fro")
    return np.array([
        min(mi, mj),
        0.5 * (mi + mj),
        abs(mi - mj),
        np.trace(Si @ Sj) / (sym_i_norm * sym_j_norm + 1e-12),
        np.trace(Ai @ Aj) / (anti_i_norm * anti_j_norm + 1e-12),
        np.linalg.norm(Gi @ Gj - Gj @ Gi, "fro"),
        sym_i_norm, sym_j_norm,
        anti_i_norm, anti_j_norm,
        np.linalg.norm(Gi - Gj, "fro"),
        vi[1] - vi[0],
        vj[1] - vj[0],
        blend_margin(Gi, Gj),
    ])


FEATURE_NAMES = ["min_margin", "mean_margin", "margin_diff", "sym_alignment",
                 "anti_alignment", "commutator_norm", "sym_norm_i", "sym_norm_j",
                 "anti_norm_i", "anti_norm_j", "frob_distance",
                 "spectral_gap_i", "spectral_gap_j", "blend_margin"]
MARGIN_IDX = [0, 1, 2]
BLEND_IDX = 13
STRUCT_IDX = list(range(3, 13))
M2_IDX = list(range(13))     # margin + structural, original column order (no blend)
ALL_IDX = list(range(len(FEATURE_NAMES)))


def make_dataset(seed=DATASET_SEED, num_attempts=NUM_ATTEMPTS, idx_range=None, return_idx=False):
    rng = np.random.default_rng(seed)
    element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]
    pool = np.arange(*idx_range) if idx_range else np.arange(NUM_ELEMENTS)
    X, y, idx = [], [], []
    for _ in range(num_attempts):
        i, j = rng.choice(pool, size=2, replace=False)
        idx.append((int(i), int(j)))
        X.append(pair_features(element_G[i], element_G[j]))
        y.append(1 if attempt_merge(element_G[i], element_G[j], CROSS_TERM_STRENGTH, rng) > 0 else 0)
    X, y = np.array(X), np.array(y, dtype=int)
    return (X, y, idx) if return_idx else (X, y)


def fit_auc(cols, Xa, Xb, ya, yb):
    m = LogisticRegression(max_iter=2000)
    m.fit(Xa[:, cols], ya)
    return roc_auc_score(yb, m.predict_proba(Xb[:, cols])[:, 1])


def coef_of(model, name):
    return float(model.coef_[0][FEATURE_NAMES.index(name)])


# ---- main dataset (random pair split over all 300 elements) ----
X, y = make_dataset()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, random_state=0, stratify=y)

auc_m1 = fit_auc(MARGIN_IDX, Xtr, Xte, ytr, yte)
auc_m2 = fit_auc(M2_IDX, Xtr, Xte, ytr, yte)                    # 13 features, no blend
auc_m3 = fit_auc([BLEND_IDX], Xtr, Xte, ytr, yte)
auc_m4 = fit_auc(ALL_IDX, Xtr, Xte, ytr, yte)

# standardized models to compare the sym_alignment coefficient, M2 vs M4
scaler2 = StandardScaler().fit(Xtr[:, M2_IDX])
clf2 = LogisticRegression(max_iter=2000).fit(scaler2.transform(Xtr[:, M2_IDX]), ytr)
scaler4 = StandardScaler().fit(Xtr[:, ALL_IDX])
clf4 = LogisticRegression(max_iter=2000).fit(scaler4.transform(Xtr[:, ALL_IDX]), ytr)

# permutation test: shuffle structural features (keeps margins + blend intact)
rng = np.random.default_rng(0)
perm_aucs = []
for _ in range(N_PERM):
    Xtr_p = Xtr.copy()
    Xte_p = Xte.copy()
    p_tr = rng.permutation(len(Xtr_p))
    p_te = rng.permutation(len(Xte_p))
    Xtr_p[:, STRUCT_IDX] = Xtr_p[p_tr][:, STRUCT_IDX]
    Xte_p[:, STRUCT_IDX] = Xte_p[p_te][:, STRUCT_IDX]
    perm_aucs.append(fit_auc(ALL_IDX, Xtr_p, Xte_p, ytr, yte))
perm_aucs = np.array(perm_aucs)
p_ge = float(np.mean(perm_aucs >= auc_m4))
p_str = f"p < {1/(N_PERM+1):.4f}" if p_ge == 0 else f"p = {p_ge:.4f}"

# univariate AUC of blend_margin for reference
uni_blend = roc_auc_score(y, X[:, BLEND_IDX])

# ---- element-wise split: train on elements 0..209, test on 210..299 ----
Xtr_ew, ytr_ew = make_dataset(num_attempts=5600, idx_range=(0, 210))
Xte_ew, yte_ew = make_dataset(num_attempts=2400, idx_range=(210, NUM_ELEMENTS))
auc_ew_m1 = fit_auc(MARGIN_IDX, Xtr_ew, Xte_ew, ytr_ew, yte_ew)
auc_ew_m3 = fit_auc([BLEND_IDX], Xtr_ew, Xte_ew, ytr_ew, yte_ew)
auc_ew_m4 = fit_auc(ALL_IDX, Xtr_ew, Xte_ew, ytr_ew, yte_ew)

print("=" * 72)
print(" Experiment 3.5 -- Blend-margin mediation")
print("=" * 72)
print(f"Dataset: {len(y)} attempts (train {len(ytr)} / test {len(yte)}), "
      f"success rate {y.mean()*100:.1f}%\n")
print("AUC on random pair split (all 300 elements):")
print(f"  M1 margin-only              {auc_m1:.4f}")
print(f"  M2 margin + structural      {auc_m2:.4f}")
print(f"  M3 blend_margin alone       {auc_m3:.4f}")
print(f"  M4 full + blend_margin      {auc_m4:.4f}")
print(f"  M4 - M3 (beyond blend):     {auc_m4 - auc_m3:+.4f}")
print(f"  univariate AUC(blend_margin): {uni_blend:.4f}")
print(f"\nPermutation null (structural features shuffled, margins+blend intact, {N_PERM} iters):")
print(f"  mean +/- std:               {perm_aucs.mean():.4f} +/- {perm_aucs.std():.4f}")
print(f"  fraction >= real M4 AUC:    {p_str}")
print("\nStandardized sym_alignment coefficient (mediator check):")
print(f"  in M2 (no blend_margin):    {coef_of(clf2, 'sym_alignment'):+.4f}")
print(f"  in M4 (with blend_margin):  {coef_of(clf4, 'sym_alignment'):+.4f}")
print(f"  blend_margin coef in M4:    {coef_of(clf4, 'blend_margin'):+.4f}")
print("\nElement-wise split (train elements 0..209, test elements 210..299):")
print(f"  M1 margin-only              {auc_ew_m1:.4f}")
print(f"  M3 blend_margin alone       {auc_ew_m3:.4f}")
print(f"  M4 full                     {auc_ew_m4:.4f}")
print(f"  M4 random-split reference:  {auc_m4:.4f}")

print("""
Interpretation guide:
- If M4 is close to M3, the deterministic blend margin carries most of the signal.
- If the sym_alignment coefficient collapses toward 0 when blend_margin is added,
  sym_alignment was a mediator of the blend margin, not an independent cause.
- If the element-wise M4 AUC is close to the random-split M4 AUC, the signal
  generalizes to unseen elements; a large drop means element-specific structure.
""")
