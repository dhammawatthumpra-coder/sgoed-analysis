import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ==========================================
# Experiment 2: Margin-matched null vs full feature model
# (Phase-2 artifact test design, Section 2.2)
#
# Elements are built with a PLANTED margin (lambda_min of the symmetric part,
# set when constructing G). If merge success is predictable from margin alone,
# the "structural" claim reduces to "the model reads back the parameter we
# implanted". This experiment fits two logistic models on per-attempt data:
#   A) margin features only
#   B) margin + full structural features
# and asks whether B adds predictive power beyond A, using a permutation test
# on the structural features: shuffle their rows (breaking their link to y
# while margin stays intact) -> the permuted-full AUC should fall back to ~A.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
NUM_ATTEMPTS = 8000
CROSS_TERM_STRENGTH = 1.4
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
DATASET_SEED = 42
N_PERM = 200

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
        min(mi, mj),                                  # min_margin
        0.5 * (mi + mj),                              # mean_margin
        abs(mi - mj),                                 # margin_diff
        np.trace(Si @ Sj) / (sym_i_norm * sym_j_norm + 1e-12),   # sym_alignment
        np.trace(Ai @ Aj) / (anti_i_norm * anti_j_norm + 1e-12), # anti_alignment
        np.linalg.norm(Gi @ Gj - Gj @ Gi, "fro"),     # commutator_norm
        sym_i_norm, sym_j_norm,                       # sym_norm_i, sym_norm_j
        anti_i_norm, anti_j_norm,                     # anti_norm_i, anti_norm_j
        np.linalg.norm(Gi - Gj, "fro"),               # frob_distance
        vi[1] - vi[0],                                # spectral_gap_i
        vj[1] - vj[0],                                # spectral_gap_j
    ])


FEATURE_NAMES = ["min_margin", "mean_margin", "margin_diff", "sym_alignment",
                 "anti_alignment", "commutator_norm", "sym_norm_i", "sym_norm_j",
                 "anti_norm_i", "anti_norm_j", "frob_distance",
                 "spectral_gap_i", "spectral_gap_j"]


def make_dataset(seed=DATASET_SEED, num_attempts=NUM_ATTEMPTS):
    rng = np.random.default_rng(seed)
    element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]
    X, y = [], []
    for _ in range(num_attempts):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        X.append(pair_features(element_G[i], element_G[j]))
        y.append(1 if attempt_merge(element_G[i], element_G[j], CROSS_TERM_STRENGTH, rng) > 0 else 0)
    return np.array(X), np.array(y, dtype=int)


X, y = make_dataset()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
margin_idx = [FEATURE_NAMES.index(f) for f in MARGIN_FEATURES]
struct_idx = [k for k in range(X.shape[1]) if k not in margin_idx]
ALL = list(range(X.shape[1]))


def fit_auc(cols, Xa, Xb, ya, yb):
    m = LogisticRegression(max_iter=2000)
    m.fit(Xa[:, cols], ya)
    return roc_auc_score(yb, m.predict_proba(Xb[:, cols])[:, 1])


auc_margin = fit_auc(margin_idx, Xtr, Xte, ytr, yte)
auc_full = fit_auc(ALL, Xtr, Xte, ytr, yte)

# Permutation test on the structural features: shuffle their rows in BOTH train
# and test (keeps margin intact, destroys the structural-feature / y link).
rng = np.random.default_rng(0)
perm_aucs = []
for _ in range(N_PERM):
    Xtr_p = Xtr.copy()
    Xte_p = Xte.copy()
    p_tr = rng.permutation(len(Xtr_p))
    p_te = rng.permutation(len(Xte_p))
    Xtr_p[:, struct_idx] = Xtr_p[p_tr][:, struct_idx]
    Xte_p[:, struct_idx] = Xte_p[p_te][:, struct_idx]
    perm_aucs.append(fit_auc(ALL, Xtr_p, Xte_p, ytr, yte))
perm_aucs = np.array(perm_aucs)
p_ge = float(np.mean(perm_aucs >= auc_full))

print("=" * 72)
print(" Experiment 2 -- Margin-matched null vs full feature model")
print("=" * 72)
print(f"Dataset: {len(y)} merge attempts (train {len(ytr)} / test {len(yte)}), "
      f"overall per-attempt success {y.mean()*100:.1f}%")
print(f"Features: {len(FEATURE_NAMES)} total; margin features = {MARGIN_FEATURES}\n")
print(f"AUC (margin-only):            {auc_margin:.4f}")
print(f"AUC (full, all features):     {auc_full:.4f}")
print(f"AUC gain (full - margin):     {auc_full - auc_margin:+.4f}")
print(f"\nPermutation null (structural features shuffled, {N_PERM} iters):")
print(f"  mean +/- std:               {perm_aucs.mean():.4f} +/- {perm_aucs.std():.4f}")
print(f"  min/max:                    {perm_aucs.min():.4f} / {perm_aucs.max():.4f}")
print(f"  fraction >= full AUC (p):   {p_ge:.4f}")

print("""
Interpretation guide (Section 2.2 criteria):
- If full AUC ~= margin AUC and the permutation null centers on the full AUC,
  merge success is explained by the planted margin alone -> "margin selection",
  not structural compatibility.
- If full AUC clearly exceeds the permutation null (which should sit near the
  margin-only AUC), structural features carry real predictive power beyond the
  planted margin.
""")
