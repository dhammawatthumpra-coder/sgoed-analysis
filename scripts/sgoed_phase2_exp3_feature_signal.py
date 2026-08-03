import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# Experiment 3: Predictive signal test
# (Phase-2 artifact test design, Section 3)
#
# Direct question: is per-attempt merge success predictable from the pair's
# matrices at all? Builds a per-attempt dataset with structural features,
# fits a full logistic model and reports:
#   - cross-validated AUC (stability)
#   - standardized coefficients (which features matter)
#   - univariate AUC per feature (single-feature signal)
#   - permutation test on y (global null; real AUC should sit far above ~0.5)
# Same dataset parameters as Experiment 2 so the numbers are comparable.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
NUM_ATTEMPTS = 8000
CROSS_TERM_STRENGTH = 1.4
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
DATASET_SEED = 42
N_PERM = 300


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

# headline: 5-fold cross-validated AUC
cv_aucs = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc")

# single-split model for coefficients / permutation comparison
scaler = StandardScaler().fit(Xtr)
Xs_tr, Xs_te = scaler.transform(Xtr), scaler.transform(Xte)
clf = LogisticRegression(max_iter=2000)
clf.fit(Xs_tr, ytr)
test_auc = roc_auc_score(yte, clf.predict_proba(Xs_te)[:, 1])
test_acc = accuracy_score(yte, clf.predict(Xs_te))
coefs = sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1]))

# univariate (rank-based) AUC: single feature used directly as the score
uni_auc = [(FEATURE_NAMES[k], roc_auc_score(y, X[:, k])) for k in range(X.shape[1])]
uni_auc.sort(key=lambda t: -t[1])

# global permutation null on y
rng = np.random.default_rng(1)
perm_aucs = []
for _ in range(N_PERM):
    ytr_p = rng.permutation(ytr)
    yte_p = rng.permutation(yte)
    clf_p = LogisticRegression(max_iter=2000).fit(Xs_tr, ytr_p)
    perm_aucs.append(roc_auc_score(yte_p, clf_p.predict_proba(Xs_te)[:, 1]))
perm_aucs = np.array(perm_aucs)
p_ge = float(np.mean(perm_aucs >= test_auc))

print("=" * 72)
print(" Experiment 3 -- Predictive signal test (full feature model)")
print("=" * 72)
print(f"Dataset: {len(y)} merge attempts, success rate {y.mean()*100:.1f}%\n")
print(f"5-fold CV AUC:                {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")
print(f"Test-set AUC:                 {test_auc:.4f}")
print(f"Test-set accuracy:            {test_acc:.4f}")
print("\nStandardized logistic coefficients (sorted by |coef|):")
for name, c in coefs:
    print(f"  {name:<18} {c:+.4f}")
print("\nUnivariate AUC per feature (rank-based, feature alone):")
for name, a in uni_auc:
    print(f"  {name:<18} {a:.4f}")
print(f"\nPermutation null on y ({N_PERM} iters):")
print(f"  mean +/- std:               {perm_aucs.mean():.4f} +/- {perm_aucs.std():.4f}")
print(f"  min/max:                    {perm_aucs.min():.4f} / {perm_aucs.max():.4f}")
print(f"  fraction >= real test AUC (p): {p_ge:.4f}")

print("""
Interpretation guide (Section 3.6 criteria for the real AUC):
- < 0.55: essentially no structural signal; 0.55-0.65: weak (possibly margin-
  only); 0.65-0.75: moderate; > 0.75: strong.
- The permutation null should sit at ~0.5. If the real AUC is far above it and
  above the univariate margin-only AUC, the matrices carry genuine signal.
""")
