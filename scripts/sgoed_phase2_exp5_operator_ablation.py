import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# Experiment 5: merge operator ablation
# (Phase-2 artifact test design, Section 6)
#
# The structural signal found in Exp 2/3 might be an artifact of the specific
# Gaussian iid cross-term in attempt_merge. Here the merge rule is swapped for
# 8 different operators and the same analyses are rerun:
#   1. blend (no noise)                -- trivial baseline, ~100% success
#   2. gaussian_iid (previous reference; distributionally identical to the
#      canonical symmetric form -- see SGOED_v2_revision.md Section 4.6.7)
#   3. symmetric noise only            -- changes I_C directly
#   4. antisymmetric noise only        -- CONTROL: cannot change lambda_min(S),
#                                          expected ~100% success
#   5. commutator coupling   alpha [Gi,Gj]
#   6. anticommutator        alpha (GiGj+GjGi)
#   7. symmetric product     alpha (SiSj+SjSi)
#   8. symmetric difference  alpha (Si-Sj)
#
# Two regimes:
#   A) raw strength fixed (alpha = 1.4): natural difficulty of each operator.
#   B) success-rate matched: alpha tuned on validation pairs to per-attempt
#      success ~0.60; logistic models fit on validation, AUC evaluated on
#      held-out test pairs. Fairer comparison of the SIGNAL at comparable task
#      difficulty.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
N_VAL = 2000
N_TEST = 3000
DATASET_SEED = 42
ALPHA_FIXED = 1.4
ALPHA_GRID = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
TARGET_SUCCESS = 0.60
TARGET_TOL = 0.05
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0

MARGIN_IDX = [0, 1, 2]
BLEND_IDX = 13
ALL_IDX = list(range(14))

OPERATORS = ["blend", "gaussian_iid", "sym_noise", "anti_noise",
             "commutator", "anticommutator", "sym_product", "sym_diff"]


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


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
        lambda_min_S(0.25 * (Gi + Gi.T + Gj + Gj.T)),   # blend_margin
    ])


FEATURE_NAMES = ["min_margin", "mean_margin", "margin_diff", "sym_alignment",
                 "anti_alignment", "commutator_norm", "sym_norm_i", "sym_norm_j",
                 "anti_norm_i", "anti_norm_j", "frob_distance",
                 "spectral_gap_i", "spectral_gap_j", "blend_margin"]


def operator_margin(name, Gi, Gj, alpha, rng):
    """Apply merge operator, return the Bendixson margin of the composite."""
    half = 0.5 * (Gi + Gj)
    if name == "blend":
        M = half
    elif name == "gaussian_iid":
        M = half + alpha * rng.standard_normal(Gi.shape)
    elif name == "sym_noise":
        E = rng.standard_normal(Gi.shape)
        M = half + alpha * 0.5 * (E + E.T)
    elif name == "anti_noise":
        E = rng.standard_normal(Gi.shape)
        M = half + alpha * 0.5 * (E - E.T)
    elif name == "commutator":
        M = half + alpha * (Gi @ Gj - Gj @ Gi)
    elif name == "anticommutator":
        M = half + alpha * (Gi @ Gj + Gj @ Gi)
    elif name == "sym_product":
        Si, Sj = 0.5 * (Gi + Gi.T), 0.5 * (Gj + Gj.T)
        M = half + alpha * (Si @ Sj + Sj @ Si)
    elif name == "sym_diff":
        Si, Sj = 0.5 * (Gi + Gi.T), 0.5 * (Gj + Gj.T)
        M = half + alpha * (Si - Sj)
    else:
        raise ValueError(name)
    return lambda_min_S(M)


def make_pairs(n_pairs, rng_pop):
    """Sample n_pairs pairs once; return (feature matrix, G-pair lists)."""
    element_margins = rng_pop.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    element_G = [random_stable_G(GDIM, m, rng_pop) for m in element_margins]
    X, Gi_list, Gj_list = [], [], []
    for _ in range(n_pairs):
        i, j = rng_pop.choice(NUM_ELEMENTS, size=2, replace=False)
        Gi_list.append(element_G[i])
        Gj_list.append(element_G[j])
        X.append(pair_features(element_G[i], element_G[j]))
    return np.array(X), Gi_list, Gj_list


def y_of(name, alpha, Gi_list, Gj_list, seed):
    rng = np.random.default_rng(seed)
    return np.array([1 if operator_margin(name, Gi, Gj, alpha, rng) > 0 else 0
                     for Gi, Gj in zip(Gi_list, Gj_list)])


def safe_auc(ytrue, yscore):
    if ytrue.std() == 0.0:
        return float("nan")
    return roc_auc_score(ytrue, yscore)


def evaluate(name, alpha, seed_val, seed_test):
    """Fit logistic models on validation (y at tuned alpha), evaluate on test."""
    yva = y_of(name, alpha, Gi_val, Gj_val, seed=seed_val)
    yte = y_of(name, alpha, Gi_test, Gj_test, seed=seed_test)
    m = {"success": yte.mean()}
    m["auc_full"] = safe_auc(yte, _proba(ALL_IDX, yva, yte))
    m["auc_margin"] = safe_auc(yte, _proba(MARGIN_IDX, yva, yte))
    m["auc_blend"] = safe_auc(yte, _proba([BLEND_IDX], yva, yte))
    m["corr_margin"] = float(np.corrcoef(X_test[:, 0], yte)[0, 1])
    m["corr_align"] = float(np.corrcoef(X_test[:, 3], yte)[0, 1])
    m["auc_anti_align"] = safe_auc(yte, X_test[:, 4])
    if yva.std() == 0.0:
        m["coef_sym_align"] = float("nan")
    else:
        scaler = StandardScaler().fit(X_val)
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_val), yva)
        m["coef_sym_align"] = float(clf.coef_[0][3])
    return m


def _proba(cols, yva, yte):
    if yva.std() == 0.0:
        return np.full(len(yte), 0.5)   # single-class labels: no model can fit
    clf = LogisticRegression(max_iter=2000).fit(X_val[:, cols], yva)
    return clf.predict_proba(X_test[:, cols])[:, 1]


rng_pop = np.random.default_rng(DATASET_SEED)
X_val, Gi_val, Gj_val = make_pairs(N_VAL, rng_pop)
X_test, Gi_test, Gj_test = make_pairs(N_TEST, rng_pop)

print("=" * 72)
print(" Experiment 5 -- Merge operator ablation")
print("=" * 72)
print(f"Validation pairs: {N_VAL}, test pairs: {N_TEST}\n")

print("-- Regime A: raw strength fixed (alpha = 1.4) --")
print(f"{'operator':<16} {'per-attempt':>12} {'AUC full':>9}")
for op_i, name in enumerate(OPERATORS):
    yte = y_of(name, ALPHA_FIXED, Gi_test, Gj_test, seed=200 + op_i)
    auc = safe_auc(yte, _proba(ALL_IDX, y_of(name, ALPHA_FIXED, Gi_val, Gj_val, seed=100 + op_i), yte))
    print(f"{name:<16} {yte.mean()*100:11.1f}% {auc:9.4f}")

print("\n-- Regime B: success-rate matched (alpha tuned on validation) --")
print("AUCs: logistic fit on validation, evaluated on held-out test.\n")
print(f"{'operator':<16} {'alpha':>6} {'per-att':>8} {'AUC full':>9} {'AUC marg':>9} "
      f"{'AUC blend':>9} {'corr_marg':>9} {'corr_align':>10} {'coef_sym_align':>14} {'auc_anti_align':>14}")

for op_i, name in enumerate(OPERATORS):
    if name in ("blend", "anti_noise"):
        # deterministic / control operators: success cannot be tuned to ~60%
        m = evaluate(name, ALPHA_FIXED, seed_val=100 + op_i, seed_test=200 + op_i)
        print(f"{name:<16} {'n/a':>6} {m['success']*100:7.1f}% {m['auc_full']:9.4f} "
              f"{m['auc_margin']:9.4f} {m['auc_blend']:9.4f} {m['corr_margin']:+9.3f} "
              f"{m['corr_align']:+10.3f} {m['coef_sym_align']:+14.4f} {m['auc_anti_align']:14.4f}")
        continue

    # tune alpha on validation: prefer in-range [0.55, 0.65], else closest to 0.6
    best_alpha, best_score, best_succ = None, 1e9, None
    for a in ALPHA_GRID:
        succ = y_of(name, a, Gi_val, Gj_val, seed=200 + op_i).mean()
        in_range = abs(succ - TARGET_SUCCESS) <= TARGET_TOL
        score = abs(succ - TARGET_SUCCESS) if in_range else abs(succ - TARGET_SUCCESS) + 1.0
        if score < best_score:
            best_score, best_alpha, best_succ = score, a, succ

    m = evaluate(name, best_alpha, seed_val=100 + op_i, seed_test=300 + op_i)
    flag = "" if abs(m["success"] - TARGET_SUCCESS) <= TARGET_TOL else "  (off-range)"
    print(f"{name:<16} {best_alpha:6.2f} {m['success']*100:7.1f}% {m['auc_full']:9.4f} "
          f"{m['auc_margin']:9.4f} {m['auc_blend']:9.4f} {m['corr_margin']:+9.3f} "
          f"{m['corr_align']:+10.3f} {m['coef_sym_align']:+14.4f} {m['auc_anti_align']:14.4f}{flag}")

print("""
Interpretation guide (Section 6.4 criteria):
- anti_noise control: per-attempt success must be ~100% (antisymmetric noise
  cannot change lambda_min(S)); if not, the control fails.
- sym_noise should behave like gaussian_iid (both act through the symmetric
  part): AUC ~0.6-0.7, positive margin correlation, negative sym_alignment.
- structured operators (commutator / anticommutator / sym_product / sym_diff)
  keeping AUC clearly above chance means the signal is not a Gaussian-noise
  artifact.
- coef_sym_align negative across many operators -> robust structural effect;
  auc_anti_align ~0.5 everywhere is expected (criterion ignores the anti part).
""")
