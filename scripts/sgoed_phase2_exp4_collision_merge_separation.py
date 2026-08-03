import numpy as np

# ==========================================
# Experiment 4: Collision/merge separation
# (Phase-2 artifact test design, Section 5)
#
# The merged fraction in the kinetic model conflates two processes: (i) how
# often the random walk brings pairs into contact (collision), and (ii) how
# selective the merge test is (sticking). Conditions:
#   A: full kinetic model (baseline, random bounce)
#   B: fixed number of attempts K per pair (no random walk at all)
#   C: kinetic, NO bounce after a failed merge
#   D: kinetic, deterministic bounce (d += 0.5)
# Metrics: merged fraction, per-attempt success, attempts/trial, correlations
# of success with margin and sym_alignment, and the hazard
# h(k) = P(success at attempt k | trial reached attempt k). With independent
# attempts h(k) should be ~constant (memoryless); drift would indicate a
# trajectory/memory effect.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
PAIR_TRIALS = 800
MAX_STEPS = 400
D_CONTACT = 0.4
D_MAX = 8.0
NOISE_STD = 0.35
CROSS_TERM_STRENGTH = 1.4
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
SEEDS = [1, 2, 3, 4, 5]
FIXED_KS = [1, 2, 3, 5, 10, 20]
HAZARD_KMAX = 10


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * CROSS_TERM_STRENGTH  # canonical: symmetric-only interference
    G_merged = 0.5 * (Gi + Gj) + cross
    return lambda_min_S(G_merged)


def sym_alignment(Gi, Gj):
    Si = 0.5 * (Gi + Gi.T)
    Sj = 0.5 * (Gj + Gj.T)
    return np.trace(Si @ Sj) / (np.linalg.norm(Si, "fro") * np.linalg.norm(Sj, "fro") + 1e-12)


def build_population(rng):
    element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]
    margin_actual = np.array([lambda_min_S(G) for G in element_G])
    return element_G, margin_actual


def run_kinetic(seed, bounce):
    """Kinetic pair-trial model. bounce: 'random' | 'none' | 'fixed'."""
    rng = np.random.default_rng(seed)
    element_G, margin_actual = build_population(rng)
    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    attempts_per_trial = []
    per_trial_outcomes = []   # 0/1 per attempt, in order, per trial
    step_at_merge = []
    per_attempt = []          # (success, min_margin, sym_alignment)
    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        Gi, Gj = element_G[i], element_G[j]
        mm = min(margin_actual[i], margin_actual[j])
        sa = sym_alignment(Gi, Gj)
        d = rng.uniform(2.0, 5.0)
        outcome, n_att, trial_outcomes = None, 0, []
        for step in range(MAX_STEPS):
            d += rng.standard_normal() * NOISE_STD
            d = max(d, 0.05)
            if d > D_MAX:
                outcome = "drifted_apart"
                break
            if d < D_CONTACT:
                n_att += 1
                success = attempt_merge(Gi, Gj, rng) > 0
                trial_outcomes.append(1 if success else 0)
                per_attempt.append((1 if success else 0, mm, sa))
                if success:
                    outcome = "merged"
                    step_at_merge.append(step)
                    break
                else:
                    if bounce == "random":
                        d += abs(rng.standard_normal()) * 0.5
                    elif bounce == "fixed":
                        d += 0.5
                    # bounce == "none": no push
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1
        attempts_per_trial.append(n_att)
        per_trial_outcomes.append(trial_outcomes)
    return counts, attempts_per_trial, per_trial_outcomes, per_attempt


def run_fixed_K(seed, K):
    """Fixed number of merge attempts per pair, no random walk."""
    rng = np.random.default_rng(seed)
    element_G, margin_actual = build_population(rng)
    merged_pairs = 0
    per_trial_outcomes = []
    per_attempt = []          # (success, min_margin, sym_alignment)
    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        Gi, Gj = element_G[i], element_G[j]
        mm = min(margin_actual[i], margin_actual[j])
        sa = sym_alignment(Gi, Gj)
        trial_outcomes = []
        for _k in range(K):
            success = attempt_merge(Gi, Gj, rng) > 0
            trial_outcomes.append(1 if success else 0)
            per_attempt.append((1 if success else 0, mm, sa))
            if success:
                break
        if trial_outcomes and trial_outcomes[0]:
            merged_pairs += 1
        per_trial_outcomes.append(trial_outcomes)
    return merged_pairs, per_trial_outcomes, per_attempt


def hazard(per_trial_outcomes, kmax=HAZARD_KMAX):
    """h(k) = P(success at attempt k | trial reached attempt k)."""
    hs = []
    for k in range(1, kmax + 1):
        reached = [t for t in per_trial_outcomes if len(t) >= k]
        if not reached:
            hs.append(float("nan"))
        else:
            hs.append(sum(1 for t in reached if t[k - 1] == 1) / len(reached))
    return hs


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(label, merged_frac, attempts_per_trial, per_trial_outcomes, per_attempt):
    per_attempt = np.array(per_attempt, dtype=float)
    hs = hazard(per_trial_outcomes)
    return {
        "label": label,
        "merged_frac": merged_frac,
        "per_attempt": per_attempt[:, 0].mean() if len(per_attempt) else float("nan"),
        "att_trial": np.mean(attempts_per_trial),
        "corr_margin": pearson(per_attempt[:, 1], per_attempt[:, 0]),
        "corr_align": pearson(per_attempt[:, 2], per_attempt[:, 0]),
        "hazard": hs,
    }


def h_mean_of(sel):
    """Mean hazard across runs, column-wise, ignoring all-nan columns."""
    h_mat = np.array([r["hazard"] for r in sel], dtype=float)
    means = []
    for k in range(HAZARD_KMAX):
        col = h_mat[:, k]
        finite = np.isfinite(col)
        means.append(float(np.mean(col[finite])) if finite.any() else float("nan"))
    return np.array(means)


print("=" * 72)
print(" Experiment 4 -- Collision/merge separation")
print("=" * 72)

rows = []

# A, C, D: kinetic runs
for label, bounce in (("A full kinetic (random bounce)", "random"),
                      ("C kinetic, no bounce", "none"),
                      ("D kinetic, fixed bounce", "fixed")):
    for seed in SEEDS:
        counts, attempts_per_trial, per_trial_outcomes, per_attempt = run_kinetic(seed, bounce)
        total = sum(counts.values())
        rows.append(aggregate(f"{label} (seed {seed})", counts["merged"] / total,
                              attempts_per_trial, per_trial_outcomes, per_attempt))

# B: fixed-K
for K in FIXED_KS:
    for seed in SEEDS:
        merged_pairs, per_trial_outcomes, per_attempt = run_fixed_K(seed, K)
        rows.append(aggregate(f"B fixed-K={K} (seed {seed})", merged_pairs / PAIR_TRIALS,
                              [K] * PAIR_TRIALS, per_trial_outcomes, per_attempt))

# print per-condition means (averaged over seeds)
print(f"\n{'condition':<38} {'merged':>7} {'per-att':>8} {'att/trial':>9} "
      f"{'corr_marg':>9} {'corr_align':>10}   hazard h(1) h(2) h(3) h(5) h(10)")
for label in ["A full kinetic (random bounce)", "C kinetic, no bounce", "D kinetic, fixed bounce"]:
    sel = [r for r in rows if r["label"].startswith(label)]
    mf = [r["merged_frac"] for r in sel]
    pa = [r["per_attempt"] for r in sel]
    at = [r["att_trial"] for r in sel]
    cm = [r["corr_margin"] for r in sel]
    ca = [r["corr_align"] for r in sel]
    h_mean = h_mean_of(sel)
    h_show = " ".join(f"{h_mean[k-1]:5.2f}" if k in (1, 2, 3, 5, 10) else "" for k in range(1, 11))
    print(f"{label:<38} {np.mean(mf)*100:6.1f}% {np.mean(pa)*100:7.1f}% {np.mean(at):9.2f} "
          f"{np.mean(cm):+9.3f} {np.mean(ca):+10.3f}   {h_show}")
for K in FIXED_KS:
    sel = [r for r in rows if r["label"].startswith(f"B fixed-K={K} ")]
    mf = [r["merged_frac"] for r in sel]
    pa = [r["per_attempt"] for r in sel]
    cm = [r["corr_margin"] for r in sel]
    ca = [r["corr_align"] for r in sel]
    h_mean = h_mean_of(sel)
    h_show = " ".join(f"{h_mean[k-1]:5.2f}" if k <= K else "" for k in range(1, 11))
    print(f"B fixed-K={K:<3} (no walk)             {np.mean(mf)*100:6.1f}% {np.mean(pa)*100:7.1f}% "
          f"{K:9d} {np.mean(cm):+9.3f} {np.mean(ca):+10.3f}   {h_show}")

print("""
Interpretation guide (Section 4.3 criteria):
- corr(min_margin, success) positive in the fixed-K model too: selectivity is
  not an artifact of the random walk / repeated attempts.
- corr(sym_alignment, success) keeps its (negative) pattern across conditions.
- per-attempt success barely changes when bounce is removed/changed: the bounce
  rule is not what drives the result.
- hazard h(k) ~ constant: attempts are independent (memoryless). Strong drift
  would indicate a trajectory/memory effect.
""")
