import numpy as np

# ==========================================
# Experiment 1: Fixed-p null vs the real kinetic model
# (Phase-2 artifact test design, Section 2.1)
#
# Question: does the real model's merge success carry information about the
# pair's matrices, or could a model that decides merges with a CONSTANT
# probability p (matched to the real per-attempt success rate) reproduce
# everything that matters?
#
# The null keeps the identical random walk (collision process), so the marginal
# statistics (merged fraction, step timing, attempts per trial) are expected to
# match by construction. The DISCRIMINATING statistics are the correlations of
# per-attempt success with element properties (margin, symmetric alignment):
# a constant-p null cannot reproduce any nonzero correlation.
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
    """Return the Bendixson margin of the merged composite (positive = stable).
    Canonical operator: symmetric-only cross-interference."""
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * cross_strength
    G_merged = 0.5 * (Gi + Gj) + cross
    return lambda_min_S(G_merged)


def sym_alignment(Gi, Gj):
    Si = 0.5 * (Gi + Gi.T)
    Sj = 0.5 * (Gj + Gj.T)
    return np.trace(Si @ Sj) / (np.linalg.norm(Si, "fro") * np.linalg.norm(Sj, "fro") + 1e-12)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_model(seed, p_merge=None):
    """One run of the pair-trial simulation.
    p_merge=None        -> real model (Bendixson test decides merges).
    p_merge=float       -> fixed-p null (constant merge probability).
    """
    rng = np.random.default_rng(seed)
    element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]
    # actual margin = lambda_min(S) of each element (>= planted target)
    element_margin_actual = np.array([lambda_min_S(G) for G in element_G])

    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    attempt_success, attempt_min_margin, attempt_sym_align = [], [], []
    step_at_merge, attempts_per_trial = [], []

    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        Gi, Gj = element_G[i], element_G[j]
        mm = min(element_margin_actual[i], element_margin_actual[j])
        sa = sym_alignment(Gi, Gj)
        d = rng.uniform(2.0, 5.0)
        outcome, n_attempts = None, 0
        for step in range(MAX_STEPS):
            d += rng.standard_normal() * NOISE_STD
            d = max(d, 0.05)
            if d > D_MAX:
                outcome = "drifted_apart"
                break
            if d < D_CONTACT:
                n_attempts += 1
                if p_merge is None:
                    success = attempt_merge(Gi, Gj, CROSS_TERM_STRENGTH, rng) > 0
                else:
                    success = rng.random() < p_merge
                attempt_success.append(1 if success else 0)
                attempt_min_margin.append(mm)
                attempt_sym_align.append(sa)
                if success:
                    outcome = "merged"
                    step_at_merge.append(step)
                    break
                else:
                    d += abs(rng.standard_normal()) * 0.5
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1
        attempts_per_trial.append(n_attempts)

    total = sum(counts.values())
    return {
        "merged_frac": counts["merged"] / total,
        "per_attempt_rate": np.mean(attempt_success) if attempt_success else float("nan"),
        "median_step": np.median(step_at_merge) if step_at_merge else float("nan"),
        "mean_attempts": np.mean(attempts_per_trial),
        "corr_margin": pearson(attempt_min_margin, attempt_success),
        "corr_sym_align": pearson(attempt_sym_align, attempt_success),
    }


print("=" * 72)
print(" Experiment 1 -- Fixed-p null vs real kinetic model")
print("=" * 72)
print("Real model first (per-attempt rate p_real), then null with p = p_real.\n")

rows_real, rows_null = [], []
for seed in SEEDS:
    s_real = run_model(seed)
    s_null = run_model(seed, p_merge=s_real["per_attempt_rate"])
    rows_real.append(s_real)
    rows_null.append(s_null)

print(f"{'seed':>4}  {'model':<5}  {'merged':>7}  {'per-att':>8}  {'medstep':>7}  "
      f"{'att/trial':>9}  {'corr_margin':>11}  {'corr_align':>11}")
for seed, (r, n) in zip(SEEDS, zip(rows_real, rows_null)):
    for tag, s in (("real", r), ("null", n)):
        print(f"{seed:>4}  {tag:<5}  {s['merged_frac']*100:6.1f}%  {s['per_attempt_rate']*100:7.1f}%  "
              f"{s['median_step']:7.1f}  {s['mean_attempts']:9.2f}  "
              f"{s['corr_margin']:+11.3f}  {s['corr_sym_align']:+11.3f}")

print("\n-- aggregate (mean over seeds) --")
labels = [("merged_frac", "merged fraction"), ("per_attempt_rate", "per-attempt success"),
          ("median_step", "median step to merge"), ("mean_attempts", "attempts per trial"),
          ("corr_margin", "corr(min_margin, success)"), ("corr_sym_align", "corr(sym_align, success)")]
for key, label in labels:
    r = float(np.nanmean([s[key] for s in rows_real]))
    n = float(np.nanmean([s[key] for s in rows_null]))
    print(f"  {label:<28} real={r:+.4f}   null={n:+.4f}")

print("""
Interpretation guide:
- merged fraction / per-attempt / step timing are expected to MATCH: the null
  was built to reproduce the marginal stochastic process with the same p.
- corr(min_margin, success): if real is clearly > 0 while null ~ 0, the merge
  test reads real structure that a constant-p process cannot fake. If real ~ 0
  as well, success is effectively noise-driven.
""")
