import numpy as np

# ==========================================
# Phase 1: Option A (two-gate separation) vs Option B (hard threshold) vs C
# (current), compared on the SAME ensembles -- WITHOUT modifying anything.
#
#   C (current): "crystallized" = gate-1 draw (Fermi-Dirac, probabilistic,
#                tail-dominated; ~22% but ~99% unstable)
#   A (two-gate): gate-1 labels a "structuring transition" (probabilistic);
#                gate-2 (I_C > 0) is the stability criterion Phase 2 consumes.
#                Two readings of A:
#                  A-clean : survivors = I_C > 0 (stability gate alone)
#                  A-strict: survivors = crystallized AND I_C > 0 (both gates)
#   B (hard): "crystallized" = "stable" = I_C > 0, deterministic.
#
# The comparison shows what each option calls "crystallized", what each hands
# to Phase 2, and the end-to-end kinetic consequence (condition A, canonical
# symmetric-noise merge test).
# ==========================================

N = 5000
T_CHAOS = 1.75
SCALE_LO, SCALE_HI = 0.5, 2.0
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
PAIR_TRIALS = 800
SEEDS = [1, 2, 3, 4, 5]
POP_CAP = 300


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA  # canonical symmetric-only interference
    return lambda_min_S(0.5 * (Gi + Gj) + cross)


def kinetic_condition_A(Gs, seed):
    rng = np.random.default_rng(seed)
    actual_margins = np.array([lambda_min_S(G) for G in Gs])
    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    per_attempt, attempts = [], []
    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        Gi, Gj = Gs[i], Gs[j]
        mm = min(actual_margins[i], actual_margins[j])
        d = rng.uniform(2.0, 5.0)
        outcome, n_att = None, 0
        for _step in range(MAX_STEPS):
            d += rng.standard_normal() * NOISE_STD
            d = max(d, 0.05)
            if d > D_MAX:
                outcome = "drifted_apart"
                break
            if d < D_CONTACT:
                n_att += 1
                ok = attempt_merge(Gi, Gj, rng) > 0
                per_attempt.append((1 if ok else 0, mm))
                if ok:
                    outcome = "merged"
                    break
                d += abs(rng.standard_normal()) * 0.5
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1
        attempts.append(n_att)
    total = sum(counts.values())
    pa = np.array(per_attempt, dtype=float)
    return {
        "merged": counts["merged"] / total,
        "per_attempt": pa[:, 0].mean() if len(pa) else float("nan"),
        "corr_margin": float(np.corrcoef(pa[:, 1], pa[:, 0])[0, 1]) if len(pa) > 3 else float("nan"),
    }


def pop_stats(i_c, mask, label):
    m = i_c[mask]
    return {
        "label": label,
        "count": int(mask.sum()),
        "frac": mask.sum() / len(i_c),
        "stable_share": (m > 0).mean() if len(m) else float("nan"),
        "mean_margin": m.mean() if len(m) else float("nan"),
        "median_margin": np.median(m) if len(m) else float("nan"),
    }


print("=" * 78)
print(" Option A (two-gate) vs Option B (hard) vs C (current) -- same ensembles")
print("=" * 78)

for dim in (3, 4):
    rng = np.random.default_rng(42)
    scales = rng.uniform(SCALE_LO, SCALE_HI, (N, 1, 1))
    G = rng.standard_normal((N, dim, dim)) * scales
    i_c = np.array([lambda_min_S(G[k]) for k in range(N)])
    stable = i_c > 0
    p = sigmoid(i_c / T_CHAOS)
    rng2 = np.random.default_rng(1)
    cryst = rng2.random(N) < p

    print(f"\n--- dim={dim} ---")
    rows = [
        pop_stats(i_c, cryst, "C: crystallized (gate-1 only)"),
        pop_stats(i_c, stable, "B: stable / A-clean survivors (I_C>0)"),
        pop_stats(i_c, cryst & stable, "A-strict survivors (crystallized & I_C>0)"),
    ]
    print(f"  {'population':<42} {'count':>6} {'frac':>7} {'stable%':>8} "
          f"{'mean marg':>10} {'med marg':>9}")
    for r in rows:
        print(f"  {r['label']:<42} {r['count']:>6} {r['frac']*100:6.2f}% "
              f"{r['stable_share']*100:7.1f}% {r['mean_margin']:+10.3f} "
              f"{r['median_margin']:+9.3f}")

print("""
Key structural fact: under option A (clean reading), Phase-2 survivors are the
stability gate I_C > 0 -- the SAME population as option B. The only difference
is what the word "crystallized" is allowed to mean (22% structuring transition
vs 0.2-2.3% stable matter). Option A-strict adds the requirement that a state
also passed the probabilistic gate-1, which keeps ~half the stable states.
""")

# ---------- end-to-end kinetic consequence (dim 3, populations large enough) ----------
print("=" * 78)
print(" End-to-end: kinetic model (condition A) on each option's survivors (dim 3)")
print("=" * 78)
dim = 3
rng = np.random.default_rng(42)
scales = rng.uniform(SCALE_LO, SCALE_HI, (N, 1, 1))
G = rng.standard_normal((N, dim, dim)) * scales
i_c = np.array([lambda_min_S(G[k]) for k in range(N)])
stable = i_c > 0
p = sigmoid(i_c / T_CHAOS)
cryst = np.random.default_rng(1).random(N) < p

G_cryst = G[cryst]
G_stable = G[stable]
G_strict = G[cryst & stable]
pops = {
    "C (crystallized)": G_cryst,
    "A-clean / B (I_C>0)": G_stable,
    "A-strict (cryst & I_C>0)": G_strict,
}

print(f"  {'population':<28} {'n':>5} {'merged':>8} {'per-att':>8} {'corr_marg':>9}")
for name, Gs in pops.items():
    if len(Gs) > POP_CAP:
        rng3 = np.random.default_rng(7)
        Gs = Gs[rng3.choice(len(Gs), POP_CAP, replace=False)]
    agg = {}
    for seed in SEEDS:
        r = kinetic_condition_A(Gs, seed)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    print(f"  {name:<28} {len(Gs):>5} {np.mean(agg['merged'])*100:7.1f}% "
          f"{np.mean(agg['per_attempt'])*100:7.1f}% "
          f"{np.mean(agg['corr_margin']):+9.3f}")

print("""
Reading:
- C (current) feeds Phase 2 with mostly-unstable elements: aggregation largely
  fails (per-attempt ~3%, merged ~15%, from the interface check).
- A-clean and B feed the SAME population (I_C > 0): the kinetic outcome is
  identical by construction. A-strict feeds a ~half-size subset of the same
  kind of elements (same margin distribution), so kinetically equivalent up to
  sampling.
- Therefore the A vs B choice is CONCEPTUAL, not numerical: it decides what the
  22% figure is allowed to mean (structuring-transition label vs removed), and
  whether the Section 2.2 probabilistic derivation keeps its object (gate-1)
  or becomes moot. The Phase-2 input and downstream numbers do not depend on it.
""")
