import numpy as np

# ==========================================
# Phase 1 -> Phase 2 interface consistency check
#
# Phase 2 assumes "individually stable survivors" (planted margins U(0.1,1.0)),
# but Phase 1's crystallization is probabilistic: ~22% of states crystallize
# while only ~0.3% have I_C > 0. What does Phase 1 actually hand to Phase 2,
# and does the kinetic model's conclusion (selectivity exists, structure
# matters) survive under Phase-1-real survivors?
#
# Populations compared (all at GDIM=3 so the states are directly feedable to
# Phase 2; the original Phase-1 script uses MATRIX_DIM=4 -- the dimension
# mismatch itself is part of the interface inconsistency, noted below):
#   P_planted : margins U(0.1, 1.0)   -- current canonical assumption
#   P_cryst   : Phase-1 "crystallized" states (passed the Fermi-Dirac test)
#   P_stable  : Phase-1 states with I_C > 0 (deterministic stability floor)
# ==========================================

GDIM = 3
NUM_STATES = 5000
T_CHAOS = 1.75
SCALE_LO, SCALE_HI = 0.5, 2.0
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
PAIR_TRIALS = 800
SEEDS = [1, 2, 3, 4, 5]
POP_SIZE = 300


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross)


# ---------- 1. run Phase 1 mechanism at GDIM=3, collect survivors ----------
print("=" * 72)
print(" 1. Phase-1 mechanism (GDIM=3) -- what survivors actually look like")
print("=" * 72)
rng = np.random.default_rng(42)
scales = rng.uniform(SCALE_LO, SCALE_HI, (NUM_STATES, 1, 1))
G_raw = rng.standard_normal((NUM_STATES, GDIM, GDIM)) * scales
i_c = np.array([lambda_min_S(G) for G in G_raw])
p_cryst = 1.0 / (1.0 + np.exp(-i_c / T_CHAOS))
crystallized = rng.random(NUM_STATES) < p_cryst
stable = i_c > 0

G_cryst = G_raw[crystallized]
G_stable = G_raw[stable]
m_cryst = i_c[crystallized]
m_stable = i_c[stable]

print(f"  states: {NUM_STATES},  crystallized (Fermi-Dirac test): "
      f"{crystallized.sum()} ({crystallized.mean()*100:.2f}%)")
print(f"  I_C > 0 (deterministic floor): {stable.sum()} ({stable.mean()*100:.2f}%)")
print(f"  of crystallized states, fraction with I_C > 0: "
      f"{(m_cryst > 0).mean()*100:.2f}%")
print(f"  margin of crystallized: mean={m_cryst.mean():+.2f}, "
      f"q50={np.median(m_cryst):+.2f}, q05/q95=({np.percentile(m_cryst,5):+.2f},"
      f"{np.percentile(m_cryst,95):+.2f})")
print(f"  margin of I_C>0 states: mean={m_stable.mean():+.3f}, "
      f"max={m_stable.max():+.3f}")
print(f"  (original Phase-1 manuscript uses MATRIX_DIM=4 and reports 0.28% "
      f"I_C>0; the dimension mismatch with GDIM=3 Phase 2 is part of the")
print(f"   interface inconsistency being checked)")

# ---------- 2. build the three populations ----------
print("\n" + "=" * 72)
print(" 2. Kinetic model (condition A) under the three survivor definitions")
print("=" * 72)
rng = np.random.default_rng(7)
planted_G = [random_stable_G(GDIM, m, rng) for m in rng.uniform(MARGIN_LOW, MARGIN_HIGH, POP_SIZE)]
if len(G_cryst) >= POP_SIZE:
    rng2 = np.random.default_rng(8)
    cryst_pop = G_cryst[rng2.choice(len(G_cryst), POP_SIZE, replace=False)]
else:
    cryst_pop = G_cryst          # use all (small population -> noisier, noted)
if len(G_stable) >= POP_SIZE:
    rng2 = np.random.default_rng(9)
    stable_pop = G_stable[rng2.choice(len(G_stable), POP_SIZE, replace=False)]
else:
    stable_pop = G_stable

pops = {"P_planted": planted_G, "P_cryst": cryst_pop, "P_stable": stable_pop}


def kinetic_condition_A(Gs, seed):
    rng = np.random.default_rng(seed)
    actual_margins = np.array([lambda_min_S(G) for G in Gs])
    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    per_attempt = []          # (success, min margin)
    attempts = []
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
        "drifted": counts["drifted_apart"] / total,
        "timeout": counts["timed_out"] / total,
        "per_attempt": pa[:, 0].mean() if len(pa) else float("nan"),
        "corr_margin": float(np.corrcoef(pa[:, 1], pa[:, 0])[0, 1]) if len(pa) > 3 else float("nan"),
        "attempts": np.mean(attempts),
    }


print(f"  {'population':<10} {'n_elem':>6} {'merged':>8} {'drifted':>8} "
      f"{'timeout':>8} {'per-att':>8} {'corr_marg':>9} {'att/trial':>9}")
results = {}
for name, Gs in pops.items():
    agg = {}
    for seed in SEEDS:
        r = kinetic_condition_A(Gs, seed)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    results[name] = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"  {name:<10} {len(Gs):>6} {results[name]['merged']*100:7.1f}% "
          f"{results[name]['drifted']*100:7.1f}% {results[name]['timeout']*100:7.1f}% "
          f"{results[name]['per_attempt']*100:7.1f}% "
          f"{results[name]['corr_margin']:+9.3f} {results[name]['attempts']:9.2f}")

print("""
Interpretation:
- P_planted (positive margins by construction) is the baseline the kinetic
  model was validated on.
- P_cryst are the states Phase 1 actually marks "crystallized"; most have
  negative margins, so they are NOT individually stable. If this is the
  survivor definition, the stability-gated aggregation premise fails.
- P_stable (I_C > 0) are individually stable but with margins clustered just
  above zero -- a much weaker population than the planted U(0.1,1.0).
- Compare merged fraction / corr(margin, success) across the three to see
  whether the selectivity conclusions of Section 4.6 survive under
  Phase-1-real survivors.
""")
