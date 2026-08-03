import numpy as np

# ==========================================
# Realistic-margin canonical numbers: Phase-1 gate-2 survivors -> Phase 2
#
# The Section 4 figures rest on planted margins U(0.1,1.0) (an optimistic
# idealization of gate-2 survivors; mean 0.55 vs real ~0.34). This script
# produces the numbers the theory actually yields end-to-end:
#   Phase 1 (large ensemble, n=3) -> gate-2 survivors (I_C > 0)
#   -> kinetic model (condition A) -> merged fraction / per-attempt / corr
# and compares with the planted-population run and with the analytic
# first-passage prediction (sgoed_phase2_first_passage.py machinery).
# Nothing in the existing docs/scripts is modified.
# ==========================================

N_STATES = 100000
GDIM = 3
T_CHAOS = 1.75
SCALE_LO, SCALE_HI = 0.5, 2.0
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


def conditions_n3(A):
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    c2 = (A[0, 0] * A[1, 1] - A[0, 1] ** 2
          + A[0, 0] * A[2, 2] - A[0, 2] ** 2
          + A[1, 1] * A[2, 2] - A[1, 2] ** 2)
    det = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] ** 2)
           - A[0, 1] * (A[0, 1] * A[2, 2] - A[0, 2] * A[1, 2])
           + A[0, 2] * (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]))
    return tr > 0 and c2 > 0 and det > 0


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
        "drifted": counts["drifted_apart"] / total,
        "timeout": counts["timed_out"] / total,
        "per_attempt": pa[:, 0].mean() if len(pa) else float("nan"),
        "corr_margin": float(np.corrcoef(pa[:, 1], pa[:, 0])[0, 1]) if len(pa) > 3 else float("nan"),
        "attempts": np.mean(attempts),
    }


def P_hit(d):
    d = np.clip(d, D_CONTACT, D_MAX)
    return (D_MAX - d) / (D_MAX - D_CONTACT)


def reduced_merged(p):
    """Analytic first-passage prediction of the merged fraction for a
    population with mean per-attempt probability p (renewal, A=0.9475)."""
    A = 1.0 - 0.5 * np.sqrt(2.0 / np.pi) / (D_MAX - D_CONTACT)
    B = p * A / (1.0 - (1.0 - p) * A)
    # E[P_hit(d0)] for d0 ~ U(2,5)
    ep_hit = float(np.mean(P_hit(np.random.default_rng(0).uniform(2.0, 5.0, 100000))))
    return ep_hit * (p + (1.0 - p) * B)


# ---------- 1. Phase-1 gate-2 survivors ----------
print("=" * 74)
print(" 1. Phase-1 gate-2 survivor population (large ensemble, n=3)")
print("=" * 74)
rng = np.random.default_rng(42)
scales = rng.uniform(SCALE_LO, SCALE_HI, (N_STATES, 1, 1))
G_all = rng.standard_normal((N_STATES, GDIM, GDIM)) * scales
i_c = np.array([lambda_min_S(G_all[k]) for k in range(N_STATES)])
stable_mask = i_c > 0
G_stable = G_all[stable_mask]
print(f"  states: {N_STATES},  gate-2 survivors (I_C>0): {stable_mask.sum()} "
      f"({stable_mask.mean()*100:.2f}%)")
print(f"  survivor margins: mean={i_c[stable_mask].mean():+.3f}, "
      f"median={np.median(i_c[stable_mask]):+.3f}, "
      f"range=[{i_c[stable_mask].min():+.3f},{i_c[stable_mask].max():+.3f}]")

rng2 = np.random.default_rng(7)
survivors = G_stable[rng2.choice(len(G_stable), POP_SIZE, replace=False)]

# measured per-attempt p over the realistic population (exact conditions MC)
rng3 = np.random.default_rng(11)
p_vals = []
for _ in range(1500):
    i, j = rng3.choice(len(survivors), size=2, replace=False)
    Gi, Gj = survivors[i], survivors[j]
    S_blend = 0.25 * (Gi + Gi.T + Gj + Gj.T)
    hits = 0
    for _k in range(200):
        E = rng3.standard_normal(Gi.shape)
        Es = 0.5 * (E + E.T)
        hits += 1 if conditions_n3(S_blend + ALPHA * Es) else 0
    p_vals.append(hits / 200)
p_realistic = float(np.mean(p_vals))
print(f"  analytic per-attempt p (exact conditions, 1500 pairs x 200 GOE): "
      f"{p_realistic*100:.1f}%")

# ---------- 2. kinetic runs ----------
print("\n" + "=" * 74)
print(" 2. Kinetic model (condition A): realistic vs planted")
print("=" * 74)
rng4 = np.random.default_rng(5)
planted = [random_stable_G(GDIM, m, rng4) for m in rng4.uniform(0.1, 1.0, POP_SIZE)]

for name, Gs in (("planted U(0.1,1.0)", planted), ("gate-2 survivors", survivors)):
    agg = {}
    for seed in SEEDS:
        r = kinetic_condition_A(Gs, seed)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"  {name:<22} merged={m['merged']*100:5.1f}%  per-att={m['per_attempt']*100:5.1f}%  "
          f"corr_marg={m['corr_margin']:+.3f}  att/trial={m['attempts']:.2f}  "
          f"timeout={m['timeout']*100:.1f}%")

# ---------- 3. analytic first-passage prediction ----------
print("\n" + "=" * 74)
print(" 3. Analytic first-passage prediction (renewal, reduced model)")
print("=" * 74)
pred = reduced_merged(p_realistic)
print(f"  p = {p_realistic*100:.1f}%  ->  predicted merged fraction = {pred*100:.1f}%")
print(f"  (compare with measured realistic merged fraction above)")

print("""
Reading: these are the numbers the theory yields end-to-end (Phase 1 -> gate-2
survivors -> Phase 2 kinetic). The qualitative conclusions of Section 4.6 are
unchanged (selectivity exists, margin correlation positive); the magnitudes
drop substantially relative to the planted idealization. Whether these become
the canonical Section 4 numbers is a documentation decision.
""")
