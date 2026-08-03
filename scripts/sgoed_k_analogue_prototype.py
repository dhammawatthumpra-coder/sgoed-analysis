import numpy as np

# ==========================================
# K-analogue prototype: antisymmetric-driven ordering (STF connection)
# See SGOED_STF_K_analogue_design.md.
#
# Orientation:  O_A(i,j) = tr([A_i,A_j] R), R = fixed antisymmetric reference.
# Merge rule:   symmetric-part stability (as in Phase 2) AND sign(O_A(i,j)) = s_ref.
#
# Tests:
#   T1 order-sensitivity  : P(outcome(i,j) != outcome(j,i))  (>0 oriented, =0 canonical)
#   T2 A=0 control        : zeroing A kills the sign-gate mechanism
#   T3 E[O_A] ~ 0         : ordering is observer-supplied (STF Source b), not intrinsic
#   T4 kinetic fractions  : canonical vs sign-gate vs magnitude-gate
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
PAIR_TRIALS = 800
SEEDS = [1, 2, 3, 4, 5]
THETA = 0.05          # magnitude-gate threshold
S_REF = +1            # reference orientation sign (convention)


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def anti_part(M):
    return 0.5 * (M - M.T)


# observer's reference orientation (so(3) generator about z)
R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def O_A(Gi, Gj):
    Ai, Aj = anti_part(Gi), anti_part(Gj)
    return float(np.trace((Ai @ Aj - Aj @ Ai) @ R))


def symmetric_stable(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross) > 0


def merge_decision(Gi, Gj, rng, gate, noise=None):
    """gate: 'canonical' | 'sign' | 'magnitude'.
    noise: optional pre-drawn E, so two orientations can be compared with the
    SAME noise realization (isolating orientation as the only variable)."""
    E = noise if noise is not None else rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    if not lambda_min_S(0.5 * (Gi + Gj) + cross) > 0:
        return False
    if gate == "canonical":
        return True
    o = O_A(Gi, Gj)
    if gate == "sign":
        return np.sign(o) == S_REF        # sign(0)=0 fails for S_REF=+1
    if gate == "magnitude":
        return abs(o) > THETA
    raise ValueError(gate)


def build_population(rng, zero_A=False):
    margins = rng.uniform(0.1, 1.0, NUM_ELEMENTS)
    Gs = [random_stable_G(GDIM, m, rng) for m in margins]
    if zero_A:
        Gs = [0.5 * (G + G.T) for G in Gs]   # kill all antisymmetric parts
    return Gs


def kinetic_condition_A(Gs, seed, gate):
    rng = np.random.default_rng(seed)
    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        Gi, Gj = Gs[i], Gs[j]
        d = rng.uniform(2.0, 5.0)
        outcome = None
        for _step in range(MAX_STEPS):
            d += rng.standard_normal() * NOISE_STD
            d = max(d, 0.05)
            if d > D_MAX:
                outcome = "drifted_apart"
                break
            if d < D_CONTACT:
                if merge_decision(Gi, Gj, rng, gate):
                    outcome = "merged"
                    break
                d += abs(rng.standard_normal()) * 0.5
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1
    return counts["merged"] / PAIR_TRIALS


print("=" * 74)
print(" K-analogue prototype: antisymmetric-driven ordering (STF connection)")
print("=" * 74)

# ---------- T1: order-sensitivity ----------
print("\n-- T1: order-sensitivity  P(outcome(i,j) != outcome(j,i)) --")
rng = np.random.default_rng(0)
Gs = build_population(rng)
n_pairs = 3000
for gate in ("canonical", "sign", "magnitude"):
    diff = 0
    for _ in range(n_pairs):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        E = rng.standard_normal(Gs[0].shape)   # one noise realization per pair
        o_ij = merge_decision(Gs[i], Gs[j], rng, gate, noise=E)
        o_ji = merge_decision(Gs[j], Gs[i], rng, gate, noise=E)
        diff += (o_ij != o_ji)
    print(f"  gate={gate:<10} order-sensitivity = {diff/n_pairs*100:5.1f}%  "
          f"(design: >0 for oriented, 0 for canonical)")

# ---------- T2: A=0 control ----------
print("\n-- T2: A=0 control (all antisymmetric parts zeroed) --")
rng = np.random.default_rng(3)
Gs0 = build_population(rng, zero_A=True)
o_zero = [O_A(Gs0[i], Gs0[j]) for _ in range(1000)
          for i, j in [rng.choice(NUM_ELEMENTS, size=2, replace=False)]]
print(f"  max |O_A| with A=0: {max(abs(x) for x in o_zero):.2e}  (must be ~0)")
for gate in ("sign", "magnitude"):
    merged = kinetic_condition_A(Gs0, 1, gate)
    print(f"  gate={gate:<10} with A=0: merged = {merged*100:.1f}%  "
          f"(design: sign-gate -> 0%; magnitude-gate -> 0% since |O|<theta)")

# ---------- T3: E[O_A] ----------
print("\n-- T3: E[O_A] over the population (intrinsic bias?) --")
rng = np.random.default_rng(5)
Gs = build_population(rng)
o_vals = [O_A(Gs[i], Gs[j]) for _ in range(5000)
          for i, j in [rng.choice(NUM_ELEMENTS, size=2, replace=False)]]
o_vals = np.array(o_vals)
print(f"  E[O_A] = {o_vals.mean():+.4f},  std = {o_vals.std():.4f},  "
      f"P(O_A>0) = {(o_vals>0).mean()*100:.1f}%")
print("  (design: E[O_A] ~ 0 -> ordering is observer-supplied (STF Source b),")
print("   not intrinsic bias)")

# ---------- T4: kinetic merged fraction ----------
print("\n-- T4: kinetic merged fraction (condition A, 5 seeds) --")
rng = np.random.default_rng(7)
Gs = build_population(rng)
for gate in ("canonical", "magnitude", "sign"):
    fracs = [kinetic_condition_A(Gs, s, gate) for s in SEEDS]
    print(f"  gate={gate:<10} merged = {np.mean(fracs)*100:5.1f}%  "
          f"(seeds {[f'{f*100:.1f}' for f in fracs]})")

print("""
Reading (design SGOED_STF_K_analogue_design.md):
- T1: sign-gate gives order-sensitivity > 0 (A generates ordering); canonical gives 0.
- T2: A is the operative ingredient (zeroing it kills the oriented mechanism).
- T3: E[O_A] ~ 0 -> ordering comes from the reference R (STF Source b), matching
  STF's Limitation 10 (intrinsic complexity alone needs a reference/boundary).
- T4: magnitude-gate keeps the merged fraction near canonical (A matters, no
  ordering); sign-gate drops it (ordering rejects half the stable pairs).
This is a toy demonstration that A CAN generate ordering -- not a claim that
this is the correct physical mechanism.
""")
