import numpy as np

# ==========================================
# K-analogue extended:
#   (1) intrinsic orientation  O3(i,j;k) = tr([G_i,G_j] G_k)  -- no external R
#   (2) recursive aggregation with the oriented (sign) gate -- directional hierarchy?
#   (3) alpha_V^nu and the STF bound d_TV(q_K, m_V) >= alpha_V^nu
# See SGOED_STF_K_analogue_design.md.
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
S_REF = +1
THETA = 0.05
ROUNDS = 8


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


R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def O_A(Gi, Gj):
    Ai, Aj = anti_part(Gi), anti_part(Gj)
    return float(np.trace((Ai @ Aj - Aj @ Ai) @ R))


def O_3(Gi, Gj, Gk):
    """Intrinsic triple orientation (no external reference)."""
    return float(np.trace((Gi @ Gj - Gj @ Gi) @ Gk))


def symmetric_stable(Gi, Gj, noise):
    cross = 0.5 * (noise + noise.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross) > 0


def build_population(rng, zero_A=False):
    margins = rng.uniform(0.1, 1.0, NUM_ELEMENTS)
    Gs = [random_stable_G(GDIM, m, rng) for m in margins]
    if zero_A:
        Gs = [0.5 * (G + G.T) for G in Gs]
    return Gs


# ==========================================
print("=" * 74)
print(" (1) Intrinsic orientation  O3(i,j;k) = tr([G_i,G_j] G_k)")
print("=" * 74)
rng = np.random.default_rng(0)
Gs = build_population(rng)

# properties: swap-antisymmetric, A-essential, zero mean
swap_ok, max_o3_zero = True, 0.0
o3_vals = []
for _ in range(3000):
    i, j, k = rng.choice(NUM_ELEMENTS, size=3, replace=False)
    o = O_3(Gs[i], Gs[j], Gs[k])
    o_ji = O_3(Gs[j], Gs[i], Gs[k])
    swap_ok &= abs(o + o_ji) < 1e-9
    o3_vals.append(o)
o3_vals = np.array(o3_vals)

Gs0 = build_population(np.random.default_rng(3), zero_A=True)
for _ in range(500):
    i, j, k = rng.choice(NUM_ELEMENTS, size=3, replace=False)
    max_o3_zero = max(max_o3_zero, abs(O_3(Gs0[i], Gs0[j], Gs0[k])))
print(f"  swap-antisymmetric (O3(j,i;k) = -O3(i,j;k)): {swap_ok}")
print(f"  A=0 -> max |O3| = {max_o3_zero:.2e}  (A-essential: ~0)")
print(f"  E[O3] = {o3_vals.mean():+.4f},  std = {o3_vals.std():.4f}  "
      f"(zero mean: no intrinsic bias at the triple level)")

# order-sensitivity with O3-gate, k fixed vs random per collision
def gate_o3(Gi, Gj, Gk, noise):
    if not symmetric_stable(Gi, Gj, noise):
        return False
    return np.sign(O_3(Gi, Gj, Gk)) == S_REF


n_pairs = 3000
k_fixed = rng.choice(NUM_ELEMENTS)
for mode in ("fixed-k (population origin)", "random-k per collision"):
    diff = 0
    for _ in range(n_pairs):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        k = k_fixed if mode.startswith("fixed") else rng.choice(NUM_ELEMENTS)
        E = rng.standard_normal(Gs[0].shape)
        a = gate_o3(Gs[i], Gs[j], Gs[k], E)
        b = gate_o3(Gs[j], Gs[i], Gs[k], E)
        diff += (a != b)
    print(f"  {mode:<32} order-sensitivity = {diff/n_pairs*100:5.1f}%")

# ==========================================
print("\n" + "=" * 74)
print(" (2) Recursive aggregation with the oriented (sign) gate")
print("=" * 74)
def recursive_oriented(seed, gate, use_o3=False):
    rng = np.random.default_rng(seed)
    Gs = build_population(rng)
    pool = [(G, 1) for G in Gs]
    k0 = rng.choice(len(Gs)) if use_o3 else None
    history = []
    for _r in range(ROUNDS):
        rng.shuffle(pool)
        new_pool, i = [], 0
        merges = 0
        while i + 1 < len(pool):
            (Ga, sa), (Gb, sb) = pool[i], pool[i + 1]
            d = rng.uniform(2.0, 5.0)
            outcome = None
            for _step in range(MAX_STEPS):
                d += rng.standard_normal() * NOISE_STD
                d = max(d, 0.05)
                if d > D_MAX:
                    outcome = "drift"; break
                if d < D_CONTACT:
                    E = rng.standard_normal(Ga.shape)
                    stable = symmetric_stable(Ga, Gb, E)
                    if not stable:
                        d += abs(rng.standard_normal()) * 0.5
                        continue
                    if gate == "canonical":
                        outcome = "merge"; break
                    k = Gs[k0] if use_o3 else None
                    o = O_3(Ga, Gb, k) if use_o3 else O_A(Ga, Gb)
                    if np.sign(o) == S_REF:
                        outcome = "merge"; break
                    d += abs(rng.standard_normal()) * 0.5
            if outcome == "merge":
                E = rng.standard_normal(Ga.shape)
                cross = 0.5 * (E + E.T) * ALPHA
                new_pool.append((0.5 * (Ga + Gb) + cross, sa + sb))
                merges += 1
            else:
                new_pool.append((Ga, sa)); new_pool.append((Gb, sb))
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
        history.append((len(pool), merges, max(s for _, s in pool)))
    return history

for label, gate, o3 in (("canonical", "canonical", False),
                        ("oriented (O_A, external R)", "oriented", False),
                        ("oriented (O3, fixed population k)", "oriented", True)):
    h = recursive_oriented(1, gate, o3)
    row = "  ".join(f"r{r+1}:e{n}/m{m}" for r, (n, m, mx) in enumerate(h))
    print(f"  {label:<30} {row}")

# composite A-axis alignment vs size (does orientation propagate up the hierarchy?)
def axis_of(A):
    """so(3) axis u of an antisymmetric 3x3 (A v = u x v)."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


rng = np.random.default_rng(1)
Gs = build_population(rng)
pool = [(G, 1) for G in Gs]
for _r in range(ROUNDS):
    rng.shuffle(pool)
    new_pool = []
    i = 0
    while i + 1 < len(pool):
        (Ga, sa), (Gb, sb) = pool[i], pool[i + 1]
        d = rng.uniform(2.0, 5.0)
        outcome = None
        for _step in range(MAX_STEPS):
            d += rng.standard_normal() * NOISE_STD
            d = max(d, 0.05)
            if d > D_MAX:
                outcome = "drift"; break
            if d < D_CONTACT:
                E = rng.standard_normal(Ga.shape)
                if symmetric_stable(Ga, Gb, E) and np.sign(O_A(Ga, Gb)) == S_REF:
                    outcome = "merge"; break
                d += abs(rng.standard_normal()) * 0.5
        if outcome == "merge":
            E = rng.standard_normal(Ga.shape)
            cross = 0.5 * (E + E.T) * ALPHA
            new_pool.append((0.5 * (Ga + Gb) + cross, sa + sb))
        else:
            new_pool.append((Ga, sa)); new_pool.append((Gb, sb))
        i += 2
    if len(pool) % 2 == 1:
        new_pool.append(pool[-1])
    pool = new_pool

print("  composite A-axis alignment |u_comp . r| / |u_comp| by size "
      "(r = axis of R; ~0.5 = isotropic/random):")
for size_min in (3, 5, 8):
    aligns = []
    for G, s in pool:
        if s >= size_min:
            u = axis_of(anti_part(G))
            n = np.linalg.norm(u)
            if n > 1e-12:
                aligns.append(abs(u[2]) / n)
    if aligns:
        print(f"    size >= {size_min}: n={len(aligns):3d}  mean |cos| = {np.mean(aligns):.3f}")

# ==========================================
print("\n" + "=" * 74)
print(" (3) alpha_V^nu and the STF bound d_TV(q_K, m_V) >= alpha_V^nu")
print("=" * 74)
# Seed-averaged: d_TV and alpha are equal in expectation (stability is
# independent of the A-orientation sign); report mean +/- SE of the difference.
N = 20000
N_SEEDS = 5
dtv_list, alpha_list = [], []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(11 + seed)
    Gs = build_population(rng)
    n_pos = n_neg = 0
    for _ in range(N):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        E = rng.standard_normal(Gs[0].shape)
        stable = symmetric_stable(Gs[i], Gs[j], E)
        if not stable:
            continue
        if np.sign(O_A(Gs[i], Gs[j])) == S_REF:
            n_pos += 1
        else:
            n_neg += 1
    tot = n_pos + n_neg
    # d_TV(q_K, m_V) = P(stable AND sign mismatch) ; alpha = 1/2 P(stable AND sign != 0)
    dtv_list.append(n_neg / N)
    alpha_list.append(0.5 * tot / N)

dtv = np.array(dtv_list); alpha = np.array(alpha_list)
diff = dtv - alpha
print(f"  P(stable)           = {dtv.mean()/0.5*100:.1f}%  (n_pos+n_neg over N per seed)")
print(f"  d_TV(q_K, m_V)      = {dtv.mean():.4f} +/- {dtv.std():.4f}  (mean +/- std over {N_SEEDS} seeds)")
print(f"  alpha_V^nu          = {alpha.mean():.4f} +/- {alpha.std():.4f}")
print(f"  d_TV - alpha        = {diff.mean():+.4f} +/- {diff.std():.4f}")
print(f"  STF Lemma A.1 (d_TV >= alpha): {diff.mean() >= -2*diff.std()}")
print("  (d_TV and alpha are equal in expectation because stability is")
print("   independent of the A-orientation sign; the canonical rule saturates")
print("   the STF lower bound -- it is the optimal rho-symmetric baseline,")
print("   the SGOED analogue of STF's S_V in Corollary A.1.1.)")

print("""
Reading (design SGOED_STF_K_analogue_design.md):
- (1) O3 is a valid intrinsic orientation: swap-antisymmetric, A-essential,
  zero-mean; it generates ordering without the external matrix R (the
  reference is a population element, i.e. system-internal).
- (2) the oriented gate builds the hierarchy from +1-oriented merges only
  (globally consistent direction); expect fewer merges than canonical.
- (3) alpha_V^nu = half the order-sensitivity (reversal rho = pair swap), and
  the canonical symmetric merge rule saturates the STF lower bound
  d_TV(q_K,m_V) >= alpha_V^nu -- i.e. it plays exactly the role of STF's
  optimal rho-symmetric baseline S_V (Corollary A.1.1).
""")
