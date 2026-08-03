import numpy as np

np.random.seed(42)

# ==========================================
# 1. Element generation: each element has BOTH
#    - a feature vector v_i, phase phi_i (for d_C, same as original Phase 2)
#    - a stability-generating matrix G_i (3x3), Bendixson-stable (from Phase 1 logic)
# ==========================================
NUM_ELEMENTS = 200
VECTOR_DIM = 4
GDIM = 3            # dimension of the stability matrix G
PAIR_COMBINATIONS = 1000
KAPPA0 = 1.0         # coupling strength scale, kappa(d) = KAPPA0 / d^2  (hard-core-like divergence)
C_ATTR = 5.5         # borrowed Yukawa strength [CALIBRATED] -- higher than before since actual
                     # per-pair strength = C_ATTR * resonance, and resonance < 1 always
SCREEN_MASS = 0.18   # screening mass m: attraction ~ -C * exp(-m*d) / d [CALIBRATED, not derived -- stated explicitly]

def random_stable_G(n, target_margin):
    """Build a matrix whose symmetric part is positive definite with min eigenvalue ~ target_margin,
    plus a random antisymmetric (rotational) component -- consistent with the Phase 1 Bendixson result
    that antisymmetric structure does not by itself destabilize."""
    A_rand = np.random.randn(n, n)
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = np.random.randn(n, n)
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti

def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()

def block_margin(Gi, Gj, kappa):
    n = Gi.shape[0]
    C = kappa * np.eye(n)
    top = np.hstack([Gi, C])
    bot = np.hstack([C.T, Gj])
    M = np.vstack([top, bot])
    return lambda_min_S(M)

def structural_compatibility(v_i, v_j, phase_i, phase_j):
    """Same d_C definition as the original Phase 2 script (gap + phase mismatch - resonance).
    Now also returns the resonance value itself, to drive attraction strength."""
    diff_vec = v_i - v_j
    gap = np.linalg.norm(diff_vec)
    cos_theta = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-9)
    phase_mismatch = 1.0 - cos_theta
    resonance = np.exp(-abs(phase_i - phase_j))
    d_c = gap + 0.8 * phase_mismatch - 0.5 * resonance
    return max(d_c, 0.1), resonance

def hybrid_potential(Gi, Gj, d_c, L_angular, kappa0, c_attr_base, screen_mass, resonance):
    """
    U_Agg = [derived repulsion from Bendixson margin degradation under coupling]
            + [attraction whose STRENGTH now scales with the pair's own phase resonance,
               instead of a universal borrowed constant C -- ties the attraction magnitude
               to a quantity the theory already defines, rather than an external import]
            + [standard centrifugal / spin barrier, L^2 / (2 d^2)]
    """
    kappa = kappa0 / (d_c ** 1.5)   # narrowed via the sweep: p>1 required, p=1.5 sufficient
    margin_far = min(lambda_min_S(Gi), lambda_min_S(Gj))
    margin_full = block_margin(Gi, Gj, kappa)
    repulsion = margin_far - margin_full
    c_attr = c_attr_base * resonance   # resonance in [0,1]; pair-specific, not universal
    attraction = -c_attr * np.exp(-screen_mass * d_c) / d_c
    spin_energy = (L_angular ** 2) / (2.0 * d_c ** 2 + 1e-9)
    return repulsion + attraction + spin_energy, repulsion, attraction, spin_energy

# ==========================================
# 2. Build the element population
#    (target_margin > 0 for all -- these are Phase-1 "survivors", already individually stable)
# ==========================================
element_vectors = np.random.randn(NUM_ELEMENTS, VECTOR_DIM)
element_phases = np.random.uniform(0, 2 * np.pi, NUM_ELEMENTS)
element_margins = np.random.uniform(0.1, 1.0, NUM_ELEMENTS)   # varying individual stability margins
element_G = [random_stable_G(GDIM, m) for m in element_margins]

# ==========================================
# 3. Simulation loop
# ==========================================
d_c_list, u_list, bound_status = [], [], []

for _ in range(PAIR_COMBINATIONS):
    idx1, idx2 = np.random.choice(NUM_ELEMENTS, size=2, replace=False)
    v1, v2 = element_vectors[idx1], element_vectors[idx2]
    p1, p2 = element_phases[idx1], element_phases[idx2]
    G1, G2 = element_G[idx1], element_G[idx2]

    dc, resonance = structural_compatibility(v1, v2, p1, p2)
    L_angular = np.random.uniform(0.1, 3.0)

    U, rep, attr, spin = hybrid_potential(G1, G2, dc, L_angular, KAPPA0, C_ATTR, SCREEN_MASS, resonance)

    d_c_list.append(dc)
    u_list.append(U)
    # Bound-state criterion: total energy negative -- standard physical definition of a bound pair
    # (not an arbitrary window like the original [-1.5, 0.5] cut)
    bound_status.append(1 if U < 0 else 0)

d_c_arr = np.array(d_c_list)
u_arr = np.array(u_list)
bound_arr = np.array(bound_status)

num_bound = np.sum(bound_arr == 1)
num_unbound = np.sum(bound_arr == 0)

print("=" * 55)
print(" SGOED Phase 2 v2 -- Hybrid (derived + borrowed) potential")
print("=" * 55)
print(f"Total pairings evaluated:        {PAIR_COMBINATIONS}")
print(f"Bound (U_total < 0):             {num_bound} ({num_bound/PAIR_COMBINATIONS*100:.2f}%)")
print(f"Unbound (U_total >= 0):          {num_unbound} ({num_unbound/PAIR_COMBINATIONS*100:.2f}%)")
print(f"d_C range: [{d_c_arr.min():.3f}, {d_c_arr.max():.3f}]")
print(f"U_total range: [{u_arr.min():.3f}, {u_arr.max():.3f}]")

# sanity check: is there an actual well? i.e. does U(d) trend negative at intermediate d
# and positive at both very small and very large d, on average?
bins = np.linspace(d_c_arr.min(), d_c_arr.max(), 12)
print("\nBinned average U_total vs d_C (checking for a genuine well shape):")
for i in range(len(bins) - 1):
    mask = (d_c_arr >= bins[i]) & (d_c_arr < bins[i + 1])
    if mask.sum() > 0:
        print(f"  d in [{bins[i]:.2f},{bins[i+1]:.2f}): mean U = {u_arr[mask].mean():.3f}  (n={mask.sum()})")
