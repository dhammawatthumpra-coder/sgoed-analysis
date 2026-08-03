import numpy as np

np.random.seed(42)

# ==========================================
# Phase 1 v2: I_C = lambda_min(S), threshold FIXED at 0 (from Bendixson's theorem,
# not a free-fit parameter as in the original version).
# Crystallization probability derived from two-state (Fermi-Dirac) statistics:
#   P(crystallize) = 1 / (1 + exp(-mu / T_chaos))
# where mu = I_C = lambda_min(S) and T_chaos is the one remaining physical parameter
# (intensity of background fluctuation -- NOT a fitted "sharpness" knob).
# ==========================================

NUM_STATES = 5000
MATRIX_DIM = 4
T_CHAOS = 1.75  # DERIVED: E[scale^2] of the same G_raw-generating distribution (U(0.5,2.0) squared),
                # i.e. T_chaos is identified with the natural variance of the primordial continuum
                # itself, not an independently fitted "sharpness" knob.

def compute_I_C_derived(G):
    S = 0.5 * (G + G.T)
    return np.linalg.eigvalsh(S).min()

def crystallization_probability(mu, T_chaos):
    return 1.0 / (1.0 + np.exp(-mu / T_chaos))

i_c_list, prob_list, status_list = [], [], []

for _ in range(NUM_STATES):
    G_raw = np.random.randn(MATRIX_DIM, MATRIX_DIM) * np.random.uniform(0.5, 2.0)
    mu = compute_I_C_derived(G_raw)
    p = crystallization_probability(mu, T_CHAOS)
    is_crystallized = 1 if np.random.rand() < p else 0
    i_c_list.append(mu)
    prob_list.append(p)
    status_list.append(is_crystallized)

i_c_arr = np.array(i_c_list)
status_arr = np.array(status_list)

num_crystallized = np.sum(status_arr == 1)
num_dissolved = np.sum(status_arr == 0)

print("=" * 55)
print(" SGOED Phase 1 v2 -- Bendixson-derived I_C + Fermi-Dirac P")
print("=" * 55)
print(f"Total chaos states evaluated:   {NUM_STATES}")
print(f"Crystallized elements:          {num_crystallized} ({num_crystallized/NUM_STATES*100:.2f}%)")
print(f"Dissolved states:                {num_dissolved} ({num_dissolved/NUM_STATES*100:.2f}%)")
print(f"Threshold used: 0 (fixed by Bendixson's theorem, not fitted)")
print(f"T_chaos (only free physical parameter): {T_CHAOS}")
print(f"mu=I_C range: [{i_c_arr.min():.3f}, {i_c_arr.max():.3f}]")
print(f"Fraction with mu > 0 (deterministic stability floor): {np.mean(i_c_arr>0)*100:.2f}%")
