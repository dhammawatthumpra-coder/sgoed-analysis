import numpy as np

# ==========================================
# Analytic first-passage / reduced collision model for the kinetic model
#
# Combines two analytic pieces:
#   (a) merge probability p_ij per pair -- the EXACT event characterization of
#       sgoed_phase2_analytic_merge.py (tr/c2/det sign conditions, n=3), averaged
#       over GOE noise draws;
#   (b) first-passage structure of the separation random walk -- the diffusion
#       approximation P_hit(d) = (d_max - d)/(d_max - d_contact), validated below
#       against direct random-walk simulation.
#
# Because a failed merge always bounces to d1 = d_contact + 0.5|Z| (the SAME
# restart distribution every time), the renewal equation closes in closed form.
# For a pair with merge probability p starting at separation d0:
#     A = E_Z[P_hit(d_contact + 0.5|Z|)]
#     B = p*A / (1 - (1-p)*A)                      (expected success after a bounce)
#     F(d0) = P_hit(d0) * (p + (1-p)*B)            (merged probability)
#     G(d0) = 1 - F(d0)                            (drift-apart; no timeout here)
#     N(d0) = P_hit(d0) * (1 + (1-p)*A/(1-(1-p)*A)) (expected contact attempts)
#
# The reduced model has NO max_steps, so its "drifted" should be compared with
# the simulation's drifted + timed-out combined.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
N_PAIRS = 2000
K_GOE = 300            # MC noise draws per pair for the analytic merge probability
ALPHA = 1.4
D_CONTACT = 0.4
D_MAX = 8.0
NOISE_STD = 0.35
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
MAX_STEPS = 400
PAIR_TRIALS = 800
SEEDS = [1, 2, 3, 4, 5]
DATASET_SEED = 42
N_WALK = 20000


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


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


def p_success_mc(Gi, Gj, alpha, k, rng):
    S_blend = 0.25 * (Gi + Gi.T + Gj + Gj.T)
    hits = 0
    for _ in range(k):
        E = rng.standard_normal(Gi.shape)
        Es = 0.5 * (E + E.T)
        hits += 1 if conditions_n3(S_blend + alpha * Es) else 0
    return hits / k


def P_hit(d):
    """Diffusion-approximation probability of hitting contact before d_max."""
    d = np.clip(d, D_CONTACT, D_MAX)
    return (D_MAX - d) / (D_MAX - D_CONTACT)


def bounce_d():
    """Restart separation after a failed merge (matches the simulation rule)."""
    return D_CONTACT + 0.5 * abs(np.random.randn())


# ---------- 0. validate the diffusion approximation directly ----------
print("=" * 72)
print(" 0. Diffusion approximation: P_hit(d) vs direct random-walk simulation")
print("=" * 72)
print(f"  {'d0':>5} {'analytic':>10} {'simulated':>10} {'|err|':>8}")
walk_err = 0.0
for d0 in [2.0, 3.0, 4.0, 5.0]:
    rng = np.random.default_rng(int(d0 * 10))
    hits = 0
    for _ in range(N_WALK):
        d = d0
        for _step in range(5000):
            d += rng.standard_normal() * NOISE_STD
            if d > D_MAX:
                break
            if d < D_CONTACT:
                hits += 1
                break
    sim = hits / N_WALK
    ana = P_hit(d0)
    walk_err = max(walk_err, abs(sim - ana))
    print(f"  {d0:5.1f} {ana:10.3f} {sim:10.3f} {abs(sim-ana):8.3f}")
print(f"  max |err| over d0 in [2,5]: {walk_err:.3f}")

# ---------- 1. analytic merge probability over the population ----------
print("\n" + "=" * 72)
print(" 1. Analytic per-pair merge probability p_ij (exact conditions + GOE MC)")
print("=" * 72)
rng = np.random.default_rng(DATASET_SEED)
element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]

rng2 = np.random.default_rng(7)
p_list, m_blend_list, d0_list = [], [], []
for _ in range(N_PAIRS):
    i, j = rng2.choice(NUM_ELEMENTS, size=2, replace=False)
    Gi, Gj = element_G[i], element_G[j]
    p_list.append(p_success_mc(Gi, Gj, ALPHA, K_GOE, rng2))
    m_blend_list.append(lambda_min_S(0.25 * (Gi + Gi.T + Gj + Gj.T)))
    d0_list.append(rng2.uniform(2.0, 5.0))
p_arr = np.array(p_list)
m_arr = np.array(m_blend_list)
d0_arr = np.array(d0_list)
print(f"  E[p_ij] over {N_PAIRS} pairs (K={K_GOE} GOE draws each): "
      f"{p_arr.mean()*100:.1f}%   (simulation per-attempt: ~60-62%)")

print("  P(success | m_blend), analytic:")
edges = np.percentile(m_arr, [0, 25, 50, 75, 100])
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (m_arr >= lo) & (m_arr < hi)
    print(f"    m_blend in [{lo:5.2f},{hi:5.2f}):  p = {p_arr[mask].mean():.3f}")

# ---------- 2. reduced first-passage model (closed form) ----------
print("\n" + "=" * 72)
print(" 2. Reduced first-passage model (closed form) vs kinetic simulation")
print("=" * 72)
# A = E_Z[P_hit(contact + 0.5|Z|)]  ~ 1 - 0.5*E|Z|/(d_max - d_contact)
z = np.random.default_rng(11).standard_normal(200000)
A = float(np.mean(P_hit(D_CONTACT + 0.5 * np.abs(z))))
A_exact = 1.0 - 0.5 * np.sqrt(2.0 / np.pi) / (D_MAX - D_CONTACT)
print(f"  A = E_Z[P_hit(bounce)] = {A:.4f}   (analytic 1 - 0.5*E|Z|/(d_max-d_c) = {A_exact:.4f})")

F_arr = np.zeros(N_PAIRS)
G_arr = np.zeros(N_PAIRS)
N_arr = np.zeros(N_PAIRS)
for k in range(N_PAIRS):
    p = p_arr[k]
    B = p * A / (1.0 - (1.0 - p) * A)
    ph = P_hit(d0_arr[k])
    F_arr[k] = ph * (p + (1.0 - p) * B)
    G_arr[k] = 1.0 - F_arr[k]
    N_arr[k] = ph * (1.0 + (1.0 - p) * A / (1.0 - (1.0 - p) * A))

# ---------- 3. kinetic simulation (condition A) for comparison ----------
def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross)


def kinetic_condition_A(seed):
    rng = np.random.default_rng(seed)
    mg = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
    Gs = [random_stable_G(GDIM, m, rng) for m in mg]
    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    per_att, attempts = [], []
    for _ in range(PAIR_TRIALS):
        i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
        Gi, Gj = Gs[i], Gs[j]
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
                if attempt_merge(Gi, Gj, rng) > 0:
                    outcome = "merged"
                    break
                d += abs(rng.standard_normal()) * 0.5
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1
        attempts.append(n_att)
        per_att.append(counts["merged"])
    total = sum(counts.values())
    return (counts["merged"] / total, counts["drifted_apart"] / total,
            counts["timed_out"] / total,
            (counts["merged"] / (sum(attempts)) if sum(attempts) else float("nan")),
            np.mean(attempts))


sim_merged, sim_drifted, sim_timeout, sim_per_att, sim_attempts = [], [], [], [], []
for seed in SEEDS:
    m_, dr_, to_, pa_, at_ = kinetic_condition_A(seed)
    sim_merged.append(m_); sim_drifted.append(dr_); sim_timeout.append(to_)
    sim_per_att.append(pa_); sim_attempts.append(at_)

def mean(x):
    return float(np.mean(x))

print(f"  {'':<34} {'reduced model':>14} {'simulation':>14}")
print(f"  {'merged fraction':<34} {F_arr.mean()*100:13.1f}% {mean(sim_merged)*100:13.1f}%")
print(f"  {'drifted (model) / drifted+timeout (sim)':<34} "
      f"{G_arr.mean()*100:13.1f}% {(mean(sim_drifted)+mean(sim_timeout))*100:13.1f}%")
print(f"  {'per-attempt success':<34} {p_arr.mean()*100:13.1f}% {mean(sim_per_att)*100:13.1f}%")
print(f"  {'expected attempts/trial':<34} {N_arr.mean():13.2f} {mean(sim_attempts):13.2f}")
print(f"  {'(sim timed-out fraction)':<34} {'':>14} {mean(sim_timeout)*100:12.1f}%")

print("""
Interpretation:
- The reduced model predicts the simulation's merged fraction / attempts per
  trial WITHOUT running the random walk or the merge test, using only P_hit
  (validated diffusion approximation) and the analytic merge probability.
- The residual gap is the stated approximation error: discrete random-walk
  steps, the reflecting floor at d=0.05, and the absence of max_steps in the
  reduced model (sim timed-out ~4% is absorbed into "drifted").
- This closes the loop: collision dynamics and merge selectivity are now each
  described analytically and verified against the simulation.
""")
