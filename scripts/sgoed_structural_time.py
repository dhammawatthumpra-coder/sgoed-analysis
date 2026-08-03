import numpy as np

# ==========================================
# Structural Time: is there a universal clock in the aggregation process?
#
# Design idea: "time" without seconds = an INTRINSIC, observer-independent
# quantity that increases monotonically along every structural transformation.
# If such a quantity exists, it is a "universal clock" (เวลากลาง): every
# observer of the same process agrees on its value, and it orders the
# transformations without any external t.
#
# Candidates tested along every merge path (a,b) -> c:
#   depth     : merge-tree rank (intrinsic, well-defined within a tree)
#   size      : composite size (trivially monotone)
#   H(S)      : spectral entropy of the symmetric part (mixing spectra should
#               increase it)  <- the non-trivial candidate
#   ||G||_F   : Frobenius norm (averaging can decrease it)
#   margin    : lambda_min(S) (known NOT to be monotone)
# Also: can any candidate give a GLOBAL ordering across all entities
# (a universal rank shared by every composite, not just within a tree)?
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
ROUNDS = 8
SEEDS = [1, 2, 3]


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def spectral_entropy(M):
    """Entropy of the normalized absolute eigenvalue spectrum of the
    symmetric part. H in [0, log(n)]; 0 = fully ordered, log(n) = maximal."""
    S = 0.5 * (M + M.T)
    lam = np.abs(np.linalg.eigvalsh(S))
    s = lam.sum()
    if s < 1e-12:
        return 0.0
    p = lam / s
    return float(-np.sum(p * np.log(p)))


def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross) > 0


def run_with_clocks(seed):
    rng = np.random.default_rng(seed)
    element_G = [random_stable_G(GDIM, m, rng) for m in rng.uniform(0.1, 1.0, NUM_ELEMENTS)]
    # entity = (G, size, depth, elements)
    pool = [(G, 1, 0, frozenset([k])) for k, G in enumerate(element_G)]
    merges = []   # (size_a, size_b, H_a, H_b, H_c, depth_c, margin_a, margin_b, margin_c)
    for _r in range(ROUNDS):
        rng.shuffle(pool)
        new_pool, i = [], 0
        while i + 1 < len(pool):
            a, b = pool[i], pool[i + 1]
            merged = False
            d = rng.uniform(2.0, 5.0)
            for _step in range(MAX_STEPS):
                d += rng.standard_normal() * NOISE_STD
                d = max(d, 0.05)
                if d > D_MAX:
                    break
                if d < D_CONTACT:
                    if attempt_merge(a[0], b[0], rng):
                        merged = True
                        break
                    d += abs(rng.standard_normal()) * 0.5
            if merged:
                E = rng.standard_normal(a[0].shape)
                cross = 0.5 * (E + E.T) * ALPHA
                Gm = 0.5 * (a[0] + b[0]) + cross
                merges.append((a[1], b[1],
                               spectral_entropy(a[0]), spectral_entropy(b[0]), spectral_entropy(Gm),
                               max(a[2], b[2]) + 1,
                               lambda_min_S(a[0]), lambda_min_S(b[0]), lambda_min_S(Gm)))
                new_pool.append((Gm, a[1] + b[1], max(a[2], b[2]) + 1, a[3] | b[3]))
            else:
                new_pool.append(a); new_pool.append(b)
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
    return merges


all_merges = []
for seed in SEEDS:
    all_merges.extend(run_with_clocks(seed))

H = np.array([(m[2], m[3], m[4]) for m in all_merges])
marg = np.array([(m[6], m[7], m[8]) for m in all_merges])

print("=" * 74)
print(" Structural Time: universal-clock candidates in the aggregation process")
print("=" * 74)
print(f"  total merges tracked: {len(all_merges)}\n")

# H monotonicity: is H(c) > max(H(a), H(b)) for EVERY merge?
h_increase = H[:, 2] > np.maximum(H[:, 0], H[:, 1])
h_strict = h_increase.mean()
print("-- candidate: spectral entropy H(S) --")
print(f"  H(c) > max(H(a),H(b)) in {h_increase.sum()}/{len(all_merges)} "
      f"merges ({h_increase.mean()*100:.1f}%)")
print(f"  mean H increase per merge: {np.mean(H[:,2]-np.maximum(H[:,0],H[:,1])):+.4f}")
print(f"  mean H(a)={H[:,0].mean():.4f}, H(b)={H[:,1].mean():.4f}, H(c)={H[:,2].mean():.4f}")

print("\n-- candidate: margin lambda_min(S) (for contrast) --")
m_inc = marg[:, 2] > np.maximum(marg[:, 0], marg[:, 1])
print(f"  margin(c) > max(a,b) in {m_inc.sum()}/{len(all_merges)} "
      f"({m_inc.mean()*100:.1f}%)  (known NOT monotone)")

print("\n-- strict monotonicity check along depth --")
# among merges that increase depth, is H strictly increasing with depth?
depth = np.array([m[5] for m in all_merges])
h_all = np.array([m[4] for m in all_merges])
for dmin, dmax in ((1, 3), (3, 5), (5, 8)):
    m = (depth >= dmin) & (depth < dmax)
    if m.sum():
        print(f"  merges at depth [{dmin},{dmax}): n={m.sum()}, "
              f"mean H(c) = {h_all[m].mean():.4f}")

print("""
Honest reading:
- If H(c) > max(H(a),H(b)) holds for (nearly) every merge, spectral entropy is
  a Lyapunov-like universal clock: intrinsic, observer-independent, increasing
  with every structural transformation -> a candidate "เวลากลาง" (universal
  time) that needs no seconds and no observer.
- Depth/size are monotone but only rank WITHIN a merge tree; H, if global, can
  also order entities ACROSS trees (a universal rank).
- Margin is confirmed NOT to be a clock (known result).
- Caveat: H gives an ORDER (rank), not a metric (no "distance" between ranks);
  that matches the STF view (ordering, not duration).
""")
