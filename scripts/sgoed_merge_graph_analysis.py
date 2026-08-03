import numpy as np

# ==========================================
# Merge-graph analysis: the "order of change" as a measurable object
#
# Tests whether the recursive aggregation hierarchy (the SGOED "order of
# change") has any spacetime-like / locality-like signature:
#   A. structural clustering : are elements within a composite closer in
#      state space (||G_i - G_j||_F) than elements across composites?
#   B. locality selection    : do merged pairs have smaller structural
#      distance than random pairs?
#   C. coagulation nulls     : is the final size distribution different from
#      generic Smoluchowski coagulation (constant/additive/multiplicative)?
#   D. tree shape            : merge-tree depth per composite.
# Honest framing: this measures the structural order the formalism produces;
# whether it resembles a manifold is assessed, not assumed.
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


def attempt_merge(Gi, Gj, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * ALPHA
    return lambda_min_S(0.5 * (Gi + Gj) + cross) > 0


def structural_dist(Ga, Gb):
    return float(np.linalg.norm(Ga - Gb, "fro"))


def recursive_with_tree(seed):
    """Recursive aggregation tracking the merge tree.
    entity = (G, size, elements(set of original indices), depth_of_tree)."""
    rng = np.random.default_rng(seed)
    element_G = [random_stable_G(GDIM, m, rng) for m in rng.uniform(0.1, 1.0, NUM_ELEMENTS)]
    pool = [(G, 1, frozenset([k]), 0) for k, G in enumerate(element_G)]
    merge_distances = []     # structural distance of each successful merge
    n_merges = 0
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
                merge_distances.append(structural_dist(a[0], b[0]))
                new_pool.append((Gm, a[1] + b[1], a[2] | b[2], max(a[3], b[3]) + 1))
                n_merges += 1
            else:
                new_pool.append(a); new_pool.append(b)
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
    return element_G, pool, merge_distances, n_merges


def coagulation_null(seed, kernel, c, rounds=ROUNDS):
    """Smoluchowski null with the same pair-up structure; merge prob = c*kernel."""
    rng = np.random.default_rng(seed)
    pool = [1] * NUM_ELEMENTS   # sizes only
    for _r in range(rounds):
        rng.shuffle(pool)
        new_pool, i = [], 0
        while i + 1 < len(pool):
            a, b = pool[i], pool[i + 1]
            if kernel == "constant":
                p = c
            elif kernel == "additive":
                p = c * (a + b) / 2.0
            elif kernel == "multiplicative":
                p = c * (a * b) / (NUM_ELEMENTS ** 2)
            else:
                raise ValueError(kernel)
            if rng.random() < min(p, 1.0):
                new_pool.append(a + b)
            else:
                new_pool.append(a); new_pool.append(b)
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
    return np.array(pool)


print("=" * 74)
print(" Merge-graph analysis: the SGOED 'order of change' as a measurable object")
print("=" * 74)

# ---- A/B: structural clustering + locality selection ----
intra_list, inter_list, mratio_list = [], [], []
for seed in SEEDS:
    element_G, pool, merge_d, n_merges = recursive_with_tree(seed)
    # composites (size >= 2)
    comps = [e for e in pool if e[1] >= 2]
    # intra: mean pairwise distance within composites
    intra = []
    for G, s, els, depth in comps:
        els = list(els)
        if len(els) < 2:
            continue
        ds = [structural_dist(element_G[i], element_G[j])
              for x, i in enumerate(els) for j in els[x + 1:]]
        intra.append(np.mean(ds))
    # inter: sample cross-composite pairs
    rng = np.random.default_rng(seed + 100)
    inter = []
    if len(comps) >= 2:
        for _ in range(500):
            c1, c2 = rng.choice(len(comps), size=2, replace=False)
            i = rng.choice(list(comps[c1][2]))
            j = rng.choice(list(comps[c2][2]))
            inter.append(structural_dist(element_G[i], element_G[j]))
    # merged-pair vs random-pair distance
    rng2 = np.random.default_rng(seed + 200)
    rand_d = [structural_dist(element_G[i], element_G[j])
              for _ in range(500) for i, j in [rng2.choice(NUM_ELEMENTS, size=2, replace=False)]]
    intra_list.append(np.mean(intra) if intra else float("nan"))
    inter_list.append(np.mean(inter) if inter else float("nan"))
    mratio_list.append(np.mean(merge_d) / np.mean(rand_d))

print("\n-- A. structural clustering (locality-like?) --")
print(f"  mean intra-composite distance : {np.nanmean(intra_list):.3f}")
print(f"  mean inter-composite distance : {np.nanmean(inter_list):.3f}")
print(f"  clustering ratio inter/intra  : {np.nanmean(inter_list)/np.nanmean(intra_list):.2f}"
      f"  (>>1 = composites are clusters in state space)")
print("\n-- B. locality selection --")
print(f"  merged-pair / random-pair distance : {np.mean(mratio_list):.3f}"
      f"  (<1 = merges prefer structurally-close pairs)")

# ---- C. coagulation nulls ----
print("\n-- C. coagulation nulls (is the size distribution generic?) --")
sgoed_sizes = []
for seed in SEEDS:
    _, pool, _, n_merges = recursive_with_tree(seed)
    sgoed_sizes.extend(e[1] for e in pool)
sgoed_sizes = np.array(sgoed_sizes)
print(f"  SGOED:     final entities={len(sgoed_sizes)}, "
      f"max size={sgoed_sizes.max()}, mean={sgoed_sizes.mean():.1f}")
for kernel in ("constant", "additive", "multiplicative"):
    # tune c so total final entities ~ SGOED's
    best = None
    for c in np.logspace(-3, 0, 60):
        sizes = np.concatenate([coagulation_null(s, kernel, c) for s in SEEDS])
        score = abs(len(sizes) - len(sgoed_sizes))
        if best is None or score < best[0]:
            best = (score, c, sizes)
    _, c, sizes = best
    print(f"  {kernel:<13}: c={c:.3f}, entities={len(sizes)}, "
          f"max={sizes.max()}, mean={sizes.mean():.1f}")

# ---- D. tree shape ----
print("\n-- D. merge-tree depth (path length leaf->root) --")
depths = []
for seed in SEEDS:
    _, pool, _, _ = recursive_with_tree(seed)
    depths.extend(e[3] for e in pool if e[1] >= 2)
depths = np.array(depths)
print(f"  composites size>=2: n={len(depths)}, mean depth={depths.mean():.2f}, "
      f"max depth={depths.max()}")

print("""
Honest reading:
- Clustering ratio ~1 -> composites are NOT clusters in state space: the
  hierarchy does not organize by structural similarity (no locality-like
  signature). ratio >>1 would be the spacetime-analogy-friendly case.
- merged/random distance ~1 -> merges are not selecting structurally-close
  pairs; the merge test is stability-gated, not similarity-gated.
- If the SGOED size distribution matches a coagulation null, the hierarchy is
  a generic aggregation process, not a geometry-forming one.
- The "order of change" is real (a hierarchy exists) but, per the K-analogue
  meta-observation, it does not amplify into manifold-like structure.
""")
