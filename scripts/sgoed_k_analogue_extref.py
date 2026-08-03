import numpy as np

# ==========================================
# Directional + parallel selection with an EXTERNAL reference r (STF Source b).
#
# Section 8 of the design doc predicted: ordering that compounds needs a
# direction that is simultaneously directional AND parallel; pure-relative
# chirality is perpendicular and cannot provide it; an external reference
# breaks rotational symmetry and should allow it.
#
# Two candidate designs (both seek compatible partners, non-i.i.d.):
#   rpar : absorb k iff stable AND u_C.r > 0 AND u_k.r > 0
#          (parallel selection + composite-direction gate; direction = r)
#   rchi : absorb k iff stable AND u_C.r > 0 AND sign(chi(C,k)) = +1
#          (chirality supplies the direction, r-gate keeps the composite
#           oriented; prediction: the perpendicular chirality absorption
#           kicks u_C off the r-axis and the r-gate then STALLS growth)
# References: canonical (no orientation), align (u_C.u_k > 0, no external r).
#
# Metrics: internal coherence |sum u_i|/n, alignment with r |u_comp.r|,
# signed net direction u_comp.r, size distribution.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
ROUNDS = 10
SEEK = 5
S_REF = +1
SEEDS = [1, 2, 3]
R = np.array([0.0, 0.0, 1.0])   # external reference direction (arbitrary, fixed)


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


def axis_of(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def chi(uC, uLast, uk):
    return float(np.dot(uC, np.cross(uLast, uk)))


def symmetric_stable(Ga, Gb, noise):
    cross = 0.5 * (noise + noise.T) * ALPHA
    return lambda_min_S(0.5 * (Ga + Gb) + cross) > 0


def build_population(rng):
    margins = rng.uniform(0.1, 1.0, NUM_ELEMENTS)
    return [random_stable_G(GDIM, m, rng) for m in margins]


def accept(mode, uC, uLast, uk):
    if mode == "canonical":
        return True
    if mode == "align":
        return float(np.dot(uC, uk)) > 0.0
    if mode == "rpar":
        return float(np.dot(uC, R)) > 0.0 and float(np.dot(uk, R)) > 0.0
    if mode == "rchi":
        return float(np.dot(uC, R)) > 0.0 and np.sign(chi(uC, uLast, uk)) == S_REF
    raise ValueError(mode)


def run(seed, mode):
    rng = np.random.default_rng(seed)
    Gs = build_population(rng)
    pool = [[G, 1, unit(axis_of(anti_part(G))), None,
             [unit(axis_of(anti_part(G)))]] for G in Gs]
    n_merges = 0
    for _r in range(ROUNDS):
        rng.shuffle(pool)
        new_pool, i = [], 0
        while i + 1 < len(pool):
            a, b = pool[i], pool[i + 1]
            merged = False
            E = rng.standard_normal(a[0].shape)
            if symmetric_stable(a[0], b[0], E):
                if a[1] == 1 and b[1] == 1:
                    merged = True
                else:
                    uC, uLast, uk = (a[2], a[3] if a[3] is not None else a[2], b[2]) \
                        if a[1] >= b[1] else (b[2], b[3] if b[3] is not None else b[2], a[2])
                    merged = accept(mode, uC, uLast, uk)
            if merged:
                new_pool.append(_merge(a, b, rng))
                n_merges += 1
            else:
                if (a[1] > 1 or b[1] > 1) and i + 2 < len(pool):
                    comp, other = (a, b) if a[1] >= b[1] else (b, a)
                    seek_success = False
                    for _s in range(SEEK):
                        j = rng.integers(i + 2, len(pool))
                        cand = pool[j]
                        E = rng.standard_normal(comp[0].shape)
                        if not symmetric_stable(comp[0], cand[0], E):
                            continue
                        uC = comp[2]; uLast = comp[3] if comp[3] is not None else comp[2]
                        if accept(mode, uC, uLast, cand[2]):
                            new_pool.append(_merge(comp, cand, rng))
                            pool.pop(j)
                            n_merges += 1
                            seek_success = True
                            break
                    if seek_success:
                        new_pool.append(other)
                        merged = True
                if not merged:
                    new_pool.append(a); new_pool.append(b)
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
    return pool, n_merges


def _merge(a, b, rng):
    E = rng.standard_normal(a[0].shape)
    cross = 0.5 * (E + E.T) * ALPHA
    Gm = 0.5 * (a[0] + b[0]) + cross
    u_new = unit(axis_of(anti_part(Gm)))
    return [Gm, a[1] + b[1], u_new, b[2], a[4] + b[4]]


print("=" * 74)
print(" Directional + parallel selection with external reference r")
print("=" * 74)
for mode in ("canonical", "align", "rpar", "rchi"):
    rows, sizes_all, total_merges = [], [], 0
    for seed in SEEDS:
        pool, nm = run(seed, mode)
        total_merges += nm
        sizes_all.extend(e[1] for e in pool)
        for e in pool:
            n = e[1]
            if n >= 3:
                cons = np.array(e[4])
                coh = np.linalg.norm(cons.sum(axis=0)) / n
                rows.append((n, coh, abs(float(np.dot(e[2], R))), float(np.dot(e[2], R))))
    rows = np.array(rows)
    sizes_all = np.array(sizes_all)
    print(f"\n  {mode}:  merges/seed={total_merges/len(SEEDS):.0f}, "
          f"size>=3: n={len(rows)}, size max={sizes_all.max()}, mean={sizes_all.mean():.1f}")
    # alignment with r by log-size bins (compounding trajectory)
    bins = [(3, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
    traj = []
    for lo, hi in bins:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum() >= 3:
            traj.append(f"[{lo},{hi}):|u.r|={rows[m,2].mean():.2f}(n={m.sum()})")
    if traj:
        print(f"    alignment trajectory: {'  '.join(traj)}")

print("""
Reading (design SGOED_STF_K_analogue_design.md, Section 9):
- rpar: directional + parallel WITH an external reference produces strong
  sustained alignment with r (0.84-0.88) and a net +r direction -- BUT the
  decomposition shows this is selection + averaging (one-shot two-vector
  average ~0.66-0.69; static LLN at these sizes 0.92-0.98 exceeds the
  recursion, which is diluted by the ungated initial merge). No dynamical
  compounding: "compounds" overstates, "selection-and-averaging" is accurate.
- rchi: adding the chirality sign to the same framework FAILS -- alignment
  decreases with size (0.45 -> 0.34), growth stalls (max 33), composites end
  pointing -r, coherence below random. The perpendicular geometry of chirality
  defeats what parallel selection achieves. Decisive structural conclusion.
""")
