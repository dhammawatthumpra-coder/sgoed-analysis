import numpy as np

# ==========================================
# Non-i.i.d. partner selection: does ordering compound?
#
# Previous metrics measured alignment with an EXTERNAL axis r -- which is
# forced to ~0 by rotational symmetry (no preferred direction exists). The
# correct rotationally-invariant order parameter is the INTERNAL coherence
# of a composite's constituent axes:
#     coh = |sum_i u_i| / n        (u_i = so(3) axes of the n original elements)
# Random constituents: E[coh] ~ 1/sqrt(n). If chirality-biased absorption
# correlates successive constituents, coh ~ c/sqrt(n) with c > 1 -- internal
# compound ordering (no global direction needed).
#
# Modes (all chirality-based, chi(C,k) = u_C . (u_last x u_k)):
#   canonical : no orientation (baseline)
#   iid       : chirality gate, uniform shuffle partners (i.i.d.)
#   seek      : chirality gate + partner RESAMPLING up to `seek` draws
#               (non-i.i.d.: composites seek compatible partners)
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


def run(seed, mode, seek=0):
    rng = np.random.default_rng(seed)
    Gs = build_population(rng)
    # entity: [G, size, u, u_last, constituents(list of unit axes)]
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
            stable = symmetric_stable(a[0], b[0], E)
            if stable:
                gated = mode != "canonical" and not (a[1] == 1 and b[1] == 1)
                if not gated or _accept(a, b, mode):
                    merged = True
            if merged:
                new_pool.append(_merge(a, b, rng))
                n_merges += 1
            else:
                # seek: composite member re-draws partners from the remaining pool
                if mode == "seek" and (a[1] > 1 or b[1] > 1) and i + 2 < len(pool):
                    comp, other = (a, b) if a[1] >= b[1] else (b, a)
                    seek_success = False
                    for _s in range(seek):
                        j = rng.integers(i + 2, len(pool))
                        cand = pool[j]
                        E = rng.standard_normal(comp[0].shape)
                        if not symmetric_stable(comp[0], cand[0], E):
                            continue
                        if _accept(comp, cand, mode):
                            new_pool.append(_merge(comp, cand, rng))
                            pool.pop(j)
                            n_merges += 1
                            seek_success = True
                            break
                    if seek_success:
                        new_pool.append(other)   # failed partner returns to the pool
                        merged = True
                if not merged:
                    new_pool.append(a); new_pool.append(b)
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
    return pool, n_merges


def _accept(a, b, mode):
    """Orientation acceptance for a gated merge of entities a (>= b in size)."""
    uC, uLast, uk = (a[2], a[3] if a[3] is not None else a[2], b[2]) if a[1] >= b[1] \
        else (b[2], b[3] if b[3] is not None else b[2], a[2])
    if mode == "iid":
        return np.sign(chi(uC, uLast, uk)) == S_REF
    if mode == "seek":
        return np.sign(chi(uC, uLast, uk)) == S_REF
    if mode == "align":
        return float(np.dot(uC, uk)) > 0.0    # parallel selection (contrast)
    raise ValueError(mode)


def _merge(a, b, rng):
    E = rng.standard_normal(a[0].shape)
    cross = 0.5 * (E + E.T) * ALPHA
    Gm = 0.5 * (a[0] + b[0]) + cross
    u_new = unit(axis_of(anti_part(Gm)))
    # constituents: a's + b's (b is the absorbed partner -> u_last = b's axis)
    cons = a[4] + b[4]
    return [Gm, a[1] + b[1], u_new, b[2], cons]


print("=" * 74)
print(" Non-i.i.d. partner selection: constituent coherence vs size")
print("=" * 74)

for mode, seek, label in (("canonical", 0, "canonical"),
                          ("iid", 0, "iid (uniform partners)"),
                          ("seek", SEEK, f"seek (resample x{SEEK})"),
                          ("align", 0, "align (u_C.u_k > 0, contrast)")):
    # collect (size, coherence) over seeds
    rows = []
    sizes_all = []
    total_merges = 0
    for seed in SEEDS:
        pool, nm = run(seed, mode, seek)
        total_merges += nm
        sizes_all.extend(e[1] for e in pool)
        for e in pool:
            n = e[1]
            if n >= 3:
                cons = np.array(e[4])
                coh = np.linalg.norm(cons.sum(axis=0)) / n
                rows.append((n, coh))
    rows = np.array(rows)
    print(f"\n  {label}:")
    print(f"    merges/seed avg = {total_merges/len(SEEDS):.0f},  final entities "
          f"size>=3: n={len(rows)}")
    for lo in (3, 5, 8):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < 16)
        if m.sum() >= 5:
            n_avg = rows[m, 0].mean()
            coh = rows[m, 1].mean()
            baseline = 1.0 / np.sqrt(n_avg)
            print(f"    size ~ {n_avg:5.1f}: coh = {coh:.4f}   "
                  f"random 1/sqrt(n) = {baseline:.4f}   ratio = {coh/baseline:.2f}")
    sizes_all = np.array(sizes_all)
    print(f"    size distribution: max={sizes_all.max()}, mean={sizes_all.mean():.1f}, "
          f"frac>=5 = {(sizes_all>=5).mean()*100:.0f}%")

print("""
Reading (design SGOED_STF_K_analogue_design.md, Section 8):
- canonical: coh ~ 1/sqrt(n) (random baseline, ratio ~1).
- iid (chirality, uniform partners): ratio ~1 -- no compounding.
- seek (chirality + resampling, non-i.i.d.): ratio ~1.33 -- weak compounding.
- align (parallel selection, contrast): ratio 1.48 -> 1.64 growing -- strong.
Structural reason: chirality selects partners in the plane PERPENDICULAR to the
composite axis (u_C x u_last is perpendicular), rotating it in random azimuths
without building alignment; only parallel selection compounds, and that is
alignment, not ordering. A directional ordering that compounds needs an
external reference (STF Source b); pure-relative chirality cannot provide it.
""")
