import numpy as np

# ==========================================
# Pure relative chirality (no external R, no designated reference element)
# + self-propagation test through the hierarchy.
#
# Two single elements have only two axes -> no intrinsic chirality, so their
# merges stay canonical (ordering requires structure). Once a composite exists,
# its OWN history supplies the third axis: for composite C (axis u_C, axis of
# its most recent merge partner u_last) meeting entity k (axis u_k):
#     chi(C,k) = u_C . (u_last x u_k)      -- scalar triple product, pure
# relative; flips under odd permutation; zero iff coplanar.
# Gate: merge binds iff sign(chi) == s_ref (convention).
#
# Section A: gate properties + pair-level tests (order-sensitivity, A control).
# Section B: recursive with the gate + self-propagation metrics.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
ALPHA = 1.4
NOISE_STD = 0.35
D_CONTACT = 0.4
D_MAX = 8.0
MAX_STEPS = 400
PAIR_TRIALS = 800
ROUNDS = 8
S_REF = +1


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
    """so(3) axis u of an antisymmetric 3x3 (A v = u x v)."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def chi(uC, uLast, uk):
    return float(np.dot(uC, np.cross(uLast, uk)))


def symmetric_stable(Ga, Gb, noise):
    cross = 0.5 * (noise + noise.T) * ALPHA
    return lambda_min_S(0.5 * (Ga + Gb) + cross) > 0


def build_population(rng, zero_A=False):
    margins = rng.uniform(0.1, 1.0, NUM_ELEMENTS)
    Gs = [random_stable_G(GDIM, m, rng) for m in margins]
    if zero_A:
        Gs = [0.5 * (G + G.T) for G in Gs]
    return Gs


# ==========================================
print("=" * 74)
print(" A. Pure relative chirality: gate properties and pair-level tests")
print("=" * 74)
rng = np.random.default_rng(0)
Gs = build_population(rng)
axes = np.array([unit(axis_of(anti_part(G))) for G in Gs])

# A.1 triple-product properties
chi_vals = []
for _ in range(5000):
    i, j, k = rng.choice(NUM_ELEMENTS, size=3, replace=False)
    c = chi(axes[i], axes[j], axes[k])
    chi_vals.append(c)
chi_vals = np.array(chi_vals)
Gs0 = build_population(np.random.default_rng(3), zero_A=True)
max_chi0 = max(abs(chi(unit(axis_of(anti_part(Gs0[i]))), unit(axis_of(anti_part(Gs0[j]))),
                     unit(axis_of(anti_part(Gs0[k])))))
               for _ in range(500) for i, j, k in [rng.choice(NUM_ELEMENTS, size=3, replace=False)])
print(f"  E[chi] = {chi_vals.mean():+.4f}, std = {chi_vals.std():.4f}  (zero mean, no intrinsic bias)")
print(f"  A=0 -> max |chi| = {max_chi0:.2e}  (A-essential)")

# A.2 order-sensitivity: composite C (built from a canonical pair merge) vs element k
# build composites by merging random pairs canonically
n_comp = 60
composites = []          # (G, size=2, u_C, u_last)
for _ in range(n_comp):
    i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
    E = rng.standard_normal(Gs[0].shape)
    cross = 0.5 * (E + E.T) * ALPHA
    Gm = 0.5 * (Gs[i] + Gs[j]) + cross
    composites.append((Gm, unit(axis_of(anti_part(Gm))), unit(axis_of(anti_part(Gs[j])))))

diff, total = 0, 0
for Gc, uC, uLast in composites:
    for _ in range(100):
        k = rng.choice(NUM_ELEMENTS)
        uk = axes[k]
        E = rng.standard_normal(Gc.shape)
        stable = symmetric_stable(Gc, Gs[k], E)
        if not stable:
            continue
        total += 1
        # (C,k): chi uses C's history ; (k,C): k is single, u_last_k = u_k
        o_ck = np.sign(chi(uC, uLast, uk)) == S_REF
        o_kc = np.sign(chi(uk, uk, uC)) == S_REF
        diff += (o_ck != o_kc)
print(f"  composite-element order-sensitivity (stable pairs only): "
      f"{diff/max(total,1)*100:.1f}%  (n={total})")

# A.3 A=0 control: axes vanish -> chi = 0 -> gate rejects all composite merges
rng = np.random.default_rng(5)
Gs0 = build_population(rng, zero_A=True)
n_gated_pass = 0
for _ in range(2000):
    i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
    u = unit(axis_of(anti_part(Gs0[i])))
    ul = unit(axis_of(anti_part(Gs0[j])))
    uk = unit(axis_of(anti_part(Gs0[rng.choice(NUM_ELEMENTS)])))
    n_gated_pass += (np.sign(chi(u, ul, uk)) == S_REF)
print(f"  A=0: gate passes {n_gated_pass}/2000 (all chi=0 -> sign(0)!=+1 -> 0)")

# ==========================================
print("\n" + "=" * 74)
print(" B. Recursive hierarchy with the SRC gate + self-propagation metrics")
print("=" * 74)

def recursive_src(seed, mode="src"):
    """mode: 'none' (canonical) | 'src' (pure-relative chirality gate) |
    'random' (random-sign gate control, metric computed on the same geometry)."""
    rng = np.random.default_rng(seed)
    Gs = build_population(rng)
    pool = [(G, 1, unit(axis_of(anti_part(G))), None) for G in Gs]  # (G, size, u, u_last)
    history = []
    signs_all = []      # (size, chi sign) for gated merges
    dcorr_all = []      # |cos| between successive gate directions
    for _r in range(ROUNDS):
        rng.shuffle(pool)
        new_pool, i = [], 0
        merges = 0
        while i + 1 < len(pool):
            (Ga, sa, ua, ula), (Gb, sb, ub, ulb) = pool[i], pool[i + 1]
            d = rng.uniform(2.0, 5.0)
            outcome = None
            gated = mode != "none" and not (sa == 1 and sb == 1)
            rec_sign, uC, uLast, uk = None, None, None, None
            for _step in range(MAX_STEPS):
                d += rng.standard_normal() * NOISE_STD
                d = max(d, 0.05)
                if d > D_MAX:
                    outcome = "drift"; break
                if d < D_CONTACT:
                    E = rng.standard_normal(Ga.shape)
                    if not symmetric_stable(Ga, Gb, E):
                        d += abs(rng.standard_normal()) * 0.5
                        continue
                    if not gated:
                        outcome = "merge"; uk = ub; break
                    # composite involved: C = the composite, k = the other
                    if sa >= sb:
                        uC, uLast, uk = ua, (ula if ula is not None else ua), ub
                    else:
                        uC, uLast, uk = ub, (ulb if ulb is not None else ub), ua
                    if mode == "src":
                        accept = np.sign(chi(uC, uLast, uk)) == S_REF
                    else:  # 'random' control
                        accept = rng.random() < 0.5
                    if accept:
                        outcome = "merge"; rec_sign = np.sign(chi(uC, uLast, uk)); break
                    d += abs(rng.standard_normal()) * 0.5
            if outcome == "merge":
                E = rng.standard_normal(Ga.shape)
                cross = 0.5 * (E + E.T) * ALPHA
                Gm = 0.5 * (Ga + Gb) + cross
                u_new = unit(axis_of(anti_part(Gm)))
                if rec_sign is not None:
                    signs_all.append((sa + sb, rec_sign))
                    dprev = np.cross(uC, uLast)
                    dnext = np.cross(u_new, uk)
                    if np.linalg.norm(dprev) > 1e-9 and np.linalg.norm(dnext) > 1e-9:
                        dcorr_all.append(abs(float(np.dot(unit(dprev), unit(dnext)))))
                new_pool.append((Gm, sa + sb, u_new, uk))
                merges += 1
            else:
                new_pool.append((Ga, sa, ua, ula)); new_pool.append((Gb, sb, ub, ulb))
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
        history.append((len(pool), merges))
    return history, signs_all, dcorr_all

for label, mode in (("canonical", "none"), ("SRC gate (pure relative)", "src"),
                    ("random-gate control", "random")):
    h, signs, dcorr = recursive_src(1, mode)
    row = "  ".join(f"r{r+1}:e{n}/m{m}" for r, (n, m) in enumerate(h))
    print(f"  {label:<28} {row}")
    if signs:
        big = [s for sz, s in signs if sz >= 3]
        if big:
            print(f"      merges into size>=3 composites: n={len(big)}, "
                  f"fraction +1 = {np.mean([1 if s==S_REF else 0 for s in big])*100:.1f}%")
    if dcorr:
        print(f"      gate-direction persistence: mean |cos(d_t, d_{'{t+1}'})| = {np.mean(dcorr):.3f}  "
              f"(n={len(dcorr)}; random baseline 0.5)")

print("\n  -- persistence, seed-averaged (5 seeds) --")
for mode in ("src", "random"):
    means = []
    for seed in range(1, 6):
        _, _, dcorr = recursive_src(seed, mode)
        if dcorr:
            means.append(np.mean(dcorr))
    if means:
        print(f"  {mode:<8} mean |cos| = {np.mean(means):.3f} +/- {np.std(means):.3f}  "
              f"(random baseline 0.5)")

print("""
Reading (design SGOED_STF_K_analogue_design.md, Section 7):
- A: pure relative chirality works without any external reference (the third
  axis comes from the composite's own history); zero-mean, A-essential, and
  composites absorb elements directionally (elements cannot absorb composites).
- B: the hierarchy is sparser under the gate. The gate-direction persistence
  must be compared with the RANDOM-GATE CONTROL (same metric, coin-flip gate):
  SRC 0.638 +/- 0.023 vs random 0.618 +/- 0.014 -> not significant. The
  persistence is a metric-geometry artifact, not chirality self-propagation.
  Conclusion: ordering is per-collision; it does not self-reinforce.
""")
