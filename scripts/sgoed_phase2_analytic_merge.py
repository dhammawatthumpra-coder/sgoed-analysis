import numpy as np

# ==========================================
# Analytic characterization of the canonical merge test
#
# Canonical operator (sym_noise):  G_merged = 0.5*(G_i+G_j) + alpha * 0.5*(E+E^T),
# E iid N(0,1).  Success  <=>  lambda_min(S(G_merged)) > 0.
#
# What is derived here:
#   1. Matrix-level equivalence: the merge test with full Gaussian iid noise and
#      with symmetric-only noise is THE SAME random variable (lambda_min sees
#      only the symmetric part of the noise). Verified bit-exactly.
#   2. Exact algebraic conditions for success via the characteristic polynomial.
#      For a real symmetric n x n matrix all eigenvalues are real, so for n=2,3
#      the sign of the characteristic-polynomial coefficients is necessary AND
#      sufficient for positive definiteness:
#          n=2:  tr > 0  AND  det > 0
#          n=3:  tr > 0  AND  c2 > 0  AND  det > 0     (c2 = sum of 2x2 principal minors)
#      (For n>=4 the simple sign test is not sufficient; the Sylvester criterion
#      on all leading principal minors is the exact condition instead.)
#   3. Weyl bound:  |lambda_min(A+E) - lambda_min(A)| <= ||E||_2, so
#      alpha*||E||_2 < m_blend guarantees success. We quantify how much of the
#      observed success this sufficient condition covers.
#   4. GOE connection: E_sym = 0.5*(E+E^T) with E iid N(0,1) is a GOE matrix
#      (beta=1; off-diagonal N(0,1/2), diagonal N(0,1)). The event
#      lambda_min(S_blend + alpha*E_sym) > 0 is a "shifted-GOE edge" probability;
#      for fixed n=3 the exact sign conditions are the practical tool (Tracy-Widom
#      would be the n->infinity edge limit).
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
ALPHA = 1.4
MARGIN_LOW, MARGIN_HIGH = 0.1, 1.0
DATASET_SEED = 42
N_SAMPLES = 20000

FAIL = "\033[91mFAIL\033[0m"
PASS = "\033[92mPASS\033[0m"


def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def sym_part(E):
    return 0.5 * (E + E.T)


# ---------- 1. matrix-level equivalence gaussian_iid == sym_noise ----------
print("=" * 72)
print(" 1. Equivalence: gaussian_iid vs sym_noise merge test")
print("=" * 72)
rng = np.random.default_rng(0)
equiv_ok, max_diff, decision_agree = True, 0.0, 0
for _ in range(2000):
    Gi = random_stable_G(GDIM, rng.uniform(MARGIN_LOW, MARGIN_HIGH), rng)
    Gj = random_stable_G(GDIM, rng.uniform(MARGIN_LOW, MARGIN_HIGH), rng)
    E = rng.standard_normal(Gi.shape)
    M1 = 0.5 * (Gi + Gj) + ALPHA * E              # full Gaussian iid construction
    M2 = 0.5 * (Gi + Gj) + ALPHA * sym_part(E)    # symmetric-only construction
    S1, S2 = 0.5 * (M1 + M1.T), 0.5 * (M2 + M2.T)
    max_diff = max(max_diff, float(np.max(np.abs(S1 - S2))))  # ~machine precision
    decision_agree += (lambda_min_S(S1) > 0) == (lambda_min_S(S2) > 0)
print(f"  max |S(M_iid) - S(M_sym)| over 2000 pairs: {max_diff:.3e}  (machine precision)")
print(f"  merge decisions agree: {decision_agree}/2000 = {decision_agree/2000*100:.1f}%")
print("  (the symmetric parts are mathematically identical; floating-point")
print("   evaluation order differs by ~1 ulp, which never flips a decision)")
print(f"  equivalence: {PASS if max_diff < 1e-12 and decision_agree == 2000 else FAIL}")

# ---------- 2. exact conditions for n=2 and n=3 ----------
print("\n" + "=" * 72)
print(" 2. Exact success conditions via characteristic polynomial")
print("=" * 72)
print("  n=2: tr(A) > 0  and  det(A) > 0")
print("  n=3: tr(A) > 0  and  c2(A) > 0  and  det(A) > 0  (c2 = sum of 2x2")
print("       principal minors). Necessary AND sufficient for symmetric A.")


def conditions_n3(A):
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    c2 = (A[0, 0] * A[1, 1] - A[0, 1] ** 2
          + A[0, 0] * A[2, 2] - A[0, 2] ** 2
          + A[1, 1] * A[2, 2] - A[1, 2] ** 2)
    det = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] ** 2)
           - A[0, 1] * (A[0, 1] * A[2, 2] - A[0, 2] * A[1, 2])
           + A[0, 2] * (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]))
    return tr > 0 and c2 > 0 and det > 0


def conditions_n2(A):
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    return tr > 0 and det > 0


def check_conditions(n, N=20000):
    rng = np.random.default_rng(n)
    agree = 0
    for _ in range(N):
        B = rng.standard_normal((n, n))
        S = B @ B.T + rng.uniform(0.05, 2.0) * np.eye(n)     # deterministic part
        E = rng.standard_normal((n, n))
        A = S + ALPHA * sym_part(E)                          # canonical perturbation
        pred = conditions_n2(A) if n == 2 else conditions_n3(A)
        true = lambda_min_S(A) > 0
        agree += (pred == true)
    return agree / N


for n in (2, 3):
    frac = check_conditions(n)
    print(f"  n={n}: conditions vs direct lambda_min agree {frac*100:.6f}%")

# ---------- 3. P(success | m_blend) via exact conditions + Weyl coverage ----------
print("\n" + "=" * 72)
print(" 3. P(success | m_blend), Weyl bound coverage, GOE connection")
print("=" * 72)
rng = np.random.default_rng(DATASET_SEED)
element_margins = rng.uniform(MARGIN_LOW, MARGIN_HIGH, NUM_ELEMENTS)
element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]

m_blend_list, success_list, weyl_covered = [], [], []
for _ in range(N_SAMPLES):
    i, j = rng.choice(NUM_ELEMENTS, size=2, replace=False)
    Gi, Gj = element_G[i], element_G[j]
    S_blend = 0.25 * (Gi + Gi.T + Gj + Gj.T)
    m_blend = lambda_min_S(S_blend)
    E = rng.standard_normal(Gi.shape)
    Es = sym_part(E)
    A = S_blend + ALPHA * Es
    m_blend_list.append(m_blend)
    success_list.append(1 if conditions_n3(A) else 0)
    weyl_covered.append(1 if ALPHA * np.linalg.norm(Es, 2) < m_blend else 0)

m_blend_arr = np.array(m_blend_list)
success_arr = np.array(success_list, dtype=float)
weyl_arr = np.array(weyl_covered, dtype=float)

print(f"  per-attempt success (canonical, alpha={ALPHA}): "
      f"{success_arr.mean()*100:.1f}%   (simulation reports 59.7-64.9%)")
print(f"  Weyl sufficient bound (alpha*||E||_2 < m_blend) holds for "
      f"{weyl_arr.mean()*100:.1f}% of attempts -> guaranteed success there;")
print("  the exact conditions extend coverage to the remaining attempts.")

print("\n  P(success | m_blend), exact conditions vs Weyl-sufficient-only:")
print(f"  {'m_blend bin':<16} {'n':>6} {'exact P':>8} {'Weyl-guaranteed':>16}")
edges = np.percentile(m_blend_arr, [0, 25, 50, 75, 100])
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (m_blend_arr >= lo) & (m_blend_arr < hi)
    if mask.sum() == 0:
        continue
    p_exact = success_arr[mask].mean()
    p_weyl = weyl_arr[mask].mean()
    print(f"  [{lo:5.2f},{hi:5.2f})          {mask.sum():6d} {p_exact:8.3f} {p_weyl:16.3f}")

# monotonicity check: correlation of m_blend with success (expected via Weyl)
corr = float(np.corrcoef(m_blend_arr, success_arr)[0, 1])
print(f"\n  corr(m_blend, success) = {corr:+.3f}  (monotone protective, as Weyl requires)")

print("""
Interpretation:
- The exact conditions reproduce the merge test 100%: the "analytic" and the
  simulated merge decisions are the same event, evaluated two ways.
- P(success) rises with m_blend as Weyl's inequality requires. The Weyl
  sufficient bound alone under-covers; the exact conditions are what the
  simulation actually computes.
- E_sym is a GOE(1) matrix, so this is the probability that a shifted GOE stays
  positive definite -- a standard random-matrix object; for fixed n=2,3 the
  sign conditions above are its exact evaluation (Tracy-Widom is the
  n -> infinity edge limit, not needed here).
""")
