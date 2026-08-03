import numpy as np

# ==========================================
# Phase 1: comparison of crystallization prescriptions
#
# The global Fermi-Dirac application is dominated by negative-mass states
# (Section 2.3 caveat): ~95-99.7% of the "crystallized" population has I_C <= 0.
# This script compares candidate prescriptions on the SAME state ensemble:
#   1. global FD   : p = sigmoid(I_C / T) for ALL states        (current)
#   2. windowed FD : p = sigmoid(I_C / T) if I_C >= -k*T, else 0
#                    (fluctuation-assisted transition only where it is
#                     physically motivated, |I_C| ~ T; deep-negative states
#                     are definitively dissolved)
#   3. hard thresh : p = 1[I_C > 0]   (deterministic; crystallized == stable
#                    by definition, but loses the probabilistic character)
#
# Metrics (expected values, no RNG noise):
#   yield         = sum p_i            (crystallized fraction)
#   stable share  = sum p_i 1[I_C>0] / sum p_i   (of what forms, how much is
#                                                  stable by the theory's own
#                                                  I_C > 0 criterion)
#   mean margin   = sum p_i I_C_i / sum p_i      (what Phase 2 would receive)
# ==========================================

N = 5000
T_CHAOS = 1.75
KS = [1, 2, 3]


def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


print("=" * 74)
print(" Crystallization prescriptions on the same 5000-state ensemble")
print("=" * 74)

for dim in (3, 4):
    rng = np.random.default_rng(42)
    scales = rng.uniform(0.5, 2.0, (N, 1, 1))
    G = rng.standard_normal((N, dim, dim)) * scales
    i_c = np.array([lambda_min_S(G[k]) for k in range(N)])
    stable = i_c > 0

    print(f"\n--- dim={dim}  (I_C range [{i_c.min():.2f}, {i_c.max():.2f}], "
          f"I_C>0: {stable.sum()} = {stable.mean()*100:.2f}%) ---")
    print(f"  {'prescription':<16} {'yield':>9} {'stable share':>13} "
          f"{'mean margin':>12} {'tail contrib':>13}")

    rows = []

    # 1. global FD
    p = sigmoid(i_c / T_CHAOS)
    y = p.sum()
    ss = (p * stable).sum() / y
    mm = (p * i_c).sum() / y
    tc = (p * (i_c < 0)).sum() / y
    rows.append(("global FD", y, ss, mm, tc))

    # 2. windowed FD, floor at -k*T
    for k in KS:
        floor = -k * T_CHAOS
        pw = np.where(i_c >= floor, sigmoid(i_c / T_CHAOS), 0.0)
        yw = pw.sum()
        ssw = (pw * stable).sum() / yw
        mmw = (pw * i_c).sum() / yw
        tcw = (pw * (i_c < 0)).sum() / yw
        rows.append((f"windowed k={k}", yw, ssw, mmw, tcw))

    # 3. hard threshold
    ph = stable.astype(float)
    yh = ph.sum()
    ssh = 1.0
    mmh = (ph * i_c).sum() / yh
    tch = 0.0
    rows.append(("hard I_C>0", yh, ssh, mmh, tch))

    for name, y_, ss_, mm_, tc_ in rows:
        print(f"  {name:<16} {y_/N*100:8.2f}% {ss_*100:12.1f}% {mm_:+11.3f} {tc_*100:12.1f}%")

print("""
Reading:
- yield        = fraction of chaos that "crystallizes"
- stable share = of that crystallized population, the fraction that is
                 individually stable by the same I_C > 0 criterion
- mean margin  = mean I_C of the crystallized population (what Phase 2
                 would receive as survivors)
- tail contrib = share of the crystallized population with I_C < 0

The self-consistency criterion is stable share ~ 1: the theory's own
definition of stability should agree with what it calls "crystallized".
""")
