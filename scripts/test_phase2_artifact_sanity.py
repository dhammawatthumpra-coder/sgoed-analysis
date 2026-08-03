import os
import sys
import numpy as np

# ==========================================
# Invariant sanity checks for the Phase 2 artifact-test suite.
# Each check asserts a mathematical invariant of the merge test or the
# experiment design. A failure here means the experiment code regressed
# (e.g. a display bug or a broken control), not that the theory changed.
# Run: python scripts/test_phase2_artifact_sanity.py
# ==========================================

HERE = os.path.dirname(os.path.abspath(__file__))


def load_defs(fname):
    """Load only the function/constant definitions of an experiment script
    (everything before its main print block)."""
    src = open(os.path.join(HERE, fname), encoding="utf-8").read()
    cut = src.index('print("=" * 72)')
    ns = {}
    exec(src[:cut], ns)
    return ns


ns5 = load_defs("sgoed_phase2_exp5_operator_ablation.py")
ns3b = load_defs("sgoed_phase2_exp3b_blend_margin_mediation.py")

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def pop(rng, n=300):
    Gs = [ns5["random_stable_G"](3, m, rng)
          for m in rng.uniform(0.1, 1.0, n)]
    return Gs


@check("fixed-K=1: merged fraction == per-attempt success")
def check_fixed_k1():
    rng = np.random.default_rng(1)
    Gs = pop(rng)
    op = ns5["operator_margin"]
    n_pairs = 800
    merged, attempts = 0, []
    for _ in range(n_pairs):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        ok = False
        for _k in range(1):
            s = 1 if op("gaussian_iid", Gs[i], Gs[j], 1.4, rng) > 0 else 0
            attempts.append(s)
            if s:
                ok = True
                break
        merged += 1 if ok else 0
    merged_frac = merged / n_pairs
    per_attempt = float(np.mean(attempts))
    return (abs(merged_frac - per_attempt) < 1e-12 and 0.5 < merged_frac < 0.8,
            f"merged={merged_frac*100:.1f}% per_attempt={per_attempt*100:.1f}%")


@check("antisymmetric noise: success == 100% (control)")
def check_anti_noise():
    rng = np.random.default_rng(2)
    Gs = pop(rng)
    op = ns5["operator_margin"]
    n, failures = 2000, 0
    for _ in range(n):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        if not op("anti_noise", Gs[i], Gs[j], 1.4, rng) > 0:
            failures += 1
    return (failures == 0, f"failures={failures}/{n}")


@check("convex blend: success == 100% (trivial baseline)")
def check_blend():
    rng = np.random.default_rng(3)
    Gs = pop(rng)
    op = ns5["operator_margin"]
    n, failures = 2000, 0
    for _ in range(n):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        if not op("blend", Gs[i], Gs[j], 1.4, rng) > 0:
            failures += 1
    return (failures == 0, f"failures={failures}/{n}")


@check("symmetric noise genuinely perturbs stability (not vacuous)")
def check_sym_noise_effective():
    rng = np.random.default_rng(4)
    Gs = pop(rng)
    op, lm = ns5["operator_margin"], ns5["lambda_min_S"]
    n, changed, successes = 2000, 0, 0
    for _ in range(n):
        i, j = rng.choice(len(Gs), size=2, replace=False)
        Gi, Gj = Gs[i], Gs[j]
        m0 = lm(0.5 * (Gi + Gj))
        m1 = op("sym_noise", Gi, Gj, 1.4, rng)
        if abs(m1 - m0) > 1e-9:
            changed += 1
        if m1 > 0:
            successes += 1
    frac_changed = changed / n
    frac_success = successes / n
    return (frac_changed > 0.99 and 0.2 < frac_success < 0.99,
            f"margins changed {frac_changed*100:.1f}%, success {frac_success*100:.1f}%")


@check("element-wise split: no element shared between train and test")
def check_element_wise_disjoint():
    md = ns3b["make_dataset"]
    _, _, idx_tr = md(num_attempts=300, idx_range=(0, 210), return_idx=True)
    _, _, idx_te = md(num_attempts=300, idx_range=(210, 300), return_idx=True)
    tr = [i for pair in idx_tr for i in pair]
    te = [i for pair in idx_te for i in pair]
    ok = (all(i < 210 for i in tr) and all(i >= 210 for i in te)
          and not set(tr).intersection(te))
    return (ok, f"train range [{min(tr)},{max(tr)}], test range [{min(te)},{max(te)}]")


def main():
    n_pass = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - report any failure clearly
            ok, detail = False, f"raised {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}  ({detail})")
        n_pass += 1 if ok else 0
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    sys.exit(0 if n_pass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
