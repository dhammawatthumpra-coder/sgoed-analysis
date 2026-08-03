import numpy as np

np.random.seed(42)

# ==========================================
# Kinetic collision model:
# - two elements undergo a random walk in separation d(t), driven by ambient chaos noise
# - no attraction potential anywhere
# - upon collision (d < d_contact), attempt a genuine merge: form a NEW composite matrix
#   G_merged (same dimension as the originals, not a block concatenation), representing
#   structural reorganization on contact, then re-run the SAME Bendixson stability test
#   used in Phase 1. If it passes, the pair is permanently bound (irreversible). If not,
#   they bounce apart and the random walk continues.
# ==========================================

GDIM = 3
NUM_ELEMENTS = 300
MAX_STEPS = 400
D_CONTACT = 0.4        # collision threshold
D_MAX = 8.0             # if they drift this far apart, count as "never met"
NOISE_STD = 0.35        # step size of the chaos-driven random walk
CROSS_TERM_STRENGTH = 1.4  # substantially larger than typical margins (0.1-1.0) to test real selectivity

def random_stable_G(n, target_margin):
    A_rand = np.random.randn(n, n)
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = np.random.randn(n, n)
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti

def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()

def attempt_merge(Gi, Gj):
    """Genuine reorganization on contact. A convex blend of two PD matrices is ALWAYS PD
    (mathematical guarantee) which made the earlier version trivially non-selective.
    Instead, overlay the two structures with genuine off-diagonal cross-coupling terms
    representing the actual interference between the two elements' internal gradients at
    the moment of contact -- this can genuinely create or destroy compatibility, not just
    average it."""
    E = np.random.randn(*Gi.shape)
    cross = 0.5 * (E + E.T) * CROSS_TERM_STRENGTH  # canonical: symmetric-only interference
    G_merged = 0.5 * (Gi + Gj) + cross  # cross term is NOT guaranteed sign-definite
    return G_merged, lambda_min_S(G_merged)

# population of individually stable "survivor" elements (Phase 1 output)
element_margins = np.random.uniform(0.1, 1.0, NUM_ELEMENTS)
element_G = [random_stable_G(GDIM, m) for m in element_margins]

results = {"merged": 0, "drifted_apart": 0, "failed_merge_attempts_total": 0, "timed_out": 0}
merge_step_record = []

PAIR_TRIALS = 800
for _ in range(PAIR_TRIALS):
    i, j = np.random.choice(NUM_ELEMENTS, size=2, replace=False)
    Gi, Gj = element_G[i], element_G[j]
    d = np.random.uniform(2.0, 5.0)  # random initial separation
    outcome = None
    for step in range(MAX_STEPS):
        d += np.random.randn() * NOISE_STD  # chaos-driven random walk, no attraction bias
        d = max(d, 0.05)
        if d > D_MAX:
            outcome = "drifted_apart"
            break
        if d < D_CONTACT:
            G_merged, margin = attempt_merge(Gi, Gj)
            if margin > 0:
                outcome = "merged"
                merge_step_record.append(step)
                break
            else:
                results["failed_merge_attempts_total"] += 1
                d += abs(np.random.randn()) * 0.5  # bounce apart after failed merge attempt
    if outcome is None:
        outcome = "timed_out"
    results[outcome] += 1

print("=" * 55)
print(" SGOED Phase 2 v3 -- Kinetic collision / recrystallization model")
print("=" * 55)
print(f"Total pair trials: {PAIR_TRIALS}")
print(f"Merged (bound permanently):   {results['merged']} ({results['merged']/PAIR_TRIALS*100:.2f}%)")
print(f"Drifted apart (never merged): {results['drifted_apart']} ({results['drifted_apart']/PAIR_TRIALS*100:.2f}%)")
print(f"Timed out (still wandering):  {results['timed_out']} ({results['timed_out']/PAIR_TRIALS*100:.2f}%)")
print(f"Total failed merge attempts (bounced, tried again): {results['failed_merge_attempts_total']}")
if merge_step_record:
    print(f"Median step-count to successful merge: {np.median(merge_step_record):.1f}")

print("\n--- Checking how selective the merge test actually is ---")
total_merge_attempts = results["merged"] + results["failed_merge_attempts_total"]
success_rate_per_attempt = results["merged"] / total_merge_attempts if total_merge_attempts else 0
print(f"Total merge attempts (successful + failed): {total_merge_attempts}")
print(f"Per-attempt success rate: {success_rate_per_attempt*100:.1f}%")
print("(Convex blend of two individually-stable matrices is mathematically guaranteed")
print(" to remain stable BEFORE noise is added -- so this test is only as selective")
print(" as the noise term MERGE_BLEND_NOISE makes it, not a strong independent filter.)")
