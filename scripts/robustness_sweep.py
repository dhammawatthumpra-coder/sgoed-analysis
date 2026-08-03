import numpy as np

GDIM = 3

def random_stable_G(n, target_margin, rng):
    A_rand = rng.standard_normal((n, n))
    S = A_rand @ A_rand.T + target_margin * np.eye(n)
    Anti = rng.standard_normal((n, n))
    Anti = 0.5 * (Anti - Anti.T)
    return S + Anti

def lambda_min_S(M):
    S = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(S).min()

def attempt_merge(Gi, Gj, cross_strength, rng):
    E = rng.standard_normal(Gi.shape)
    cross = 0.5 * (E + E.T) * cross_strength  # canonical: symmetric-only interference
    G_merged = 0.5 * (Gi + Gj) + cross
    return lambda_min_S(G_merged)

def run_trial(seed, noise_std, d_contact, cross_strength, num_elements, pair_trials,
              margin_low, margin_high, max_steps=400, d_max=8.0):
    rng = np.random.default_rng(seed)
    element_margins = rng.uniform(margin_low, margin_high, num_elements)
    element_G = [random_stable_G(GDIM, m, rng) for m in element_margins]

    counts = {"merged": 0, "drifted_apart": 0, "timed_out": 0}
    merge_attempts = 0
    merge_successes = 0

    for _ in range(pair_trials):
        i, j = rng.choice(num_elements, size=2, replace=False)
        Gi, Gj = element_G[i], element_G[j]
        d = rng.uniform(2.0, 5.0)
        outcome = None
        for step in range(max_steps):
            d += rng.standard_normal() * noise_std
            d = max(d, 0.05)
            if d > d_max:
                outcome = "drifted_apart"
                break
            if d < d_contact:
                merge_attempts += 1
                margin = attempt_merge(Gi, Gj, cross_strength, rng)
                if margin > 0:
                    merge_successes += 1
                    outcome = "merged"
                    break
                else:
                    d += abs(rng.standard_normal()) * 0.5
        if outcome is None:
            outcome = "timed_out"
        counts[outcome] += 1

    per_attempt_rate = merge_successes / merge_attempts if merge_attempts else float("nan")
    return counts, per_attempt_rate, merge_attempts

print("=" * 70)
print(" Robustness sweep: kinetic collision/recrystallization model")
print("=" * 70)

baseline = dict(noise_std=0.35, d_contact=0.4, cross_strength=1.4,
                num_elements=300, pair_trials=500, margin_low=0.1, margin_high=1.0)

print("\n-- 1. Seed variation (all other params fixed at baseline) --")
for seed in [1, 2, 3, 4, 5]:
    counts, rate, attempts = run_trial(seed=seed, **baseline)
    total = sum(counts.values())
    print(f"seed={seed}: merged={counts['merged']/total*100:5.1f}%  "
          f"drifted={counts['drifted_apart']/total*100:5.1f}%  "
          f"timeout={counts['timed_out']/total*100:5.1f}%  "
          f"per-attempt success={rate*100:5.1f}%  (n_attempts={attempts})")

print("\n-- 2. Noise (random walk step size) variation --")
for noise in [0.15, 0.25, 0.35, 0.5, 0.8]:
    params = dict(baseline); params["noise_std"] = noise
    counts, rate, attempts = run_trial(seed=1, **params)
    total = sum(counts.values())
    print(f"noise_std={noise}: merged={counts['merged']/total*100:5.1f}%  "
          f"drifted={counts['drifted_apart']/total*100:5.1f}%  "
          f"per-attempt success={rate*100:5.1f}%  (n_attempts={attempts})")

print("\n-- 3. Contact threshold variation --")
for dc in [0.2, 0.4, 0.6, 1.0]:
    params = dict(baseline); params["d_contact"] = dc
    counts, rate, attempts = run_trial(seed=1, **params)
    total = sum(counts.values())
    print(f"d_contact={dc}: merged={counts['merged']/total*100:5.1f}%  "
          f"drifted={counts['drifted_apart']/total*100:5.1f}%  "
          f"per-attempt success={rate*100:5.1f}%  (n_attempts={attempts})")

print("\n-- 4. Cross-term strength variation (the calibrated parameter) --")
for cs in [0.5, 1.0, 1.4, 2.0, 3.0]:
    params = dict(baseline); params["cross_strength"] = cs
    counts, rate, attempts = run_trial(seed=1, **params)
    total = sum(counts.values())
    print(f"cross_strength={cs}: merged={counts['merged']/total*100:5.1f}%  "
          f"per-attempt success={rate*100:5.1f}%  (n_attempts={attempts})")

print("\n-- 5. Individual element margin range (weaker vs stronger Phase-1 survivors) --")
for lo, hi in [(0.05, 0.3), (0.1, 1.0), (0.5, 1.5), (1.0, 2.0)]:
    params = dict(baseline); params["margin_low"] = lo; params["margin_high"] = hi
    counts, rate, attempts = run_trial(seed=1, **params)
    total = sum(counts.values())
    print(f"margin_range=({lo},{hi}): merged={counts['merged']/total*100:5.1f}%  "
          f"per-attempt success={rate*100:5.1f}%  (n_attempts={attempts})")
