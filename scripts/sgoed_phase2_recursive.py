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

def attempt_merge(Ga, Gb, cross_strength, rng):
    E = rng.standard_normal(Ga.shape)
    cross = 0.5 * (E + E.T) * cross_strength  # canonical: symmetric-only interference
    G_merged = 0.5 * (Ga + Gb) + cross
    return G_merged, lambda_min_S(G_merged)

def kinetic_collision(Ga, Gb, cross_strength, noise_std, d_contact, d_max, max_steps, rng):
    """One collision attempt between two entities (single elements or existing composites)."""
    d = rng.uniform(2.0, 5.0)
    for step in range(max_steps):
        d += rng.standard_normal() * noise_std
        d = max(d, 0.05)
        if d > d_max:
            return None  # drifted apart
        if d < d_contact:
            G_merged, margin = attempt_merge(Ga, Gb, cross_strength, rng)
            if margin > 0:
                return G_merged
            else:
                d += abs(rng.standard_normal()) * 0.5
    return None  # timed out


def run_recursive_aggregation(seed, num_elements, rounds, cross_strength=1.4,
                                noise_std=0.35, d_contact=0.4, d_max=8.0, max_steps=400,
                                margin_low=0.1, margin_high=1.0):
    rng = np.random.default_rng(seed)
    # each "entity" = (G matrix, size = number of original elements it represents)
    pool = []
    for _ in range(num_elements):
        m = rng.uniform(margin_low, margin_high)
        pool.append((random_stable_G(GDIM, m, rng), 1))

    history = []
    for r in range(rounds):
        rng.shuffle(pool)
        new_pool = []
        i = 0
        merges_this_round = 0
        while i + 1 < len(pool):
            (Ga, sa), (Gb, sb) = pool[i], pool[i+1]
            result = kinetic_collision(Ga, Gb, cross_strength, noise_std, d_contact, d_max, max_steps, rng)
            if result is not None:
                new_pool.append((result, sa + sb))
                merges_this_round += 1
            else:
                new_pool.append((Ga, sa))
                new_pool.append((Gb, sb))
            i += 2
        if len(pool) % 2 == 1:
            new_pool.append(pool[-1])
        pool = new_pool
        sizes = [s for _, s in pool]
        history.append({
            "round": r + 1,
            "num_entities": len(pool),
            "merges": merges_this_round,
            "max_size": max(sizes),
            "mean_size": np.mean(sizes),
            "size_distribution": np.bincount(sizes)[1:] if sizes else [],
        })
    return pool, history


pool, history = run_recursive_aggregation(seed=1, num_elements=300, rounds=8)

print("=" * 70)
print(" Recursive aggregation: multi-round kinetic collision/recrystallization")
print("=" * 70)
for h in history:
    print(f"round {h['round']}: entities={h['num_entities']:4d}  merges={h['merges']:4d}  "
          f"max_size={h['max_size']:3d}  mean_size={h['mean_size']:.2f}")

final_sizes = [s for _, s in pool]
print(f"\nFinal state after {history[-1]['round']} rounds:")
print(f"  Number of surviving entities: {len(pool)} (started from 300 individual elements)")
print(f"  Largest composite: {max(final_sizes)} original elements")
print(f"  Size distribution (count of entities by size):")
dist = np.bincount(final_sizes)
for size, count in enumerate(dist):
    if count > 0:
        print(f"    size {size}: {count} entities")

print("\n--- Checking whether margin degrades with repeated merging (composite fragility) ---")
def lambda_min_S2(M):
    S = 0.5*(M+M.T)
    return np.linalg.eigvalsh(S).min()

sizes_and_margins = [(s, lambda_min_S2(G)) for G, s in pool]
sizes_and_margins.sort()
for s, m in sizes_and_margins:
    print(f"  size={s:3d}  current margin={m:.3f}")
