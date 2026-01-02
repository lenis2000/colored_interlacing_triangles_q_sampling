#!/usr/bin/env python3
"""
Verification script for the depth-2 triangle sampler.

Compares sampled psi distribution with exact enumeration for small n.
"""

import sys
from itertools import permutations
from collections import Counter
import random

def check_interlacing(level1, level2, n):
    """Check if levels satisfy interlacing condition."""
    for color in range(1, n + 1):
        levels_of_color = []
        for i in range(n):
            if level2[2 * i] == color:
                levels_of_color.append(2)
            if level1[i] == color:
                levels_of_color.append(1)
            if level2[2 * i + 1] == color:
                levels_of_color.append(2)
        if levels_of_color != [2, 1, 2]:
            return False
    return True

def compute_psi(level1, level2, n):
    """Compute psi statistic."""
    active = {c: 1 for c in range(1, n + 1)}
    psi = 0

    for i in range(n - 1, -1, -1):
        # Top right exits
        active[level2[2 * i + 1]] = 0
        # Bottom enters
        c_bottom = level1[i]
        psi += sum(active[c] for c in range(c_bottom + 1, n + 1))
        active[c_bottom] = 1
        # Top left exits
        active[level2[2 * i]] = 0

    return psi

def enumerate_triangles(n):
    """Enumerate all valid depth-2 triangles."""
    triangles = []

    for level1 in permutations(range(1, n + 1)):
        level1 = list(level1)

        # Generate all level2 with each color appearing twice
        colors = list(range(1, n + 1)) * 2
        seen = set()

        for level2 in permutations(colors):
            if level2 in seen:
                continue
            seen.add(level2)
            level2 = list(level2)

            if check_interlacing(level1, level2, n):
                triangles.append((tuple(level1), tuple(level2)))

    return triangles

def exact_psi_distribution(triangles, n, q):
    """Compute exact q-weighted psi distribution."""
    psi_weights = {}
    Z = 0.0

    for level1, level2 in triangles:
        psi = compute_psi(level1, level2, n)
        w = q ** psi
        psi_weights[psi] = psi_weights.get(psi, 0) + w
        Z += w

    return {psi: w / Z for psi, w in psi_weights.items()}, Z

def try_level2_swap(level1, level2, pos, n):
    """Try to swap adjacent entries at level 2."""
    if pos < 0 or pos >= 2 * n - 1:
        return None
    if level2[pos] == level2[pos + 1]:
        return None

    new_level2 = list(level2)
    new_level2[pos], new_level2[pos + 1] = new_level2[pos + 1], new_level2[pos]

    if check_interlacing(level1, new_level2, n):
        return (level1, tuple(new_level2))
    return None

def try_level1_swap(level1, level2, pos, n):
    """Try to swap adjacent entries at level 1 with reconciliation."""
    if pos < 0 or pos >= n - 1:
        return None

    a = level1[pos]
    b = level1[pos + 1]
    if a == b:
        return None

    new_level1 = list(level1)
    new_level2 = list(level2)

    new_level1[pos], new_level1[pos + 1] = new_level1[pos + 1], new_level1[pos]

    between_left = level2[2 * pos]
    between_right = level2[2 * pos + 1]

    # Count how many a's and b's in between
    a_count = (1 if between_left == a else 0) + (1 if between_right == a else 0)
    b_count = (1 if between_left == b else 0) + (1 if between_right == b else 0)

    if a_count == 0 and b_count == 0:
        # Neither a nor b in between - no reconciliation needed
        pass
    elif a_count > 0 and b_count == 0:
        # Only a in between: replace a's with b's, then find b's to right and replace with a's
        if between_left == a:
            new_level2[2 * pos] = b
        if between_right == a:
            new_level2[2 * pos + 1] = b

        # Find a_count many b's to the right and replace with a
        replaced = 0
        for i in range(2 * (pos + 1), 2 * n):
            if new_level2[i] == b and replaced < a_count:
                new_level2[i] = a
                replaced += 1
    elif a_count == 0 and b_count > 0:
        # Only b in between: replace b's with a's, then find a's to left and replace with b's
        if between_left == b:
            new_level2[2 * pos] = a
        if between_right == b:
            new_level2[2 * pos + 1] = a

        # Find b_count many a's to the left and replace with b
        replaced = 0
        for i in range(2 * pos - 1, -1, -1):
            if new_level2[i] == a and replaced < b_count:
                new_level2[i] = b
                replaced += 1
    else:
        # Both a and b in between
        # Find occurrences of a and b OUTSIDE the between region that are closest to it
        # and swap them

        # Find rightmost a to the left of position 2*pos
        a_left = -1
        for i in range(2 * pos - 1, -1, -1):
            if new_level2[i] == a:
                a_left = i
                break

        # Find leftmost b to the right of position 2*pos+1
        b_right = -1
        for i in range(2 * (pos + 1), 2 * n):
            if new_level2[i] == b:
                b_right = i
                break

        # Swap these occurrences
        if a_left >= 0 and b_right >= 0:
            new_level2[a_left] = b
            new_level2[b_right] = a

    if check_interlacing(new_level1, new_level2, n):
        return (tuple(new_level1), tuple(new_level2))
    return None

def mcmc_sample(n, q, num_samples, burn_in=5000, thin=50):
    """Run MCMC and collect samples."""
    # Initialize to identity
    level1 = tuple(range(1, n + 1))
    level2 = tuple(c for c in range(1, n + 1) for _ in range(2))

    current = (level1, level2)
    current_psi = compute_psi(level1, level2, n)
    current_weight = q ** current_psi

    num_level1_swaps = n - 1
    num_level2_swaps = 2 * n - 1
    total_swaps = num_level1_swaps + num_level2_swaps

    samples = []
    step = 0

    while len(samples) < num_samples:
        # Propose
        swap_choice = random.randint(0, total_swaps - 1)

        if swap_choice < num_level1_swaps:
            proposed = try_level1_swap(current[0], current[1], swap_choice, n)
        else:
            proposed = try_level2_swap(current[0], current[1], swap_choice - num_level1_swaps, n)

        if proposed is not None:
            proposed_psi = compute_psi(proposed[0], proposed[1], n)
            proposed_weight = q ** proposed_psi

            accept_prob = min(1.0, proposed_weight / current_weight)

            if random.random() < accept_prob:
                current = proposed
                current_psi = proposed_psi
                current_weight = proposed_weight

        step += 1

        if step > burn_in and (step - burn_in) % thin == 0:
            samples.append(current_psi)

    return samples

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 verify.py <n> <q> <num_samples>")
        print("Example: python3 verify.py 3 0.5 50000")
        sys.exit(1)

    n = int(sys.argv[1])
    q = float(sys.argv[2])
    num_samples = int(sys.argv[3])

    if n > 5:
        print(f"Warning: n={n} may be slow for exact enumeration")

    print(f"=== Verification for n={n}, q={q} ===")
    print()

    # Exact enumeration
    print("Enumerating all triangles...")
    triangles = enumerate_triangles(n)
    print(f"Total triangles: {len(triangles)}")

    exact_dist, Z = exact_psi_distribution(triangles, n, q)
    print(f"Normalizing constant Z = T_2({n}; {q}) = {Z:.6f}")
    print()

    # MCMC sampling
    print(f"Running MCMC for {num_samples} samples...")
    samples = mcmc_sample(n, q, num_samples)

    sampled_counts = Counter(samples)
    sampled_dist = {psi: count / num_samples for psi, count in sampled_counts.items()}

    # Compare
    print()
    print("Comparison (exact vs sampled):")
    print("-" * 50)
    print(f"{'psi':>5} {'exact':>10} {'sampled':>10} {'diff':>10}")
    print("-" * 50)

    all_psi = sorted(set(exact_dist.keys()) | set(sampled_dist.keys()))
    total_diff = 0.0

    for psi in all_psi:
        exact = exact_dist.get(psi, 0)
        sampled = sampled_dist.get(psi, 0)
        diff = abs(exact - sampled)
        total_diff += diff
        print(f"{psi:>5} {exact:>10.4f} {sampled:>10.4f} {diff:>10.4f}")

    print("-" * 50)
    print(f"Total variation distance: {total_diff / 2:.4f}")
    print()

    # Expected TV for this sample size
    expected_tv = 1.0 / (2 * num_samples ** 0.5)
    print(f"Expected TV for {num_samples} samples: ~{expected_tv:.4f}")

    if total_diff / 2 < 3 * expected_tv:
        print("PASS: Sampled distribution matches exact distribution")
    else:
        print("WARNING: Large deviation from exact distribution")

if __name__ == "__main__":
    main()
