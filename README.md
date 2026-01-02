# MCMC Sampling of Depth-2 Colored Interlacing Triangles

Metropolis-Hastings sampler for q-weighted colored interlacing n-triangles with depth N=2.
See also 
[https://github.com/lenis2000/colored_interlacing_triangles_enumeration](https://github.com/lenis2000/colored_interlacing_triangles_enumeration) for enumeration, and
[https://github.com/lenis2000/colored_interlacing_triangles_q_enumeration](https://github.com/lenis2000/colored_interlacing_triangles_q_enumeration) for q-enumeration.


arXiv paper: (TO BE INSTERTED)

## Quick Start

```bash
# Build
make

# Draw a single triangle (n=50, q=0.2, 10M steps)
python3 draw_triangle.py 50 0.2 10000000

# Collect samples and generate plots (one command)
python3 plot_samples.py 25 0.5

# Verify against exact enumeration (small n only)
python3 verify.py 4 0.5 50000
```

## Files

| File | Description |
|------|-------------|
| `sampler.cpp` | Basic MCMC sampler with progress reporting |
| `sampler_collect.cpp` | Collects samples and outputs statistics to CSV |
| `draw_triangle.py` | Draw a single sampled triangle with colored lines |
| `plot_samples.py` | One-command workflow: compile, sample, and generate all plots |
| `verify.py` | Verification against exact enumeration (n ≤ 5) |
| `Makefile` | Build system |

## Target Distribution

The sampler targets the q-weighted distribution:

```
P(triangle) ∝ q^{ψ(triangle)}
```

where ψ is the inter-level inversion statistic from Section 4 of the paper.

| q value | Behavior |
|---------|----------|
| q = 1 | Uniform sampling |
| 0 < q < 1 | Favors low ψ (more "ordered" triangles) |
| q > 1 | Favors high ψ |

## Usage

### Draw Single Triangle

Runs MCMC and draws a triangle visualization with lines connecting each color's occurrences.

```bash
python3 draw_triangle.py <n> <q> <num_steps> [seed]

# Examples:
python3 draw_triangle.py 50 0.2 10000000      # Low q, ordered
python3 draw_triangle.py 50 0.98 10000000     # High q, disordered
python3 draw_triangle.py 25 0.5 5000000 42    # With seed
```

Output: `triangle_n<n>_q<q>_psi<psi>.png`

### Collect Samples and Plot (Recommended)

One command to compile, run sampler, and generate all visualizations:

```bash
python3 plot_samples.py <n> <q> [num_samples] [burn_in] [thin]

# Examples:
python3 plot_samples.py 25 0.5                    # Use defaults
python3 plot_samples.py 25 0.5 10000 100000 1000  # Custom parameters
python3 plot_samples.py 50 0.2 100000 500000 5000 # Large n
```

**Default parameters:**
- `num_samples = 10,000`
- `burn_in = 100,000`
- `thin = 1,000`
- `total_steps = burn_in + num_samples × thin = 10,100,000`

**Output files** (prefix = `samples_n<n>_q<q>`):
- `<prefix>_psi.csv` - Histogram of ψ values
- `<prefix>_level1.csv` - Position frequencies at level 1 (n × n)
- `<prefix>_level2.csv` - Position frequencies at level 2 (n × 2n)
- `<prefix>_psi.png` - ψ histogram with mean line
- `<prefix>_level1.png` - Level 1 heatmap
- `<prefix>_level2.png` - Level 2 heatmap
- `<prefix>_combined.png` - All plots in one figure

### Basic Sampler (Direct)

Low-level sampler with progress reporting:

```bash
./sampler <n> <q> <num_steps> [seed]

# Examples:
./sampler 5 1.0 1000000      # Uniform sampling
./sampler 50 0.2 10000000    # q-weighted, large n
```

Outputs final triangle state and ψ value.

### Sample Collector (Direct)

Collect multiple samples for statistics:

```bash
./sampler_collect <n> <q> <num_samples> [burn_in] [thin]

# Example:
./sampler_collect 25 0.5 10000 100000 1000
```

### Verification

Compare sampled ψ distribution with exact enumeration (only feasible for n ≤ 5):

```bash
python3 verify.py <n> <q> <num_samples>

# Example:
python3 verify.py 4 0.5 50000
```

## Algorithm

### State Space

A depth-2 colored interlacing n-triangle consists of:
- **Level 1**: A permutation λ¹ = (λ¹₁, ..., λ¹ₙ) of {1, ..., n}
- **Level 2**: A sequence λ² = (λ²₁, ..., λ²₂ₙ) where each color appears exactly twice

**Interlacing constraint**: For each color c, the positions must satisfy:
```
(first c in λ²) < (c in λ¹) < (second c in λ²)
```
in the linear order: λ²₁ < λ¹₁ < λ²₂ < λ²₃ < λ¹₂ < λ²₄ < ... < λ²₂ₙ

### Swap Operations

The Markov chain uses two types of moves:

| Move | Description |
|------|-------------|
| **Level-2 swap** | Adjacent transposition at level 2 (valid if preserves interlacing) |
| **Level-1 swap** | Adjacent transposition at level 1 with deterministic reconciliation |

### Level-1 Swap Reconciliation

When swapping colors a and b at positions i ↔ i+1 at level 1, the "between region" at level 2 consists of positions 2i-1 and 2i. The reconciliation rule:

| Case | Between Region | Action |
|------|----------------|--------|
| 1 | Neither a nor b | No change needed |
| 2a | Only a | Replace a→b in between; find first b to right, replace b→a |
| 2b | Only b | Replace b→a in between; find first a to left, replace a→b |
| 3 | Both a and b | Swap closest a (left of between) and b (right of between) |

### Metropolis-Hastings Acceptance

At each step:
1. Uniformly choose one of the 3n-2 possible swap positions
2. Attempt the swap (check validity for level-2 swaps)
3. Accept with probability:
```
α = min(1, q^{ψ(proposed) - ψ(current)})
```

This ensures detailed balance with the target distribution P ∝ q^ψ.

### Connectedness

The swap operations connect all states in T₂(n) (verified for n ≤ 5):

| n | |T₂(n)| | Connected |
|---|--------|-----------|
| 3 | 48 | ✓ |
| 4 | 1,344 | ✓ |
| 5 | 72,960 | ✓ |

## Performance

On Apple M2 Pro:

| n | Steps/second |
|---|--------------|
| 10 | ~5M |
| 25 | ~2M |
| 50 | ~1M |
| 100 | ~400K |

**Recommended parameters for production:**
- `burn_in = 100,000` (let chain mix before sampling)
- `thin = 1,000` (reduce autocorrelation between samples)
- `num_samples = 10,000` (for good statistics)

For larger n, increase burn_in and thin proportionally.

## Verified Results

Comparison with exact enumeration (q=1 uniform case):

| n | q | Samples | TV Distance |
|---|---|---------|-------------|
| 3 | 1.0 | 50,000 | < 0.01 |
| 4 | 1.0 | 50,000 | < 0.01 |
| 5 | 1.0 | 50,000 | < 0.01 |

For q ≠ 1, the sampler correctly weights by q^ψ (verified by comparing marginal ψ distributions).

## Mathematical Background

The ψ statistic counts inter-level inversions. For a triangle with levels λ¹ and λ²:

```
ψ = #{(i,j) : i appears before j at level 1, but j's right occurrence
     at level 2 appears before i's right occurrence}
```

See Section 4 of the paper for the precise definition.

## Reference

Blitvic-Petrov, "Colored interlacing triangles and Genocchi medians", Section 5.

Code repository: https://github.com/lenis2000/colored_interlacing_triangles_q_sampling
