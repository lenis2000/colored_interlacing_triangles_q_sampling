#!/usr/bin/env python3
"""
Visualize sampling results from sampler_collect.

Reads CSV files and produces:
  - Psi histogram
  - Level 1 heatmap (n x n)
  - Level 2 heatmap (n x 2n)

Usage:
  python3 plot_samples.py <prefix>

Example:
  python3 plot_samples.py samples_n50_q0.200000

Recommended sampling parameters for n=50:
  burn_in = 100000
  thin = 1000
  num_samples = 10000
  total_steps = burn_in + num_samples * thin = 10,100,000
"""

# Recommended defaults
DEFAULT_BURN_IN = 100000
DEFAULT_THIN = 1000
DEFAULT_NUM_SAMPLES = 10000

def compute_total_steps(num_samples=DEFAULT_NUM_SAMPLES, burn_in=DEFAULT_BURN_IN, thin=DEFAULT_THIN):
    """Compute total MCMC steps."""
    return burn_in + num_samples * thin

def print_sampling_info(n, num_samples=DEFAULT_NUM_SAMPLES, burn_in=DEFAULT_BURN_IN, thin=DEFAULT_THIN):
    """Print recommended sampling parameters."""
    total = compute_total_steps(num_samples, burn_in, thin)
    print(f"Sampling parameters for n={n}:")
    print(f"  burn_in     = {burn_in:,}")
    print(f"  thin        = {thin:,}")
    print(f"  num_samples = {num_samples:,}")
    print(f"  total_steps = {total:,}")
    print()

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def read_csv_psi(filename):
    """Read psi CSV file, return (psi_values, frequencies)."""
    psi_vals, freqs = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psi_vals.append(int(row['psi']))
            freqs.append(float(row['frequency']))
    return np.array(psi_vals), np.array(freqs)

def read_csv_matrix(filename):
    """Read level CSV file, return numpy array."""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = []
        for row in reader:
            data.append([float(x) for x in row[1:]])  # skip first column (color)
    return np.array(data)

def plot_psi_histogram(psi_file, output_file):
    """Plot histogram of psi values."""
    psi_vals, freqs = read_csv_psi(psi_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(psi_vals, freqs, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'$\psi$', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(r'Distribution of $\psi$', fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    mean_psi = (psi_vals * freqs).sum()
    ax.axvline(x=mean_psi, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_psi:.1f}')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_file}')

def plot_level1_heatmap(level1_file, output_file):
    """Plot heatmap of level 1 position frequencies."""
    data = read_csv_matrix(level1_file)
    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), max(6, n * 0.25)))

    cmap = LinearSegmentedColormap.from_list('custom', ['white', 'steelblue', 'darkblue'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=data.max())

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Color', fontsize=12)
    ax.set_title(f'Level 1 position frequencies ($n={n}$)', fontsize=14)

    # Ticks
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_yticklabels(range(1, n+1))
    else:
        # Sparse ticks for large n
        tick_step = max(1, n // 10)
        ax.set_xticks(range(0, n, tick_step))
        ax.set_yticks(range(0, n, tick_step))
        ax.set_yticklabels(range(1, n+1, tick_step))

    plt.colorbar(im, ax=ax, label='Frequency')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_file}')

def plot_level2_heatmap(level2_file, output_file):
    """Plot heatmap of level 2 position frequencies."""
    data = read_csv_matrix(level2_file)
    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.4), max(6, n * 0.25)))

    cmap = LinearSegmentedColormap.from_list('custom', ['white', 'coral', 'darkred'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=data.max())

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Color', fontsize=12)
    ax.set_title(f'Level 2 position frequencies ($n={n}$, $2n={2*n}$ positions)', fontsize=14)

    # Ticks
    if n <= 20:
        ax.set_xticks(range(2*n))
        ax.set_yticks(range(n))
        ax.set_yticklabels(range(1, n+1))
    else:
        tick_step = max(1, n // 10)
        ax.set_xticks(range(0, 2*n, tick_step * 2))
        ax.set_yticks(range(0, n, tick_step))
        ax.set_yticklabels(range(1, n+1, tick_step))

    plt.colorbar(im, ax=ax, label='Frequency')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_file}')

def plot_combined(prefix):
    """Create a combined figure with all three plots."""
    psi_file = prefix + '_psi.csv'
    level1_file = prefix + '_level1.csv'
    level2_file = prefix + '_level2.csv'

    psi_vals, freqs = read_csv_psi(psi_file)
    data_l1 = read_csv_matrix(level1_file)
    data_l2 = read_csv_matrix(level2_file)

    n = data_l1.shape[0]

    fig = plt.figure(figsize=(16, 12))

    # Psi histogram (top)
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.bar(psi_vals, freqs, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel(r'$\psi$', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    mean_psi = (psi_vals * freqs).sum()
    ax1.axvline(x=mean_psi, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_psi:.1f}')
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_title(r'Distribution of $\psi$', fontsize=16)

    # Level 1 heatmap (bottom left)
    ax2 = fig.add_subplot(2, 2, 3)
    cmap1 = LinearSegmentedColormap.from_list('custom', ['white', 'steelblue', 'darkblue'])
    im1 = ax2.imshow(data_l1, cmap=cmap1, aspect='auto', vmin=0)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Color', fontsize=12)
    ax2.set_title(f'Level 1 ($n \\times n$)', fontsize=14)
    plt.colorbar(im1, ax=ax2)

    # Level 2 heatmap (bottom right)
    ax3 = fig.add_subplot(2, 2, 4)
    cmap2 = LinearSegmentedColormap.from_list('custom', ['white', 'coral', 'darkred'])
    im2 = ax3.imshow(data_l2, cmap=cmap2, aspect='auto', vmin=0)
    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Color', fontsize=12)
    ax3.set_title(f'Level 2 ($n \\times 2n$)', fontsize=14)
    plt.colorbar(im2, ax=ax3)

    plt.suptitle(f'Sampling results: {prefix}', fontsize=18, y=1.02)
    plt.tight_layout()

    output_file = prefix + '_combined.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_file}')

def run_sampler(n, q, num_samples=DEFAULT_NUM_SAMPLES, burn_in=DEFAULT_BURN_IN, thin=DEFAULT_THIN):
    """Run the C++ sampler and return the output prefix."""
    import subprocess
    import os

    # Compile if needed
    sampler_path = os.path.join(os.path.dirname(__file__) or '.', 'sampler_collect')
    source_path = os.path.join(os.path.dirname(__file__) or '.', 'sampler_collect.cpp')

    if not os.path.exists(sampler_path) or os.path.getmtime(source_path) > os.path.getmtime(sampler_path):
        print("Compiling sampler_collect.cpp...")
        result = subprocess.run(
            ['g++', '-O3', '-std=c++17', source_path, '-o', sampler_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            sys.exit(1)

    # Run sampler
    print_sampling_info(n, num_samples, burn_in, thin)
    cmd = [sampler_path, str(n), str(q), str(num_samples), str(burn_in), str(thin)]
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Sampler failed!")
        sys.exit(1)

    # Return prefix (match C++ formatting)
    return f"samples_n{n}_q{q:f}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_samples.py <n> <q> [num_samples] [burn_in] [thin]")
        print()
        print("Examples:")
        print("  python3 plot_samples.py 25 0.2              # Use defaults")
        print("  python3 plot_samples.py 50 0.5 10000 100000 1000")
        print()
        print(f"Defaults: num_samples={DEFAULT_NUM_SAMPLES}, burn_in={DEFAULT_BURN_IN}, thin={DEFAULT_THIN}")
        print(f"Total steps = {compute_total_steps():,}")
        sys.exit(1)

    n = int(sys.argv[1])
    q = float(sys.argv[2])
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NUM_SAMPLES
    burn_in = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_BURN_IN
    thin = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_THIN

    # Run sampler
    prefix = run_sampler(n, q, num_samples, burn_in, thin)

    # Check files exist
    import os
    for suffix in ['_psi.csv', '_level1.csv', '_level2.csv']:
        if not os.path.exists(prefix + suffix):
            print(f"Error: {prefix + suffix} not found")
            sys.exit(1)

    # Generate plots
    print("\nGenerating plots...")
    plot_psi_histogram(prefix + '_psi.csv', prefix + '_psi.png')
    plot_level1_heatmap(prefix + '_level1.csv', prefix + '_level1.png')
    plot_level2_heatmap(prefix + '_level2.csv', prefix + '_level2.png')
    plot_combined(prefix)

    print("\nDone!")

if __name__ == "__main__":
    main()
