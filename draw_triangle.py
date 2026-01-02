#!/usr/bin/env python3
"""
Draw a single colored interlacing triangle with arcs showing dependencies.

Usage:
  python3 draw_triangle.py <n> <q> <num_steps> [seed]

Examples:
  python3 draw_triangle.py 5 1.0 100000        # Uniform random sample
  python3 draw_triangle.py 50 0.5 5000000      # q-weighted sample
  python3 draw_triangle.py 50 0.5 5000000 42   # With specific seed
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, ConnectionPatch
import random
from verify import (
    enumerate_triangles, compute_psi,
    try_level1_swap, try_level2_swap
)

def sample_triangle(n, q, num_steps):
    """Sample a triangle using MCMC with progress reporting."""
    import sys

    level1 = list(range(1, n + 1))
    level2 = [c for c in range(1, n + 1) for _ in range(2)]

    current = (tuple(level1), tuple(level2))
    current_psi = compute_psi(level1, level2, n)
    current_weight = q ** current_psi

    num_level1_swaps = n - 1
    num_level2_swaps = 2 * n - 1
    total_swaps = num_level1_swaps + num_level2_swaps

    # Progress interval: ~10 updates, min 100k steps
    progress_interval = max(100000, num_steps // 10)

    # Stats for each window
    psi_sum = 0
    psi_count = 0
    psi_min = psi_max = current_psi
    accepted = 0
    total = 0

    for step in range(num_steps):
        swap_choice = random.randint(0, total_swaps - 1)

        if swap_choice < num_level1_swaps:
            proposed = try_level1_swap(current[0], current[1], swap_choice, n)
        else:
            proposed = try_level2_swap(current[0], current[1], swap_choice - num_level1_swaps, n)

        total += 1
        if proposed is not None:
            proposed_psi = compute_psi(proposed[0], proposed[1], n)
            proposed_weight = q ** proposed_psi

            accept_prob = min(1.0, proposed_weight / current_weight)

            if random.random() < accept_prob:
                current = proposed
                current_psi = proposed_psi
                current_weight = proposed_weight
                accepted += 1

        # Track psi stats
        psi_sum += current_psi
        psi_count += 1
        psi_min = min(psi_min, current_psi)
        psi_max = max(psi_max, current_psi)

        # Progress report
        if (step + 1) % progress_interval == 0:
            pct = 100.0 * (step + 1) / num_steps
            avg_psi = psi_sum / psi_count
            accept_rate = 100.0 * accepted / total
            print(f"[{pct:.1f}%] step {step+1}/{num_steps} | psi={current_psi} avg={avg_psi:.2f} [{psi_min},{psi_max}] | accept={accept_rate:.1f}%", file=sys.stderr)
            # Reset window stats
            psi_sum = 0
            psi_count = 0
            psi_min = psi_max = current_psi

    return current, current_psi

def get_color_palette(n):
    """Get a nice color palette for n colors."""
    if n <= 10:
        # Use tab10 for small n
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(n)]
    else:
        # Use full rainbow (HSV) for larger n
        cmap = plt.cm.hsv
        return [cmap(i / n) for i in range(n)]

def draw_triangle(level1, level2, n, psi=None, filename='triangle.png'):
    """Draw the triangle with arcs showing color dependencies."""

    # Scale parameters based on n
    if n <= 10:
        box_size = 0.5
        x_spacing = 1.0
        arc_width = 2.0
        arc_alpha = 0.6
        fontsize = 9
        fig_width = max(10, n * 1.2)
    elif n <= 30:
        box_size = 0.35
        x_spacing = 0.5
        arc_width = 1.2
        arc_alpha = 0.5
        fontsize = 6
        fig_width = max(12, n * 0.6)
    elif n <= 60:
        box_size = 0.25
        x_spacing = 0.3
        arc_width = 1.5
        arc_alpha = 0.5
        fontsize = 0  # No labels for large n
        fig_width = max(16, n * 0.4)
    else:
        box_size = 0.18
        x_spacing = 0.22
        arc_width = 0.6
        arc_alpha = 0.6
        fontsize = 0
        fig_width = min(40, max(25, n * 0.3))

    fig_height = 5 if n <= 30 else 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = get_color_palette(n)
    level1_y = 0
    level2_y = 2.0 if n > 30 else 1.5

    level2_x_start = 0
    level1_x_start = x_spacing / 2  # Offset so level1[i] is between level2[2i] and level2[2i+1]

    # Draw level 2 boxes (top)
    for i, c in enumerate(level2):
        x = level2_x_start + i * x_spacing
        color = colors[c - 1]
        rect = FancyBboxPatch(
            (x - box_size/2, level2_y - box_size/2),
            box_size, box_size,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)
        # Add color number (only for small n)
        if fontsize > 0:
            text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
            ax.text(x, level2_y, str(c), ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=text_color)

    # Draw level 1 boxes (bottom, staggered)
    for i, c in enumerate(level1):
        x = level1_x_start + i * (2 * x_spacing)  # Position between level2[2i] and level2[2i+1]
        color = colors[c - 1]
        rect = FancyBboxPatch(
            (x - box_size/2, level1_y - box_size/2),
            box_size, box_size,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)
        if fontsize > 0:
            text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
            ax.text(x, level1_y, str(c), ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=text_color)

    # Draw arcs connecting each color's occurrences
    for c in range(1, n + 1):
        color = colors[c - 1]

        # Find positions
        level1_pos = None
        level2_positions = []

        for i, val in enumerate(level1):
            if val == c:
                level1_pos = level1_x_start + i * (2 * x_spacing)
                break

        for i, val in enumerate(level2):
            if val == c:
                level2_positions.append(level2_x_start + i * x_spacing)

        # Draw arcs from level1 to each level2 occurrence
        for l2_x in level2_positions:
            # Draw a line (from level1 top to level2 bottom)
            # Use straight lines - simpler and cleaner
            ax.plot([level1_pos, l2_x],
                    [level1_y + box_size/2, level2_y - box_size/2],
                    color=color, alpha=arc_alpha, linewidth=arc_width)

    # No labels or title - info goes in filename

    # Adjust axes
    ax.set_xlim(-2, level2_x_start + (2*n - 1) * x_spacing + 1)
    ax.set_ylim(level1_y - 1.0, level2_y + 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    dpi = 150 if n <= 30 else 300
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {filename}')

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 draw_triangle.py <n> <q> <num_steps> [seed]")
        print("Example: python3 draw_triangle.py 50 0.5 5000000")
        sys.exit(1)

    n = int(sys.argv[1])
    q = float(sys.argv[2])
    num_steps = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if seed is not None:
        random.seed(seed)

    print(f"Sampling triangle with n={n}, q={q}, steps={num_steps}", file=sys.stderr)
    (level1, level2), psi = sample_triangle(n, q, num_steps)

    print(f"Level 1: {level1}", file=sys.stderr)
    print(f"Level 2: {level2}", file=sys.stderr)
    print(f"psi = {psi}", file=sys.stderr)

    filename = f"triangle_n{n}_q{q}_psi{psi}.png"
    draw_triangle(level1, level2, n, psi, filename)

if __name__ == "__main__":
    main()
