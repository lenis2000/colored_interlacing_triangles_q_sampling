/*
 * Metropolis-Hastings Sampler for Depth-2 Colored Interlacing Triangles
 *
 * Samples from the q-weighted measure on colored interlacing n-triangles
 * with N=2 levels. The equilibrium distribution is:
 *
 *     P(triangle) proportional to q^{psi(triangle)}
 *
 * where psi is the inter-level statistic from the LLT Central Limit Theorem.
 *
 * The Markov chain uses two types of moves:
 *   1. Level-2 swaps: swap adjacent entries at level 2
 *   2. Level-1 swaps with reconciliation: swap adjacent entries at level 1,
 *      then adjust level 2 to maintain interlacing
 *
 * Both move types preserve interlacing and together form an ergodic chain.
 *
 * Reference: Blitvic-Petrov, "Colored interlacing triangles and Genocchi medians"
 *
 * Compile: g++ -O3 -std=c++17 sampler.cpp -o sampler
 * Run: ./sampler <n> <q> <num_steps> [seed]
 * Example: ./sampler 5 0.5 1000000
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;

// ============================================================================
// TRIANGLE REPRESENTATION
// ============================================================================

struct Triangle {
    int n;                    // Number of colors
    vector<int> level1;       // Bottom level: n entries (permutation of 1..n)
    vector<int> level2;       // Top level: 2n entries (each color appears twice)

    Triangle(int n) : n(n), level1(n), level2(2 * n) {}

    // Initialize to identity triangle: level1 = (1,2,...,n), level2 = (1,1,2,2,...,n,n)
    void initialize_identity() {
        for (int i = 0; i < n; i++) {
            level1[i] = i + 1;
            level2[2 * i] = i + 1;
            level2[2 * i + 1] = i + 1;
        }
    }

    void print(ostream& out = cout) const {
        out << "Level 1: ";
        for (int c : level1) out << c << " ";
        out << "\nLevel 2: ";
        for (int c : level2) out << c << " ";
        out << "\n";
    }
};

// ============================================================================
// INTERLACING CHECK
// ============================================================================

/*
 * Linear order between levels 1 and 2 (from Definition 1.1 in the paper):
 *
 * For each triangle i = 1,...,n:
 *   lambda[i]^2_1, lambda[i]^1_1, lambda[i]^2_2
 *
 * Concatenated: (L2[0], L1[0], L2[1]), (L2[2], L1[1], L2[3]), ...
 *
 * Interlacing condition: For each color c, in the linear order,
 * the pattern of levels for c must be (2, 1, 2).
 */
bool check_interlacing(const Triangle& tri) {
    const int n = tri.n;

    for (int color = 1; color <= n; color++) {
        // Find positions of this color in the interlaced sequence
        vector<int> levels_of_color;

        for (int i = 0; i < n; i++) {
            // Position 3i: level2[2i]
            if (tri.level2[2 * i] == color) levels_of_color.push_back(2);
            // Position 3i+1: level1[i]
            if (tri.level1[i] == color) levels_of_color.push_back(1);
            // Position 3i+2: level2[2i+1]
            if (tri.level2[2 * i + 1] == color) levels_of_color.push_back(2);
        }

        // Must be exactly (2, 1, 2)
        if (levels_of_color.size() != 3) return false;
        if (levels_of_color[0] != 2 || levels_of_color[1] != 1 || levels_of_color[2] != 2) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// PSI STATISTIC (q-weight)
// ============================================================================

/*
 * Compute psi(lambda^1, lambda^2) using the vertex model from Section 4.
 *
 * Process the interlaced sequence from right to left:
 * - Maintain set A of "active" colors (on the horizontal line)
 * - When color exits upward (level 2): remove from A
 * - When color enters from below (level 1): add contribution |{c' in A : c' > c}|
 */
int compute_psi(const Triangle& tri) {
    const int n = tri.n;

    // Initialize: all colors active
    vector<int> active(n + 1, 1);  // active[c] = 1 if color c is active

    int psi = 0;

    // Process right to left
    for (int i = n - 1; i >= 0; i--) {
        // Level 2 right position: exits upward
        int c_top_right = tri.level2[2 * i + 1];
        active[c_top_right] = 0;

        // Level 1 position: enters from below
        int c_bottom = tri.level1[i];
        // Count active colors greater than c_bottom
        for (int c = c_bottom + 1; c <= n; c++) {
            psi += active[c];
        }
        active[c_bottom] = 1;

        // Level 2 left position: exits upward
        int c_top_left = tri.level2[2 * i];
        active[c_top_left] = 0;
    }

    return psi;
}

// ============================================================================
// MCMC MOVES
// ============================================================================

/*
 * Level-2 swap: swap level2[pos] with level2[pos+1].
 * Returns true if the swap preserves interlacing.
 */
bool try_level2_swap(Triangle& tri, int pos) {
    if (pos < 0 || pos >= 2 * tri.n - 1) return false;
    if (tri.level2[pos] == tri.level2[pos + 1]) return false;  // No change

    swap(tri.level2[pos], tri.level2[pos + 1]);

    if (check_interlacing(tri)) {
        return true;
    }

    // Revert
    swap(tri.level2[pos], tri.level2[pos + 1]);
    return false;
}

/*
 * Level-1 swap with reconciliation: swap level1[pos] with level1[pos+1],
 * then adjust level2 to maintain interlacing.
 *
 * Reconciliation rule (from notes on swap operations):
 * - Case 1: Neither a nor b in between region -> no change at level 2
 * - Case 2: Only a (or b) in between -> replace with b (or a), compensate outside
 * - Case 3: Both a and b in between -> swap closest occurrences OUTSIDE the between region
 */
bool try_level1_swap(Triangle& tri, int pos) {
    if (pos < 0 || pos >= tri.n - 1) return false;

    const int n = tri.n;
    const int a = tri.level1[pos];
    const int b = tri.level1[pos + 1];

    if (a == b) return false;  // No change

    // Save original state
    vector<int> old_level1 = tri.level1;
    vector<int> old_level2 = tri.level2;

    // Swap at level 1
    swap(tri.level1[pos], tri.level1[pos + 1]);

    // The "between" region in level2: positions 2*pos and 2*pos+1
    int between_left = old_level2[2 * pos];
    int between_right = old_level2[2 * pos + 1];

    // Count occurrences of a and b in between
    int a_count = (between_left == a ? 1 : 0) + (between_right == a ? 1 : 0);
    int b_count = (between_left == b ? 1 : 0) + (between_right == b ? 1 : 0);

    // Reconciliation cases
    if (a_count == 0 && b_count == 0) {
        // Case 1: Neither a nor b in between region - no reconciliation needed
    }
    else if (a_count > 0 && b_count == 0) {
        // Case 2a: Only a is between
        // Replace a with b in between, then find a_count many b's to the right and replace with a
        if (between_left == a) tri.level2[2 * pos] = b;
        if (between_right == a) tri.level2[2 * pos + 1] = b;

        int replaced = 0;
        for (int i = 2 * (pos + 1); i < 2 * n && replaced < a_count; i++) {
            if (tri.level2[i] == b) {
                tri.level2[i] = a;
                replaced++;
            }
        }
    }
    else if (a_count == 0 && b_count > 0) {
        // Case 2b: Only b is between
        // Replace b with a in between, then find b_count many a's to the left and replace with b
        if (between_left == b) tri.level2[2 * pos] = a;
        if (between_right == b) tri.level2[2 * pos + 1] = a;

        int replaced = 0;
        for (int i = 2 * pos - 1; i >= 0 && replaced < b_count; i--) {
            if (tri.level2[i] == a) {
                tri.level2[i] = b;
                replaced++;
            }
        }
    }
    else {
        // Case 3: Both a and b are in between
        // Swap the occurrences of a and b OUTSIDE the between region that are closest to it

        // Find rightmost a to the left of position 2*pos
        int a_left = -1;
        for (int i = 2 * pos - 1; i >= 0; i--) {
            if (tri.level2[i] == a) {
                a_left = i;
                break;
            }
        }

        // Find leftmost b to the right of position 2*pos+1
        int b_right = -1;
        for (int i = 2 * (pos + 1); i < 2 * n; i++) {
            if (tri.level2[i] == b) {
                b_right = i;
                break;
            }
        }

        // Swap these occurrences
        if (a_left >= 0 && b_right >= 0) {
            tri.level2[a_left] = b;
            tri.level2[b_right] = a;
        }
    }

    // Check if result is valid
    if (check_interlacing(tri)) {
        return true;
    }

    // Revert
    tri.level1 = old_level1;
    tri.level2 = old_level2;
    return false;
}

// ============================================================================
// METROPOLIS-HASTINGS SAMPLER
// ============================================================================

struct SamplerStats {
    long long total_steps = 0;
    long long accepted = 0;
    long long rejected_interlacing = 0;
    long long rejected_mh = 0;

    void print(ostream& out = cout) const {
        out << "Total steps: " << total_steps << "\n";
        out << "Accepted: " << accepted << " ("
            << fixed << setprecision(2) << (100.0 * accepted / total_steps) << "%)\n";
        out << "Rejected (interlacing): " << rejected_interlacing << "\n";
        out << "Rejected (MH): " << rejected_mh << "\n";
    }
};

/*
 * Run Metropolis-Hastings sampler.
 *
 * Parameters:
 *   n: number of colors
 *   q: weight parameter (0 < q). For q=1, samples uniformly.
 *   num_steps: number of MCMC steps
 *   seed: random seed
 *   collect_samples: if true, store all samples
 *   thin: thinning interval for collected samples
 *   burn_in: number of initial steps to discard
 *
 * Returns: vector of sampled triangles (empty if collect_samples=false)
 */
struct SamplerResult {
    vector<Triangle> samples;
    SamplerStats stats;
    Triangle final_state;
    int final_psi;
};

SamplerResult run_sampler(
    int n, double q, long long num_steps,
    unsigned int seed = 42,
    bool collect_samples = false,
    int thin = 1,
    long long burn_in = 0,
    long long progress_interval = 0,  // 0 = auto
    bool verbose = true
) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> uniform(0.0, 1.0);

    // Initialize
    Triangle current(n);
    current.initialize_identity();

    int current_psi = compute_psi(current);
    double current_weight = pow(q, current_psi);

    SamplerStats stats;
    vector<Triangle> samples;

    // Number of possible swaps
    const int num_level1_swaps = n - 1;
    const int num_level2_swaps = 2 * n - 1;
    const int total_swaps = num_level1_swaps + num_level2_swaps;

    // Auto progress interval: ~10 updates, but at least every 100000 steps
    if (progress_interval == 0) {
        progress_interval = max(100000LL, num_steps / 10);
    }

    // Track psi statistics for equilibration monitoring
    long long psi_sum = 0;
    long long psi_count = 0;
    int psi_min = current_psi, psi_max = current_psi;

    for (long long step = 0; step < num_steps; step++) {
        stats.total_steps++;

        // Propose a swap
        Triangle proposed = current;
        int swap_choice = (int)(uniform(rng) * total_swaps);

        bool valid_swap;
        if (swap_choice < num_level1_swaps) {
            valid_swap = try_level1_swap(proposed, swap_choice);
        } else {
            valid_swap = try_level2_swap(proposed, swap_choice - num_level1_swaps);
        }

        if (!valid_swap) {
            stats.rejected_interlacing++;
            // Stay at current state
        } else {
            // Metropolis-Hastings acceptance
            int proposed_psi = compute_psi(proposed);
            double proposed_weight = pow(q, proposed_psi);

            double accept_prob = min(1.0, proposed_weight / current_weight);

            if (uniform(rng) < accept_prob) {
                current = proposed;
                current_psi = proposed_psi;
                current_weight = proposed_weight;
                stats.accepted++;
            } else {
                stats.rejected_mh++;
            }
        }

        // Track psi stats
        psi_sum += current_psi;
        psi_count++;
        psi_min = min(psi_min, current_psi);
        psi_max = max(psi_max, current_psi);

        // Progress report
        if (verbose && (step + 1) % progress_interval == 0) {
            double pct = 100.0 * (step + 1) / num_steps;
            double avg_psi = (double)psi_sum / psi_count;
            double accept_rate = 100.0 * stats.accepted / stats.total_steps;
            cerr << "[" << fixed << setprecision(1) << pct << "%] "
                 << "step " << (step + 1) << "/" << num_steps
                 << " | psi=" << current_psi
                 << " avg=" << setprecision(2) << avg_psi
                 << " [" << psi_min << "," << psi_max << "]"
                 << " | accept=" << setprecision(1) << accept_rate << "%"
                 << endl;
            // Reset stats for next window
            psi_sum = 0;
            psi_count = 0;
            psi_min = psi_max = current_psi;
        }

        // Collect sample
        if (collect_samples && step >= burn_in && (step - burn_in) % thin == 0) {
            samples.push_back(current);
        }
    }

    return {samples, stats, current, current_psi};
}

// ============================================================================
// MAIN
// ============================================================================

void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " <n> <q> <num_steps> [seed]\n";
    cerr << "\n";
    cerr << "Arguments:\n";
    cerr << "  n          Number of colors (1 <= n <= 100)\n";
    cerr << "  q          Weight parameter (q > 0). Use q=1 for uniform sampling.\n";
    cerr << "  num_steps  Number of MCMC steps\n";
    cerr << "  seed       (Optional) Random seed (default: 42)\n";
    cerr << "\n";
    cerr << "Example: " << prog << " 5 0.5 1000000\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    int n = stoi(argv[1]);
    double q = stod(argv[2]);
    long long num_steps = stoll(argv[3]);
    unsigned int seed = (argc == 5) ? stoul(argv[4]) : 42;

    if (n < 1 || n > 100) {
        cerr << "Error: n must be between 1 and 100\n";
        return 1;
    }
    if (q <= 0) {
        cerr << "Error: q must be positive\n";
        return 1;
    }
    if (num_steps < 1) {
        cerr << "Error: num_steps must be positive\n";
        return 1;
    }

    cerr << "=== Depth-2 Triangle Sampler ===" << endl;
    cerr << "n = " << n << " colors" << endl;
    cerr << "q = " << q << endl;
    cerr << "num_steps = " << num_steps << endl;
    cerr << "seed = " << seed << endl;
    cerr << "================================\n" << endl;

    // Run sampler
    auto result = run_sampler(n, q, num_steps, seed, false);

    cerr << "\n=== Results ===" << endl;
    result.stats.print(cerr);
    cerr << "\nFinal triangle:" << endl;
    result.final_state.print(cerr);
    cerr << "Final psi = " << result.final_psi << endl;

    return 0;
}
