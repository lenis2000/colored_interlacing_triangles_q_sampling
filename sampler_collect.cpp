/*
 * Sample Collector for Depth-2 Colored Interlacing Triangles
 *
 * Runs the MCMC sampler and collects statistics on the sampled triangles.
 * Outputs:
 *   - Histogram of psi values
 *   - Position frequencies for each color at each level
 *   - Comparison with exact distribution (for small n)
 *
 * Compile: g++ -O3 -std=c++17 sampler_collect.cpp -o sampler_collect
 * Run: ./sampler_collect <n> <q> <num_samples> [burn_in] [thin]
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <fstream>

using namespace std;

// ============================================================================
// TRIANGLE REPRESENTATION AND OPERATIONS (same as sampler.cpp)
// ============================================================================

struct Triangle {
    int n;
    vector<int> level1;
    vector<int> level2;

    Triangle(int n) : n(n), level1(n), level2(2 * n) {}

    void initialize_identity() {
        for (int i = 0; i < n; i++) {
            level1[i] = i + 1;
            level2[2 * i] = i + 1;
            level2[2 * i + 1] = i + 1;
        }
    }

    bool operator<(const Triangle& other) const {
        if (level1 != other.level1) return level1 < other.level1;
        return level2 < other.level2;
    }

    bool operator==(const Triangle& other) const {
        return level1 == other.level1 && level2 == other.level2;
    }
};

bool check_interlacing(const Triangle& tri) {
    const int n = tri.n;
    for (int color = 1; color <= n; color++) {
        vector<int> levels_of_color;
        for (int i = 0; i < n; i++) {
            if (tri.level2[2 * i] == color) levels_of_color.push_back(2);
            if (tri.level1[i] == color) levels_of_color.push_back(1);
            if (tri.level2[2 * i + 1] == color) levels_of_color.push_back(2);
        }
        if (levels_of_color.size() != 3) return false;
        if (levels_of_color[0] != 2 || levels_of_color[1] != 1 || levels_of_color[2] != 2) {
            return false;
        }
    }
    return true;
}

int compute_psi(const Triangle& tri) {
    const int n = tri.n;
    vector<int> active(n + 1, 1);
    int psi = 0;

    for (int i = n - 1; i >= 0; i--) {
        int c_top_right = tri.level2[2 * i + 1];
        active[c_top_right] = 0;

        int c_bottom = tri.level1[i];
        for (int c = c_bottom + 1; c <= n; c++) {
            psi += active[c];
        }
        active[c_bottom] = 1;

        int c_top_left = tri.level2[2 * i];
        active[c_top_left] = 0;
    }

    return psi;
}

bool try_level2_swap(Triangle& tri, int pos) {
    if (pos < 0 || pos >= 2 * tri.n - 1) return false;
    if (tri.level2[pos] == tri.level2[pos + 1]) return false;

    swap(tri.level2[pos], tri.level2[pos + 1]);

    if (check_interlacing(tri)) {
        return true;
    }

    swap(tri.level2[pos], tri.level2[pos + 1]);
    return false;
}

bool try_level1_swap(Triangle& tri, int pos) {
    if (pos < 0 || pos >= tri.n - 1) return false;

    const int n = tri.n;
    const int a = tri.level1[pos];
    const int b = tri.level1[pos + 1];

    if (a == b) return false;

    vector<int> old_level1 = tri.level1;
    vector<int> old_level2 = tri.level2;

    swap(tri.level1[pos], tri.level1[pos + 1]);

    int between_left = old_level2[2 * pos];
    int between_right = old_level2[2 * pos + 1];

    int a_count = (between_left == a ? 1 : 0) + (between_right == a ? 1 : 0);
    int b_count = (between_left == b ? 1 : 0) + (between_right == b ? 1 : 0);

    if (a_count == 0 && b_count == 0) {
        // Case 1: No reconciliation needed
    }
    else if (a_count > 0 && b_count == 0) {
        // Case 2a: Only a in between
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
        // Case 2b: Only b in between
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
        // Case 3: Both a and b in between - swap OUTSIDE occurrences
        int a_left = -1;
        for (int i = 2 * pos - 1; i >= 0; i--) {
            if (tri.level2[i] == a) { a_left = i; break; }
        }
        int b_right = -1;
        for (int i = 2 * (pos + 1); i < 2 * n; i++) {
            if (tri.level2[i] == b) { b_right = i; break; }
        }
        if (a_left >= 0 && b_right >= 0) {
            tri.level2[a_left] = b;
            tri.level2[b_right] = a;
        }
    }

    if (check_interlacing(tri)) {
        return true;
    }

    tri.level1 = old_level1;
    tri.level2 = old_level2;
    return false;
}

// ============================================================================
// SAMPLE COLLECTION
// ============================================================================

struct SampleStats {
    int n;
    map<int, long long> psi_histogram;
    vector<vector<long long>> level1_freq;  // level1_freq[color-1][pos]
    vector<vector<long long>> level2_freq;  // level2_freq[color-1][pos]
    long long total_samples;

    SampleStats(int n) : n(n), total_samples(0) {
        level1_freq.resize(n, vector<long long>(n, 0));
        level2_freq.resize(n, vector<long long>(2 * n, 0));
    }

    void add_sample(const Triangle& tri, int psi) {
        total_samples++;
        psi_histogram[psi]++;

        for (int i = 0; i < n; i++) {
            level1_freq[tri.level1[i] - 1][i]++;
        }
        for (int i = 0; i < 2 * n; i++) {
            level2_freq[tri.level2[i] - 1][i]++;
        }
    }

    void print(ostream& out = cout) const {
        out << "=== Sample Statistics ===" << endl;
        out << "Total samples: " << total_samples << endl;

        out << "\nPsi histogram:" << endl;
        for (auto& [psi, count] : psi_histogram) {
            double freq = (double)count / total_samples;
            out << "  psi=" << psi << ": " << count << " (" << fixed << setprecision(4) << freq << ")" << endl;
        }

        out << "\nLevel 1 position frequencies (color -> positions):" << endl;
        for (int c = 0; c < n; c++) {
            out << "  Color " << (c + 1) << ": ";
            for (int p = 0; p < n; p++) {
                double freq = (double)level1_freq[c][p] / total_samples;
                out << fixed << setprecision(3) << freq << " ";
            }
            out << endl;
        }

        out << "\nLevel 2 position frequencies (color -> positions):" << endl;
        for (int c = 0; c < n; c++) {
            out << "  Color " << (c + 1) << ": ";
            for (int p = 0; p < 2 * n; p++) {
                double freq = (double)level2_freq[c][p] / total_samples;
                out << fixed << setprecision(3) << freq << " ";
            }
            out << endl;
        }
    }

    void write_csv(const string& prefix) const {
        // Psi histogram
        ofstream psi_file(prefix + "_psi.csv");
        psi_file << "psi,count,frequency" << endl;
        for (auto& [psi, count] : psi_histogram) {
            psi_file << psi << "," << count << "," << (double)count / total_samples << endl;
        }

        // Level 1 frequencies
        ofstream l1_file(prefix + "_level1.csv");
        l1_file << "color";
        for (int p = 0; p < n; p++) l1_file << ",pos" << p;
        l1_file << endl;
        for (int c = 0; c < n; c++) {
            l1_file << (c + 1);
            for (int p = 0; p < n; p++) {
                l1_file << "," << (double)level1_freq[c][p] / total_samples;
            }
            l1_file << endl;
        }

        // Level 2 frequencies
        ofstream l2_file(prefix + "_level2.csv");
        l2_file << "color";
        for (int p = 0; p < 2 * n; p++) l2_file << ",pos" << p;
        l2_file << endl;
        for (int c = 0; c < n; c++) {
            l2_file << (c + 1);
            for (int p = 0; p < 2 * n; p++) {
                l2_file << "," << (double)level2_freq[c][p] / total_samples;
            }
            l2_file << endl;
        }
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <n> <q> <num_samples> [burn_in] [thin]" << endl;
        cerr << "  n           Number of colors (1-100)" << endl;
        cerr << "  q           Weight parameter (q > 0)" << endl;
        cerr << "  num_samples Number of samples to collect" << endl;
        cerr << "  burn_in     Burn-in steps (default: 10000)" << endl;
        cerr << "  thin        Thinning interval (default: 100)" << endl;
        return 1;
    }

    int n = stoi(argv[1]);
    double q = stod(argv[2]);
    long long num_samples = stoll(argv[3]);
    long long burn_in = (argc >= 5) ? stoll(argv[4]) : 10000;
    int thin = (argc >= 6) ? stoi(argv[5]) : 100;

    if (n < 1 || n > 100) {
        cerr << "Error: n must be between 1 and 100" << endl;
        return 1;
    }
    if (q <= 0) {
        cerr << "Error: q must be positive" << endl;
        return 1;
    }

    long long total_steps = burn_in + num_samples * (long long)thin;

    cerr << "=== Sample Collector ===" << endl;
    cerr << "n = " << n << ", q = " << q << endl;
    cerr << "num_samples = " << num_samples << endl;
    cerr << "burn_in = " << burn_in << ", thin = " << thin << endl;
    cerr << "total_steps = " << total_steps << endl;
    cerr << "========================\n" << endl;

    // Initialize
    mt19937_64 rng(42);
    uniform_real_distribution<double> uniform(0.0, 1.0);

    Triangle current(n);
    current.initialize_identity();
    int current_psi = compute_psi(current);
    double current_weight = pow(q, current_psi);

    const int num_level1_swaps = n - 1;
    const int num_level2_swaps = 2 * n - 1;
    const int total_swaps = num_level1_swaps + num_level2_swaps;

    SampleStats stats(n);
    long long step = 0;
    long long samples_collected = 0;

    // Progress tracking
    long long progress_interval = max(1LL, num_samples / 10);
    long long psi_sum = 0, psi_count = 0;
    int psi_min = current_psi, psi_max = current_psi;

    cerr << "Running MCMC (burn_in=" << burn_in << ", thin=" << thin << ")..." << endl;

    while (samples_collected < num_samples) {
        // MCMC step
        Triangle proposed = current;
        int swap_choice = (int)(uniform(rng) * total_swaps);

        bool valid_swap;
        if (swap_choice < num_level1_swaps) {
            valid_swap = try_level1_swap(proposed, swap_choice);
        } else {
            valid_swap = try_level2_swap(proposed, swap_choice - num_level1_swaps);
        }

        if (valid_swap) {
            int proposed_psi = compute_psi(proposed);
            double proposed_weight = pow(q, proposed_psi);
            double accept_prob = min(1.0, proposed_weight / current_weight);

            if (uniform(rng) < accept_prob) {
                current = proposed;
                current_psi = proposed_psi;
                current_weight = proposed_weight;
            }
        }

        step++;

        // Track psi during burn-in and sampling
        psi_sum += current_psi;
        psi_count++;
        psi_min = min(psi_min, current_psi);
        psi_max = max(psi_max, current_psi);

        // Collect sample after burn-in, with thinning
        if (step > burn_in && (step - burn_in) % thin == 0) {
            stats.add_sample(current, current_psi);
            samples_collected++;

            if (samples_collected % progress_interval == 0) {
                double pct = 100.0 * samples_collected / num_samples;
                double avg_psi = (double)psi_sum / psi_count;
                cerr << "[" << fixed << setprecision(0) << pct << "%] "
                     << samples_collected << "/" << num_samples
                     << " | psi=" << current_psi
                     << " avg=" << setprecision(1) << avg_psi
                     << " [" << psi_min << "," << psi_max << "]" << endl;
                psi_sum = 0; psi_count = 0;
                psi_min = psi_max = current_psi;
            }
        }
    }

    cout << "\n";
    stats.print();

    // Write CSV files
    string prefix = "samples_n" + to_string(n) + "_q" + to_string(q);
    stats.write_csv(prefix);
    cout << "\nWrote CSV files with prefix: " << prefix << endl;

    return 0;
}
