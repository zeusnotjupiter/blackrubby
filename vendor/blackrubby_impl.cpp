#pragma once

// BlackRubby: Black Regression (random-Fourier-feature ridge) fused with Rubby (Rubik's-cube-walk symbolic search).
//
// Stage 1 (Rubby): a short cube-walk GA on a subsample finds closed-form candidate expressions.
// Stage 2 (Black): ridge over [cos/sin Fourier features | standardized inputs | standardized Rubby expressions].
// Bandwidth, alpha and "use Rubby columns?" are picked by exact leave-one-out error, so the fit can only
// lean on the symbolic columns when they help. Pure C++17 + Eigen, OpenMP optional.
//
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -DNDEBUG -I eigen-3.4.0 blackrubby.cpp -o blackrubby.exe

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BLACKRUBBY_NO_MAIN
#include "black_regression/black_regression.hpp"
#endif

namespace blackrubby {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Clock = std::chrono::steady_clock;

struct Options {
    int expansion_dim = 1024;      // number of cos/sin pairs
    double max_frequency = 54.5981500331; // largest 1D Fourier frequency in standardized-input units
    double alpha = 0.0;            // ridge strength for the full data set; 0 = auto (CV)
    double bandwidth = 0.0;        // Fourier bandwidth; 0 = auto (CV)
    std::uint64_t seed = 42;
    bool use_symbolic = true;
    int sym_population = 128;
    int sym_generations = 100;
    int sym_turns = 8;             // cube-walk length (max 32)
    int sym_rows = 1024;           // rows used by the symbolic search
    int sym_top_k = 6;             // symbolic columns offered to the ridge
    int sym_mcts_sims = 24;
    double sym_time_limit = 5.0;   // seconds
    int select_rows = 2048;        // rows used for hyper-parameter selection
    int select_dim = 128;          // cos/sin pairs used only while selecting (strided subset of the full set)
    int solve_rows = -1;           // rows in the final solve: -1 = auto cap (>= 50000 and 20x features), 0 = all
    int batch_size = 2048;
};

namespace detail {

enum class Op : std::uint8_t { Const, Var, Add, Sub, Mul, Div, Log, Exp, Sqrt, Abs, Pow2, Pow3 };

constexpr int kSize = 3;
constexpr int kCells = kSize * kSize * kSize;
constexpr int kMaxTurns = 32;

struct Node {
    Op op = Op::Const;
    double c = 0.0;
    int var = 0;
};

struct Genome {
    std::array<Node, kCells> cube{};
    std::vector<std::uint8_t> path;
    double a = 1.0, b = 0.0;
    double nmse = 1e12;
};

struct Search {
    MatrixXd X;
    VectorXd y;
    double mean_y = 0.0, var_y = 1.0;
};

using Rng = std::mt19937_64;

inline std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
    std::uint64_t z = a + 0x9E3779B97F4A7C15ULL * (b + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline double unit(Rng& r) { return std::uniform_real_distribution<double>(0.0, 1.0)(r); }

// Wrapping move on the 3x3x3 cube; 0/1 = +/-x, 2/3 = +/-y, 4/5 = +/-z.
inline int step(int cell, int dir) {
    int x = cell / 9, y = (cell / 3) % 3, z = cell % 3;
    switch (dir) {
        case 0: x = (x + 1) % 3; break;
        case 1: x = (x + 2) % 3; break;
        case 2: y = (y + 1) % 3; break;
        case 3: y = (y + 2) % 3; break;
        case 4: z = (z + 1) % 3; break;
        default: z = (z + 2) % 3; break;
    }
    return x * 9 + y * 3 + z;
}

inline int active_cells(const Genome& g, int* cells) {
    int cell = 0;
    cells[0] = 0;
    const int turns = static_cast<int>(g.path.size());
    for (int i = 0; i < turns; ++i) {
        cell = step(cell, g.path[i]);
        cells[i + 1] = cell;
    }
    return turns + 1;
}

inline double safe_denom(double v) { return std::abs(v) >= 1e-4 ? v : (v >= 0.0 ? 1e-4 : -1e-4); }

// Walks the genome's path through the cube, evaluating the stack program column-wise over all rows.
inline void run_genome(const Genome& g, const MatrixXd& X, MatrixXd& st, VectorXd& out) {
    int cells[kMaxTurns + 1];
    const int m = active_cells(g, cells);
    const Index n = X.rows();
    int top = -1;
    for (int i = 0; i < m; ++i) {
        const Node& nd = g.cube[cells[i]];
        switch (nd.op) {
            case Op::Const: st.col(++top).setConstant(nd.c); break;
            case Op::Var: st.col(++top) = X.col(nd.var); break;
            case Op::Add:
                if (top < 1) break;
                st.col(top - 1).array() = st.col(top - 1).array() + st.col(top).array();
                --top;
                break;
            case Op::Sub:
                if (top < 1) break;
                st.col(top - 1).array() = st.col(top - 1).array() - st.col(top).array();
                --top;
                break;
            case Op::Mul:
                if (top < 1) break;
                st.col(top - 1).array() = st.col(top - 1).array() * st.col(top).array();
                --top;
                break;
            case Op::Div:
                if (top < 1) break;
                st.col(top - 1).array() = st.col(top - 1).array() / st.col(top).array().unaryExpr(&safe_denom);
                --top;
                break;
            case Op::Log:
                if (top < 0) break;
                st.col(top).array() = (st.col(top).array().abs() + 1e-4).log();
                break;
            case Op::Exp:
                if (top < 0) break;
                st.col(top).array() = st.col(top).array().max(-20.0).min(20.0).exp();
                break;
            case Op::Sqrt:
                if (top < 0) break;
                st.col(top).array() = st.col(top).array().abs().sqrt();
                break;
            case Op::Abs:
                if (top < 0) break;
                st.col(top).array() = st.col(top).array().abs();
                break;
            case Op::Pow2:
                if (top < 0) break;
                st.col(top).array() = st.col(top).array().square();
                break;
            case Op::Pow3:
                if (top < 0) break;
                st.col(top).array() = st.col(top).array().cube();
                break;
        }
    }
    if (top < 0) {
        out.setZero(n);
    } else {
        out.resize(n);
        out.array() = st.col(top).array().isFinite().select(st.col(top).array(), 0.0).min(1e5).max(-1e5);
    }
}

// Least-squares scale/offset so the walk only has to find the right *shape*.
inline void score(Genome& g, const Search& s, MatrixXd& st, VectorXd& pred) {
    run_genome(g, s.X, st, pred);
    const double mp = pred.mean();
    const auto pc = pred.array() - mp;
    const double vp = pc.square().sum();
    double a = 0.0, b = s.mean_y;
    if (vp > 1e-12) {
        a = (pc * (s.y.array() - s.mean_y)).sum() / vp;
        b = s.mean_y - a * mp;
    }
    const double mse = (a * pred.array() + b - s.y.array()).square().mean();
    g.a = a;
    g.b = b;
    g.nmse = std::isfinite(mse) ? mse / s.var_y : 1e12;
}

inline Node random_node(Rng& r, int nfeat) {
    Node n;
    const double p = unit(r);
    if (p < 0.25) {
        n.op = Op::Const;
        n.c = std::round(std::uniform_real_distribution<double>(-5.0, 5.0)(r) * 100.0) / 100.0;
    } else if (p < 0.55) {
        n.op = Op::Var;
        n.var = static_cast<int>(r() % static_cast<std::uint64_t>(nfeat));
    } else {
        n.op = static_cast<Op>(2 + static_cast<int>(r() % 10));
    }
    return n;
}

inline Genome random_genome(Rng& r, int nfeat, int turns) {
    Genome g;
    for (auto& n : g.cube) n = random_node(r, nfeat);
    g.path.resize(turns);
    for (auto& d : g.path) d = static_cast<std::uint8_t>(r() % 6);
    return g;
}

inline void mutate_active(Genome& g, Rng& r, int nfeat) {
    int cells[kMaxTurns + 1];
    const int m = active_cells(g, cells);
    Node& n = g.cube[cells[r() % static_cast<std::uint64_t>(m)]];
    if (n.op == Op::Const && unit(r) < 0.5) {
        n.c += std::normal_distribution<double>(0.0, 0.2 * std::max(1.0, std::abs(n.c)))(r);
    } else {
        n = random_node(r, nfeat);
    }
}

inline void mutate_full(Genome& g, Rng& r, int nfeat) {
    const int count = std::max(1, static_cast<int>(kCells * 0.2));
    for (int i = 0; i < count; ++i) g.cube[r() % kCells] = random_node(r, nfeat);
    const int path_count = std::max(1, static_cast<int>(g.path.size() * 0.2));
    for (int i = 0; i < path_count; ++i) g.path[r() % g.path.size()] = static_cast<std::uint8_t>(r() % 6);
}

inline Genome crossover(const Genome& p1, const Genome& p2, Rng& r) {
    Genome c;
    for (int i = 0; i < kCells; ++i) c.cube[i] = (unit(r) < 0.5) ? p1.cube[i] : p2.cube[i];
    const int turns = static_cast<int>(p1.path.size());
    const int split = 1 + static_cast<int>(r() % static_cast<std::uint64_t>(std::max(1, turns - 1)));
    c.path.resize(turns);
    for (int i = 0; i < turns; ++i) c.path[i] = (i < split) ? p1.path[i] : p2.path[i];
    return c;
}

// Tree distance between paths: 2 * (moves after the shared prefix).
inline int path_distance(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    size_t shared = 0;
    while (shared < a.size() && a[shared] == b[shared]) ++shared;
    return 2 * static_cast<int>(a.size() - shared);
}

inline std::pair<int, int> select_parents(const std::vector<Genome>& pop, Rng& r) {
    auto tournament = [&]() {
        int best = static_cast<int>(r() % pop.size());
        for (int i = 1; i < 5; ++i) {
            const int c = static_cast<int>(r() % pop.size());
            if (pop[c].nmse < pop[best].nmse) best = c;
        }
        return best;
    };
    const int p1 = tournament();
    int p2 = tournament();
    for (int attempt = 1; attempt < 5; ++attempt) {
        if (path_distance(pop[p1].path, pop[p2].path) >= 4) break;
        p2 = tournament();
    }
    return {p1, p2};
}

struct MctsNode {
    int visits = 0;
    double total = 0.0, best = 0.0;
    int child[6] = {-1, -1, -1, -1, -1, -1};
};

// UCT search over the path alone (cube contents fixed); keeps the best path found.
inline void mcts_refine(Genome& g, const Search& s, int sims, Rng& r, MatrixXd& st, VectorXd& pred) {
    const int turns = static_cast<int>(g.path.size());
    std::vector<MctsNode> pool;
    pool.reserve(static_cast<size_t>(sims) * (turns + 1) + 1);
    std::vector<int> ids(static_cast<size_t>(turns + 1) * kCells, -1);
    std::vector<int> visited;
    auto get = [&](int depth, int cell) {
        int& id = ids[static_cast<size_t>(depth) * kCells + cell];
        if (id == -1) {
            pool.emplace_back();
            id = static_cast<int>(pool.size()) - 1;
        }
        return id;
    };
    Genome cand = g;
    Genome best = g;
    for (int sim = 0; sim < sims; ++sim) {
        int cell = 0;
        visited.clear();
        visited.push_back(get(0, cell));
        bool in_tree = true;
        for (int depth = 0; depth < turns; ++depth) {
            int dir;
            if (in_tree) {
                const MctsNode parent = pool[visited.back()];
                int expand = -1;
                for (int d = 0; d < 6; ++d) {
                    if (parent.child[d] == -1) { expand = d; break; }
                }
                if (expand != -1) {
                    dir = expand;
                    in_tree = false;
                } else {
                    double best_ucb = -1e18;
                    dir = 0;
                    for (int d = 0; d < 6; ++d) {
                        const MctsNode& ch = pool[parent.child[d]];
                        const double mean = ch.visits ? ch.total / ch.visits : 0.0;
                        const double exploit = 0.9 * mean + 0.1 * ch.best;
                        const double prior = ch.best > 0.1 ? 2.0 : 1.0;
                        const double ucb = exploit + 1.4 * prior *
                            std::sqrt(std::log(static_cast<double>(parent.visits + 1)) / (ch.visits + 1e-9));
                        if (ucb > best_ucb) { best_ucb = ucb; dir = d; }
                    }
                }
                const int next = step(cell, dir);
                const int child = get(depth + 1, next);
                pool[visited.back()].child[dir] = child;
                visited.push_back(child);
                cell = next;
            } else {
                dir = static_cast<int>(r() % 6);
                cell = step(cell, dir);
            }
            cand.path[depth] = static_cast<std::uint8_t>(dir);
        }
        score(cand, s, st, pred);
        const double reward = 1.0 / (1.0 + cand.nmse);
        if (cand.nmse < best.nmse) best = cand;
        for (int id : visited) {
            pool[id].visits += 1;
            pool[id].total += reward;
            pool[id].best = std::max(pool[id].best, reward);
        }
    }
    if (best.nmse < g.nmse) g = best;
}

// Levenberg-Marquardt over the constants on the active path (variable projection for scale/offset).
inline void polish_lm(Genome& g, const Search& s, MatrixXd& st, VectorXd& pred, int steps = 25) {
    int cells[kMaxTurns + 1];
    const int m = active_cells(g, cells);
    std::vector<int> idx;
    for (int i = 0; i < m; ++i) {
        if (g.cube[cells[i]].op == Op::Const &&
            std::find(idx.begin(), idx.end(), cells[i]) == idx.end()) {
            idx.push_back(cells[i]);
        }
    }
    const int K = static_cast<int>(idx.size());
    if (K == 0) return;
    const Index n = s.X.rows();
    VectorXd r(n), rh(n);
    MatrixXd J(n, K);
    auto residual = [&](VectorXd& res) {
        score(g, s, st, pred);
        res = s.y.array() - (g.a * pred.array() + g.b);
    };
    double lambda = 1e-2;
    residual(r);
    double current = g.nmse;
    for (int it = 0; it < steps; ++it) {
        std::vector<double> saved(K);
        for (int k = 0; k < K; ++k) saved[k] = g.cube[idx[k]].c;
        for (int k = 0; k < K; ++k) {
            const double h = 1e-6 * std::max(1.0, std::abs(saved[k]));
            g.cube[idx[k]].c = saved[k] + h;
            residual(rh);
            J.col(k) = (rh - r) / h;
            g.cube[idx[k]].c = saved[k];
        }
        const MatrixXd H = J.transpose() * J;
        const VectorXd grad = J.transpose() * r;
        bool accepted = false;
        for (int tries = 0; tries < 6 && !accepted; ++tries) {
            MatrixXd Hd = H;
            Hd.diagonal().array() += lambda * (H.diagonal().array() + 1e-9);
            const VectorXd delta = Hd.ldlt().solve(-grad);
            if (delta.allFinite()) {
                for (int k = 0; k < K; ++k) g.cube[idx[k]].c = saved[k] + delta[k];
                residual(rh);
                if (g.nmse < current) {
                    current = g.nmse;
                    r = rh;
                    lambda = std::max(1e-9, lambda / 3.0);
                    accepted = true;
                    continue;
                }
            }
            for (int k = 0; k < K; ++k) g.cube[idx[k]].c = saved[k];
            lambda = std::min(1e8, lambda * 4.0);
        }
        if (!accepted) {
            residual(r);
            break;
        }
    }
    score(g, s, st, pred);
}

inline void add_hall_of_fame(std::vector<Genome>& hof, const Genome& c, size_t top_k) {
    for (const auto& h : hof) {
        if (std::abs(h.nmse - c.nmse) <= 1e-7 * h.nmse + 1e-15) return;
    }
    hof.push_back(c);
    std::sort(hof.begin(), hof.end(), [](const Genome& a, const Genome& b) { return a.nmse < b.nmse; });
    if (hof.size() > top_k) hof.resize(top_k);
}

inline std::vector<Genome> symbolic_search(const Search& s, const Options& o) {
    const int nfeat = static_cast<int>(s.X.cols());
    const int turns = std::clamp(o.sym_turns, 2, kMaxTurns);
    const int P = std::max(8, o.sym_population);
    const Index n = s.X.rows();
    const int nthreads = max_threads();
    std::vector<MatrixXd> st(nthreads, MatrixXd(n, turns + 1));
    std::vector<VectorXd> pr(nthreads, VectorXd(n));
    const auto t0 = Clock::now();

    std::vector<Genome> pop(P);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < P; ++i) {
        Rng r(mix(o.seed, static_cast<std::uint64_t>(i)));
        pop[i] = random_genome(r, nfeat, turns);
        score(pop[i], s, st[thread_id()], pr[thread_id()]);
    }

    auto by_nmse = [](const Genome& a, const Genome& b) { return a.nmse < b.nmse; };
    std::vector<Genome> hof;
    const size_t top_k = static_cast<size_t>(std::max(1, o.sym_top_k));
    double last_best = std::numeric_limits<double>::infinity();
    int stagnant = 0;
    const int elites = std::max(2, P / 20);

    for (int gen = 0; gen < o.sym_generations; ++gen) {
        std::sort(pop.begin(), pop.end(), by_nmse);
        if (gen % 10 == 9) {
            Rng r(mix(o.seed ^ 0xABCDULL, static_cast<std::uint64_t>(gen)));
            for (int i = 0; i < 2; ++i) {
                mcts_refine(pop[i], s, o.sym_mcts_sims * 4, r, st[0], pr[0]);
                polish_lm(pop[i], s, st[0], pr[0]);
            }
            std::sort(pop.begin(), pop.end(), by_nmse);
        }
        for (int i = 0; i < std::min(4, P); ++i) add_hall_of_fame(hof, pop[i], top_k);
        if (pop[0].nmse < 1e-12) break;
        if (o.sym_time_limit > 0.0 && std::chrono::duration<double>(Clock::now() - t0).count() > o.sym_time_limit) break;

        if (pop[0].nmse < last_best * (1.0 - 1e-6)) {
            last_best = pop[0].nmse;
            stagnant = 0;
        } else {
            ++stagnant;
        }
        const bool inject = stagnant >= 8;
        if (inject) stagnant = 0;

        std::vector<Genome> next(P);
        for (int i = 0; i < elites; ++i) next[i] = pop[i];
        #pragma omp parallel for schedule(static)
        for (int i = elites; i < P; ++i) {
            Rng r(mix(o.seed, (static_cast<std::uint64_t>(gen) + 1) * 1000003ULL + i));
            Genome child;
            if (inject && i < elites + P / 2) {
                child = random_genome(r, nfeat, turns);
            } else {
                const auto parents = select_parents(pop, r);
                child = crossover(pop[parents.first], pop[parents.second], r);
                if (unit(r) < 0.5) mutate_active(child, r, nfeat); else mutate_full(child, r, nfeat);
                score(child, s, st[thread_id()], pr[thread_id()]);
                if (o.sym_mcts_sims > 0 && unit(r) < 0.15) {
                    mcts_refine(child, s, o.sym_mcts_sims, r, st[thread_id()], pr[thread_id()]);
                }
            }
            if (child.nmse >= 1e12 || inject) score(child, s, st[thread_id()], pr[thread_id()]);
            next[i] = std::move(child);
        }
        pop = std::move(next);
    }

    std::sort(pop.begin(), pop.end(), by_nmse);
    for (int i = 0; i < std::min(4, P); ++i) add_hall_of_fame(hof, pop[i], top_k);
    const size_t polish_count = std::min<size_t>(3, hof.size());
    for (size_t i = 0; i < polish_count; ++i) {
        Genome polished = hof[i];
        polish_lm(polished, s, st[0], pr[0], 40);
        add_hall_of_fame(hof, polished, top_k);
    }
    return hof;
}

} // namespace detail

// Greedy optimal-design row selection on random Fourier features. Selection uses X only, so only the
// chosen rows ever need a label. Mode 0 (D-optimal) maximises log-det information gain and favours extreme rows;
// mode 1 (V-optimal) minimises the average predictive variance over a reference set, which favours typical rows,
// and mode 2 alternates the two so the picks cover both the edges and the interior of the data.
struct InfoSelection {
    std::vector<Index> rows;
    std::vector<double> gains;     // information gained by each pick, in nats
    std::vector<double> avg_var;   // mean predictive variance over the reference set after each pick
};

inline InfoSelection greedy_info_rows(const MatrixXd& X, Index m, int mode = 1, double bw = 1.0,
                                      double lambda = 1e-2, int dim = 128, Index ref_rows = 256,
                                      std::uint64_t seed = 42) {
    const Index n = X.rows(), d = X.cols();
    m = std::min(m, n);
    Eigen::RowVectorXd mu = X.colwise().mean();
    Eigen::RowVectorXd sd(d);
    for (Index c = 0; c < d; ++c) {
        const double var = (X.col(c).array() - mu[c]).square().mean();
        sd[c] = var > 1e-24 ? std::sqrt(var) : 1.0;
    }
    MatrixXd xs = X;
    xs.rowwise() -= mu;
    xs.array().rowwise() /= sd.array();
    MatrixXd W(dim, d);
    if (d == 1) {
        for (int j = 0; j < dim; ++j) W(j, 0) = std::exp(4.0 * j / (dim - 1));
    } else {
        detail::Rng r(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int j = 0; j < dim; ++j)
            for (Index c = 0; c < d; ++c) W(j, c) = normal(r);
    }
    const MatrixXd ang = (xs * W.transpose()) / bw;
    const double s = 1.0 / std::sqrt(static_cast<double>(dim));
    MatrixXd F(n, 2 * dim + d);
    F.leftCols(dim) = ang.array().cos() * s;
    F.middleCols(dim, dim) = ang.array().sin() * s;
    F.rightCols(d) = xs;

    MatrixXd Z = F / lambda;  // Z = F A^-1 with A = lambda I
    VectorXd var = (F.array() * Z.array()).rowwise().sum();
    const Index R = std::min<Index>(ref_rows, n);
    MatrixXd C(R, n);  // C = F_R A^-1 F^T, the covariance of reference predictions with every candidate
    double total_var = 0.0;
    {
        MatrixXd FR(R, F.cols());
        for (Index r = 0; r < R; ++r) FR.row(r) = F.row(r * n / R);
        C.noalias() = FR * F.transpose() / lambda;
        total_var = FR.squaredNorm() / lambda;
    }
    InfoSelection out;
    VectorXd score(n);
    for (Index k = 0; k < m; ++k) {
        const int step_mode = mode == 2 ? static_cast<int>(k % 2) : mode;  // 2 = alternate boundary / interior
        if (step_mode == 0) {
            score = var;
        } else {
            score = C.array().square().colwise().sum().transpose() / (1.0 + var.array().max(0.0));
            for (Index i = 0; i < n; ++i) if (!std::isfinite(var[i])) score[i] = -std::numeric_limits<double>::infinity();
        }
        Index pick = 0;
        score.maxCoeff(&pick);
        if (!(var[pick] > 0.0)) break;
        out.rows.push_back(pick);
        out.gains.push_back(std::log1p(var[pick]));
        const VectorXd c = Z * F.row(pick).transpose();
        const VectorXd a = Z.row(pick).transpose();
        const double denom = 1.0 + var[pick];
        const VectorXd cr = C.col(pick);
        total_var -= cr.squaredNorm() / denom;
        C.noalias() -= (cr / denom) * c.transpose();
        Z.noalias() -= (c / denom) * a.transpose();
        var.array() -= c.array().square() / denom;
        var[pick] = -std::numeric_limits<double>::infinity();
        out.avg_var.push_back(total_var / static_cast<double>(R));
    }
    return out;
}

class BlackRubby {
public:
    explicit BlackRubby(const Options& options = Options{}) : opts_(options) {
        if (opts_.expansion_dim < 2) throw std::invalid_argument("expansion_dim must be >= 2.");
        if (opts_.alpha < 0.0 || opts_.bandwidth < 0.0) throw std::invalid_argument("alpha/bandwidth must be >= 0.");
        if (!(opts_.max_frequency >= 1.0) || !std::isfinite(opts_.max_frequency)) {
            throw std::invalid_argument("max_frequency must be finite and at least 1.");
        }
        if (opts_.batch_size < 1) throw std::invalid_argument("batch_size must be positive.");
    }

    void fit(const Eigen::Ref<const MatrixXd>& x, const Eigen::Ref<const VectorXd>& y) {
        using namespace detail;
        fitted_ = false;
        has_best_ = false;
        champions_.clear();
        if (x.rows() == 0 || x.cols() == 0 || x.rows() != y.size()) {
            throw std::invalid_argument("x and y must have the same nonzero length.");
        }
        if (!x.allFinite() || !y.allFinite()) throw std::invalid_argument("x and y must be finite.");
        const Index N = x.rows(), d = x.cols(), D = opts_.expansion_dim;

        mean_ = x.colwise().mean();
        scale_.resize(d);
        for (Index c = 0; c < d; ++c) {
            const double var = (x.col(c).array() - mean_[c]).square().mean();
            scale_[c] = var > 1e-24 ? std::sqrt(var) : 1.0;
        }
        build_projections(d);

        champions_.clear();
        symbolic_seconds_ = 0.0;
        auto t = Clock::now();
        if (opts_.use_symbolic) {
            Search s;
            const Index ns = std::min<Index>(N, opts_.sym_rows);
            s.X.resize(ns, d);
            s.y.resize(ns);
            for (Index i = 0; i < ns; ++i) {
                const Index src = i * N / ns;
                s.X.row(i) = x.row(src);
                s.y[i] = y[src];
            }
            s.mean_y = s.y.mean();
            s.var_y = std::max((s.y.array() - s.mean_y).square().mean(), 1e-12);
            std::vector<Genome> hof = symbolic_search(s, opts_);
            if (!hof.empty()) best_ = hof.front();
            prepare_symbolic_columns(x, std::move(hof));
        }
        symbolic_seconds_ = seconds_since(t);

        t = Clock::now();
        const Index ns0 = std::min<Index>(N, opts_.select_rows);
        MatrixXd xs0(ns0, d);
        VectorXd ys0(ns0);
        for (Index i = 0; i < ns0; ++i) {
            const Index src = i * N / ns0;
            xs0.row(i) = x.row(src);
            ys0[i] = y[src];
        }
        // Exact duplicate rows leak into cross-validation and favour overfitting, so keep one of each.
        std::vector<Index> order(ns0);
        std::iota(order.begin(), order.end(), Index(0));
        auto row_less = [&](Index a, Index b) {
            for (Index c = 0; c < d; ++c) if (xs0(a, c) != xs0(b, c)) return xs0(a, c) < xs0(b, c);
            return false;
        };
        std::sort(order.begin(), order.end(), row_less);
        std::vector<Index> keep;
        for (Index k = 0; k < ns0; ++k) {
            if (k == 0 || row_less(order[k - 1], order[k])) keep.push_back(order[k]);
        }
        const Index ns = static_cast<Index>(keep.size());
        MatrixXd xs(ns, d);
        VectorXd ys(ns);
        for (Index i = 0; i < ns; ++i) {
            xs.row(i) = xs0.row(keep[i]);
            ys[i] = ys0[keep[i]];
        }
        select_hyperparameters(xs, ys, static_cast<double>(N));
        select_seconds_ = seconds_since(t);

        t = Clock::now();
        solve(x, y, N, D);
        solve_seconds_ = seconds_since(t);
        fitted_ = true;
    }

    VectorXd predict(const Eigen::Ref<const MatrixXd>& x) const {
        require_fitted(x);
        VectorXd out(x.rows());
        const Index bs = opts_.batch_size;
        const Index nb = (x.rows() + bs - 1) / bs;
        #pragma omp parallel for schedule(static)
        for (Index b = 0; b < nb; ++b) {
            const Index begin = b * bs, count = std::min(bs, x.rows() - begin);
            const MatrixXd F = features(x.middleRows(begin, count), bandwidth_, use_sym_, W_);
            out.segment(begin, count) = F * weights_;
            out.segment(begin, count).array() += intercept_;
        }
        return out;
    }

    // Rubby-only prediction (best symbolic champion with fitted scale/offset).
    VectorXd predict_symbolic(const Eigen::Ref<const MatrixXd>& x) const {
        require_fitted(x);
        if (!has_best_) throw std::logic_error("no symbolic champion (use_symbolic=false).");
        MatrixXd st(x.rows(), best_.path.size() + 1);
        VectorXd pred;
        const MatrixXd xm = x;
        detail::run_genome(best_, xm, st, pred);
        return (best_.a * pred.array() + best_.b).matrix();
    }

    bool is_fitted() const noexcept { return fitted_; }
    Index input_dim() const noexcept { return mean_.size(); }
    int expansion_dim() const noexcept { return opts_.expansion_dim; }
    const Options& options() const noexcept { return opts_; }

    double score(const Eigen::Ref<const MatrixXd>& x, const Eigen::Ref<const VectorXd>& y) const {
        if (x.rows() != y.size() || y.size() == 0 || !y.allFinite()) {
            throw std::invalid_argument("score requires one finite target per input row.");
        }
        const VectorXd predictions = predict(x);
        const double total = (y.array() - y.mean()).square().sum();
        if (total <= std::numeric_limits<double>::epsilon()) {
            return (predictions - y).squaredNorm() <= std::numeric_limits<double>::epsilon() ? 1.0 : 0.0;
        }
        return 1.0 - (predictions - y).squaredNorm() / total;
    }

    // Token walk of the best Rubby champion: operators/operands in visit order, plus its scale/offset.
    std::string champion_formula() const {
        if (!has_best_) return "";
        static const char* names[] = {"c", "x", "+", "-", "*", "/", "log", "exp", "sqrt", "abs", "^2", "^3"};
        std::ostringstream out;
        out << best_.a << " * ( ";
        int cell = 0;
        auto emit = [&](int c) {
            const detail::Node& n = best_.cube[c];
            if (n.op == detail::Op::Const) out << '[' << n.c << "] ";
            else if (n.op == detail::Op::Var) out << 'x' << n.var << ' ';
            else out << names[static_cast<int>(n.op)] << ' ';
        };
        emit(cell);
        for (std::uint8_t d : best_.path) {
            cell = detail::step(cell, d);
            emit(cell);
        }
        out << ") + " << best_.b << "   (postfix)";
        return out.str();
    }
    double chosen_bandwidth() const noexcept { return bandwidth_; }
    double chosen_alpha() const noexcept { return alpha_abs_; }
    bool uses_symbolic() const noexcept { return use_sym_; }
    int symbolic_columns() const noexcept { return static_cast<int>(champions_.size()); }
    double champion_nmse() const noexcept { return has_best_ ? best_.nmse : std::numeric_limits<double>::quiet_NaN(); }
    double symbolic_seconds() const noexcept { return symbolic_seconds_; }
    double select_seconds() const noexcept { return select_seconds_; }
    double solve_seconds() const noexcept { return solve_seconds_; }

private:
    Options opts_;
    Eigen::RowVectorXd mean_, scale_;
    MatrixXd W_;                       // D x d projection
    MatrixXd W_sel_;                   // strided subset of W_ used only for hyper-parameter selection
    std::vector<detail::Genome> champions_;
    VectorXd sym_mean_, sym_scale_;
    detail::Genome best_;
    bool has_best_ = false;
    VectorXd weights_;
    double intercept_ = 0.0, bandwidth_ = 1.0, alpha_abs_ = 1.0;
    bool use_sym_ = false, fitted_ = false;
    double symbolic_seconds_ = 0.0, select_seconds_ = 0.0, solve_seconds_ = 0.0;

    static double seconds_since(Clock::time_point t) {
        return std::chrono::duration<double>(Clock::now() - t).count();
    }

    void require_fitted(const Eigen::Ref<const MatrixXd>& x) const {
        if (!fitted_) throw std::logic_error("fit must be called before predict.");
        if (x.cols() != mean_.size() || !x.allFinite()) {
            throw std::invalid_argument("x has the wrong number of columns or non-finite values.");
        }
    }

    void build_projections(Index d) {
        const Index D = opts_.expansion_dim;
        W_.resize(D, d);
        if (d == 1) {
            const double log_max = std::log(std::max(1.0, opts_.max_frequency));
            for (Index j = 0; j < D; ++j) W_(j, 0) = std::exp(log_max * static_cast<double>(j) / (D - 1));
        } else {
            detail::Rng r(opts_.seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            for (Index j = 0; j < D; ++j)
                for (Index c = 0; c < d; ++c) W_(j, c) = normal(r);
        }
        const Index stride = std::max<Index>(1, D / std::max(1, opts_.select_dim));
        const Index count = (D + stride - 1) / stride;
        W_sel_.resize(count, d);
        for (Index j = 0; j < count; ++j) W_sel_.row(j) = W_.row(j * stride);
    }

    MatrixXd symbolic_raw(const Eigen::Ref<const MatrixXd>& xb) const {
        const MatrixXd xm = xb;
        const Index n = xm.rows(), K = static_cast<Index>(champions_.size());
        MatrixXd out(n, K);
        MatrixXd st(n, detail::kMaxTurns + 1);
        VectorXd pred;
        for (Index k = 0; k < K; ++k) {
            detail::run_genome(champions_[k], xm, st, pred);
            out.col(k) = pred;
        }
        return out;
    }

    // Drops constant / near-duplicate champions and records standardization over the full data.
    void prepare_symbolic_columns(const Eigen::Ref<const MatrixXd>& x, std::vector<detail::Genome> cands) {
        champions_.clear();
        has_best_ = !cands.empty();
        if (cands.empty()) return;
        champions_ = cands;
        const Index K0 = static_cast<Index>(cands.size()), N = x.rows();
        VectorXd sum = VectorXd::Zero(K0);
        MatrixXd ss = MatrixXd::Zero(K0, K0);
        const Index bs = opts_.batch_size, nb = (N + bs - 1) / bs;
        #pragma omp parallel
        {
            VectorXd ls = VectorXd::Zero(K0);
            MatrixXd lss = MatrixXd::Zero(K0, K0);
            #pragma omp for schedule(static) nowait
            for (Index b = 0; b < nb; ++b) {
                const Index begin = b * bs, count = std::min(bs, N - begin);
                const MatrixXd S = symbolic_raw(x.middleRows(begin, count));
                ls += S.colwise().sum().transpose();
                lss.noalias() += S.transpose() * S;
            }
            #pragma omp critical
            { sum += ls; ss += lss; }
        }
        const VectorXd mean = sum / static_cast<double>(N);
        MatrixXd cov = ss / static_cast<double>(N) - mean * mean.transpose();
        std::vector<Index> kept;
        for (Index k = 0; k < K0; ++k) {
            const double var = cov(k, k);
            if (!(var > 1e-20) || std::sqrt(var) <= 1e-9 * (1.0 + std::abs(mean[k]))) continue;
            bool duplicate = false;
            for (Index j : kept) {
                if (std::abs(cov(k, j)) / std::sqrt(var * cov(j, j)) > 0.99999) { duplicate = true; break; }
            }
            if (!duplicate) kept.push_back(k);
        }
        std::vector<detail::Genome> filtered;
        sym_mean_.resize(kept.size());
        sym_scale_.resize(kept.size());
        for (size_t i = 0; i < kept.size(); ++i) {
            filtered.push_back(cands[kept[i]]);
            sym_mean_[i] = mean[kept[i]];
            sym_scale_[i] = std::sqrt(cov(kept[i], kept[i]));
        }
        champions_ = std::move(filtered);
    }

    MatrixXd features(const Eigen::Ref<const MatrixXd>& xb, double bw, bool with_sym, const MatrixXd& W) const {
        const Index n = xb.rows(), d = xb.cols(), D = W.rows();
        const Index K = with_sym ? static_cast<Index>(champions_.size()) : 0;
        MatrixXd xs = xb;
        xs.rowwise() -= mean_;
        xs.array().rowwise() /= scale_.array();
        MatrixXd angles = xs * W.transpose();
        angles /= bw;
        const double s = 1.0 / std::sqrt(static_cast<double>(D));
        MatrixXd F(n, 2 * D + d + K);
        F.leftCols(D) = angles.array().cos() * s;
        F.middleCols(D, D) = angles.array().sin() * s;
        F.middleCols(2 * D, d) = xs;
        if (K > 0) {
            MatrixXd S = symbolic_raw(xb);
            S.rowwise() -= sym_mean_.transpose();
            S.array().rowwise() /= sym_scale_.transpose().array();
            F.rightCols(K) = S;
        }
        return F;
    }

    // Exact leave-one-out scores for every alpha via one eigendecomposition of the centered Gram matrix.
    static std::vector<double> loo_scores(const MatrixXd& Phi, const VectorXd& y, const std::vector<double>& alphas) {
        const Index n = Phi.rows(), P = Phi.cols();
        const Eigen::RowVectorXd mu = Phi.colwise().mean();
        const MatrixXd Fc = Phi.rowwise() - mu;
        const VectorXd yc = y.array() - y.mean();
        std::vector<double> out;
        if (n < P) {
            // Dual form: eigendecompose the n x n kernel instead of the P x P covariance.
            MatrixXd K = MatrixXd::Zero(n, n);
            K.selfadjointView<Eigen::Lower>().rankUpdate(Fc);
            Eigen::SelfAdjointEigenSolver<MatrixXd> es(K);
            const MatrixXd& U = es.eigenvectors();
            const VectorXd uty = U.transpose() * yc;
            const MatrixXd U2 = U.array().square();
            for (double a : alphas) {
                const VectorXd shrink = es.eigenvalues().array().max(0.0) /
                                        (es.eigenvalues().array().max(0.0) + a * static_cast<double>(n));
                const VectorXd pred = U * uty.cwiseProduct(shrink);
                const VectorXd h = U2 * shrink;
                const VectorXd e = (yc - pred).array() / (1.0 - h.array()).max(1e-8);
                out.push_back(e.squaredNorm() / static_cast<double>(n));
            }
            return out;
        }
        MatrixXd G = MatrixXd::Zero(P, P);
        G.selfadjointView<Eigen::Lower>().rankUpdate(Fc.transpose());
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(G);
        const MatrixXd M = Fc * es.eigenvectors();
        const VectorXd bty = M.transpose() * yc;
        const MatrixXd M2 = M.array().square();
        for (double a : alphas) {
            const VectorXd dinv = (es.eigenvalues().array().max(0.0) + a * static_cast<double>(n)).inverse();
            const VectorXd pred = M * bty.cwiseProduct(dinv);
            const VectorXd h = M2 * dinv;
            const VectorXd e = (yc - pred).array() / (1.0 - h.array()).max(1e-8);
            out.push_back(e.squaredNorm() / static_cast<double>(n));
        }
        return out;
    }

    void select_hyperparameters(const MatrixXd& xs, const VectorXd& ys, double N_total) {
        std::vector<double> bws = opts_.bandwidth > 0.0 ? std::vector<double>{opts_.bandwidth}
                                                        : std::vector<double>{0.5, 1.0, 2.0, 4.0};
        std::vector<double> alphas;  // ascending
        if (opts_.alpha > 0.0) {
            alphas.push_back(opts_.alpha / N_total);
        } else {
            for (int e = -9; e <= 1; ++e) alphas.push_back(std::pow(10.0, e));
        }
        const Index K = static_cast<Index>(champions_.size());

        // Per config: pick the largest alpha within 2% of the best score, which avoids the grid-edge overfit.
        struct Pick { double score; double bw; double alpha; bool sym; };
        auto evaluate = [&](double bw, bool with_sym) {
            const MatrixXd Phi = features(xs, bw, with_sym, W_sel_);
            const std::vector<double> scores = loo_scores(Phi, ys, alphas);
            const double best = *std::min_element(scores.begin(), scores.end());
            size_t pick = 0;
            for (size_t i = 0; i < scores.size(); ++i) if (scores[i] <= best * 1.02) pick = i;
            return Pick{scores[pick], bw, alphas[pick], with_sym};
        };

        // Stage 1: bandwidth, with the symbolic columns if there are any. Stage 2: compare against no symbolic columns.
        Pick best{std::numeric_limits<double>::infinity(), bws[0], alphas[0], false};
        for (double bw : bws) {
            const Pick p = evaluate(bw, K > 0);
            if (p.score < best.score) best = p;
        }
        if (K > 0) {
            const Pick p = evaluate(best.bw, false);
            if (p.score < best.score) best = p;
        }
        bandwidth_ = best.bw;
        alpha_abs_ = best.alpha * N_total;
        use_sym_ = best.sym;
        // The 128-term screen can mis-rank the symbolic columns, so re-decide at full width on a 2-fold holdout.
        if (K > 0) {
            const double with = holdout_mse(xs, ys, best.bw, best.alpha, true);
            const double without = holdout_mse(xs, ys, best.bw, best.alpha, false);
            use_sym_ = with < without;
        }
    }

    // 2-fold holdout MSE of a dual-form ridge at the full feature width.
    double holdout_mse(const MatrixXd& xs, const VectorXd& ys, double bw, double alpha_rel, bool with_sym) const {
        const Index n = xs.rows();
        if (n < 8) return std::numeric_limits<double>::infinity();
        const MatrixXd F = features(xs, bw, with_sym, W_);
        double total = 0.0;
        for (int fold = 0; fold < 2; ++fold) {
            std::vector<Index> tr, te;
            for (Index i = 0; i < n; ++i) ((i % 2) == fold ? te : tr).push_back(i);
            MatrixXd Ft(tr.size(), F.cols()), Fv(te.size(), F.cols());
            VectorXd yt(tr.size()), yv(te.size());
            for (size_t i = 0; i < tr.size(); ++i) { Ft.row(i) = F.row(tr[i]); yt[i] = ys[tr[i]]; }
            for (size_t i = 0; i < te.size(); ++i) { Fv.row(i) = F.row(te[i]); yv[i] = ys[te[i]]; }
            const Eigen::RowVectorXd mu = Ft.colwise().mean();
            Ft.rowwise() -= mu;
            Fv.rowwise() -= mu;
            const double ym = yt.mean();
            MatrixXd K = Ft * Ft.transpose();
            K.diagonal().array() += alpha_rel * static_cast<double>(tr.size());
            const VectorXd c = K.ldlt().solve(VectorXd(yt.array() - ym));
            VectorXd pred = Fv * (Ft.transpose() * c);
            pred.array() += ym;
            total += (pred - yv).squaredNorm();
        }
        return total / static_cast<double>(n);
    }

    void solve(const Eigen::Ref<const MatrixXd>& x, const Eigen::Ref<const VectorXd>& y, Index N, Index D) {
        const Index d = x.cols();
        const Index P = 2 * D + d + (use_sym_ ? static_cast<Index>(champions_.size()) : 0);
        Index M = N;
        if (opts_.solve_rows > 0) {
            M = std::min<Index>(N, opts_.solve_rows);
        } else if (opts_.solve_rows < 0) {
            M = std::min<Index>(N, std::max<Index>(50000, 20 * P));
        }
        const bool strided = M < N;
        MatrixXd G = MatrixXd::Zero(P, P);
        VectorXd fs = VectorXd::Zero(P), fy = VectorXd::Zero(P);
        double ysum = 0.0;
        const Index bs = opts_.batch_size, nb = (M + bs - 1) / bs;
        #pragma omp parallel
        {
            MatrixXd Gl = MatrixXd::Zero(P, P);
            VectorXd fsl = VectorXd::Zero(P), fyl = VectorXd::Zero(P);
            double ysl = 0.0;
            #pragma omp for schedule(dynamic) nowait
            for (Index b = 0; b < nb; ++b) {
                const Index begin = b * bs, count = std::min(bs, M - begin);
                MatrixXd F;
                VectorXd yb(count);
                if (strided) {
                    MatrixXd xb(count, d);
                    for (Index i = 0; i < count; ++i) {
                        const Index src = (begin + i) * N / M;
                        xb.row(i) = x.row(src);
                        yb[i] = y[src];
                    }
                    F = features(xb, bandwidth_, use_sym_, W_);
                } else {
                    F = features(x.middleRows(begin, count), bandwidth_, use_sym_, W_);
                    yb = y.segment(begin, count);
                }
                fsl += F.colwise().sum().transpose();
                fyl.noalias() += F.transpose() * yb;
                ysl += yb.sum();
                Gl.selfadjointView<Eigen::Lower>().rankUpdate(F.transpose());
            }
            #pragma omp critical
            { G += Gl; fs += fsl; fy += fyl; ysum += ysl; }
        }
        const double inv_n = 1.0 / static_cast<double>(M);
        const double ymean = ysum * inv_n;
        G.noalias() -= fs * fs.transpose() * inv_n;
        G.diagonal().array() += alpha_abs_ * static_cast<double>(M) / static_cast<double>(N);
        const VectorXd rhs = fy - fs * ymean;
        Eigen::LLT<MatrixXd> llt(G.selfadjointView<Eigen::Lower>());
        if (llt.info() == Eigen::Success) {
            weights_ = llt.solve(rhs);
        } else {
            Eigen::LDLT<MatrixXd> ldlt(G.selfadjointView<Eigen::Lower>());
            weights_ = ldlt.solve(rhs);
        }
        if (!weights_.allFinite()) throw std::runtime_error("Ridge solve failed.");
        intercept_ = ymean - (fs * inv_n).dot(weights_);
    }
};

// Active fit: picks rows by optimal design on X alone, asks for labels only for those rows, and doubles the
// budget until the model's predictions stop changing. No y is read for rows that were never chosen.
struct ActiveOptions {
    Index min_rows = 32;
    Index max_rows = 2048;
    double tol = 1e-3;       // stop when RMS prediction change / prediction std falls below this
    int mode = 2;            // 0 D-optimal, 1 V-optimal, 2 hybrid
    Index pool = 20000;      // candidate rows considered
    Options model;
};

struct ActiveResult {
    std::vector<Index> rows;           // indices into X of the rows that were labelled
    std::vector<double> change;        // relative prediction change at each doubling
};

inline ActiveResult fit_active(BlackRubby& model, const MatrixXd& X, const std::function<double(Index)>& label,
                               const ActiveOptions& opt = ActiveOptions{}) {
    const Index n = X.rows(), d = X.cols();
    const Index pool = std::min<Index>(n, opt.pool);
    MatrixXd Xpool(pool, d);
    std::vector<Index> src(pool);
    for (Index i = 0; i < pool; ++i) { src[i] = i * n / pool; Xpool.row(i) = X.row(src[i]); }
    const InfoSelection sel = greedy_info_rows(Xpool, std::min(opt.max_rows, pool), opt.mode);

    const Index nref = std::min<Index>(pool, 256);
    MatrixXd Xref(nref, d);
    for (Index i = 0; i < nref; ++i) Xref.row(i) = Xpool.row(i * pool / nref);

    ActiveResult out;
    std::vector<double> labels;
    VectorXd prev;
    const Index limit = static_cast<Index>(sel.rows.size());
    for (Index k = std::min(opt.min_rows, limit);; k = std::min(limit, k * 2)) {
        while (static_cast<Index>(labels.size()) < k) {
            const Index row = src[sel.rows[labels.size()]];
            out.rows.push_back(row);
            labels.push_back(label(row));
        }
        MatrixXd Xk(k, d);
        VectorXd yk(k);
        for (Index i = 0; i < k; ++i) { Xk.row(i) = X.row(out.rows[i]); yk[i] = labels[i]; }
        BlackRubby candidate(opt.model);
        candidate.fit(Xk, yk);
        const VectorXd pred = candidate.predict(Xref);
        model = candidate;
        if (prev.size() == pred.size()) {
            const double sd = std::sqrt((pred.array() - pred.mean()).square().mean()) + 1e-12;
            const double change = std::sqrt((pred - prev).squaredNorm() / static_cast<double>(pred.size())) / sd;
            out.change.push_back(change);
            if (change < opt.tol) break;
        }
        prev = pred;
        if (k >= limit) break;
    }
    return out;
}

} // namespace blackrubby

#ifndef BLACKRUBBY_NO_MAIN

namespace {

// Edit these to change the data size; a command-line argument still overrides them.
constexpr int kSuiteRows = 100000;         // total rows per problem, split 80/20 train/test
constexpr long kScaleRows = 1000000;     // rows for the `scale` timing mode

using blackrubby::Clock;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Problem {
    std::string name;
    std::vector<double> lo, hi, ext;   // train range [lo,hi]; extrapolation box [lo,ext]
    bool integer = false;
    std::function<double(const double*)> f;
};

std::vector<int> g_pi;  // prime-counting table

void sample(const Problem& p, int n, std::uint64_t seed, bool extrapolate, MatrixXd& X, VectorXd& y) {
    const int d = static_cast<int>(p.lo.size());
    X.resize(n, d);
    y.resize(n);
    std::mt19937_64 r(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n;) {
        double row[8];
        bool outside = !extrapolate;
        for (int c = 0; c < d; ++c) {
            const double top = extrapolate ? p.ext[c] : p.hi[c];
            row[c] = p.lo[c] + u(r) * (top - p.lo[c]);
            if (p.integer) row[c] = std::floor(row[c]);
            if (row[c] > p.hi[c]) outside = true;
        }
        if (!outside) continue;
        for (int c = 0; c < d; ++c) X(i, c) = row[c];
        y[i] = p.f(row);
        ++i;
    }
}

// Score against the training-target variance so near-constant extrapolation targets don't blow up R2.
double r2(const VectorXd& pred, const VectorXd& y, double ref_var) {
    return 1.0 - (pred - y).squaredNorm() / (static_cast<double>(y.size()) * ref_var);
}

double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::vector<Problem> problems() {
    std::vector<Problem> p;
    p.push_back({"Feynman sqrt(k/m)", {1, 1}, {10, 5}, {15, 7.5}, false,
                 [](const double* x) { return std::sqrt(x[0] / x[1]); }});
    p.push_back({"Relativistic mass", {1, 0}, {10, 0.9}, {15, 0.97}, false,
                 [](const double* x) { return x[0] / std::sqrt(1.0 - x[1] * x[1]); }});
    p.push_back({"Kepler sqrt(a^3/(M1+M2))", {1, 1, 1}, {5, 10, 10}, {7.5, 15, 15}, false,
                 [](const double* x) { return std::sqrt(x[0] * x[0] * x[0] / (x[1] + x[2])); }});
    p.push_back({"Gravity m1*m2/r^2", {1, 1, 1}, {25, 25, 25}, {37, 37, 37}, false,
                 [](const double* x) { return x[0] * x[1] / (x[2] * x[2]); }});
    p.push_back({"Kinetic energy m*v^2/2", {1, 1}, {25, 25}, {37, 37}, false,
                 [](const double* x) { return 0.5 * x[0] * x[1] * x[1]; }});
    p.push_back({"Prime counting pi(n)", {500}, {1000000}, {1500000}, true,
                 [](const double* x) { return static_cast<double>(g_pi[static_cast<size_t>(x[0])]); }});
    p.push_back({"Centripetal v^2/r", {1, 1}, {15, 10}, {22, 15}, false,
                 [](const double* x) { return x[0] * x[0] / x[1]; }});
    p.push_back({"Logistic sigmoid", {-6}, {6}, {9}, false,
                 [](const double* x) { return 1.0 / (1.0 + std::exp(-x[0])); }});
    p.push_back({"Smooth sin(2xy)+.2x-.1y", {-1, -1}, {1, 1}, {1.5, 1.5}, false,
                 [](const double* x) { return std::sin(2.0 * x[0] * x[1]) + 0.2 * x[0] - 0.1 * x[1]; }});
    return p;
}

void run_suite(int n_total) {
    const int n_train = n_total * 4 / 5;
    const int n_test = n_total - n_train;
    {
        const int limit = 1500000;
        std::vector<char> composite(limit + 1, 0);
        g_pi.assign(limit + 1, 0);
        for (int i = 2; i <= limit; ++i) {
            if (!composite[i]) {
                for (long long j = 2LL * i; j <= limit; j += i) composite[static_cast<size_t>(j)] = 1;
            }
            g_pi[i] = g_pi[i - 1] + (composite[i] ? 0 : 1);
        }
    }
    std::cout << "total rows = " << n_total << " (80/20 split: " << n_train << " train, " << n_test
              << " test), plus 4000 extrapolation rows\n";
    std::cout << std::left << std::setw(26) << "problem" << std::setw(18) << "model"
              << std::right << std::setw(10) << "fit s" << std::setw(12) << "R2 interp"
              << std::setw(12) << "R2 extrap" << "  notes\n";
    std::cout << std::string(96, '-') << "\n";
    std::cout << std::fixed;

    double sum_fit[4] = {0, 0, 0, 0}, sum_i[4] = {0, 0, 0, 0}, sum_e[4] = {0, 0, 0, 0};
    const char* names[4] = {"Black(1024,a=1)", "Black-tuned", "Rubby-only", "BlackRubby"};
    int count = 0;
    for (const Problem& p : problems()) {
        MatrixXd Xall, Xtr, Xi, Xe;
        VectorXd yall, ytr, yi, ye;
        sample(p, n_total, 1, false, Xall, yall);
        Xtr = Xall.topRows(n_train);
        ytr = yall.head(n_train);
        Xi = Xall.bottomRows(n_test);
        yi = yall.tail(n_test);
        sample(p, 4000, 3, true, Xe, ye);
        const double ref_var = (ytr.array() - ytr.mean()).square().mean();

        auto report = [&](int slot, const std::string& note, double fit_s, const VectorXd& pi, const VectorXd& pe) {
            const double a = r2(pi, yi, ref_var), b = r2(pe, ye, ref_var);
            sum_fit[slot] += fit_s; sum_i[slot] += a; sum_e[slot] += b;
            std::cout << std::left << std::setw(26) << (slot == 0 ? p.name : "") << std::setw(18) << names[slot]
                      << std::right << std::setprecision(3) << std::setw(10) << fit_s
                      << std::setprecision(5) << std::setw(12) << a << std::setw(12) << b << "  " << note << "\n";
        };

        {
            black_regression::BlackRegression::Options o;
            o.expansion_dim = 1024;
            black_regression::BlackRegression m(o);
            auto t = Clock::now();
            m.fit(Xtr, ytr);
            const double fit_s = seconds(t);
            report(0, "", fit_s, m.predict(Xi), m.predict(Xe));
        }
        {
            blackrubby::Options o;
            o.use_symbolic = false;
            blackrubby::BlackRubby m(o);
            auto t = Clock::now();
            m.fit(Xtr, ytr);
            const double fit_s = seconds(t);
            std::ostringstream note;
            note << "bw=" << m.chosen_bandwidth() << " alpha=" << std::scientific << std::setprecision(1) << m.chosen_alpha();
            report(1, note.str(), fit_s, m.predict(Xi), m.predict(Xe));
        }
        blackrubby::BlackRubby m{blackrubby::Options{}};
        auto t = Clock::now();
        m.fit(Xtr, ytr);
        const double fit_s = seconds(t);
        report(2, "champion nmse=" + std::to_string(m.champion_nmse()), m.symbolic_seconds(),
               m.predict_symbolic(Xi), m.predict_symbolic(Xe));
        std::ostringstream note;
        note << "sym=" << (m.uses_symbolic() ? "yes" : "no") << " cols=" << m.symbolic_columns()
             << " bw=" << m.chosen_bandwidth() << " [rubby " << std::fixed << std::setprecision(2)
             << m.symbolic_seconds() << "s, select " << m.select_seconds() << "s, solve " << m.solve_seconds() << "s]";
        report(3, note.str(), fit_s, m.predict(Xi), m.predict(Xe));
        std::cout << "    champion: " << m.champion_formula() << "\n";
        std::cout << "\n";
        ++count;
    }
    std::cout << std::string(96, '-') << "\nMEAN over " << count << " problems\n";
    for (int s = 0; s < 4; ++s) {
        std::cout << std::left << std::setw(26) << "" << std::setw(18) << names[s] << std::right
                  << std::setprecision(3) << std::setw(10) << sum_fit[s] / count
                  << std::setprecision(5) << std::setw(12) << sum_i[s] / count << std::setw(12) << sum_e[s] / count << "\n";
    }
}

struct DataSpec {
    const char* file;
    const char* label;
    std::vector<std::string> features;
    std::string target;
};

// Edit this list to choose which CSV columns are features and which is the target.
const std::vector<DataSpec>& data_specs() {
    static const std::vector<DataSpec> specs = {
        {"Classificationtestset.csv", "Iris: petal_width", {"sepal_length", "sepal_width", "petal_length"}, "petal_width"},
        {"clean_2d_ready.csv", "clean_2d: target(Rank)", {"Rank"}, "target"},
        {"colebrook_white_dataset.csv", "Colebrook: f", {"reynolds_number", "relative_roughness"}, "friction_factor_darcy"},
        {"colebrook_white_dataset.csv", "Colebrook: log10 f", {"log10_Re", "log10_eps_D"}, "log10_f"},
        {"combined_gravitational_wave_data.csv", "GW: rh+ (t)", {"t (sec)"}, "rh+ (cm)"},
        {"framingham.csv", "Framingham: sysBP",
         {"male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp",
          "diabetes", "totChol", "diaBP", "BMI", "heartRate", "glucose"}, "sysBP"},
        {"Human_vs_AI_stock_Prediciton.csv", "Stock: actual return",
         {"Human_predicted_return_pct", "AI_predicted_return_pct", "Human_confidence", "AI_confidence"},
         "Acutal_return_pct"},
        {"1m.csv", "1m: Num(Rank)", {"Rank"}, "Num"},
    };
    return specs;
}

bool load_csv(const std::string& path, const DataSpec& spec, MatrixXd& X, VectorXd& y, size_t& dropped) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;
    auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cell;
        std::stringstream ss(s);
        while (std::getline(ss, cell, ',')) {
            while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
            out.push_back(cell);
        }
        return out;
    };
    const std::vector<std::string> header = split(line);
    auto find = [&](const std::string& name) {
        for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return static_cast<int>(i);
        throw std::runtime_error(std::string(spec.file) + ": missing column '" + name + "'");
    };
    std::vector<int> cols;
    for (const auto& f : spec.features) cols.push_back(find(f));
    cols.push_back(find(spec.target));
    const size_t d = spec.features.size();

    std::vector<double> flat;
    size_t rows = 0;
    dropped = 0;
    std::vector<double> row(d + 1);
    while (std::getline(in, line)) {
        const std::vector<std::string> cells = split(line);
        bool ok = true;
        for (size_t k = 0; k <= d && ok; ++k) {
            if (static_cast<size_t>(cols[k]) >= cells.size() || cells[cols[k]].empty()) { ok = false; break; }
            char* end = nullptr;
            const double v = std::strtod(cells[cols[k]].c_str(), &end);
            if (*end != '\0' || !std::isfinite(v)) ok = false; else row[k] = v;
        }
        if (!ok) { ++dropped; continue; }
        flat.insert(flat.end(), row.begin(), row.end());
        ++rows;
    }
    X.resize(rows, d);
    y.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < d; ++k) X(i, k) = flat[i * (d + 1) + k];
        y[i] = flat[i * (d + 1) + d];
    }
    return true;
}

double r2_test(const VectorXd& pred, const VectorXd& y) {
    return 1.0 - (pred - y).squaredNorm() / (y.array() - y.mean()).square().sum();
}

struct InputChart {
    enum class Kind { Raw, Whitened, SignedLog, Polar, PositiveLog } kind;
    Eigen::RowVectorXd center;
    Eigen::RowVectorXd scale;
    MatrixXd whitening;
    std::string name;
};

InputChart fit_chart(const MatrixXd& train, InputChart::Kind kind, const std::string& name) {
    InputChart chart;
    chart.kind = kind;
    chart.name = name;
    chart.center = train.colwise().mean();
    chart.scale.resize(train.cols());
    for (Eigen::Index c = 0; c < train.cols(); ++c) {
        const double variance = (train.col(c).array() - chart.center[c]).square().mean();
        chart.scale[c] = variance > 1e-24 ? std::sqrt(variance) : 1.0;
    }
    if (kind == InputChart::Kind::Whitened) {
        MatrixXd centered = train.rowwise() - chart.center;
        MatrixXd covariance = centered.transpose() * centered / static_cast<double>(train.rows());
        Eigen::SelfAdjointEigenSolver<MatrixXd> eig(covariance);
        if (eig.info() != Eigen::Success) throw std::runtime_error("PCA chart eigensolve failed.");
        VectorXd inverse_scale = eig.eigenvalues().cwiseMax(1e-10).cwiseSqrt().cwiseInverse();
        chart.whitening = eig.eigenvectors() * inverse_scale.asDiagonal();
    }
    if (kind == InputChart::Kind::PositiveLog && (train.array() <= 0.0).any()) {
        throw std::invalid_argument("positive-log chart requires strictly positive values.");
    }
    return chart;
}

MatrixXd transform_chart(const InputChart& chart, const Eigen::Ref<const MatrixXd>& input) {
    MatrixXd centered = input.rowwise() - chart.center;
    MatrixXd scaled = centered;
    scaled.array().rowwise() /= chart.scale.array();
    switch (chart.kind) {
        case InputChart::Kind::Raw:
            return input;
        case InputChart::Kind::Whitened:
            return centered * chart.whitening;
        case InputChart::Kind::SignedLog:
            return (scaled.array().sign() * scaled.array().abs().log1p()).matrix();
        case InputChart::Kind::Polar: {
            if (input.cols() != 2) throw std::invalid_argument("polar chart requires two input columns.");
            MatrixXd out(input.rows(), 3);
            const auto x = scaled.col(0).array();
            const auto y = scaled.col(1).array();
            const auto radius = (x.square() + y.square()).sqrt();
            out.col(0) = radius.matrix();
            out.col(1) = (x / radius.max(1e-12)).matrix();
            out.col(2) = (y / radius.max(1e-12)).matrix();
            return out;
        }
        case InputChart::Kind::PositiveLog:
            if ((input.array() <= 0.0).any()) throw std::invalid_argument("positive-log chart received non-positive values.");
            return input.array().log().matrix();
    }
    throw std::logic_error("unknown coordinate chart.");
}

blackrubby::BlackRubby make_chart_model(std::uint64_t seed = 42) {
    blackrubby::Options options;
    options.expansion_dim = 256;
    options.select_dim = 64;
    options.select_rows = 1024;
    options.use_symbolic = false;
    options.seed = seed;
    return blackrubby::BlackRubby(options);
}

void run_charts() {
    std::cout << "Coordinate atlas: chart selected by inner validation; outer test untouched; lower 1-R2 is better\n";
    std::cout << std::left << std::setw(24) << "dataset" << std::setw(16) << "selected chart" << std::right
              << std::setw(10) << "rows" << std::setw(16) << "base 1-R2" << std::setw(16) << "atlas 1-R2"
              << std::setw(10) << "fit s" << "\n" << std::string(92, '-') << "\n";
    for (const DataSpec& spec : data_specs()) {
        MatrixXd X;
        VectorXd y;
        size_t dropped = 0;
        if (!load_csv(spec.file, spec, X, y, dropped) || X.rows() < 100) continue;
        const Eigen::Index n = X.rows();
        std::vector<Eigen::Index> perm(n);
        std::iota(perm.begin(), perm.end(), Eigen::Index(0));
        std::mt19937_64 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        const Eigen::Index ntest = n / 5;
        const Eigen::Index ntrain = n - ntest;
        const Eigen::Index fit_rows = std::min<Eigen::Index>(ntrain, 8000);
        MatrixXd Xtr(fit_rows, X.cols()), Xte(ntest, X.cols());
        VectorXd ytr(fit_rows), yte(ntest);
        for (Eigen::Index i = 0; i < fit_rows; ++i) { Xtr.row(i) = X.row(perm[i]); ytr[i] = y[perm[i]]; }
        for (Eigen::Index i = 0; i < ntest; ++i) { Xte.row(i) = X.row(perm[ntrain + i]); yte[i] = y[perm[ntrain + i]]; }

        std::vector<std::pair<InputChart::Kind, std::string>> charts = {
            {InputChart::Kind::Raw, "raw"},
            {InputChart::Kind::Whitened, "PCA-whitened"},
            {InputChart::Kind::SignedLog, "signed-log"}
        };
        if (X.cols() == 2) charts.emplace_back(InputChart::Kind::Polar, "polar-chart");
        if ((Xtr.array() > 0.0).all()) charts.emplace_back(InputChart::Kind::PositiveLog, "positive-log");

        std::vector<std::vector<double>> chart_mse(charts.size());
        for (size_t fold = 0; fold < 3; ++fold) {
            std::vector<Eigen::Index> train_rows, val_rows;
            for (Eigen::Index i = 0; i < fit_rows; ++i) {
                (static_cast<size_t>(i) % 3 == fold ? val_rows : train_rows).push_back(i);
            }
            MatrixXd Xinner(train_rows.size(), X.cols()), Xval(val_rows.size(), X.cols());
            VectorXd yinner(train_rows.size()), yval(val_rows.size());
            for (size_t i = 0; i < train_rows.size(); ++i) {
                Xinner.row(i) = Xtr.row(train_rows[i]);
                yinner[i] = ytr[train_rows[i]];
            }
            for (size_t i = 0; i < val_rows.size(); ++i) {
                Xval.row(i) = Xtr.row(val_rows[i]);
                yval[i] = ytr[val_rows[i]];
            }
            for (size_t j = 0; j < charts.size(); ++j) {
                const auto& [kind, name] = charts[j];
                try {
                    const InputChart chart = fit_chart(Xinner, kind, name);
                    const MatrixXd transformed_train = transform_chart(chart, Xinner);
                    const MatrixXd transformed_val = transform_chart(chart, Xval);
                    auto model = make_chart_model(42 + fold);
                    model.fit(transformed_train, yinner);
                    const double mse = (model.predict(transformed_val) - yval).squaredNorm() /
                                       static_cast<double>(val_rows.size());
                    chart_mse[j].push_back(mse);
                } catch (const std::exception&) {
                }
            }
        }

        std::vector<double> means(charts.size(), std::numeric_limits<double>::infinity());
        std::vector<double> ses(charts.size(), std::numeric_limits<double>::infinity());
        double best_mse = std::numeric_limits<double>::infinity();
        size_t best_index = 0;
        for (size_t j = 0; j < charts.size(); ++j) {
            if (chart_mse[j].size() != 3) continue;
            means[j] = std::accumulate(chart_mse[j].begin(), chart_mse[j].end(), 0.0) / 3.0;
            double ss = 0.0;
            for (double value : chart_mse[j]) ss += (value - means[j]) * (value - means[j]);
            ses[j] = std::sqrt(ss / 2.0) / std::sqrt(3.0);
            if (means[j] < best_mse) { best_mse = means[j]; best_index = j; }
        }
        const double threshold = best_mse + ses[best_index];
        InputChart::Kind selected = InputChart::Kind::Raw;
        std::string selected_name = "raw";
        for (size_t j = 0; j < charts.size(); ++j) {
            if (means[j] <= threshold) {
                selected = charts[j].first;
                selected_name = charts[j].second;
                break;
            }
        }

        const InputChart base_chart = fit_chart(Xtr, InputChart::Kind::Raw, "raw");
        auto base_model = make_chart_model();
        base_model.fit(transform_chart(base_chart, Xtr), ytr);
        const double base_r2 = r2_test(base_model.predict(transform_chart(base_chart, Xte)), yte);

        const InputChart final_chart = fit_chart(Xtr, selected, selected_name);
        auto atlas_model = make_chart_model();
        const auto start = Clock::now();
        atlas_model.fit(transform_chart(final_chart, Xtr), ytr);
        const double elapsed = seconds(start);
        const double atlas_r2 = r2_test(atlas_model.predict(transform_chart(final_chart, Xte)), yte);
        std::cout << std::left << std::setw(24) << spec.label << std::setw(16) << selected_name << std::right
                  << std::setw(10) << fit_rows << std::scientific << std::setprecision(3)
                  << std::setw(16) << (1.0 - base_r2) << std::setw(16) << (1.0 - atlas_r2)
                  << std::fixed << std::setprecision(2) << std::setw(10) << elapsed << std::endl;
    }
}

void run_gw() {
    const DataSpec spec = {"combined_gravitational_wave_data.csv", "GW: rh+ (t,rhx)",
                           {"t (sec)", "rhx (cm)"}, "rh+ (cm)"};
    MatrixXd X;
    VectorXd y;
    size_t dropped = 0;
    if (!load_csv(spec.file, spec, X, y, dropped) || X.rows() < 100) {
        throw std::runtime_error("could not load enough gravitational-wave samples.");
    }

    std::vector<Eigen::Index> order(X.rows());
    std::iota(order.begin(), order.end(), Eigen::Index(0));
    std::sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return X(a, 0) < X(b, 0); });
    const Eigen::Index nominal_train = X.rows() * 4 / 5;
    const double test_start = X(order[nominal_train], 0);
    Eigen::Index ntrain = 0;
    while (ntrain < X.rows() && X(order[ntrain], 0) < test_start) ++ntrain;
    const Eigen::Index ntest = X.rows() - ntrain;
    MatrixXd Xtr(ntrain, 1), Xte(ntest, 1), Xtr_joint(ntrain, 2), Xte_joint(ntest, 2);
    VectorXd ytr(ntrain), yte(ntest);
    for (Eigen::Index i = 0; i < ntrain; ++i) {
        Xtr(i, 0) = X(order[i], 0);
        Xtr_joint.row(i) = X.row(order[i]);
        ytr[i] = y[order[i]];
    }
    for (Eigen::Index i = 0; i < ntest; ++i) {
        Xte(i, 0) = X(order[ntrain + i], 0);
        Xte_joint.row(i) = X.row(order[ntrain + i]);
        yte[i] = y[order[ntrain + i]];
    }

    Eigen::Index nval = ntrain / 5;
    const double val_start = Xtr(ntrain - nval, 0);
    while (ntrain - nval > 0 && Xtr(ntrain - nval - 1, 0) == val_start) ++nval;
    const Eigen::Index nfit = ntrain - nval;
    MatrixXd Xfit = Xtr.topRows(nfit), Xval = Xtr.bottomRows(nval);
    MatrixXd Xfit_joint = Xtr_joint.topRows(nfit), Xval_joint = Xtr_joint.bottomRows(nval);
    VectorXd yfit = ytr.head(nfit), yval = ytr.tail(nval);

    std::vector<double> spacings;
    spacings.reserve(static_cast<size_t>(nfit - 1));
    for (Eigen::Index i = 1; i < nfit; ++i) {
        const double dt = Xfit(i, 0) - Xfit(i - 1, 0);
        if (dt > 1e-15) spacings.push_back(dt);
    }
    if (spacings.empty()) throw std::runtime_error("time column has no distinct samples.");
    const auto mid = spacings.begin() + spacings.size() / 2;
    std::nth_element(spacings.begin(), mid, spacings.end());
    const double time_scale = std::sqrt((Xfit.col(0).array() - Xfit.col(0).mean()).square().mean());
    const double nyquist = std::min(20000.0, std::acos(-1.0) * time_scale / *mid);
    std::vector<double> candidates = {54.5981500331, 256.0, 1024.0, 4096.0, nyquist};
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-6 * std::max(a, b);
    }), candidates.end());

    const double target_scale = std::sqrt(yfit.squaredNorm() / static_cast<double>(nfit));
    double best_mse = std::numeric_limits<double>::infinity();
    double chosen = candidates.front();
    bool use_asinh = false;
    for (double ceiling : candidates) {
        for (int transform = 0; transform < 2; ++transform) {
            const VectorXd fit_target = transform == 0
                ? yfit
                : (yfit.array() / target_scale).unaryExpr([](double v) { return std::asinh(v); }).matrix();
            blackrubby::Options options;
            options.expansion_dim = 256;
            options.select_dim = 64;
            options.select_rows = 1024;
            options.use_symbolic = false;
            options.max_frequency = ceiling;
            blackrubby::BlackRubby candidate(options);
            candidate.fit(Xfit, fit_target);
            VectorXd prediction = candidate.predict(Xval);
            if (transform != 0) {
                prediction = (prediction.array().max(-20.0).min(20.0).sinh() * target_scale).matrix();
            }
            const double mse = (prediction - yval).squaredNorm() / static_cast<double>(nval);
            std::cout << "validation max-frequency=" << std::fixed << std::setprecision(1) << ceiling
                      << " target=" << (transform == 0 ? "raw" : "asinh")
                      << " mse=" << std::scientific << std::setprecision(6) << mse << "\n";
            if (mse < best_mse) {
                best_mse = mse;
                chosen = ceiling;
                use_asinh = transform != 0;
            }
        }
    }

    blackrubby::Options options;
    options.expansion_dim = 256;
    options.select_dim = 64;
    options.select_rows = 1024;
    options.use_symbolic = false;
    options.max_frequency = chosen;
    const VectorXd train_target = use_asinh
        ? (ytr.array() / target_scale).unaryExpr([](double v) { return std::asinh(v); }).matrix()
        : ytr;
    blackrubby::BlackRubby model(options);
    const auto start = Clock::now();
    model.fit(Xtr, train_target);
    const double elapsed = seconds(start);
    VectorXd prediction = model.predict(Xte);
    if (use_asinh) prediction = (prediction.array().max(-20.0).min(20.0).sinh() * target_scale).matrix();
    const double score = r2_test(prediction, yte);
    blackrubby::Options joint_options;
    joint_options.expansion_dim = 256;
    joint_options.select_dim = 64;
    joint_options.select_rows = 1024;
    joint_options.use_symbolic = false;
    blackrubby::BlackRubby joint_validation_model(joint_options);
    joint_validation_model.fit(Xfit_joint, yfit);
    const double joint_val_mse =
        (joint_validation_model.predict(Xval_joint) - yval).squaredNorm() / static_cast<double>(nval);
    blackrubby::BlackRubby joint_model(joint_options);
    const auto joint_start = Clock::now();
    joint_model.fit(Xtr_joint, ytr);
    const double joint_elapsed = seconds(joint_start);
    const double joint_score = r2_test(joint_model.predict(Xte_joint), yte);
    std::cout << "Chronological GW split: " << ntrain << " train, " << ntest << " future test rows\n"
              << "Distinct-time median spacing: " << *mid << " s; Nyquist-scaled ceiling: " << nyquist << "\n"
              << "Chosen ceiling: " << chosen << "; target transform: " << (use_asinh ? "asinh" : "raw")
              << "; test R2: " << std::setprecision(8) << score
              << "; fit seconds: " << elapsed << "\n"
              << "time+rhx raw-target model: validation MSE " << joint_val_mse << "; test R2 " << joint_score
              << "; fit seconds " << joint_elapsed << "\n";
}

void run_datasets() {
    std::cout << "80/20 shuffled split per file; R2 shown to 12 decimals, plus 1-R2 for resolution near 1\n";
    std::cout << std::left << std::setw(24) << "dataset" << std::setw(18) << "model" << std::right
              << std::setw(9) << "fit s" << std::setw(18) << "test R2" << std::setw(12) << "1-R2" << "  notes\n";
    std::cout << std::string(110, '-') << "\n";

    for (const DataSpec& spec : data_specs()) {
        MatrixXd X;
        VectorXd y;
        size_t dropped = 0;
        if (!load_csv(spec.file, spec, X, y, dropped)) {
            std::cout << std::left << std::setw(24) << spec.label << "skipped: cannot open " << spec.file << "\n\n";
            continue;
        }
        const Eigen::Index n = X.rows();
        if (n < 20) {
            std::cout << std::left << std::setw(24) << spec.label << "skipped: only " << n << " usable rows\n\n";
            continue;
        }
        std::vector<Eigen::Index> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        const Eigen::Index ntr = n * 4 / 5, nte = n - ntr;
        MatrixXd Xtr(ntr, X.cols()), Xte(nte, X.cols());
        VectorXd ytr(ntr), yte(nte);
        for (Eigen::Index i = 0; i < ntr; ++i) { Xtr.row(i) = X.row(perm[i]); ytr[i] = y[perm[i]]; }
        for (Eigen::Index i = 0; i < nte; ++i) { Xte.row(i) = X.row(perm[ntr + i]); yte[i] = y[perm[ntr + i]]; }

        auto report = [&](const char* model, double fit_s, const VectorXd& pred, const std::string& note) {
            const double r = r2_test(pred, yte);
            std::cout << std::left << std::setw(24) << (std::string(model) == "Black" ? spec.label : "")
                      << std::setw(18) << model << std::right << std::fixed << std::setprecision(3) << std::setw(9) << fit_s
                      << std::setprecision(12) << std::setw(18) << r << std::scientific << std::setprecision(2)
                      << std::setw(12) << (1.0 - r) << "  " << note << "\n";
        };

        {
            black_regression::BlackRegression::Options o;
            o.expansion_dim = ntr > 200000 ? 256 : 1024;
            black_regression::BlackRegression m(o);
            auto t = Clock::now();
            m.fit(Xtr, ytr);
            const double fit_s = seconds(t);
            VectorXd pred(nte);
            for (Eigen::Index b = 0; b < nte; b += 8192) {
                const Eigen::Index c = std::min<Eigen::Index>(8192, nte - b);
                pred.segment(b, c) = m.predict(Xte.middleRows(b, c));
            }
            std::ostringstream note;
            note << "rows " << ntr << "/" << nte << " (dropped " << dropped << "), D=" << o.expansion_dim;
            report("Black", fit_s, pred, note.str());
        }
        {
            blackrubby::Options o;
            o.use_symbolic = false;
            blackrubby::BlackRubby m(o);
            auto t = Clock::now();
            m.fit(Xtr, ytr);
            const double fit_s = seconds(t);
            std::ostringstream note;
            note << "bw=" << std::fixed << std::setprecision(2) << m.chosen_bandwidth() << " alpha="
                 << std::scientific << std::setprecision(1) << m.chosen_alpha();
            report("Black-tuned", fit_s, m.predict(Xte), note.str());
        }
        {
            blackrubby::BlackRubby m{blackrubby::Options{}};
            auto t = Clock::now();
            m.fit(Xtr, ytr);
            const double fit_s = seconds(t);
            report("Rubby-only", m.symbolic_seconds(), m.predict_symbolic(Xte), m.champion_formula());
            std::ostringstream note;
            note << "sym=" << (m.uses_symbolic() ? "yes" : "no") << " cols=" << m.symbolic_columns()
                 << " bw=" << std::fixed << std::setprecision(2) << m.chosen_bandwidth()
                 << " [rubby " << m.symbolic_seconds() << "s, select " << m.select_seconds()
                 << "s, solve " << m.solve_seconds() << "s]";
            report("BlackRubby", fit_s, m.predict(Xte), note.str());
        }
        std::cout << std::endl;
    }
}

// Does choosing rows by information gain beat random rows at the same row budget?
void run_infogeo() {
    const std::vector<Eigen::Index> budgets = {16, 32, 64, 128, 256, 512};
    std::cout << "Rows chosen by optimal design vs random rows, same budget, scored on a held-out 20% (1-R2, lower is better)\n";
    std::cout << std::left << std::setw(22) << "dataset" << std::right << std::setw(6) << "rows"
              << std::setw(11) << "random" << std::setw(11) << "D-opt" << std::setw(11) << "V-opt"
              << std::setw(11) << "hybrid" << std::setw(12) << "avg var" << "\n" << std::string(84, '-') << "\n";

    for (const DataSpec& spec : data_specs()) {
        const std::string label = spec.label;
        if (label.find("GW") != std::string::npos || label.find("1m") != std::string::npos) continue;
        MatrixXd X;
        VectorXd y;
        size_t dropped = 0;
        if (!load_csv(spec.file, spec, X, y, dropped) || X.rows() < 200) continue;
        const Eigen::Index n = X.rows();
        std::vector<Eigen::Index> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        const Eigen::Index ntr = n * 4 / 5, nte = n - ntr;
        MatrixXd Xtr(ntr, X.cols()), Xte(nte, X.cols());
        VectorXd ytr(ntr), yte(nte);
        for (Eigen::Index i = 0; i < ntr; ++i) { Xtr.row(i) = X.row(perm[i]); ytr[i] = y[perm[i]]; }
        for (Eigen::Index i = 0; i < nte; ++i) { Xte.row(i) = X.row(perm[ntr + i]); yte[i] = y[perm[ntr + i]]; }

        // Candidate pool: an even stride over the training rows, capped so selection stays cheap.
        const Eigen::Index pool = std::min<Eigen::Index>(ntr, 20000);
        MatrixXd Xpool(pool, X.cols());
        std::vector<Eigen::Index> pool_src(pool);
        for (Eigen::Index i = 0; i < pool; ++i) { pool_src[i] = i * ntr / pool; Xpool.row(i) = Xtr.row(pool_src[i]); }
        const blackrubby::InfoSelection dsel = blackrubby::greedy_info_rows(Xpool, budgets.back(), 0);
        const blackrubby::InfoSelection vsel = blackrubby::greedy_info_rows(Xpool, budgets.back(), 1);
        const blackrubby::InfoSelection hsel = blackrubby::greedy_info_rows(Xpool, budgets.back(), 2);

        auto score = [&](const MatrixXd& xs, const VectorXd& ys) {
            try {
                blackrubby::BlackRubby m{blackrubby::Options{}};
                m.fit(xs, ys);
                return r2_test(m.predict(Xte), yte);
            } catch (const std::exception&) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        };

        auto pick_rows = [&](const blackrubby::InfoSelection& sel, Eigen::Index m, MatrixXd& xo, VectorXd& yo) {
            xo.resize(m, X.cols());
            yo.resize(m);
            for (Eigen::Index i = 0; i < m; ++i) {
                xo.row(i) = Xtr.row(pool_src[sel.rows[i]]);
                yo[i] = ytr[pool_src[sel.rows[i]]];
            }
        };

        for (Eigen::Index m : budgets) {
            if (m > ntr || m > static_cast<Eigen::Index>(vsel.rows.size()) ||
                m > static_cast<Eigen::Index>(dsel.rows.size())) break;
            MatrixXd Xd, Xv, Xh;
            VectorXd yd, yv, yh;
            pick_rows(dsel, m, Xd, yd);
            pick_rows(vsel, m, Xv, yv);
            pick_rows(hsel, m, Xh, yh);
            const double a = score(Xtr.topRows(m), ytr.head(m)), b = score(Xd, yd), c = score(Xv, yv);
            const double h = score(Xh, yh);
            std::cout << std::left << std::setw(22) << (m == budgets.front() ? label : "") << std::right
                      << std::setw(6) << m << std::scientific << std::setprecision(1) << std::setw(11) << (1.0 - a)
                      << std::setw(11) << (1.0 - b) << std::setw(11) << (1.0 - c) << std::setw(11) << (1.0 - h)
                      << std::setw(12) << vsel.avg_var[m - 1] << std::endl;
        }
        std::cout << std::endl;
    }
}

// How much labelled data does the active learner need, compared with random rows and with all rows?
void run_active() {
    std::cout << std::left << std::setw(22) << "dataset" << std::right << std::setw(9) << "labelled" << std::setw(8)
              << "% data" << std::setw(12) << "active 1-R2" << std::setw(12) << "random 1-R2" << std::setw(11)
              << "full 1-R2" << std::setw(9) << "act s" << "\n" << std::string(83, '-') << "\n";
    for (const DataSpec& spec : data_specs()) {
        const std::string label = spec.label;
        if (label.find("GW") != std::string::npos || label.find("1m") != std::string::npos) continue;
        MatrixXd X;
        VectorXd y;
        size_t dropped = 0;
        if (!load_csv(spec.file, spec, X, y, dropped) || X.rows() < 200) continue;
        const Eigen::Index n = X.rows();
        std::vector<Eigen::Index> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        const Eigen::Index ntr = n * 4 / 5, nte = n - ntr;
        MatrixXd Xtr(ntr, X.cols()), Xte(nte, X.cols());
        VectorXd ytr(ntr), yte(nte);
        for (Eigen::Index i = 0; i < ntr; ++i) { Xtr.row(i) = X.row(perm[i]); ytr[i] = y[perm[i]]; }
        for (Eigen::Index i = 0; i < nte; ++i) { Xte.row(i) = X.row(perm[ntr + i]); yte[i] = y[perm[ntr + i]]; }

        blackrubby::BlackRubby active{blackrubby::Options{}};
        auto t = Clock::now();
        const blackrubby::ActiveResult res = blackrubby::fit_active(
            active, Xtr, [&](Eigen::Index i) { return ytr[i]; });
        const double act_s = seconds(t);
        const Eigen::Index used = static_cast<Eigen::Index>(res.rows.size());
        const double ra = r2_test(active.predict(Xte), yte);

        blackrubby::BlackRubby rnd{blackrubby::Options{}};
        rnd.fit(Xtr.topRows(used), ytr.head(used));
        const double rr = r2_test(rnd.predict(Xte), yte);

        blackrubby::BlackRubby full{blackrubby::Options{}};
        full.fit(Xtr, ytr);
        const double rf = r2_test(full.predict(Xte), yte);

        std::cout << std::left << std::setw(22) << label << std::right << std::setw(9) << used << std::fixed
                  << std::setprecision(2) << std::setw(8) << 100.0 * used / static_cast<double>(ntr)
                  << std::scientific << std::setprecision(1) << std::setw(12) << (1.0 - ra) << std::setw(12)
                  << (1.0 - rr) << std::setw(11) << (1.0 - rf) << std::fixed << std::setprecision(2) << std::setw(9)
                  << act_s << std::endl;
    }
}

void run_scale(Eigen::Index N) {
    const int d = 5;
    MatrixXd X(N, d);
    VectorXd y(N);
    std::mt19937_64 r(7);
    std::uniform_real_distribution<double> u(1.0, 10.0);
    for (Eigen::Index i = 0; i < N; ++i) {
        for (int c = 0; c < d; ++c) X(i, c) = u(r);
        y[i] = std::sqrt(X(i, 0) * X(i, 0) * X(i, 0) / (X(i, 1) + X(i, 2))) + 0.1 * X(i, 3) * X(i, 4);
    }
    const Eigen::Index test = std::min<Eigen::Index>(N, 50000);
    const double ref_var = (y.array() - y.mean()).square().mean();
    std::cout << std::fixed << std::setprecision(3) << "scale test: N=" << N << ", d=" << d << "\n";
    {
        black_regression::BlackRegression::Options o;
        o.expansion_dim = 256;
        black_regression::BlackRegression m(o);
        auto t = Clock::now();
        m.fit(X, y);
        const double fit_s = seconds(t);
        std::cout << "Black(256)   fit " << fit_s << "s  R2 " << r2(m.predict(X.topRows(test)), y.head(test), ref_var) << "\n";
    }
    {
        blackrubby::BlackRubby m{blackrubby::Options{}};
        auto t = Clock::now();
        m.fit(X, y);
        const double fit_s = seconds(t);
        std::cout << "BlackRubby   fit " << fit_s << "s  R2 " << r2(m.predict(X.topRows(test)), y.head(test), ref_var)
                  << "  [rubby " << m.symbolic_seconds() << "s, select " << m.select_seconds()
                  << "s, solve " << m.solve_seconds() << "s]\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "gw") {
            run_gw();
        } else if (argc > 1 && std::string(argv[1]) == "charts") {
            run_charts();
        } else if (argc > 1 && std::string(argv[1]) == "active") {
            run_active();
        } else if (argc > 1 && std::string(argv[1]) == "infogeo") {
            run_infogeo();
        } else if (argc > 1 && std::string(argv[1]) == "data") {
            run_datasets();
        } else if (argc > 1 && std::string(argv[1]) == "scale") {
            run_scale(argc > 2 ? std::atol(argv[2]) : kScaleRows);
        } else {
            run_suite(argc > 1 ? std::atoi(argv[1]) : kSuiteRows);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#endif
