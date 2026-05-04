#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include "OptimizeTPE.h"
#include "TestMeshes.h"
#include "TPE.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct CliOptions {
    int torus_nu = 24;
    int torus_nv = 12;
    int max_iters = 200;
    bool remesh = true;
    int remesh_every = 10;
    std::string out_dir = "out/gallery_genus1";
    bool dump_every_iter = true;
};

struct EnergyRow {
    int iter = 0;
    double energy = 0.0;
    double grad_norm = 0.0;
    double step_size = 0.0;
    int n_backtracks = 0;
    int did_remesh = 0;
};

std::string frame_path(const std::string &dir, int idx) {
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx
        << ".obj";
    return oss.str();
}

int parse_int_value(int argc,
                    char **argv,
                    int &i,
                    const std::string &arg,
                    const std::string &name) {
    const std::string eq_prefix = name + "=";
    if (arg == name) {
        if (i + 1 >= argc) {
            throw std::runtime_error(name + " requires an integer value");
        }
        return std::stoi(argv[++i]);
    }
    if (arg.rfind(eq_prefix, 0) == 0) {
        return std::stoi(arg.substr(eq_prefix.size()));
    }
    throw std::runtime_error("internal parse error for " + name);
}

std::string parse_string_value(int argc,
                               char **argv,
                               int &i,
                               const std::string &arg,
                               const std::string &name) {
    const std::string eq_prefix = name + "=";
    if (arg == name) {
        if (i + 1 >= argc) {
            throw std::runtime_error(name + " requires a value");
        }
        return argv[++i];
    }
    if (arg.rfind(eq_prefix, 0) == 0) {
        return arg.substr(eq_prefix.size());
    }
    throw std::runtime_error("internal parse error for " + name);
}

CliOptions parse_args(int argc, char **argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--torus_nu" || arg.rfind("--torus_nu=", 0) == 0) {
            opts.torus_nu = parse_int_value(argc, argv, i, arg, "--torus_nu");
        } else if (arg == "--torus-nu" ||
                   arg.rfind("--torus-nu=", 0) == 0) {
            opts.torus_nu = parse_int_value(argc, argv, i, arg, "--torus-nu");
        } else if (arg == "--torus_nv" ||
                   arg.rfind("--torus_nv=", 0) == 0) {
            opts.torus_nv = parse_int_value(argc, argv, i, arg, "--torus_nv");
        } else if (arg == "--torus-nv" ||
                   arg.rfind("--torus-nv=", 0) == 0) {
            opts.torus_nv = parse_int_value(argc, argv, i, arg, "--torus-nv");
        } else if (arg == "--max-iters" ||
                   arg.rfind("--max-iters=", 0) == 0) {
            opts.max_iters = parse_int_value(argc, argv, i, arg, "--max-iters");
        } else if (arg == "--out-dir" || arg.rfind("--out-dir=", 0) == 0) {
            opts.out_dir = parse_string_value(argc, argv, i, arg, "--out-dir");
        } else if (arg == "--remesh") {
            opts.remesh = true;
        } else if (arg == "--no-remesh") {
            opts.remesh = false;
        } else if (arg == "--remesh-every" ||
                   arg.rfind("--remesh-every=", 0) == 0) {
            opts.remesh_every =
                parse_int_value(argc, argv, i, arg, "--remesh-every");
        } else if (arg == "--dump-every-iter") {
            opts.dump_every_iter = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.torus_nu < 3 || opts.torus_nv < 3) {
        throw std::runtime_error("torus dimensions must both be at least 3");
    }
    if (opts.max_iters < 0) {
        throw std::runtime_error("--max-iters must be nonnegative");
    }
    if (opts.remesh_every < 0) {
        throw std::runtime_error("--remesh-every must be nonnegative");
    }
    if (opts.remesh_every == 0) {
        opts.remesh = false;
    }
    return opts;
}

Eigen::Vector3d bbox_extents(const rsh::MeshData &mesh) {
    const Eigen::Vector3d mn = mesh.V.colwise().minCoeff();
    const Eigen::Vector3d mx = mesh.V.colwise().maxCoeff();
    return (mx - mn).eval();
}

double tpe_energy_for(const rsh::MeshData &mesh,
                      double alpha,
                      double theta) {
    const rsh::FaceGeom g = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, g);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    return rsh::tpe_energy_bh(g, bvh, bp, alpha);
}

void perturb_vertices(rsh::MeshData &mesh) {
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 0.05 * mesh.L0);
    for (int v = 0; v < mesh.n_vertices(); ++v) {
        for (int c = 0; c < 3; ++c) {
            mesh.V(v, c) += normal(rng);
        }
    }
}

std::pair<int, int> edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

bool edge_manifold(const rsh::MeshData &mesh) {
    std::map<std::pair<int, int>, int> edge_count;
    for (int f = 0; f < mesh.n_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            ++edge_count[edge_key(mesh.F(f, k), mesh.F(f, (k + 1) % 3))];
        }
    }
    return std::all_of(edge_count.begin(),
                       edge_count.end(),
                       [](const auto &item) { return item.second <= 2; });
}

std::vector<EnergyRow> read_energy_rows(const std::string &path) {
    std::ifstream in(path);
    std::vector<EnergyRow> rows;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        EnergyRow row;
        if (iss >> row.iter >> row.energy >> row.grad_norm >>
            row.step_size >> row.n_backtracks >> row.did_remesh) {
            rows.push_back(row);
        }
    }
    return rows;
}

bool energy_monotone(const std::vector<EnergyRow> &rows) {
    if (rows.size() < 2) return true;
    for (size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].energy > rows[i - 1].energy + 1e-8) return false;
    }
    return true;
}

bool accepted_nonremesh_energy_monotone(const std::vector<EnergyRow> &rows) {
    if (rows.size() < 2) return true;
    for (size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].did_remesh) continue;
        if (rows[i].energy > rows[i - 1].energy + 1e-8) return false;
    }
    return true;
}

int remesh_energy_increase_count(const std::vector<EnergyRow> &rows) {
    int count = 0;
    for (size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].did_remesh &&
            rows[i].energy > rows[i - 1].energy + 1e-8) {
            ++count;
        }
    }
    return count;
}

bool two_largest_extents_close(const Eigen::Vector3d &ext,
                               double rel_tol,
                               double *relative_gap) {
    std::array<double, 3> sorted = {ext(0), ext(1), ext(2)};
    std::sort(sorted.begin(), sorted.end());
    const double gap =
        std::abs(sorted[2] - sorted[1]) / std::max(sorted[2], 1e-16);
    if (relative_gap != nullptr) *relative_gap = gap;
    return gap <= rel_tol;
}

int frame_count(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return 0;
    int count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("frame_", 0) == 0 &&
            entry.path().extension() == ".obj") {
            ++count;
        }
    }
    return count;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const CliOptions opts = parse_args(argc, argv);

        rsh::MeshData mesh = rsh::make_torus(1.0, 0.4,
                                             opts.torus_nu,
                                             opts.torus_nv);
        mesh.normalize();
        perturb_vertices(mesh);
        mesh.normalize();

        rsh::OptimizeTPEParams params;
        params.max_iters = opts.max_iters;
        params.grad_tol = 1e-4;
        params.remesh_every = opts.remesh ? opts.remesh_every : 0;
        params.out_dir = opts.out_dir;
        params.dump_every_iter = opts.dump_every_iter;
        params.constraints.pin_barycenter = true;

        const double e0 = tpe_energy_for(mesh, params.tpe_alpha,
                                         params.bvh_theta);
        const Eigen::Vector3d ext0 = bbox_extents(mesh);
        const int f0 = mesh.n_faces();

#ifdef _OPENMP
        const int omp_threads = omp_get_max_threads();
#else
        const int omp_threads = 1;
#endif

        const auto total_t0 = std::chrono::steady_clock::now();
        const auto opt_t0 = std::chrono::steady_clock::now();
        const rsh::OptimizeTPEResult result = rsh::optimize_tpe(mesh, params);
        const auto opt_t1 = std::chrono::steady_clock::now();

        result.final_mesh.save_obj(
            frame_path(opts.out_dir, result.iterations_completed));

        const std::vector<EnergyRow> rows =
            read_energy_rows(opts.out_dir + "/energy.csv");
        const Eigen::Vector3d ext1 = bbox_extents(result.final_mesh);
        double bbox_symmetry_gap = 0.0;
        const bool symmetry_ok =
            two_largest_extents_close(ext1, 0.05, &bbox_symmetry_gap);
        const bool monotone_ok = energy_monotone(rows);
        const bool accepted_monotone_ok =
            accepted_nonremesh_energy_monotone(rows);
        const int remesh_energy_increases = remesh_energy_increase_count(rows);
        const bool manifold_ok = edge_manifold(result.final_mesh);
        const bool face_count_ok =
            result.final_mesh.n_faces() >= 0.5 * static_cast<double>(f0) &&
            result.final_mesh.n_faces() <= 3.0 * static_cast<double>(f0);
        const bool remesh_ok = result.remeshes_completed >= 5;
        const bool remesh_acceptance_ok = !opts.remesh || remesh_ok;
        const int n_frames = frame_count(opts.out_dir);
        const auto total_t1 = std::chrono::steady_clock::now();
        const double optimize_seconds =
            std::chrono::duration<double>(opt_t1 - opt_t0).count();
        const double total_seconds =
            std::chrono::duration<double>(total_t1 - total_t0).count();

        std::cout << "demo_gallery_genus1"
                  << ",torus_nu=" << opts.torus_nu
                  << ",torus_nv=" << opts.torus_nv
                  << ",initial_faces=" << f0
                  << ",final_faces=" << result.final_mesh.n_faces()
                  << ",omp_threads=" << omp_threads
                  << ",remesh_enabled=" << (opts.remesh ? 1 : 0)
                  << ",remesh_every=" << params.remesh_every
                  << ",initial_energy=" << e0
                  << ",final_energy=" << result.final_energy
                  << ",initial_bbox_extents=(" << ext0(0) << ";" << ext0(1)
                  << ";" << ext0(2) << ")"
                  << ",final_bbox_extents=(" << ext1(0) << ";" << ext1(1)
                  << ";" << ext1(2) << ")"
                  << ",bbox_largest_gap=" << bbox_symmetry_gap
                  << ",iterations=" << result.iterations_completed
                  << ",remeshes=" << result.remeshes_completed
                  << ",remeshes_rejected=" << result.remeshes_rejected
                  << ",remeshes_rejected_face_budget="
                  << result.remeshes_rejected_face_budget
                  << ",frames=" << n_frames
                  << ",stop_reason=" << result.stop_reason
                  << ",optimize_seconds=" << optimize_seconds
                  << ",total_seconds=" << total_seconds
                  << "\n";

        std::cout << "acceptance"
                  << ",csv_energy_monotone=" << (monotone_ok ? 1 : 0)
                  << ",accepted_nonremesh_energy_monotone="
                  << (accepted_monotone_ok ? 1 : 0)
                  << ",remesh_energy_increases=" << remesh_energy_increases
                  << ",two_largest_bbox_extents_within_5pct="
                  << (symmetry_ok ? 1 : 0)
                  << ",edge_manifold=" << (manifold_ok ? 1 : 0)
                  << ",face_count_in_range=" << (face_count_ok ? 1 : 0)
                  << ",at_least_5_remeshes=" << (remesh_ok ? 1 : 0)
                  << ",remesh_acceptance_ok="
                  << (remesh_acceptance_ok ? 1 : 0)
                  << "\n";

        std::cout << "outputs"
                  << ",frames=" << opts.out_dir << "/frame_*.obj"
                  << ",energy_csv=" << opts.out_dir << "/energy.csv"
                  << "\n";
    } catch (const std::exception &e) {
        std::cerr << "demo_gallery_genus1: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
