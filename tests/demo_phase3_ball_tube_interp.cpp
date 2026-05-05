#include "MeshData.h"
#include "Obstacle.h"
#include "PathEnergy.h"
#include "ShellEnergy.h"
#include "SurfaceBarrier.h"
#include "TestMeshes.h"
#include "TrustRegionSolver.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using rsh::MeshData;

namespace {

std::string frame_path(const std::string &dir, int idx) {
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx
        << ".obj";
    return oss.str();
}

void remove_stale_outputs(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if ((name.rfind("frame_", 0) == 0 &&
             entry.path().extension() == ".obj") ||
            (name.rfind("energy_level_", 0) == 0 &&
             entry.path().extension() == ".csv") ||
            name == "energy.csv" || name == "obstacle.obj") {
            std::filesystem::remove(entry.path());
        }
    }
}

struct CliOptions {
    int icosphere_subdiv = 2;
    int num_frames = 9;
    int max_tr_iters = rsh::TrustRegionParams().max_iters;
    int max_cg_iters = rsh::TrustRegionParams().max_cg_iters;
    double tr_tol = rsh::TrustRegionParams().grad_tol;
    double initial_radius = 0.02;
    double max_radius = 0.20;
    double block_preconditioner_regularization =
        rsh::TrustRegionParams().block_preconditioner_regularization;
    double block_preconditioner_regularization_floor =
        rsh::TrustRegionParams().block_preconditioner_regularization_floor;
    double obstacle_weight = 1e-4;
    double sdf_guard_weight = 1e-6;
    double graph_beta = 1e-7;
    double rigid_translation_weight = 1.0;
    double rigid_rotation_weight = 1e-2;
    double ball_radius = 0.30;
    double endpoint_x = 1.80;
    double tube_inner_radius = 0.22;
    double tube_outer_radius = 0.30;
    double tube_half_length = 0.25;
    double squeeze_margin = 0.95;
    double shell_thickness = 0.01;
    double shell_lambda = rsh::ShellEnergyParams().lambda;
    double shell_mu = rsh::ShellEnergyParams().mu;
    double self_tpe_weight = rsh::PathEnergyParams().self_tpe_weight;
    double max_centroid_gap_factor = 2.5;
    int adaptive_depth = 12;
    int continuation_stages = 1;
    int temporal_refinement_levels = 1;
    int bootstrap_iters = 6;
    bool check_init_only = false;
    bool bootstrap_project_clearance = false;
    bool use_graph_residual_hessian = false;
    bool use_block_preconditioner =
        rsh::TrustRegionParams().use_block_diagonal_preconditioner;
    std::string init_mode = "piecewise";
    std::string out_dir = "out/ball_tube_interp";
};

int parse_int(int argc, char **argv, int &i, const std::string &name) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
    return std::stoi(argv[++i]);
}

double parse_double(int argc, char **argv, int &i, const std::string &name) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
    return std::stod(argv[++i]);
}

std::string parse_string(int argc, char **argv, int &i,
                         const std::string &name) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
    return argv[++i];
}

CliOptions parse_args(int argc, char **argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--num-frames") {
            opts.num_frames = parse_int(argc, argv, i, arg);
        } else if (arg == "--icosphere-subdiv") {
            opts.icosphere_subdiv = parse_int(argc, argv, i, arg);
        } else if (arg == "--max-tr-iters") {
            opts.max_tr_iters = parse_int(argc, argv, i, arg);
        } else if (arg == "--max-cg-iters") {
            opts.max_cg_iters = parse_int(argc, argv, i, arg);
        } else if (arg == "--tr-tol") {
            opts.tr_tol = parse_double(argc, argv, i, arg);
        } else if (arg == "--tr-initial-radius") {
            opts.initial_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--tr-max-radius") {
            opts.max_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--no-block-preconditioner") {
            opts.use_block_preconditioner = false;
        } else if (arg == "--graph-residual-hessian") {
            opts.use_graph_residual_hessian = true;
        } else if (arg == "--no-graph-residual-hessian") {
            opts.use_graph_residual_hessian = false;
        } else if (arg == "--block-preconditioner-regularization") {
            opts.block_preconditioner_regularization =
                parse_double(argc, argv, i, arg);
        } else if (arg == "--block-preconditioner-regularization-floor") {
            opts.block_preconditioner_regularization_floor =
                parse_double(argc, argv, i, arg);
        } else if (arg == "--obstacle-weight" ||
                   arg == "--tpe-barrier-weight") {
            opts.obstacle_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--sdf-guard-weight") {
            opts.sdf_guard_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--graph-beta") {
            opts.graph_beta = parse_double(argc, argv, i, arg);
        } else if (arg == "--rigid-translation-weight") {
            opts.rigid_translation_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--rigid-rotation-weight") {
            opts.rigid_rotation_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--ball-radius") {
            opts.ball_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--endpoint-x" || arg == "--path-half-length") {
            opts.endpoint_x = parse_double(argc, argv, i, arg);
        } else if (arg == "--tube-radius" ||
                   arg == "--tube-inner-radius") {
            opts.tube_inner_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--tube-outer-radius") {
            opts.tube_outer_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--tube-wall-thickness") {
            const double thickness = parse_double(argc, argv, i, arg);
            opts.tube_outer_radius = opts.tube_inner_radius + thickness;
        } else if (arg == "--tube-half-length") {
            opts.tube_half_length = parse_double(argc, argv, i, arg);
        } else if (arg == "--squeeze-margin") {
            opts.squeeze_margin = parse_double(argc, argv, i, arg);
        } else if (arg == "--shell-thickness") {
            opts.shell_thickness = parse_double(argc, argv, i, arg);
        } else if (arg == "--shell-lambda") {
            opts.shell_lambda = parse_double(argc, argv, i, arg);
        } else if (arg == "--shell-mu") {
            opts.shell_mu = parse_double(argc, argv, i, arg);
        } else if (arg == "--self-tpe-weight") {
            opts.self_tpe_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--max-centroid-gap-factor") {
            opts.max_centroid_gap_factor =
                parse_double(argc, argv, i, arg);
        } else if (arg == "--adaptive-depth") {
            opts.adaptive_depth = parse_int(argc, argv, i, arg);
        } else if (arg == "--continuation-stages") {
            opts.continuation_stages = parse_int(argc, argv, i, arg);
        } else if (arg == "--temporal-refinement-levels") {
            opts.temporal_refinement_levels = parse_int(argc, argv, i, arg);
        } else if (arg == "--bootstrap-iters") {
            opts.bootstrap_iters = parse_int(argc, argv, i, arg);
        } else if (arg == "--bootstrap-project-clearance") {
            opts.bootstrap_project_clearance = true;
        } else if (arg == "--no-bootstrap-project-clearance") {
            opts.bootstrap_project_clearance = false;
        } else if (arg == "--init-mode") {
            opts.init_mode = parse_string(argc, argv, i, arg);
        } else if (arg == "--check-init-only") {
            opts.check_init_only = true;
        } else if (arg == "--out-dir") {
            opts.out_dir = parse_string(argc, argv, i, arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.num_frames < 3) {
        throw std::runtime_error("--num-frames must be >= 3");
    }
    if (opts.icosphere_subdiv < 0) {
        throw std::runtime_error("--icosphere-subdiv must be nonnegative");
    }
    if (!(opts.ball_radius > 0.0 && opts.tube_inner_radius > 0.0 &&
          opts.tube_outer_radius > opts.tube_inner_radius &&
          opts.tube_half_length > 0.0 && opts.endpoint_x > 0.0)) {
        throw std::runtime_error("ball/tube geometry must be positive");
    }
    if (!(opts.endpoint_x > opts.tube_half_length + opts.ball_radius)) {
        throw std::runtime_error(
            "--endpoint-x must place the full ball outside the tube");
    }
    if (!(opts.obstacle_weight >= 0.0 && std::isfinite(opts.obstacle_weight))) {
        throw std::runtime_error("--obstacle-weight must be finite and >= 0");
    }
    if (!(opts.sdf_guard_weight >= 0.0 &&
          std::isfinite(opts.sdf_guard_weight))) {
        throw std::runtime_error("--sdf-guard-weight must be finite and >= 0");
    }
    if (!(opts.graph_beta >= 0.0 && std::isfinite(opts.graph_beta))) {
        throw std::runtime_error("--graph-beta must be finite and >= 0");
    }
    if (!(opts.rigid_translation_weight >= 0.0 &&
          std::isfinite(opts.rigid_translation_weight))) {
        throw std::runtime_error(
            "--rigid-translation-weight must be finite and >= 0");
    }
    if (!(opts.rigid_rotation_weight >= 0.0 &&
          std::isfinite(opts.rigid_rotation_weight))) {
        throw std::runtime_error(
            "--rigid-rotation-weight must be finite and >= 0");
    }
    if (!(opts.squeeze_margin > 0.0 && opts.squeeze_margin < 1.0 &&
          std::isfinite(opts.squeeze_margin))) {
        throw std::runtime_error("--squeeze-margin must be finite and in (0,1)");
    }
    if (!(opts.block_preconditioner_regularization > 0.0 &&
          std::isfinite(opts.block_preconditioner_regularization))) {
        throw std::runtime_error(
            "--block-preconditioner-regularization must be finite and > 0");
    }
    if (!(opts.block_preconditioner_regularization_floor > 0.0 &&
          std::isfinite(opts.block_preconditioner_regularization_floor))) {
        throw std::runtime_error(
            "--block-preconditioner-regularization-floor must be finite and > 0");
    }
    if (!(opts.shell_thickness > 0.0 &&
          std::isfinite(opts.shell_thickness))) {
        throw std::runtime_error("--shell-thickness must be finite and > 0");
    }
    if (!(opts.shell_lambda > 0.0 && std::isfinite(opts.shell_lambda))) {
        throw std::runtime_error("--shell-lambda must be finite and > 0");
    }
    if (!(opts.shell_mu > 0.0 && std::isfinite(opts.shell_mu))) {
        throw std::runtime_error("--shell-mu must be finite and > 0");
    }
    if (!(opts.self_tpe_weight >= 0.0 &&
          std::isfinite(opts.self_tpe_weight))) {
        throw std::runtime_error("--self-tpe-weight must be finite and >= 0");
    }
    if (!(opts.max_centroid_gap_factor > 0.0 &&
          std::isfinite(opts.max_centroid_gap_factor))) {
        throw std::runtime_error(
            "--max-centroid-gap-factor must be finite and > 0");
    }
    if (opts.adaptive_depth < 0) {
        throw std::runtime_error("--adaptive-depth must be nonnegative");
    }
    if (opts.continuation_stages < 1) {
        throw std::runtime_error("--continuation-stages must be >= 1");
    }
    if (opts.temporal_refinement_levels < 0) {
        throw std::runtime_error(
            "--temporal-refinement-levels must be nonnegative");
    }
    if (opts.bootstrap_iters < 0) {
        throw std::runtime_error("--bootstrap-iters must be nonnegative");
    }
    if (opts.init_mode != "linear" && opts.init_mode != "piecewise" &&
        opts.init_mode != "piecewise-smooth" &&
        opts.init_mode != "midpoint-pin" &&
        opts.init_mode != "tube-clearance") {
        throw std::runtime_error(
            "--init-mode must be one of: linear, piecewise, piecewise-smooth, "
            "midpoint-pin, tube-clearance");
    }
    return opts;
}

double min_signed_distance(const MeshData &mesh, const rsh::Obstacle &obs) {
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        m = std::min(m, obs.signed_distance(mesh.V.row(i).transpose()));
    }
    return m;
}

double centroid_x(const MeshData &mesh) {
    return mesh.V.col(0).mean();
}

bool centroid_x_monotone(const std::vector<MeshData> &frames) {
    constexpr double kTol = 1e-6;
    for (size_t k = 1; k < frames.size(); ++k) {
        if (centroid_x(frames[k]) + kTol < centroid_x(frames[k - 1])) {
            return false;
        }
    }
    return true;
}

double max_centroid_x_gap(const std::vector<MeshData> &frames) {
    double max_gap = 0.0;
    for (size_t k = 1; k < frames.size(); ++k) {
        max_gap =
            std::max(max_gap,
                     std::abs(centroid_x(frames[k]) -
                              centroid_x(frames[k - 1])));
    }
    return max_gap;
}

void write_hollow_tube_visual_obj(double half_length,
                                  double inner_radius,
                                  double outer_radius,
                                  int n_circ,
                                  const std::string &path) {
    std::ofstream out(path);
    out << std::setprecision(8);
    const double xs[2] = {-half_length, half_length};
    const double radii[2] = {outer_radius, inner_radius};
    for (double x : xs) {
        for (double r : radii) {
            for (int i = 0; i < n_circ; ++i) {
                const double t = 2.0 * M_PI * i / n_circ;
                out << "v " << x << " " << r * std::cos(t) << " "
                    << r * std::sin(t) << "\n";
            }
        }
    }

    auto idx = [&](int side, int radius_id, int i) {
        return 1 + (side * 2 + radius_id) * n_circ + (i % n_circ);
    };
    for (int i = 0; i < n_circ; ++i) {
        const int j = (i + 1) % n_circ;
        out << "f " << idx(0, 0, i) << " " << idx(1, 0, i) << " "
            << idx(1, 0, j) << "\n";
        out << "f " << idx(0, 0, i) << " " << idx(1, 0, j) << " "
            << idx(0, 0, j) << "\n";
        out << "f " << idx(0, 1, i) << " " << idx(1, 1, j) << " "
            << idx(1, 1, i) << "\n";
        out << "f " << idx(0, 1, i) << " " << idx(0, 1, j) << " "
            << idx(1, 1, j) << "\n";
        out << "f " << idx(0, 1, i) << " " << idx(0, 0, i) << " "
            << idx(0, 0, j) << "\n";
        out << "f " << idx(0, 1, i) << " " << idx(0, 0, j) << " "
            << idx(0, 1, j) << "\n";
        out << "f " << idx(1, 1, i) << " " << idx(1, 0, j) << " "
            << idx(1, 0, i) << "\n";
        out << "f " << idx(1, 1, i) << " " << idx(1, 1, j) << " "
            << idx(1, 0, j) << "\n";
    }
}

MeshData make_hollow_tube_mesh(double half_length,
                               double inner_radius,
                               double outer_radius,
                               int n_circ) {
    MeshData mesh;
    mesh.V.resize(4 * n_circ, 3);
    mesh.F.resize(8 * n_circ, 3);
    const double xs[2] = {-half_length, half_length};
    const double radii[2] = {outer_radius, inner_radius};
    int v = 0;
    for (double x : xs) {
        for (double r : radii) {
            for (int i = 0; i < n_circ; ++i) {
                const double t = 2.0 * M_PI * i / n_circ;
                mesh.V.row(v++) =
                    Eigen::RowVector3d(x, r * std::cos(t), r * std::sin(t));
            }
        }
    }

    auto idx = [&](int side, int radius_id, int i) {
        return (side * 2 + radius_id) * n_circ + (i % n_circ);
    };
    int f = 0;
    auto add_face = [&](int a, int b, int c) {
        mesh.F.row(f++) = Eigen::Vector3i(a, b, c);
    };
    for (int i = 0; i < n_circ; ++i) {
        const int j = (i + 1) % n_circ;
        add_face(idx(0, 0, i), idx(1, 0, i), idx(1, 0, j));
        add_face(idx(0, 0, i), idx(1, 0, j), idx(0, 0, j));
        add_face(idx(0, 1, i), idx(1, 1, j), idx(1, 1, i));
        add_face(idx(0, 1, i), idx(0, 1, j), idx(1, 1, j));
        add_face(idx(0, 1, i), idx(0, 0, i), idx(0, 0, j));
        add_face(idx(0, 1, i), idx(0, 0, j), idx(0, 1, j));
        add_face(idx(1, 1, i), idx(1, 0, j), idx(1, 0, i));
        add_face(idx(1, 1, i), idx(1, 1, j), idx(1, 0, j));
    }
    mesh.L0 = mesh.compute_L0();
    return mesh;
}

MeshData make_ball(double radius, int subdiv, const Eigen::RowVector3d &center) {
    MeshData ball = rsh::make_icosphere(subdiv);
    ball.normalize();
    const Eigen::RowVector3d c = ball.V.colwise().mean();
    ball.V.rowwise() -= c;
    const double normalized_radius = ball.V.rowwise().norm().maxCoeff();
    if (!(normalized_radius > 0.0)) {
        throw std::runtime_error("invalid procedural ball mesh");
    }
    ball.V *= radius / normalized_radius;
    ball.V.rowwise() += center;
    ball.L0 = ball.compute_L0();
    return ball;
}

MeshData lerp_mesh(const MeshData &a, const MeshData &b, double t) {
    MeshData out = a;
    out.V = (1.0 - t) * a.V + t * b.V;
    out.L0 = a.L0 > 0.0 ? a.L0 : a.compute_L0();
    return out;
}

double smoothstep01(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
}

// Feasible visual initializer: enforce radial clearance only where a vertex is
// in or near the tube. This is a diagnostic scaffold, not a paper objective.
double allowed_radial_at_tube_x(double x, const CliOptions &opts) {
    const double target_radial = opts.squeeze_margin * opts.tube_inner_radius;
    const double ax = std::abs(x);
    if (ax <= opts.tube_half_length) {
        return target_radial;
    }

    const double transition_width = opts.ball_radius;
    const double transition_end = opts.tube_half_length + transition_width;
    if (ax >= transition_end) {
        return std::numeric_limits<double>::infinity();
    }

    const double u = (ax - opts.tube_half_length) / transition_width;
    return target_radial +
           smoothstep01(u) * (opts.ball_radius - target_radial);
}

void apply_local_tube_clearance(MeshData &frame, const CliOptions &opts) {
    for (int i = 0; i < frame.n_vertices(); ++i) {
        const double allowed = allowed_radial_at_tube_x(frame.V(i, 0), opts);
        if (!std::isfinite(allowed)) continue;
        const double r = std::hypot(frame.V(i, 1), frame.V(i, 2));
        if (!(r > allowed) || !(r > 0.0)) continue;
        const double s = allowed / r;
        frame.V(i, 1) *= s;
        frame.V(i, 2) *= s;
    }
}

std::vector<MeshData> initial_path(const MeshData &x0,
                                   const MeshData &xN,
                                   const CliOptions &opts) {
    std::vector<MeshData> frames(static_cast<size_t>(opts.num_frames), x0);
    const int n = opts.num_frames - 1;

    if (opts.init_mode == "piecewise") {
        for (int k = 0; k <= n; ++k) {
            frames[static_cast<size_t>(k)] = (k < opts.num_frames / 2) ? x0 : xN;
        }
    } else if (opts.init_mode == "piecewise-smooth") {
        for (int k = 0; k <= n; ++k) {
            frames[static_cast<size_t>(k)] = (k < opts.num_frames / 2) ? x0 : xN;
        }
        std::vector<MeshData> smooth = frames;
        for (int k = 1; k < n; ++k) {
            smooth[static_cast<size_t>(k)].V =
                0.25 * frames[static_cast<size_t>(k - 1)].V +
                0.50 * frames[static_cast<size_t>(k)].V +
                0.25 * frames[static_cast<size_t>(k + 1)].V;
        }
        smooth.front() = x0;
        smooth.back() = xN;
        frames = std::move(smooth);
    } else if (opts.init_mode == "midpoint-pin" ||
               opts.init_mode == "tube-clearance") {
        for (int k = 0; k <= n; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(n);
            frames[static_cast<size_t>(k)] = lerp_mesh(x0, xN, t);
            if (k > 0 && k < n) {
                apply_local_tube_clearance(frames[static_cast<size_t>(k)],
                                           opts);
            }
        }
    } else {
        for (int k = 0; k <= n; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(n);
            frames[static_cast<size_t>(k)] = lerp_mesh(x0, xN, t);
        }
    }

    frames.front() = x0;
    frames.back() = xN;
    for (MeshData &f : frames) {
        f.F = x0.F;
        f.L0 = x0.L0 > 0.0 ? x0.L0 : x0.compute_L0();
    }
    return frames;
}

std::vector<MeshData> temporal_refine_piecewise_constant(
    const std::vector<MeshData> &frames) {
    if (frames.size() < 2) {
        return frames;
    }
    std::vector<MeshData> refined;
    refined.reserve(2 * frames.size() - 1);
    for (size_t k = 0; k + 1 < frames.size(); ++k) {
        refined.push_back(frames[k]);
        // RS Section 6.4.1 inserts x_{k+1/2} = x_k. This keeps every
        // individual time sample feasible even though the piecewise path has
        // jumps before the finer solve relaxes it.
        refined.push_back(frames[k]);
    }
    refined.push_back(frames.back());
    return refined;
}

std::string energy_csv_path(const std::string &out_dir,
                            int refinement_level,
                            int final_refinement_level) {
    if (refinement_level == final_refinement_level) {
        return out_dir + "/energy.csv";
    }
    std::ostringstream oss;
    oss << out_dir << "/energy_level_" << std::setfill('0') << std::setw(2)
        << refinement_level << ".csv";
    return oss.str();
}

void write_energy_header(std::ofstream &csv, int num_frames) {
    csv << "iter,total,shell_sum,repulsive_graph,accepted,converged,"
           "grad_norm,radius,step_norm,rho,centroid_x_monotone,min_phi_min,"
           "max_centroid_x_gap,rigid_sum";
    for (int k = 0; k < num_frames; ++k) {
        csv << ",obstacle_phi_" << k;
    }
    for (int k = 0; k < num_frames; ++k) {
        csv << ",min_phi_" << k;
    }
    for (int k = 0; k < num_frames; ++k) {
        csv << ",centroid_x_" << k;
    }
    csv << "\n";
}

void write_energy_row(std::ofstream &csv,
                      int iter,
                      const std::vector<MeshData> &frames,
                      const rsh::PathEnergyParams &params,
                      const rsh::Obstacle &obs,
                      bool accepted,
                      bool converged,
                      double grad_norm,
                      double radius,
                      double step_norm,
                      double rho) {
    const rsh::PathEnergyResult energy = rsh::path_energy(frames, params);
    std::vector<double> obstacle_phi(frames.size(), 0.0);
    std::vector<double> min_phi(frames.size(), 0.0);
    std::vector<double> cx(frames.size(), 0.0);
    double min_phi_min = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < frames.size(); ++k) {
        if (params.tpe_barrier_mesh != nullptr) {
            obstacle_phi[k] = rsh::surface_tpe_barrier_energy(
                frames[k], *params.tpe_barrier_mesh, params.tpe_alpha);
        } else {
            obstacle_phi[k] = rsh::obstacle_energy(frames[k], obs);
        }
        min_phi[k] = min_signed_distance(frames[k], obs);
        cx[k] = centroid_x(frames[k]);
        min_phi_min = std::min(min_phi_min, min_phi[k]);
    }

    csv << iter << "," << energy.terms.total << ","
        << energy.terms.shell_sum << "," << energy.terms.repulsive_sum
        << "," << (accepted ? 1 : 0) << "," << (converged ? 1 : 0)
        << "," << grad_norm << "," << radius << "," << step_norm
        << "," << rho << "," << (centroid_x_monotone(frames) ? 1 : 0)
        << "," << min_phi_min << "," << max_centroid_x_gap(frames)
        << "," << energy.terms.rigid_sum;
    for (double v : obstacle_phi) csv << "," << v;
    for (double v : min_phi) csv << "," << v;
    for (double v : cx) csv << "," << v;
    csv << "\n";
    csv.flush();
}

void validate_initial_path_feasible(const std::vector<MeshData> &frames,
                                    const rsh::Obstacle &obs) {
    for (size_t k = 0; k < frames.size(); ++k) {
        const double min_phi = min_signed_distance(frames[k], obs);
        const double phi = rsh::obstacle_energy(frames[k], obs);
        if (!(min_phi > 0.0) || !std::isfinite(phi)) {
            std::ostringstream oss;
            oss << "initial path frame " << k
                << " intersects the tube wall or has non-finite obstacle "
                   "energy: min_phi="
                << min_phi << ", obstacle_phi=" << phi;
            throw std::runtime_error(oss.str());
        }
    }
}

double membrane_density_from_cauchy_green(
    const Eigen::Matrix2d &A,
    const rsh::ShellEnergyParams &params) {
    const double detA = std::max(A.determinant(), params.det_smoothing);
    return 0.5 * params.mu * A.trace() +
           0.25 * params.lambda * detA -
           0.25 * (2.0 * params.mu + params.lambda) * std::log(detA) -
           params.mu - 0.25 * params.lambda;
}

double max_membrane_density_against_rest(
    const MeshData &rest,
    const MeshData &frame,
    const rsh::ShellEnergyParams &params = rsh::ShellEnergyParams()) {
    double max_density = 0.0;
    for (int f = 0; f < rest.n_faces(); ++f) {
        const int i = rest.F(f, 0);
        const int j = rest.F(f, 1);
        const int k = rest.F(f, 2);

        Eigen::Matrix<double, 3, 2> J_ref;
        J_ref.col(0) = rest.V.row(j) - rest.V.row(i);
        J_ref.col(1) = rest.V.row(k) - rest.V.row(i);
        Eigen::Matrix<double, 3, 2> J_def;
        J_def.col(0) = frame.V.row(j) - frame.V.row(i);
        J_def.col(1) = frame.V.row(k) - frame.V.row(i);

        const Eigen::Matrix2d I_ref = J_ref.transpose() * J_ref;
        const double det_ref = I_ref.determinant();
        if (!(det_ref > 1e-16) || !std::isfinite(det_ref)) {
            continue;
        }
        const Eigen::Matrix2d I_def = J_def.transpose() * J_def;
        const Eigen::Matrix2d A = I_ref.inverse() * I_def;
        max_density =
            std::max(max_density,
                     membrane_density_from_cauchy_green(A, params));
    }
    return max_density;
}

double max_interior_membrane_strain(const MeshData &rest,
                                    const std::vector<MeshData> &frames,
                                    const rsh::ShellEnergyParams &params) {
    double max_density = 0.0;
    for (size_t k = 1; k + 1 < frames.size(); ++k) {
        max_density =
            std::max(max_density,
                     max_membrane_density_against_rest(rest, frames[k],
                                                       params));
    }
    return max_density;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "demo_phase3_ball_tube_interp: " << e.what() << "\n";
        return 1;
    }

    rsh::HollowTubeObstacle tube(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::UnitX(),
        opts.tube_half_length,
        opts.tube_inner_radius,
        opts.tube_outer_radius);
    const MeshData tube_mesh = make_hollow_tube_mesh(
        opts.tube_half_length,
        opts.tube_inner_radius,
        opts.tube_outer_radius,
        48);

    const MeshData x0 = make_ball(
        opts.ball_radius, opts.icosphere_subdiv,
        Eigen::RowVector3d(-opts.endpoint_x, 0.0, 0.0));
    const MeshData xN = make_ball(
        opts.ball_radius, opts.icosphere_subdiv,
        Eigen::RowVector3d(opts.endpoint_x, 0.0, 0.0));

    std::vector<MeshData> frames = initial_path(x0, xN, opts);
    try {
        validate_initial_path_feasible(frames, tube);
    } catch (const std::exception &e) {
        std::cerr << "demo_phase3_ball_tube_interp: initial path intersects "
                     "the tube wall. "
                  << e.what()
                  << ". Try a smaller --ball-radius, larger "
                     "--tube-inner-radius, lower --squeeze-margin, or "
                     "--init-mode midpoint-pin.\n";
        return 2;
    }

    double initial_min_phi_min = std::numeric_limits<double>::infinity();
    for (const MeshData &m : frames) {
        initial_min_phi_min =
            std::min(initial_min_phi_min, min_signed_distance(m, tube));
    }
    if (opts.check_init_only) {
        std::cout << "demo_phase3_ball_tube_interp: init_only=1"
                  << " frames=" << opts.num_frames
                  << " ball_r=" << opts.ball_radius
                  << " tube_inner_r=" << opts.tube_inner_radius
                  << " tube_half_length=" << opts.tube_half_length
                  << " squeeze_margin=" << opts.squeeze_margin
                  << " init_mode=" << opts.init_mode
                  << " initial_min_phi_min=" << initial_min_phi_min
                  << " centroid_x_monotone="
                  << (centroid_x_monotone(frames) ? 1 : 0) << "\n";
        return 0;
    }

    std::filesystem::create_directories(opts.out_dir);
    remove_stale_outputs(opts.out_dir);
    tube_mesh.save_obj(opts.out_dir + "/obstacle.obj");

    rsh::PathEnergyParams target_energy_params;
    target_energy_params.shell.thickness = opts.shell_thickness;
    target_energy_params.shell.lambda = opts.shell_lambda;
    target_energy_params.shell.mu = opts.shell_mu;
    target_energy_params.graph_beta = opts.graph_beta;
    target_energy_params.self_tpe_weight = opts.self_tpe_weight;
    target_energy_params.tpe_barrier_mesh = &tube_mesh;
    target_energy_params.tpe_barrier_weight = opts.obstacle_weight;
    // The fixed-surface TPE barrier is the paper-aligned graph term. The SDF
    // term is a small discrete guard so coarse trial steps with vertices inside
    // the tube wall are rejected instead of accepted through quadrature gaps.
    // Keep it active through continuation: once an early stage enters the wall,
    // the infinite guard cannot be enabled later because its gradient is
    // undefined on an infeasible path.
    target_energy_params.obstacle = &tube;
    target_energy_params.obstacle_weight = opts.sdf_guard_weight;
    target_energy_params.rigid_translation_weight =
        opts.rigid_translation_weight;
    target_energy_params.rigid_rotation_weight =
        opts.rigid_rotation_weight;
    target_energy_params.tpe_alpha = 6.0;
    target_energy_params.tpe_theta = 0.5;
    target_energy_params.tpe_adaptive.enabled = true;
    target_energy_params.tpe_adaptive.theta = 10.0;
    target_energy_params.tpe_adaptive.max_depth = opts.adaptive_depth;
    target_energy_params.tpe_adaptive.max_stack_items = 1048576;

    auto stage_energy_params = [&](int stage) {
        rsh::PathEnergyParams p = target_energy_params;
        if (opts.continuation_stages > 1) {
            const double t = static_cast<double>(stage + 1) /
                             static_cast<double>(opts.continuation_stages);
            p.graph_beta = opts.graph_beta * t;
            p.obstacle_weight = opts.sdf_guard_weight;
        }
        return p;
    };

    rsh::PathEnergyParams active_energy_params = stage_energy_params(0);
    auto radius_scale_for = [&](const std::vector<MeshData> &pass_frames) {
        const int optimized_frame_count =
            std::max(0, static_cast<int>(pass_frames.size()) - 2);
        const int optimized_dofs =
            optimized_frame_count * x0.n_vertices() * 3;
        return std::sqrt(
            static_cast<double>(std::max(1, optimized_dofs)));
    };
    auto make_tr_params = [&](const std::vector<MeshData> &pass_frames) {
        rsh::TrustRegionParams tr_params;
        tr_params.max_iters = opts.max_tr_iters;
        tr_params.max_cg_iters = opts.max_cg_iters;
        tr_params.grad_tol = opts.tr_tol;
        const double radius_scale = radius_scale_for(pass_frames);
        tr_params.initial_radius = opts.initial_radius * radius_scale;
        tr_params.max_radius = opts.max_radius * radius_scale;
        tr_params.optimize_end_frame = false;
        tr_params.use_block_diagonal_preconditioner =
            opts.use_block_preconditioner;
        tr_params.use_graph_residual_hessian =
            opts.use_graph_residual_hessian;
        tr_params.block_preconditioner_regularization =
            opts.block_preconditioner_regularization;
        tr_params.block_preconditioner_regularization_floor =
            opts.block_preconditioner_regularization_floor;
        return tr_params;
    };
    const double initial_radius_scale = radius_scale_for(frames);

    std::cout << "demo_phase3_ball_tube_interp: frames=" << opts.num_frames
              << " ball_r=" << opts.ball_radius
              << " endpoint_x=" << opts.endpoint_x
              << " tube_inner_r=" << opts.tube_inner_radius
              << " tube_outer_r=" << opts.tube_outer_radius
              << " tube_half_length=" << opts.tube_half_length
              << " tpe_barrier_weight=" << opts.obstacle_weight
              << " sdf_guard_weight=" << opts.sdf_guard_weight
              << " graph_beta=" << opts.graph_beta
              << " rigid_translation_weight="
              << opts.rigid_translation_weight
              << " rigid_rotation_weight=" << opts.rigid_rotation_weight
              << " use_block_preconditioner="
              << (opts.use_block_preconditioner ? 1 : 0)
              << " use_graph_residual_hessian="
              << (opts.use_graph_residual_hessian ? 1 : 0)
              << " block_preconditioner_regularization="
              << opts.block_preconditioner_regularization
              << " block_preconditioner_regularization_floor="
              << opts.block_preconditioner_regularization_floor
              << " shell_thickness=" << opts.shell_thickness
              << " shell_lambda=" << opts.shell_lambda
              << " shell_mu=" << opts.shell_mu
              << " self_tpe_weight=" << opts.self_tpe_weight
              << " adaptive_depth=" << opts.adaptive_depth
              << " continuation_stages=" << opts.continuation_stages
              << " temporal_refinement_levels="
              << opts.temporal_refinement_levels
              << " bootstrap_project_clearance="
              << (opts.bootstrap_project_clearance ? 1 : 0)
              << " tr_initial_radius_rms=" << opts.initial_radius
              << " tr_max_radius_rms=" << opts.max_radius
              << " tr_radius_scale=" << initial_radius_scale
              << " squeeze_margin=" << opts.squeeze_margin
              << " init_mode=" << opts.init_mode << "\n";

    std::vector<MeshData> current_frames = frames;
    rsh::TrustRegionResult result;
    int total_accepted_steps = 0;
    int total_outer_iterations = 0;
    bool final_converged = false;
    if (opts.bootstrap_project_clearance && opts.init_mode == "piecewise" &&
        opts.bootstrap_iters > 0) {
        rsh::PathEnergyParams bootstrap_energy_params = target_energy_params;
        bootstrap_energy_params.self_tpe_weight = 0.0;
        bootstrap_energy_params.tpe_barrier_weight = 0.0;
        bootstrap_energy_params.obstacle_weight = 0.0;

        rsh::TrustRegionParams bootstrap_tr_params =
            make_tr_params(current_frames);
        bootstrap_tr_params.max_iters = opts.bootstrap_iters;
        bootstrap_tr_params.iteration_callback = nullptr;

        std::cout << "demo_phase3_ball_tube_interp: bootstrap_project_clearance=1"
                  << " bootstrap_iters=" << opts.bootstrap_iters
                  << " starting_gap=" << max_centroid_x_gap(current_frames)
                  << "\n";
        rsh::TrustRegionResult bootstrap_result =
            rsh::interpolate_geodesic_trust_region(
                current_frames, bootstrap_energy_params,
                bootstrap_tr_params);
        current_frames = std::move(bootstrap_result.frames);
        for (int k = 1; k + 1 < static_cast<int>(current_frames.size());
             ++k) {
            apply_local_tube_clearance(current_frames[static_cast<size_t>(k)],
                                       opts);
            current_frames[static_cast<size_t>(k)].L0 =
                current_frames[static_cast<size_t>(k)].compute_L0();
        }

        double bootstrap_min_phi =
            std::numeric_limits<double>::infinity();
        for (const MeshData &m : current_frames) {
            bootstrap_min_phi =
                std::min(bootstrap_min_phi, min_signed_distance(m, tube));
        }
        std::cout << "demo_phase3_ball_tube_interp: bootstrap_done"
                  << " accepted_steps=" << bootstrap_result.accepted_steps
                  << " outer_iterations="
                  << bootstrap_result.outer_iterations
                  << " projected_min_phi=" << bootstrap_min_phi
                  << " projected_max_centroid_gap="
                  << max_centroid_x_gap(current_frames)
                  << "\n";
        if (!(bootstrap_min_phi > 0.0)) {
            std::cerr
                << "demo_phase3_ball_tube_interp: bootstrap projection "
                   "failed to restore feasibility; min_phi="
                << bootstrap_min_phi << "\n";
            return 3;
        }
    }
    for (int refinement_level = 0;
         refinement_level <= opts.temporal_refinement_levels;
         ++refinement_level) {
        if (refinement_level > 0) {
            current_frames =
                temporal_refine_piecewise_constant(current_frames);
            double refined_min_phi =
                std::numeric_limits<double>::infinity();
            for (const MeshData &m : current_frames) {
                refined_min_phi =
                    std::min(refined_min_phi, min_signed_distance(m, tube));
            }
            std::cout
                << "demo_phase3_ball_tube_interp: temporal_refinement_level="
                << refinement_level << "/" << opts.temporal_refinement_levels
                << " frames=" << current_frames.size()
                << " inserted=piecewise_constant"
                << " min_phi_min=" << refined_min_phi
                << " max_centroid_gap="
                << max_centroid_x_gap(current_frames) << "\n";
            if (!(refined_min_phi > 0.0)) {
                std::cerr
                    << "demo_phase3_ball_tube_interp: temporal refinement "
                       "produced an infeasible frame; min_phi="
                    << refined_min_phi << "\n";
                return 3;
            }
        }

        const std::string csv_path = energy_csv_path(
            opts.out_dir, refinement_level,
            opts.temporal_refinement_levels);
        std::ofstream csv(csv_path);
        csv << std::setprecision(10);
        write_energy_header(csv, static_cast<int>(current_frames.size()));
        int iteration_offset = 0;

        for (int stage = 0; stage < opts.continuation_stages; ++stage) {
            active_energy_params = stage_energy_params(stage);
            const double t =
                (opts.continuation_stages == 1)
                    ? 1.0
                    : static_cast<double>(stage) /
                          static_cast<double>(opts.continuation_stages - 1);
            std::cout << "demo_phase3_ball_tube_interp: refinement_level="
                      << refinement_level << "/"
                      << opts.temporal_refinement_levels
                      << " continuation_stage=" << (stage + 1) << "/"
                      << opts.continuation_stages
                      << " frames=" << current_frames.size()
                      << " ramp=" << t
                      << " stage_self_tpe_weight="
                      << active_energy_params.self_tpe_weight
                      << " stage_tpe_barrier_weight="
                      << active_energy_params.tpe_barrier_weight
                      << " stage_graph_beta="
                      << active_energy_params.graph_beta
                      << " stage_sdf_guard_weight="
                      << active_energy_params.obstacle_weight
                      << " tr_radius_scale="
                      << radius_scale_for(current_frames) << "\n";
            if (active_energy_params.obstacle_weight > 0.0) {
                double stage_min_phi =
                    std::numeric_limits<double>::infinity();
                for (const MeshData &m : current_frames) {
                    stage_min_phi = std::min(
                        stage_min_phi, min_signed_distance(m, tube));
                }
                if (!(stage_min_phi > 0.0)) {
                    std::cerr
                        << "demo_phase3_ball_tube_interp: skipping positive "
                           "SDF-guard continuation stage because the incoming "
                           "path is already infeasible; min_phi="
                        << stage_min_phi << "\n";
                    break;
                }
            }

            rsh::TrustRegionParams tr_params =
                make_tr_params(current_frames);
            tr_params.iteration_callback =
                [&](const rsh::TrustRegionIterationInfo &info,
                    const std::vector<MeshData> &callback_frames) {
                    write_energy_row(
                        csv, iteration_offset + info.iteration,
                        callback_frames, active_energy_params, tube,
                        info.accepted, info.converged, info.grad_norm,
                        info.radius, info.step_norm, info.rho);
                };

            write_energy_row(csv, iteration_offset, current_frames,
                             active_energy_params, tube, true, false,
                             0.0, 0.0, 0.0, 0.0);
            result = rsh::interpolate_geodesic_trust_region(
                current_frames, active_energy_params, tr_params);
            current_frames = result.frames;
            total_accepted_steps += result.accepted_steps;
            total_outer_iterations += result.outer_iterations;
            final_converged = result.converged;
            write_energy_row(csv,
                             iteration_offset +
                                 result.outer_iterations + 1,
                             current_frames, active_energy_params, tube,
                             true, result.converged, 0.0, 0.0, 0.0, 0.0);
            iteration_offset += result.outer_iterations + 1;
        }
    }

    for (int k = 0; k < static_cast<int>(current_frames.size()); ++k) {
        current_frames[static_cast<size_t>(k)].save_obj(
            frame_path(opts.out_dir, k));
    }

    double min_phi_min = std::numeric_limits<double>::infinity();
    for (const MeshData &m : current_frames) {
        min_phi_min = std::min(min_phi_min, min_signed_distance(m, tube));
    }
    const bool monotone = centroid_x_monotone(current_frames);
    const double max_centroid_gap = max_centroid_x_gap(current_frames);
    const double uniform_centroid_gap =
        (2.0 * opts.endpoint_x) /
        static_cast<double>(current_frames.size() - 1);
    const double max_allowed_centroid_gap =
        opts.max_centroid_gap_factor * uniform_centroid_gap;
    const bool centroid_gap_ok =
        max_centroid_gap <= max_allowed_centroid_gap;
    const double max_membrane_strain =
        max_interior_membrane_strain(x0, current_frames,
                                     target_energy_params.shell);
    std::cout << "demo_phase3_ball_tube_interp: "
              << "final_frames=" << current_frames.size()
              << " temporal_refinement_levels="
              << opts.temporal_refinement_levels
              << " "
              << "accepted_steps=" << total_accepted_steps
              << " outer_iterations=" << total_outer_iterations
              << " converged=" << (final_converged ? 1 : 0)
              << " min_phi_min=" << min_phi_min
              << " centroid_x_monotone=" << (monotone ? 1 : 0)
              << " max_centroid_x_gap=" << max_centroid_gap
              << " max_allowed_centroid_x_gap="
              << max_allowed_centroid_gap
              << " centroid_gap_ok=" << (centroid_gap_ok ? 1 : 0)
              << " max_membrane_strain=" << max_membrane_strain << "\n";
    std::cout << "final_min_phi_min=" << min_phi_min << "\n";
    std::cout << "final_centroid_x_monotone=" << (monotone ? 1 : 0) << "\n";
    std::cout << "final_max_centroid_x_gap=" << max_centroid_gap << "\n";
    std::cout << "final_centroid_gap_ok=" << (centroid_gap_ok ? 1 : 0)
              << "\n";
    std::cout << "final_max_membrane_strain=" << max_membrane_strain << "\n";
    std::cout << "outputs,frames=" << opts.out_dir << "/frame_*.obj"
              << ",obstacle=" << opts.out_dir << "/obstacle.obj"
              << ",energy_csv=" << opts.out_dir << "/energy.csv\n";

    return (min_phi_min > 0.0 && monotone && centroid_gap_ok) ? 0 : 3;
}
