#include "MeshData.h"
#include "PathEnergy.h"
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

struct CliOptions {
    int icosphere_subdiv = 2;
    int num_frames = 9;
    int max_tr_iters = 80;
    int max_cg_iters = 40;
    double tr_tol = 1e-7;
    double initial_radius = 0.20;
    double max_radius = 2.00;
    double rigid_translation_weight = 1.0;
    double rigid_rotation_weight = 0.0;
    double shell_thickness = 0.01;
    double self_tpe_weight = 0.0;
    double endpoint_x = 1.80;
    double ball_radius = 0.30;
    double max_gap_factor = 1.25;
    double max_centroid_rms_error = 0.05;
    bool use_block_preconditioner = false;
    double block_preconditioner_regularization =
        rsh::TrustRegionParams().block_preconditioner_regularization;
    double block_preconditioner_regularization_floor =
        rsh::TrustRegionParams().block_preconditioner_regularization_floor;
    std::string out_dir = "out/translated_sphere_path";
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
        } else if (arg == "--rigid-translation-weight") {
            opts.rigid_translation_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--rigid-rotation-weight") {
            opts.rigid_rotation_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--shell-thickness") {
            opts.shell_thickness = parse_double(argc, argv, i, arg);
        } else if (arg == "--self-tpe-weight") {
            opts.self_tpe_weight = parse_double(argc, argv, i, arg);
        } else if (arg == "--endpoint-x" || arg == "--path-half-length") {
            opts.endpoint_x = parse_double(argc, argv, i, arg);
        } else if (arg == "--ball-radius") {
            opts.ball_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--max-gap-factor") {
            opts.max_gap_factor = parse_double(argc, argv, i, arg);
        } else if (arg == "--max-centroid-rms-error") {
            opts.max_centroid_rms_error = parse_double(argc, argv, i, arg);
        } else if (arg == "--block-preconditioner") {
            opts.use_block_preconditioner = true;
        } else if (arg == "--block-preconditioner-regularization") {
            opts.block_preconditioner_regularization =
                parse_double(argc, argv, i, arg);
        } else if (arg == "--block-preconditioner-regularization-floor") {
            opts.block_preconditioner_regularization_floor =
                parse_double(argc, argv, i, arg);
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
    if (!(opts.endpoint_x > 0.0 && opts.ball_radius > 0.0)) {
        throw std::runtime_error("--endpoint-x and --ball-radius must be > 0");
    }
    if (!(opts.initial_radius > 0.0 && opts.max_radius >= opts.initial_radius)) {
        throw std::runtime_error(
            "trust-region radii must satisfy 0 < initial <= max");
    }
    if (!(opts.rigid_translation_weight > 0.0 &&
          std::isfinite(opts.rigid_translation_weight))) {
        throw std::runtime_error(
            "--rigid-translation-weight must be finite and > 0");
    }
    if (!(opts.rigid_rotation_weight >= 0.0 &&
          std::isfinite(opts.rigid_rotation_weight))) {
        throw std::runtime_error(
            "--rigid-rotation-weight must be finite and >= 0");
    }
    if (!(opts.shell_thickness >= 0.0 &&
          std::isfinite(opts.shell_thickness))) {
        throw std::runtime_error("--shell-thickness must be finite and >= 0");
    }
    if (!(opts.self_tpe_weight >= 0.0 &&
          std::isfinite(opts.self_tpe_weight))) {
        throw std::runtime_error("--self-tpe-weight must be finite and >= 0");
    }
    if (!(opts.max_gap_factor > 0.0 && std::isfinite(opts.max_gap_factor))) {
        throw std::runtime_error("--max-gap-factor must be finite and > 0");
    }
    if (!(opts.max_centroid_rms_error > 0.0 &&
          std::isfinite(opts.max_centroid_rms_error))) {
        throw std::runtime_error(
            "--max-centroid-rms-error must be finite and > 0");
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
    return opts;
}

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
            name == "energy.csv") {
            std::filesystem::remove(entry.path());
        }
    }
}

MeshData make_ball(double radius, int subdiv, const Eigen::RowVector3d &center) {
    MeshData ball = rsh::make_icosphere(subdiv);
    ball.normalize();
    ball.V.rowwise() -= ball.V.colwise().mean();
    const double r = ball.V.rowwise().norm().maxCoeff();
    if (!(r > 0.0)) {
        throw std::runtime_error("invalid procedural ball mesh");
    }
    ball.V *= radius / r;
    ball.V.rowwise() += center;
    ball.L0 = ball.compute_L0();
    return ball;
}

std::vector<MeshData> piecewise_path(const MeshData &x0,
                                     const MeshData &xN,
                                     int num_frames) {
    std::vector<MeshData> frames(static_cast<size_t>(num_frames), x0);
    for (int k = 0; k < num_frames; ++k) {
        frames[static_cast<size_t>(k)] = (k < num_frames / 2) ? x0 : xN;
    }
    frames.front() = x0;
    frames.back() = xN;
    for (MeshData &f : frames) {
        f.F = x0.F;
        f.L0 = x0.L0;
    }
    return frames;
}

double centroid_x(const MeshData &mesh) {
    return mesh.V.col(0).mean();
}

std::vector<double> centroid_xs(const std::vector<MeshData> &frames) {
    std::vector<double> out;
    out.reserve(frames.size());
    for (const MeshData &f : frames) out.push_back(centroid_x(f));
    return out;
}

double target_centroid_x(const CliOptions &opts, int k) {
    const double t = static_cast<double>(k) /
                     static_cast<double>(opts.num_frames - 1);
    return (1.0 - t) * (-opts.endpoint_x) + t * opts.endpoint_x;
}

double max_centroid_gap(const std::vector<double> &cx) {
    double max_gap = 0.0;
    for (size_t k = 1; k < cx.size(); ++k) {
        max_gap = std::max(max_gap, std::abs(cx[k] - cx[k - 1]));
    }
    return max_gap;
}

double centroid_rms_error(const std::vector<double> &cx,
                          const CliOptions &opts) {
    double sum = 0.0;
    for (int k = 0; k < opts.num_frames; ++k) {
        const double e = cx[static_cast<size_t>(k)] -
                         target_centroid_x(opts, k);
        sum += e * e;
    }
    return std::sqrt(sum / static_cast<double>(opts.num_frames));
}

bool centroid_monotone(const std::vector<double> &cx) {
    constexpr double kTol = 1e-7;
    for (size_t k = 1; k < cx.size(); ++k) {
        if (cx[k] + kTol < cx[k - 1]) return false;
    }
    return true;
}

void write_csv_row(std::ofstream &csv,
                   int iter,
                   const std::vector<MeshData> &frames,
                   const rsh::PathEnergyParams &params,
                   bool accepted,
                   bool converged,
                   double grad_norm,
                   double radius,
                   double step_norm,
                   double rho,
                   const CliOptions &opts) {
    const rsh::PathEnergyResult e = rsh::path_energy(frames, params);
    const std::vector<double> cx = centroid_xs(frames);
    csv << iter << "," << e.terms.total << "," << e.terms.shell_sum << ","
        << e.terms.repulsive_sum << "," << e.terms.rigid_sum << ","
        << (accepted ? 1 : 0) << "," << (converged ? 1 : 0) << ","
        << grad_norm << "," << radius << "," << step_norm << "," << rho
        << "," << max_centroid_gap(cx) << ","
        << centroid_rms_error(cx, opts) << ","
        << (centroid_monotone(cx) ? 1 : 0);
    for (double v : cx) csv << "," << v;
    csv << "\n";
    csv.flush();
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "demo_phase3_translated_sphere_path: " << e.what()
                  << "\n";
        return 1;
    }

    const MeshData x0 = make_ball(
        opts.ball_radius, opts.icosphere_subdiv,
        Eigen::RowVector3d(-opts.endpoint_x, 0.0, 0.0));
    const MeshData xN = make_ball(
        opts.ball_radius, opts.icosphere_subdiv,
        Eigen::RowVector3d(opts.endpoint_x, 0.0, 0.0));
    std::vector<MeshData> frames =
        piecewise_path(x0, xN, opts.num_frames);

    std::filesystem::create_directories(opts.out_dir);
    remove_stale_outputs(opts.out_dir);

    rsh::PathEnergyParams energy_params;
    energy_params.shell.thickness = opts.shell_thickness;
    energy_params.self_tpe_weight = opts.self_tpe_weight;
    energy_params.rigid_translation_weight =
        opts.rigid_translation_weight;
    energy_params.rigid_rotation_weight = opts.rigid_rotation_weight;
    energy_params.tpe_alpha = 6.0;
    energy_params.tpe_theta = 0.5;
    energy_params.tpe_adaptive.enabled = true;
    energy_params.tpe_adaptive.theta = 10.0;
    energy_params.tpe_adaptive.max_depth = 8;
    energy_params.tpe_adaptive.max_stack_items = 1048576;

    std::ofstream csv(opts.out_dir + "/energy.csv");
    csv << std::setprecision(10);
    csv << "iter,total,shell_sum,repulsive_graph,rigid_sum,accepted,"
           "converged,grad_norm,radius,step_norm,rho,max_centroid_gap,"
           "centroid_rms_error,centroid_monotone";
    for (int k = 0; k < opts.num_frames; ++k) {
        csv << ",centroid_x_" << k;
    }
    csv << "\n";
    write_csv_row(csv, 0, frames, energy_params, true, false,
                  0.0, 0.0, 0.0, 0.0, opts);

    rsh::TrustRegionParams tr_params;
    tr_params.max_iters = opts.max_tr_iters;
    tr_params.max_cg_iters = opts.max_cg_iters;
    tr_params.grad_tol = opts.tr_tol;
    const int optimized_frame_count = std::max(0, opts.num_frames - 2);
    const int optimized_dofs =
        optimized_frame_count * x0.n_vertices() * 3;
    const double radius_scale =
        std::sqrt(static_cast<double>(std::max(1, optimized_dofs)));
    tr_params.initial_radius = opts.initial_radius * radius_scale;
    tr_params.max_radius = opts.max_radius * radius_scale;
    tr_params.use_block_diagonal_preconditioner =
        opts.use_block_preconditioner;
    tr_params.block_preconditioner_regularization =
        opts.block_preconditioner_regularization;
    tr_params.block_preconditioner_regularization_floor =
        opts.block_preconditioner_regularization_floor;
    tr_params.iteration_callback =
        [&](const rsh::TrustRegionIterationInfo &info,
            const std::vector<MeshData> &current_frames) {
            write_csv_row(csv, info.iteration, current_frames, energy_params,
                          info.accepted, info.converged, info.grad_norm,
                          info.radius, info.step_norm, info.rho, opts);
        };

    std::cout << "demo_phase3_translated_sphere_path: frames="
              << opts.num_frames
              << " verts=" << x0.n_vertices()
              << " endpoint_x=" << opts.endpoint_x
              << " ball_radius=" << opts.ball_radius
              << " rigid_translation_weight="
              << opts.rigid_translation_weight
              << " rigid_rotation_weight=" << opts.rigid_rotation_weight
              << " shell_thickness=" << opts.shell_thickness
              << " self_tpe_weight=" << opts.self_tpe_weight
              << " use_block_preconditioner="
              << (opts.use_block_preconditioner ? 1 : 0)
              << " tr_initial_radius_rms=" << opts.initial_radius
              << " tr_max_radius_rms=" << opts.max_radius
              << " tr_radius_scale=" << radius_scale << "\n";

    const rsh::TrustRegionResult result =
        rsh::interpolate_geodesic_trust_region(
            frames, energy_params, tr_params);

    for (int k = 0; k < static_cast<int>(result.frames.size()); ++k) {
        result.frames[static_cast<size_t>(k)].save_obj(
            frame_path(opts.out_dir, k));
    }
    write_csv_row(csv, result.outer_iterations + 1, result.frames,
                  energy_params, true, result.converged, 0.0, 0.0,
                  0.0, 0.0, opts);

    const std::vector<double> cx = centroid_xs(result.frames);
    const double uniform_gap =
        (2.0 * opts.endpoint_x) /
        static_cast<double>(opts.num_frames - 1);
    const double max_allowed_gap = opts.max_gap_factor * uniform_gap;
    const double gap = max_centroid_gap(cx);
    const double rms = centroid_rms_error(cx, opts);
    const bool monotone = centroid_monotone(cx);
    const bool gap_ok = gap <= max_allowed_gap;
    const bool rms_ok = rms <= opts.max_centroid_rms_error;

    std::cout << "demo_phase3_translated_sphere_path: accepted_steps="
              << result.accepted_steps
              << " outer_iterations=" << result.outer_iterations
              << " converged=" << (result.converged ? 1 : 0)
              << " max_centroid_gap=" << gap
              << " max_allowed_centroid_gap=" << max_allowed_gap
              << " centroid_rms_error=" << rms
              << " max_allowed_centroid_rms_error="
              << opts.max_centroid_rms_error
              << " centroid_monotone=" << (monotone ? 1 : 0) << "\n";
    std::cout << "outputs,frames=" << opts.out_dir << "/frame_*.obj"
              << ",energy_csv=" << opts.out_dir << "/energy.csv\n";

    return (gap_ok && rms_ok && monotone) ? 0 : 3;
}
