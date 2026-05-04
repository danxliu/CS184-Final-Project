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
    double initial_radius = rsh::TrustRegionParams().initial_radius;
    double max_radius = rsh::TrustRegionParams().max_radius;
    double obstacle_weight = 1.0;
    double ball_radius = 0.30;
    double endpoint_x = 1.45;
    double tube_inner_radius = 0.22;
    double tube_outer_radius = 0.36;
    double tube_half_length = 1.0;
    double squeeze_margin = 0.95;
    bool check_init_only = false;
    std::string init_mode = "midpoint-pin";
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
        } else if (arg == "--obstacle-weight") {
            opts.obstacle_weight = parse_double(argc, argv, i, arg);
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
    if (!(opts.squeeze_margin > 0.0 && opts.squeeze_margin < 1.0 &&
          std::isfinite(opts.squeeze_margin))) {
        throw std::runtime_error("--squeeze-margin must be finite and in (0,1)");
    }
    if (opts.init_mode != "linear" && opts.init_mode != "piecewise" &&
        opts.init_mode != "midpoint-pin") {
        throw std::runtime_error(
            "--init-mode must be one of: linear, piecewise, midpoint-pin");
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
    for (size_t k = 1; k < frames.size(); ++k) {
        if (centroid_x(frames[k]) + 1e-10 < centroid_x(frames[k - 1])) {
            return false;
        }
    }
    return true;
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

MeshData squeezed_midpoint_pose(const MeshData &base,
                                double ball_radius,
                                double tube_inner_radius,
                                double squeeze_margin) {
    MeshData out = base;
    const Eigen::RowVector3d c = out.V.colwise().mean();
    out.V.rowwise() -= c;
    const double s = squeeze_margin * tube_inner_radius / ball_radius;
    if (!(s > 0.0) || !std::isfinite(s)) {
        throw std::runtime_error("invalid midpoint squeeze scale");
    }
    out.V.col(0) *= 1.0 / (s * s);
    out.V.col(1) *= s;
    out.V.col(2) *= s;
    out.V.rowwise() += Eigen::RowVector3d(0.0, 0.0, 0.0);
    out.L0 = base.L0 > 0.0 ? base.L0 : base.compute_L0();
    return out;
}

void squeeze_frame_to_radial_clearance(MeshData &frame,
                                       double tube_inner_radius,
                                       double squeeze_margin) {
    const double target_radial = squeeze_margin * tube_inner_radius;
    double max_radial = 0.0;
    for (int i = 0; i < frame.n_vertices(); ++i) {
        max_radial = std::max(max_radial,
                              std::hypot(frame.V(i, 1), frame.V(i, 2)));
    }
    if (!(max_radial > target_radial) || !(target_radial > 0.0)) {
        return;
    }

    const double s = target_radial / max_radial;
    const double cx = frame.V.col(0).mean();
    frame.V.col(0).array() = cx + (frame.V.col(0).array() - cx) / (s * s);
    frame.V.col(1) *= s;
    frame.V.col(2) *= s;
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
    } else if (opts.init_mode == "midpoint-pin") {
        MeshData mid = squeezed_midpoint_pose(x0,
                                              opts.ball_radius,
                                              opts.tube_inner_radius,
                                              opts.squeeze_margin);
        for (int k = 0; k <= n; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(n);
            if (t <= 0.5) {
                frames[static_cast<size_t>(k)] = lerp_mesh(x0, mid, 2.0 * t);
            } else {
                frames[static_cast<size_t>(k)] =
                    lerp_mesh(mid, xN, 2.0 * (t - 0.5));
            }
        }
        // The default 9-frame path enters the tube before the pure lerp is
        // fully squeezed, so keep every interior frame inside the channel.
        for (int k = 1; k < n; ++k) {
            squeeze_frame_to_radial_clearance(frames[static_cast<size_t>(k)],
                                              opts.tube_inner_radius,
                                              opts.squeeze_margin);
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

void write_energy_header(std::ofstream &csv, int num_frames) {
    csv << "iter,total,shell_sum,repulsive_graph,accepted,converged,"
           "grad_norm,radius,step_norm,rho,centroid_x_monotone,min_phi_min";
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
        << "," << min_phi_min;
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
                                    const std::vector<MeshData> &frames) {
    double max_density = 0.0;
    for (size_t k = 1; k + 1 < frames.size(); ++k) {
        max_density =
            std::max(max_density,
                     max_membrane_density_against_rest(rest, frames[k]));
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

    rsh::PathEnergyParams energy_params;
    energy_params.tpe_barrier_mesh = &tube_mesh;
    energy_params.tpe_barrier_weight = opts.obstacle_weight;
    energy_params.tpe_alpha = 6.0;
    energy_params.tpe_theta = 0.5;
    energy_params.tpe_adaptive.enabled = true;
    energy_params.tpe_adaptive.theta = 10.0;
    energy_params.tpe_adaptive.max_depth = 6;

    std::ofstream csv(opts.out_dir + "/energy.csv");
    csv << std::setprecision(10);
    write_energy_header(csv, opts.num_frames);
    write_energy_row(csv, 0, frames, energy_params, tube,
                     true, false, 0.0, 0.0, 0.0, 0.0);

    rsh::TrustRegionParams tr_params;
    tr_params.max_iters = opts.max_tr_iters;
    tr_params.max_cg_iters = opts.max_cg_iters;
    tr_params.grad_tol = opts.tr_tol;
    tr_params.initial_radius = opts.initial_radius;
    tr_params.max_radius = opts.max_radius;
    tr_params.optimize_end_frame = false;
    tr_params.iteration_callback =
        [&](const rsh::TrustRegionIterationInfo &info,
            const std::vector<MeshData> &current_frames) {
            write_energy_row(csv, info.iteration, current_frames,
                             energy_params, tube, info.accepted,
                             info.converged, info.grad_norm, info.radius,
                             info.step_norm, info.rho);
        };

    std::cout << "demo_phase3_ball_tube_interp: frames=" << opts.num_frames
              << " ball_r=" << opts.ball_radius
              << " endpoint_x=" << opts.endpoint_x
              << " tube_inner_r=" << opts.tube_inner_radius
              << " obstacle_weight=" << opts.obstacle_weight
              << " squeeze_margin=" << opts.squeeze_margin
              << " init_mode=" << opts.init_mode << "\n";

    const rsh::TrustRegionResult result =
        rsh::interpolate_geodesic_trust_region(
            frames, energy_params, tr_params);

    for (int k = 0; k < static_cast<int>(result.frames.size()); ++k) {
        result.frames[static_cast<size_t>(k)].save_obj(
            frame_path(opts.out_dir, k));
    }

    write_energy_row(csv, result.outer_iterations + 1, result.frames,
                     energy_params, tube, true, result.converged,
                     0.0, 0.0, 0.0, 0.0);

    double min_phi_min = std::numeric_limits<double>::infinity();
    for (const MeshData &m : result.frames) {
        min_phi_min = std::min(min_phi_min, min_signed_distance(m, tube));
    }
    const bool monotone = centroid_x_monotone(result.frames);
    const double max_membrane_strain =
        max_interior_membrane_strain(x0, result.frames);
    std::cout << "demo_phase3_ball_tube_interp: "
              << "accepted_steps=" << result.accepted_steps
              << " outer_iterations=" << result.outer_iterations
              << " converged=" << (result.converged ? 1 : 0)
              << " min_phi_min=" << min_phi_min
              << " centroid_x_monotone=" << (monotone ? 1 : 0)
              << " max_membrane_strain=" << max_membrane_strain << "\n";
    std::cout << "final_min_phi_min=" << min_phi_min << "\n";
    std::cout << "final_centroid_x_monotone=" << (monotone ? 1 : 0) << "\n";
    std::cout << "final_max_membrane_strain=" << max_membrane_strain << "\n";
    std::cout << "outputs,frames=" << opts.out_dir << "/frame_*.obj"
              << ",obstacle=" << opts.out_dir << "/obstacle.obj"
              << ",energy_csv=" << opts.out_dir << "/energy.csv\n";

    return (min_phi_min > 0.0 && monotone) ? 0 : 3;
}
