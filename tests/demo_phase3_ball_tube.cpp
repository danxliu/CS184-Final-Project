#include "ExtrapolationSolver.h"
#include "MeshData.h"
#include "Obstacle.h"
#include "PathEnergy.h"
#include "TestMeshes.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
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

void remove_stale_frames(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("frame_", 0) == 0 &&
            entry.path().extension() == ".obj") {
            std::filesystem::remove(entry.path());
        }
    }
}

// Phase 3.5 ball-through-tube smoke. Status (2026-05-04):
//
// The obstacle barrier wiring (PathEnergy + ExtrapolationSolver) is in place
// and energy / gradient FD-checks pass. The current ExtrapolationSolver
// uses an FD-on-FD Jacobian (FD on the bending grad_ref FD-gradient) plus
// an unpreconditioned GMRES, which diverges in the obstacle-near regime at
// even icosphere(1) scale. Until that is fixed (Sherman-Morrison rank-1
// trick from RS Eq. 23 + analytical bending grad_ref + a GMRES
// preconditioner), the demo is best run far from the obstacle to confirm
// the integrator follows constant velocity. Close-to-obstacle runs do
// raise the obstacle energy correctly and halt on intersection, but Newton
// does not actually deflect the trajectory.
struct CliOptions {
    int icosphere_subdiv = 1;
    int num_frames = 20;
    double ball_radius = 0.3;
    double start_x = -3.0;
    double speed = 0.05;
    double tube_radius = 0.5;
    double tube_half_length = 1.5;
    int max_newton_iters = 8;
    double newton_tol = 1e-2;
    std::string out_dir = "out/ball_tube";
};

int parse_int(int argc, char **argv, int &i, const std::string &name) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
    return std::stoi(argv[++i]);
}

double parse_double(int argc, char **argv, int &i, const std::string &name) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
    return std::stod(argv[++i]);
}

std::string parse_string(int argc, char **argv, int &i, const std::string &name) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
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
        } else if (arg == "--ball-radius") {
            opts.ball_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--start-x") {
            opts.start_x = parse_double(argc, argv, i, arg);
        } else if (arg == "--speed") {
            opts.speed = parse_double(argc, argv, i, arg);
        } else if (arg == "--tube-radius") {
            opts.tube_radius = parse_double(argc, argv, i, arg);
        } else if (arg == "--tube-half-length") {
            opts.tube_half_length = parse_double(argc, argv, i, arg);
        } else if (arg == "--max-newton-iters") {
            opts.max_newton_iters = parse_int(argc, argv, i, arg);
        } else if (arg == "--newton-tol") {
            opts.newton_tol = parse_double(argc, argv, i, arg);
        } else if (arg == "--out-dir") {
            opts.out_dir = parse_string(argc, argv, i, arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.num_frames < 3) {
        throw std::runtime_error("--num-frames must be >= 3");
    }
    if (!(opts.ball_radius > 0.0 && opts.tube_radius > 0.0 &&
          opts.tube_half_length > 0.0)) {
        throw std::runtime_error("ball/tube geometry must be positive");
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

} // namespace

int main(int argc, char **argv) {
    CliOptions opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "demo_phase3_ball_tube: " << e.what() << "\n";
        return 1;
    }

    // Ball: icosphere shrunk to ball_radius, translated so its initial
    // centroid is at (start_x, 0, 0). It moves in +x.
    MeshData ball = rsh::make_icosphere(opts.icosphere_subdiv);
    ball.normalize();
    ball.V *= opts.ball_radius;
    ball.V.rowwise() += Eigen::RowVector3d(opts.start_x, 0.0, 0.0);
    ball.L0 = ball.compute_L0();

    // Tube: vertical capsule at the origin, axis along Z, radius tube_radius.
    // Ball moves in +x and meets the tube; tube_radius < ball_radius +
    // tube_half_length so the ball must deform around the tube to clear it.
    rsh::CapsuleObstacle tube(
        Eigen::Vector3d(0.0, 0.0, -opts.tube_half_length),
        Eigen::Vector3d(0.0, 0.0,  opts.tube_half_length),
        opts.tube_radius);

    // Initial 2-frame stencil: x_km1 at start, x_k = x_km1 + (speed, 0, 0).
    MeshData x_km1 = ball;
    MeshData x_k = ball;
    x_k.V.rowwise() += Eigen::RowVector3d(opts.speed, 0.0, 0.0);
    x_k.L0 = x_k.compute_L0();

    // Sanity: both initial frames must be feasible (vertices outside tube).
    if (!std::isfinite(rsh::obstacle_energy(x_km1, tube)) ||
        !std::isfinite(rsh::obstacle_energy(x_k,   tube))) {
        std::cerr << "demo_phase3_ball_tube: initial frame intersects tube. "
                     "Try larger --start-x (more negative) or smaller "
                     "--ball-radius.\n";
        return 1;
    }

    std::filesystem::create_directories(opts.out_dir);
    remove_stale_frames(opts.out_dir);

    rsh::PathEnergyParams params;
    params.obstacle = &tube;
    params.tpe_alpha = 6.0;
    params.tpe_theta = 0.5;

    rsh::ExtrapolationParams ext;
    ext.max_newton_iters = opts.max_newton_iters;
    ext.newton_tol = opts.newton_tol;

    x_km1.save_obj(frame_path(opts.out_dir, 0));
    x_k.save_obj(frame_path(opts.out_dir, 1));

    std::ofstream csv(opts.out_dir + "/energy.csv");
    csv << "frame,total,shell,repulsive,obstacle,min_phi,centroid_x,"
           "newton_iters,converged\n";
    csv << std::setprecision(10);

    auto log_frame = [&](int frame, const MeshData &m_prev,
                         const MeshData &m_cur, int newton_iters,
                         bool converged) {
        const std::vector<MeshData> two = {m_prev, m_cur};
        const rsh::PathEnergyResult r = rsh::path_energy(two, params);
        const double min_phi = min_signed_distance(m_cur, tube);
        const double cx = m_cur.V.col(0).mean();
        csv << frame << "," << r.terms.total << "," << r.terms.shell_sum
            << "," << r.terms.repulsive_sum << "," << r.terms.obstacle_sum
            << "," << min_phi << "," << cx << "," << newton_iters << ","
            << (converged ? 1 : 0) << "\n";
        csv.flush();
    };

    log_frame(0, x_km1, x_km1, 0, true);
    log_frame(1, x_km1, x_k, 0, true);

    std::cout << "demo_phase3_ball_tube: "
              << "ball_r=" << opts.ball_radius
              << " tube_r=" << opts.tube_radius
              << " speed=" << opts.speed
              << " frames=" << opts.num_frames << "\n";

    for (int frame = 2; frame < opts.num_frames; ++frame) {
        rsh::ExtrapolationResult res =
            rsh::extrapolate_geodesic(x_km1, x_k, params, ext);

        if (!std::isfinite(rsh::obstacle_energy(res.next_frame, tube))) {
            std::cerr << "frame " << frame
                      << ": next_frame intersects tube; halting.\n";
            return 2;
        }

        x_km1 = x_k;
        x_k = res.next_frame;
        x_k.save_obj(frame_path(opts.out_dir, frame));
        log_frame(frame, x_km1, x_k, res.newton_iters, res.converged);

        std::cout << "  frame " << frame
                  << "  newton_iters=" << res.newton_iters
                  << (res.converged ? "" : " (NOT converged)")
                  << "  centroid_x=" << x_k.V.col(0).mean()
                  << "  min_phi=" << min_signed_distance(x_k, tube) << "\n";
    }

    std::cout << "demo_phase3_ball_tube: wrote " << opts.num_frames
              << " frames to " << opts.out_dir << "\n";
    return 0;
}
