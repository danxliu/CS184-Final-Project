#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "HsPreconditioner.h"
#include "MeshData.h"
#include "ShellEnergy.h"
#include "TPE.h"
#include "TestMeshes.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
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
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx << ".obj";
    return oss.str();
}

std::string partial_frame_path(const std::string &dir, int frame) {
    std::ostringstream oss;
    oss << dir << "/partial_frame_" << std::setfill('0') << std::setw(4)
        << frame << ".obj";
    return oss.str();
}

void remove_stale_frames(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if ((name.rfind("frame_", 0) == 0 ||
             name.rfind("partial_frame_", 0) == 0) &&
            entry.path().extension() == ".obj") {
            std::filesystem::remove(entry.path());
        }
    }
}

struct CliOptions {
    double initial_tau = 1.0;
    std::vector<std::string> mesh_paths;
};

CliOptions parse_cli(int argc, char **argv) {
    CliOptions out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--initial-tau" || arg == "--tau0") {
            if (i + 1 >= argc) {
                throw std::runtime_error(arg + " requires a numeric value");
            }
            out.initial_tau = std::stod(argv[++i]);
        } else if (arg.rfind("--initial-tau=", 0) == 0) {
            out.initial_tau = std::stod(arg.substr(std::string("--initial-tau=").size()));
        } else if (arg.rfind("--tau0=", 0) == 0) {
            out.initial_tau = std::stod(arg.substr(std::string("--tau0=").size()));
        } else {
            out.mesh_paths.push_back(arg);
        }
    }

    if (!std::isfinite(out.initial_tau) || out.initial_tau <= 0.0) {
        throw std::runtime_error("--initial-tau must be positive and finite");
    }
    if (!out.mesh_paths.empty() && out.mesh_paths.size() != 2) {
        throw std::runtime_error(
            "expected either zero mesh paths or exactly two mesh paths");
    }
    return out;
}

MeshData combine_meshes(const MeshData &m1, const MeshData &m2) {
    MeshData out;
    int v1 = m1.n_vertices();
    int v2 = m2.n_vertices();
    int f1 = m1.n_faces();
    int f2 = m2.n_faces();

    out.V.resize(v1 + v2, 3);
    out.V.topRows(v1) = m1.V;
    out.V.bottomRows(v2) = m2.V;

    out.F.resize(f1 + f2, 3);
    out.F.topRows(f1) = m1.F;
    out.F.bottomRows(f2) = m2.F.array() + v1;

    return out;
}

} // namespace

int main(int argc, char **argv) {
    const CliOptions cli = parse_cli(argc, argv);
    MeshData m_left, m_right;

    if (cli.mesh_paths.size() == 2) {
        std::cout << "Loading custom meshes: " << cli.mesh_paths[0]
                  << " and " << cli.mesh_paths[1] << "\n";
        m_left = rsh::MeshData::load_obj(cli.mesh_paths[0]);
        m_right = rsh::MeshData::load_obj(cli.mesh_paths[1]);
    } else {
        std::cout << "Using default icosphere(2) meshes.\n";
        m_left = rsh::make_icosphere(2);
        m_right = rsh::make_icosphere(2);
        
        // Scale down the icospheres slightly if they are too big
        m_left.V *= 0.5;
        m_right.V *= 0.5;
    }

    // Ensure meshes are separated on the X axis.
    // Calculate bounding boxes
    double left_max_x = m_left.V.col(0).maxCoeff();
    double left_min_x = m_left.V.col(0).minCoeff();
    double right_min_x = m_right.V.col(0).minCoeff();
    double right_max_x = m_right.V.col(0).maxCoeff();

    // Center them on Y and Z
    Eigen::RowVector3d c_left = m_left.V.colwise().mean();
    Eigen::RowVector3d c_right = m_right.V.colwise().mean();
    m_left.V.col(1).array() -= c_left(1);
    m_left.V.col(2).array() -= c_left(2);
    m_right.V.col(1).array() -= c_right(1);
    m_right.V.col(2).array() -= c_right(2);

    // Translate so they face each other
    // Left mesh ends at X = -0.1, right mesh starts at X = 0.1
    double target_left_max_x = -0.1;
    double target_right_min_x = 0.1;

    m_left.V.col(0).array() += (target_left_max_x - left_max_x);
    m_right.V.col(0).array() += (target_right_min_x - right_min_x);

    // Recompute exact bounds
    left_min_x = m_left.V.col(0).minCoeff();
    right_max_x = m_right.V.col(0).maxCoeff();

    // Combine meshes into the reference state
    MeshData m_ref = combine_meshes(m_left, m_right);

    // Identify handle vertices
    double epsilon = 0.1;
    std::vector<bool> is_left_handle(m_ref.n_vertices(), false);
    std::vector<bool> is_right_handle(m_ref.n_vertices(), false);
    int num_left_handles = 0;
    int num_right_handles = 0;

    for (int i = 0; i < m_ref.n_vertices(); ++i) {
        double x = m_ref.V(i, 0);
        if (x <= left_min_x + epsilon) {
            is_left_handle[i] = true;
            num_left_handles++;
        } else if (x >= right_max_x - epsilon) {
            is_right_handle[i] = true;
            num_right_handles++;
        }
    }

    std::cout << "Left handles: " << num_left_handles << ", Right handles: " << num_right_handles << "\n";

    // Simulation parameters
    int num_frames = 40;
    double dx_per_frame = 0.025; // Move each side inwards by 0.025 per frame
    
    // Optimization parameters
    const double alpha = 6.0;
    const double theta = 0.5;
    const int max_inner_iters = 100;
    const double armijo_c1 = 1e-4;
    const double shrink = 0.5;
    const int max_backtracks = 60;
    const double grad_tol = 1e-4;
    const double tpe_weight = 0.005;

    rsh::ShellEnergyParams shell_params;
    shell_params.thickness = 0.005;
    shell_params.lambda = 1.0;
    shell_params.mu = 1.0;

    rsh::HsPreconditionerParams hs_params;
    hs_params.s = 1.5;
    hs_params.sigma = 1.0;
    hs_params.mass_weight = 1.0;

    const std::string out_dir = "out/compress_sequence";
    std::filesystem::create_directories(out_dir);
    remove_stale_frames(out_dir);

    MeshData m_curr = m_ref;
    m_curr.save_obj(frame_path(out_dir, 0));

    std::ofstream csv(out_dir + "/energy.csv");
    csv << "frame,inner,energy,tpe_energy_weighted,shell_energy,grad_norm,step_size,n_backtracks,g_dot_dir,used_identity_fallback\n";
    std::ofstream frame_csv(out_dir + "/frame_summary.csv");
    frame_csv << "frame,status,initial_energy,final_energy,accepted_steps,safeguard_fallbacks,total_backtracks\n";

    std::cout << "Starting compression simulation...\n"
              << "Initial Armijo trial tau: " << cli.initial_tau << "\n";

    for (int frame = 1; frame <= num_frames; ++frame) {
        std::cout << "\n--- Frame " << frame << " ---\n";
        
        // 1. Move handles
        for (int i = 0; i < m_curr.n_vertices(); ++i) {
            if (is_left_handle[i]) {
                m_curr.V(i, 0) += dx_per_frame; // Move left handle right
            } else if (is_right_handle[i]) {
                m_curr.V(i, 0) -= dx_per_frame; // Move right handle left
            }
        }

        // 2. Relax free vertices
        double tau = cli.initial_tau;
        double frame_initial_energy = std::numeric_limits<double>::quiet_NaN();
        double frame_final_energy = std::numeric_limits<double>::quiet_NaN();
        int frame_fallbacks = 0;
        int frame_backtracks = 0;
        int accepted_steps = 0;
        std::string frame_status = "max_iters";
        
        for (int it = 0; it < max_inner_iters; ++it) {
            // Compute TPE
            const rsh::FaceGeom g = rsh::compute_face_geom(m_curr);
            const rsh::BVH bvh = rsh::build_bvh(m_curr, g);
            const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);

            double E_tpe = rsh::tpe_energy_bh(g, bvh, bp, alpha);
            Eigen::MatrixXd G_tpe = rsh::tpe_gradient_bh(m_curr, g, bvh, bp, alpha);

            // Compute Shell
            rsh::ShellEnergyGradientResult shell_res = rsh::shell_energy_with_gradient(m_ref, m_curr, shell_params);
            
            double E_total = tpe_weight * E_tpe + shell_res.energy.total;
            Eigen::MatrixXd G_total = tpe_weight * G_tpe + shell_res.grad_def;
            if (it == 0) {
                frame_initial_energy = E_total;
            }
            frame_final_energy = E_total;

            // Zero out gradient on handles
            for (int i = 0; i < m_curr.n_vertices(); ++i) {
                if (is_left_handle[i] || is_right_handle[i]) {
                    G_total.row(i).setZero();
                }
            }

            double gnorm = G_total.norm();
            if (it == 0) {
                std::cout << "  initial inner energy = " << E_total
                          << " (TPE=" << (tpe_weight * E_tpe)
                          << ", Shell=" << shell_res.energy.total
                          << "), |g| = " << gnorm << "\n";
                csv << frame << ",-1," << E_total << ","
                    << (tpe_weight * E_tpe) << "," << shell_res.energy.total
                    << "," << gnorm << ",0,0,0,0\n";
                csv.flush();
            }
            if (gnorm < grad_tol) {
                std::cout << "  [Inner " << it << "] converged! gnorm = " << gnorm << "\n";
                frame_status = "grad_tol";
                break;
            }

            // Direction
            rsh::HsDirectionResult hs_res = rsh::hs_preconditioned_direction(m_curr, G_total, hs_params);
            Eigen::MatrixXd dir = hs_res.direction;
            if (hs_res.used_identity_fallback) {
                ++frame_fallbacks;
            }

            // Zero out direction on handles
            for (int i = 0; i < m_curr.n_vertices(); ++i) {
                if (is_left_handle[i] || is_right_handle[i]) {
                    dir.row(i).setZero();
                }
            }

            // Backtracking
            tau = std::min(tau * 2.0, cli.initial_tau);
            int n_bt = 0;
            MeshData m_try = m_curr;
            rsh::BVH bvh_try = bvh;
            double E_new = E_total;
            
            for (; n_bt < max_backtracks; ++n_bt) {
                m_try.V = m_curr.V - tau * dir;
                
                const rsh::FaceGeom g_try = rsh::compute_face_geom(m_try);
                rsh::update_bvh_aggregates(bvh_try, g_try);
                
                double E_tpe_try = rsh::tpe_energy_bh(g_try, bvh_try, bp, alpha);
                rsh::ShellEnergyValue shell_try = rsh::shell_energy(m_ref, m_try, shell_params);
                
                E_new = tpe_weight * E_tpe_try + shell_try.total;
                
                if (E_new <= E_total - armijo_c1 * tau * hs_res.g_dot_dir) {
                    break;
                }
                tau *= shrink;
            }

            if (n_bt == max_backtracks) {
                std::cout << "  [Inner " << it << "] Armijo failed, gnorm = " << gnorm << "\n";
                frame_status = "armijo_failed";
                break; // proceed to next frame
            }

            m_curr = m_try;
            m_curr.save_obj(partial_frame_path(out_dir, frame));
            frame_final_energy = E_new;
            frame_backtracks += n_bt;
            ++accepted_steps;
            std::printf("  [Inner %2d] E = %.4e (TPE=%.4e, Shell=%.4e) |g| = %.2e tau = %.2e bt = %d\n",
                        it, E_new, tpe_weight * E_tpe, shell_res.energy.total, gnorm, tau, n_bt);
            csv << frame << "," << it << "," << E_new << ","
                << (tpe_weight * E_tpe) << "," << shell_res.energy.total
                << "," << gnorm << "," << tau << "," << n_bt << ","
                << hs_res.g_dot_dir << ","
                << (hs_res.used_identity_fallback ? 1 : 0) << "\n";
            csv.flush();
        }

        m_curr.save_obj(frame_path(out_dir, frame));
        frame_csv << frame << "," << frame_status << ","
                  << frame_initial_energy << "," << frame_final_energy << ","
                  << accepted_steps << "," << frame_fallbacks << ","
                  << frame_backtracks << "\n";
        frame_csv.flush();
        std::cout << "Frame " << frame << " summary: status=" << frame_status
                  << ", initial_E=" << frame_initial_energy
                  << ", final_E=" << frame_final_energy
                  << ", accepted_steps=" << accepted_steps
                  << ", fallbacks=" << frame_fallbacks
                  << ", total_backtracks=" << frame_backtracks << "\n";
    }

    std::cout << "\nCompression demo complete. Output saved to " << out_dir << "/\n";
    return 0;
}
