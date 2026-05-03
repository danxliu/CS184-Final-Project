// Phase 2.13 smoke --- genus-0 canonical embedding with H^s descent.
//
// This is intentionally only the easiest canonical-embedding case: a noisy
// icosphere(2) minimized under TPE alone. It uses the approved H^s
// B+B_0+sandwich machinery, removes the uniform scale mode, and normalizes
// trial meshes inside Armijo backtracking so the TPE scale covariance cannot
// fake energy decrease by inflation.
//
// Outputs: out/canonical_genus0/frame_XXXX.obj and energy.csv.

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "HsPreconditioner.h"
#include "MeshData.h"
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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

using rsh::MeshData;

namespace {

std::string frame_path(const std::string &dir, int idx) {
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx << ".obj";
    return oss.str();
}

void remove_stale_frames(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("frame_", 0) == 0 && entry.path().extension() == ".obj") {
            std::filesystem::remove(entry.path());
        }
    }
}

double parse_initial_tau(int argc, char **argv) {
    double initial_tau = 1.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--initial-tau" || arg == "--tau0") {
            if (i + 1 >= argc) {
                throw std::runtime_error(arg + " requires a numeric value");
            }
            initial_tau = std::stod(argv[++i]);
        } else if (arg.rfind("--initial-tau=", 0) == 0) {
            initial_tau = std::stod(arg.substr(std::string("--initial-tau=").size()));
        } else if (arg.rfind("--tau0=", 0) == 0) {
            initial_tau = std::stod(arg.substr(std::string("--tau0=").size()));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (!std::isfinite(initial_tau) || initial_tau <= 0.0) {
        throw std::runtime_error("--initial-tau must be positive and finite");
    }
    return initial_tau;
}

Eigen::Vector3d bbox_extents(const MeshData &mesh) {
    const Eigen::Vector3d mn = mesh.V.colwise().minCoeff();
    const Eigen::Vector3d mx = mesh.V.colwise().maxCoeff();
    return (mx - mn).eval();
}

double bbox_diagonal(const MeshData &mesh) {
    return bbox_extents(mesh).norm();
}

void remove_scale_mode(const Eigen::MatrixXd &V, Eigen::MatrixXd &G) {
    const Eigen::RowVector3d c = V.colwise().mean();
    const Eigen::MatrixXd R = V.rowwise() - c;
    const double den = R.squaredNorm();
    if (den > 0.0) {
        const double num = R.cwiseProduct(G).sum();
        G -= (num / den) * R;
    }
}

void project_scale_direction(const Eigen::MatrixXd &V, Eigen::MatrixXd &D) {
    const Eigen::RowVector3d c = V.colwise().mean();
    const Eigen::MatrixXd R = V.rowwise() - c;
    const double den = R.squaredNorm();
    if (den > 0.0) {
        const double num = R.cwiseProduct(D).sum();
        D -= (num / den) * R;
    }
}

void perturb_vertices(MeshData &mesh) {
    const double sigma = 0.05 * bbox_diagonal(mesh);
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, sigma);
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        for (int j = 0; j < 3; ++j) {
            mesh.V(i, j) += normal(rng);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    const double initial_tau = parse_initial_tau(argc, argv);

    const int ico_subdiv = 2;
    const double alpha = 6.0;
    const double tpe_theta = 0.0;
    const int max_iters = 200;
    const double armijo_c1 = 1e-4;
    const double shrink = 0.5;
    const int max_backtracks = 60;
    const double hs_grad_tol = 1e-3;

    rsh::HsPreconditionerParams hs_params;
    hs_params.s = 5.0 / 3.0;
    hs_params.sigma = 1.0;
    hs_params.mass_weight = 0.0;

    const std::string out_dir = "out/canonical_genus0";
    std::filesystem::create_directories(out_dir);
    remove_stale_frames(out_dir);

    MeshData m = rsh::make_icosphere(ico_subdiv);
    perturb_vertices(m);
    m.normalize();

    const Eigen::Vector3d ext0 = bbox_extents(m);
    std::cout << "Canonical genus-0 smoke --- noisy icosphere(" << ico_subdiv << ")\n"
              << "  n_v = " << m.n_vertices()
              << ", n_f = " << m.n_faces() << "\n"
              << "  alpha = " << alpha << ", TPE theta = " << tpe_theta << "\n"
              << "  Hs s = " << hs_params.s
              << ", sigma = " << hs_params.sigma
              << ", theta = " << hs_params.theta << "\n"
              << "  initial Armijo trial tau = " << initial_tau << "\n"
              << "  initial bbox extents: (" << ext0(0) << ", "
              << ext0(1) << ", " << ext0(2) << ")\n"
              << "  output: " << out_dir << "/\n\n";

    std::ofstream csv(out_dir + "/energy.csv");
    csv << "iter,energy,grad_norm,hs_grad_norm,step_size,n_backtracks,g_dot_dir,used_identity_fallback\n";

    m.save_obj(frame_path(out_dir, 0));

    double tau = initial_tau;
    bool stopped_by_tol = false;
    bool stopped_by_armijo = false;
    double final_energy = std::numeric_limits<double>::quiet_NaN();
    double final_hs_norm = std::numeric_limits<double>::quiet_NaN();

    for (int it = 0; it < max_iters; ++it) {
        const rsh::FaceGeom g = rsh::compute_face_geom(m);
        const rsh::BVH bvh = rsh::build_bvh(m, g);
        const rsh::BlockPairs bp = rsh::build_bct_self(bvh, tpe_theta);

        const double E = rsh::tpe_energy_bh(g, bvh, bp, alpha);
        Eigen::MatrixXd G = rsh::tpe_gradient_bh(m, g, bvh, bp, alpha);
        remove_scale_mode(m.V, G);
        const double gnorm = G.norm();

        rsh::HsDirectionResult hs = rsh::hs_preconditioned_direction(m, G, hs_params);
        Eigen::MatrixXd dir = hs.direction;
        project_scale_direction(m.V, dir);
        const double g_dot_dir = (G.array() * dir.array()).sum();
        if (!std::isfinite(g_dot_dir) || g_dot_dir <= 0.0) {
            std::cout << "  projected Hs direction became non-descent at iter "
                      << it << " (g.d = " << g_dot_dir << ") --- stopping\n";
            final_energy = E;
            final_hs_norm = std::numeric_limits<double>::quiet_NaN();
            break;
        }

        const double hs_norm = std::sqrt(g_dot_dir);
        final_energy = E;
        final_hs_norm = hs_norm;
        if (hs_norm < hs_grad_tol) {
            std::cout << "  Hs gradient norm below tol --- stopped at iter " << it << "\n";
            csv << it << "," << E << "," << gnorm << "," << hs_norm
                << ",0,0," << g_dot_dir << ","
                << (hs.used_identity_fallback ? 1 : 0) << "\n";
            stopped_by_tol = true;
            break;
        }

        tau = std::min(tau * 2.0, initial_tau);
        int n_bt = 0;
        MeshData m_try = m;
        rsh::BVH bvh_try = bvh;
        double E_new = E;
        for (; n_bt < max_backtracks; ++n_bt) {
            m_try.V = m.V - tau * dir;
            m_try.normalize();
            const rsh::FaceGeom g_try = rsh::compute_face_geom(m_try);
            rsh::update_bvh_aggregates(bvh_try, g_try);
            E_new = rsh::tpe_energy_bh(g_try, bvh_try, bp, alpha);
            if (E_new <= E - armijo_c1 * tau * g_dot_dir) break;
            tau *= shrink;
        }

        if (n_bt == max_backtracks) {
            std::cout << "  Armijo failed at iter " << it << " --- stopping\n";
            stopped_by_armijo = true;
            break;
        }

        m = m_try;

        std::printf("iter %3d  E = %.6e  |g| = %.3e  |g|_Hs = %.3e  tau = %.3e  bt = %2d%s\n",
                    it, E, gnorm, hs_norm, tau, n_bt,
                    hs.used_identity_fallback ? "  fallback" : "");
        csv << it << "," << E << "," << gnorm << "," << hs_norm << ","
            << tau << "," << n_bt << "," << g_dot_dir << ","
            << (hs.used_identity_fallback ? 1 : 0) << "\n";
        csv.flush();

        m.save_obj(frame_path(out_dir, it + 1));
    }

    const Eigen::Vector3d extF = bbox_extents(m);
    const double mean_extent = extF.mean();
    const double roundness_spread =
        (extF.maxCoeff() - extF.minCoeff()) / std::max(mean_extent, 1e-16);
    {
        const rsh::FaceGeom g = rsh::compute_face_geom(m);
        const rsh::BVH bvh = rsh::build_bvh(m, g);
        const rsh::BlockPairs bp = rsh::build_bct_self(bvh, tpe_theta);
        final_energy = rsh::tpe_energy_bh(g, bvh, bp, alpha);
        Eigen::MatrixXd G = rsh::tpe_gradient_bh(m, g, bvh, bp, alpha);
        remove_scale_mode(m.V, G);
        rsh::HsDirectionResult hs = rsh::hs_preconditioned_direction(m, G, hs_params);
        Eigen::MatrixXd dir = hs.direction;
        project_scale_direction(m.V, dir);
        const double g_dot_dir = (G.array() * dir.array()).sum();
        final_hs_norm = (std::isfinite(g_dot_dir) && g_dot_dir > 0.0)
            ? std::sqrt(g_dot_dir)
            : std::numeric_limits<double>::quiet_NaN();
    }

    std::cout << "\nfinal energy: " << final_energy
              << "\nfinal Hs gradient norm: " << final_hs_norm
              << "\nfinal bbox extents: (" << extF(0) << ", "
              << extF(1) << ", " << extF(2) << ")"
              << "\nroundness spread: " << roundness_spread
              << "\nstopped by tolerance: " << (stopped_by_tol ? "yes" : "no")
              << "\nstopped by Armijo: " << (stopped_by_armijo ? "yes" : "no")
              << "\nFrames: " << out_dir << "/frame_*.obj"
              << "\nEnergy log: " << out_dir << "/energy.csv\n";
    return 0;
}
