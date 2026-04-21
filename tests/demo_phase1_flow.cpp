// Phase 1.8 --- L2 gradient flow sanity demo.
//
// Vanilla gradient descent on the Barnes-Hut TPE, Armijo backtracking, starting
// from an *ellipsoid* built by anisotropic scaling of icosphere(2) (162 verts /
// 320 tris). TPE alone --- no preconditioner, no shell energy, no remeshing
// --- this is the "slow, mesh-dependent" baseline that motivates the H^s
// preconditioner in the next phase.
//
// Why an ellipsoid and not a plain icosphere: the genus-0 TPE minimizer is the
// round sphere, so an icosphere is already near the discrete minimum --- descent
// moves invisibly. An ellipsoid gives descent a clearly visible job: round out.
//
// Why *not* random noise: at alpha = 6, Gaussian vertex noise creates near-
// contact pairs whose |grad E| explodes (~1e12), Armijo shrinks tau far enough
// that tau*G is still O(1), and the first step can mangle the mesh into huge
// degenerate triangles from which descent never recovers.
//
// The anisotropic start is smooth and strictly non-self-intersecting, so the
// kernel is well-conditioned from iter 0; descent gradually rounds the shape
// toward a sphere, then stalls above the round-sphere minimum --- the expected
// L^2 failure mode (RSu Fig. 5) that the H^s preconditioner fixes.
//
// Scale covariance (E(lambda*x) = lambda^(4-alpha) E(x), alpha=6 -> lambda^-2)
// means raw descent would inflate the mesh indefinitely to lower the energy.
// We remove the uniform-radial mode from the gradient before each step and
// renormalize to unit bbox after the step; descent then moves shape, not size.
//
// Outputs: out/phase1_flow/frame_XXXX.obj per iteration, energy.csv with
// (iter, energy, grad_norm, step_size, n_backtracks).

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include "TPE.h"
#include "TestMeshes.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using rsh::MeshData;

namespace {

std::string frame_path(const std::string &dir, int idx) {
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx << ".obj";
    return oss.str();
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

void scale_axes(MeshData &m, double sx, double sy, double sz) {
    m.V.col(0) *= sx;
    m.V.col(1) *= sy;
    m.V.col(2) *= sz;
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    const int ico_subdiv = 2;
    const double sx = 1.7, sy = 0.75, sz = 0.75;
    const double alpha = 6.0;
    const double theta = 0.5;
    const int max_iters = 150;
    const double armijo_c1 = 1e-4;
    const double shrink = 0.5;
    const int max_backtracks = 60;
    const double grad_tol = 1e-6;
    const double rel_progress_tol = 1e-6;
    const int stall_window = 5;

    const std::string out_dir = "out/phase1_flow";
    std::filesystem::create_directories(out_dir);

    MeshData m = rsh::make_icosphere(ico_subdiv);
    scale_axes(m, sx, sy, sz);
    m.normalize();

    auto bbox_extents = [](const MeshData &mesh) {
        const Eigen::Vector3d mn = mesh.V.colwise().minCoeff();
        const Eigen::Vector3d mx = mesh.V.colwise().maxCoeff();
        return (mx - mn).eval();
    };

    const Eigen::Vector3d ext0 = bbox_extents(m);
    std::cout << "Phase 1.8 demo --- L2 gradient flow on ellipsoid (icosphere(" << ico_subdiv << ") scaled)\n"
              << "  n_v = " << m.n_vertices()
              << ", n_f = " << m.n_faces() << "\n"
              << "  axis scales: (" << sx << ", " << sy << ", " << sz << ")\n"
              << "  alpha = " << alpha << ", theta = " << theta << "\n"
              << "  initial bbox extents: (" << ext0(0) << ", "
              << ext0(1) << ", " << ext0(2) << ")\n"
              << "  round sphere target:  (0.5774, 0.5774, 0.5774)\n"
              << "  output: " << out_dir << "/\n\n";

    std::ofstream csv(out_dir + "/energy.csv");
    csv << "iter,energy,grad_norm,step_size,n_backtracks\n";

    m.save_obj(frame_path(out_dir, 0));

    double tau = 1.0;
    double E_stall_ref = std::numeric_limits<double>::infinity();
    int stall_ref_iter = -stall_window;

    for (int it = 0; it < max_iters; ++it) {
        const rsh::FaceGeom g = rsh::compute_face_geom(m);
        const rsh::BVH bvh = rsh::build_bvh(m, g);
        const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);

        const double E = rsh::tpe_energy_bh(g, bvh, bp, alpha);
        Eigen::MatrixXd G = rsh::tpe_gradient_bh(m, g, bvh, bp, alpha);
        remove_scale_mode(m.V, G);
        const double gnorm2 = G.squaredNorm();
        const double gnorm = std::sqrt(gnorm2);

        if (gnorm < grad_tol) {
            std::cout << "  gradient below tol --- stopped at iter " << it << "\n";
            csv << it << "," << E << "," << gnorm << ",0,0\n";
            break;
        }

        tau = std::min(tau * 2.0, 1.0);
        int n_bt = 0;
        MeshData m_try = m;
        rsh::BVH bvh_try = bvh;
        double E_new = E;
        for (; n_bt < max_backtracks; ++n_bt) {
            m_try.V = m.V - tau * G;
            m_try.normalize();
            const rsh::FaceGeom g_try = rsh::compute_face_geom(m_try);
            rsh::update_bvh_aggregates(bvh_try, g_try);
            E_new = rsh::tpe_energy_bh(g_try, bvh_try, bp, alpha);
            if (E_new <= E - armijo_c1 * tau * gnorm2) break;
            tau *= shrink;
        }

        if (n_bt == max_backtracks) {
            std::cout << "  Armijo failed at iter " << it << " --- stopping\n";
            break;
        }

        m = m_try;

        std::printf("iter %3d  E = %.6e  |g| = %.3e  tau = %.3e  bt = %2d\n",
                    it, E, gnorm, tau, n_bt);
        csv << it << "," << E << "," << gnorm << "," << tau << "," << n_bt << "\n";
        csv.flush();

        m.save_obj(frame_path(out_dir, it + 1));

        if (it - stall_ref_iter >= stall_window) {
            const double rel = (E_stall_ref - E_new) / std::max(std::abs(E_stall_ref), 1.0);
            if (rel < rel_progress_tol) {
                std::cout << "  relative progress < " << rel_progress_tol
                          << " over " << stall_window
                          << " iters --- stopping (L2 stalled; see next phase's preconditioner)\n";
                break;
            }
            E_stall_ref = E_new;
            stall_ref_iter = it;
        }
    }

    const Eigen::Vector3d extF = bbox_extents(m);
    std::cout << "\nfinal bbox extents:   (" << extF(0) << ", "
              << extF(1) << ", " << extF(2) << ")\n"
              << "Frames: " << out_dir << "/frame_*.obj\n"
              << "Energy log: " << out_dir << "/energy.csv\n";
    return 0;
}
