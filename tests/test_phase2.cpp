#include "GradCheck.h"
#include "MeshData.h"
#include "TestMeshes.h"
#include "PathEnergy.h"
#include "TrustRegionSolver.h"
#include "ExtrapolationSolver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace rsh;

namespace {

int failures = 0;

void check(bool cond, const std::string &name) {
    if (cond) {
        std::cout << "  [ok] " << name << "\n";
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        ++failures;
    }
}

void test_interpolation_decreases_energy() {
    std::cout << "-- trust-region interpolation decreases path energy --\n";
    MeshData x0 = make_icosphere(1);
    x0.normalize();
    MeshData x3 = x0;
    x3.V.col(0) *= 1.15;
    x3.V.col(1) *= 0.9;

    std::vector<MeshData> frames(4, x0);
    frames[3] = x3;
    // piecewise-constant initialization of interior frames
    frames[1] = x0;
    frames[2] = x3;

    PathEnergyParams ep;
    ep.tpe_adaptive.enabled = true;
    ep.tpe_adaptive.max_depth = 2;

    TrustRegionParams tp;
    tp.max_iters = 8;
    tp.max_cg_iters = 25;
    tp.initial_radius = 1e-2;
    tp.max_radius = 5e-2;

    const double e0 = path_energy(frames, ep).terms.total;
    const TrustRegionResult res =
        interpolate_geodesic_trust_region(frames, ep, tp);
    const double e1 = path_energy(res.frames, ep).terms.total;
    std::cout << "    E0 = " << e0
              << ", E1 = " << e1
              << ", accepted = " << res.accepted_steps
              << ", iters = " << res.outer_iterations << "\n";
    check(e1 <= e0 + 1e-10, "trust-region does not increase objective");
}

void test_extrapolation_stability() {
    std::cout << "-- extrapolation stability (constant velocity) --\n";
    MeshData x0 = make_icosphere(1);
    x0.normalize();
    
    // Constant translation
    MeshData x1 = x0;
    Eigen::RowVector3d vel(0.1, 0.05, -0.02);
    x1.V.rowwise() += vel;
    
    PathEnergyParams ep;
    // disable TPE for pure translation test to perfectly preserve velocity
    ep.tpe_alpha = 0.0;
    
    ExtrapolationParams ext_p;
    ext_p.max_newton_iters = 5;
    
    ExtrapolationResult res = extrapolate_geodesic(x0, x1, ep, ext_p);
    
    MeshData x2_expected = x1;
    x2_expected.V.rowwise() += vel;
    
    double max_err = (res.next_frame.V - x2_expected.V).cwiseAbs().maxCoeff();
    std::cout << "    max_err = " << max_err << "\n";
    check(max_err < 1e-5, "Extrapolation follows constant velocity for pure translation without repulsion");
}

void test_extrapolation_with_repulsion() {
    std::cout << "-- extrapolation with repulsion (parallel plates) --\n";
    // Two parallel triangles at z=0 and z=0.5, moving towards each other.
    MeshData m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           0, 0, 0.5,
           1, 0, 0.5,
           0, 1, 0.5;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           3, 4, 5;
           
    MeshData m_km1 = m;
    m_km1.V(3, 2) = 0.6; // Triangle 2 further away in past
    m_km1.V(4, 2) = 0.6;
    m_km1.V(5, 2) = 0.6;
    
    MeshData m_k = m; // Triangle 2 at z=0.5
    
    PathEnergyParams ep;
    ep.tpe_alpha = 6.0;
    ep.tpe_theta = 0.5;
    ep.tpe_adaptive.enabled = true;
    ep.tpe_adaptive.max_depth = 2;
    
    ExtrapolationParams ext_p;
    
    ExtrapolationResult res = extrapolate_geodesic(m_km1, m_k, ep, ext_p);
    
    // Constant velocity would put triangle 2 at z=0.4.
    // Repulsion should keep it further away than 0.4.
    double z_final = res.next_frame.V(3, 2);
    std::cout << "    z_km1=0.6, z_k=0.5, z_kp1_linear=0.4, z_kp1_actual=" << z_final << "\n";
    check(z_final > 0.4 + 1e-5, "Extrapolation slows down/deflects due to repulsion");
}

} // namespace

int main() {
    std::cout << "=== Phase 2/3 smoke tests ===\n";
    test_interpolation_decreases_energy();
    test_extrapolation_stability();
    test_extrapolation_with_repulsion();

    std::cout << "\n" << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
