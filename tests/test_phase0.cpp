#include "GradCheck.h"
#include "MeshData.h"
#include "TestMeshes.h"

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

void test_round_trip() {
    std::cout << "-- OpenMesh round-trip --\n";
    MeshData a = make_icosphere(2);
    OMesh om = a.to_openmesh();
    MeshData b = MeshData::from_openmesh(om);
    check(a.V.rows() == b.V.rows(), "vertex count preserved");
    check(a.F.rows() == b.F.rows(), "face count preserved");
    check((a.V - b.V).cwiseAbs().maxCoeff() < 1e-12, "V bit-close after round-trip");
    check((a.F - b.F).cwiseAbs().maxCoeff() == 0, "F identical after round-trip");
}

void test_normalize() {
    std::cout << "-- normalize invariants --\n";
    MeshData m = make_torus(2.5, 0.8, 32, 16);
    m.normalize();
    check(m.centroid().norm() < 1e-12, "centroid at origin");
    check(std::abs(m.bbox_diagonal() - 1.0) < 1e-12, "bbox diagonal == 1");
    check(m.L0 > 0.0, "L0 positive");
    const double L0_recomputed = m.compute_L0();
    check(std::abs(m.L0 - L0_recomputed) < 1e-15, "cached L0 matches recompute");
}

void test_grad_check_quadratic() {
    std::cout << "-- gradient checker on E = 0.5 x^T A x + b^T x --\n";
    const int n = 9;
    Eigen::MatrixXd M = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd A = 0.5 * (M + M.transpose());
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    auto f = [&](const Eigen::VectorXd &x) {
        return 0.5 * x.dot(A * x) + b.dot(x);
    };
    auto grad = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        return A * x + b;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    auto res = finite_diff_gradient_check(f, grad, x);
    std::cout << "    max_abs_err = " << res.max_abs_err
              << "  max_rel_err = " << res.max_rel_err
              << "  worst_index = " << res.worst_index << "\n";
    check(res.pass(1e-6), "FD gradient agrees with analytic (quadratic)");
}

void test_test_meshes() {
    std::cout << "-- test mesh generators --\n";
    for (int k = 0; k <= 3; ++k) {
        MeshData ic = make_icosphere(k);
        const int expected_v = 10 * (1 << (2 * k)) + 2;
        const int expected_f = 20 * (1 << (2 * k));
        check(ic.n_vertices() == expected_v,
              "icosphere(" + std::to_string(k) + ") vertex count");
        check(ic.n_faces() == expected_f,
              "icosphere(" + std::to_string(k) + ") face count");
        const double max_r = ic.V.rowwise().norm().maxCoeff();
        const double min_r = ic.V.rowwise().norm().minCoeff();
        check(std::abs(max_r - 1.0) < 1e-12 && std::abs(min_r - 1.0) < 1e-12,
              "icosphere(" + std::to_string(k) + ") all vertices on unit sphere");
    }

    const int nu = 40, nv = 20;
    MeshData t = make_torus(1.0, 0.3, nu, nv);
    check(t.n_vertices() == nu * nv, "torus vertex count");
    check(t.n_faces() == 2 * nu * nv, "torus face count");

    t.normalize();
    check(std::abs(t.bbox_diagonal() - 1.0) < 1e-12, "torus normalizes");
}

} // namespace

int main() {
    std::cout << "=== Phase 0 smoke tests ===\n";
    test_round_trip();
    test_normalize();
    test_grad_check_quadratic();
    test_test_meshes();

    std::cout << "\n" << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
