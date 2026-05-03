// Validation for the high-order H^s operator B in HsPreconditioner.cpp.
// The test includes the implementation file so it can exercise the internal
// matrix-free HsOperator without making that type public API.
//
// RSu Sec. 3.2.1 Eq. 12 discretizes the high-order term B as
//
//   sum_{S != T} <Df u(S)-Df u(T), Df v(S)-Df v(T)>
//                * area(S) area(T) / |X(S)-X(T)|^(2 sigma + 2),
//
// where RSu Sec. 2.4 sets sigma = s - 1 for the gradient-elevated
// high-order term. On a surface, the distance exponent is therefore
// 2(s - 1) + 2 = 2s. This test uses the same ordered face-pair convention as
// the Phase 1 TPE code and validates the matvec through v^T (B u).

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "TestMeshes.h"
#include "../src/core/HsPreconditioner.cpp"

namespace {

int failures = 0;

void check(bool cond, const std::string &msg) {
    if (cond) {
        std::cout << "  [ok] " << msg << "\n";
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failures;
    }
}

double rel_err(double a, double b) {
    return std::abs(a - b) / std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

rsh::MeshData make_parallel_triangles(double d) {
    rsh::MeshData mesh;
    mesh.V.resize(6, 3);
    mesh.V << 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, d,
              1.0, 0.0, d,
              0.0, 1.0, d;
    mesh.F.resize(2, 3);
    mesh.F << 0, 1, 2,
              3, 4, 5;
    mesh.L0 = 0.0;
    return mesh;
}

Eigen::Vector3d face_grad(const rsh::MeshData &mesh,
                          const rsh::FaceGeom &g,
                          const Eigen::VectorXd &u,
                          int f) {
    const int i0 = mesh.F(f, 0);
    const int i1 = mesh.F(f, 1);
    const int i2 = mesh.F(f, 2);

    const Eigen::Vector3d v0 = mesh.V.row(i0).transpose();
    const Eigen::Vector3d v1 = mesh.V.row(i1).transpose();
    const Eigen::Vector3d v2 = mesh.V.row(i2).transpose();

    Eigen::Vector3d E0;
    Eigen::Vector3d E1;
    Eigen::Vector3d E2;
    rsh::opposite_edges(v0, v1, v2, E0, E1, E2);

    const Eigen::Vector3d n = g.N.row(f).transpose();
    return (u(i0) * n.cross(E0) +
            u(i1) * n.cross(E1) +
            u(i2) * n.cross(E2)) / (2.0 * g.A(f));
}

void scatter_face_adjoint(const rsh::MeshData &mesh,
                          const rsh::FaceGeom &g,
                          int f,
                          const Eigen::Vector3d &dual,
                          Eigen::VectorXd &y) {
    const int i0 = mesh.F(f, 0);
    const int i1 = mesh.F(f, 1);
    const int i2 = mesh.F(f, 2);

    const Eigen::Vector3d v0 = mesh.V.row(i0).transpose();
    const Eigen::Vector3d v1 = mesh.V.row(i1).transpose();
    const Eigen::Vector3d v2 = mesh.V.row(i2).transpose();

    Eigen::Vector3d E0;
    Eigen::Vector3d E1;
    Eigen::Vector3d E2;
    rsh::opposite_edges(v0, v1, v2, E0, E1, E2);

    const Eigen::Vector3d n = g.N.row(f).transpose();
    const double inv_2a = 1.0 / (2.0 * g.A(f));
    y(i0) += dual.dot(inv_2a * n.cross(E0));
    y(i1) += dual.dot(inv_2a * n.cross(E1));
    y(i2) += dual.dot(inv_2a * n.cross(E2));
}

Eigen::VectorXd apply_hs_operator(const rsh::MeshData &mesh,
                                  const Eigen::VectorXd &u,
                                  double s,
                                  double theta,
                                  double mass_weight = 0.0) {
    rsh::HsPreconditionerParams params;
    params.s = s;
    params.sigma = 1.0;
    params.mass_weight = mass_weight;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);

    const rsh::HsOperator op(mesh, params, hs, geom, bvh, bp);
    return op * u;
}

Eigen::VectorXd brute_high_order_b(const rsh::MeshData &mesh,
                                   const Eigen::VectorXd &u,
                                   double s) {
    const int nf = mesh.n_faces();
    const int nv = mesh.n_vertices();
    const rsh::FaceGeom g = rsh::compute_face_geom(mesh);

    std::vector<Eigen::Vector3d> gu(static_cast<size_t>(nf));
    for (int f = 0; f < nf; ++f) {
        gu[static_cast<size_t>(f)] = face_grad(mesh, g, u, f);
    }

    std::vector<Eigen::Vector3d> dual(
        static_cast<size_t>(nf), Eigen::Vector3d::Zero());
    for (int a = 0; a < nf; ++a) {
        for (int b = 0; b < nf; ++b) {
            if (a == b) continue;
            const double r2 = (g.C.row(a) - g.C.row(b)).squaredNorm();
            if (r2 == 0.0) continue;
            const double K = 1.0 / std::pow(r2, s);
            dual[static_cast<size_t>(a)] +=
                2.0 * g.A(a) * g.A(b) * K *
                (gu[static_cast<size_t>(a)] - gu[static_cast<size_t>(b)]);
        }
    }

    Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
    for (int f = 0; f < nf; ++f) {
        scatter_face_adjoint(mesh, g, f, dual[static_cast<size_t>(f)], y);
    }
    return y;
}

double slope_log_log(const std::vector<double> &x,
                     const std::vector<double> &y) {
    const int n = static_cast<int>(x.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double lx = std::log(x[i]);
        const double ly = std::log(y[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

Eigen::VectorXd deterministic_field(int n, double phase) {
    Eigen::VectorXd out(n);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1);
        out(i) = std::sin(0.37 * t + phase) + 0.25 * std::cos(0.11 * t);
    }
    return out;
}

void test_two_triangle_hand_case() {
    std::cout << "-- 2-triangle hand-computed high-order B --\n";
    const double s = 5.0 / 3.0;
    const double d = 4.0;
    const rsh::MeshData mesh = make_parallel_triangles(d);

    // Face 0 has u = x, hence Df u = (1,0,0). Face 1 is constant 1/3,
    // hence Df u = 0. With ordered face-pair convention:
    //
    //   <B u,u> = 2 * area0 * area1 * |(1,0,0)-0|^2 / d^(2s)
    //           = 0.5 / d^(2s).
    Eigen::VectorXd u(6);
    u << 0.0, 1.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;

    const Eigen::VectorXd y = apply_hs_operator(mesh, u, s, 0.0);
    const double actual = u.dot(y);
    const double expected = 0.5 / std::pow(d, 2.0 * s);
    const double err = rel_err(actual, expected);

    std::cout << "    d = " << d << "\n";
    std::cout << "    actual <B u,u>   = " << actual << "\n";
    std::cout << "    expected <B u,u> = " << expected << "\n";
    std::cout << "    rel err          = " << err << "\n";
    check(err < 1e-12, "operator matches the hand-computed Eq. 12 value");
}

void test_distance_sweep() {
    std::cout << "-- distance sweep for exponent -2s --\n";
    const double s = 5.0 / 3.0;
    std::vector<double> distances = {1.0, 2.0, 4.0, 8.0, 16.0};
    std::vector<double> values;
    values.reserve(distances.size());

    std::cout << "    d, actual_<B u,u>, expected\n";
    for (double d : distances) {
        const rsh::MeshData mesh = make_parallel_triangles(d);
        Eigen::VectorXd u(6);
        u << 0.0, 1.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;

        const Eigen::VectorXd y = apply_hs_operator(mesh, u, s, 0.0);
        const double actual = u.dot(y);
        const double expected = 0.5 / std::pow(d, 2.0 * s);
        values.push_back(actual);
        std::cout << "    " << d << ", " << actual << ", " << expected << "\n";
        check(rel_err(actual, expected) < 1e-12,
              "sweep value matches 0.5 / d^(2s)");
    }

    const double slope = slope_log_log(distances, values);
    std::cout << "    slope log(<B u,u>) vs log(d) = " << slope << "\n";
    check(std::abs(slope + 2.0 * s) < 1e-12,
          "distance sweep identifies exponent -2s");
}

void test_linearity() {
    std::cout << "-- linearity on icosphere(2) --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;

    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.1);
    const Eigen::VectorXd y = deterministic_field(mesh.n_vertices(), 0.9);
    const double alpha = 1.7;
    const double beta = -0.4;

    const Eigen::VectorXd lhs =
        apply_hs_operator(mesh, alpha * x + beta * y, s, 0.5);
    const Eigen::VectorXd rhs =
        alpha * apply_hs_operator(mesh, x, s, 0.5) +
        beta * apply_hs_operator(mesh, y, s, 0.5);

    const double err = (lhs - rhs).norm() / std::max(1.0, rhs.norm());
    std::cout << "    relative linearity error = " << err << "\n";
    check(err < 1e-12, "B(alpha*x + beta*y) == alpha*Bx + beta*By");
}

void test_symmetry() {
    std::cout << "-- symmetry proxy --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;

    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.2);
    const Eigen::VectorXd y = deterministic_field(mesh.n_vertices(), 1.3);

    for (double theta : {0.0, 0.5}) {
        const double xBy = x.dot(apply_hs_operator(mesh, y, s, theta));
        const double yBx = y.dot(apply_hs_operator(mesh, x, s, theta));
        const double denom = std::max(1.0, std::max(std::abs(xBy), std::abs(yBx)));
        const double err = std::abs(xBy - yBx) / denom;
        const double gate = (theta == 0.0) ? 1e-12 : 1e-2;
        std::cout << "    theta = " << theta << ", x^T B y = " << xBy
                  << ", y^T B x = " << yBx << ", rel = " << err << "\n";
        check(err < gate, "x^T B y ~= y^T B x");
    }
}

void test_brute_reference() {
    std::cout << "-- theta=0 hierarchical matvec matches brute B --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.4);

    const Eigen::VectorXd fast = apply_hs_operator(mesh, x, s, 0.0);
    const Eigen::VectorXd brute = brute_high_order_b(mesh, x, s);
    const double err = (fast - brute).norm() / std::max(1.0, brute.norm());
    std::cout << "    relative ||fast - brute|| = " << err << "\n";
    check(err < 1e-12, "theta=0 HsOperator matches O(n_f^2) brute B");
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "=== Hs high-order B validation ===\n";

    test_two_triangle_hand_case();
    test_distance_sweep();
    test_linearity();
    test_symmetry();
    test_brute_reference();

    std::cout << "\n"
              << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures))
              << "\n";
    return failures == 0 ? 0 : 1;
}
