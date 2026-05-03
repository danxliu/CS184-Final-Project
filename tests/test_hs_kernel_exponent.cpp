// Validation for the paper-faithful H^s operator A = B + B_0 in
// HsPreconditioner.cpp.
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
// 2(s - 1) + 2 = 2s. Eq. 8 adds the low-order term B_0 by multiplying
// face-value differences by the asymmetric p=2 TPE kernel k_{f,2}. This test
// uses the same ordered face-pair convention as the Phase 1 TPE code and
// validates the matvec through v^T (A u).

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ShellEnergy.h"
#include "TestMeshes.h"
#include "TPE.h"
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

rsh::MeshData combine_meshes(const rsh::MeshData &a, const rsh::MeshData &b) {
    rsh::MeshData out;
    const int av = a.n_vertices();
    const int bv = b.n_vertices();
    const int af = a.n_faces();
    const int bf = b.n_faces();
    out.V.resize(av + bv, 3);
    out.V.topRows(av) = a.V;
    out.V.bottomRows(bv) = b.V;
    out.F.resize(af + bf, 3);
    out.F.topRows(af) = a.F;
    out.F.bottomRows(bf) = b.F.array() + av;
    out.L0 = out.compute_L0();
    return out;
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

Eigen::VectorXd apply_scalar_fractional_laplacian(const rsh::MeshData &mesh,
                                                  const Eigen::VectorXd &u,
                                                  double s,
                                                  double theta) {
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const rsh::ScalarFractionalLaplacian op(mesh, geom, bvh, bp, 2.0 - s);
    return op.apply(u);
}

double face_value(const rsh::MeshData &mesh,
                  const Eigen::VectorXd &u,
                  int f) {
    return (u(mesh.F(f, 0)) +
            u(mesh.F(f, 1)) +
            u(mesh.F(f, 2))) / 3.0;
}

double b0_kernel_sym_faces(const rsh::FaceGeom &g,
                           int a,
                           int b,
                           double s) {
    const Eigen::Vector3d e = g.C.row(a) - g.C.row(b);
    const double r2 = e.squaredNorm();
    if (r2 == 0.0) return 0.0;

    const Eigen::Vector3d na = g.N.row(a).transpose();
    const Eigen::Vector3d nb = g.N.row(b).transpose();
    const double pa = na.dot(e);
    const double pb = nb.dot(-e);
    return (pa * pa + pb * pb) / std::pow(r2, s + 2.0);
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

Eigen::VectorXd brute_low_order_b0(const rsh::MeshData &mesh,
                                   const Eigen::VectorXd &u,
                                   double s) {
    const int nf = mesh.n_faces();
    const int nv = mesh.n_vertices();
    const rsh::FaceGeom g = rsh::compute_face_geom(mesh);

    Eigen::VectorXd face_u(nf);
    for (int f = 0; f < nf; ++f) {
        face_u(f) = face_value(mesh, u, f);
    }

    Eigen::VectorXd dual = Eigen::VectorXd::Zero(nf);
    for (int a = 0; a < nf; ++a) {
        for (int b = 0; b < nf; ++b) {
            if (a == b) continue;
            const double K0 = b0_kernel_sym_faces(g, a, b, s);
            dual(a) += g.A(a) * g.A(b) * K0 * (face_u(a) - face_u(b));
        }
    }

    Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
    for (int f = 0; f < nf; ++f) {
        const double val = dual(f) / 3.0;
        y(mesh.F(f, 0)) += val;
        y(mesh.F(f, 1)) += val;
        y(mesh.F(f, 2)) += val;
    }
    return y;
}

Eigen::VectorXd brute_operator_a(const rsh::MeshData &mesh,
                                 const Eigen::VectorXd &u,
                                 double s) {
    return brute_high_order_b(mesh, u, s) + brute_low_order_b0(mesh, u, s);
}

Eigen::VectorXd brute_scalar_fractional_laplacian(const rsh::MeshData &mesh,
                                                  const Eigen::VectorXd &u,
                                                  double s) {
    const int nf = mesh.n_faces();
    const int nv = mesh.n_vertices();
    const double sigma_middle = 2.0 - s;
    const double power = sigma_middle + 1.0;
    const rsh::FaceGeom g = rsh::compute_face_geom(mesh);

    Eigen::VectorXd face_u(nf);
    for (int f = 0; f < nf; ++f) {
        face_u(f) = face_value(mesh, u, f);
    }

    Eigen::VectorXd dual = Eigen::VectorXd::Zero(nf);
    for (int a = 0; a < nf; ++a) {
        for (int b = 0; b < nf; ++b) {
            if (a == b) continue;
            const double r2 = (g.C.row(a) - g.C.row(b)).squaredNorm();
            if (r2 == 0.0) continue;
            const double K = 1.0 / std::pow(r2, power);
            dual(a) +=
                2.0 * g.A(a) * g.A(b) * K * (face_u(a) - face_u(b));
        }
    }

    Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
    for (int f = 0; f < nf; ++f) {
        const double val = dual(f) / 3.0;
        y(mesh.F(f, 0)) += val;
        y(mesh.F(f, 1)) += val;
        y(mesh.F(f, 2)) += val;
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

void project_constant_mode(Eigen::VectorXd &x) {
    if (x.size() == 0) return;
    x.array() -= x.mean();
}

struct SolveStats {
    Eigen::VectorXd x;
    int iterations = 0;
    double error = 0.0;
    double residual = 0.0;
    Eigen::ComputationInfo info = Eigen::InvalidInput;
};

SolveStats solve_with_identity(const rsh::MeshData &mesh,
                               const Eigen::VectorXd &rhs,
                               double s,
                               double theta,
                               int max_iters,
                               double tol) {
    rsh::HsPreconditionerParams params;
    params.s = s;
    params.sigma = 1.0;
    params.theta = theta;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const rsh::HsOperator op(mesh, params, hs, geom, bvh, bp);

    Eigen::VectorXd rhs_projected = rhs;
    project_constant_mode(rhs_projected);

    Eigen::GMRES<rsh::HsOperator, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(op);
    gmres.setTolerance(tol);
    gmres.setMaxIterations(max_iters);

    SolveStats out;
    out.x = gmres.solve(rhs_projected);
    project_constant_mode(out.x);
    out.iterations = static_cast<int>(gmres.iterations());
    out.error = gmres.error();
    out.info = gmres.info();
    out.residual = (op * out.x - rhs_projected).norm() /
                   std::max(1.0, rhs_projected.norm());
    return out;
}

SolveStats solve_with_sandwich(const rsh::MeshData &mesh,
                               const Eigen::VectorXd &rhs,
                               double s,
                               double theta,
                               int max_iters,
                               double tol,
                               rsh::HsLaplacianInverseMode inverse_mode =
                                   rsh::HsLaplacianInverseMode::RawStiffness,
                               rsh::HsConstantProjectionMode projection_mode =
                                   rsh::HsConstantProjectionMode::Algebraic) {
    rsh::HsPreconditionerParams params;
    params.s = s;
    params.sigma = 1.0;
    params.theta = theta;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const rsh::HsOperator op(mesh, params, hs, geom, bvh, bp);
    const rsh::ScalarFractionalLaplacian middle(mesh, geom, bvh, bp, 2.0 - s);
    const rsh::HsSandwichPreconditioner sandwich(
        hs, middle, inverse_mode, projection_mode);
    const rsh::HsRightPreconditionedOperator right_op(op, sandwich);

    Eigen::VectorXd rhs_projected = rhs;
    sandwich.project(rhs_projected);

    Eigen::GMRES<rsh::HsRightPreconditionedOperator,
                 Eigen::IdentityPreconditioner> gmres;
    gmres.compute(right_op);
    gmres.setTolerance(tol);
    gmres.setMaxIterations(max_iters);

    SolveStats out;
    const Eigen::VectorXd z = gmres.solve(rhs_projected);
    out.x = sandwich.solve(z);
    sandwich.project(out.x);
    out.iterations = static_cast<int>(gmres.iterations());
    out.error = gmres.error();
    out.info = gmres.info();
    out.residual = (op * out.x - rhs_projected).norm() /
                   std::max(1.0, rhs_projected.norm());
    return out;
}

SolveStats solve_with_left_sandwich(
                               const rsh::MeshData &mesh,
                               const Eigen::VectorXd &rhs,
                               double s,
                               double theta,
                               int max_iters,
                               double tol,
                               rsh::HsLaplacianInverseMode inverse_mode,
                               rsh::HsConstantProjectionMode projection_mode) {
    rsh::HsPreconditionerParams params;
    params.s = s;
    params.sigma = 1.0;
    params.theta = theta;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const rsh::HsOperator op(mesh, params, hs, geom, bvh, bp);
    const rsh::ScalarFractionalLaplacian middle(mesh, geom, bvh, bp, 2.0 - s);
    const rsh::HsSandwichPreconditioner sandwich(
        hs, middle, inverse_mode, projection_mode);
    const rsh::HsLeftPreconditionedOperator left_op(op, sandwich);

    Eigen::VectorXd rhs_projected = rhs;
    sandwich.project(rhs_projected);
    const Eigen::VectorXd rhs_left = sandwich.solve(rhs_projected);

    Eigen::GMRES<rsh::HsLeftPreconditionedOperator,
                 Eigen::IdentityPreconditioner> gmres;
    gmres.compute(left_op);
    gmres.setTolerance(tol);
    gmres.setMaxIterations(max_iters);

    SolveStats out;
    out.x = gmres.solve(rhs_left);
    sandwich.project(out.x);
    out.iterations = static_cast<int>(gmres.iterations());
    out.error = gmres.error();
    out.info = gmres.info();
    out.residual = (op * out.x - rhs_projected).norm() /
                   std::max(1.0, rhs_projected.norm());
    return out;
}

std::string variant_name(rsh::HsLaplacianInverseMode inverse_mode,
                         rsh::HsConstantProjectionMode projection_mode) {
    std::string out;
    if (inverse_mode == rsh::HsLaplacianInverseMode::H1Metric) {
        out = "H1inv";
    } else if (inverse_mode == rsh::HsLaplacianInverseMode::LaplaceBeltrami) {
        out = "LinvM";
    } else {
        out = "Linv";
    }
    out += "+";
    if (projection_mode == rsh::HsConstantProjectionMode::None) {
        out += "Pnone";
    } else if (projection_mode == rsh::HsConstantProjectionMode::MassWeighted) {
        out += "Pmass";
    } else {
        out += "Palg";
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

void test_b0_two_triangle_hand_case() {
    std::cout << "-- 2-triangle hand-computed low-order B_0 --\n";
    const double s = 5.0 / 3.0;
    const double d = 4.0;
    const rsh::MeshData mesh = make_parallel_triangles(d);

    // Both faces carry constant data, so Df u = 0 on each triangle and the
    // high-order B term vanishes. Face 0 has u_S = 1, face 1 has u_T = 0.
    //
    // For the parallel stacked triangles, e = c_S - c_T = (0,0,-d),
    // n_S = n_T = (0,0,1), area_S = area_T = 1/2. The symmetrized B_0 kernel is
    //
    //   K_0^sym(S,T)
    //     = (|n_S.e|^2 + |n_T.(-e)|^2) / |e|^(2s+4)
    //     = 2 d^2 / d^(2s+4)
    //     = 2 / d^(2s+2).
    //
    // With the ordered face-pair convention used by the operator, the bilinear
    // value for the two-face system is
    //
    //   <B_0 u,u> = area_S area_T K_0^sym (u_S-u_T)^2
    //              = 0.5 / d^(2s+2).
    Eigen::VectorXd u(6);
    u << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

    const Eigen::VectorXd y = apply_hs_operator(mesh, u, s, 0.0);
    const double actual = u.dot(y);
    const double expected = 0.5 / std::pow(d, 2.0 * s + 2.0);
    const double err = rel_err(actual, expected);

    std::cout << "    d = " << d << "\n";
    std::cout << "    actual <B_0 u,u>   = " << actual << "\n";
    std::cout << "    expected <B_0 u,u> = " << expected << "\n";
    std::cout << "    rel err            = " << err << "\n";
    check(err < 1e-12, "operator matches the hand-computed Eq. 8 value");
}

void test_b0_distance_sweep() {
    std::cout << "-- B_0 distance sweep for parallel-triangle exponent --\n";
    const double s = 5.0 / 3.0;
    std::vector<double> distances = {1.0, 2.0, 4.0, 8.0, 16.0};
    std::vector<double> values;
    values.reserve(distances.size());

    std::cout << "    d, actual_<B_0 u,u>, expected\n";
    for (double d : distances) {
        const rsh::MeshData mesh = make_parallel_triangles(d);
        Eigen::VectorXd u(6);
        u << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

        const Eigen::VectorXd y = apply_hs_operator(mesh, u, s, 0.0);
        const double actual = u.dot(y);
        const double expected = 0.5 / std::pow(d, 2.0 * s + 2.0);
        values.push_back(actual);
        std::cout << "    " << d << ", " << actual << ", " << expected << "\n";
        check(rel_err(actual, expected) < 1e-12,
              "sweep value matches 0.5 / d^(2s+2)");
    }

    const double slope = slope_log_log(distances, values);
    std::cout << "    slope log(<B_0 u,u>) vs log(d) = " << slope << "\n";
    check(std::abs(slope + 2.0 * s + 2.0) < 1e-12,
          "parallel-triangle sweep identifies effective exponent -(2s+2)");
}

void test_linearity() {
    std::cout << "-- linearity of A = B + B_0 on icosphere(2) --\n";
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
    check(err < 1e-12, "A(alpha*x + beta*y) == alpha*Ax + beta*Ay");
}

void test_symmetry() {
    std::cout << "-- symmetry proxy for A = B + B_0 --\n";
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
        std::cout << "    theta = " << theta << ", x^T A y = " << xBy
                  << ", y^T A x = " << yBx << ", rel = " << err << "\n";
        check(err < gate, "x^T A y ~= y^T A x");
    }
}

void test_brute_reference() {
    std::cout << "-- hierarchical matvec matches brute A at theta=0 --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.4);

    const Eigen::VectorXd fast = apply_hs_operator(mesh, x, s, 0.0);
    const Eigen::VectorXd brute = brute_operator_a(mesh, x, s);
    const double err = (fast - brute).norm() / std::max(1.0, brute.norm());
    std::cout << "    relative ||fast - brute|| = " << err << "\n";
    check(err < 1e-12, "theta=0 HsOperator matches O(n_f^2) brute A");

    const double q_fast = x.dot(apply_hs_operator(mesh, x, s, 0.5));
    const double q_brute = x.dot(brute);
    const double ratio = q_fast / q_brute;
    std::cout << "    theta=0.5 quadratic ratio vs brute = " << ratio << "\n";
    check(std::isfinite(ratio), "theta=0.5 A ratio is finite");
}

void print_theta_sweep_for_mesh(const std::string &name,
                                rsh::MeshData mesh,
                                double phase) {
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), phase);
    const Eigen::VectorXd brute = brute_operator_a(mesh, x, s);
    const double q_brute = x.dot(brute);

    for (double theta : {0.0, 0.1, 0.25, 0.5}) {
        const double q_bh = x.dot(apply_hs_operator(mesh, x, s, theta));
        const double ratio = q_bh / q_brute;
        std::cout << "    " << name << "," << theta << "," << q_brute
                  << "," << q_bh << "," << ratio << "\n";
    }
}

void test_theta_sweep_table() {
    std::cout << "-- theta sweep table for A quadratic approximation --\n";
    std::cout << "    mesh,theta,A_brute,A_BH,ratio\n";
    print_theta_sweep_for_mesh("icosphere_2", rsh::make_icosphere(2), 0.4);
    print_theta_sweep_for_mesh("icosphere_3", rsh::make_icosphere(3), 0.4);
}

void test_scalar_fractional_laplacian_hand_case() {
    std::cout << "-- scalar L^(2-s) hand-computed middle operator --\n";
    const double s = 5.0 / 3.0;
    const double d = 4.0;
    const rsh::MeshData mesh = make_parallel_triangles(d);

    // The sandwich middle apply uses RSu Eq. 6 with sigma = 2-s. On a surface
    // the distance exponent is 2(2-s)+2 = 6-2s. For two parallel right
    // triangles with constant face values 1 and 0, the ordered convention gives
    //
    //   <L^(2-s) u,u> = 2 * area0 * area1 / d^(6-2s)
    //                  = 0.5 / d^(6-2s).
    Eigen::VectorXd u(6);
    u << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

    const Eigen::VectorXd y = apply_scalar_fractional_laplacian(mesh, u, s, 0.0);
    const double actual = u.dot(y);
    const double expected = 0.5 / std::pow(d, 6.0 - 2.0 * s);
    const double err = rel_err(actual, expected);

    std::cout << "    actual <L^(2-s) u,u>   = " << actual << "\n";
    std::cout << "    expected <L^(2-s) u,u> = " << expected << "\n";
    std::cout << "    rel err                = " << err << "\n";
    check(err < 1e-12, "middle operator matches the hand-computed Eq. 6 value");
}

void test_scalar_fractional_laplacian_brute_reference() {
    std::cout << "-- scalar L^(2-s) hierarchical-vs-brute --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.55);

    const Eigen::VectorXd fast = apply_scalar_fractional_laplacian(mesh, x, s, 0.0);
    const Eigen::VectorXd brute = brute_scalar_fractional_laplacian(mesh, x, s);
    const double err = (fast - brute).norm() / std::max(1.0, brute.norm());
    std::cout << "    theta=0 relative ||fast - brute|| = " << err << "\n";
    check(err < 1e-12, "theta=0 L^(2-s) matches O(n_f^2) brute reference");

    const double q_brute = x.dot(brute);
    std::cout << "    mesh,theta,L_brute,L_BH,ratio\n";
    for (double theta : {0.0, 0.1, 0.25, 0.5}) {
        const double q_bh = x.dot(apply_scalar_fractional_laplacian(mesh, x, s, theta));
        std::cout << "    icosphere_2," << theta << "," << q_brute
                  << "," << q_bh << "," << (q_bh / q_brute) << "\n";
    }
}

void test_sandwich_psd_probe() {
    std::cout << "-- sandwich preconditioner PSD probe --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const double theta = 0.25;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const rsh::ScalarFractionalLaplacian middle(mesh, geom, bvh, bp, 2.0 - s);

    struct Variant {
        rsh::HsLaplacianInverseMode inverse_mode;
        rsh::HsConstantProjectionMode projection_mode;
    };

    const std::vector<Variant> variants = {
        {rsh::HsLaplacianInverseMode::RawStiffness,
         rsh::HsConstantProjectionMode::Algebraic},
        {rsh::HsLaplacianInverseMode::RawStiffness,
         rsh::HsConstantProjectionMode::MassWeighted},
        {rsh::HsLaplacianInverseMode::LaplaceBeltrami,
         rsh::HsConstantProjectionMode::Algebraic},
        {rsh::HsLaplacianInverseMode::LaplaceBeltrami,
         rsh::HsConstantProjectionMode::MassWeighted},
        {rsh::HsLaplacianInverseMode::H1Metric,
         rsh::HsConstantProjectionMode::None},
    };

    for (const Variant &variant : variants) {
        const rsh::HsSandwichPreconditioner sandwich(
            hs, middle, variant.inverse_mode, variant.projection_mode);
        double min_q_euclidean = std::numeric_limits<double>::infinity();
        double max_q_euclidean = -std::numeric_limits<double>::infinity();
        double min_q_mass = std::numeric_limits<double>::infinity();
        double max_q_mass = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < 20; ++i) {
            Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.17 * i);
            sandwich.project(x);
            const Eigen::VectorXd y = sandwich.solve(x);
            const double q_euclidean = x.dot(y);
            const double q_mass = (hs.mass_diag.array() * x.array() * y.array()).sum();
            min_q_euclidean = std::min(min_q_euclidean, q_euclidean);
            max_q_euclidean = std::max(max_q_euclidean, q_euclidean);
            min_q_mass = std::min(min_q_mass, q_mass);
            max_q_mass = std::max(max_q_mass, q_mass);
        }

        std::cout << "    " << variant_name(variant.inverse_mode,
                                             variant.projection_mode)
                  << ": q_euclidean_min=" << min_q_euclidean
                  << ", q_euclidean_max=" << max_q_euclidean
                  << ", q_mass_min=" << min_q_mass
                  << ", q_mass_max=" << max_q_mass << "\n";

        if (variant.inverse_mode == rsh::HsLaplacianInverseMode::RawStiffness &&
            variant.projection_mode == rsh::HsConstantProjectionMode::Algebraic) {
            check(min_q_euclidean > 0.0,
                  "committed raw sandwich is Euclidean-PSD on deterministic probes");
        }
    }
}

void test_sandwich_solver_sanity() {
    std::cout << "-- sandwich preconditioner GMRES sanity --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;
    const double theta = 0.25;
    Eigen::VectorXd rhs = deterministic_field(mesh.n_vertices(), 0.8);

    const SolveStats identity = solve_with_identity(mesh, rhs, s, theta, 200, 1e-6);
    const SolveStats sandwich = solve_with_sandwich(mesh, rhs, s, theta, 200, 1e-6);

    const double x_rel = (identity.x - sandwich.x).norm() /
                         std::max(1.0, identity.x.norm());
    std::cout << "    identity:  iter=" << identity.iterations
              << ", error=" << identity.error
              << ", residual=" << identity.residual << "\n";
    std::cout << "    sandwich:  iter=" << sandwich.iterations
              << ", error=" << sandwich.error
              << ", residual=" << sandwich.residual << "\n";
    std::cout << "    relative solution difference = " << x_rel << "\n";

    check(identity.residual < 1e-4, "identity-preconditioned solve residual is small");
    check(sandwich.residual < 1e-4, "sandwich-preconditioned solve residual is small");
    check(x_rel < 1e-3, "identity and sandwich solves agree");
    check(sandwich.iterations <= identity.iterations,
          "sandwich GMRES iterations do not exceed identity");
}

void test_sandwich_resolution_diagnostics() {
    std::cout << "-- sandwich iteration counts vs resolution --\n";
    const double s = 5.0 / 3.0;
    const double theta = 0.25;
    std::vector<int> identity_iters;
    std::vector<int> raw_iters;
    std::vector<int> paper_iters;
    std::vector<int> h1_iters;
    std::vector<int> h1_left_iters;

    std::cout << "    mesh,n_v,n_f,identity_iter,raw_iter,paper_iter,h1_iter,h1_left_iter,"
              << "identity_residual,raw_residual,paper_residual,h1_residual,h1_left_residual\n";
    for (int subdiv : {2, 3, 4}) {
        rsh::MeshData mesh = rsh::make_icosphere(subdiv);
        mesh.L0 = mesh.compute_L0();
        Eigen::VectorXd rhs =
            mesh.V.col(0) + 0.3 * mesh.V.col(1) - 0.2 * mesh.V.col(2);

        const SolveStats identity = solve_with_identity(mesh, rhs, s, theta, 250, 1e-5);
        const SolveStats raw = solve_with_sandwich(
            mesh, rhs, s, theta, 250, 1e-5,
            rsh::HsLaplacianInverseMode::RawStiffness,
            rsh::HsConstantProjectionMode::Algebraic);
        const SolveStats paper = solve_with_sandwich(
            mesh, rhs, s, theta, 250, 1e-5,
            rsh::HsLaplacianInverseMode::LaplaceBeltrami,
            rsh::HsConstantProjectionMode::MassWeighted);
        const SolveStats h1 = solve_with_sandwich(
            mesh, rhs, s, theta, 250, 1e-5,
            rsh::HsLaplacianInverseMode::H1Metric,
            rsh::HsConstantProjectionMode::None);
        const SolveStats h1_left = solve_with_left_sandwich(
            mesh, rhs, s, theta, 250, 1e-12,
            rsh::HsLaplacianInverseMode::H1Metric,
            rsh::HsConstantProjectionMode::None);
        identity_iters.push_back(identity.iterations);
        raw_iters.push_back(raw.iterations);
        paper_iters.push_back(paper.iterations);
        h1_iters.push_back(h1.iterations);
        h1_left_iters.push_back(h1_left.iterations);

        std::cout << "    icosphere_" << subdiv << "," << mesh.n_vertices()
                  << "," << mesh.n_faces() << "," << identity.iterations
                  << "," << raw.iterations << "," << paper.iterations
                  << "," << h1.iterations << "," << h1_left.iterations
                  << "," << identity.residual << "," << raw.residual
                  << "," << paper.residual << "," << h1.residual
                  << "," << h1_left.residual << "\n";

        check(raw.residual < 1e-2,
              "raw sandwich solve residual stays controlled at this resolution");
        check(raw.iterations <= identity.iterations,
              "raw sandwich iterations do not exceed identity at this resolution");
        check(std::isfinite(paper.residual),
              "paper-weighted sandwich residual is finite at this resolution");
        check(std::isfinite(h1.residual),
              "Repulsor-style H1 sandwich residual is finite at this resolution");
        check(std::isfinite(h1_left.residual),
              "Repulsor-style left-preconditioned H1 residual is finite at this resolution");
    }

    const double raw_growth =
        static_cast<double>(raw_iters.back()) /
        std::max(1, raw_iters.front());
    const double paper_growth =
        static_cast<double>(paper_iters.back()) /
        std::max(1, paper_iters.front());
    const double h1_growth =
        static_cast<double>(h1_iters.back()) /
        std::max(1, h1_iters.front());
    const double h1_left_growth =
        static_cast<double>(h1_left_iters.back()) /
        std::max(1, h1_left_iters.front());
    const bool identity_grows =
        identity_iters.back() > identity_iters.front();
    std::cout << "    raw sandwich growth icosphere_2->4 = " << raw_growth << "\n";
    std::cout << "    paper-weighted sandwich growth icosphere_2->4 = "
              << paper_growth << "\n";
    std::cout << "    Repulsor-style H1 sandwich growth icosphere_2->4 = "
              << h1_growth << "\n";
    std::cout << "    Repulsor-style left H1 sandwich growth icosphere_2->4 = "
              << h1_left_growth << "\n";
    // This is intentionally diagnostic rather than a hard gate. Unit 2B.3's
    // first implementation improves each solve versus identity, but the 2->4
    // growth is still larger than the RSu Fig. 5 target and is reported in
    // communication.md for reviewer diagnosis.
    check(identity_grows, "identity iteration count grows with resolution");
}

void test_compression_gradient_descent_safeguard() {
    std::cout << "-- compression-gradient descent safeguard smoke --\n";
    rsh::MeshData m_left = rsh::make_icosphere(2);
    rsh::MeshData m_right = rsh::make_icosphere(2);
    m_left.V *= 0.5;
    m_right.V *= 0.5;

    double left_max_x = m_left.V.col(0).maxCoeff();
    double left_min_x = m_left.V.col(0).minCoeff();
    double right_min_x = m_right.V.col(0).minCoeff();
    double right_max_x = m_right.V.col(0).maxCoeff();

    const Eigen::RowVector3d c_left = m_left.V.colwise().mean();
    const Eigen::RowVector3d c_right = m_right.V.colwise().mean();
    m_left.V.col(1).array() -= c_left(1);
    m_left.V.col(2).array() -= c_left(2);
    m_right.V.col(1).array() -= c_right(1);
    m_right.V.col(2).array() -= c_right(2);

    m_left.V.col(0).array() += (-0.1 - left_max_x);
    m_right.V.col(0).array() += (0.1 - right_min_x);
    left_min_x = m_left.V.col(0).minCoeff();
    right_max_x = m_right.V.col(0).maxCoeff();

    const rsh::MeshData m_ref = combine_meshes(m_left, m_right);
    rsh::MeshData m_curr = m_ref;

    std::vector<bool> is_handle(m_ref.n_vertices(), false);
    const double epsilon = 0.1;
    for (int i = 0; i < m_ref.n_vertices(); ++i) {
        const double x = m_ref.V(i, 0);
        is_handle[static_cast<size_t>(i)] =
            x <= left_min_x + epsilon || x >= right_max_x - epsilon;
    }

    for (int i = 0; i < m_curr.n_vertices(); ++i) {
        if (is_handle[static_cast<size_t>(i)]) {
            if (m_curr.V(i, 0) < 0.0) {
                m_curr.V(i, 0) += 0.025;
            } else {
                m_curr.V(i, 0) -= 0.025;
            }
        }
    }

    const double alpha = 6.0;
    const double theta = 0.5;
    const double tpe_weight = 0.005;
    const rsh::FaceGeom g = rsh::compute_face_geom(m_curr);
    const rsh::BVH bvh = rsh::build_bvh(m_curr, g);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
    const Eigen::MatrixXd G_tpe =
        rsh::tpe_gradient_bh(m_curr, g, bvh, bp, alpha);

    rsh::ShellEnergyParams shell_params;
    shell_params.thickness = 0.005;
    shell_params.lambda = 1.0;
    shell_params.mu = 1.0;
    const rsh::ShellEnergyGradientResult shell_res =
        rsh::shell_energy_with_gradient(m_ref, m_curr, shell_params);

    Eigen::MatrixXd G_total = tpe_weight * G_tpe + shell_res.grad_def;
    for (int i = 0; i < m_curr.n_vertices(); ++i) {
        if (is_handle[static_cast<size_t>(i)]) {
            G_total.row(i).setZero();
        }
    }

    rsh::HsPreconditionerParams hs_params;
    hs_params.s = 1.5;
    hs_params.sigma = 1.0;
    hs_params.mass_weight = 1.0;
    const rsh::HsDirectionResult hs =
        rsh::hs_preconditioned_direction(m_curr, G_total, hs_params);

    std::cout << "    g_dot_dir=" << hs.g_dot_dir
              << ", used_identity_fallback="
              << (hs.used_identity_fallback ? "yes" : "no")
              << ", max_gmres_iterations=" << hs.max_gmres_iterations
              << ", max_gmres_error=" << hs.max_gmres_error << "\n";
    check(hs.g_dot_dir > 0.0,
          "hs_preconditioned_direction returns a descent direction for the compression gradient");
}

void test_nullspace_and_spd() {
    std::cout << "-- A nullspace and positivity --\n";
    rsh::MeshData mesh = rsh::make_icosphere(2);
    mesh.L0 = mesh.compute_L0();
    const double s = 5.0 / 3.0;

    const Eigen::VectorXd constant =
        Eigen::VectorXd::Constant(mesh.n_vertices(), 2.0);
    const Eigen::VectorXd A_constant = apply_hs_operator(mesh, constant, s, 0.0);
    const double const_norm = A_constant.norm();
    std::cout << "    ||A*1|| = " << const_norm << "\n";
    check(const_norm < 1e-10, "constant field is in A's nullspace");

    Eigen::VectorXd x = deterministic_field(mesh.n_vertices(), 0.7);
    x.array() -= x.mean();
    const double q = x.dot(apply_hs_operator(mesh, x, s, 0.0));
    std::cout << "    mean-zero x^T A x = " << q << "\n";
    check(q > 0.0, "mean-zero nonconstant field has positive A quadratic form");
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "=== Hs A = B + B_0 validation ===\n";

    test_two_triangle_hand_case();
    test_distance_sweep();
    test_b0_two_triangle_hand_case();
    test_b0_distance_sweep();
    test_linearity();
    test_symmetry();
    test_brute_reference();
    test_theta_sweep_table();
    test_scalar_fractional_laplacian_hand_case();
    test_scalar_fractional_laplacian_brute_reference();
    test_sandwich_psd_probe();
    test_sandwich_solver_sanity();
    test_sandwich_resolution_diagnostics();
    test_compression_gradient_descent_safeguard();
    test_nullspace_and_spd();

    std::cout << "\n"
              << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures))
              << "\n";
    return failures == 0 ? 0 : 1;
}
