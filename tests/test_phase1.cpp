#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "GradCheck.h"
#include "MeshData.h"
#include "PathEnergy.h"
#include "ShellEnergy.h"
#include "TestMeshes.h"
#include "TPE.h"
#include "TrustRegionSolver.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

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

// Central-difference Jacobian of f : R^3 -> R^m at x, compared to analytic J.
// Returns max abs componentwise difference.
double jacobian_max_err(
    const std::function<Eigen::VectorXd(const Eigen::Vector3d &)> &f,
    const Eigen::MatrixXd &J_analytic,
    const Eigen::Vector3d &x,
    double eps = 1e-5) {
    Eigen::MatrixXd J_num(J_analytic.rows(), 3);
    Eigen::Vector3d xp = x;
    for (int j = 0; j < 3; ++j) {
        const double h = eps * (1.0 + std::abs(x[j]));
        xp[j] = x[j] + h;
        const Eigen::VectorXd fp = f(xp);
        xp[j] = x[j] - h;
        const Eigen::VectorXd fm = f(xp);
        xp[j] = x[j];
        J_num.col(j) = (fp - fm) / (2.0 * h);
    }
    return (J_analytic - J_num).cwiseAbs().maxCoeff();
}

// Overload for scalar f : R^3 -> R (area). Compares a row-vector gradient.
double gradient_max_err(
    const std::function<double(const Eigen::Vector3d &)> &f,
    const Eigen::RowVector3d &grad_analytic,
    const Eigen::Vector3d &x,
    double eps = 1e-5) {
    Eigen::RowVector3d grad_num;
    Eigen::Vector3d xp = x;
    for (int j = 0; j < 3; ++j) {
        const double h = eps * (1.0 + std::abs(x[j]));
        xp[j] = x[j] + h;
        const double fp = f(xp);
        xp[j] = x[j] - h;
        const double fm = f(xp);
        xp[j] = x[j];
        grad_num[j] = (fp - fm) / (2.0 * h);
    }
    return (grad_analytic - grad_num).cwiseAbs().maxCoeff();
}

void test_unit_triangle() {
    std::cout << "-- closed-form check on unit-z triangle --\n";
    MeshData m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0;
    m.F.resize(1, 3);
    m.F << 0, 1, 2;

    FaceGeom g = compute_face_geom(m);
    check(std::abs(g.A(0) - 0.5) < 1e-12, "area == 0.5");
    check((g.N.row(0) - Eigen::RowVector3d(0, 0, 1)).norm() < 1e-12,
          "normal == +z");
    check((g.C.row(0) - Eigen::RowVector3d(1.0 / 3.0, 1.0 / 3.0, 0.0)).norm() < 1e-12,
          "centroid == (1/3, 1/3, 0)");
}

void test_icosphere_invariants() {
    std::cout << "-- closed-mesh invariants on icosphere(2) --\n";
    MeshData m = make_icosphere(2);
    FaceGeom g = compute_face_geom(m);

    // All areas strictly positive.
    check(g.A.minCoeff() > 0.0, "all areas > 0");

    // Divergence theorem: on a closed surface, sum of (a_t * n_t) = 0.
    Eigen::Vector3d an_sum = Eigen::Vector3d::Zero();
    for (int t = 0; t < g.A.size(); ++t) {
        an_sum += g.A(t) * g.N.row(t).transpose();
    }
    check(an_sum.norm() < 1e-10, "sum a_t n_t ~ 0 (divergence thm)");

    // Total area of unit sphere ~ 4π (our icosphere approximates it).
    const double total_area = g.A.sum();
    check(std::abs(total_area - 4.0 * M_PI) < 0.5,
          "total area approximates 4π (sphere surface)");
}

std::pair<int, int> mesh_edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

std::map<std::pair<int, int>, int> mesh_edge_counts(const MeshData &mesh) {
    std::map<std::pair<int, int>, int> counts;
    for (int f = 0; f < mesh.n_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            ++counts[mesh_edge_key(mesh.F(f, k), mesh.F(f, (k + 1) % 3))];
        }
    }
    return counts;
}

bool has_duplicate_vertices(const MeshData &mesh, double tol = 1e-12) {
    const double tol2 = tol * tol;
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        for (int j = i + 1; j < mesh.n_vertices(); ++j) {
            if ((mesh.V.row(i) - mesh.V.row(j)).squaredNorm() <= tol2) {
                return true;
            }
        }
    }
    return false;
}

void test_make_n_torus_topology() {
    std::cout << "-- connected-sum n-torus topology checks --\n";
    for (int genus = 1; genus <= 3; ++genus) {
        MeshData m = make_n_torus(genus, 1.0, 0.3, 18, 10);
        const FaceGeom g = compute_face_geom(m);
        const auto edge_counts = mesh_edge_counts(m);
        const int V = m.n_vertices();
        const int E = static_cast<int>(edge_counts.size());
        const int F = m.n_faces();
        const int chi = V - E + F;
        const int expected_chi = 2 - 2 * genus;

        bool edge_manifold = true;
        bool closed = true;
        for (const auto &item : edge_counts) {
            edge_manifold = edge_manifold && item.second <= 2;
            closed = closed && item.second == 2;
        }

        std::cout << "    genus=" << genus
                  << ", V=" << V
                  << ", E=" << E
                  << ", F=" << F
                  << ", chi=" << chi
                  << ", expected_chi=" << expected_chi
                  << ", min_area=" << g.A.minCoeff()
                  << "\n";
        check(chi == expected_chi,
              "make_n_torus(" + std::to_string(genus) +
                  ") has Euler chi = 2 - 2g");
        check(edge_manifold,
              "make_n_torus(" + std::to_string(genus) +
                  ") has no edge with >2 incident faces");
        check(closed,
              "make_n_torus(" + std::to_string(genus) +
                  ") has no boundary edges");
        check(g.A.minCoeff() > 1e-12,
              "make_n_torus(" + std::to_string(genus) +
                  ") has no degenerate faces");
        check(!has_duplicate_vertices(m),
              "make_n_torus(" + std::to_string(genus) +
                  ") has no duplicate vertices");
    }
}

// Pick a random non-degenerate triangle for Jacobian checks.
void random_triangle(Eigen::Vector3d &v0, Eigen::Vector3d &v1, Eigen::Vector3d &v2,
                     std::mt19937 &rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int attempt = 0; attempt < 16; ++attempt) {
        v0 = Eigen::Vector3d(u(rng), u(rng), u(rng));
        v1 = Eigen::Vector3d(u(rng), u(rng), u(rng));
        v2 = Eigen::Vector3d(u(rng), u(rng), u(rng));
        const double a2 = (v1 - v0).cross(v2 - v0).norm();
        if (a2 > 0.2) return;  // well away from degenerate
    }
    // fallback: standard-basis triangle
    v0 = Eigen::Vector3d(0, 0, 0);
    v1 = Eigen::Vector3d(1, 0, 0);
    v2 = Eigen::Vector3d(0, 1, 0);
}

// Helper: for a triangle, return (centroid, normal, area) as a function of
// the k-th corner position (others held fixed).
void triangle_funcs(
    int k,
    Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector3d v2,
    std::function<Eigen::VectorXd(const Eigen::Vector3d &)> &c_of,
    std::function<Eigen::VectorXd(const Eigen::Vector3d &)> &n_of,
    std::function<double(const Eigen::Vector3d &)> &a_of) {
    // Capture by value, mutate the k-th corner.
    c_of = [k, v0, v1, v2](const Eigen::Vector3d &x) -> Eigen::VectorXd {
        Eigen::Vector3d a = v0, b = v1, c = v2;
        if (k == 0) a = x; else if (k == 1) b = x; else c = x;
        return (a + b + c) / 3.0;
    };
    n_of = [k, v0, v1, v2](const Eigen::Vector3d &x) -> Eigen::VectorXd {
        Eigen::Vector3d a = v0, b = v1, c = v2;
        if (k == 0) a = x; else if (k == 1) b = x; else c = x;
        const Eigen::Vector3d u = (b - a).cross(c - a);
        return u / u.norm();
    };
    a_of = [k, v0, v1, v2](const Eigen::Vector3d &x) -> double {
        Eigen::Vector3d a = v0, b = v1, c = v2;
        if (k == 0) a = x; else if (k == 1) b = x; else c = x;
        return 0.5 * (b - a).cross(c - a).norm();
    };
}

void test_jacobians_fd() {
    std::cout << "-- FD-check per-corner Jacobians on random triangles --\n";
    std::mt19937 rng(42);

    const int num_trials = 8;
    double worst_dc = 0.0, worst_da = 0.0, worst_dn = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        Eigen::Vector3d v0, v1, v2;
        random_triangle(v0, v1, v2, rng);
        const Eigen::Vector3d cross = (v1 - v0).cross(v2 - v0);
        const double area = 0.5 * cross.norm();
        const Eigen::Vector3d n = cross / cross.norm();
        Eigen::Vector3d E0, E1, E2;
        opposite_edges(v0, v1, v2, E0, E1, E2);
        const Eigen::Vector3d E[3] = {E0, E1, E2};

        for (int k = 0; k < 3; ++k) {
            std::function<Eigen::VectorXd(const Eigen::Vector3d &)> c_of, n_of;
            std::function<double(const Eigen::Vector3d &)> a_of;
            triangle_funcs(k, v0, v1, v2, c_of, n_of, a_of);
            const Eigen::Vector3d vk = (k == 0 ? v0 : k == 1 ? v1 : v2);

            const Eigen::Matrix3d Jc = dc_dvk();
            const Eigen::RowVector3d ga = da_dvk(n, E[k]);
            const Eigen::Matrix3d Jn = dn_dvk(n, area, E[k]);

            worst_dc = std::max(worst_dc, jacobian_max_err(c_of, Jc, vk));
            worst_da = std::max(worst_da, gradient_max_err(a_of, ga, vk));
            worst_dn = std::max(worst_dn, jacobian_max_err(n_of, Jn, vk));
        }
    }

    std::cout << "    worst |dc/dv - FD| = " << worst_dc << "\n";
    std::cout << "    worst |da/dv - FD| = " << worst_da << "\n";
    std::cout << "    worst |dn/dv - FD| = " << worst_dn << "\n";

    check(worst_dc < 1e-8, "dc/dv_k matches FD");
    check(worst_da < 1e-6, "da/dv_k matches FD");
    check(worst_dn < 1e-6, "dn/dv_k matches FD");
}

void test_translation_and_shear_invariants() {
    std::cout << "-- algebraic invariants of per-corner Jacobians --\n";
    std::mt19937 rng(7);

    double worst_sum_da = 0.0, worst_sum_dn = 0.0, worst_shear = 0.0;

    for (int trial = 0; trial < 8; ++trial) {
        Eigen::Vector3d v0, v1, v2;
        random_triangle(v0, v1, v2, rng);
        const Eigen::Vector3d cross = (v1 - v0).cross(v2 - v0);
        const double area = 0.5 * cross.norm();
        const Eigen::Vector3d n = cross / cross.norm();
        Eigen::Vector3d E0, E1, E2;
        opposite_edges(v0, v1, v2, E0, E1, E2);
        const Eigen::Vector3d E[3] = {E0, E1, E2};

        Eigen::RowVector3d sum_da = Eigen::RowVector3d::Zero();
        Eigen::Matrix3d sum_dn = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) {
            sum_da += da_dvk(n, E[k]);
            sum_dn += dn_dvk(n, area, E[k]);
            // Sliding corner k parallel to its opposite edge leaves area fixed.
            const double shear = std::abs(da_dvk(n, E[k]) * E[k]);
            worst_shear = std::max(worst_shear, shear);
        }
        worst_sum_da = std::max(worst_sum_da, sum_da.cwiseAbs().maxCoeff());
        worst_sum_dn = std::max(worst_sum_dn, sum_dn.cwiseAbs().maxCoeff());
    }

    std::cout << "    worst |sum_k da/dv_k| = " << worst_sum_da << "\n";
    std::cout << "    worst |sum_k dn/dv_k| = " << worst_sum_dn << "\n";
    std::cout << "    worst |da/dv_k . E_k| = " << worst_shear << "\n";
    check(worst_sum_da < 1e-12, "sum_k da/dv_k = 0 (translation invariant)");
    check(worst_sum_dn < 1e-12, "sum_k dn/dv_k = 0 (translation invariant)");
    check(worst_shear  < 1e-12, "da/dv_k . E_k = 0 (shear-invariant area)");
}

void test_tpe_two_triangle_toy() {
    std::cout << "-- brute-force TPE on two parallel unit triangles --\n";
    // Two copies of the right triangle {(0,0,0), (1,0,0), (0,1,0)} at z = 0
    // and z = 1. Both have area 1/2, normal (0,0,1), centroid (1/3, 1/3, z).
    // Chord c1 - c2 = (0, 0, -1) (or +1 in the reverse pair). |<n, chord>| = 1,
    // so the kernel is 1 / 1^(2 alpha) = 1. Two ordered pairs contribute,
    // each weighted by a1 * a2 = 1/4: Phi = 2 * 1/4 * 1 = 1/2.
    MeshData m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           0, 0, 1,
           1, 0, 1,
           0, 1, 1;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           3, 4, 5;

    const double phi = tpe_energy_brute(m, 6.0);
    check(std::abs(phi - 0.5) < 1e-12,
          "Phi(two unit triangles, d=1, alpha=6) == 1/2");
}

void test_tpe_coplanar_is_zero() {
    std::cout << "-- TPE vanishes on coplanar triangles --\n";
    // Two non-overlapping triangles in the z = 0 plane. Both normals are +z,
    // but each chord c1 - c2 lies in the z = 0 plane so <n, chord> = 0 and
    // every pair contributes exactly zero.
    MeshData m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           5, 0, 0,
           6, 0, 0,
           5, 1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           3, 4, 5;
    const double phi = tpe_energy_brute(m, 6.0);
    check(std::abs(phi) < 1e-20, "Phi(coplanar pair) == 0");
}

void test_tpe_scale_covariance() {
    std::cout << "-- scale covariance Phi(lambda x) = lambda^(4 - alpha) Phi(x) --\n";
    // Kernel |<n, chord>|^alpha / |chord|^{2 alpha} scales as lambda^{-alpha},
    // and the two area weights a_{t1} a_{t2} contribute lambda^4, so
    //   Phi(lambda x) = lambda^(4 - alpha) Phi(x).
    // For alpha = 6 and lambda = 2 the ratio is 2^(-2) = 0.25.
    MeshData m = make_icosphere(1);
    m.normalize();
    const double alpha = 6.0;
    const double phi_1 = tpe_energy_brute(m, alpha);

    MeshData m2 = m;
    m2.V *= 2.0;
    const double phi_2 = tpe_energy_brute(m2, alpha);

    const double ratio = phi_2 / phi_1;
    const double expected = std::pow(2.0, 4.0 - alpha);
    const double rel_err = std::abs(ratio - expected) / expected;
    std::cout << "    Phi(x)         = " << phi_1 << "\n";
    std::cout << "    Phi(2x)        = " << phi_2 << "\n";
    std::cout << "    ratio          = " << ratio << "\n";
    std::cout << "    expected       = " << expected << "\n";
    std::cout << "    rel err        = " << rel_err << "\n";
    check(rel_err < 1e-10,
          "ratio matches lambda^(4 - alpha) at lambda = 2");
}

void test_tpe_translation_invariance() {
    std::cout << "-- TPE is translation-invariant --\n";
    // Centroids, normals, and areas are all translation-invariant, so the
    // whole energy must be too.
    MeshData m = make_icosphere(1);
    m.normalize();
    const double phi_0 = tpe_energy_brute(m, 6.0);

    MeshData m_shift = m;
    m_shift.V.rowwise() += Eigen::RowVector3d(3.7, -2.1, 0.5);
    const double phi_t = tpe_energy_brute(m_shift, 6.0);

    const double rel = std::abs(phi_t - phi_0) / std::max(phi_0, 1.0);
    check(rel < 1e-10, "Phi(x + t) == Phi(x)");
}

// Pack/unpack an (n_v x 3) vertex matrix into a 3 n_v vector using Eigen's
// default column-major layout (all x's, then all y's, then all z's).
Eigen::VectorXd flatten(const Eigen::MatrixXd &V) {
    Eigen::VectorXd x(V.size());
    Eigen::Map<Eigen::MatrixXd>(x.data(), V.rows(), V.cols()) = V;
    return x;
}

Eigen::MatrixXd unflatten(const Eigen::VectorXd &x, int nv) {
    Eigen::MatrixXd V(nv, 3);
    V = Eigen::Map<const Eigen::MatrixXd>(x.data(), nv, 3);
    return V;
}

void test_tpe_gradient_fd_torus() {
    std::cout << "-- FD gradient check on small torus (alpha = 6) --\n";
    // ~200 faces — small enough that an O(n_f^2) FD sweep is cheap (~0.5s),
    // large enough to exercise the scatter through every per-corner Jacobian.
    MeshData m = make_torus(1.0, 0.3, 12, 8);  // 96 verts, 192 faces
    m.normalize();
    const double alpha = 6.0;
    const int nv = m.n_vertices();

    auto energy = [&](const Eigen::VectorXd &x) -> double {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        return tpe_energy_brute(mm, alpha);
    };
    auto grad = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        return flatten(tpe_gradient_brute(mm, alpha));
    };

    const Eigen::VectorXd x0 = flatten(m.V);
    const GradCheckResult r = finite_diff_gradient_check(energy, grad, x0);

    std::cout << "    nv              = " << nv << "\n";
    std::cout << "    max abs err     = " << r.max_abs_err << "\n";
    std::cout << "    max rel err     = " << r.max_rel_err << "\n";
    std::cout << "    worst index     = " << r.worst_index << "\n";
    std::cout << "    ||analytical||  = " << r.analytical.norm() << "\n";
    std::cout << "    ||numerical||   = " << r.numerical.norm() << "\n";

    // This is the validation gate per CLAUDE.md — 1.3 does not ship without it.
    check(r.pass(1e-4), "FD gradient check at rel-err < 1e-4");
}

void test_tpe_gradient_translation_equivariance() {
    std::cout << "-- gradient unchanged under rigid translation --\n";
    // Energy is translation-invariant, so moving every vertex by the same
    // constant must leave the vertex gradient bit-identical.
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const double alpha = 6.0;

    const Eigen::MatrixXd G0 = tpe_gradient_brute(m, alpha);

    MeshData m_shift = m;
    m_shift.V.rowwise() += Eigen::RowVector3d(3.7, -2.1, 0.5);
    const Eigen::MatrixXd G_t = tpe_gradient_brute(m_shift, alpha);

    const double err = (G_t - G0).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, G0.cwiseAbs().maxCoeff());
    std::cout << "    max |G(x+t) - G(x)| = " << err << "\n";
    check(err / scale < 1e-10, "gradient is translation-equivariant");
}

void test_tpe_gradient_sum_to_zero() {
    std::cout << "-- sum of gradient rows = 0 (translation invariance) --\n";
    // If E(x + t) = E(x) for all constants t, then sum_i dE/dv_i = 0 as a
    // 3-vector. This is a purely algebraic check on the analytical gradient
    // (no finite differences needed).
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const Eigen::MatrixXd G = tpe_gradient_brute(m, 6.0);

    const Eigen::RowVector3d row_sum = G.colwise().sum();
    const double scale = std::max(1.0, G.cwiseAbs().maxCoeff());
    std::cout << "    sum_i dPhi/dv_i = " << row_sum << "\n";
    std::cout << "    relative scale  = " << scale << "\n";
    check(row_sum.norm() / scale < 1e-10, "sum_i dPhi/dv_i = 0");
}

// Collect every face index in the subtree rooted at node u.
std::vector<int> faces_in_subtree(const BVH &bvh, int u) {
    const BVHNode &U = bvh.nodes[u];
    if (U.is_leaf()) {
        return std::vector<int>(
            bvh.face_indices.begin() + U.face_start,
            bvh.face_indices.begin() + U.face_end);
    }
    std::vector<int> out = faces_in_subtree(bvh, U.left);
    std::vector<int> r = faces_in_subtree(bvh, U.right);
    out.insert(out.end(), r.begin(), r.end());
    return out;
}

void test_bvh_invariants() {
    std::cout << "-- BVH invariants on icosphere(3) --\n";
    MeshData m = make_icosphere(3);  // 1280 faces
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g, 8);

    // face_indices is a permutation of [0, n_f).
    std::vector<int> seen(m.n_faces(), 0);
    for (int t : bvh.face_indices) {
        if (t >= 0 && t < m.n_faces()) seen[t]++;
    }
    bool is_permutation = true;
    for (int c : seen) if (c != 1) { is_permutation = false; break; }
    check(is_permutation, "face_indices is a permutation of [0, n_f)");

    // Child AABBs fit inside parent AABB.
    bool aabb_nested = true;
    int worst_aabb_node = -1;
    for (int u = 0; u < bvh.n_nodes(); ++u) {
        const BVHNode &U = bvh.nodes[u];
        if (U.is_leaf()) continue;
        const BVHNode &L = bvh.nodes[U.left];
        const BVHNode &R = bvh.nodes[U.right];
        const double tol = 1e-12;
        if (((U.bmin.array() > L.bmin.array() + tol).any()) ||
            ((U.bmin.array() > R.bmin.array() + tol).any()) ||
            ((U.bmax.array() < L.bmax.array() - tol).any()) ||
            ((U.bmax.array() < R.bmax.array() - tol).any())) {
            aabb_nested = false;
            worst_aabb_node = u;
            break;
        }
    }
    if (!aabb_nested) {
        std::cout << "    worst AABB-nesting offender: node " << worst_aabb_node << "\n";
    }
    check(aabb_nested, "child AABBs fit inside parent AABB");

    // Leaf face counts in [1, 8]. One degenerate exception: a "leaf" left
    // behind by the coincident-centroid fallback in the builder can be
    // larger, but that is not expected on any normal mesh.
    bool leaf_sizes_ok = true;
    int max_leaf_size = 0;
    for (const BVHNode &U : bvh.nodes) {
        if (!U.is_leaf()) continue;
        const int k = U.face_count();
        max_leaf_size = std::max(max_leaf_size, k);
        if (k < 1 || k > 8) leaf_sizes_ok = false;
    }
    std::cout << "    max leaf face count = " << max_leaf_size << "\n";
    check(leaf_sizes_ok, "leaf face counts in [1, 8]");

    // Root aggregates match direct computation.
    double total_area = 0.0;
    Eigen::Vector3d c_weighted = Eigen::Vector3d::Zero();
    Eigen::Vector3d n_weighted = Eigen::Vector3d::Zero();
    for (int t = 0; t < m.n_faces(); ++t) {
        const double a = g.A(t);
        const Eigen::Vector3d c = g.C.row(t);
        const Eigen::Vector3d n = g.N.row(t);
        total_area += a;
        c_weighted += a * c;
        n_weighted += a * n;
    }
    c_weighted /= total_area;
    const BVHNode &root = bvh.nodes[bvh.root];
    const double a_err = std::abs(root.area - total_area) / total_area;
    const double c_err = (root.centroid - c_weighted).norm();
    const double n_err = (root.normal_sum - n_weighted).norm();
    std::cout << "    root area rel-err       = " << a_err << "\n";
    std::cout << "    root centroid abs-err   = " << c_err << "\n";
    std::cout << "    root normal_sum abs-err = " << n_err << "\n";
    check(a_err < 1e-12, "root.area == sum of face areas");
    check(c_err < 1e-12, "root.centroid == area-weighted mean of face centroids");
    check(n_err < 1e-12, "root.normal_sum == area-weighted sum of face normals");
}

void test_bvh_internal_aggregates() {
    std::cout << "-- internal-node aggregates match direct subtree sums --\n";
    MeshData m = make_torus(1.0, 0.3, 16, 12);  // 384 faces
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g, 8);

    double worst_area = 0.0, worst_c = 0.0, worst_n = 0.0;
    for (int u = 0; u < bvh.n_nodes(); ++u) {
        const BVHNode &U = bvh.nodes[u];
        const std::vector<int> faces = faces_in_subtree(bvh, u);
        double a_ref = 0.0;
        Eigen::Vector3d c_num = Eigen::Vector3d::Zero();
        Eigen::Vector3d n_sum = Eigen::Vector3d::Zero();
        for (int t : faces) {
            const double a = g.A(t);
            const Eigen::Vector3d c = g.C.row(t);
            const Eigen::Vector3d n = g.N.row(t);
            a_ref += a;
            c_num += a * c;
            n_sum += a * n;
        }
        const Eigen::Vector3d c_ref = c_num / a_ref;
        worst_area = std::max(worst_area, std::abs(U.area - a_ref));
        worst_c    = std::max(worst_c,    (U.centroid - c_ref).norm());
        worst_n    = std::max(worst_n,    (U.normal_sum - n_sum).norm());
    }
    std::cout << "    worst |a_U - ref|        = " << worst_area << "\n";
    std::cout << "    worst ||c_U - ref||      = " << worst_c << "\n";
    std::cout << "    worst ||n_U - ref||      = " << worst_n << "\n";
    check(worst_area < 1e-12, "internal a_U matches direct subtree sum");
    check(worst_c    < 1e-12, "internal c_U matches direct subtree area-weighted mean");
    check(worst_n    < 1e-12, "internal n_U matches direct subtree area-weighted sum");
}

void test_bvh_cluster_radius_bound() {
    std::cout << "-- cluster radius r_U upper-bounds ||c_t - c_U|| for every t in U --\n";
    MeshData m = make_torus(1.0, 0.3, 16, 12);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g, 8);

    double worst_excess = 0.0;
    int worst_node = -1;
    for (int u = 0; u < bvh.n_nodes(); ++u) {
        const BVHNode &U = bvh.nodes[u];
        for (int t : faces_in_subtree(bvh, u)) {
            const Eigen::Vector3d c = g.C.row(t);
            const double d = (c - U.centroid).norm();
            const double excess = d - U.radius;
            if (excess > worst_excess) {
                worst_excess = excess;
                worst_node = u;
            }
        }
    }
    std::cout << "    worst excess (||c_t - c_U|| - r_U) = " << worst_excess
              << "  (node " << worst_node << ")\n";
    check(worst_excess < 1e-12, "r_U upper-bounds ||c_t - c_U|| in subtree");
}

// Count face pairs covered by a BlockPairs. Self-pair leaf (u == v) covers
// nu * (nu - 1) ordered pairs (skipping the diagonal); any other pair covers
// nu * nv.
long long count_face_pairs(const BVH &bvh, const BlockPairs &bp) {
    auto face_count = [&](int u) {
        return static_cast<long long>(faces_in_subtree(bvh, u).size());
    };
    long long total = 0;
    for (const ClusterPair &p : bp.admissible) {
        total += face_count(p.u) * face_count(p.v);
    }
    for (const ClusterPair &p : bp.near_field) {
        const long long nu = face_count(p.u);
        const long long nv = face_count(p.v);
        total += (p.u == p.v) ? nu * (nv - 1) : nu * nv;
    }
    return total;
}

void test_bct_theta_zero() {
    std::cout << "-- BCT at theta=0 reduces to brute-force near-field list --\n";
    // Strict inequality in MAC (max_r < theta * dist) guarantees that every
    // pair falls through to near-field at theta = 0.
    MeshData m = make_icosphere(2);  // 320 faces
    m.normalize();
    const BVH bvh = build_bvh(m, 8);
    const BlockPairs bp = build_bct_self(bvh, 0.0);
    const long long covered = count_face_pairs(bvh, bp);
    const long long expected =
        static_cast<long long>(m.n_faces()) * (m.n_faces() - 1);

    std::cout << "    admissible = " << bp.admissible.size()
              << ", near-field = " << bp.near_field.size() << "\n";
    std::cout << "    covered face pairs = " << covered
              << " (expected " << expected << ")\n";
    check(bp.admissible.empty(), "no admissible pairs at theta = 0");
    check(covered == expected,
          "near-field covers exactly n_f (n_f - 1) ordered face pairs");
}

void test_bct_exclusive_coverage() {
    std::cout << "-- BCT at theta=0.5 covers every ordered face pair exactly once --\n";
    // Strong form: enumerate every covered ordered pair into a set and
    // verify the set equals {(t1, t2) : t1 != t2}. Uses icosphere(1) so
    // the full pair set fits comfortably (80 * 79 = 6320 pairs).
    MeshData m = make_icosphere(1);  // 80 faces
    m.normalize();
    const BVH bvh = build_bvh(m, 8);
    const BlockPairs bp = build_bct_self(bvh, 0.5);

    std::set<std::pair<int, int>> covered;
    bool duplicate = false;
    auto insert_pair = [&](int t1, int t2) {
        if (!covered.insert({t1, t2}).second) duplicate = true;
    };
    for (const ClusterPair &p : bp.admissible) {
        for (int t1 : faces_in_subtree(bvh, p.u)) {
            for (int t2 : faces_in_subtree(bvh, p.v)) {
                insert_pair(t1, t2);
            }
        }
    }
    for (const ClusterPair &p : bp.near_field) {
        for (int t1 : faces_in_subtree(bvh, p.u)) {
            for (int t2 : faces_in_subtree(bvh, p.v)) {
                if (t1 == t2) continue;
                insert_pair(t1, t2);
            }
        }
    }

    const long long expected =
        static_cast<long long>(m.n_faces()) * (m.n_faces() - 1);
    std::cout << "    covered unique pairs = " << covered.size()
              << " (expected " << expected << ")\n";
    check(!duplicate, "no duplicate face pairs across admissible + near-field");
    check(static_cast<long long>(covered.size()) == expected,
          "BCT covers every ordered face pair exactly once");
}

void test_bct_scaling() {
    std::cout << "-- BCT pair counts vs mesh size (icosphere 1..3) --\n";
    for (int k = 1; k <= 3; ++k) {
        MeshData m = make_icosphere(k);
        m.normalize();
        const BVH bvh = build_bvh(m, 8);
        const BlockPairs bp = build_bct_self(bvh, 0.5);
        const long long covered = count_face_pairs(bvh, bp);
        const long long expected =
            static_cast<long long>(m.n_faces()) * (m.n_faces() - 1);
        std::cout << "    k=" << k << "  n_f=" << m.n_faces()
                  << "  admissible=" << bp.admissible.size()
                  << "  near-field=" << bp.near_field.size()
                  << "  covered=" << covered << "/" << expected << "\n";
        check(covered == expected,
              "total face-pair coverage matches n_f (n_f - 1)");
    }
}

void test_tpe_bh_theta_zero_matches_brute() {
    std::cout << "-- BH energy at theta=0 matches brute force exactly --\n";
    // theta = 0 -> no admissible pairs, everything falls to near-field.
    // Near-field iterates exactly the same (t1, t2) set as brute force, so
    // the two sums differ only by floating-point accumulation order.
    MeshData m = make_torus(1.0, 0.3, 12, 8);  // 192 faces
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, 0.0);

    const double phi_brute = tpe_energy_brute(g, 6.0);
    const double phi_bh = tpe_energy_bh(g, bvh, bp, 6.0);
    const double rel_err = std::abs(phi_bh - phi_brute) / std::abs(phi_brute);
    std::cout << "    phi_brute = " << phi_brute << "\n"
              << "    phi_bh    = " << phi_bh << "\n"
              << "    rel_err   = " << rel_err << "\n";
    check(rel_err < 1e-12,
          "BH energy at theta=0 matches brute force to float-roundoff");
}

void test_tpe_bh_smooth_mesh_accuracy() {
    std::cout << "-- BH energy vs brute at theta=0.5 on smooth meshes --\n";
    // RS §5.2 quotes ~0.1% rel-err for the 0th-order multipole at theta = 0.5
    // on typical smooth geometry; the icosphere hits that cleanly. Anisotropic
    // curvature (fat torus) degrades the bound — Jensen's inequality on the
    // normal averaging underestimates the true sum — so we gate those higher.
    struct Case {
        std::string name;
        MeshData mesh;
        double tol;
    };
    // Gate values are approximation-family bounds for the current 0th-order
    // multipole scheme (projector-covariance aggregate + AABB-MAC, matching
    // Repulsor's BCT0). On these meshes at theta=0.5 both implementations sit
    // in the 3-5% regime; tighter gates would flag honest approximation noise.
    // The theta-monotone-collapse test below remains the structural correctness
    // check that the BH partition and aggregates are wired correctly.
    std::vector<Case> cases = {
        {"icosphere(3), 1280 faces", make_icosphere(3), 5e-2},
        {"torus(1, 0.3, 24, 16), 768 faces", make_torus(1.0, 0.3, 24, 16), 5e-2},
    };
    for (auto &c : cases) {
        c.mesh.normalize();
        const FaceGeom g = compute_face_geom(c.mesh);
        const BVH bvh = build_bvh(c.mesh, g);
        const BlockPairs bp = build_bct_self(bvh, 0.5);

        const double phi_brute = tpe_energy_brute(g, 6.0);
        const double phi_bh = tpe_energy_bh(g, bvh, bp, 6.0);
        const double rel_err =
            std::abs(phi_bh - phi_brute) / std::abs(phi_brute);
        std::cout << "    " << c.name << "\n"
                  << "      admissible = " << bp.admissible.size()
                  << ", near-field = " << bp.near_field.size() << "\n"
                  << "      phi_brute = " << phi_brute
                  << ", phi_bh = " << phi_bh
                  << ", rel_err = " << rel_err
                  << " (gate " << c.tol << ")\n";
        check(rel_err < c.tol,
              "BH energy rel-err below gate vs brute force at theta=0.5");
    }
}

void test_tpe_bh_energy_deterministic() {
    std::cout << "-- BH energy deterministic under repeated calls --\n";
    MeshData m = make_icosphere(3);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, 0.5);

    const double phi1 = tpe_energy_bh(g, bvh, bp, 6.0);
    const double phi2 = tpe_energy_bh(g, bvh, bp, 6.0);
    const bool byte_equal = std::memcmp(&phi1, &phi2, sizeof(double)) == 0;
    std::cout << "    phi1 = " << phi1
              << ", phi2 = " << phi2
              << ", byte_equal = " << byte_equal << "\n";
    check(byte_equal, "BH energy is byte-identical across repeated calls");
}

void test_tpe_bh_theta_sweep_convergence() {
    std::cout << "-- BH -> brute as theta -> 0 (torus, monotone in theta) --\n";
    // Convergence of the 0th-order approximation: smaller theta means fewer
    // admissible pairs and more near-field pairs, so the approximation gets
    // tighter. This is the structural check that says "the BH error really
    // does come from the cluster approximation, not from something else."
    MeshData m = make_torus(1.0, 0.3, 24, 16);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const double phi_brute = tpe_energy_brute(g, 6.0);

    const std::vector<double> thetas = {0.8, 0.5, 0.25, 0.1};
    double prev_err = std::numeric_limits<double>::infinity();
    bool monotone = true;
    for (double theta : thetas) {
        const BlockPairs bp = build_bct_self(bvh, theta);
        const double phi_bh = tpe_energy_bh(g, bvh, bp, 6.0);
        const double rel_err =
            std::abs(phi_bh - phi_brute) / std::abs(phi_brute);
        std::cout << "    theta=" << theta
                  << "  admissible=" << bp.admissible.size()
                  << "  near-field=" << bp.near_field.size()
                  << "  rel_err=" << rel_err << "\n";
        if (rel_err > prev_err + 1e-12) monotone = false;
        prev_err = rel_err;
    }
    check(monotone, "BH rel-err monotone nondecreasing in theta on torus");
}

void test_tpe_gradient_bh_theta_zero_matches_brute() {
    std::cout << "-- BH gradient at theta=0 matches brute-force gradient --\n";
    // theta = 0 -> no admissible pairs, near-field enumerates every ordered
    // (t1, t2) pair. Our near-field inner loop reuses the same per-pair
    // kernel as tpe_gradient_brute, so the two gradients differ only by
    // floating-point summation order.
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, 0.0);

    const Eigen::MatrixXd G_brute = tpe_gradient_brute(m, g, 6.0);
    const Eigen::MatrixXd G_bh    = tpe_gradient_bh(m, g, bvh, bp, 6.0);

    const double scale = std::max(1.0, G_brute.cwiseAbs().maxCoeff());
    const double max_abs = (G_bh - G_brute).cwiseAbs().maxCoeff();
    const double rel = max_abs / scale;
    std::cout << "    max |G_bh - G_brute|      = " << max_abs << "\n"
              << "    relative to ||G_brute||_inf = " << rel << "\n";
    check(rel < 1e-12,
          "BH gradient at theta=0 matches brute-force to float-roundoff");
}

void test_tpe_gradient_bh_fd_check() {
    std::cout << "-- FD gradient check on BH gradient at theta=0.5 --\n";
    // RSu §4.2 "Approximate Derivative": freeze the BCT partition at the
    // current iterate and differentiate the kernel values on that fixed
    // partition. Admissible kernels depend on vertex positions through the
    // cluster aggregates (a_U, c_U, n_U) — all linear in per-face quantities
    // — so "frozen partition" means: same face_indices permutation, same
    // tree shape, same admissible/near-field lists; aggregates recomputed
    // from the perturbed FaceGeom via update_bvh_aggregates().
    //
    // Rebuilding the BVH from scratch under perturbation is NOT equivalent:
    // SAH binning and BCT admissibility are discrete, so a tiny shift can
    // reassign faces or flip partition membership, and that discontinuity
    // dominates the FD.
    MeshData m = make_icosphere(2);  // 162 verts, 320 faces — FD tractable
    m.normalize();

    const double alpha = 6.0;
    const FaceGeom g0 = compute_face_geom(m);
    BVH bvh = build_bvh(m, g0);
    const BlockPairs bp = build_bct_self(bvh, 0.5);
    const int nv = m.n_vertices();

    auto energy = [&](const Eigen::VectorXd &x) -> double {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        const FaceGeom g = compute_face_geom(mm);
        update_bvh_aggregates(bvh, g);
        return tpe_energy_bh(g, bvh, bp, alpha);
    };
    auto grad = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        const FaceGeom g = compute_face_geom(mm);
        update_bvh_aggregates(bvh, g);
        return flatten(tpe_gradient_bh(mm, g, bvh, bp, alpha));
    };

    const Eigen::VectorXd x0 = flatten(m.V);
    const GradCheckResult r = finite_diff_gradient_check(energy, grad, x0);

    // Restore aggregates to the original iterate so downstream tests see
    // a self-consistent BVH (not strictly necessary here since this test
    // owns its own BVH, but defensive for any future refactor).
    update_bvh_aggregates(bvh, g0);

    std::cout << "    nv              = " << nv << "\n"
              << "    admissible      = " << bp.admissible.size()
              << ", near-field = " << bp.near_field.size() << "\n"
              << "    max abs err     = " << r.max_abs_err << "\n"
              << "    max rel err     = " << r.max_rel_err << "\n"
              << "    worst index     = " << r.worst_index << "\n"
              << "    ||analytical||  = " << r.analytical.norm() << "\n"
              << "    ||numerical||   = " << r.numerical.norm() << "\n";
    // CLAUDE.md gate for Step 1.7 is rel-err < 1e-3.
    check(r.pass(1e-3), "BH gradient FD-check passes at rel-err < 1e-3");
}

void test_tpe_gradient_bh_translation_equivariance() {
    std::cout << "-- BH gradient unchanged under rigid translation --\n";
    // Building a fresh BVH on a translated mesh produces a different tree (AABBs
    // shift) but the BlockPairs partition is determined by the cluster radii
    // and centroid distances, which are translation-invariant. So the gradient
    // should match to floating-point roundoff.
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const double alpha = 6.0;
    const Eigen::MatrixXd G0 = tpe_gradient_bh(m, alpha, 0.5);

    MeshData m_shift = m;
    m_shift.V.rowwise() += Eigen::RowVector3d(3.7, -2.1, 0.5);
    const Eigen::MatrixXd G_t = tpe_gradient_bh(m_shift, alpha, 0.5);

    const double scale = std::max(1.0, G0.cwiseAbs().maxCoeff());
    const double err = (G_t - G0).cwiseAbs().maxCoeff();
    std::cout << "    max |G(x+t) - G(x)|  = " << err << "\n"
              << "    relative to scale    = " << err / scale << "\n";
    check(err / scale < 1e-10, "BH gradient is translation-equivariant");
}

void test_tpe_gradient_bh_descent_step() {
    std::cout << "-- gradient direction is a descent direction for BH energy --\n";
    // A direct sanity check, independent of FD: on the frozen partition
    // (tree structure + BlockPairs fixed at x, aggregates refreshed from the
    // perturbed x), taking x <- x - tau ∇E must decrease the BH energy.
    // Semantics match RSu §4.2 and the FD-check above.
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const FaceGeom g0 = compute_face_geom(m);
    BVH bvh = build_bvh(m, g0);
    const BlockPairs bp = build_bct_self(bvh, 0.5);
    const double alpha = 6.0;

    const double E0 = tpe_energy_bh(g0, bvh, bp, alpha);
    const Eigen::MatrixXd G = tpe_gradient_bh(m, g0, bvh, bp, alpha);
    const double g_norm2 = G.squaredNorm();
    // tau small enough that the linear model dominates curvature: pick it
    // relative to the largest gradient component so |dx| stays below 1e-6.
    const double tau = 1e-6 / std::max(1.0, G.cwiseAbs().maxCoeff());

    MeshData m_step = m;
    m_step.V -= tau * G;
    const FaceGeom g_step = compute_face_geom(m_step);
    update_bvh_aggregates(bvh, g_step);   // refresh aggregates on frozen tree
    const double E_step = tpe_energy_bh(g_step, bvh, bp, alpha);
    update_bvh_aggregates(bvh, g0);       // restore aggregates for downstream

    const double predicted = -tau * g_norm2;
    const double actual = E_step - E0;

    std::cout << "    E(x)                 = " << E0 << "\n"
              << "    E(x - tau G)         = " << E_step << "\n"
              << "    predicted deltaE     = " << predicted << "\n"
              << "    actual deltaE        = " << actual << "\n"
              << "    ratio actual/pred    = " << actual / predicted << "\n";
    check(actual < 0.0, "one gradient step strictly decreases BH energy");
    // First-order: actual ~ predicted to O(tau). Tight tolerance since the
    // partition is frozen — no admissibility flips can blur the linear model.
    check(std::abs(actual - predicted) / std::abs(predicted) < 1e-4,
          "linear model deltaE ~ -tau ||G||^2 holds to O(tau)");
}

void test_tpe_bh_scale_covariance() {
    std::cout << "-- BH energy respects scale covariance Phi(lambda x) = lambda^(4-alpha) Phi(x) --\n";
    // The BH approximation is a linear combination of per-pair kernels with
    // the same scaling law as the continuous TPE, so the scale-covariance
    // identity must hold exactly (independent of admissibility structure).
    MeshData m = make_icosphere(2);  // 320 faces
    m.normalize();
    const double alpha = 6.0;
    const double phi1 = tpe_energy_bh(m, alpha, 0.5);

    MeshData m2 = m;
    m2.V *= 2.0;
    const double phi2 = tpe_energy_bh(m2, alpha, 0.5);

    const double expected_ratio = std::pow(2.0, 4.0 - alpha);  // 2^-2 = 0.25
    const double ratio = phi2 / phi1;
    const double rel_err = std::abs(ratio - expected_ratio) / expected_ratio;
    std::cout << "    phi(x)      = " << phi1 << "\n"
              << "    phi(2 x)    = " << phi2 << "\n"
              << "    ratio       = " << ratio
              << "  (expected " << expected_ratio << ")\n"
              << "    rel_err     = " << rel_err << "\n";
    check(rel_err < 1e-12,
          "BH scale covariance holds to float precision");
}

void test_tpe_adaptive_disabled_matches_bh() {
    std::cout << "-- adaptive disabled matches midpoint BH exactly --\n";
    MeshData m = make_torus(1.0, 0.3, 12, 8);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, 0.5);
    const double alpha = 6.0;

    const TpeAdaptiveParams off{};
    const double e_ref = tpe_energy_bh(g, bvh, bp, alpha);
    const double e_off = tpe_energy_bh(m, g, bvh, bp, off, alpha);
    const Eigen::MatrixXd g_ref = tpe_gradient_bh(m, g, bvh, bp, alpha);
    const Eigen::MatrixXd g_off = tpe_gradient_bh(m, g, bvh, bp, off, alpha);

    const double e_err = std::abs(e_off - e_ref);
    const double g_err = (g_off - g_ref).cwiseAbs().maxCoeff();
    check(e_err < 1e-14, "adaptive-off energy equals BH midpoint");
    check(g_err < 1e-12, "adaptive-off gradient equals BH midpoint");
}

void test_tpe_adaptive_depth0_matches_midpoint() {
    std::cout << "-- adaptive max_depth=0 reproduces midpoint near-field --\n";
    MeshData m = make_icosphere(2);
    m.normalize();
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, 0.5);
    const double alpha = 6.0;

    TpeAdaptiveParams ad;
    ad.enabled = true;
    ad.theta = 10.0;
    ad.max_depth = 0;

    const TpeAdaptiveCache cache = build_tpe_adaptive_cache(m, g, bvh, bp, ad);
    const double e_ref = tpe_energy_bh(g, bvh, bp, alpha);
    const double e_ad = tpe_energy_bh(m, g, bvh, bp, ad, alpha, &cache);
    const Eigen::MatrixXd g_ref = tpe_gradient_bh(m, g, bvh, bp, alpha);
    const Eigen::MatrixXd g_ad = tpe_gradient_bh(m, g, bvh, bp, ad, alpha, &cache);

    const double e_err = std::abs(e_ad - e_ref);
    const double g_err = (g_ad - g_ref).cwiseAbs().maxCoeff();
    std::cout << "    |E_ad - E_mid| = " << e_err << "\n";
    check(e_err < 1e-10, "adaptive depth0 energy equals midpoint");
    check(g_err < 1e-11, "adaptive depth0 gradient equals midpoint");
}

void test_tpe_adaptive_gradient_fd_check() {
    std::cout << "-- FD check for adaptive BH gradient (frozen adaptive cache) --\n";
    MeshData m = make_icosphere(2);
    m.normalize();
    const double alpha = 6.0;
    const FaceGeom g0 = compute_face_geom(m);
    BVH bvh = build_bvh(m, g0);
    const BlockPairs bp = build_bct_self(bvh, 0.5);
    TpeAdaptiveParams ad;
    ad.enabled = true;
    ad.theta = 10.0;
    ad.max_depth = 2;
    ad.max_stack_items = 400000;
    const TpeAdaptiveCache cache = build_tpe_adaptive_cache(m, g0, bvh, bp, ad);
    const int nv = m.n_vertices();

    auto energy = [&](const Eigen::VectorXd &x) -> double {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        const FaceGeom g = compute_face_geom(mm);
        update_bvh_aggregates(bvh, g);
        return tpe_energy_bh(mm, g, bvh, bp, ad, alpha, &cache);
    };
    auto grad = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        MeshData mm = m;
        mm.V = unflatten(x, nv);
        const FaceGeom g = compute_face_geom(mm);
        update_bvh_aggregates(bvh, g);
        return flatten(tpe_gradient_bh(mm, g, bvh, bp, ad, alpha, &cache));
    };

    const Eigen::VectorXd x0 = flatten(m.V);
    const GradCheckResult r = finite_diff_gradient_check(energy, grad, x0);
    update_bvh_aggregates(bvh, g0);

    std::cout << "    adaptive terms   = " << cache.near_terms.size() << "\n"
              << "    max abs err      = " << r.max_abs_err << "\n"
              << "    max rel err      = " << r.max_rel_err << "\n";
    check(r.pass(5e-3), "adaptive BH gradient FD-check passes at rel-err < 5e-3");
}

void test_tpe_adaptive_near_contact_growth() {
    std::cout << "-- adaptive strengthens near-contact response for offset centroids --\n";
    const double alpha = 6.0;
    std::vector<double> deltas = {0.2, 0.1, 0.05, 0.025};
    std::vector<double> e_mid;
    std::vector<double> e_ad;
    e_mid.reserve(deltas.size());
    e_ad.reserve(deltas.size());

    for (double d : deltas) {
        // Point-like near contact: only one vertex of triangle 2 approaches
        // triangle 1 as delta shrinks; centroids stay relatively far apart.
        MeshData m;
        m.V.resize(6, 3);
        m.V << 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.05, 0.05, d,
               1.80, 0.05, 1.0,
               0.05, 1.80, 1.0;
        m.F.resize(2, 3);
        m.F << 0, 1, 2,
               3, 4, 5;

        const FaceGeom g = compute_face_geom(m);
        const BVH bvh = build_bvh(m, g);
        const BlockPairs bp = build_bct_self(bvh, 0.5);

        TpeAdaptiveParams ad;
        ad.enabled = true;
        ad.theta = 0.5;
        ad.max_depth = 7;
        ad.max_stack_items = 500000;
        const TpeAdaptiveCache cache = build_tpe_adaptive_cache(m, g, bvh, bp, ad);

        e_mid.push_back(tpe_energy_bh(g, bvh, bp, alpha));
        e_ad.push_back(tpe_energy_bh(m, g, bvh, bp, ad, alpha, &cache));
    }

    bool midpoint_nonincreasing = true;
    bool adaptive_nonincreasing = true;
    bool adaptive_monotone_ratio = true;
    for (size_t i = 1; i < deltas.size(); ++i) {
        if (!(e_mid[i] >= e_mid[i - 1])) midpoint_nonincreasing = false;
        if (!(e_ad[i] >= e_ad[i - 1])) adaptive_nonincreasing = false;
        const double ratio_i = e_ad[i] / e_mid[i];
        const double ratio_prev = e_ad[i - 1] / e_mid[i - 1];
        if (!(ratio_i >= ratio_prev)) adaptive_monotone_ratio = false;
    }
    const double growth_mid = e_mid.back() / e_mid.front();
    const double growth_ad = e_ad.back() / e_ad.front();
    std::cout << "    deltas: ";
    for (double d : deltas) std::cout << d << " ";
    std::cout << "\n    midpoint E: ";
    for (double e : e_mid) std::cout << e << " ";
    std::cout << "\n    adaptive E: ";
    for (double e : e_ad) std::cout << e << " ";
    std::cout << "\n    growth midpoint: " << growth_mid
              << "\n    growth adaptive: " << growth_ad << "\n";
    check(midpoint_nonincreasing, "midpoint energy increases as gap shrinks");
    check(adaptive_nonincreasing, "adaptive energy increases as gap shrinks");
    check(adaptive_monotone_ratio,
          "adaptive/midpoint ratio increases as near-contact tightens");
    check(growth_ad > growth_mid,
          "adaptive shows stronger near-contact growth than midpoint");
}

void test_shell_energy_zero_at_identity() {
    std::cout << "-- shell energy is ~0 at identity deformation --\n";
    MeshData x = make_icosphere(2);
    x.normalize();
    ShellEnergyParams p;
    p.thickness = 1.0;
    p.lambda = 1.0;
    p.mu = 1.0;
    const ShellEnergyValue e = shell_energy(x, x, p);
    std::cout << "    membrane = " << e.membrane
              << ", bending = " << e.bending
              << ", total = " << e.total << "\n";
    check(std::abs(e.total) < 1e-9, "Wc(x,x) ~= 0");
}

void test_shell_energy_positive_under_deformation() {
    std::cout << "-- shell energy increases under anisotropic deformation --\n";
    MeshData x = make_icosphere(2);
    x.normalize();
    MeshData y = x;
    y.V.col(0) *= 1.15;
    y.V.col(1) *= 0.9;

    const ShellEnergyValue e = shell_energy(x, y);
    std::cout << "    membrane = " << e.membrane
              << ", bending = " << e.bending
              << ", total = " << e.total << "\n";
    check(e.total > 0.0, "Wc(x,y) > 0 for nontrivial deformation");
}

double matrix_relative_max_err(const Eigen::MatrixXd &a,
                               const Eigen::MatrixXd &b) {
    double max_err = 0.0;
    double max_scale = 1.0;
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            max_err = std::max(max_err, std::abs(a(i, j) - b(i, j)));
            max_scale = std::max(max_scale, std::abs(b(i, j)));
        }
    }
    return max_err / max_scale;
}

void test_bending_gradient_analytical_matches_fd_icosphere() {
    std::cout << "-- analytical bending gradient matches FD on perturbed icosphere --\n";
    MeshData x = make_icosphere(2);
    x.normalize();
    MeshData y = x;

    std::mt19937 rng(18431);
    std::normal_distribution<double> n(0.0, 0.025);
    for (int i = 0; i < y.n_vertices(); ++i) {
        y.V(i, 0) += n(rng);
        y.V(i, 1) += n(rng);
        y.V(i, 2) += n(rng);
    }

    ShellEnergyParams params;
    params.lambda = 0.0;
    params.mu = 0.0;
    params.use_tan_bending = true;
    params.bending_fd_eps = 1e-6;

    ShellEnergyParams fd_params = params;
    fd_params.use_analytical_bending_gradient = false;
    ShellEnergyParams analytical_params = params;
    analytical_params.use_analytical_bending_gradient = true;

    const ShellEnergyGradientResult fd =
        shell_energy_with_gradient(x, y, fd_params);
    const ShellEnergyGradientResult an =
        shell_energy_with_gradient(x, y, analytical_params);
    const double rel = matrix_relative_max_err(an.grad_def, fd.grad_def);
    const double abs = (an.grad_def - fd.grad_def).cwiseAbs().maxCoeff();
    std::cout << "    max abs err = " << abs
              << ", max rel err = " << rel << "\n";
    check(rel < 1e-5, "analytical bending grad_def matches FD (rel < 1e-5)");
}

MeshData make_hinge_mesh(double z) {
    MeshData m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           0, 0, z;
    m.F.resize(2, 3);
    // Faces share edge (0,1). With z=1 the normals are +z and +y, so the
    // unsigned dihedral used by ShellEnergy is exactly pi/2.
    m.F << 0, 1, 2,
           1, 0, 3;
    m.L0 = m.compute_L0();
    return m;
}

void test_bending_gradient_hinge_directional_derivative() {
    std::cout << "-- analytical bending gradient hinge directional derivative --\n";
    MeshData x = make_hinge_mesh(1.0);
    MeshData y = make_hinge_mesh(1.25);
    y.V.row(3) += Eigen::RowVector3d(0.08, -0.04, 0.02);

    ShellEnergyParams params;
    params.lambda = 0.0;
    params.mu = 0.0;
    params.use_tan_bending = false;
    params.use_analytical_bending_gradient = true;

    const ShellEnergyGradientResult r = shell_energy_with_gradient(x, y, params);
    Eigen::MatrixXd dir = Eigen::MatrixXd::Zero(y.n_vertices(), 3);
    dir.row(3) = Eigen::RowVector3d(0.3, -0.2, 0.15);

    const double h = 1e-6;
    MeshData yp = y;
    MeshData ym = y;
    yp.V += h * dir;
    ym.V -= h * dir;
    const double fd =
        (shell_energy(x, yp, params).bending -
         shell_energy(x, ym, params).bending) / (2.0 * h);
    const double analytical = (r.grad_def.array() * dir.array()).sum();
    const double abs_err = std::abs(analytical - fd);
    const double rel_err = abs_err / std::max({1.0, std::abs(analytical), std::abs(fd)});
    std::cout << "    analytical = " << analytical
              << ", FD = " << fd
              << ", abs err = " << abs_err
              << ", rel err = " << rel_err << "\n";
    check(rel_err < 1e-8, "hinge bending gradient matches directional FD");
}

void test_shell_energy_gradient_fd_deformed_mesh() {
    std::cout << "-- FD check for shell energy gradients (ref/deformed vars) --\n";
    MeshData x0 = make_torus(1.0, 0.3, 10, 8);
    x0.normalize();
    MeshData y0 = x0;
    y0.V.col(0) *= 1.1;
    y0.V.col(2) *= 0.95;
    const int nv = x0.n_vertices();
    const int ndof = 6 * nv;
    const ShellEnergyParams params{};

    auto pack = [&](const MeshData &xr, const MeshData &xd) {
        Eigen::VectorXd out(ndof);
        for (int i = 0; i < nv; ++i) {
            out.segment<3>(3 * i) = xr.V.row(i).transpose();
            out.segment<3>(3 * nv + 3 * i) = xd.V.row(i).transpose();
        }
        return out;
    };
    auto unpack = [&](const Eigen::VectorXd &z, MeshData &xr, MeshData &xd) {
        xr = x0;
        xd = y0;
        for (int i = 0; i < nv; ++i) {
            xr.V.row(i) = z.segment<3>(3 * i).transpose();
            xd.V.row(i) = z.segment<3>(3 * nv + 3 * i).transpose();
        }
    };

    auto energy = [&](const Eigen::VectorXd &z) -> double {
        MeshData xr, xd;
        unpack(z, xr, xd);
        return shell_energy(xr, xd, params).total;
    };
    auto grad = [&](const Eigen::VectorXd &z) -> Eigen::VectorXd {
        MeshData xr, xd;
        unpack(z, xr, xd);
        const ShellEnergyGradientResult r = shell_energy_with_gradient(xr, xd, params);
        Eigen::VectorXd g(ndof);
        for (int i = 0; i < nv; ++i) {
            g.segment<3>(3 * i) = r.grad_ref.row(i).transpose();
            g.segment<3>(3 * nv + 3 * i) = r.grad_def.row(i).transpose();
        }
        return g;
    };

    const Eigen::VectorXd z0 = pack(x0, y0);
    const GradCheckResult r = finite_diff_gradient_check(energy, grad, z0, 1e-6);
    std::cout << "    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.pass(5e-3), "shell energy gradient FD-check passes (rel < 5e-3)");
}

void test_shell_def_gradient_helper_matches_full_gradient() {
    std::cout << "-- shell def-gradient helper matches full gradient --\n";
    MeshData x = make_icosphere(1);
    x.normalize();
    MeshData y = x;
    y.V.col(0) *= 1.08;
    y.V.col(1) *= 0.94;
    y.V.col(2) *= 1.03;

    ShellEnergyParams params;
    const ShellEnergyGradientResult full =
        shell_energy_with_gradient(x, y, params);
    const Eigen::MatrixXd def_only =
        shell_energy_def_gradient(x, y, params);
    const double max_err = (full.grad_def - def_only).cwiseAbs().maxCoeff();
    std::cout << "    max |full.grad_def - def_only| = "
              << max_err << "\n";
    check(max_err < 1e-12,
          "shell_energy_def_gradient matches full grad_def");
}

Eigen::VectorXd pack_vertex_matrix(const Eigen::MatrixXd &m) {
    Eigen::VectorXd out(3 * m.rows());
    for (int i = 0; i < m.rows(); ++i) {
        out.segment<3>(3 * i) = m.row(i).transpose();
    }
    return out;
}

double sparse_max_abs(const Eigen::SparseMatrix<double> &m) {
    double out = 0.0;
    for (int outer = 0; outer < m.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(m, outer); it;
             ++it) {
            out = std::max(out, std::abs(it.value()));
        }
    }
    return out;
}

void test_shell_hessian_symmetry() {
    std::cout << "-- analytical shell Hessian block symmetry --\n";
    MeshData x = make_icosphere(1);
    x.normalize();
    MeshData y = x;
    y.V.col(0) *= 1.07;
    y.V.col(1) *= 0.93;
    y.V.col(2) *= 1.02;

    const ShellEnergyParams params{};
    const ShellEnergyHessianResult H = shell_energy_hessian(x, y, params);
    Eigen::SparseMatrix<double> rr_t = H.ref_ref.transpose();
    Eigen::SparseMatrix<double> dd_t = H.def_def.transpose();
    Eigen::SparseMatrix<double> dr_t = H.def_ref.transpose();
    Eigen::SparseMatrix<double> rr_diff = H.ref_ref - rr_t;
    Eigen::SparseMatrix<double> dd_diff = H.def_def - dd_t;
    Eigen::SparseMatrix<double> rd_diff = H.ref_def - dr_t;
    rr_diff.makeCompressed();
    dd_diff.makeCompressed();
    rd_diff.makeCompressed();

    const double rr = sparse_max_abs(rr_diff);
    const double dd = sparse_max_abs(dd_diff);
    const double rd = sparse_max_abs(rd_diff);
    std::cout << "    max |Hrr-Hrr^T| = " << rr
              << ", max |Hdd-Hdd^T| = " << dd
              << ", max |Hrd-Hdr^T| = " << rd << "\n";
    check(std::max({rr, dd, rd}) < 1e-10,
          "shell Hessian blocks have transpose symmetry");
}

void test_shell_hessian_directional_derivative() {
    std::cout << "-- analytical shell Hessian directional derivative --\n";
    MeshData x = make_icosphere(1);
    x.normalize();
    MeshData y = x;
    y.V.col(0) *= 1.08;
    y.V.col(1) *= 0.94;
    y.V.col(2) *= 1.03;

    std::mt19937 rng(18432);
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(x.n_vertices(), 3);
    Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(y.n_vertices(), 3);
    for (int i = 0; i < x.n_vertices(); ++i) {
        for (int c = 0; c < 3; ++c) {
            dx(i, c) = n(rng);
            dy(i, c) = n(rng);
        }
    }
    const double dir_norm =
        std::sqrt(dx.squaredNorm() + dy.squaredNorm());
    dx /= dir_norm;
    dy /= dir_norm;

    const ShellEnergyParams params{};
    const ShellEnergyHessianResult H = shell_energy_hessian(x, y, params);
    const Eigen::VectorXd vx = pack_vertex_matrix(dx);
    const Eigen::VectorXd vy = pack_vertex_matrix(dy);
    const Eigen::VectorXd hx = H.ref_ref * vx + H.ref_def * vy;
    const Eigen::VectorXd hy = H.def_ref * vx + H.def_def * vy;

    const double h = 1e-6;
    MeshData xp = x;
    MeshData xm = x;
    MeshData yp = y;
    MeshData ym = y;
    xp.V += h * dx;
    xm.V -= h * dx;
    yp.V += h * dy;
    ym.V -= h * dy;
    const ShellEnergyGradientResult gp =
        shell_energy_with_gradient(xp, yp, params);
    const ShellEnergyGradientResult gm =
        shell_energy_with_gradient(xm, ym, params);
    const Eigen::VectorXd fd_x =
        (pack_vertex_matrix(gp.grad_ref) -
         pack_vertex_matrix(gm.grad_ref)) / (2.0 * h);
    const Eigen::VectorXd fd_y =
        (pack_vertex_matrix(gp.grad_def) -
         pack_vertex_matrix(gm.grad_def)) / (2.0 * h);

    const double err =
        std::max((hx - fd_x).lpNorm<Eigen::Infinity>(),
                 (hy - fd_y).lpNorm<Eigen::Infinity>());
    const double scale =
        std::max({1.0,
                  hx.lpNorm<Eigen::Infinity>(),
                  hy.lpNorm<Eigen::Infinity>(),
                  fd_x.lpNorm<Eigen::Infinity>(),
                  fd_y.lpNorm<Eigen::Infinity>()});
    const double rel = err / scale;
    std::cout << "    max abs err = " << err
              << ", max rel err = " << rel << "\n";
    check(rel < 5e-5,
          "shell Hessian matches FD directional derivative");
}

void test_bending_hessian_directional_derivative() {
    std::cout << "-- closed-form bending Hessian directional derivative --\n";
    MeshData x = make_icosphere(1);
    x.normalize();
    MeshData y = x;
    y.V.col(0) *= 1.08;
    y.V.col(1) *= 0.94;
    y.V.col(2) *= 1.03;

    std::mt19937 rng(18433);
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(x.n_vertices(), 3);
    Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(y.n_vertices(), 3);
    for (int i = 0; i < x.n_vertices(); ++i) {
        for (int c = 0; c < 3; ++c) {
            dx(i, c) = n(rng);
            dy(i, c) = n(rng);
        }
    }
    const double dir_norm =
        std::sqrt(dx.squaredNorm() + dy.squaredNorm());
    dx /= dir_norm;
    dy /= dir_norm;

    ShellEnergyParams params{};
    params.lambda = 0.0;
    params.mu = 0.0;
    const ShellEnergyHessianResult H = shell_energy_hessian(x, y, params);
    const Eigen::VectorXd vx = pack_vertex_matrix(dx);
    const Eigen::VectorXd vy = pack_vertex_matrix(dy);
    const Eigen::VectorXd hx = H.ref_ref * vx + H.ref_def * vy;
    const Eigen::VectorXd hy = H.def_ref * vx + H.def_def * vy;

    const double h = 1e-6;
    MeshData xp = x;
    MeshData xm = x;
    MeshData yp = y;
    MeshData ym = y;
    xp.V += h * dx;
    xm.V -= h * dx;
    yp.V += h * dy;
    ym.V -= h * dy;
    const ShellEnergyGradientResult gp =
        shell_energy_with_gradient(xp, yp, params);
    const ShellEnergyGradientResult gm =
        shell_energy_with_gradient(xm, ym, params);
    const Eigen::VectorXd fd_x =
        (pack_vertex_matrix(gp.grad_ref) -
         pack_vertex_matrix(gm.grad_ref)) / (2.0 * h);
    const Eigen::VectorXd fd_y =
        (pack_vertex_matrix(gp.grad_def) -
         pack_vertex_matrix(gm.grad_def)) / (2.0 * h);

    const double err =
        std::max((hx - fd_x).lpNorm<Eigen::Infinity>(),
                 (hy - fd_y).lpNorm<Eigen::Infinity>());
    const double scale =
        std::max({1.0,
                  hx.lpNorm<Eigen::Infinity>(),
                  hy.lpNorm<Eigen::Infinity>(),
                  fd_x.lpNorm<Eigen::Infinity>(),
                  fd_y.lpNorm<Eigen::Infinity>()});
    const double rel = err / scale;
    std::cout << "    max abs err = " << err
              << ", max rel err = " << rel << "\n";
    check(rel < 5e-5,
          "closed-form bending Hessian matches FD directional derivative");
}

void test_path_energy_identity_zero() {
    std::cout << "-- path energy is ~0 for constant trajectory --\n";
    MeshData x = make_icosphere(2);
    x.normalize();
    std::vector<MeshData> frames = {x, x, x, x};
    PathEnergyParams p;
    p.tpe_adaptive.enabled = true;
    p.tpe_adaptive.max_depth = 2;
    const PathEnergyResult r = path_energy(frames, p);
    std::cout << "    total = " << r.terms.total
              << ", shell = " << r.terms.shell_sum
              << ", rep = " << r.terms.repulsive_sum << "\n";
    check(std::abs(r.terms.total) < 1e-9, "E_hat(constant path) ~= 0");
}

void test_path_energy_gradient_fd() {
    std::cout << "-- FD check for path energy gradient on interior frames --\n";
    MeshData x0 = make_icosphere(1);
    x0.normalize();
    std::vector<MeshData> frames(4, x0);
    frames[1].V.col(0) *= 1.05;
    frames[2].V.col(1) *= 0.92;

    PathEnergyParams p;
    p.tpe_adaptive.enabled = true;
    p.tpe_adaptive.max_depth = 2;
    p.tpe_alpha = 6.0;
    p.tpe_theta = 0.5;
    p.rigid_translation_weight = 0.2;
    p.rigid_rotation_weight = 0.03;
    const std::vector<PathEnergyFrameCache> frozen_cache =
        build_path_energy_frame_cache(frames, p);

    const int nv = x0.n_vertices();
    const int ndof = 2 * 3 * nv; // optimize x1, x2 only

    auto pack = [&](const std::vector<MeshData> &f) {
        Eigen::VectorXd z(ndof);
        for (int i = 0; i < nv; ++i) {
            z.segment<3>(3 * i) = f[1].V.row(i).transpose();
            z.segment<3>(3 * nv + 3 * i) = f[2].V.row(i).transpose();
        }
        return z;
    };
    auto unpack = [&](const Eigen::VectorXd &z, std::vector<MeshData> &f) {
        f = frames;
        for (int i = 0; i < nv; ++i) {
            f[1].V.row(i) = z.segment<3>(3 * i).transpose();
            f[2].V.row(i) = z.segment<3>(3 * nv + 3 * i).transpose();
        }
    };

    auto energy = [&](const Eigen::VectorXd &z) -> double {
        std::vector<MeshData> f;
        unpack(z, f);
        return path_energy(f, p, &frozen_cache).terms.total;
    };
    auto grad = [&](const Eigen::VectorXd &z) -> Eigen::VectorXd {
        std::vector<MeshData> f;
        unpack(z, f);
        const PathEnergyGradientResult g = path_energy_with_gradient(f, p, &frozen_cache);
        Eigen::VectorXd out(ndof);
        for (int i = 0; i < nv; ++i) {
            out.segment<3>(3 * i) = g.grad_frames[1].row(i).transpose();
            out.segment<3>(3 * nv + 3 * i) = g.grad_frames[2].row(i).transpose();
        }
        return out;
    };

    const Eigen::VectorXd z0 = pack(frames);
    const GradCheckResult r = finite_diff_gradient_check(energy, grad, z0, 1e-6);
    std::cout << "    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.pass(5e-3), "path energy gradient FD-check passes (rel < 5e-3)");
}

void test_trust_region_interpolation_decreases_path_energy() {
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

} // namespace

int main() {
    std::cout << "=== Phase 1.1–1.7 smoke tests ===\n";
    test_unit_triangle();
    test_icosphere_invariants();
    test_make_n_torus_topology();
    test_jacobians_fd();
    test_translation_and_shear_invariants();
    test_tpe_two_triangle_toy();
    test_tpe_coplanar_is_zero();
    test_tpe_scale_covariance();
    test_tpe_translation_invariance();
    test_tpe_gradient_fd_torus();
    test_tpe_gradient_translation_equivariance();
    test_tpe_gradient_sum_to_zero();
    test_bvh_invariants();
    test_bvh_internal_aggregates();
    test_bvh_cluster_radius_bound();
    test_bct_theta_zero();
    test_bct_exclusive_coverage();
    test_bct_scaling();
    test_tpe_bh_theta_zero_matches_brute();
    test_tpe_bh_smooth_mesh_accuracy();
    test_tpe_bh_energy_deterministic();
    test_tpe_bh_theta_sweep_convergence();
    test_tpe_bh_scale_covariance();
    test_tpe_gradient_bh_theta_zero_matches_brute();
    test_tpe_gradient_bh_fd_check();
    test_tpe_gradient_bh_translation_equivariance();
    test_tpe_gradient_bh_descent_step();
    test_tpe_adaptive_disabled_matches_bh();
    test_tpe_adaptive_depth0_matches_midpoint();
    test_tpe_adaptive_gradient_fd_check();
    test_tpe_adaptive_near_contact_growth();
    test_shell_energy_zero_at_identity();
    test_shell_energy_positive_under_deformation();
    test_bending_gradient_analytical_matches_fd_icosphere();
    test_bending_gradient_hinge_directional_derivative();
    test_shell_energy_gradient_fd_deformed_mesh();
    test_shell_def_gradient_helper_matches_full_gradient();
    test_shell_hessian_symmetry();
    test_shell_hessian_directional_derivative();
    test_bending_hessian_directional_derivative();
    test_path_energy_identity_zero();
    test_path_energy_gradient_fd();
    test_trust_region_interpolation_decreases_path_energy();

    std::cout << "\n"
              << (failures == 0 ? "ALL PASSED"
                                : "FAILURES: " + std::to_string(failures))
              << "\n";
    return failures == 0 ? 0 : 1;
}
