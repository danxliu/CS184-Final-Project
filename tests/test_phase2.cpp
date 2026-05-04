#include "GradCheck.h"
#include "MeshData.h"
#include "TestMeshes.h"
#include "PathEnergy.h"
#include "TrustRegionSolver.h"
#include "ExtrapolationSolver.h"
#include "BCT.h"
#include "BVH.h"
#include "Constraints.h"
#include "FaceGeom.h"
#include "HsPreconditioner.h"
#include "OptimizeTPE.h"
#include "Remesh.h"
#include "TPE.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
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

struct MeshQuality {
    double min_edge = std::numeric_limits<double>::infinity();
    double max_edge = 0.0;
    int min_valence = std::numeric_limits<int>::max();
    int max_valence = 0;
    double max_aspect = 0.0;
    double mean_aspect = 0.0;
    bool edge_manifold = true;
};

std::pair<int, int> edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

MeshQuality measure_mesh_quality(const MeshData &m) {
    MeshQuality q;
    std::map<std::pair<int, int>, int> edges;
    std::vector<std::set<int>> nbr(static_cast<size_t>(m.n_vertices()));
    for (int f = 0; f < m.n_faces(); ++f) {
        const Eigen::Vector3d p0 = m.V.row(m.F(f, 0));
        const Eigen::Vector3d p1 = m.V.row(m.F(f, 1));
        const Eigen::Vector3d p2 = m.V.row(m.F(f, 2));
        const double area2 = (p1 - p0).cross(p2 - p0).norm();
        double longest = 0.0;
        for (int k = 0; k < 3; ++k) {
            const int a = m.F(f, k);
            const int b = m.F(f, (k + 1) % 3);
            edges[edge_key(a, b)]++;
            nbr[static_cast<size_t>(a)].insert(b);
            nbr[static_cast<size_t>(b)].insert(a);
            longest = std::max(longest, (m.V.row(a) - m.V.row(b)).norm());
        }
        if (area2 > 0.0) {
            const double aspect = longest * longest / area2;
            q.max_aspect = std::max(q.max_aspect, aspect);
            q.mean_aspect += aspect;
        } else {
            q.max_aspect = std::numeric_limits<double>::infinity();
            q.mean_aspect = std::numeric_limits<double>::infinity();
        }
    }
    if (m.n_faces() > 0 && std::isfinite(q.mean_aspect)) {
        q.mean_aspect /= static_cast<double>(m.n_faces());
    }

    for (const auto &item : edges) {
        q.edge_manifold = q.edge_manifold && item.second <= 2;
        const double len = (m.V.row(item.first.first) -
                            m.V.row(item.first.second)).norm();
        q.min_edge = std::min(q.min_edge, len);
        q.max_edge = std::max(q.max_edge, len);
    }
    if (edges.empty()) q.min_edge = 0.0;

    for (const auto &n : nbr) {
        const int val = static_cast<int>(n.size());
        q.min_valence = std::min(q.min_valence, val);
        q.max_valence = std::max(q.max_valence, val);
    }
    if (nbr.empty()) q.min_valence = 0;
    return q;
}

bool has_vertex_near(const MeshData &m, const Eigen::Vector3d &p, double tol) {
    for (int i = 0; i < m.n_vertices(); ++i) {
        if ((m.V.row(i).transpose() - p).norm() <= tol) return true;
    }
    return false;
}

bool has_edge(const MeshData &m, int a, int b) {
    const auto key = edge_key(a, b);
    for (int f = 0; f < m.n_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            if (edge_key(m.F(f, k), m.F(f, (k + 1) % 3)) == key) {
                return true;
            }
        }
    }
    return false;
}

Eigen::Vector3d triangle_circumcenter_test(const Eigen::Vector3d &a,
                                           const Eigen::Vector3d &b,
                                           const Eigen::Vector3d &c) {
    const Eigen::Vector3d u = b - a;
    const Eigen::Vector3d v = c - a;
    const Eigen::Vector3d n = u.cross(v);
    const double n2 = n.squaredNorm();
    if (!(n2 > 1e-30)) return (a + b + c) / 3.0;
    return a + (u.squaredNorm() * v.cross(n) +
                v.squaredNorm() * n.cross(u)) / (2.0 * n2);
}

Eigen::Vector3d vertex_normal(const MeshData &m, int vertex) {
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    for (int f = 0; f < m.n_faces(); ++f) {
        bool incident = false;
        for (int k = 0; k < 3; ++k) incident = incident || m.F(f, k) == vertex;
        if (!incident) continue;
        const Eigen::Vector3d p0 = m.V.row(m.F(f, 0)).transpose();
        const Eigen::Vector3d p1 = m.V.row(m.F(f, 1)).transpose();
        const Eigen::Vector3d p2 = m.V.row(m.F(f, 2)).transpose();
        n += (p1 - p0).cross(p2 - p0);
    }
    const double nn = n.norm();
    if (nn > 0.0) return n / nn;
    return Eigen::Vector3d::UnitZ();
}

MeshData make_smoothing_fan() {
    MeshData m;
    m.V.resize(7, 3);
    m.V.row(0) << 0.2, 0.0, 0.0;
    for (int i = 0; i < 6; ++i) {
        const double a = 2.0 * M_PI * static_cast<double>(i) / 6.0;
        m.V.row(i + 1) << std::cos(a), std::sin(a), 0.0;
    }
    m.F.resize(6, 3);
    for (int i = 0; i < 6; ++i) {
        m.F.row(i) << 0, i + 1, 1 + ((i + 1) % 6);
    }
    m.L0 = m.compute_L0();
    return m;
}

Eigen::Vector3d smoothing_target_for_vertex(const MeshData &m, int vertex) {
    Eigen::Vector3d weighted = Eigen::Vector3d::Zero();
    double area_sum = 0.0;
    for (int f = 0; f < m.n_faces(); ++f) {
        bool incident = false;
        for (int k = 0; k < 3; ++k) incident = incident || m.F(f, k) == vertex;
        if (!incident) continue;
        const Eigen::Vector3d p0 = m.V.row(m.F(f, 0)).transpose();
        const Eigen::Vector3d p1 = m.V.row(m.F(f, 1)).transpose();
        const Eigen::Vector3d p2 = m.V.row(m.F(f, 2)).transpose();
        const double area = 0.5 * (p1 - p0).cross(p2 - p0).norm();
        weighted += area * triangle_circumcenter_test(p0, p1, p2);
        area_sum += area;
    }
    if (area_sum > 0.0) return weighted / area_sum;
    return m.V.row(vertex).transpose();
}

MeshData deterministically_perturbed_icosphere(int subdiv) {
    MeshData m = make_icosphere(subdiv);
    m.normalize();
    const double amp = 0.6 * m.L0;
    for (int i = 0; i < m.n_vertices(); ++i) {
        Eigen::Vector3d p = m.V.row(i).transpose();
        const double s = std::sin(1.7 * p.x() + 2.3 * p.y() - 0.9 * p.z());
        m.V.row(i) += (amp * s * p.normalized()).transpose();
    }
    return m;
}

MeshData make_scaled_icosphere(int subdiv) {
    MeshData m = make_icosphere(subdiv);
    m.V.col(0) *= 1.7;
    m.V.col(1) *= 0.75;
    m.V.col(2) *= 0.75;
    m.normalize();
    return m;
}

double bbox_aspect(const MeshData &m) {
    const Eigen::Vector3d mn = m.V.colwise().minCoeff();
    const Eigen::Vector3d mx = m.V.colwise().maxCoeff();
    const Eigen::Vector3d ext = (mx - mn).eval();
    const double min_ext = ext.minCoeff();
    if (!(min_ext > 0.0)) return std::numeric_limits<double>::infinity();
    return ext.maxCoeff() / min_ext;
}

double tpe_energy_for_test(const MeshData &m, double alpha, double theta) {
    const FaceGeom g = compute_face_geom(m);
    const BVH bvh = build_bvh(m, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    return tpe_energy_bh(g, bvh, bp, alpha);
}

void reset_test_dir(const std::string &dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
}

struct EnergyRow {
    int iter = 0;
    double energy = 0.0;
    double grad_norm = 0.0;
    double step_size = 0.0;
    int n_backtracks = 0;
    int did_remesh = 0;
};

std::vector<EnergyRow> read_energy_rows(const std::string &path) {
    std::ifstream in(path);
    std::vector<EnergyRow> rows;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        EnergyRow r;
        if (iss >> r.iter >> r.energy >> r.grad_norm >> r.step_size >>
            r.n_backtracks >> r.did_remesh) {
            rows.push_back(r);
        }
    }
    return rows;
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

double max_abs_column_sum(const Eigen::MatrixXd &x) {
    return x.colwise().sum().cwiseAbs().maxCoeff();
}

void test_project_barycenter_zero_mean() {
    std::cout << "-- barycenter projection has zero column sums --\n";
    Eigen::MatrixXd d(17, 3);
    for (int i = 0; i < d.rows(); ++i) {
        for (int j = 0; j < d.cols(); ++j) {
            d(i, j) = std::sin(0.37 * (i + 1) * (j + 2)) +
                      0.2 * std::cos(0.11 * (i + 3) * (j + 1));
        }
    }

    project_barycenter(d);
    const double max_sum = max_abs_column_sum(d);
    std::cout << "    max |sum column| = " << max_sum << "\n";
    check(max_sum < 1e-14, "project_barycenter makes each component sum to zero");
}

void test_apply_pin_mask_zeroes_only_pinned_rows() {
    std::cout << "-- pin mask zeroes pinned rows only --\n";
    Eigen::MatrixXd d(12, 3);
    for (int i = 0; i < d.rows(); ++i) {
        d.row(i) = Eigen::RowVector3d(0.1 * (i + 1),
                                      -0.2 * (i + 2),
                                      0.3 * (i + 3));
    }
    const Eigen::MatrixXd before = d;
    std::vector<bool> pin(static_cast<size_t>(d.rows()), false);
    pin[0] = true;
    pin[5] = true;
    pin[11] = true;

    apply_pin_mask(d, pin);

    double max_pinned = 0.0;
    double max_unpinned_delta = 0.0;
    for (int i = 0; i < d.rows(); ++i) {
        if (pin[static_cast<size_t>(i)]) {
            max_pinned = std::max(max_pinned, d.row(i).norm());
        } else {
            max_unpinned_delta =
                std::max(max_unpinned_delta, (d.row(i) - before.row(i)).norm());
        }
    }
    std::cout << "    max pinned norm = " << max_pinned
              << ", max unpinned delta = " << max_unpinned_delta << "\n";
    check(max_pinned == 0.0, "apply_pin_mask zeros pinned rows exactly");
    check(max_unpinned_delta == 0.0, "apply_pin_mask leaves unpinned rows unchanged");
}

void test_barycenter_fixed_under_hs_tpe_descent() {
    std::cout << "-- barycenter-constrained Hs TPE descent preserves centroid --\n";
    MeshData m = make_icosphere(2);
    m.normalize();

    std::mt19937 rng(18437);
    std::normal_distribution<double> normal(0.0, 0.01);
    for (int i = 0; i < m.n_vertices(); ++i) {
        for (int j = 0; j < 3; ++j) {
            m.V(i, j) += normal(rng);
        }
    }
    const Eigen::RowVector3d c0 = m.V.colwise().mean();

    HsPreconditionerParams hs_params;
    hs_params.s = 5.0 / 3.0;
    hs_params.sigma = 1.0;
    hs_params.theta = 0.25;
    HsConstraints constraints;
    constraints.pin_barycenter = true;

    double max_step_sum = 0.0;
    double min_g_dot_dir = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 10; ++it) {
        const FaceGeom g = compute_face_geom(m);
        const BVH bvh = build_bvh(m, g);
        const BlockPairs bp = build_bct_self(bvh, 0.5);
        Eigen::MatrixXd G = tpe_gradient_bh(m, g, bvh, bp, 6.0);

        const HsDirectionResult hs =
            hs_preconditioned_direction(m, G, hs_params, constraints);
        max_step_sum = std::max(max_step_sum, max_abs_column_sum(hs.direction));
        min_g_dot_dir = std::min(min_g_dot_dir, hs.g_dot_dir);

        const double tau = 1e-3 / std::max(1.0, hs.direction.norm());
        m.V -= tau * hs.direction;
    }

    const Eigen::RowVector3d c1 = m.V.colwise().mean();
    const double drift = (c1 - c0).norm();
    std::cout << "    centroid drift = " << drift
              << ", max |sum direction column| = " << max_step_sum
              << ", min g_dot_dir = " << min_g_dot_dir << "\n";
    check(max_step_sum < 1e-10, "Hs constrained direction has zero column sums");
    check(drift < 1e-10, "barycenter remains fixed after constrained Hs steps");
    check(min_g_dot_dir > 0.0, "constrained Hs directions remain descent directions");
}

void test_remesh_single_edge_split() {
    std::cout << "-- remesh single-edge split --\n";
    MeshData m;
    m.V.resize(3, 3);
    m.V << 0.0, 0.0, 0.0,
           1.6, 0.0, 0.0,
           0.8, 1.0, 0.0;
    m.F.resize(1, 3);
    m.F << 0, 1, 2;
    m.L0 = 1.0;

    const MeshData r = remesh_split_collapse(m);
    std::cout << "    vertices = " << r.n_vertices()
              << ", faces = " << r.n_faces() << "\n";
    check(r.n_vertices() == 4, "split inserts one midpoint vertex");
    check(r.n_faces() == 2, "split replaces one triangle by two triangles");
    check(has_vertex_near(r, Eigen::Vector3d(0.8, 0.0, 0.0), 1e-12),
          "split midpoint is on the long edge");
}

void test_remesh_single_edge_collapse() {
    std::cout << "-- remesh single-edge collapse --\n";
    MeshData m;
    m.V.resize(4, 3);
    m.V << 0.0, 0.0, 0.0,
           0.5, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0,-1.0, 0.0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           1, 0, 3;
    m.L0 = 1.0;

    const MeshData r = remesh_split_collapse(m);
    std::cout << "    vertices = " << r.n_vertices()
              << ", faces = " << r.n_faces() << "\n";
    check(r.n_faces() == 0, "collapse removes the two incident triangles");
    check(has_vertex_near(r, Eigen::Vector3d(0.25, 0.0, 0.0), 1e-12),
          "collapse keeps merged vertex at midpoint");
}

void test_remesh_foldover_rejection() {
    std::cout << "-- remesh foldover rejection --\n";
    MeshData m;
    m.V.resize(6, 3);
    m.V << 0.0, 0.0, 0.0,
           0.5, 0.0, 0.0,
           0.0, 3.0, 0.0,
           0.0,-3.0, 0.0,
           0.4,-3.0, 0.0,
           0.4, 3.0, 0.0;
    m.F.resize(3, 3);
    m.F << 0, 1, 2,
           1, 0, 3,
           1, 4, 5;
    m.L0 = 5.0;

    const MeshData r = remesh_split_collapse(m);
    std::cout << "    vertices = " << r.n_vertices()
              << ", faces = " << r.n_faces() << "\n";
    check(r.n_vertices() == m.n_vertices() && r.n_faces() == m.n_faces(),
          "foldover candidate collapse is rejected");
}

void test_remesh_valence_guard_rejection() {
    std::cout << "-- remesh valence-guard rejection --\n";
    MeshData m;
    m.V.resize(6, 3);
    m.V << 0.0, 0.0, 0.0,
           0.5, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0,-1.0, 0.0,
           1.0, 1.0, 0.0,
           1.0,-1.0, 0.0;
    m.F.resize(4, 3);
    m.F << 0, 1, 2,
           1, 0, 3,
           1, 4, 2,
           1, 3, 5;
    m.L0 = 1.6;

    const MeshData r = remesh_split_collapse(m);
    std::cout << "    vertices = " << r.n_vertices()
              << ", faces = " << r.n_faces() << "\n";
    check(r.n_vertices() == m.n_vertices() && r.n_faces() == m.n_faces(),
          "collapse that would leave low-valence vertices is rejected");
}

void test_remesh_icosphere2_bounds() {
    std::cout << "-- remesh perturbed icosphere_2 bounds --\n";
    MeshData m = deterministically_perturbed_icosphere(2);
    const int f0 = m.n_faces();
    const MeshData r = remesh_split_collapse(m);
    const MeshQuality q = measure_mesh_quality(r);
    std::cout << "    faces = " << r.n_faces()
              << ", min_edge = " << q.min_edge
              << ", max_edge = " << q.max_edge
              << ", valence = [" << q.min_valence << ", "
              << q.max_valence << "]\n";
    check(q.max_edge <= (4.0 / 3.0) * m.L0 * 1.05,
          "post-remesh max edge stays below split threshold slack");
    check(q.min_edge >= (4.0 / 5.0) * m.L0 * 0.95,
          "post-remesh min edge stays above collapse threshold slack");
    check(r.n_faces() >= 0.5 * f0 && r.n_faces() <= 2.0 * f0,
          "post-remesh face count remains in sanity range");
    check(q.edge_manifold, "post-remesh edges have at most two incident faces");
    check(q.min_valence >= 3, "post-remesh vertices keep valence at least three");
}

void test_remesh_icosphere3_quality() {
    std::cout << "-- remesh perturbed icosphere_3 quality --\n";
    MeshData m = deterministically_perturbed_icosphere(3);
    const int f0 = m.n_faces();
    const MeshData r = remesh_split_collapse(m);
    const MeshQuality q = measure_mesh_quality(r);
    std::cout << "    faces = " << r.n_faces()
              << ", max_aspect = " << q.max_aspect
              << ", min_edge = " << q.min_edge
              << ", max_edge = " << q.max_edge << "\n";
    check(q.max_aspect < 50.0, "icosphere_3 remesh has no degenerate triangles");
    check(q.edge_manifold, "icosphere_3 remesh remains edge-manifold");
    check(r.n_faces() >= 0.5 * f0 && r.n_faces() <= 2.0 * f0,
          "icosphere_3 remesh face count remains reasonable");
}

void test_remesh_delaunay_flip_yes_case() {
    std::cout << "-- remesh Delaunay flip yes-case --\n";
    MeshData m;
    m.V.resize(4, 3);
    m.V << 0.0, 0.0, 0.0,
           2.0, 0.0, 0.0,
           2.0, 1.0, 0.0,
           0.0, 0.2, 0.0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           0, 2, 3;
    m.L0 = m.compute_L0();

    const MeshData r = remesh_delaunay_flip(m);
    std::cout << "    had edge 0-2 = " << has_edge(m, 0, 2)
              << ", result edge 1-3 = " << has_edge(r, 1, 3) << "\n";
    check(!has_edge(r, 0, 2), "non-Delaunay diagonal is removed");
    check(has_edge(r, 1, 3), "Delaunay diagonal is inserted");
}

void test_remesh_delaunay_flip_no_case() {
    std::cout << "-- remesh Delaunay flip no-case --\n";
    MeshData m;
    m.V.resize(4, 3);
    m.V << 0.0, 0.0, 0.0,
           2.0, 0.0, 0.0,
           2.0, 1.0, 0.0,
           0.0, 0.2, 0.0;
    m.F.resize(2, 3);
    m.F << 0, 1, 3,
           1, 2, 3;
    m.L0 = m.compute_L0();

    const MeshData r = remesh_delaunay_flip(m);
    check(has_edge(r, 1, 3), "Delaunay diagonal remains in place");
    check(!has_edge(r, 0, 2), "no alternate diagonal is inserted");
}

void test_remesh_tangential_smooth_circumcenter_step() {
    std::cout << "-- remesh tangential smooth moves toward circumcenter target --\n";
    MeshData m = make_smoothing_fan();
    const Eigen::Vector3d p0 = m.V.row(0).transpose();
    const Eigen::Vector3d target = smoothing_target_for_vertex(m, 0);
    const MeshData r = remesh_tangential_smooth(m, 0.5, 1);
    const Eigen::Vector3d p1 = r.V.row(0).transpose();
    const Eigen::Vector3d expected = p0 + 0.5 * (target - p0);
    const double err = (p1 - expected).norm();
    std::cout << "    target = " << target.transpose()
              << ", moved = " << (p1 - p0).transpose()
              << ", err = " << err << "\n";
    check(err < 1e-12, "center vertex follows rho-scaled circumcenter target");
}

void test_remesh_tangential_smooth_tangent_plane() {
    std::cout << "-- remesh tangential smooth stays tangent on icosphere --\n";
    MeshData m = make_icosphere(2);
    m.normalize();
    double max_normal_motion = 0.0;
    for (int iter = 0; iter < 5; ++iter) {
        std::vector<Eigen::Vector3d> normals(static_cast<size_t>(m.n_vertices()));
        for (int v = 0; v < m.n_vertices(); ++v) {
            normals[static_cast<size_t>(v)] = vertex_normal(m, v);
        }
        const MeshData next = remesh_tangential_smooth(m, 0.5, 1);
        for (int v = 0; v < m.n_vertices(); ++v) {
            const Eigen::Vector3d delta =
                next.V.row(v).transpose() - m.V.row(v).transpose();
            max_normal_motion = std::max(
                max_normal_motion,
                std::abs(delta.dot(normals[static_cast<size_t>(v)])));
        }
        m = next;
    }
    std::cout << "    max normal displacement component = "
              << max_normal_motion << "\n";
    check(max_normal_motion < 1e-10,
          "smoothing displacement is tangent to the pre-step normals");
}

void test_remesh_full_icosphere2_quality() {
    std::cout << "-- remesh_full perturbed icosphere_2 quality --\n";
    MeshData m = deterministically_perturbed_icosphere(2);
    const MeshData sc = remesh_split_collapse(m);
    const MeshQuality q_sc = measure_mesh_quality(sc);
    const MeshData flipped = remesh_delaunay_flip(sc);
    const MeshData full = remesh_tangential_smooth(flipped);
    const MeshQuality q = measure_mesh_quality(full);
    const double displacement = (full.n_vertices() == flipped.n_vertices())
                                    ? (full.V - flipped.V).rowwise().norm().maxCoeff()
                                    : 0.0;
    std::cout << "    split/collapse mean_aspect = " << q_sc.mean_aspect
              << ", full mean_aspect = " << q.mean_aspect
              << ", full max_aspect = " << q.max_aspect
              << ", displacement = " << displacement << "\n";
    check(q.max_edge <= (4.0 / 3.0) * m.L0 * 1.05,
          "remesh_full max edge stays below split threshold slack");
    check(q.min_edge >= (4.0 / 5.0) * m.L0 * 0.95,
          "remesh_full min edge stays above collapse threshold slack");
    check(q.mean_aspect <= q_sc.mean_aspect + 1e-12,
          "flip and smooth do not worsen mean aspect ratio");
    check(q.edge_manifold, "remesh_full output remains edge-manifold");
    check(displacement < 0.1 * m.bbox_diagonal(),
          "smoothing displacement remains small relative to mesh diameter");
}

void test_remesh_full_icosphere3_quality() {
    std::cout << "-- remesh_full perturbed icosphere_3 quality --\n";
    MeshData m = deterministically_perturbed_icosphere(3);
    const MeshQuality q0 = measure_mesh_quality(m);
    const MeshData full = remesh_full(m);
    const MeshQuality q = measure_mesh_quality(full);
    std::cout << "    input mean_aspect = " << q0.mean_aspect
              << ", full mean_aspect = " << q.mean_aspect
              << ", full max_aspect = " << q.max_aspect << "\n";
    check(q.mean_aspect <= q0.mean_aspect + 1e-12,
          "icosphere_3 full remesh improves mean aspect ratio");
    check(q.max_aspect < 5.0, "icosphere_3 full remesh max aspect stays below 5");
    check(q.edge_manifold, "icosphere_3 full remesh remains edge-manifold");
}

void test_optimize_tpe_empty_call() {
    std::cout << "-- optimize_tpe empty-call sanity --\n";
    MeshData m = make_scaled_icosphere(2);
    OptimizeTPEParams p;
    p.max_iters = 0;
    const OptimizeTPEResult r = optimize_tpe(m, p);
    const double vertex_delta = (r.final_mesh.V - m.V).norm();
    std::cout << "    iterations = " << r.iterations_completed
              << ", vertex_delta = " << vertex_delta
              << ", stop = " << r.stop_reason << "\n";
    check(r.iterations_completed == 0, "empty optimize_tpe returns zero iterations");
    check(vertex_delta == 0.0, "empty optimize_tpe leaves vertices unchanged");
}

void test_optimize_tpe_single_iteration() {
    std::cout << "-- optimize_tpe single iteration --\n";
    const std::string out_dir = "/tmp/rsh_opt_single";
    reset_test_dir(out_dir);
    MeshData m = make_scaled_icosphere(2);
    const Eigen::RowVector3d c0 = m.V.colwise().mean();
    OptimizeTPEParams p;
    p.max_iters = 1;
    p.out_dir = out_dir;
    const double e0 = tpe_energy_for_test(m, p.tpe_alpha, p.bvh_theta);
    const OptimizeTPEResult r = optimize_tpe(m, p);
    const double drift = (r.final_mesh.V.colwise().mean() - c0).norm();
    const bool frame_written =
        std::filesystem::exists(out_dir + "/frame_0001.obj");
    std::cout << "    E0 = " << e0
              << ", E1 = " << r.final_energy
              << ", drift = " << drift
              << ", frame_0001 = " << frame_written << "\n";
    check(r.final_energy < e0, "single optimize_tpe step decreases energy");
    check(drift < 1e-12, "single optimize_tpe step preserves barycenter");
    check(frame_written, "single optimize_tpe writes frame_0001.obj");
}

void test_optimize_tpe_icosphere2_twenty_iters() {
    std::cout << "-- optimize_tpe 20-iter icosphere_2 smoke --\n";
    const std::string out_dir = "/tmp/rsh_opt_ico2";
    reset_test_dir(out_dir);
    MeshData m = make_scaled_icosphere(2);
    const Eigen::RowVector3d c0 = m.V.colwise().mean();
    OptimizeTPEParams p;
    p.max_iters = 20;
    p.out_dir = out_dir;
    const OptimizeTPEResult r = optimize_tpe(m, p);
    const std::vector<EnergyRow> rows = read_energy_rows(out_dir + "/energy.csv");
    bool monotone = rows.size() >= 2;
    int max_bt = 0;
    int remesh_rows = 0;
    for (size_t i = 1; i < rows.size(); ++i) {
        monotone = monotone && rows[i].energy <= rows[i - 1].energy + 1e-8;
        max_bt = std::max(max_bt, rows[i].n_backtracks);
        remesh_rows += rows[i].did_remesh;
    }
    const double drift = (r.final_mesh.V.colwise().mean() - c0).norm();
    std::cout << "    rows = " << rows.size()
              << ", final_E = " << r.final_energy
              << ", remeshes = " << r.remeshes_completed
              << ", max_bt = " << max_bt
              << ", drift = " << drift << "\n";
    check(monotone, "20-iter optimize_tpe energy log is monotone");
    check(max_bt <= p.armijo_max_backtracks, "20-iter optimize_tpe backtracks stay capped");
    check(drift < 1e-10, "20-iter optimize_tpe preserves barycenter");
    check(r.iterations_completed < p.remesh_every ||
              (r.remeshes_completed >= 1 && remesh_rows >= 1),
          "20-iter optimize_tpe fires remeshing if it reaches the remesh interval");
}

void test_optimize_tpe_icosphere3_fifty_iters() {
    std::cout << "-- optimize_tpe 50-iter icosphere_3 completion smoke --\n";
    const std::string out_dir = "/tmp/rsh_opt_ico3";
    reset_test_dir(out_dir);
    MeshData m = make_scaled_icosphere(3);
    const double e0 = tpe_energy_for_test(m, 6.0, 0.5);
    const double aspect0 = bbox_aspect(m);
    const int f0 = m.n_faces();
    OptimizeTPEParams p;
    p.max_iters = 50;
    p.out_dir = out_dir;
    const OptimizeTPEResult r = optimize_tpe(m, p);
    const MeshQuality q = measure_mesh_quality(r.final_mesh);
    const double aspect1 = bbox_aspect(r.final_mesh);
    std::cout << "    E0 = " << e0
              << ", E1 = " << r.final_energy
              << ", aspect0 = " << aspect0
              << ", aspect1 = " << aspect1
              << ", remeshes = " << r.remeshes_completed
              << ", faces = " << r.final_mesh.n_faces() << "\n";
    check(r.final_energy <= 0.5 * e0,
          "50-iter optimize_tpe cuts icosphere_3 energy by at least half");
    check(std::abs(aspect1 - 1.0) < std::abs(aspect0 - 1.0),
          "50-iter optimize_tpe moves bbox aspect ratio toward one");
    const int scheduled_remeshes =
        p.remesh_every > 0 ? r.iterations_completed / p.remesh_every : 0;
    check(r.remeshes_completed >= scheduled_remeshes,
          "50-iter optimize_tpe fires reached scheduled remeshes");
    check(q.edge_manifold, "50-iter optimize_tpe final mesh remains edge-manifold");
    check(r.final_mesh.n_faces() >= 0.5 * f0 &&
              r.final_mesh.n_faces() <= 2.0 * f0,
          "50-iter optimize_tpe face count remains bounded");
}

} // namespace

int main() {
    std::cout << "=== Phase 2/3 smoke tests ===\n";
    test_project_barycenter_zero_mean();
    test_apply_pin_mask_zeroes_only_pinned_rows();
    test_barycenter_fixed_under_hs_tpe_descent();
    test_remesh_single_edge_split();
    test_remesh_single_edge_collapse();
    test_remesh_foldover_rejection();
    test_remesh_valence_guard_rejection();
    test_remesh_icosphere2_bounds();
    test_remesh_icosphere3_quality();
    test_remesh_delaunay_flip_yes_case();
    test_remesh_delaunay_flip_no_case();
    test_remesh_tangential_smooth_circumcenter_step();
    test_remesh_tangential_smooth_tangent_plane();
    test_remesh_full_icosphere2_quality();
    test_remesh_full_icosphere3_quality();
    test_optimize_tpe_empty_call();
    test_optimize_tpe_single_iteration();
    test_optimize_tpe_icosphere2_twenty_iters();
    test_optimize_tpe_icosphere3_fifty_iters();
    test_interpolation_decreases_energy();
    test_extrapolation_stability();
    test_extrapolation_with_repulsion();

    std::cout << "\n" << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
