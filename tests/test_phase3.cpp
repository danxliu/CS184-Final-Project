#include "GradCheck.h"
#include "MeshData.h"
#include "Obstacle.h"
#include "PathEnergy.h"
#include "SurfaceBarrier.h"
#include "TestMeshes.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
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

bool close(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

bool close_vec(const Eigen::Vector3d &a,
               const Eigen::Vector3d &b,
               double tol = 1e-12) {
    return (a - b).norm() <= tol;
}

Eigen::VectorXd flatten_matrix(const Eigen::MatrixXd &M) {
    return Eigen::Map<const Eigen::VectorXd>(M.data(), M.size());
}

Eigen::MatrixXd unflatten_vertices(const Eigen::VectorXd &x) {
    return Eigen::Map<const Eigen::Matrix<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::ColMajor>>(
        x.data(), x.size() / 3, 3);
}

MeshData with_vertices(const MeshData &base, const Eigen::VectorXd &x) {
    MeshData out = base;
    out.V = unflatten_vertices(x);
    out.L0 = out.compute_L0();
    return out;
}

double min_signed_distance(const MeshData &mesh, const Obstacle &obs) {
    double min_phi = std::numeric_limits<double>::infinity();
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        min_phi = std::min(
            min_phi,
            obs.signed_distance(mesh.V.row(i).transpose()));
    }
    return min_phi;
}

void test_sdf_hand_cases() {
    std::cout << "-- obstacle SDF hand cases --\n";

    SphereObstacle sphere(Eigen::Vector3d::Zero(), 1.0);
    check(close(sphere.signed_distance({2.0, 0.0, 0.0}), 1.0),
          "sphere distance outside");
    check(close(sphere.signed_distance(Eigen::Vector3d::Zero()), -1.0),
          "sphere distance inside");
    check(close_vec(sphere.signed_distance_gradient({2.0, 0.0, 0.0}),
                    Eigen::Vector3d::UnitX()),
          "sphere gradient outside");

    CapsuleObstacle capsule({0.0, 0.0, 0.0},
                            {0.0, 0.0, 1.0},
                            0.5);
    check(close(capsule.signed_distance({1.0, 0.0, 0.5}), 0.5),
          "capsule side distance");
    check(close(capsule.signed_distance({0.0, 0.0, -1.0}), 0.5),
          "capsule lower endpoint distance");
    check(close(capsule.signed_distance({0.0, 0.0, 2.0}), 0.5),
          "capsule upper endpoint distance");

    HollowTubeObstacle tube(Eigen::Vector3d::Zero(),
                            Eigen::Vector3d::UnitX(),
                            1.0,
                            0.5,
                            0.8);
    check(close(tube.signed_distance({0.0, 0.0, 0.0}), 0.5),
          "hollow tube centerline is feasible channel");
    check(close(tube.signed_distance({0.0, 0.6, 0.0}), -0.1),
          "hollow tube wall interior is infeasible");
    check(close(tube.signed_distance({0.0, 1.0, 0.0}), 0.2),
          "hollow tube outside radius distance");
    check(close(tube.signed_distance({1.3, 0.6, 0.0}), 0.3),
          "hollow tube end distance through wall annulus");

    BoxObstacle box(Eigen::Vector3d::Zero(), {1.0, 1.0, 1.0});
    check(close(box.signed_distance({2.0, 0.0, 0.0}), 1.0),
          "box face distance");
    check(close(box.signed_distance(Eigen::Vector3d::Zero()), -1.0),
          "box interior distance");
    check(close(box.signed_distance({1.5, 1.5, 0.0}), std::sqrt(0.5)),
          "box edge-corner outside distance");

    HalfPlaneObstacle plane(Eigen::Vector3d::Zero(),
                            Eigen::Vector3d::UnitY());
    check(close(plane.signed_distance({5.0, 3.0, -2.0}), 3.0),
          "half-plane distance");
    check(close_vec(plane.signed_distance_gradient({5.0, 3.0, -2.0}),
                    Eigen::Vector3d::UnitY()),
          "half-plane gradient");
}

void test_sdf_gradient_fd() {
    std::cout << "-- obstacle SDF gradient FD checks --\n";

    struct Shape {
        std::string name;
        std::unique_ptr<Obstacle> obs;
    };

    std::vector<Shape> shapes;
    shapes.push_back({"sphere",
                      std::make_unique<SphereObstacle>(
                          Eigen::Vector3d(0.2, -0.1, 0.3), 0.8)});
    shapes.push_back({"capsule",
                      std::make_unique<CapsuleObstacle>(
                          Eigen::Vector3d(-0.2, 0.1, -0.5),
                          Eigen::Vector3d(0.3, -0.4, 0.8),
                          0.35)});
    shapes.push_back({"hollow tube",
                      std::make_unique<HollowTubeObstacle>(
                          Eigen::Vector3d(0.1, -0.2, 0.2),
                          Eigen::Vector3d(1.0, 0.2, -0.1),
                          0.9,
                          0.45,
                          0.75)});
    shapes.push_back({"box",
                      std::make_unique<BoxObstacle>(
                          Eigen::Vector3d(0.1, -0.2, 0.2),
                          Eigen::Vector3d(0.7, 0.8, 0.6))});
    shapes.push_back({"half-plane",
                      std::make_unique<HalfPlaneObstacle>(
                          Eigen::Vector3d(0.0, -0.3, 0.0),
                          Eigen::Vector3d(0.2, 1.0, 0.4))});

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-2.5, 2.5);
    const double h = 1e-5;
    for (const Shape &shape : shapes) {
        double worst_rel = 0.0;
        int checked = 0;
        int attempts = 0;
        while (checked < 5 && attempts < 1000) {
            ++attempts;
            Eigen::Vector3d x(dist(rng), dist(rng), dist(rng));
            if (shape.obs->signed_distance(x) <= 0.25) continue;

            Eigen::Vector3d numerical = Eigen::Vector3d::Zero();
            for (int k = 0; k < 3; ++k) {
                Eigen::Vector3d xp = x;
                Eigen::Vector3d xm = x;
                xp(k) += h;
                xm(k) -= h;
                numerical(k) =
                    (shape.obs->signed_distance(xp) -
                     shape.obs->signed_distance(xm)) / (2.0 * h);
            }
            const Eigen::Vector3d analytical =
                shape.obs->signed_distance_gradient(x);
            const double denom =
                std::max({1.0, analytical.norm(), numerical.norm()});
            worst_rel =
                std::max(worst_rel,
                         (analytical - numerical).norm() / denom);
            ++checked;
        }
        std::cout << "    " << shape.name
                  << " worst rel err = " << worst_rel << "\n";
        check(checked == 5 && worst_rel < 1e-6,
              shape.name + " SDF gradient matches FD");
    }
}

void test_obstacle_energy_gradient_fd() {
    std::cout << "-- obstacle barrier energy gradient FD check --\n";
    MeshData mesh = make_icosphere(2);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(2.0, 0.1, -0.05);
    mesh.L0 = mesh.compute_L0();
    SphereObstacle obs(Eigen::Vector3d::Zero(), 1.0);

    const Eigen::VectorXd x = flatten_matrix(mesh.V);
    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &v) {
            return obstacle_energy(with_vertices(mesh, v), obs);
        },
        [&](const Eigen::VectorXd &v) {
            return flatten_matrix(obstacle_gradient(with_vertices(mesh, v),
                                                    obs));
        },
        x,
        1e-6);

    std::cout << "    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.max_rel_err < 1e-5,
          "obstacle energy gradient matches central FD");
}

void test_surface_tpe_barrier_gradient_fd() {
    std::cout << "-- surface TPE barrier gradient FD check --\n";
    MeshData mesh = make_icosphere(0);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(1.3, 0.08, -0.03);
    mesh.L0 = mesh.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.55;
    barrier.L0 = barrier.compute_L0();

    const Eigen::VectorXd x = flatten_matrix(mesh.V);
    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &v) {
            return surface_tpe_barrier_energy(
                with_vertices(mesh, v), barrier, 6.0);
        },
        [&](const Eigen::VectorXd &v) {
            return flatten_matrix(surface_tpe_barrier_gradient(
                with_vertices(mesh, v), barrier, 6.0));
        },
        x,
        1e-6);

    std::cout << "    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.pass(1e-4),
          "surface TPE barrier gradient matches central FD");
}

void test_surface_tpe_barrier_bh_theta_zero_matches_brute() {
    std::cout << "-- surface TPE barrier BH theta=0 parity --\n";
    MeshData mesh = make_icosphere(0);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(1.2, 0.07, -0.04);
    mesh.L0 = mesh.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.55;
    barrier.L0 = barrier.compute_L0();

    const double e_brute = surface_tpe_barrier_energy(mesh, barrier, 6.0);
    const double e_bh = surface_tpe_barrier_energy_bh(
        mesh, barrier, 6.0, 0.0);
    const Eigen::MatrixXd g_brute =
        surface_tpe_barrier_gradient(mesh, barrier, 6.0);
    const Eigen::MatrixXd g_bh =
        surface_tpe_barrier_gradient_bh(mesh, barrier, 6.0, 0.0);
    const double e_rel = std::abs(e_brute - e_bh) /
                         std::max({1.0, std::abs(e_brute), std::abs(e_bh)});
    const double g_abs = (g_brute - g_bh).cwiseAbs().maxCoeff();
    const double g_rel =
        g_abs / std::max({1.0, g_brute.cwiseAbs().maxCoeff(),
                          g_bh.cwiseAbs().maxCoeff()});

    std::cout << "    energy rel err = " << e_rel
              << ", grad rel err = " << g_rel
              << ", grad max abs err = " << g_abs << "\n";
    check(e_rel < 1e-12, "BH theta=0 barrier energy matches brute");
    check(g_rel < 1e-12, "BH theta=0 barrier gradient matches brute");
}

void test_surface_tpe_barrier_bh_gradient_fd() {
    std::cout << "-- surface TPE barrier BH frozen-partition FD check --\n";
    MeshData mesh = make_icosphere(0);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(1.45, 0.06, 0.03);
    mesh.L0 = mesh.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.50;
    barrier.L0 = barrier.compute_L0();

    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, 0.5);

    const Eigen::VectorXd x = flatten_matrix(mesh.V);
    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &v) {
            return surface_tpe_barrier_energy_bh(
                with_vertices(mesh, v), barrier, cache, 6.0);
        },
        [&](const Eigen::VectorXd &v) {
            return flatten_matrix(surface_tpe_barrier_gradient_bh(
                with_vertices(mesh, v), barrier, cache, 6.0));
        },
        x,
        1e-6);

    std::cout << "    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.pass(1e-4),
          "BH surface barrier gradient matches frozen-partition FD");
}

void test_surface_tpe_barrier_adaptive_depth0_matches_midpoint() {
    std::cout << "-- surface TPE barrier adaptive depth0 parity --\n";
    MeshData mesh = make_icosphere(0);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(1.35, 0.05, -0.02);
    mesh.L0 = mesh.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.55;
    barrier.L0 = barrier.compute_L0();

    TpeAdaptiveParams adaptive;
    adaptive.enabled = true;
    adaptive.theta = 10.0;
    adaptive.max_depth = 0;

    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, 0.0, adaptive);
    const double e_mid =
        surface_tpe_barrier_energy_bh(mesh, barrier, 6.0, 0.0);
    const double e_ad =
        surface_tpe_barrier_energy_bh(mesh, barrier, cache, 6.0);
    const Eigen::MatrixXd g_mid =
        surface_tpe_barrier_gradient_bh(mesh, barrier, 6.0, 0.0);
    const Eigen::MatrixXd g_ad =
        surface_tpe_barrier_gradient_bh(mesh, barrier, cache, 6.0);

    const double e_err = std::abs(e_ad - e_mid);
    const double g_err = (g_ad - g_mid).cwiseAbs().maxCoeff();
    std::cout << "    adaptive terms = " << cache.near_terms.size()
              << ", |E_ad - E_mid| = " << e_err
              << ", grad max abs err = " << g_err << "\n";
    check(e_err < 1e-10,
          "surface barrier adaptive depth0 energy equals midpoint");
    check(g_err < 1e-10,
          "surface barrier adaptive depth0 gradient equals midpoint");
}

void test_surface_tpe_barrier_adaptive_gradient_fd() {
    std::cout << "-- surface TPE barrier adaptive FD check --\n";
    MeshData mesh = make_icosphere(0);
    mesh.normalize();
    mesh.V.rowwise() += Eigen::RowVector3d(1.05, 0.04, 0.02);
    mesh.L0 = mesh.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.60;
    barrier.L0 = barrier.compute_L0();

    TpeAdaptiveParams adaptive;
    adaptive.enabled = true;
    adaptive.theta = 10.0;
    adaptive.max_depth = 2;
    adaptive.max_stack_items = 400000;
    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, 0.0, adaptive);

    const Eigen::VectorXd x = flatten_matrix(mesh.V);
    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &v) {
            return surface_tpe_barrier_energy_bh(
                with_vertices(mesh, v), barrier, cache, 6.0);
        },
        [&](const Eigen::VectorXd &v) {
            return flatten_matrix(surface_tpe_barrier_gradient_bh(
                with_vertices(mesh, v), barrier, cache, 6.0));
        },
        x,
        1e-6);

    std::cout << "    adaptive terms = " << cache.near_terms.size()
              << "\n    max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index << "\n";
    check(r.pass(5e-3),
          "adaptive surface barrier gradient matches frozen-cache FD");
}

void test_surface_tpe_barrier_adaptive_near_contact_growth() {
    std::cout << "-- surface TPE barrier adaptive near-contact growth --\n";
    const double alpha = 6.0;
    const std::vector<double> deltas = {0.2, 0.1, 0.05, 0.025};
    std::vector<double> e_mid;
    std::vector<double> e_ad;
    e_mid.reserve(deltas.size());
    e_ad.reserve(deltas.size());

    MeshData barrier;
    barrier.V.resize(3, 3);
    barrier.V << 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0;
    barrier.F.resize(1, 3);
    barrier.F << 0, 1, 2;
    barrier.L0 = barrier.compute_L0();

    for (double d : deltas) {
        MeshData mesh;
        mesh.V.resize(3, 3);
        mesh.V << 0.05, 0.05, d,
                  1.80, 0.05, 1.0,
                  0.05, 1.80, 1.0;
        mesh.F.resize(1, 3);
        mesh.F << 0, 1, 2;
        mesh.L0 = mesh.compute_L0();

        TpeAdaptiveParams adaptive;
        adaptive.enabled = true;
        adaptive.theta = 0.5;
        adaptive.max_depth = 7;
        adaptive.max_stack_items = 500000;
        const SurfaceBarrierCache cache =
            build_surface_tpe_barrier_cache(mesh, barrier, 0.0, adaptive);

        e_mid.push_back(surface_tpe_barrier_energy_bh(
            mesh, barrier, alpha, 0.0));
        e_ad.push_back(surface_tpe_barrier_energy_bh(
            mesh, barrier, cache, alpha));
    }

    bool midpoint_monotone = true;
    bool adaptive_monotone = true;
    bool adaptive_ratio_monotone = true;
    for (size_t i = 1; i < deltas.size(); ++i) {
        if (!(e_mid[i] >= e_mid[i - 1])) midpoint_monotone = false;
        if (!(e_ad[i] >= e_ad[i - 1])) adaptive_monotone = false;
        if (!(e_ad[i] / e_mid[i] >= e_ad[i - 1] / e_mid[i - 1])) {
            adaptive_ratio_monotone = false;
        }
    }
    const double growth_mid = e_mid.back() / e_mid.front();
    const double growth_ad = e_ad.back() / e_ad.front();
    std::cout << "    midpoint E: ";
    for (double e : e_mid) std::cout << e << " ";
    std::cout << "\n    adaptive E: ";
    for (double e : e_ad) std::cout << e << " ";
    std::cout << "\n    growth midpoint: " << growth_mid
              << "\n    growth adaptive: " << growth_ad << "\n";
    check(midpoint_monotone,
          "surface barrier midpoint energy increases as gap shrinks");
    check(adaptive_monotone,
          "surface barrier adaptive energy increases as gap shrinks");
    check(adaptive_ratio_monotone,
          "surface barrier adaptive/midpoint ratio grows near contact");
    check(growth_ad > growth_mid,
          "surface barrier adaptive response grows faster than midpoint");
}

void test_path_energy_uses_adaptive_surface_tpe_barrier() {
    std::cout << "-- path-energy wires adaptive surface TPE barrier --\n";
    const double alpha = 6.0;

    MeshData barrier;
    barrier.V.resize(3, 3);
    barrier.V << 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0;
    barrier.F.resize(1, 3);
    barrier.F << 0, 1, 2;
    barrier.L0 = barrier.compute_L0();

    MeshData mesh;
    mesh.V.resize(3, 3);
    mesh.V << 0.05, 0.05, 0.035,
              1.80, 0.05, 1.0,
              0.05, 1.80, 1.0;
    mesh.F.resize(1, 3);
    mesh.F << 0, 1, 2;
    mesh.L0 = mesh.compute_L0();

    PathEnergyParams midpoint_params;
    midpoint_params.self_tpe_weight = 0.0;
    midpoint_params.tpe_barrier_mesh = &barrier;
    midpoint_params.tpe_barrier_weight = 1.0;
    midpoint_params.tpe_alpha = alpha;
    midpoint_params.tpe_theta = 0.0;
    midpoint_params.tpe_adaptive.enabled = false;

    PathEnergyParams adaptive_params = midpoint_params;
    adaptive_params.tpe_adaptive.enabled = true;
    adaptive_params.tpe_adaptive.theta = 0.5;
    adaptive_params.tpe_adaptive.max_depth = 7;
    adaptive_params.tpe_adaptive.max_stack_items = 500000;

    const double path_mid =
        path_energy({mesh}, midpoint_params).phi_per_frame[0];
    const double path_ad_uncached =
        path_energy({mesh}, adaptive_params).phi_per_frame[0];
    const std::vector<PathEnergyFrameCache> cache =
        build_path_energy_frame_cache({mesh}, adaptive_params);
    const double path_ad_cached =
        path_energy({mesh}, adaptive_params, &cache).phi_per_frame[0];
    const SurfaceBarrierCache direct_cache =
        build_surface_tpe_barrier_cache(
            mesh, barrier, 0.0, adaptive_params.tpe_adaptive);
    const double direct_ad =
        surface_tpe_barrier_energy_bh(mesh, barrier, direct_cache, alpha);

    std::cout << "    path midpoint = " << path_mid
              << ", path adaptive uncached = " << path_ad_uncached
              << ", path adaptive cached = " << path_ad_cached
              << ", direct adaptive = " << direct_ad
              << ", adaptive terms = " << cache[0].barrier_cache.near_terms.size()
              << "\n";
    check(cache[0].has_barrier_cache &&
              cache[0].barrier_cache.has_adaptive &&
              !cache[0].barrier_cache.near_terms.empty(),
          "path-energy barrier cache contains adaptive cross terms");
    check(path_ad_uncached > path_mid * 1.1,
          "path-energy uncached surface barrier uses adaptive quadrature");
    check(std::abs(path_ad_cached - direct_ad) <=
              1e-10 * std::max(1.0, std::abs(direct_ad)),
          "path-energy cached surface barrier matches direct adaptive barrier");
}

void test_barrier_divergence() {
    std::cout << "-- obstacle barrier divergence near contact --\n";

    MeshData base = make_icosphere(2);
    base.normalize();
    SphereObstacle obs(Eigen::Vector3d::Zero(), 1.0);

    const double start_gap = 1.0;
    const double end_gap = 5.0e-4;
    double prev_energy = -1.0;
    bool monotone = true;
    double final_energy = 0.0;
    double final_min_phi = 0.0;
    for (int step = 0; step < 10; ++step) {
        const double t = static_cast<double>(step) / 9.0;
        const double gap = (1.0 - t) * start_gap + t * end_gap;

        double lo = 0.0;
        double hi = 4.0;
        auto min_phi_at = [&](double center_x) {
            MeshData shifted = base;
            shifted.V.rowwise() += Eigen::RowVector3d(center_x, 0.0, 0.0);
            return min_signed_distance(shifted, obs);
        };
        while (min_phi_at(hi) < gap) hi *= 2.0;
        for (int it = 0; it < 80; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (min_phi_at(mid) < gap) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        MeshData mesh = base;
        const double center_x = hi;
        mesh.V.rowwise() += Eigen::RowVector3d(center_x, 0.0, 0.0);
        const double energy = obstacle_energy(mesh, obs);
        const double min_phi = min_signed_distance(mesh, obs);
        if (step > 0 && energy <= prev_energy) monotone = false;
        prev_energy = energy;
        final_energy = energy;
        final_min_phi = min_phi;
    }

    std::cout << "    final min phi = " << final_min_phi
              << ", final energy = " << final_energy << "\n";
    check(monotone, "barrier energy increases as closest gap shrinks");
    check(std::isfinite(final_energy) && final_energy > 1e6,
          "barrier energy is finite but large before contact");
}

void test_path_energy_with_obstacle() {
    std::cout << "-- path-energy gradient FD check with obstacle graph potential --\n";
    MeshData x0 = make_icosphere(1);
    x0.normalize();
    x0.V.rowwise() += Eigen::RowVector3d(2.5, 0.0, 0.0);
    x0.L0 = x0.compute_L0();
    std::vector<MeshData> frames(3, x0);
    frames[1].V.col(0) *= 1.02;
    frames[2].V.col(1) *= 0.97;
    for (auto &f : frames) f.L0 = f.compute_L0();

    SphereObstacle obs(Eigen::Vector3d::Zero(), 1.0);
    PathEnergyParams params;
    params.obstacle = &obs;
    params.tpe_alpha = 6.0;
    params.tpe_theta = 0.5;
    const std::vector<PathEnergyFrameCache> frozen_cache =
        build_path_energy_frame_cache(frames, params);

    const std::vector<MeshData> identical = {x0, x0};
    const PathEnergyResult same =
        path_energy(identical, params);
    check(std::isfinite(same.terms.obstacle_sum) &&
              same.terms.obstacle_sum > 0.0 &&
              std::abs(same.terms.total) < 1e-10,
          "constant obstacle barrier does not add ordinary path energy");

    PathEnergyParams without_obs = params;
    without_obs.obstacle = nullptr;
    const double e_with =
        path_energy(frames, params, &frozen_cache).terms.total;
    const double e_without =
        path_energy(frames, without_obs, &frozen_cache).terms.total;
    check(std::isfinite(e_with) && e_with > e_without,
          "path energy with obstacle graph potential > path energy without obstacle");

    const int nv = x0.n_vertices();
    auto pack = [&](const std::vector<MeshData> &f) {
        return flatten_matrix(f[1].V);
    };
    auto unpack = [&](const Eigen::VectorXd &z, std::vector<MeshData> &f) {
        f = frames;
        f[1].V = unflatten_vertices(z);
        f[1].L0 = f[1].compute_L0();
    };

    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &z) {
            std::vector<MeshData> f;
            unpack(z, f);
            return path_energy(f, params, &frozen_cache).terms.total;
        },
        [&](const Eigen::VectorXd &z) {
            std::vector<MeshData> f;
            unpack(z, f);
            const PathEnergyGradientResult g =
                path_energy_with_gradient(f, params, &frozen_cache);
            return flatten_matrix(g.grad_frames[1]);
        },
        pack(frames),
        1e-5);

    std::cout << "    frame-1 grad: max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index;
    if (r.worst_index >= 0) {
        std::cout << ", analytical = " << r.analytical[r.worst_index]
                  << ", numerical = " << r.numerical[r.worst_index];
    }
    std::cout << "\n";
    check(r.pass(1e-4),
          "path-energy gradient with obstacle graph potential matches central FD");
    (void)nv;
}

void test_path_energy_with_surface_tpe_barrier() {
    std::cout << "-- path-energy with surface TPE barrier --\n";
    MeshData x0 = make_icosphere(0);
    x0.normalize();
    x0.V.rowwise() += Eigen::RowVector3d(1.35, 0.04, 0.02);
    x0.L0 = x0.compute_L0();

    MeshData barrier = make_icosphere(0);
    barrier.normalize();
    barrier.V *= 0.55;
    barrier.L0 = barrier.compute_L0();

    std::vector<MeshData> frames(3, x0);
    frames[1].V.col(1).array() += 0.04;
    frames[2].V.col(0).array() += 0.08;
    for (MeshData &f : frames) f.L0 = f.compute_L0();

    PathEnergyParams params;
    params.tpe_barrier_mesh = &barrier;
    params.tpe_barrier_weight = 1.0;
    params.tpe_alpha = 6.0;
    params.tpe_theta = 0.0;
    params.shell.thickness = 0.0;

    const std::vector<PathEnergyFrameCache> frozen_cache =
        build_path_energy_frame_cache(frames, params);

    const PathEnergyResult same = path_energy({x0, x0}, params);
    check(std::isfinite(same.terms.obstacle_sum) &&
              same.terms.obstacle_sum > 0.0 &&
              std::abs(same.terms.total) < 1e-10,
          "constant surface TPE barrier does not add ordinary path energy");

    const PathEnergyResult beta_one = path_energy(frames, params);
    PathEnergyParams beta_scaled = params;
    beta_scaled.graph_beta = 0.25;
    const PathEnergyResult beta_quarter = path_energy(frames, beta_scaled);
    check(std::abs(beta_quarter.terms.repulsive_sum -
                   0.25 * beta_one.terms.repulsive_sum) <=
              1e-10 * std::max(1.0, beta_one.terms.repulsive_sum),
          "graph_beta scales the squared graph-potential difference");

    GradCheckResult r = finite_diff_gradient_check(
        [&](const Eigen::VectorXd &z) {
            std::vector<MeshData> f = frames;
            f[1].V = unflatten_vertices(z);
            f[1].L0 = f[1].compute_L0();
            return path_energy(f, params, &frozen_cache).terms.total;
        },
        [&](const Eigen::VectorXd &z) {
            std::vector<MeshData> f = frames;
            f[1].V = unflatten_vertices(z);
            f[1].L0 = f[1].compute_L0();
            return flatten_matrix(path_energy_with_gradient(f, params,
                                                            &frozen_cache)
                                      .grad_frames[1]);
        },
        flatten_matrix(frames[1].V),
        1e-6);
    std::cout << "    frame-1 grad: max abs err = " << r.max_abs_err
              << ", max rel err = " << r.max_rel_err
              << ", worst index = " << r.worst_index;
    if (r.worst_index >= 0) {
        std::cout << ", analytical = " << r.analytical[r.worst_index]
                  << ", numerical = " << r.numerical[r.worst_index];
    }
    std::cout << "\n";
    check(r.pass(1e-4),
          "path-energy surface TPE barrier gradient matches central FD");
}

void test_infeasibility_signal() {
    std::cout << "-- obstacle infeasibility signal --\n";
    MeshData mesh = make_icosphere(2);
    mesh.normalize();
    SphereObstacle obs(Eigen::Vector3d::Zero(), 1.0);
    const double energy = obstacle_energy(mesh, obs);
    check(std::isinf(energy),
          "obstacle energy is infinity when vertices are inside");
    const Eigen::MatrixXd grad = obstacle_gradient(mesh, obs);
    check(grad.isZero(0.0),
          "obstacle gradient returns zero matrix on infeasible mesh");
}

} // namespace

int main() {
    std::cout << "=== Phase 3 obstacle barrier tests ===\n";
    test_sdf_hand_cases();
    test_sdf_gradient_fd();
    test_obstacle_energy_gradient_fd();
    test_surface_tpe_barrier_gradient_fd();
    test_surface_tpe_barrier_bh_theta_zero_matches_brute();
    test_surface_tpe_barrier_bh_gradient_fd();
    test_surface_tpe_barrier_adaptive_depth0_matches_midpoint();
    test_surface_tpe_barrier_adaptive_gradient_fd();
    test_surface_tpe_barrier_adaptive_near_contact_growth();
    test_path_energy_uses_adaptive_surface_tpe_barrier();
    test_barrier_divergence();
    test_path_energy_with_obstacle();
    test_path_energy_with_surface_tpe_barrier();
    test_infeasibility_signal();

    if (failures == 0) {
        std::cout << "\nALL PASSED\n";
        return 0;
    }
    std::cout << "\n" << failures << " checks failed\n";
    return 1;
}
