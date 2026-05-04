#include "GradCheck.h"
#include "MeshData.h"
#include "Obstacle.h"
#include "PathEnergy.h"
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
    std::cout << "-- path-energy gradient FD check with obstacle term --\n";
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

    PathEnergyParams without_obs = params;
    without_obs.obstacle = nullptr;
    const double e_with =
        path_energy(frames, params, &frozen_cache).terms.total;
    const double e_without =
        path_energy(frames, without_obs, &frozen_cache).terms.total;
    check(std::isfinite(e_with) && e_with > e_without,
          "path energy with obstacle > path energy without obstacle");

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
              << ", max rel err = " << r.max_rel_err << "\n";
    check(r.pass(1e-4),
          "path-energy gradient with obstacle matches central FD");
    (void)nv;
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
    test_barrier_divergence();
    test_path_energy_with_obstacle();
    test_infeasibility_signal();

    if (failures == 0) {
        std::cout << "\nALL PASSED\n";
        return 0;
    }
    std::cout << "\n" << failures << " checks failed\n";
    return 1;
}
