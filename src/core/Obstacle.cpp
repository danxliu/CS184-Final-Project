#include "Obstacle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rsh {

namespace {

constexpr double kPhiFloor = 1.0e-12;

Eigen::Vector3d fallback_unit_x() {
    return Eigen::Vector3d::UnitX();
}

double require_positive_radius(double radius, const char *name) {
    if (!(radius > 0.0) || !std::isfinite(radius)) {
        throw std::runtime_error(std::string(name) +
                                 ": radius must be positive and finite");
    }
    return radius;
}

} // namespace

SphereObstacle::SphereObstacle(const Eigen::Vector3d &center, double radius)
    : center_(center), radius_(require_positive_radius(radius,
                                                       "SphereObstacle")) {}

double SphereObstacle::signed_distance(const Eigen::Vector3d &x) const {
    return (x - center_).norm() - radius_;
}

Eigen::Vector3d
SphereObstacle::signed_distance_gradient(const Eigen::Vector3d &x) const {
    const Eigen::Vector3d d = x - center_;
    const double n = d.norm();
    if (n <= 0.0) return fallback_unit_x();
    return d / n;
}

CapsuleObstacle::CapsuleObstacle(const Eigen::Vector3d &p0,
                                 const Eigen::Vector3d &p1,
                                 double radius)
    : p0_(p0), p1_(p1),
      radius_(require_positive_radius(radius, "CapsuleObstacle")) {}

Eigen::Vector3d
CapsuleObstacle::closest_point(const Eigen::Vector3d &x) const {
    const Eigen::Vector3d axis = p1_ - p0_;
    const double axis2 = axis.squaredNorm();
    if (axis2 <= 0.0) return p0_;
    const double t = std::clamp((x - p0_).dot(axis) / axis2, 0.0, 1.0);
    return p0_ + t * axis;
}

double CapsuleObstacle::signed_distance(const Eigen::Vector3d &x) const {
    return (x - closest_point(x)).norm() - radius_;
}

Eigen::Vector3d
CapsuleObstacle::signed_distance_gradient(const Eigen::Vector3d &x) const {
    const Eigen::Vector3d d = x - closest_point(x);
    const double n = d.norm();
    if (n <= 0.0) return fallback_unit_x();
    return d / n;
}

BoxObstacle::BoxObstacle(const Eigen::Vector3d &center,
                         const Eigen::Vector3d &half_extents)
    : center_(center), half_extents_(half_extents) {
    if ((half_extents_.array() <= 0.0).any() || !half_extents_.allFinite()) {
        throw std::runtime_error(
            "BoxObstacle: half extents must be positive and finite");
    }
}

double BoxObstacle::signed_distance(const Eigen::Vector3d &x) const {
    const Eigen::Vector3d q =
        (x - center_).cwiseAbs() - half_extents_;
    const Eigen::Vector3d outside = q.cwiseMax(0.0);
    const double outside_dist = outside.norm();
    const double inside_dist = std::min(q.maxCoeff(), 0.0);
    return outside_dist + inside_dist;
}

Eigen::Vector3d
BoxObstacle::signed_distance_gradient(const Eigen::Vector3d &x) const {
    const Eigen::Vector3d p = x - center_;
    const Eigen::Vector3d q = p.cwiseAbs() - half_extents_;
    const Eigen::Vector3d outside = q.cwiseMax(0.0);

    if (outside.squaredNorm() > 0.0) {
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        for (int i = 0; i < 3; ++i) {
            if (q(i) > 0.0) {
                g(i) = outside(i) * (p(i) >= 0.0 ? 1.0 : -1.0);
            }
        }
        const double n = g.norm();
        if (n > 0.0) return g / n;
    }

    int axis = 0;
    if (q(1) > q(axis)) axis = 1;
    if (q(2) > q(axis)) axis = 2;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    g(axis) = (p(axis) >= 0.0 ? 1.0 : -1.0);
    return g;
}

HalfPlaneObstacle::HalfPlaneObstacle(
    const Eigen::Vector3d &point_on_plane,
    const Eigen::Vector3d &outward_normal)
    : point_(point_on_plane), normal_(outward_normal) {
    const double n = normal_.norm();
    if (!(n > 0.0) || !std::isfinite(n)) {
        throw std::runtime_error(
            "HalfPlaneObstacle: outward normal must be nonzero and finite");
    }
    normal_ /= n;
}

double HalfPlaneObstacle::signed_distance(const Eigen::Vector3d &x) const {
    return (x - point_).dot(normal_);
}

Eigen::Vector3d
HalfPlaneObstacle::signed_distance_gradient(const Eigen::Vector3d &) const {
    return normal_;
}

double obstacle_energy(const MeshData &mesh, const Obstacle &obs) {
    double energy = 0.0;
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        const double phi =
            obs.signed_distance(mesh.V.row(i).transpose());
        if (!(phi > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        const double denom = std::max(phi, kPhiFloor);
        energy += 1.0 / (denom * denom);
    }
    return energy;
}

Eigen::MatrixXd obstacle_gradient(const MeshData &mesh,
                                  const Obstacle &obs) {
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        const Eigen::Vector3d x = mesh.V.row(i).transpose();
        const double phi = obs.signed_distance(x);
        if (!(phi > 0.0)) {
            return Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
        }
        const double denom = std::max(phi, kPhiFloor);
        grad.row(i) =
            (-2.0 / (denom * denom * denom) *
             obs.signed_distance_gradient(x)).transpose();
    }
    return grad;
}

} // namespace rsh
