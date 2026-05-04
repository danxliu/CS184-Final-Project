#ifndef OBSTACLE_H
#define OBSTACLE_H

#include "MeshData.h"

#include <Eigen/Dense>

namespace rsh {

class Obstacle {
public:
    virtual ~Obstacle() = default;

    // Signed distance convention: positive outside, negative inside.
    virtual double signed_distance(const Eigen::Vector3d &x) const = 0;
    virtual Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const = 0;
};

class SphereObstacle final : public Obstacle {
public:
    SphereObstacle(const Eigen::Vector3d &center, double radius);

    double signed_distance(const Eigen::Vector3d &x) const override;
    Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const override;

private:
    Eigen::Vector3d center_;
    double radius_ = 1.0;
};

class CapsuleObstacle final : public Obstacle {
public:
    CapsuleObstacle(const Eigen::Vector3d &p0,
                    const Eigen::Vector3d &p1,
                    double radius);

    double signed_distance(const Eigen::Vector3d &x) const override;
    Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const override;

private:
    Eigen::Vector3d closest_point(const Eigen::Vector3d &x) const;

    Eigen::Vector3d p0_;
    Eigen::Vector3d p1_;
    double radius_ = 1.0;
};

class HollowTubeObstacle final : public Obstacle {
public:
    HollowTubeObstacle(const Eigen::Vector3d &center,
                       const Eigen::Vector3d &axis,
                       double half_length,
                       double inner_radius,
                       double outer_radius);

    double signed_distance(const Eigen::Vector3d &x) const override;
    Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const override;

private:
    Eigen::Vector2d tube_coordinates(const Eigen::Vector3d &x) const;
    Eigen::Vector2d signed_distance_gradient_2d(
        const Eigen::Vector2d &coord) const;
    Eigen::Vector3d radial_direction(const Eigen::Vector3d &x) const;

    Eigen::Vector3d center_;
    Eigen::Vector3d axis_;
    double half_length_ = 1.0;
    double inner_radius_ = 0.5;
    double outer_radius_ = 0.75;
};

class BoxObstacle final : public Obstacle {
public:
    BoxObstacle(const Eigen::Vector3d &center,
                const Eigen::Vector3d &half_extents);

    double signed_distance(const Eigen::Vector3d &x) const override;
    Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const override;

private:
    Eigen::Vector3d center_;
    Eigen::Vector3d half_extents_;
};

class HalfPlaneObstacle final : public Obstacle {
public:
    HalfPlaneObstacle(const Eigen::Vector3d &point_on_plane,
                      const Eigen::Vector3d &outward_normal);

    double signed_distance(const Eigen::Vector3d &x) const override;
    Eigen::Vector3d
    signed_distance_gradient(const Eigen::Vector3d &x) const override;

private:
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
};

double obstacle_energy(const MeshData &mesh, const Obstacle &obs);
Eigen::MatrixXd obstacle_gradient(const MeshData &mesh,
                                  const Obstacle &obs);

} // namespace rsh

#endif
