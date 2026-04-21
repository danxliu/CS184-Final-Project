#ifndef FACEGEOM_H
#define FACEGEOM_H

#include "MeshData.h"
#include <Eigen/Dense>

namespace rsh {

// Per-face geometric quantities, batched across all faces of a mesh.
//
//   C.row(t)   — centroid c_t = (v0 + v1 + v2) / 3
//   N.row(t)   — unit face normal n_t = normalize( (v1-v0) × (v2-v0) )
//   A(t)       — face area     a_t = 0.5 * |(v1-v0) × (v2-v0)|
struct FaceGeom {
    Eigen::MatrixXd C;
    Eigen::MatrixXd N;
    Eigen::VectorXd A;
};

FaceGeom compute_face_geom(const MeshData &mesh);

// Opposite-edge convention used throughout the Jacobian formulas:
//   E_0 = v2 - v1   (edge opposite corner 0)
//   E_1 = v0 - v2   (edge opposite corner 1)
//   E_2 = v1 - v0   (edge opposite corner 2)
// With this convention, E_0 + E_1 + E_2 = 0 and moving corner k parallel
// to E_k leaves the triangle's area unchanged.
void opposite_edges(const Eigen::Vector3d &v0,
                    const Eigen::Vector3d &v1,
                    const Eigen::Vector3d &v2,
                    Eigen::Vector3d &E0,
                    Eigen::Vector3d &E1,
                    Eigen::Vector3d &E2);

// Per-corner Jacobians for a single triangle. Each returns the partial
// derivative of the face quantity with respect to the position of corner k.
//
//   dc / dv_k = (1/3) I_3            (3×3, independent of k)
//   da / dv_k = 0.5 * (n × E_k)^T     (1×3 row vector)
//   dn / dv_k = (1/(2a)) (I - n n^T) [E_k]_x   (3×3)
//
// [E]_x is the skew-symmetric matrix satisfying [E]_x w = E × w.
Eigen::Matrix3d dc_dvk();

Eigen::RowVector3d da_dvk(const Eigen::Vector3d &n,
                          const Eigen::Vector3d &Ek);

Eigen::Matrix3d dn_dvk(const Eigen::Vector3d &n,
                       double area,
                       const Eigen::Vector3d &Ek);

// Skew-symmetric cross-product matrix: skew(a) * b == a.cross(b).
Eigen::Matrix3d skew(const Eigen::Vector3d &a);

} // namespace rsh

#endif
