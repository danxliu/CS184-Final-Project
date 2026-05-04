#ifndef BVH_H
#define BVH_H

#include "FaceGeom.h"
#include "MeshData.h"
#include <Eigen/Dense>
#include <vector>

namespace rsh {

// Bounding volume hierarchy over the faces of a triangle mesh.
//
// Each node caches:
//   bmin, bmax      — axis-aligned bounding box over all faces in the subtree.
//   area            — a_U = sum of face areas in the subtree (RS Eq. 21).
//   centroid        — c_U = (1/a_U) sum_{t} a_t c_t, area-weighted. The
//                     multipole expansion point in RSu §4.1 and RS Eq. 19.
//   normal_sum      — n_U = sum_{t} a_t n_t, area-weighted but NOT unit. The
//                     magnitude |n_U| / a_U is in [0, 1] and encodes how
//                     coherently oriented the cluster is.
//                     Used by the H^s low-order operator (B_0). Avoid for
//                     TPE-side multipole — see projector_sum below.
//   projector_sum   — PS_U = sum_{t} a_t * (n_t * n_t^T), 3x3 symmetric PSD.
//                     The covariance of the cluster's face normals (weighted
//                     by area). Repulsor's BCT0 uses the same form. Replaces
//                     normal_sum for the TPE multipole kernel because the
//                     Jensen gap of |E[N]·d|^α is large at α=6 when normals
//                     in the cluster disagree (e.g. opposing patches across
//                     a torus handle pinch); ((1/a) d^T PS_U d)^(α/2) is a
//                     much tighter approximation.
//   radius          — r_U = max_{t in subtree} ||c_t - c_U||. Used by the
//                     MAC in Phase 1.5 together with AABB separation.
//                     Computed exactly at leaves; upper-bounded via the
//                     triangle inequality at internal nodes (the upper
//                     bound is safe for MAC — it only makes admissibility
//                     more conservative).
//
// Every node (leaf or internal) carries [face_start, face_end) — a contiguous
// range into BVH::face_indices identifying every face in its subtree. The
// face_indices array is partitioned in place during build, so an internal
// node's range equals the union of its children's ranges and callers can
// iterate a cluster's faces without a recursive subtree walk.
struct BVHNode {
    Eigen::Vector3d bmin;
    Eigen::Vector3d bmax;
    double area = 0.0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_sum = Eigen::Vector3d::Zero();
    Eigen::Matrix3d projector_sum = Eigen::Matrix3d::Zero();
    double radius = 0.0;

    int left = -1;   // child indices into BVH::nodes; -1 if leaf
    int right = -1;
    int face_start = 0;  // only valid when is_leaf()
    int face_end = 0;

    bool is_leaf() const { return left < 0; }
    int face_count() const { return face_end - face_start; }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int> face_indices;  // permutation of [0, n_f); leaves own slices
    int root = 0;

    int n_faces() const { return static_cast<int>(face_indices.size()); }
    int n_nodes() const { return static_cast<int>(nodes.size()); }
};

// Build a BVH with a top-down SAH split at each internal node.
// max_leaf_size: stop subdividing once a node owns this many or fewer faces.
BVH build_bvh(const MeshData &mesh, const FaceGeom &g, int max_leaf_size = 8);
BVH build_bvh(const MeshData &mesh, int max_leaf_size = 8);

// Refresh per-node aggregates (area, centroid, normal_sum, radius) from a new
// FaceGeom while keeping the tree structure (face_indices permutation and
// shape) fixed. Used when differentiating a BH approximation through the
// aggregate chain rule — the partition is frozen but the aggregates must
// respond to vertex perturbations. AABBs are NOT refreshed since they are
// consumed by BCT.cpp's AABB-distance MAC. Callers that freeze a tree for
// derivative checks must reuse the original BCT partition.
void update_bvh_aggregates(BVH &bvh, const FaceGeom &g);

} // namespace rsh

#endif
