#ifndef TPE_H
#define TPE_H

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include <vector>

namespace rsh {

struct TpeAdaptiveParams {
    bool enabled = false;
    double theta = 10.0;
    int max_depth = 8;
    int max_stack_items = 262144;
};

struct TpeNearFieldTerm {
    int t1 = -1;
    int t2 = -1;
    Eigen::Vector3d w1 = Eigen::Vector3d::Constant(1.0 / 3.0);
    Eigen::Vector3d w2 = Eigen::Vector3d::Constant(1.0 / 3.0);
    double area_scale_1 = 1.0;
    double area_scale_2 = 1.0;
};

struct TpeAdaptiveCache {
    TpeAdaptiveParams params;
    std::vector<TpeNearFieldTerm> near_terms;
};

// Discrete tangent-point energy via midpoint quadrature (RSu Eq. 10, RS Eq. 14):
//
//   Phi(x) = sum_{t1 != t2} a_{t1} * a_{t2} *
//            |<n_{t1}, c_{t1} - c_{t2}>|^alpha / |c_{t1} - c_{t2}|^{2 alpha}
//
// The sum runs over ordered pairs (so each unordered pair is visited twice with
// different "side" normals). Skips the t1 == t2 diagonal. Brute O(n_f^2) — this
// is the reference that the hierarchical (Barnes-Hut) version in Phase 1.6 has
// to match on small meshes.
double tpe_energy_brute(const MeshData &mesh, double alpha = 6.0);
double tpe_energy_brute(const FaceGeom &g, double alpha = 6.0);

// Analytical gradient of tpe_energy_brute w.r.t. vertex positions.
// Returns an (n_v x 3) matrix where row i holds dPhi / dv_i.
//
// Chain-rules the per-pair kernel derivatives through FaceGeom's per-corner
// Jacobians (dc/dv_k, dn/dv_k, da/dv_k). The kernel depends on n_{t1} only
// (not n_{t2}), so the "t2 side" of each ordered pair contributes through c
// and a but not through n — that asymmetry is restored by also visiting the
// swapped pair (t2, t1).
Eigen::MatrixXd tpe_gradient_brute(const MeshData &mesh, double alpha = 6.0);
Eigen::MatrixXd tpe_gradient_brute(const MeshData &mesh,
                                   const FaceGeom &g,
                                   double alpha = 6.0);

// Barnes-Hut tangent-point energy via a prebuilt BVH + block-cluster pair list.
//
// Admissible cluster pair (U, V) contributes the 0th-order multipole proxy
//   K(U, V) = a_U a_V * |<n_U/a_U, c_U - c_V>|^alpha / ||c_U - c_V||^(2 alpha),
// where n_U / a_U is the area-weighted mean unit normal (RSu §4.1 / RS §5.2).
// Near-field leaf-leaf pairs evaluate the exact brute-force kernel on every
// face pair in the leaves, skipping the t1 == t2 diagonal on self-pairs.
//
// With bp = build_bct_self(bvh, theta = 0), this collapses to the brute-force
// energy (modulo floating-point summation-order differences).
double tpe_energy_bh(const FaceGeom &g, const BVH &bvh, const BlockPairs &bp,
                     double alpha = 6.0);

TpeAdaptiveCache build_tpe_adaptive_cache(const MeshData &mesh,
                                          const FaceGeom &g,
                                          const BVH &bvh,
                                          const BlockPairs &bp,
                                          const TpeAdaptiveParams &adaptive);

double tpe_energy_bh(const MeshData &mesh,
                     const FaceGeom &g,
                     const BVH &bvh,
                     const BlockPairs &bp,
                     const TpeAdaptiveParams &adaptive,
                     double alpha = 6.0,
                     const TpeAdaptiveCache *cache = nullptr);

// Convenience wrapper: build FaceGeom + BVH + BCT internally. Intended for
// one-shot use; for repeated evaluations on the same topology, build the
// BVH/BCT once and call the three-argument overload above.
double tpe_energy_bh(const MeshData &mesh, double alpha = 6.0,
                     double theta = 0.5);

double tpe_energy_bh(const MeshData &mesh,
                     const TpeAdaptiveParams &adaptive,
                     double alpha = 6.0,
                     double theta = 0.5);

// Barnes-Hut tangent-point gradient. Returns an (n_v x 3) matrix where row i
// holds dPhi_BH / dv_i for the same hierarchical approximation evaluated by
// tpe_energy_bh above.
//
// Subtlety (RSu §4.2 "Approximate Derivative"): the set of admissible blocks
// depends on x, so naively differentiating the BH approximation picks up a
// jump contribution every time a pair switches between admissible and near-
// field. We instead freeze the BVH / BlockPairs at the current iterate and
// differentiate only the per-pair kernel values on that fixed partition — the
// same scheme the paper ships.
//
// Admissible pair (U, V): the cluster aggregates (a_U, c_U, n_U) are linear
// in the per-face quantities (a_t, c_t, n_t), so the chain rule to vertex
// positions is just a scatter followed by the usual FaceGeom Jacobian apply.
// Only the "near" side (U) depends on the normal — the V side contributes
// through area and centroid only (symmetry is restored by visiting (V, U) as
// a separate admissible pair).
//
// Near-field pair: exact per-face pair gradient, identical to the brute-force
// inner kernel.
//
// With theta = 0 this matches tpe_gradient_brute to floating-point roundoff.
Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const FaceGeom &g,
                                const BVH &bvh,
                                const BlockPairs &bp,
                                double alpha = 6.0);

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const FaceGeom &g,
                                const BVH &bvh,
                                const BlockPairs &bp,
                                const TpeAdaptiveParams &adaptive,
                                double alpha = 6.0,
                                const TpeAdaptiveCache *cache = nullptr);

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh, double alpha = 6.0,
                                double theta = 0.5);

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const TpeAdaptiveParams &adaptive,
                                double alpha = 6.0,
                                double theta = 0.5);

} // namespace rsh

#endif
