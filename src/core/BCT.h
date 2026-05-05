#ifndef BCT_H
#define BCT_H

#include "BVH.h"
#include <vector>

namespace rsh {

// A pair of BVH nodes. For self traversals both indices refer to the same
// BVH; for cross traversals `u` indexes the left BVH and `v` indexes the
// right BVH.
struct ClusterPair {
    int u;
    int v;
};

// Output of a block-cluster-tree traversal. Every ordered face pair (t1, t2)
// with t1 != t2 is covered by exactly one of:
//   - An entry in `admissible`, where cluster-level 0th-order multipole is OK.
//   - An entry in `near_field`, where exact per-face evaluation is required.
// The caller iterates these lists to evaluate hierarchical energies/gradients.
struct BlockPairs {
    std::vector<ClusterPair> admissible;
    std::vector<ClusterPair> near_field;
};

// Enumerate block-cluster pairs of a BVH against itself.
//
// Acceptance criterion (RSu §4.1, radius form):
//   admissible  <=>  max(r_U, r_V) < theta * dist(c_U, c_V)
// Strict inequality guarantees that `theta = 0` admits no pair (so the
// near-field list reduces to the exact brute-force ordered-pair enumeration).
// RSu's default for global Barnes-Hut is theta = 0.5; RS uses the same value
// for the global tree and theta ≈ 10 for the near-field adaptive scheme in
// Phase 3 (not relevant here).
BlockPairs build_bct_self(const BVH &bvh, double theta = 0.5);

// Enumerate block-cluster pairs between two distinct BVHs. The output covers
// every Cartesian product face pair exactly once. The same MAC as
// build_bct_self is used, with `u` indexing `left.nodes` and `v` indexing
// `right.nodes`.
BlockPairs build_bct_cross(const BVH &left,
                           const BVH &right,
                           double theta = 0.5);

} // namespace rsh

#endif
