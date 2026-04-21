#include "BCT.h"

#include <algorithm>

namespace rsh {

namespace {

bool mac_admissible(const BVHNode &U, const BVHNode &V, double theta) {
    const double dist = (V.centroid - U.centroid).norm();
    const double max_r = std::max(U.radius, V.radius);
    return dist > 0.0 && max_r < theta * dist;
}

void visit(const BVH &bvh, int u, int v, double theta, BlockPairs &out) {
    const BVHNode &U = bvh.nodes[u];
    const BVHNode &V = bvh.nodes[v];

    // Self-pair. Can't admit (dist = 0); must recurse into children, or emit
    // as near-field at a leaf (the near-field evaluator skips the t1 == t2
    // diagonal).
    if (u == v) {
        if (U.is_leaf()) {
            out.near_field.push_back({u, v});
            return;
        }
        visit(bvh, U.left, U.left, theta, out);
        visit(bvh, U.left, U.right, theta, out);
        visit(bvh, U.right, U.left, theta, out);
        visit(bvh, U.right, U.right, theta, out);
        return;
    }

    if (mac_admissible(U, V, theta)) {
        out.admissible.push_back({u, v});
        return;
    }

    if (U.is_leaf() && V.is_leaf()) {
        out.near_field.push_back({u, v});
        return;
    }

    if (U.is_leaf()) {
        visit(bvh, u, V.left, theta, out);
        visit(bvh, u, V.right, theta, out);
    } else if (V.is_leaf()) {
        visit(bvh, U.left, v, theta, out);
        visit(bvh, U.right, v, theta, out);
    } else {
        visit(bvh, U.left, V.left, theta, out);
        visit(bvh, U.left, V.right, theta, out);
        visit(bvh, U.right, V.left, theta, out);
        visit(bvh, U.right, V.right, theta, out);
    }
}

} // anonymous namespace

BlockPairs build_bct_self(const BVH &bvh, double theta) {
    BlockPairs out;
    visit(bvh, bvh.root, bvh.root, theta, out);
    return out;
}

} // namespace rsh
