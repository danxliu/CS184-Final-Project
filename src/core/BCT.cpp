#include "BCT.h"

#include <algorithm>

namespace rsh {

namespace {

// AABB squared-distance: 0 if boxes overlap, otherwise the squared distance
// between the closest pair of points. For nearby clusters whose AABBs overlap
// (e.g. opposite sides of a torus handle pinch), this is zero — forcing the
// pair into recursion / near-field exact computation. Repulsor uses the same
// formulation (GJK.hpp:1182).
double aabb_squared_distance(const BVHNode &U, const BVHNode &V) {
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double over = U.bmin[k] - V.bmax[k];   // > 0 iff U_min > V_max
        const double under = V.bmin[k] - U.bmax[k];  // > 0 iff V_min > U_max
        const double gap = std::max({0.0, over, under});
        d2 += gap * gap;
    }
    return d2;
}

bool mac_admissible(const BVHNode &U, const BVHNode &V, double theta) {
    const double d2 = aabb_squared_distance(U, V);
    const double max_r2 = std::max(U.radius * U.radius, V.radius * V.radius);
    return d2 > 0.0 && max_r2 < (theta * theta) * d2;
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

void visit_cross(const BVH &left,
                 const BVH &right,
                 int u,
                 int v,
                 double theta,
                 BlockPairs &out) {
    const BVHNode &U = left.nodes[u];
    const BVHNode &V = right.nodes[v];

    if (mac_admissible(U, V, theta)) {
        out.admissible.push_back({u, v});
        return;
    }

    if (U.is_leaf() && V.is_leaf()) {
        out.near_field.push_back({u, v});
        return;
    }

    if (U.is_leaf()) {
        visit_cross(left, right, u, V.left, theta, out);
        visit_cross(left, right, u, V.right, theta, out);
    } else if (V.is_leaf()) {
        visit_cross(left, right, U.left, v, theta, out);
        visit_cross(left, right, U.right, v, theta, out);
    } else {
        visit_cross(left, right, U.left, V.left, theta, out);
        visit_cross(left, right, U.left, V.right, theta, out);
        visit_cross(left, right, U.right, V.left, theta, out);
        visit_cross(left, right, U.right, V.right, theta, out);
    }
}

} // anonymous namespace

BlockPairs build_bct_self(const BVH &bvh, double theta) {
    BlockPairs out;
    visit(bvh, bvh.root, bvh.root, theta, out);
    return out;
}

BlockPairs build_bct_cross(const BVH &left, const BVH &right, double theta) {
    BlockPairs out;
    if (left.nodes.empty() || right.nodes.empty()) {
        return out;
    }
    visit_cross(left, right, left.root, right.root, theta, out);
    return out;
}

} // namespace rsh
