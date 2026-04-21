#include "BVH.h"

#include <algorithm>
#include <array>
#include <limits>

namespace rsh {

namespace {

constexpr int kNumBins = 16;

struct FaceBox {
    Eigen::Vector3d bmin;
    Eigen::Vector3d bmax;
    Eigen::Vector3d centroid;  // c_t from FaceGeom, used for split positioning
};

std::vector<FaceBox> compute_face_boxes(const MeshData &mesh,
                                        const FaceGeom &g) {
    const int nf = static_cast<int>(mesh.F.rows());
    std::vector<FaceBox> out(nf);
    for (int t = 0; t < nf; ++t) {
        const Eigen::Vector3d v0 = mesh.V.row(mesh.F(t, 0));
        const Eigen::Vector3d v1 = mesh.V.row(mesh.F(t, 1));
        const Eigen::Vector3d v2 = mesh.V.row(mesh.F(t, 2));
        out[t].bmin = v0.cwiseMin(v1).cwiseMin(v2);
        out[t].bmax = v0.cwiseMax(v1).cwiseMax(v2);
        out[t].centroid = g.C.row(t);
    }
    return out;
}

double aabb_surface_area(const Eigen::Vector3d &bmin,
                         const Eigen::Vector3d &bmax) {
    const Eigen::Vector3d d = (bmax - bmin).cwiseMax(0.0);
    return 2.0 * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

struct SplitChoice {
    int axis = -1;
    int bin = -1;
    double cost = std::numeric_limits<double>::infinity();
};

// SAH split with axis-aligned centroid binning. Returns the (axis, bin_cut)
// with minimum cost; `bin_cut` is the first bin index that goes to the right
// child. If no valid split found (e.g. all centroids coincident), returns
// axis = -1 and the caller falls back to a median split.
SplitChoice find_sah_split(const std::vector<FaceBox> &fa,
                           const std::vector<int> &face_indices,
                           int start, int end) {
    SplitChoice best;

    Eigen::Vector3d cmin = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    Eigen::Vector3d cmax = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (int i = start; i < end; ++i) {
        cmin = cmin.cwiseMin(fa[face_indices[i]].centroid);
        cmax = cmax.cwiseMax(fa[face_indices[i]].centroid);
    }

    for (int axis = 0; axis < 3; ++axis) {
        const double extent = cmax[axis] - cmin[axis];
        if (extent < 1e-20) continue;

        struct Bin {
            Eigen::Vector3d bmin;
            Eigen::Vector3d bmax;
            int count = 0;
            Bin() {
                bmin = Eigen::Vector3d::Constant(
                    std::numeric_limits<double>::infinity());
                bmax = Eigen::Vector3d::Constant(
                    -std::numeric_limits<double>::infinity());
            }
        };
        std::array<Bin, kNumBins> bins;

        const double scale = kNumBins / extent;
        for (int i = start; i < end; ++i) {
            const FaceBox &f = fa[face_indices[i]];
            int b = static_cast<int>(scale * (f.centroid[axis] - cmin[axis]));
            if (b < 0) b = 0;
            if (b >= kNumBins) b = kNumBins - 1;
            bins[b].bmin = bins[b].bmin.cwiseMin(f.bmin);
            bins[b].bmax = bins[b].bmax.cwiseMax(f.bmax);
            bins[b].count++;
        }

        // Prefix sweep left-to-right, suffix sweep right-to-left. For each
        // candidate cut (between bin k-1 and bin k), the SAH cost is
        //    cost(k) = SA(L)·count(L) + SA(R)·count(R).
        std::array<Eigen::Vector3d, kNumBins> lmin, lmax, rmin, rmax;
        std::array<int, kNumBins> lcount{}, rcount{};
        lmin[0] = bins[0].bmin;
        lmax[0] = bins[0].bmax;
        lcount[0] = bins[0].count;
        for (int k = 1; k < kNumBins; ++k) {
            lmin[k] = lmin[k - 1].cwiseMin(bins[k].bmin);
            lmax[k] = lmax[k - 1].cwiseMax(bins[k].bmax);
            lcount[k] = lcount[k - 1] + bins[k].count;
        }
        rmin[kNumBins - 1] = bins[kNumBins - 1].bmin;
        rmax[kNumBins - 1] = bins[kNumBins - 1].bmax;
        rcount[kNumBins - 1] = bins[kNumBins - 1].count;
        for (int k = kNumBins - 2; k >= 0; --k) {
            rmin[k] = rmin[k + 1].cwiseMin(bins[k].bmin);
            rmax[k] = rmax[k + 1].cwiseMax(bins[k].bmax);
            rcount[k] = rcount[k + 1] + bins[k].count;
        }

        for (int k = 1; k < kNumBins; ++k) {
            if (lcount[k - 1] == 0 || rcount[k] == 0) continue;
            const double cost =
                lcount[k - 1] * aabb_surface_area(lmin[k - 1], lmax[k - 1]) +
                rcount[k] * aabb_surface_area(rmin[k], rmax[k]);
            if (cost < best.cost) {
                best.cost = cost;
                best.axis = axis;
                best.bin = k;
            }
        }
    }

    return best;
}

void fill_aabb(BVHNode &node, const std::vector<FaceBox> &fa,
               const std::vector<int> &face_indices, int start, int end) {
    Eigen::Vector3d bmin = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    Eigen::Vector3d bmax = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (int i = start; i < end; ++i) {
        bmin = bmin.cwiseMin(fa[face_indices[i]].bmin);
        bmax = bmax.cwiseMax(fa[face_indices[i]].bmax);
    }
    node.bmin = bmin;
    node.bmax = bmax;
}

void make_leaf(BVHNode &node, const FaceGeom &g,
               const std::vector<int> &face_indices, int start, int end) {
    node.left = -1;
    node.right = -1;
    node.face_start = start;
    node.face_end = end;

    double area_sum = 0.0;
    Eigen::Vector3d c_num = Eigen::Vector3d::Zero();
    Eigen::Vector3d n_sum = Eigen::Vector3d::Zero();
    for (int i = start; i < end; ++i) {
        const int t = face_indices[i];
        const double a = g.A(t);
        const Eigen::Vector3d c = g.C.row(t);
        const Eigen::Vector3d n = g.N.row(t);
        area_sum += a;
        c_num += a * c;
        n_sum += a * n;
    }
    node.area = area_sum;
    node.centroid = c_num / area_sum;
    node.normal_sum = n_sum;

    double r = 0.0;
    for (int i = start; i < end; ++i) {
        const int t = face_indices[i];
        const Eigen::Vector3d c = g.C.row(t);
        r = std::max(r, (c - node.centroid).norm());
    }
    node.radius = r;
}

// Recursive builder. Returns the index of the node it just created.
int build_recursive(BVH &bvh, const std::vector<FaceBox> &fa,
                    const FaceGeom &g, int start, int end, int max_leaf_size) {
    const int node_idx = static_cast<int>(bvh.nodes.size());
    bvh.nodes.emplace_back();
    fill_aabb(bvh.nodes[node_idx], fa, bvh.face_indices, start, end);

    const int n = end - start;
    if (n <= max_leaf_size) {
        make_leaf(bvh.nodes[node_idx], g, bvh.face_indices, start, end);
        return node_idx;
    }

    SplitChoice split = find_sah_split(fa, bvh.face_indices, start, end);

    int split_pos = -1;
    if (split.axis >= 0) {
        // Partition face_indices[start, end) into {bin < cut} and {bin >= cut}.
        Eigen::Vector3d cmin = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::infinity());
        Eigen::Vector3d cmax = Eigen::Vector3d::Constant(
            -std::numeric_limits<double>::infinity());
        for (int i = start; i < end; ++i) {
            cmin = cmin.cwiseMin(fa[bvh.face_indices[i]].centroid);
            cmax = cmax.cwiseMax(fa[bvh.face_indices[i]].centroid);
        }
        const int axis = split.axis;
        const double scale = kNumBins / (cmax[axis] - cmin[axis]);
        const int bin_cut = split.bin;
        auto in_left = [&](int face_id) -> bool {
            const double c = fa[face_id].centroid[axis];
            int b = static_cast<int>(scale * (c - cmin[axis]));
            if (b < 0) b = 0;
            if (b >= kNumBins) b = kNumBins - 1;
            return b < bin_cut;
        };
        split_pos = static_cast<int>(
            std::partition(bvh.face_indices.begin() + start,
                           bvh.face_indices.begin() + end, in_left) -
            bvh.face_indices.begin());
    }

    // Fallback: median split on axis with largest centroid spread.
    // Happens if SAH found no valid split (all faces in one bin) or if
    // partition degenerated to one side.
    if (split.axis < 0 || split_pos <= start || split_pos >= end) {
        Eigen::Vector3d cmin = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::infinity());
        Eigen::Vector3d cmax = Eigen::Vector3d::Constant(
            -std::numeric_limits<double>::infinity());
        for (int i = start; i < end; ++i) {
            cmin = cmin.cwiseMin(fa[bvh.face_indices[i]].centroid);
            cmax = cmax.cwiseMax(fa[bvh.face_indices[i]].centroid);
        }
        const Eigen::Vector3d spread = cmax - cmin;
        int axis = 0;
        if (spread.y() > spread[axis]) axis = 1;
        if (spread.z() > spread[axis]) axis = 2;

        if (spread[axis] < 1e-20) {
            // All centroids coincide — keep as leaf even if large.
            make_leaf(bvh.nodes[node_idx], g, bvh.face_indices, start, end);
            return node_idx;
        }

        const int mid = (start + end) / 2;
        std::nth_element(bvh.face_indices.begin() + start,
                         bvh.face_indices.begin() + mid,
                         bvh.face_indices.begin() + end,
                         [&](int a_id, int b_id) {
                             return fa[a_id].centroid[axis] <
                                    fa[b_id].centroid[axis];
                         });
        split_pos = mid;
    }

    const int left = build_recursive(bvh, fa, g, start, split_pos, max_leaf_size);
    const int right = build_recursive(bvh, fa, g, split_pos, end, max_leaf_size);

    // Re-access by index (vector may have reallocated during recursion).
    BVHNode &node = bvh.nodes[node_idx];
    const BVHNode &L = bvh.nodes[left];
    const BVHNode &R = bvh.nodes[right];
    node.left = left;
    node.right = right;
    // face_indices is partitioned in-place, so every node (leaf or internal)
    // owns a contiguous slice [face_start, face_end). Setting these on
    // internal nodes lets consumers iterate an admissible cluster's faces
    // without recursing through its subtree.
    node.face_start = start;
    node.face_end = end;
    node.area = L.area + R.area;
    node.centroid = (L.area * L.centroid + R.area * R.centroid) / node.area;
    node.normal_sum = L.normal_sum + R.normal_sum;
    // Triangle inequality upper bound: any face t in L-subtree is within
    // r_L of c_L, and c_L is ||c_L - c_U|| from c_U, so ||c_t - c_U|| <=
    // ||c_L - c_U|| + r_L. Same for R. Take the max.
    node.radius = std::max(
        (L.centroid - node.centroid).norm() + L.radius,
        (R.centroid - node.centroid).norm() + R.radius);
    return node_idx;
}

// Post-order refresh of cluster aggregates. Internal nodes are rebuilt from
// children using the same identities the builder uses, so this is bit-for-bit
// identical to a fresh build's aggregates up to float summation order.
void refresh_aggregates(BVH &bvh, int u, const FaceGeom &g) {
    BVHNode &node = bvh.nodes[u];
    if (node.is_leaf()) {
        double area_sum = 0.0;
        Eigen::Vector3d c_num = Eigen::Vector3d::Zero();
        Eigen::Vector3d n_sum = Eigen::Vector3d::Zero();
        for (int i = node.face_start; i < node.face_end; ++i) {
            const int t = bvh.face_indices[i];
            const double a = g.A(t);
            const Eigen::Vector3d c = g.C.row(t);
            const Eigen::Vector3d n = g.N.row(t);
            area_sum += a;
            c_num += a * c;
            n_sum += a * n;
        }
        node.area = area_sum;
        node.centroid = c_num / area_sum;
        node.normal_sum = n_sum;
        double r = 0.0;
        for (int i = node.face_start; i < node.face_end; ++i) {
            const int t = bvh.face_indices[i];
            const Eigen::Vector3d c = g.C.row(t);
            r = std::max(r, (c - node.centroid).norm());
        }
        node.radius = r;
    } else {
        refresh_aggregates(bvh, node.left, g);
        refresh_aggregates(bvh, node.right, g);
        const BVHNode &L = bvh.nodes[node.left];
        const BVHNode &R = bvh.nodes[node.right];
        node.area = L.area + R.area;
        node.centroid = (L.area * L.centroid + R.area * R.centroid) / node.area;
        node.normal_sum = L.normal_sum + R.normal_sum;
        node.radius = std::max(
            (L.centroid - node.centroid).norm() + L.radius,
            (R.centroid - node.centroid).norm() + R.radius);
    }
}

} // anonymous namespace

BVH build_bvh(const MeshData &mesh, const FaceGeom &g, int max_leaf_size) {
    BVH bvh;
    const int nf = static_cast<int>(mesh.F.rows());
    bvh.face_indices.resize(nf);
    for (int i = 0; i < nf; ++i) bvh.face_indices[i] = i;
    bvh.nodes.reserve(2 * nf);

    std::vector<FaceBox> fa = compute_face_boxes(mesh, g);
    bvh.root = build_recursive(bvh, fa, g, 0, nf, max_leaf_size);
    return bvh;
}

BVH build_bvh(const MeshData &mesh, int max_leaf_size) {
    const FaceGeom g = compute_face_geom(mesh);
    return build_bvh(mesh, g, max_leaf_size);
}

void update_bvh_aggregates(BVH &bvh, const FaceGeom &g) {
    if (bvh.nodes.empty()) return;
    refresh_aggregates(bvh, bvh.root, g);
}

} // namespace rsh
