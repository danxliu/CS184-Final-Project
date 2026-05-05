#include "SurfaceBarrier.h"

#include "DeterministicReduction.h"
#include "FaceGeom.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rsh {

namespace {

using Vec3 = Eigen::Vector3d;
struct SubTriState {
    std::array<Vec3, 3> bary;
    double area_scale = 1.0;
    int depth = 0;
};

struct AdaptivePairItem {
    SubTriState moving;
    SubTriState barrier;
};

inline Vec3 face_vertex(const MeshData &mesh, int t, int k) {
    return mesh.V.row(mesh.F(t, k));
}

Vec3 bary_blend(const std::array<Vec3, 3> &tri, const Vec3 &bary) {
    return bary.x() * tri[0] + bary.y() * tri[1] + bary.z() * tri[2];
}

Vec3 weighted_face_point(const MeshData &mesh, int t, const Vec3 &w) {
    return w.x() * face_vertex(mesh, t, 0) +
           w.y() * face_vertex(mesh, t, 1) +
           w.z() * face_vertex(mesh, t, 2);
}

double clamp01(double v) {
    return std::max(0.0, std::min(1.0, v));
}

double triangle_diameter_sq(const std::array<Vec3, 3> &tri) {
    const double d01 = (tri[0] - tri[1]).squaredNorm();
    const double d12 = (tri[1] - tri[2]).squaredNorm();
    const double d20 = (tri[2] - tri[0]).squaredNorm();
    return std::max(d01, std::max(d12, d20));
}

double point_triangle_distance_sq(const Vec3 &p,
                                  const Vec3 &a,
                                  const Vec3 &b,
                                  const Vec3 &c) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return ap.squaredNorm();

    const Vec3 bp = p - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return bp.squaredNorm();

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return (p - (a + v * ab)).squaredNorm();
    }

    const Vec3 cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return cp.squaredNorm();

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return (p - (a + w * ac)).squaredNorm();
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + w * (c - b))).squaredNorm();
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    return (p - (a + ab * v + ac * w)).squaredNorm();
}

double segment_segment_distance_sq(const Vec3 &p1,
                                   const Vec3 &q1,
                                   const Vec3 &p2,
                                   const Vec3 &q2) {
    constexpr double eps = 1e-15;
    const Vec3 d1 = q1 - p1;
    const Vec3 d2 = q2 - p2;
    const Vec3 r = p1 - p2;
    const double a = d1.dot(d1);
    const double e = d2.dot(d2);
    const double f = d2.dot(r);

    double s = 0.0;
    double t = 0.0;

    if (a <= eps && e <= eps) {
        return r.squaredNorm();
    }
    if (a <= eps) {
        t = clamp01(f / e);
    } else {
        const double c = d1.dot(r);
        if (e <= eps) {
            s = clamp01(-c / a);
        } else {
            const double b = d1.dot(d2);
            const double denom = a * e - b * b;
            if (std::abs(denom) > eps) {
                s = clamp01((b * f - c * e) / denom);
            }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp01(-c / a);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp01((b - c) / a);
            }
        }
    }

    const Vec3 c1 = p1 + s * d1;
    const Vec3 c2 = p2 + t * d2;
    return (c1 - c2).squaredNorm();
}

double triangle_triangle_distance_sq(const std::array<Vec3, 3> &t1,
                                     const std::array<Vec3, 3> &t2) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        best = std::min(best,
                        point_triangle_distance_sq(t1[i], t2[0], t2[1], t2[2]));
        best = std::min(best,
                        point_triangle_distance_sq(t2[i], t1[0], t1[1], t1[2]));
    }
    const std::array<std::pair<int, int>, 3> edges = {
        std::make_pair(0, 1), std::make_pair(1, 2), std::make_pair(2, 0)};
    for (const auto &e1 : edges) {
        for (const auto &e2 : edges) {
            best = std::min(best,
                            segment_segment_distance_sq(
                                t1[e1.first], t1[e1.second],
                                t2[e2.first], t2[e2.second]));
        }
    }
    return best;
}

SubTriState root_subtri() {
    SubTriState out;
    out.bary = {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                Vec3(0.0, 0.0, 1.0)};
    out.area_scale = 1.0;
    out.depth = 0;
    return out;
}

void subdivide_subtri(const SubTriState &in,
                      std::array<SubTriState, 4> &children) {
    const Vec3 &b0 = in.bary[0];
    const Vec3 &b1 = in.bary[1];
    const Vec3 &b2 = in.bary[2];
    const Vec3 m01 = 0.5 * (b0 + b1);
    const Vec3 m12 = 0.5 * (b1 + b2);
    const Vec3 m20 = 0.5 * (b2 + b0);
    const double child_area = 0.25 * in.area_scale;
    const int child_depth = in.depth + 1;

    children[0] = {{b0, m01, m20}, child_area, child_depth};
    children[1] = {{m01, b1, m12}, child_area, child_depth};
    children[2] = {{m20, m12, b2}, child_area, child_depth};
    children[3] = {{m01, m12, m20}, child_area, child_depth};
}

void subtri_world_triangle(const SubTriState &sub,
                           const std::array<Vec3, 3> &parent,
                           std::array<Vec3, 3> &out) {
    out[0] = bary_blend(parent, sub.bary[0]);
    out[1] = bary_blend(parent, sub.bary[1]);
    out[2] = bary_blend(parent, sub.bary[2]);
}

Vec3 subtri_centroid_weights(const SubTriState &sub) {
    return (sub.bary[0] + sub.bary[1] + sub.bary[2]) / 3.0;
}

void emit_subtri_term(std::vector<TpeNearFieldTerm> &terms,
                      int tm,
                      int tb,
                      const SubTriState &moving,
                      const SubTriState &barrier) {
    TpeNearFieldTerm term;
    term.t1 = tm;
    term.t2 = tb;
    term.w1 = subtri_centroid_weights(moving);
    term.w2 = subtri_centroid_weights(barrier);
    term.area_scale_1 = moving.area_scale;
    term.area_scale_2 = barrier.area_scale;
    terms.push_back(term);
}

void append_adaptive_terms_for_cross_face_pair(
    const MeshData &mesh,
    const MeshData &barrier,
    int tm,
    int tb,
    const TpeAdaptiveParams &adaptive,
    std::vector<TpeNearFieldTerm> &terms) {
    const std::array<Vec3, 3> parent_m = {
        face_vertex(mesh, tm, 0), face_vertex(mesh, tm, 1),
        face_vertex(mesh, tm, 2)};
    const std::array<Vec3, 3> parent_b = {
        face_vertex(barrier, tb, 0), face_vertex(barrier, tb, 1),
        face_vertex(barrier, tb, 2)};

    const double theta = std::max(0.0, adaptive.theta);
    const double theta_sq = theta * theta;
    const int max_depth = std::max(0, adaptive.max_depth);
    const int max_stack_items = std::max(16, adaptive.max_stack_items);
    std::vector<AdaptivePairItem> stack;
    stack.reserve(128);
    stack.push_back({root_subtri(), root_subtri()});

    while (!stack.empty()) {
        const AdaptivePairItem item = stack.back();
        stack.pop_back();

        std::array<Vec3, 3> tri_m;
        std::array<Vec3, 3> tri_b;
        subtri_world_triangle(item.moving, parent_m, tri_m);
        subtri_world_triangle(item.barrier, parent_b, tri_b);

        const double dist_sq = triangle_triangle_distance_sq(tri_m, tri_b);
        const double diam_m_sq = triangle_diameter_sq(tri_m);
        const double diam_b_sq = triangle_diameter_sq(tri_b);
        const bool mac_ok =
            dist_sq > 0.0 &&
            std::max(diam_m_sq, diam_b_sq) <= theta_sq * dist_sq;
        const bool depth_cap = item.moving.depth >= max_depth;
        const bool stack_cap =
            static_cast<int>(stack.size()) + 16 > max_stack_items;
        if (mac_ok || depth_cap || stack_cap) {
            emit_subtri_term(terms, tm, tb, item.moving, item.barrier);
            continue;
        }

        std::array<SubTriState, 4> children_m;
        std::array<SubTriState, 4> children_b;
        subdivide_subtri(item.moving, children_m);
        subdivide_subtri(item.barrier, children_b);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                stack.push_back({children_m[i], children_b[j]});
            }
        }
    }
}

std::vector<std::array<Vec3, 3>> build_opposite_edges_for_mesh(
    const MeshData &mesh) {
    std::vector<std::array<Vec3, 3>> E(static_cast<size_t>(mesh.n_faces()));
    for (int t = 0; t < mesh.n_faces(); ++t) {
        const Vec3 v0 = mesh.V.row(mesh.F(t, 0));
        const Vec3 v1 = mesh.V.row(mesh.F(t, 1));
        const Vec3 v2 = mesh.V.row(mesh.F(t, 2));
        opposite_edges(v0, v1, v2, E[static_cast<size_t>(t)][0],
                       E[static_cast<size_t>(t)][1],
                       E[static_cast<size_t>(t)][2]);
    }
    return E;
}

double ordered_tpe_term(double a_source,
                        double a_target,
                        const Vec3 &c_source,
                        const Vec3 &c_target,
                        const Vec3 &n_source,
                        double alpha) {
    const Vec3 d = c_source - c_target;
    const double r2 = d.squaredNorm();
    if (!(r2 > 0.0)) return 0.0;
    const double s = n_source.dot(d);
    const double num = std::pow(std::abs(s), alpha);
    const double den = std::pow(r2, alpha);
    return a_source * a_target * num / den;
}

double ordered_cluster_tpe_term(const BVHNode &source,
                                const BVHNode &target,
                                double alpha) {
    if (!(source.area > 0.0) || !(target.area > 0.0)) {
        return 0.0;
    }
    const Vec3 d = source.centroid - target.centroid;
    const double r2 = d.squaredNorm();
    if (!(r2 > 0.0)) return 0.0;
    const double q = d.dot(source.projector_sum * d);
    if (!(q > 0.0)) return 0.0;
    const double half_alpha = 0.5 * alpha;
    return target.area * std::pow(source.area, 1.0 - half_alpha) *
           std::pow(q, half_alpha) / std::pow(r2, alpha);
}

void accumulate_dynamic_source_gradient(const MeshData &mesh,
                                        const FaceGeom &gm,
                                        const FaceGeom &gb,
                                        const std::vector<std::array<Vec3, 3>> &E,
                                        int tm,
                                        int tb,
                                        double alpha,
                                        Eigen::MatrixXd &G) {
    const Vec3 c1 = gm.C.row(tm);
    const Vec3 n1 = gm.N.row(tm);
    const double a1 = gm.A(tm);
    const Vec3 c2 = gb.C.row(tb);
    const double a2 = gb.A(tb);
    const Vec3 d = c1 - c2;
    const double r2 = d.squaredNorm();
    if (!(r2 > 0.0)) return;

    const double s = n1.dot(d);
    const double s2 = s * s;
    const double inv_r2alpha = std::pow(r2, -alpha);
    const double s_abs_alpha = std::pow(s2, 0.5 * alpha);
    const double s_signed_power =
        alpha * s * std::pow(s2, 0.5 * (alpha - 2.0));

    const double K_factor = s_abs_alpha * inv_r2alpha;
    const double dK_da1 = a2 * K_factor;
    const double coef_n = a1 * a2 * s_signed_power * inv_r2alpha;
    const double coef_d = 2.0 * alpha * a1 * a2 * K_factor / r2;

    const Eigen::RowVector3d dK_dc1 =
        (coef_n * n1 - coef_d * d).transpose();
    const Eigen::RowVector3d dK_dn1 = (coef_n * d).transpose();
    const Eigen::RowVector3d dK_dc1_over3 = dK_dc1 / 3.0;

    for (int k = 0; k < 3; ++k) {
        const int vi = mesh.F(tm, k);
        const Eigen::Matrix3d Jn1 =
            dn_dvk(n1, a1, E[static_cast<size_t>(tm)][k]);
        const Eigen::RowVector3d Ja1 =
            da_dvk(n1, E[static_cast<size_t>(tm)][k]);
        G.row(vi) += dK_dc1_over3 + dK_dn1 * Jn1 + dK_da1 * Ja1;
    }
}

void accumulate_dynamic_target_gradient(const MeshData &mesh,
                                        const FaceGeom &gm,
                                        const FaceGeom &gb,
                                        const std::vector<std::array<Vec3, 3>> &E,
                                        int tm,
                                        int tb,
                                        double alpha,
                                        Eigen::MatrixXd &G) {
    const Vec3 c_source = gb.C.row(tb);
    const Vec3 n_source = gb.N.row(tb);
    const double a_source = gb.A(tb);
    const Vec3 c_target = gm.C.row(tm);
    const double a_target = gm.A(tm);
    const Vec3 d = c_source - c_target;
    const double r2 = d.squaredNorm();
    if (!(r2 > 0.0)) return;

    const double s = n_source.dot(d);
    const double s2 = s * s;
    const double inv_r2alpha = std::pow(r2, -alpha);
    const double s_abs_alpha = std::pow(s2, 0.5 * alpha);
    const double s_signed_power =
        alpha * s * std::pow(s2, 0.5 * (alpha - 2.0));

    const double K_factor = s_abs_alpha * inv_r2alpha;
    const double dK_da_target = a_source * K_factor;
    const double coef_n = a_source * a_target * s_signed_power * inv_r2alpha;
    const double coef_d =
        2.0 * alpha * a_source * a_target * K_factor / r2;
    const Eigen::RowVector3d dK_dc_source =
        (coef_n * n_source - coef_d * d).transpose();
    const Eigen::RowVector3d dK_dc_target = -dK_dc_source;
    const Eigen::RowVector3d dK_dc_target_over3 = dK_dc_target / 3.0;

    const Vec3 n_target = gm.N.row(tm);
    for (int k = 0; k < 3; ++k) {
        const int vi = mesh.F(tm, k);
        const Eigen::RowVector3d Ja_target =
            da_dvk(n_target, E[static_cast<size_t>(tm)][k]);
        G.row(vi) += dK_dc_target_over3 + dK_da_target * Ja_target;
    }
}

void accumulate_admissible_gradient(const MeshData &mesh,
                                    const FaceGeom &gm,
                                    const BVH &moving_bvh,
                                    const BVH &barrier_bvh,
                                    const BlockPairs &bp,
                                    double alpha,
                                    const std::vector<std::array<Vec3, 3>> &E,
                                    Eigen::MatrixXd &G) {
    const int nf = mesh.n_faces();
    const double half_alpha = 0.5 * alpha;

    Eigen::VectorXd dE_da_face = Eigen::VectorXd::Zero(nf);
    Eigen::MatrixXd dE_dc_face = Eigen::MatrixXd::Zero(nf, 3);
    Eigen::MatrixXd dE_dn_face = Eigen::MatrixXd::Zero(nf, 3);

    for (const ClusterPair &cp : bp.admissible) {
        const BVHNode &U = moving_bvh.nodes[cp.u];
        const BVHNode &V = barrier_bvh.nodes[cp.v];
        const double aU = U.area;
        const double aV = V.area;
        if (!(aU > 0.0) || !(aV > 0.0)) {
            continue;
        }

        // Ordered term where the moving cluster is the source. Its gradient
        // sees moving area, centroid, and normal/projector aggregates.
        {
            const Vec3 d = U.centroid - V.centroid;
            const double r2 = d.squaredNorm();
            const Vec3 PSd = U.projector_sum * d;
            const double q = d.dot(PSd);
            if (r2 > 0.0 && q > 0.0) {
                const double K =
                    aV * std::pow(aU, 1.0 - half_alpha) *
                    std::pow(q, half_alpha) / std::pow(r2, alpha);
                const Vec3 dK_dcU =
                    K * (alpha / q * PSd - (2.0 * alpha / r2) * d);
                const double dK_daU_via_area =
                    (1.0 - half_alpha) * K / aU;
                const double K_alpha_over_2q = 0.5 * K * alpha / q;
                const double K_alpha_over_q = K * alpha / q;

                for (int i = U.face_start; i < U.face_end; ++i) {
                    const int t = moving_bvh.face_indices[i];
                    const double a_t = gm.A(t);
                    const Vec3 c_t = gm.C.row(t);
                    const Vec3 n_t = gm.N.row(t);
                    const double nd = n_t.dot(d);
                    dE_da_face(t) +=
                        dK_daU_via_area +
                        dK_dcU.dot(c_t - U.centroid) / aU +
                        K_alpha_over_2q * nd * nd;
                    dE_dc_face.row(t) +=
                        (a_t / aU) * dK_dcU.transpose();
                    dE_dn_face.row(t) +=
                        (K_alpha_over_q * a_t * nd) * d.transpose();
                }
            }
        }

        // Ordered term where the fixed barrier cluster is the source and the
        // moving cluster is only the target. No moving normal derivative.
        {
            const Vec3 d = V.centroid - U.centroid;
            const double r2 = d.squaredNorm();
            const Vec3 PSd = V.projector_sum * d;
            const double q = d.dot(PSd);
            if (r2 > 0.0 && q > 0.0) {
                const double K =
                    aU * std::pow(aV, 1.0 - half_alpha) *
                    std::pow(q, half_alpha) / std::pow(r2, alpha);
                const Vec3 dK_dc_source =
                    K * (alpha / q * PSd - (2.0 * alpha / r2) * d);
                const Vec3 dK_dcU = -dK_dc_source;
                const double dK_daU = K / aU;

                for (int i = U.face_start; i < U.face_end; ++i) {
                    const int t = moving_bvh.face_indices[i];
                    const double a_t = gm.A(t);
                    const Vec3 c_t = gm.C.row(t);
                    dE_da_face(t) +=
                        dK_daU + dK_dcU.dot(c_t - U.centroid) / aU;
                    dE_dc_face.row(t) +=
                        (a_t / aU) * dK_dcU.transpose();
                }
            }
        }
    }

    for (int t = 0; t < nf; ++t) {
        if (dE_da_face(t) == 0.0 && dE_dc_face.row(t).isZero() &&
            dE_dn_face.row(t).isZero()) {
            continue;
        }
        const double a_t = gm.A(t);
        const Vec3 n_t = gm.N.row(t);
        const Eigen::RowVector3d dE_dc_t = dE_dc_face.row(t);
        const Eigen::RowVector3d dE_dn_t = dE_dn_face.row(t);
        const double dE_da_t = dE_da_face(t);
        const Eigen::RowVector3d dc_over3 = dE_dc_t / 3.0;
        for (int k = 0; k < 3; ++k) {
            const Eigen::Matrix3d Jn = dn_dvk(n_t, a_t, E[t][k]);
            const Eigen::RowVector3d Ja = da_dvk(n_t, E[t][k]);
            G.row(mesh.F(t, k)) += dc_over3 + dE_dn_t * Jn + dE_da_t * Ja;
        }
    }
}

double admissible_energy(const BVH &moving_bvh,
                         const BVH &barrier_bvh,
                         const BlockPairs &bp,
                         double alpha) {
    const int n_pairs = static_cast<int>(bp.admissible.size());
    const int lanes = canonical_reduction_lanes();
    std::vector<double> partial(static_cast<size_t>(lanes), 0.0);
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const IndexRange range =
            canonical_static_range(n_pairs, lane, lanes);
        double local = 0.0;
        for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
            const ClusterPair &cp =
                bp.admissible[static_cast<size_t>(pair_idx)];
            const BVHNode &U = moving_bvh.nodes[cp.u];
            const BVHNode &V = barrier_bvh.nodes[cp.v];
            local += ordered_cluster_tpe_term(U, V, alpha);
            local += ordered_cluster_tpe_term(V, U, alpha);
        }
        partial[static_cast<size_t>(lane)] = local;
    }
    double phi = 0.0;
    for (double p : partial) phi += p;
    return phi;
}

double nearfield_energy(const FaceGeom &gm,
                        const FaceGeom &gb,
                        const BVH &moving_bvh,
                        const BVH &barrier_bvh,
                        const BlockPairs &bp,
                        double alpha) {
    const int n_pairs = static_cast<int>(bp.near_field.size());
    const int lanes = canonical_reduction_lanes();
    std::vector<double> partial(static_cast<size_t>(lanes), 0.0);
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const IndexRange range =
            canonical_static_range(n_pairs, lane, lanes);
        double local = 0.0;
        for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
            const ClusterPair &cp =
                bp.near_field[static_cast<size_t>(pair_idx)];
            const BVHNode &U = moving_bvh.nodes[cp.u];
            const BVHNode &V = barrier_bvh.nodes[cp.v];
            for (int i = U.face_start; i < U.face_end; ++i) {
                const int tm = moving_bvh.face_indices[i];
                const Vec3 cm = gm.C.row(tm);
                const Vec3 nm = gm.N.row(tm);
                const double am = gm.A(tm);
                for (int j = V.face_start; j < V.face_end; ++j) {
                    const int tb = barrier_bvh.face_indices[j];
                    const Vec3 cb = gb.C.row(tb);
                    const Vec3 nb = gb.N.row(tb);
                    const double ab = gb.A(tb);
                    local += ordered_tpe_term(am, ab, cm, cb, nm, alpha);
                    local += ordered_tpe_term(ab, am, cb, cm, nb, alpha);
                }
            }
        }
        partial[static_cast<size_t>(lane)] = local;
    }
    double phi = 0.0;
    for (double p : partial) phi += p;
    return phi;
}

double adaptive_nearfield_energy(const MeshData &mesh,
                                 const MeshData &barrier,
                                 const FaceGeom &gm,
                                 const FaceGeom &gb,
                                 const std::vector<TpeNearFieldTerm> &terms,
                                 double alpha) {
    const int n_terms = static_cast<int>(terms.size());
    const int lanes = canonical_reduction_lanes();
    std::vector<double> partial(static_cast<size_t>(lanes), 0.0);
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        const IndexRange range =
            canonical_static_range(n_terms, lane, lanes);
        double local = 0.0;
        for (int term_idx = range.begin; term_idx < range.end; ++term_idx) {
            const TpeNearFieldTerm &term =
                terms[static_cast<size_t>(term_idx)];
            const Vec3 cm = weighted_face_point(mesh, term.t1, term.w1);
            const Vec3 cb = weighted_face_point(barrier, term.t2, term.w2);
            const Vec3 nm = gm.N.row(term.t1);
            const Vec3 nb = gb.N.row(term.t2);
            const double am = term.area_scale_1 * gm.A(term.t1);
            const double ab = term.area_scale_2 * gb.A(term.t2);
            local += ordered_tpe_term(am, ab, cm, cb, nm, alpha);
            local += ordered_tpe_term(ab, am, cb, cm, nb, alpha);
        }
        partial[static_cast<size_t>(lane)] = local;
    }
    double phi = 0.0;
    for (double p : partial) phi += p;
    return phi;
}

void accumulate_nearfield_gradient(
    const MeshData &mesh,
    const FaceGeom &gm,
    const FaceGeom &gb,
    const BVH &moving_bvh,
    const BVH &barrier_bvh,
    const BlockPairs &bp,
    double alpha,
    const std::vector<std::array<Vec3, 3>> &E,
    Eigen::MatrixXd &G) {
    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = moving_bvh.nodes[cp.u];
        const BVHNode &V = barrier_bvh.nodes[cp.v];
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int tm = moving_bvh.face_indices[i];
            for (int j = V.face_start; j < V.face_end; ++j) {
                const int tb = barrier_bvh.face_indices[j];
                accumulate_dynamic_source_gradient(
                    mesh, gm, gb, E, tm, tb, alpha, G);
                accumulate_dynamic_target_gradient(
                    mesh, gm, gb, E, tm, tb, alpha, G);
            }
        }
    }
}

void accumulate_adaptive_cross_term_gradient(
    const MeshData &mesh,
    const MeshData &barrier,
    const FaceGeom &gm,
    const FaceGeom &gb,
    const std::vector<std::array<Vec3, 3>> &E,
    const TpeNearFieldTerm &term,
    double alpha,
    Eigen::MatrixXd &G) {
    const int tm = term.t1;
    const int tb = term.t2;
    const Vec3 cm = weighted_face_point(mesh, tm, term.w1);
    const Vec3 cb = weighted_face_point(barrier, tb, term.w2);
    const Vec3 nm = gm.N.row(tm);
    const Vec3 nb = gb.N.row(tb);
    const double am = term.area_scale_1 * gm.A(tm);
    const double ab = term.area_scale_2 * gb.A(tb);

    // Ordered term with the moving shell as source.
    {
        const Vec3 d = cm - cb;
        const double r2 = d.squaredNorm();
        if (r2 > 0.0) {
            const double s = nm.dot(d);
            const double s2 = s * s;
            const double inv_r2alpha = std::pow(r2, -alpha);
            const double s_abs_alpha = std::pow(s2, 0.5 * alpha);
            const double s_signed_power =
                alpha * s * std::pow(s2, 0.5 * (alpha - 2.0));
            const double K_factor = s_abs_alpha * inv_r2alpha;
            const double dK_da_m = ab * K_factor;
            const double coef_n = am * ab * s_signed_power * inv_r2alpha;
            const double coef_d = 2.0 * alpha * am * ab * K_factor / r2;
            const Eigen::RowVector3d dK_dc_m =
                (coef_n * nm - coef_d * d).transpose();
            const Eigen::RowVector3d dK_dn_m = (coef_n * d).transpose();

            for (int k = 0; k < 3; ++k) {
                const int vi = mesh.F(tm, k);
                const Eigen::Matrix3d Jn_m =
                    dn_dvk(nm, gm.A(tm), E[static_cast<size_t>(tm)][k]);
                const Eigen::RowVector3d Ja_m =
                    term.area_scale_1 *
                    da_dvk(nm, E[static_cast<size_t>(tm)][k]);
                G.row(vi) +=
                    term.w1(k) * dK_dc_m + dK_dn_m * Jn_m +
                    dK_da_m * Ja_m;
            }
        }
    }

    // Ordered term with the fixed obstacle as source and the moving shell as
    // target. Only the moving centroid and area vary.
    {
        const Vec3 d = cb - cm;
        const double r2 = d.squaredNorm();
        if (r2 > 0.0) {
            const double s = nb.dot(d);
            const double s2 = s * s;
            const double inv_r2alpha = std::pow(r2, -alpha);
            const double s_abs_alpha = std::pow(s2, 0.5 * alpha);
            const double s_signed_power =
                alpha * s * std::pow(s2, 0.5 * (alpha - 2.0));
            const double K_factor = s_abs_alpha * inv_r2alpha;
            const double dK_da_m = ab * K_factor;
            const double coef_n = ab * am * s_signed_power * inv_r2alpha;
            const double coef_d = 2.0 * alpha * ab * am * K_factor / r2;
            const Eigen::RowVector3d dK_dc_source =
                (coef_n * nb - coef_d * d).transpose();
            const Eigen::RowVector3d dK_dc_m = -dK_dc_source;

            for (int k = 0; k < 3; ++k) {
                const int vi = mesh.F(tm, k);
                const Eigen::RowVector3d Ja_m =
                    term.area_scale_1 *
                    da_dvk(nm, E[static_cast<size_t>(tm)][k]);
                G.row(vi) += term.w1(k) * dK_dc_m + dK_da_m * Ja_m;
            }
        }
    }
}

void accumulate_adaptive_nearfield_gradient(
    const MeshData &mesh,
    const MeshData &barrier,
    const FaceGeom &gm,
    const FaceGeom &gb,
    const std::vector<TpeNearFieldTerm> &terms,
    double alpha,
    const std::vector<std::array<Vec3, 3>> &E,
    Eigen::MatrixXd &G) {
    const int lanes = canonical_reduction_lanes();
    std::vector<Eigen::MatrixXd> tls_G;
    tls_G.reserve(static_cast<size_t>(lanes));
    for (int tid = 0; tid < lanes; ++tid) {
        tls_G.push_back(Eigen::MatrixXd::Zero(G.rows(), G.cols()));
    }

    const int n_terms = static_cast<int>(terms.size());
#pragma omp parallel for schedule(static)
    for (int lane = 0; lane < lanes; ++lane) {
        Eigen::MatrixXd &G_local = tls_G[static_cast<size_t>(lane)];
        const IndexRange range =
            canonical_static_range(n_terms, lane, lanes);
        for (int term_idx = range.begin; term_idx < range.end; ++term_idx) {
            accumulate_adaptive_cross_term_gradient(
                mesh, barrier, gm, gb, E,
                terms[static_cast<size_t>(term_idx)], alpha, G_local);
        }
    }
    for (int tid = 0; tid < lanes; ++tid) {
        G += tls_G[static_cast<size_t>(tid)];
    }
}

} // namespace

double surface_tpe_barrier_energy(const MeshData &mesh,
                                  const MeshData &barrier,
                                  double alpha) {
    const FaceGeom gm = compute_face_geom(mesh);
    const FaceGeom gb = compute_face_geom(barrier);
    double phi = 0.0;
    for (int tm = 0; tm < mesh.n_faces(); ++tm) {
        const Vec3 cm = gm.C.row(tm);
        const Vec3 nm = gm.N.row(tm);
        const double am = gm.A(tm);
        for (int tb = 0; tb < barrier.n_faces(); ++tb) {
            const Vec3 cb = gb.C.row(tb);
            const Vec3 nb = gb.N.row(tb);
            const double ab = gb.A(tb);
            phi += ordered_tpe_term(am, ab, cm, cb, nm, alpha);
            phi += ordered_tpe_term(ab, am, cb, cm, nb, alpha);
        }
    }
    return phi;
}

Eigen::MatrixXd surface_tpe_barrier_gradient(const MeshData &mesh,
                                             const MeshData &barrier,
                                             double alpha) {
    const FaceGeom gm = compute_face_geom(mesh);
    const FaceGeom gb = compute_face_geom(barrier);
    const std::vector<std::array<Vec3, 3>> E =
        build_opposite_edges_for_mesh(mesh);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
    for (int tm = 0; tm < mesh.n_faces(); ++tm) {
        for (int tb = 0; tb < barrier.n_faces(); ++tb) {
            accumulate_dynamic_source_gradient(mesh, gm, gb, E, tm, tb,
                                               alpha, G);
            accumulate_dynamic_target_gradient(mesh, gm, gb, E, tm, tb,
                                               alpha, G);
        }
    }
    return G;
}

SurfaceBarrierCache build_surface_tpe_barrier_cache(
    const MeshData &mesh,
    const MeshData &barrier,
    double theta,
    const TpeAdaptiveParams &adaptive) {
    SurfaceBarrierCache out;
    const FaceGeom gm = compute_face_geom(mesh);
    out.barrier_geom = compute_face_geom(barrier);
    out.moving_bvh = build_bvh(mesh, gm);
    out.barrier_bvh = build_bvh(barrier, out.barrier_geom);
    out.bp = build_bct_cross(out.moving_bvh, out.barrier_bvh, theta);
    out.theta = theta;
    out.adaptive = adaptive;
    if (adaptive.enabled) {
        out.adaptive.theta = std::max(0.0, adaptive.theta);
        out.adaptive.max_depth = std::max(0, adaptive.max_depth);
        out.adaptive.max_stack_items =
            std::max(16, adaptive.max_stack_items);
        for (const ClusterPair &cp : out.bp.near_field) {
            const BVHNode &U = out.moving_bvh.nodes[cp.u];
            const BVHNode &V = out.barrier_bvh.nodes[cp.v];
            for (int i = U.face_start; i < U.face_end; ++i) {
                const int tm = out.moving_bvh.face_indices[i];
                for (int j = V.face_start; j < V.face_end; ++j) {
                    const int tb = out.barrier_bvh.face_indices[j];
                    append_adaptive_terms_for_cross_face_pair(
                        mesh, barrier, tm, tb, out.adaptive,
                        out.near_terms);
                }
            }
        }
        out.has_adaptive = true;
    }
    return out;
}

double surface_tpe_barrier_energy_bh(
    const MeshData &mesh,
    const MeshData &barrier,
    const SurfaceBarrierCache &cache,
    double alpha) {
    const FaceGeom gm = compute_face_geom(mesh);
    BVH moving_bvh = cache.moving_bvh;
    update_bvh_aggregates(moving_bvh, gm);
    double phi =
        admissible_energy(moving_bvh, cache.barrier_bvh, cache.bp, alpha);
    if (cache.has_adaptive && cache.adaptive.enabled) {
        phi += adaptive_nearfield_energy(mesh, barrier, gm, cache.barrier_geom,
                                         cache.near_terms, alpha);
    } else {
        phi += nearfield_energy(gm, cache.barrier_geom, moving_bvh,
                                cache.barrier_bvh, cache.bp, alpha);
    }
    return phi;
}

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(
    const MeshData &mesh,
    const MeshData &barrier,
    const SurfaceBarrierCache &cache,
    double alpha) {
    const FaceGeom gm = compute_face_geom(mesh);
    BVH moving_bvh = cache.moving_bvh;
    update_bvh_aggregates(moving_bvh, gm);
    const std::vector<std::array<Vec3, 3>> E =
        build_opposite_edges_for_mesh(mesh);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
    accumulate_admissible_gradient(mesh, gm, moving_bvh, cache.barrier_bvh,
                                   cache.bp, alpha, E, G);
    if (cache.has_adaptive && cache.adaptive.enabled) {
        accumulate_adaptive_nearfield_gradient(
            mesh, barrier, gm, cache.barrier_geom, cache.near_terms, alpha, E,
            G);
    } else {
        accumulate_nearfield_gradient(mesh, gm, cache.barrier_geom, moving_bvh,
                                      cache.barrier_bvh, cache.bp, alpha, E, G);
    }
    return G;
}

double surface_tpe_barrier_energy_bh(const MeshData &mesh,
                                     const MeshData &barrier,
                                     double alpha,
                                     double theta) {
    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, theta);
    return surface_tpe_barrier_energy_bh(mesh, barrier, cache, alpha);
}

double surface_tpe_barrier_energy_bh(const MeshData &mesh,
                                     const MeshData &barrier,
                                     const TpeAdaptiveParams &adaptive,
                                     double alpha,
                                     double theta) {
    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, theta, adaptive);
    return surface_tpe_barrier_energy_bh(mesh, barrier, cache, alpha);
}

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(const MeshData &mesh,
                                                const MeshData &barrier,
                                                double alpha,
                                                double theta) {
    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, theta);
    return surface_tpe_barrier_gradient_bh(mesh, barrier, cache, alpha);
}

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(
    const MeshData &mesh,
    const MeshData &barrier,
    const TpeAdaptiveParams &adaptive,
    double alpha,
    double theta) {
    const SurfaceBarrierCache cache =
        build_surface_tpe_barrier_cache(mesh, barrier, theta, adaptive);
    return surface_tpe_barrier_gradient_bh(mesh, barrier, cache, alpha);
}

} // namespace rsh
