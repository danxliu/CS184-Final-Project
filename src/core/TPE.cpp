#include "TPE.h"
#include "FaceGeom.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    SubTriState s1;
    SubTriState s2;
};

inline Vec3 face_vertex(const MeshData &mesh, int t, int k) {
    return mesh.V.row(mesh.F(t, k));
}

Vec3 bary_blend(const std::array<Vec3, 3> &tri, const Vec3 &bary) {
    return bary.x() * tri[0] + bary.y() * tri[1] + bary.z() * tri[2];
}

Vec3 weighted_face_point(const MeshData &mesh, int t, const Vec3 &w) {
    const Vec3 v0 = face_vertex(mesh, t, 0);
    const Vec3 v1 = face_vertex(mesh, t, 1);
    const Vec3 v2 = face_vertex(mesh, t, 2);
    return w.x() * v0 + w.y() * v1 + w.z() * v2;
}

bool faces_adjacent(const MeshData &mesh, int t1, int t2) {
    int shared = 0;
    for (int i = 0; i < 3; ++i) {
        const int a = mesh.F(t1, i);
        for (int j = 0; j < 3; ++j) {
            if (a == mesh.F(t2, j)) {
                ++shared;
                break;
            }
        }
    }
    return shared > 0;
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
        const Vec3 proj = a + v * ab;
        return (p - proj).squaredNorm();
    }

    const Vec3 cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return cp.squaredNorm();

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const Vec3 proj = a + w * ac;
        return (p - proj).squaredNorm();
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3 proj = b + w * (c - b);
        return (p - proj).squaredNorm();
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    const Vec3 proj = a + ab * v + ac * w;
    return (p - proj).squaredNorm();
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
        s = 0.0;
        t = clamp01(f / e);
    } else {
        const double c = d1.dot(r);
        if (e <= eps) {
            t = 0.0;
            s = clamp01(-c / a);
        } else {
            const double b = d1.dot(d2);
            const double denom = a * e - b * b;
            if (std::abs(denom) > eps) {
                s = clamp01((b * f - c * e) / denom);
            } else {
                s = 0.0;
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
            best = std::min(
                best,
                segment_segment_distance_sq(
                    t1[e1.first], t1[e1.second], t2[e2.first], t2[e2.second]));
        }
    }

    return best;
}

SubTriState root_subtri() {
    SubTriState out;
    out.bary = {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)};
    out.area_scale = 1.0;
    out.depth = 0;
    return out;
}

void subdivide_subtri(const SubTriState &in, std::array<SubTriState, 4> &children) {
    const Vec3 &b0 = in.bary[0];
    const Vec3 &b1 = in.bary[1];
    const Vec3 &b2 = in.bary[2];
    const Vec3 m01 = 0.5 * (b0 + b1);
    const Vec3 m12 = 0.5 * (b1 + b2);
    const Vec3 m20 = 0.5 * (b2 + b0);

    const double child_area = in.area_scale * 0.25;
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

void emit_midpoint_term(std::vector<TpeNearFieldTerm> &terms, int t1, int t2) {
    TpeNearFieldTerm term;
    term.t1 = t1;
    term.t2 = t2;
    terms.push_back(term);
}

void emit_subtri_term(std::vector<TpeNearFieldTerm> &terms,
                      int t1,
                      int t2,
                      const SubTriState &s1,
                      const SubTriState &s2) {
    TpeNearFieldTerm term;
    term.t1 = t1;
    term.t2 = t2;
    term.w1 = subtri_centroid_weights(s1);
    term.w2 = subtri_centroid_weights(s2);
    term.area_scale_1 = s1.area_scale;
    term.area_scale_2 = s2.area_scale;
    terms.push_back(term);
}

void append_adaptive_terms_for_face_pair(const MeshData &mesh,
                                         int t1,
                                         int t2,
                                         const TpeAdaptiveParams &adaptive,
                                         std::vector<TpeNearFieldTerm> &terms) {
    if (faces_adjacent(mesh, t1, t2)) {
        emit_midpoint_term(terms, t1, t2);
        return;
    }

    const std::array<Vec3, 3> parent1 = {
        face_vertex(mesh, t1, 0), face_vertex(mesh, t1, 1), face_vertex(mesh, t1, 2)};
    const std::array<Vec3, 3> parent2 = {
        face_vertex(mesh, t2, 0), face_vertex(mesh, t2, 1), face_vertex(mesh, t2, 2)};

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

        std::array<Vec3, 3> tri1;
        std::array<Vec3, 3> tri2;
        subtri_world_triangle(item.s1, parent1, tri1);
        subtri_world_triangle(item.s2, parent2, tri2);

        const double dist_sq = triangle_triangle_distance_sq(tri1, tri2);
        const double diam1_sq = triangle_diameter_sq(tri1);
        const double diam2_sq = triangle_diameter_sq(tri2);

        const bool mac_ok =
            (dist_sq > 0.0) && (std::max(diam1_sq, diam2_sq) <= theta_sq * dist_sq);
        const bool depth_cap = item.s1.depth >= max_depth;
        const bool stack_cap =
            static_cast<int>(stack.size()) + 16 > max_stack_items;
        if (mac_ok || depth_cap || stack_cap) {
            emit_subtri_term(terms, t1, t2, item.s1, item.s2);
            continue;
        }

        std::array<SubTriState, 4> c1;
        std::array<SubTriState, 4> c2;
        subdivide_subtri(item.s1, c1);
        subdivide_subtri(item.s2, c2);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                stack.push_back({c1[i], c2[j]});
            }
        }
    }
}

std::vector<std::array<Vec3, 3>> build_opposite_edges(const MeshData &mesh) {
    const int nf = mesh.n_faces();
    std::vector<std::array<Vec3, 3>> E(nf);
    for (int t = 0; t < nf; ++t) {
        const Vec3 v0 = face_vertex(mesh, t, 0);
        const Vec3 v1 = face_vertex(mesh, t, 1);
        const Vec3 v2 = face_vertex(mesh, t, 2);
        opposite_edges(v0, v1, v2, E[t][0], E[t][1], E[t][2]);
    }
    return E;
}

double admissible_energy(const FaceGeom &g,
                         const BVH &bvh,
                         const BlockPairs &bp,
                         double alpha) {
    double phi = 0.0;
    for (const ClusterPair &cp : bp.admissible) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const Vec3 d = U.centroid - V.centroid;
        const double r2 = d.squaredNorm();
        const Vec3 nU_mean = U.normal_sum / U.area;
        const double s = nU_mean.dot(d);
        const double num = std::pow(std::abs(s), alpha);
        const double den = std::pow(r2, alpha);
        phi += U.area * V.area * num / den;
    }
    return phi;
}

double midpoint_nearfield_energy(const FaceGeom &g,
                                 const BVH &bvh,
                                 const BlockPairs &bp,
                                 double alpha) {
    double phi = 0.0;
    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const bool self = (cp.u == cp.v);
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t1 = bvh.face_indices[i];
            const Vec3 c1 = g.C.row(t1);
            const Vec3 n1 = g.N.row(t1);
            const double a1 = g.A(t1);
            for (int j = V.face_start; j < V.face_end; ++j) {
                const int t2 = bvh.face_indices[j];
                if (self && t1 == t2) continue;
                const Vec3 c2 = g.C.row(t2);
                const double a2 = g.A(t2);
                const Vec3 chord = c1 - c2;
                const double num = std::pow(std::abs(n1.dot(chord)), alpha);
                const double den = std::pow(chord.squaredNorm(), alpha);
                phi += a1 * a2 * num / den;
            }
        }
    }
    return phi;
}

double adaptive_term_energy(const MeshData &mesh,
                            const FaceGeom &g,
                            const TpeNearFieldTerm &term,
                            double alpha) {
    const Vec3 c1 = weighted_face_point(mesh, term.t1, term.w1);
    const Vec3 c2 = weighted_face_point(mesh, term.t2, term.w2);
    const Vec3 n1 = g.N.row(term.t1);
    const double a1 = term.area_scale_1 * g.A(term.t1);
    const double a2 = term.area_scale_2 * g.A(term.t2);
    const Vec3 d = c1 - c2;
    const double r2 = d.squaredNorm();
    if (r2 == 0.0) return 0.0;
    const double num = std::pow(std::abs(n1.dot(d)), alpha);
    const double den = std::pow(r2, alpha);
    return a1 * a2 * num / den;
}

void accumulate_admissible_gradient(const MeshData &mesh,
                                    const FaceGeom &g,
                                    const BVH &bvh,
                                    const BlockPairs &bp,
                                    double alpha,
                                    const std::vector<std::array<Vec3, 3>> &E,
                                    Eigen::MatrixXd &G) {
    const int nf = static_cast<int>(g.A.size());

    Eigen::VectorXd dE_da_face = Eigen::VectorXd::Zero(nf);
    Eigen::MatrixXd dE_dc_face = Eigen::MatrixXd::Zero(nf, 3);
    Eigen::MatrixXd dE_dn_face = Eigen::MatrixXd::Zero(nf, 3);

    for (const ClusterPair &cp : bp.admissible) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const double aU = U.area;
        const double aV = V.area;
        const Vec3 cU = U.centroid;
        const Vec3 cV = V.centroid;
        const Vec3 SU = U.normal_sum;
        const Vec3 mU = SU / aU;
        const Vec3 d = cU - cV;
        const double r2 = d.squaredNorm();
        if (r2 == 0.0) continue;

        const double s = mU.dot(d);
        const double s2 = s * s;
        const double inv_r2alpha = std::pow(r2, -alpha);
        const double psi = std::pow(s2, alpha * 0.5);
        const double s_signed_pow = s * std::pow(s2, (alpha - 2.0) * 0.5);

        const double dK_daU = (1.0 - alpha) * aV * psi * inv_r2alpha;
        const double dK_daV = aU * psi * inv_r2alpha;
        const Vec3 dK_dSU = aV * alpha * s_signed_pow * inv_r2alpha * d;
        const Vec3 dK_dcU =
            aU * aV * alpha * inv_r2alpha *
            (s_signed_pow * mU - 2.0 * psi * d / r2);
        const Vec3 dK_dcV = -dK_dcU;

        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t = bvh.face_indices[i];
            const double a_t = g.A(t);
            const Vec3 c_t = g.C.row(t);
            const Vec3 n_t = g.N.row(t);
            dE_da_face(t) +=
                dK_daU + dK_dSU.dot(n_t) + dK_dcU.dot(c_t - cU) / aU;
            dE_dc_face.row(t) += (a_t / aU) * dK_dcU.transpose();
            dE_dn_face.row(t) += a_t * dK_dSU.transpose();
        }
        for (int j = V.face_start; j < V.face_end; ++j) {
            const int t = bvh.face_indices[j];
            const double a_t = g.A(t);
            const Vec3 c_t = g.C.row(t);
            dE_da_face(t) += dK_daV + dK_dcV.dot(c_t - cV) / aV;
            dE_dc_face.row(t) += (a_t / aV) * dK_dcV.transpose();
        }
    }

    for (int t = 0; t < nf; ++t) {
        if (dE_da_face(t) == 0.0 && dE_dc_face.row(t).isZero() &&
            dE_dn_face.row(t).isZero()) {
            continue;
        }
        const double a_t = g.A(t);
        const Vec3 n_t = g.N.row(t);
        const Eigen::RowVector3d dE_dc_t = dE_dc_face.row(t);
        const Eigen::RowVector3d dE_dn_t = dE_dn_face.row(t);
        const double dE_da_t = dE_da_face(t);
        const Eigen::RowVector3d dc_over3 = dE_dc_t / 3.0;
        for (int k = 0; k < 3; ++k) {
            const Eigen::Matrix3d Jn_k = dn_dvk(n_t, a_t, E[t][k]);
            const Eigen::RowVector3d Ja_k = da_dvk(n_t, E[t][k]);
            G.row(mesh.F(t, k)) += dc_over3 + dE_dn_t * Jn_k + dE_da_t * Ja_k;
        }
    }
}

void accumulate_nearfield_midpoint_gradient(
    const MeshData &mesh,
    const FaceGeom &g,
    const BVH &bvh,
    const BlockPairs &bp,
    double alpha,
    const std::vector<std::array<Vec3, 3>> &E,
    Eigen::MatrixXd &G) {
    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const bool self = (cp.u == cp.v);
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t1 = bvh.face_indices[i];
            const Vec3 c1 = g.C.row(t1);
            const Vec3 n1 = g.N.row(t1);
            const double a1 = g.A(t1);
            Eigen::Matrix3d Jn1[3];
            Eigen::RowVector3d Ja1[3];
            for (int k = 0; k < 3; ++k) {
                Jn1[k] = dn_dvk(n1, a1, E[t1][k]);
                Ja1[k] = da_dvk(n1, E[t1][k]);
            }
            for (int j = V.face_start; j < V.face_end; ++j) {
                const int t2 = bvh.face_indices[j];
                if (self && t1 == t2) continue;
                const Vec3 c2 = g.C.row(t2);
                const Vec3 n2 = g.N.row(t2);
                const double a2 = g.A(t2);

                const Vec3 d = c1 - c2;
                const double r2 = d.squaredNorm();
                if (r2 == 0.0) continue;

                const double s = n1.dot(d);
                const double s2 = s * s;
                const double inv_r2alpha = std::pow(r2, -alpha);
                const double s_abs_alpha = std::pow(s2, alpha * 0.5);
                const double s_signed_power =
                    alpha * s * std::pow(s2, (alpha - 2.0) * 0.5);

                const double K_factor = s_abs_alpha * inv_r2alpha;
                const double dK_da1 = a2 * K_factor;
                const double dK_da2 = a1 * K_factor;
                const double coef_n = a1 * a2 * s_signed_power * inv_r2alpha;
                const double coef_d = 2.0 * alpha * a1 * a2 * K_factor / r2;

                const Eigen::RowVector3d dK_dc1 =
                    (coef_n * n1 - coef_d * d).transpose();
                const Eigen::RowVector3d dK_dn1 = (coef_n * d).transpose();

                const Eigen::RowVector3d dK_dc1_over3 = dK_dc1 / 3.0;
                const Eigen::RowVector3d dK_dc2_over3 = -dK_dc1_over3;

                for (int k = 0; k < 3; ++k) {
                    G.row(mesh.F(t1, k)) +=
                        dK_dc1_over3 + dK_dn1 * Jn1[k] + dK_da1 * Ja1[k];
                }
                for (int k = 0; k < 3; ++k) {
                    const Eigen::RowVector3d Ja2_k = da_dvk(n2, E[t2][k]);
                    G.row(mesh.F(t2, k)) += dK_dc2_over3 + dK_da2 * Ja2_k;
                }
            }
        }
    }
}

void accumulate_adaptive_term_gradient(
    const MeshData &mesh,
    const FaceGeom &g,
    const std::vector<std::array<Vec3, 3>> &E,
    const TpeNearFieldTerm &term,
    double alpha,
    Eigen::MatrixXd &G) {
    const int t1 = term.t1;
    const int t2 = term.t2;
    const Vec3 c1 = weighted_face_point(mesh, t1, term.w1);
    const Vec3 c2 = weighted_face_point(mesh, t2, term.w2);
    const Vec3 n1 = g.N.row(t1);
    const Vec3 n2 = g.N.row(t2);
    const double a1 = term.area_scale_1 * g.A(t1);
    const double a2 = term.area_scale_2 * g.A(t2);

    const Vec3 d = c1 - c2;
    const double r2 = d.squaredNorm();
    if (r2 == 0.0) return;

    const double s = n1.dot(d);
    const double s2 = s * s;
    const double inv_r2alpha = std::pow(r2, -alpha);
    const double s_abs_alpha = std::pow(s2, alpha * 0.5);
    const double s_signed_power =
        alpha * s * std::pow(s2, (alpha - 2.0) * 0.5);

    const double K_factor = s_abs_alpha * inv_r2alpha;
    const double dK_da1 = a2 * K_factor;
    const double dK_da2 = a1 * K_factor;
    const double coef_n = a1 * a2 * s_signed_power * inv_r2alpha;
    const double coef_d = 2.0 * alpha * a1 * a2 * K_factor / r2;

    const Eigen::RowVector3d dK_dc1 = (coef_n * n1 - coef_d * d).transpose();
    const Eigen::RowVector3d dK_dn1 = (coef_n * d).transpose();
    const Eigen::RowVector3d dK_dc2 = -dK_dc1;

    for (int k = 0; k < 3; ++k) {
        const int vi = mesh.F(t1, k);
        const Eigen::Matrix3d Jn1 = dn_dvk(n1, g.A(t1), E[t1][k]);
        const Eigen::RowVector3d Ja1 = term.area_scale_1 * da_dvk(n1, E[t1][k]);
        G.row(vi) += term.w1(k) * dK_dc1 + dK_dn1 * Jn1 + dK_da1 * Ja1;
    }
    for (int k = 0; k < 3; ++k) {
        const int vi = mesh.F(t2, k);
        const Eigen::RowVector3d Ja2 = term.area_scale_2 * da_dvk(n2, E[t2][k]);
        G.row(vi) += term.w2(k) * dK_dc2 + dK_da2 * Ja2;
    }
}

} // namespace

double tpe_energy_brute(const FaceGeom &g, double alpha) {
    const int nf = static_cast<int>(g.A.size());
    double phi = 0.0;
    for (int t1 = 0; t1 < nf; ++t1) {
        const Vec3 c1 = g.C.row(t1);
        const Vec3 n1 = g.N.row(t1);
        const double a1 = g.A(t1);
        for (int t2 = 0; t2 < nf; ++t2) {
            if (t2 == t1) continue;
            const Vec3 c2 = g.C.row(t2);
            const double a2 = g.A(t2);
            const Vec3 chord = c1 - c2;
            const double num = std::pow(std::abs(n1.dot(chord)), alpha);
            const double den = std::pow(chord.norm(), 2.0 * alpha);
            phi += a1 * a2 * num / den;
        }
    }
    return phi;
}

double tpe_energy_brute(const MeshData &mesh, double alpha) {
    const FaceGeom g = compute_face_geom(mesh);
    return tpe_energy_brute(g, alpha);
}

Eigen::MatrixXd tpe_gradient_brute(const MeshData &mesh,
                                   const FaceGeom &g,
                                   double alpha) {
    const int nf = static_cast<int>(g.A.size());
    const int nv = mesh.n_vertices();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nv, 3);
    const auto E = build_opposite_edges(mesh);

    for (int t1 = 0; t1 < nf; ++t1) {
        const Vec3 c1 = g.C.row(t1);
        const Vec3 n1 = g.N.row(t1);
        const double a1 = g.A(t1);

        Eigen::Matrix3d Jn1[3];
        Eigen::RowVector3d Ja1[3];
        for (int k = 0; k < 3; ++k) {
            Jn1[k] = dn_dvk(n1, a1, E[t1][k]);
            Ja1[k] = da_dvk(n1, E[t1][k]);
        }

        for (int t2 = 0; t2 < nf; ++t2) {
            if (t2 == t1) continue;
            const Vec3 c2 = g.C.row(t2);
            const Vec3 n2 = g.N.row(t2);
            const double a2 = g.A(t2);

            const Vec3 d = c1 - c2;
            const double r2 = d.squaredNorm();
            if (r2 == 0.0) continue;

            const double s = n1.dot(d);
            const double s2 = s * s;
            const double inv_r2alpha = std::pow(r2, -alpha);
            const double s_abs_alpha = std::pow(s2, alpha * 0.5);
            const double s_signed_power =
                alpha * s * std::pow(s2, (alpha - 2.0) * 0.5);

            const double K_factor = s_abs_alpha * inv_r2alpha;
            const double dK_da1 = a2 * K_factor;
            const double dK_da2 = a1 * K_factor;
            const double coef_n = a1 * a2 * s_signed_power * inv_r2alpha;
            const double coef_d = 2.0 * alpha * a1 * a2 * K_factor / r2;

            const Eigen::RowVector3d dK_dc1 =
                (coef_n * n1 - coef_d * d).transpose();
            const Eigen::RowVector3d dK_dn1 = (coef_n * d).transpose();

            const Eigen::RowVector3d dK_dc1_over3 = dK_dc1 / 3.0;
            const Eigen::RowVector3d dK_dc2_over3 = -dK_dc1_over3;

            for (int k = 0; k < 3; ++k) {
                G.row(mesh.F(t1, k)) +=
                    dK_dc1_over3 + dK_dn1 * Jn1[k] + dK_da1 * Ja1[k];
            }
            for (int k = 0; k < 3; ++k) {
                const Eigen::RowVector3d Ja2_k = da_dvk(n2, E[t2][k]);
                G.row(mesh.F(t2, k)) += dK_dc2_over3 + dK_da2 * Ja2_k;
            }
        }
    }
    return G;
}

Eigen::MatrixXd tpe_gradient_brute(const MeshData &mesh, double alpha) {
    const FaceGeom g = compute_face_geom(mesh);
    return tpe_gradient_brute(mesh, g, alpha);
}

double tpe_energy_bh(const FaceGeom &g, const BVH &bvh, const BlockPairs &bp,
                     double alpha) {
    return admissible_energy(g, bvh, bp, alpha) +
           midpoint_nearfield_energy(g, bvh, bp, alpha);
}

TpeAdaptiveCache build_tpe_adaptive_cache(const MeshData &mesh,
                                          const FaceGeom &g,
                                          const BVH &bvh,
                                          const BlockPairs &bp,
                                          const TpeAdaptiveParams &adaptive) {
    (void)g;
    TpeAdaptiveCache out;
    out.params = adaptive;
    if (!adaptive.enabled) {
        return out;
    }

    TpeAdaptiveParams params = adaptive;
    params.theta = std::max(0.0, params.theta);
    params.max_depth = std::max(0, params.max_depth);
    params.max_stack_items = std::max(16, params.max_stack_items);
    out.params = params;

    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const bool self = (cp.u == cp.v);
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t1 = bvh.face_indices[i];
            for (int j = V.face_start; j < V.face_end; ++j) {
                const int t2 = bvh.face_indices[j];
                if (self && t1 == t2) continue;
                append_adaptive_terms_for_face_pair(mesh, t1, t2, params, out.near_terms);
            }
        }
    }

    return out;
}

double tpe_energy_bh(const MeshData &mesh,
                     const FaceGeom &g,
                     const BVH &bvh,
                     const BlockPairs &bp,
                     const TpeAdaptiveParams &adaptive,
                     double alpha,
                     const TpeAdaptiveCache *cache) {
    if (!adaptive.enabled) {
        return tpe_energy_bh(g, bvh, bp, alpha);
    }

    TpeAdaptiveCache local_cache;
    const TpeAdaptiveCache *active_cache = cache;
    if (active_cache == nullptr) {
        local_cache = build_tpe_adaptive_cache(mesh, g, bvh, bp, adaptive);
        active_cache = &local_cache;
    }

    double phi = admissible_energy(g, bvh, bp, alpha);
    for (const TpeNearFieldTerm &term : active_cache->near_terms) {
        phi += adaptive_term_energy(mesh, g, term, alpha);
    }
    return phi;
}

double tpe_energy_bh(const MeshData &mesh, double alpha, double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    return tpe_energy_bh(g, bvh, bp, alpha);
}

double tpe_energy_bh(const MeshData &mesh,
                     const TpeAdaptiveParams &adaptive,
                     double alpha,
                     double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    const TpeAdaptiveCache cache =
        build_tpe_adaptive_cache(mesh, g, bvh, bp, adaptive);
    return tpe_energy_bh(mesh, g, bvh, bp, adaptive, alpha, &cache);
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const FaceGeom &g,
                                const BVH &bvh,
                                const BlockPairs &bp,
                                double alpha) {
    const int nv = mesh.n_vertices();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nv, 3);
    const auto E = build_opposite_edges(mesh);
    accumulate_admissible_gradient(mesh, g, bvh, bp, alpha, E, G);
    accumulate_nearfield_midpoint_gradient(mesh, g, bvh, bp, alpha, E, G);
    return G;
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const FaceGeom &g,
                                const BVH &bvh,
                                const BlockPairs &bp,
                                const TpeAdaptiveParams &adaptive,
                                double alpha,
                                const TpeAdaptiveCache *cache) {
    if (!adaptive.enabled) {
        return tpe_gradient_bh(mesh, g, bvh, bp, alpha);
    }

    TpeAdaptiveCache local_cache;
    const TpeAdaptiveCache *active_cache = cache;
    if (active_cache == nullptr) {
        local_cache = build_tpe_adaptive_cache(mesh, g, bvh, bp, adaptive);
        active_cache = &local_cache;
    }

    const int nv = mesh.n_vertices();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nv, 3);
    const auto E = build_opposite_edges(mesh);
    accumulate_admissible_gradient(mesh, g, bvh, bp, alpha, E, G);
    for (const TpeNearFieldTerm &term : active_cache->near_terms) {
        accumulate_adaptive_term_gradient(mesh, g, E, term, alpha, G);
    }
    return G;
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh, double alpha, double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    return tpe_gradient_bh(mesh, g, bvh, bp, alpha);
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const TpeAdaptiveParams &adaptive,
                                double alpha,
                                double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    const TpeAdaptiveCache cache =
        build_tpe_adaptive_cache(mesh, g, bvh, bp, adaptive);
    return tpe_gradient_bh(mesh, g, bvh, bp, adaptive, alpha, &cache);
}

} // namespace rsh
