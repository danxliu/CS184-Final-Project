#include "TPE.h"
#include "FaceGeom.h"

#include <array>
#include <cmath>
#include <vector>

namespace rsh {

double tpe_energy_brute(const FaceGeom &g, double alpha) {
    const int nf = static_cast<int>(g.A.size());
    double phi = 0.0;
    for (int t1 = 0; t1 < nf; ++t1) {
        const Eigen::Vector3d c1 = g.C.row(t1);
        const Eigen::Vector3d n1 = g.N.row(t1);
        const double a1 = g.A(t1);
        for (int t2 = 0; t2 < nf; ++t2) {
            if (t2 == t1) continue;
            const Eigen::Vector3d c2 = g.C.row(t2);
            const double a2 = g.A(t2);
            const Eigen::Vector3d chord = c1 - c2;
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

    // Cache per-triangle opposite edges (reused across all t2 partners).
    std::vector<std::array<Eigen::Vector3d, 3>> E(nf);
    for (int t = 0; t < nf; ++t) {
        const Eigen::Vector3d v0 = mesh.V.row(mesh.F(t, 0));
        const Eigen::Vector3d v1 = mesh.V.row(mesh.F(t, 1));
        const Eigen::Vector3d v2 = mesh.V.row(mesh.F(t, 2));
        opposite_edges(v0, v1, v2, E[t][0], E[t][1], E[t][2]);
    }

    for (int t1 = 0; t1 < nf; ++t1) {
        const Eigen::Vector3d c1 = g.C.row(t1);
        const Eigen::Vector3d n1 = g.N.row(t1);
        const double a1 = g.A(t1);

        // Jacobians that depend only on t1 — hoisted out of the t2 inner loop.
        Eigen::Matrix3d Jn1[3];
        Eigen::RowVector3d Ja1[3];
        for (int k = 0; k < 3; ++k) {
            Jn1[k] = dn_dvk(n1, a1, E[t1][k]);
            Ja1[k] = da_dvk(n1, E[t1][k]);
        }

        for (int t2 = 0; t2 < nf; ++t2) {
            if (t2 == t1) continue;
            const Eigen::Vector3d c2 = g.C.row(t2);
            const Eigen::Vector3d n2 = g.N.row(t2);
            const double a2 = g.A(t2);

            const Eigen::Vector3d d = c1 - c2;
            const double r2 = d.squaredNorm();
            if (r2 == 0.0) continue;

            const double s = n1.dot(d);
            const double s2 = s * s;
            const double inv_r2alpha = std::pow(r2, -alpha);
            const double s_abs_alpha = std::pow(s2, alpha * 0.5);
            // alpha * |s|^{alpha-1} * sign(s) = alpha * s * |s|^{alpha-2}.
            // Well-defined at s=0 for alpha >= 2 (limit is 0).
            const double s_signed_power =
                alpha * s * std::pow(s2, (alpha - 2.0) * 0.5);

            // Per-pair derivatives w.r.t. intermediates (a1, a2, c1, c2, n1).
            // The kernel involves only n1, so n2 has no partial (its turn
            // comes when the swapped pair (t2, t1) is visited).
            const double K_factor = s_abs_alpha * inv_r2alpha;
            const double dK_da1   = a2 * K_factor;
            const double dK_da2   = a1 * K_factor;
            const double coef_n   = a1 * a2 * s_signed_power * inv_r2alpha;
            const double coef_d   = 2.0 * alpha * a1 * a2 * K_factor / r2;

            const Eigen::RowVector3d dK_dc1 =
                (coef_n * n1 - coef_d * d).transpose();
            const Eigen::RowVector3d dK_dn1 = (coef_n * d).transpose();

            // dc/dv_k = (1/3) I for every corner k of either triangle.
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
    double phi = 0.0;

    // Admissible cluster pairs — one kernel evaluation per pair using the
    // area-weighted centroid / normal aggregates. n_U / a_U is the mean
    // unit normal direction of the cluster (has norm <= 1 with equality
    // iff all face normals are parallel).
    for (const ClusterPair &cp : bp.admissible) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const Eigen::Vector3d d = U.centroid - V.centroid;
        const double r2 = d.squaredNorm();
        const Eigen::Vector3d nU_mean = U.normal_sum / U.area;
        const double s = nU_mean.dot(d);
        const double num = std::pow(std::abs(s), alpha);
        const double den = std::pow(r2, alpha);
        phi += U.area * V.area * num / den;
    }

    // Near-field leaf-leaf pairs — exact per-face kernel, same as brute force.
    // Self-pairs (u == v) skip the t1 == t2 diagonal.
    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const bool self = (cp.u == cp.v);
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t1 = bvh.face_indices[i];
            const Eigen::Vector3d c1 = g.C.row(t1);
            const Eigen::Vector3d n1 = g.N.row(t1);
            const double a1 = g.A(t1);
            for (int j = V.face_start; j < V.face_end; ++j) {
                const int t2 = bvh.face_indices[j];
                if (self && t1 == t2) continue;
                const Eigen::Vector3d c2 = g.C.row(t2);
                const double a2 = g.A(t2);
                const Eigen::Vector3d chord = c1 - c2;
                const double num = std::pow(std::abs(n1.dot(chord)), alpha);
                const double den = std::pow(chord.squaredNorm(), alpha);
                phi += a1 * a2 * num / den;
            }
        }
    }

    return phi;
}

double tpe_energy_bh(const MeshData &mesh, double alpha, double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    return tpe_energy_bh(g, bvh, bp, alpha);
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh,
                                const FaceGeom &g,
                                const BVH &bvh,
                                const BlockPairs &bp,
                                double alpha) {
    const int nf = static_cast<int>(g.A.size());
    const int nv = mesh.n_vertices();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nv, 3);

    // Cache per-face opposite-edge vectors once (used by both admissible
    // scatter and near-field per-pair gradient).
    std::vector<std::array<Eigen::Vector3d, 3>> E(nf);
    for (int t = 0; t < nf; ++t) {
        const Eigen::Vector3d v0 = mesh.V.row(mesh.F(t, 0));
        const Eigen::Vector3d v1 = mesh.V.row(mesh.F(t, 1));
        const Eigen::Vector3d v2 = mesh.V.row(mesh.F(t, 2));
        opposite_edges(v0, v1, v2, E[t][0], E[t][1], E[t][2]);
    }

    // --- Admissible pair pass ----------------------------------------------
    // Per-face accumulators for dE/da_t, dE/dc_t, dE/dn_t contributed by
    // admissible pairs. Faces are chain-ruled to vertex positions once at the
    // end so a face appearing in K admissible pairs pays only one Jacobian
    // apply, not K.
    Eigen::VectorXd dE_da_face = Eigen::VectorXd::Zero(nf);
    Eigen::MatrixXd dE_dc_face = Eigen::MatrixXd::Zero(nf, 3);
    Eigen::MatrixXd dE_dn_face = Eigen::MatrixXd::Zero(nf, 3);

    for (const ClusterPair &cp : bp.admissible) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const double aU = U.area;
        const double aV = V.area;
        const Eigen::Vector3d cU = U.centroid;
        const Eigen::Vector3d cV = V.centroid;
        const Eigen::Vector3d SU = U.normal_sum;       // = sum_{t in U} a_t n_t
        const Eigen::Vector3d mU = SU / aU;            // area-weighted mean unit normal
        const Eigen::Vector3d d = cU - cV;
        const double r2 = d.squaredNorm();
        if (r2 == 0.0) continue;

        const double s = mU.dot(d);
        const double s2 = s * s;
        const double inv_r2alpha = std::pow(r2, -alpha);     // phi = r^(-2 alpha)
        const double psi = std::pow(s2, alpha * 0.5);        // |s|^alpha
        // s * |s|^(alpha - 2) = s * (s^2)^((alpha - 2)/2).  Well-defined at
        // s = 0 for alpha >= 2 (limit is 0).
        const double s_signed_pow = s * std::pow(s2, (alpha - 2.0) * 0.5);

        // Kernel K(U, V) = a_U a_V psi phi with s = (S_U / a_U) . d. Treat
        // (a_U, S_U, c_U) and (a_V, c_V) as the independent aggregates (S_V
        // doesn't enter K since only the near side contributes a normal).
        //
        //   dK/da_U = (1 - alpha) a_V psi phi
        //   dK/da_V = a_U psi phi
        //   dK/dS_U = a_V alpha (s |s|^(a-2)) d phi
        //   dK/dc_U = a_U a_V alpha phi (s |s|^(a-2) m_U - 2 psi d / r^2)
        //   dK/dc_V = -dK/dc_U
        const double dK_daU = (1.0 - alpha) * aV * psi * inv_r2alpha;
        const double dK_daV = aU * psi * inv_r2alpha;
        const Eigen::Vector3d dK_dSU =
            aV * alpha * s_signed_pow * inv_r2alpha * d;
        const Eigen::Vector3d dK_dcU =
            aU * aV * alpha * inv_r2alpha *
            (s_signed_pow * mU - 2.0 * psi * d / r2);
        const Eigen::Vector3d dK_dcV = -dK_dcU;

        // Scatter U-side: a_U = sum a_t, S_U = sum a_t n_t, a_U c_U = sum a_t c_t.
        //   da_U/da_t = 1,      dS_U/da_t = n_t,      dc_U/da_t = (c_t - c_U)/a_U
        //   da_U/dn_t = 0,      dS_U/dn_t = a_t I,    dc_U/dn_t = 0
        //   da_U/dc_t = 0,      dS_U/dc_t = 0,        dc_U/dc_t = (a_t/a_U) I
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t = bvh.face_indices[i];
            const double a_t = g.A(t);
            const Eigen::Vector3d c_t = g.C.row(t);
            const Eigen::Vector3d n_t = g.N.row(t);
            dE_da_face(t) +=
                dK_daU + dK_dSU.dot(n_t) + dK_dcU.dot(c_t - cU) / aU;
            dE_dc_face.row(t) += (a_t / aU) * dK_dcU.transpose();
            dE_dn_face.row(t) += a_t * dK_dSU.transpose();
        }
        // Scatter V-side: kernel doesn't depend on n_V, so only (a_V, c_V)
        // chain-rule back. (n_V's contribution is picked up when (V, U) is
        // processed as its own admissible pair.)
        for (int j = V.face_start; j < V.face_end; ++j) {
            const int t = bvh.face_indices[j];
            const double a_t = g.A(t);
            const Eigen::Vector3d c_t = g.C.row(t);
            dE_da_face(t) += dK_daV + dK_dcV.dot(c_t - cV) / aV;
            dE_dc_face.row(t) += (a_t / aV) * dK_dcV.transpose();
        }
    }

    // Collapse the per-face admissible accumulators into vertex gradients.
    // dc_t/dv_k = (1/3) I, so the c-contribution is just dE_dc_face.row(t)/3
    // for every corner k.
    for (int t = 0; t < nf; ++t) {
        if (dE_da_face(t) == 0.0 && dE_dc_face.row(t).isZero() &&
            dE_dn_face.row(t).isZero()) {
            continue;
        }
        const double a_t = g.A(t);
        const Eigen::Vector3d n_t = g.N.row(t);
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

    // --- Near-field pair pass ----------------------------------------------
    // Exact per-face pair gradient, same inner kernel as tpe_gradient_brute.
    // Self-pairs (u == v) skip the t1 == t2 diagonal.
    for (const ClusterPair &cp : bp.near_field) {
        const BVHNode &U = bvh.nodes[cp.u];
        const BVHNode &V = bvh.nodes[cp.v];
        const bool self = (cp.u == cp.v);
        for (int i = U.face_start; i < U.face_end; ++i) {
            const int t1 = bvh.face_indices[i];
            const Eigen::Vector3d c1 = g.C.row(t1);
            const Eigen::Vector3d n1 = g.N.row(t1);
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
                const Eigen::Vector3d c2 = g.C.row(t2);
                const Eigen::Vector3d n2 = g.N.row(t2);
                const double a2 = g.A(t2);

                const Eigen::Vector3d d = c1 - c2;
                const double r2 = d.squaredNorm();
                if (r2 == 0.0) continue;

                const double s = n1.dot(d);
                const double s2 = s * s;
                const double inv_r2alpha = std::pow(r2, -alpha);
                const double s_abs_alpha = std::pow(s2, alpha * 0.5);
                const double s_signed_power =
                    alpha * s * std::pow(s2, (alpha - 2.0) * 0.5);

                const double K_factor = s_abs_alpha * inv_r2alpha;
                const double dK_da1   = a2 * K_factor;
                const double dK_da2   = a1 * K_factor;
                const double coef_n   = a1 * a2 * s_signed_power * inv_r2alpha;
                const double coef_d   = 2.0 * alpha * a1 * a2 * K_factor / r2;

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

    return G;
}

Eigen::MatrixXd tpe_gradient_bh(const MeshData &mesh, double alpha, double theta) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, theta);
    return tpe_gradient_bh(mesh, g, bvh, bp, alpha);
}

} // namespace rsh
