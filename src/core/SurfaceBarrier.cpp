#include "SurfaceBarrier.h"

#include "FaceGeom.h"

#include <array>
#include <cmath>
#include <vector>

namespace rsh {

namespace {

using Vec3 = Eigen::Vector3d;

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

} // namespace rsh
