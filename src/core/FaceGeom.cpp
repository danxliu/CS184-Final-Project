#include "FaceGeom.h"

namespace rsh {

FaceGeom compute_face_geom(const MeshData &mesh) {
    const int nf = static_cast<int>(mesh.F.rows());
    FaceGeom g;
    g.C.resize(nf, 3);
    g.N.resize(nf, 3);
    g.A.resize(nf);

    for (int t = 0; t < nf; ++t) {
        const Eigen::Vector3d v0 = mesh.V.row(mesh.F(t, 0));
        const Eigen::Vector3d v1 = mesh.V.row(mesh.F(t, 1));
        const Eigen::Vector3d v2 = mesh.V.row(mesh.F(t, 2));

        const Eigen::Vector3d cross = (v1 - v0).cross(v2 - v0);
        const double cross_norm = cross.norm();

        g.C.row(t) = ((v0 + v1 + v2) / 3.0).transpose();
        g.A(t) = 0.5 * cross_norm;
        g.N.row(t) = (cross / cross_norm).transpose();
    }
    return g;
}

void opposite_edges(const Eigen::Vector3d &v0,
                    const Eigen::Vector3d &v1,
                    const Eigen::Vector3d &v2,
                    Eigen::Vector3d &E0,
                    Eigen::Vector3d &E1,
                    Eigen::Vector3d &E2) {
    E0 = v2 - v1;
    E1 = v0 - v2;
    E2 = v1 - v0;
}

Eigen::Matrix3d skew(const Eigen::Vector3d &a) {
    Eigen::Matrix3d S;
    S <<    0.0, -a.z(),  a.y(),
          a.z(),    0.0, -a.x(),
         -a.y(),  a.x(),    0.0;
    return S;
}

Eigen::Matrix3d dc_dvk() {
    return (1.0 / 3.0) * Eigen::Matrix3d::Identity();
}

Eigen::RowVector3d da_dvk(const Eigen::Vector3d &n,
                          const Eigen::Vector3d &Ek) {
    return 0.5 * n.cross(Ek).transpose();
}

Eigen::Matrix3d dn_dvk(const Eigen::Vector3d &n,
                       double area,
                       const Eigen::Vector3d &Ek) {
    const Eigen::Matrix3d P =
        Eigen::Matrix3d::Identity() - n * n.transpose();
    return (1.0 / (2.0 * area)) * P * skew(Ek);
}

} // namespace rsh
