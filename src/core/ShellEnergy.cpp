#include "ShellEnergy.h"
#include "FaceGeom.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rsh {

namespace {

using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double, 3, 2>;

struct TriFrame {
    Mat32 J_ref;
    Mat32 J_def;
    Mat2 I_ref;
    Mat2 I_def;
    double area_ref = 0.0;
};

struct BendingEdge {
    int i = -1;
    int j = -1;
    int f0 = -1;
    int f1 = -1;
};

double safe_det(double det, double eps) {
    return std::max(det, eps);
}

double triangle_area(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

Vec3 triangle_normal_unit(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const Vec3 n = (b - a).cross(c - a);
    const double nn = n.norm();
    if (nn <= 0.0 || !std::isfinite(nn)) {
        return Vec3::UnitZ();
    }
    return n / nn;
}

Mat2 membrane_density_derivative(const Mat2 &A,
                                 const ShellEnergyParams &params) {
    const double mu = params.mu;
    const double lambda = params.lambda;
    const double eps = params.det_smoothing;

    const double detA_raw = A.determinant();
    if (!(detA_raw > eps)) {
        return 0.5 * mu * Mat2::Identity();
    }

    const Mat2 A_inv_t = A.inverse().transpose();
    return 0.5 * mu * Mat2::Identity() +
           0.25 * (lambda * detA_raw - (2.0 * mu + lambda)) * A_inv_t;
}

double membrane_density(const Mat2 &A, const ShellEnergyParams &params) {
    const double mu = params.mu;
    const double lambda = params.lambda;
    const double eps = params.det_smoothing;

    const double trA = A.trace();
    const double detA = safe_det(A.determinant(), eps);
    return 0.5 * mu * trA +
           0.25 * lambda * detA -
           0.25 * (2.0 * mu + lambda) * std::log(detA) -
           mu - 0.25 * lambda;
}

TriFrame build_triangle_frame(const MeshData &x_ref,
                              const MeshData &x_def,
                              int f) {
    const int i = x_ref.F(f, 0);
    const int j = x_ref.F(f, 1);
    const int k = x_ref.F(f, 2);

    const Vec3 xr0 = x_ref.V.row(i);
    const Vec3 xr1 = x_ref.V.row(j);
    const Vec3 xr2 = x_ref.V.row(k);
    const Vec3 xd0 = x_def.V.row(i);
    const Vec3 xd1 = x_def.V.row(j);
    const Vec3 xd2 = x_def.V.row(k);

    TriFrame out;
    out.J_ref.col(0) = xr1 - xr0;
    out.J_ref.col(1) = xr2 - xr0;
    out.J_def.col(0) = xd1 - xd0;
    out.J_def.col(1) = xd2 - xd0;
    out.I_ref = out.J_ref.transpose() * out.J_ref;
    out.I_def = out.J_def.transpose() * out.J_def;
    out.area_ref = triangle_area(xr0, xr1, xr2);
    return out;
}

std::vector<TriFrame> build_frames(const MeshData &x_ref, const MeshData &x_def) {
    const int nf = x_ref.n_faces();
    std::vector<TriFrame> frames(static_cast<size_t>(nf));
    for (int f = 0; f < nf; ++f) {
        frames[static_cast<size_t>(f)] = build_triangle_frame(x_ref, x_def, f);
    }
    return frames;
}

std::vector<BendingEdge> build_interior_edges(const MeshData &mesh) {
    struct HalfEdgeInfo {
        int f = -1;
    };

    std::unordered_map<long long, HalfEdgeInfo> edge_face;
    edge_face.reserve(static_cast<size_t>(mesh.n_faces() * 3));
    std::vector<BendingEdge> out;

    auto key = [](int a, int b) -> long long {
        if (a > b) std::swap(a, b);
        return (static_cast<long long>(a) << 32) ^
               static_cast<unsigned long long>(b);
    };

    for (int f = 0; f < mesh.n_faces(); ++f) {
        for (int e = 0; e < 3; ++e) {
            const int a = mesh.F(f, e);
            const int b = mesh.F(f, (e + 1) % 3);
            const long long k = key(a, b);
            auto it = edge_face.find(k);
            if (it == edge_face.end()) {
                edge_face.emplace(k, HalfEdgeInfo{f});
            } else {
                BendingEdge be;
                be.i = std::min(a, b);
                be.j = std::max(a, b);
                be.f0 = it->second.f;
                be.f1 = f;
                out.push_back(be);
                edge_face.erase(it);
            }
        }
    }

    return out;
}

double edge_length(const MeshData &mesh, int i, int j) {
    const Vec3 vi = mesh.V.row(i);
    const Vec3 vj = mesh.V.row(j);
    return (vi - vj).norm();
}

double face_area(const MeshData &mesh, int f) {
    const int i = mesh.F(f, 0);
    const int j = mesh.F(f, 1);
    const int k = mesh.F(f, 2);
    const Vec3 a = mesh.V.row(i);
    const Vec3 b = mesh.V.row(j);
    const Vec3 c = mesh.V.row(k);
    return triangle_area(a, b, c);
}

double dihedral_angle(const MeshData &mesh, int f0, int f1, int i, int j) {
    (void)i;
    (void)j;
    const Vec3 n0 = triangle_normal_unit(
        mesh.V.row(mesh.F(f0, 0)), mesh.V.row(mesh.F(f0, 1)), mesh.V.row(mesh.F(f0, 2)));
    const Vec3 n1 = triangle_normal_unit(
        mesh.V.row(mesh.F(f1, 0)), mesh.V.row(mesh.F(f1, 1)), mesh.V.row(mesh.F(f1, 2)));

    const double c = std::max(-1.0, std::min(1.0, n0.dot(n1)));
    return std::acos(c);
}

double bending_energy_only(const MeshData &x_ref,
                           const MeshData &x_def,
                           const ShellEnergyParams &params,
                           const std::vector<BendingEdge> &edges) {
    double wb = 0.0;
    for (const BendingEdge &e : edges) {
        const double theta_ref = dihedral_angle(x_ref, e.f0, e.f1, e.i, e.j);
        const double theta_def = dihedral_angle(x_def, e.f0, e.f1, e.i, e.j);

        const double area_e =
            (face_area(x_ref, e.f0) + face_area(x_ref, e.f1)) / 3.0;
        if (!(area_e > 0.0)) continue;
        const double ell = edge_length(x_ref, e.i, e.j);

        double val_ref = theta_ref;
        double val_def = theta_def;
        if (params.use_tan_bending) {
            const double clamp_hi = M_PI - params.angle_clamp_eps;
            val_ref = 2.0 * std::tan(0.5 * std::min(theta_ref, clamp_hi));
            val_def = 2.0 * std::tan(0.5 * std::min(theta_def, clamp_hi));
        }
        const double diff = val_def - val_ref;
        wb += std::pow(params.thickness, 3.0) * diff * diff * ell * ell / area_e;
    }
    return wb;
}

void accumulate_membrane_gradient(const MeshData &x_ref,
                                  const MeshData &x_def,
                                  const std::vector<TriFrame> &frames,
                                  const ShellEnergyParams &params,
                                  Eigen::MatrixXd &grad_ref,
                                  Eigen::MatrixXd &grad_def) {
    const int nf = x_ref.n_faces();
    for (int f = 0; f < nf; ++f) {
        const TriFrame &tf = frames[static_cast<size_t>(f)];
        if (!(tf.area_ref > 0.0)) continue;

        const Mat2 A = tf.I_ref.inverse() * tf.I_def;
        const Mat2 dW_dA = membrane_density_derivative(A, params);
        const Mat2 dW_dIdef_raw = tf.I_ref.inverse().transpose() * dW_dA;
        const Mat2 dW_dIdef =
            0.5 * (dW_dIdef_raw + dW_dIdef_raw.transpose());
        const Mat32 dW_dJdef = 2.0 * tf.J_def * dW_dIdef;

        const Mat2 dW_dIref_raw =
            -tf.I_ref.inverse().transpose() * dW_dA * A.transpose();
        const Mat2 dW_dIref =
            0.5 * (dW_dIref_raw + dW_dIref_raw.transpose());
        const Mat32 dW_dJref = 2.0 * tf.J_ref * dW_dIref;

        const double scale = params.thickness * tf.area_ref;
        const int i = x_ref.F(f, 0);
        const int j = x_ref.F(f, 1);
        const int k = x_ref.F(f, 2);

        const Vec3 gdef_j = scale * dW_dJdef.col(0);
        const Vec3 gdef_k = scale * dW_dJdef.col(1);
        const Vec3 gdef_i = -gdef_j - gdef_k;

        grad_def.row(i) += gdef_i.transpose();
        grad_def.row(j) += gdef_j.transpose();
        grad_def.row(k) += gdef_k.transpose();

        const Vec3 gref_j = scale * dW_dJref.col(0);
        const Vec3 gref_k = scale * dW_dJref.col(1);
        const Vec3 gref_i = -gref_j - gref_k;

        grad_ref.row(i) += gref_i.transpose();
        grad_ref.row(j) += gref_j.transpose();
        grad_ref.row(k) += gref_k.transpose();

        const Vec3 vr0 = x_ref.V.row(i);
        const Vec3 vr1 = x_ref.V.row(j);
        const Vec3 vr2 = x_ref.V.row(k);
        const Vec3 nr = triangle_normal_unit(vr0, vr1, vr2);
        Vec3 E0, E1, E2;
        opposite_edges(vr0, vr1, vr2, E0, E1, E2);
        const double w = membrane_density(A, params);
        grad_ref.row(i) += (params.thickness * w) * da_dvk(nr, E0);
        grad_ref.row(j) += (params.thickness * w) * da_dvk(nr, E1);
        grad_ref.row(k) += (params.thickness * w) * da_dvk(nr, E2);
    }
}

void accumulate_bending_gradient_fd(const MeshData &x_ref,
                                    const MeshData &x_def,
                                    const ShellEnergyParams &params,
                                    const std::vector<BendingEdge> &edges,
                                    Eigen::MatrixXd &grad_ref,
                                    Eigen::MatrixXd &grad_def) {
    const double eps = params.bending_fd_eps;
    if (!(eps > 0.0)) return;

    auto wb_ref_def = [&](const MeshData &xr, const MeshData &xd) -> double {
        return bending_energy_only(xr, xd, params, edges);
    };

    MeshData xr = x_ref;
    MeshData xd = x_def;

    for (int v = 0; v < x_ref.n_vertices(); ++v) {
        for (int c = 0; c < 3; ++c) {
            const double h = eps * (1.0 + std::abs(x_ref.V(v, c)));

            xr.V(v, c) += h;
            const double ep = wb_ref_def(xr, xd);
            xr.V(v, c) -= 2.0 * h;
            const double em = wb_ref_def(xr, xd);
            xr.V(v, c) += h;
            grad_ref(v, c) += (ep - em) / (2.0 * h);
        }
    }

    for (int v = 0; v < x_def.n_vertices(); ++v) {
        for (int c = 0; c < 3; ++c) {
            const double h = eps * (1.0 + std::abs(x_def.V(v, c)));

            xd.V(v, c) += h;
            const double ep = wb_ref_def(xr, xd);
            xd.V(v, c) -= 2.0 * h;
            const double em = wb_ref_def(xr, xd);
            xd.V(v, c) += h;
            grad_def(v, c) += (ep - em) / (2.0 * h);
        }
    }
}

void validate_mesh_pair(const MeshData &x_ref, const MeshData &x_def) {
    if (x_ref.n_vertices() != x_def.n_vertices() ||
        x_ref.n_faces() != x_def.n_faces()) {
        throw std::runtime_error("shell_energy: meshes must have identical topology");
    }
    if ((x_ref.F.rows() != x_def.F.rows()) || ((x_ref.F - x_def.F).array() != 0).any()) {
        throw std::runtime_error("shell_energy: face connectivity must match exactly");
    }
}

} // namespace

ShellEnergyValue shell_energy(const MeshData &x_ref,
                              const MeshData &x_def,
                              const ShellEnergyParams &params) {
    validate_mesh_pair(x_ref, x_def);

    ShellEnergyValue out;
    const std::vector<TriFrame> frames = build_frames(x_ref, x_def);
    for (const TriFrame &tf : frames) {
        if (!(tf.area_ref > 0.0)) continue;
        const Mat2 A = tf.I_ref.inverse() * tf.I_def;
        out.membrane += params.thickness * tf.area_ref * membrane_density(A, params);
    }

    const std::vector<BendingEdge> edges = build_interior_edges(x_ref);
    out.bending = bending_energy_only(x_ref, x_def, params, edges);
    out.total = out.membrane + out.bending;
    return out;
}

ShellEnergyGradientResult shell_energy_with_gradient(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params) {
    validate_mesh_pair(x_ref, x_def);

    ShellEnergyGradientResult out;
    out.grad_ref = Eigen::MatrixXd::Zero(x_ref.n_vertices(), 3);
    out.grad_def = Eigen::MatrixXd::Zero(x_def.n_vertices(), 3);

    const std::vector<TriFrame> frames = build_frames(x_ref, x_def);
    for (const TriFrame &tf : frames) {
        if (!(tf.area_ref > 0.0)) continue;
        const Mat2 A = tf.I_ref.inverse() * tf.I_def;
        out.energy.membrane +=
            params.thickness * tf.area_ref * membrane_density(A, params);
    }
    accumulate_membrane_gradient(
        x_ref, x_def, frames, params, out.grad_ref, out.grad_def);

    const std::vector<BendingEdge> edges = build_interior_edges(x_ref);
    out.energy.bending = bending_energy_only(x_ref, x_def, params, edges);
    accumulate_bending_gradient_fd(
        x_ref, x_def, params, edges, out.grad_ref, out.grad_def);

    out.energy.total = out.energy.membrane + out.energy.bending;
    return out;
}

} // namespace rsh
