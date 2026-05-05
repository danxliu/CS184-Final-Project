#include "ShellEnergy.h"
#include "FaceGeom.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

namespace rsh {

namespace {

using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double, 3, 2>;

struct AD2 {
    double val = 0.0;
    Eigen::VectorXd grad;
    Eigen::MatrixXd hess;

    AD2() = default;
    AD2(double v, int dim)
        : val(v),
          grad(Eigen::VectorXd::Zero(dim)),
          hess(Eigen::MatrixXd::Zero(dim, dim)) {}

    int dim() const { return static_cast<int>(grad.size()); }
};

AD2 ad_constant(double v, int dim) {
    return AD2(v, dim);
}

AD2 ad_variable(double v, int dim, int idx) {
    AD2 out(v, dim);
    out.grad(idx) = 1.0;
    return out;
}

AD2 operator+(const AD2 &a, const AD2 &b) {
    AD2 out;
    out.val = a.val + b.val;
    out.grad = a.grad + b.grad;
    out.hess = a.hess + b.hess;
    return out;
}

AD2 operator+(const AD2 &a, double b) {
    AD2 out = a;
    out.val += b;
    return out;
}

AD2 operator+(double a, const AD2 &b) { return b + a; }

AD2 operator-(const AD2 &a, const AD2 &b) {
    AD2 out;
    out.val = a.val - b.val;
    out.grad = a.grad - b.grad;
    out.hess = a.hess - b.hess;
    return out;
}

AD2 operator-(const AD2 &a, double b) {
    AD2 out = a;
    out.val -= b;
    return out;
}

AD2 operator-(double a, const AD2 &b) {
    AD2 out;
    out.val = a - b.val;
    out.grad = -b.grad;
    out.hess = -b.hess;
    return out;
}

AD2 operator-(const AD2 &a) {
    AD2 out;
    out.val = -a.val;
    out.grad = -a.grad;
    out.hess = -a.hess;
    return out;
}

AD2 operator*(const AD2 &a, const AD2 &b) {
    AD2 out;
    out.val = a.val * b.val;
    out.grad = b.val * a.grad + a.val * b.grad;
    out.hess = b.val * a.hess + a.val * b.hess +
               a.grad * b.grad.transpose() +
               b.grad * a.grad.transpose();
    return out;
}

AD2 operator*(const AD2 &a, double b) {
    AD2 out;
    out.val = a.val * b;
    out.grad = a.grad * b;
    out.hess = a.hess * b;
    return out;
}

AD2 operator*(double a, const AD2 &b) { return b * a; }

AD2 compose_unary(const AD2 &a, double val, double d1, double d2) {
    AD2 out;
    out.val = val;
    out.grad = d1 * a.grad;
    out.hess = d1 * a.hess + d2 * (a.grad * a.grad.transpose());
    return out;
}

AD2 ad_inv(const AD2 &a) {
    const double inv = 1.0 / a.val;
    return compose_unary(a, inv, -inv * inv, 2.0 * inv * inv * inv);
}

AD2 operator/(const AD2 &a, const AD2 &b) { return a * ad_inv(b); }
AD2 operator/(const AD2 &a, double b) { return a * (1.0 / b); }
AD2 operator/(double a, const AD2 &b) { return a * ad_inv(b); }

AD2 ad_sqrt(const AD2 &a) {
    const double s = std::sqrt(std::max(0.0, a.val));
    if (!(s > 0.0)) {
        return ad_constant(0.0, a.dim());
    }
    return compose_unary(a, s, 0.5 / s, -0.25 / (s * s * s));
}

AD2 ad_log(const AD2 &a) {
    return compose_unary(a, std::log(a.val), 1.0 / a.val,
                         -1.0 / (a.val * a.val));
}

AD2 ad_clamped_det(const AD2 &a, double eps) {
    if (a.val > eps) {
        return a;
    }
    return ad_constant(eps, a.dim());
}

struct ADVec3 {
    AD2 x;
    AD2 y;
    AD2 z;
};

ADVec3 operator+(const ADVec3 &a, const ADVec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

ADVec3 operator-(const ADVec3 &a, const ADVec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

ADVec3 operator*(const ADVec3 &a, double b) {
    return {a.x * b, a.y * b, a.z * b};
}

ADVec3 operator/(const ADVec3 &a, const AD2 &b) {
    return {a.x / b, a.y / b, a.z / b};
}

AD2 dot_ad(const ADVec3 &a, const ADVec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

ADVec3 cross_ad(const ADVec3 &a, const ADVec3 &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

AD2 norm_ad(const ADVec3 &a) {
    return ad_sqrt(dot_ad(a, a));
}

ADVec3 make_ad_vec3(const MeshData &mesh,
                    int vertex,
                    int dim,
                    int base_idx) {
    return {
        ad_variable(mesh.V(vertex, 0), dim, base_idx),
        ad_variable(mesh.V(vertex, 1), dim, base_idx + 1),
        ad_variable(mesh.V(vertex, 2), dim, base_idx + 2)};
}

struct ADMat2 {
    AD2 a00;
    AD2 a01;
    AD2 a10;
    AD2 a11;
};

ADMat2 mat2_mul(const ADMat2 &A, const ADMat2 &B) {
    return {
        A.a00 * B.a00 + A.a01 * B.a10,
        A.a00 * B.a01 + A.a01 * B.a11,
        A.a10 * B.a00 + A.a11 * B.a10,
        A.a10 * B.a01 + A.a11 * B.a11};
}

AD2 mat2_det(const ADMat2 &A) {
    return A.a00 * A.a11 - A.a01 * A.a10;
}

AD2 mat2_trace(const ADMat2 &A) {
    return A.a00 + A.a11;
}

ADMat2 mat2_inverse(const ADMat2 &A) {
    const AD2 inv_det = ad_inv(mat2_det(A));
    return { A.a11 * inv_det, -A.a01 * inv_det,
            -A.a10 * inv_det,  A.a00 * inv_det };
}

struct LocalDof {
    bool ref = true;
    int vertex = -1;
    int coord = 0;
};

struct BendingEdge {
    int i = -1;
    int j = -1;
    int f0 = -1;
    int f1 = -1;
};

int opposite_vertex(const MeshData &mesh, int f, int i, int j);
void scatter_local_gradient(const std::vector<LocalDof> &dofs,
                            const Eigen::VectorXd &grad,
                            Eigen::MatrixXd &grad_ref,
                            Eigen::MatrixXd &grad_def);
void scatter_local_hessian(
    const std::vector<LocalDof> &dofs,
    const Eigen::MatrixXd &hess,
    std::vector<Eigen::Triplet<double>> &rr,
    std::vector<Eigen::Triplet<double>> &rd,
    std::vector<Eigen::Triplet<double>> &dr,
    std::vector<Eigen::Triplet<double>> &dd,
    double drop_tol);

struct ScalarDerivatives {
    double value = 0.0;
    Eigen::VectorXd grad;
    Eigen::MatrixXd hess;
};

std::array<Vec3, 4> local_positions(const MeshData &mesh,
                                    const std::array<int, 4> &verts) {
    return {mesh.V.row(verts[0]), mesh.V.row(verts[1]),
            mesh.V.row(verts[2]), mesh.V.row(verts[3])};
}

int local_index_of(const std::array<int, 4> &verts, int v) {
    for (int i = 0; i < 4; ++i) {
        if (verts[static_cast<size_t>(i)] == v) {
            return i;
        }
    }
    throw std::runtime_error("local hinge vertex lookup failed");
}

std::array<int, 3> local_face_indices(const MeshData &mesh,
                                      int f,
                                      const std::array<int, 4> &verts) {
    return {local_index_of(verts, mesh.F(f, 0)),
            local_index_of(verts, mesh.F(f, 1)),
            local_index_of(verts, mesh.F(f, 2))};
}

Vec3 basis_delta(int local_vertex, int coord, int target_vertex) {
    if (local_vertex != target_vertex) {
        return Vec3::Zero();
    }
    Vec3 out = Vec3::Zero();
    out(coord) = 1.0;
    return out;
}

void raw_normal_derivatives(const std::array<Vec3, 4> &pos,
                            const std::array<int, 3> &face,
                            Vec3 &normal,
                            std::array<Vec3, 12> &dn,
                            std::array<std::array<Vec3, 12>, 12> &d2n) {
    const Vec3 e1 = pos[static_cast<size_t>(face[1])] -
                    pos[static_cast<size_t>(face[0])];
    const Vec3 e2 = pos[static_cast<size_t>(face[2])] -
                    pos[static_cast<size_t>(face[0])];
    normal = e1.cross(e2);

    std::array<Vec3, 12> de1;
    std::array<Vec3, 12> de2;
    for (int p = 0; p < 12; ++p) {
        const int lv = p / 3;
        const int coord = p % 3;
        de1[static_cast<size_t>(p)] =
            basis_delta(lv, coord, face[1]) -
            basis_delta(lv, coord, face[0]);
        de2[static_cast<size_t>(p)] =
            basis_delta(lv, coord, face[2]) -
            basis_delta(lv, coord, face[0]);
        dn[static_cast<size_t>(p)] =
            de1[static_cast<size_t>(p)].cross(e2) +
            e1.cross(de2[static_cast<size_t>(p)]);
    }

    for (int p = 0; p < 12; ++p) {
        for (int q = 0; q < 12; ++q) {
            d2n[static_cast<size_t>(p)][static_cast<size_t>(q)] =
                de1[static_cast<size_t>(p)].cross(
                    de2[static_cast<size_t>(q)]) +
                de1[static_cast<size_t>(q)].cross(
                    de2[static_cast<size_t>(p)]);
        }
    }
}

ScalarDerivatives norm_derivatives(
    const Vec3 &x,
    const std::array<Vec3, 12> &dx,
    const std::array<std::array<Vec3, 12>, 12> &d2x,
    double scale = 1.0) {
    ScalarDerivatives out;
    out.grad = Eigen::VectorXd::Zero(12);
    out.hess = Eigen::MatrixXd::Zero(12, 12);
    const double len = x.norm();
    if (!(len > 0.0) || !std::isfinite(len)) {
        return out;
    }

    out.value = scale * len;
    const Vec3 unit = x / len;
    for (int p = 0; p < 12; ++p) {
        out.grad(p) = scale * unit.dot(dx[static_cast<size_t>(p)]);
    }
    const Eigen::Matrix3d P =
        Eigen::Matrix3d::Identity() - unit * unit.transpose();
    for (int p = 0; p < 12; ++p) {
        for (int q = 0; q < 12; ++q) {
            out.hess(p, q) = scale *
                (dx[static_cast<size_t>(q)].dot(
                     P * dx[static_cast<size_t>(p)]) / len +
                 unit.dot(d2x[static_cast<size_t>(p)]
                             [static_cast<size_t>(q)]));
        }
    }
    return out;
}

void unit_vector_derivatives(
    const Vec3 &x,
    const std::array<Vec3, 12> &dx,
    const std::array<std::array<Vec3, 12>, 12> &d2x,
    Vec3 &unit,
    std::array<Vec3, 12> &du,
    std::array<std::array<Vec3, 12>, 12> &d2u) {
    const double len = x.norm();
    if (!(len > 0.0) || !std::isfinite(len)) {
        unit = Vec3::UnitZ();
        for (int p = 0; p < 12; ++p) {
            du[static_cast<size_t>(p)] = Vec3::Zero();
            for (int q = 0; q < 12; ++q) {
                d2u[static_cast<size_t>(p)][static_cast<size_t>(q)] =
                    Vec3::Zero();
            }
        }
        return;
    }

    unit = x / len;
    std::array<double, 12> dlen;
    for (int p = 0; p < 12; ++p) {
        dlen[static_cast<size_t>(p)] =
            unit.dot(dx[static_cast<size_t>(p)]);
        du[static_cast<size_t>(p)] =
            (dx[static_cast<size_t>(p)] -
             unit * dlen[static_cast<size_t>(p)]) / len;
    }

    for (int p = 0; p < 12; ++p) {
        for (int q = 0; q < 12; ++q) {
            const double d2len =
                (dx[static_cast<size_t>(p)].dot(
                     dx[static_cast<size_t>(q)]) +
                 x.dot(d2x[static_cast<size_t>(p)]
                         [static_cast<size_t>(q)])) / len -
                dlen[static_cast<size_t>(p)] *
                    dlen[static_cast<size_t>(q)] / len;
            d2u[static_cast<size_t>(p)][static_cast<size_t>(q)] =
                d2x[static_cast<size_t>(p)][static_cast<size_t>(q)] / len -
                dx[static_cast<size_t>(p)] *
                    (dlen[static_cast<size_t>(q)] / (len * len)) -
                dx[static_cast<size_t>(q)] *
                    (dlen[static_cast<size_t>(p)] / (len * len)) -
                x * (d2len / (len * len)) +
                x * (2.0 * dlen[static_cast<size_t>(p)] *
                     dlen[static_cast<size_t>(q)] / (len * len * len));
        }
    }
}

ScalarDerivatives triangle_area_derivatives(
    const std::array<Vec3, 4> &pos,
    const std::array<int, 3> &face) {
    Vec3 normal;
    std::array<Vec3, 12> dn;
    std::array<std::array<Vec3, 12>, 12> d2n;
    raw_normal_derivatives(pos, face, normal, dn, d2n);
    return norm_derivatives(normal, dn, d2n, 0.5);
}

ScalarDerivatives edge_length_squared_derivatives(int a,
                                                  int b,
                                                  const std::array<Vec3, 4> &pos) {
    ScalarDerivatives out;
    out.grad = Eigen::VectorXd::Zero(12);
    out.hess = Eigen::MatrixXd::Zero(12, 12);
    const Vec3 e = pos[static_cast<size_t>(b)] - pos[static_cast<size_t>(a)];
    out.value = e.squaredNorm();
    for (int c = 0; c < 3; ++c) {
        out.grad(3 * a + c) = -2.0 * e(c);
        out.grad(3 * b + c) = 2.0 * e(c);
        out.hess(3 * a + c, 3 * a + c) += 2.0;
        out.hess(3 * b + c, 3 * b + c) += 2.0;
        out.hess(3 * a + c, 3 * b + c) -= 2.0;
        out.hess(3 * b + c, 3 * a + c) -= 2.0;
    }
    return out;
}

ScalarDerivatives hinge_weight_derivatives(
    const std::array<Vec3, 4> &pos,
    int edge_a,
    int edge_b,
    const std::array<int, 3> &face0,
    const std::array<int, 3> &face1,
    const ShellEnergyParams &params) {
    const ScalarDerivatives l2 =
        edge_length_squared_derivatives(edge_a, edge_b, pos);
    const ScalarDerivatives a0 = triangle_area_derivatives(pos, face0);
    const ScalarDerivatives a1 = triangle_area_derivatives(pos, face1);
    const double area_e = (a0.value + a1.value) / 3.0;

    ScalarDerivatives area;
    area.value = area_e;
    area.grad = (a0.grad + a1.grad) / 3.0;
    area.hess = (a0.hess + a1.hess) / 3.0;

    ScalarDerivatives out;
    out.grad = Eigen::VectorXd::Zero(12);
    out.hess = Eigen::MatrixXd::Zero(12, 12);
    if (!(area.value > 0.0)) {
        return out;
    }

    const double coef = std::pow(params.thickness, 3.0);
    out.value = coef * l2.value / area.value;
    out.grad = coef *
        (l2.grad / area.value -
         l2.value * area.grad / (area.value * area.value));
    out.hess = coef *
        (l2.hess / area.value -
         (l2.grad * area.grad.transpose() +
          area.grad * l2.grad.transpose()) /
             (area.value * area.value) -
         l2.value * area.hess / (area.value * area.value) +
         2.0 * l2.value *
             (area.grad * area.grad.transpose()) /
             (area.value * area.value * area.value));
    return out;
}

ScalarDerivatives hinge_angle_derivatives(
    const std::array<Vec3, 4> &pos,
    const std::array<int, 3> &face0,
    const std::array<int, 3> &face1) {
    Vec3 normal0;
    Vec3 normal1;
    std::array<Vec3, 12> dn0_raw;
    std::array<Vec3, 12> dn1_raw;
    std::array<std::array<Vec3, 12>, 12> d2n0_raw;
    std::array<std::array<Vec3, 12>, 12> d2n1_raw;
    raw_normal_derivatives(pos, face0, normal0, dn0_raw, d2n0_raw);
    raw_normal_derivatives(pos, face1, normal1, dn1_raw, d2n1_raw);

    Vec3 n0;
    Vec3 n1;
    std::array<Vec3, 12> dn0;
    std::array<Vec3, 12> dn1;
    std::array<std::array<Vec3, 12>, 12> d2n0;
    std::array<std::array<Vec3, 12>, 12> d2n1;
    unit_vector_derivatives(normal0, dn0_raw, d2n0_raw, n0, dn0, d2n0);
    unit_vector_derivatives(normal1, dn1_raw, d2n1_raw, n1, dn1, d2n1);

    double s = n0.dot(n1);
    s = std::max(-1.0, std::min(1.0, s));
    const double sin2 = std::max(0.0, 1.0 - s * s);
    const double sin_theta = std::sqrt(sin2);

    ScalarDerivatives out;
    out.value = std::acos(s);
    out.grad = Eigen::VectorXd::Zero(12);
    out.hess = Eigen::MatrixXd::Zero(12, 12);
    if (!(sin_theta > 1e-10) || !std::isfinite(sin_theta)) {
        return out;
    }

    Eigen::VectorXd ds = Eigen::VectorXd::Zero(12);
    Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(12, 12);
    for (int p = 0; p < 12; ++p) {
        ds(p) = dn0[static_cast<size_t>(p)].dot(n1) +
                n0.dot(dn1[static_cast<size_t>(p)]);
    }
    for (int p = 0; p < 12; ++p) {
        for (int q = 0; q < 12; ++q) {
            Hs(p, q) =
                d2n0[static_cast<size_t>(p)][static_cast<size_t>(q)].dot(n1) +
                dn0[static_cast<size_t>(p)].dot(
                    dn1[static_cast<size_t>(q)]) +
                dn0[static_cast<size_t>(q)].dot(
                    dn1[static_cast<size_t>(p)]) +
                n0.dot(d2n1[static_cast<size_t>(p)]
                            [static_cast<size_t>(q)]);
        }
    }

    const double d1 = -1.0 / sin_theta;
    const double d2 = -s / (sin2 * sin_theta);
    out.grad = d1 * ds;
    out.hess = d1 * Hs + d2 * (ds * ds.transpose());
    return out;
}

ScalarDerivatives bending_angle_value_derivatives(
    const ScalarDerivatives &theta,
    const ShellEnergyParams &params) {
    ScalarDerivatives out;
    out.value = theta.value;
    out.grad = theta.grad;
    out.hess = theta.hess;
    if (!params.use_tan_bending) {
        return out;
    }

    const double clamp_hi = M_PI - params.angle_clamp_eps;
    if (theta.value >= clamp_hi) {
        out.value = 2.0 * std::tan(0.5 * clamp_hi);
        out.grad.setZero();
        out.hess.setZero();
        return out;
    }

    const double half = 0.5 * theta.value;
    const double t = std::tan(half);
    const double sec2 = 1.0 + t * t;
    const double d1 = sec2;
    const double d2 = sec2 * t;
    out.value = 2.0 * t;
    out.grad = d1 * theta.grad;
    out.hess = d1 * theta.hess +
               d2 * (theta.grad * theta.grad.transpose());
    return out;
}

void scatter_bending_hessian_closed_form(
    const MeshData &x_ref,
    const MeshData &x_def,
    const BendingEdge &e,
    const ShellEnergyParams &params,
    Eigen::MatrixXd *grad_ref,
    Eigen::MatrixXd *grad_def,
    std::vector<Eigen::Triplet<double>> *rr,
    std::vector<Eigen::Triplet<double>> *rd,
    std::vector<Eigen::Triplet<double>> *dr,
    std::vector<Eigen::Triplet<double>> *dd,
    double drop_tol) {
    // Tamstorf/Grinspun Eq. 11-12: bending Hessian is a weighted sum of the
    // hinge-angle Hessian and grad(theta) grad(theta)^T. Since our rest mesh
    // can also be optimized, include derivatives of the rest length/area weight.
    const int c = opposite_vertex(x_ref, e.f0, e.i, e.j);
    const int d = opposite_vertex(x_ref, e.f1, e.i, e.j);
    if (c < 0 || d < 0) {
        return;
    }

    const std::array<int, 4> verts = {e.i, e.j, c, d};
    std::vector<LocalDof> dofs;
    dofs.reserve(24);
    for (int v : verts) {
        for (int coord = 0; coord < 3; ++coord) {
            dofs.push_back(LocalDof{true, v, coord});
        }
    }
    for (int v : verts) {
        for (int coord = 0; coord < 3; ++coord) {
            dofs.push_back(LocalDof{false, v, coord});
        }
    }

    const std::array<int, 3> face0 = local_face_indices(x_ref, e.f0, verts);
    const std::array<int, 3> face1 = local_face_indices(x_ref, e.f1, verts);
    const std::array<Vec3, 4> ref_pos = local_positions(x_ref, verts);
    const std::array<Vec3, 4> def_pos = local_positions(x_def, verts);
    const int local_i = local_index_of(verts, e.i);
    const int local_j = local_index_of(verts, e.j);

    const ScalarDerivatives weight =
        hinge_weight_derivatives(
            ref_pos, local_i, local_j, face0, face1, params);
    if (!(weight.value > 0.0) || !std::isfinite(weight.value)) {
        return;
    }

    const ScalarDerivatives theta_ref =
        hinge_angle_derivatives(ref_pos, face0, face1);
    const ScalarDerivatives theta_def =
        hinge_angle_derivatives(def_pos, face0, face1);
    const ScalarDerivatives val_ref =
        bending_angle_value_derivatives(theta_ref, params);
    const ScalarDerivatives val_def =
        bending_angle_value_derivatives(theta_def, params);
    const double q = val_def.value - val_ref.value;

    Eigen::VectorXd local_grad = Eigen::VectorXd::Zero(24);
    local_grad.segment(0, 12) =
        q * q * weight.grad -
        2.0 * weight.value * q * val_ref.grad;
    local_grad.segment(12, 12) =
        2.0 * weight.value * q * val_def.grad;

    if (grad_ref != nullptr && grad_def != nullptr) {
        scatter_local_gradient(dofs, local_grad, *grad_ref, *grad_def);
    }

    if (rr == nullptr || rd == nullptr || dr == nullptr || dd == nullptr) {
        return;
    }

    Eigen::MatrixXd local_hess = Eigen::MatrixXd::Zero(24, 24);
    local_hess.block(0, 0, 12, 12) =
        q * q * weight.hess -
        2.0 * q *
            (weight.grad * val_ref.grad.transpose() +
             val_ref.grad * weight.grad.transpose()) +
        weight.value *
            (2.0 * val_ref.grad * val_ref.grad.transpose() -
             2.0 * q * val_ref.hess);
    local_hess.block(12, 12, 12, 12) =
        weight.value *
        (2.0 * val_def.grad * val_def.grad.transpose() +
         2.0 * q * val_def.hess);
    local_hess.block(0, 12, 12, 12) =
        (2.0 * q * weight.grad -
         2.0 * weight.value * val_ref.grad) *
        val_def.grad.transpose();
    local_hess.block(12, 0, 12, 12) =
        local_hess.block(0, 12, 12, 12).transpose();

    scatter_local_hessian(
        dofs, local_hess, *rr, *rd, *dr, *dd, drop_tol);
}

struct TriFrame {
    Mat32 J_ref;
    Mat32 J_def;
    Mat2 I_ref;
    Mat2 I_def;
    double area_ref = 0.0;
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

int opposite_vertex(const MeshData &mesh, int f, int i, int j) {
    for (int k = 0; k < 3; ++k) {
        const int v = mesh.F(f, k);
        if (v != i && v != j) {
            return v;
        }
    }
    return -1;
}

AD2 triangle_area_ad(const ADVec3 &a, const ADVec3 &b, const ADVec3 &c) {
    return 0.5 * norm_ad(cross_ad(b - a, c - a));
}

AD2 membrane_local_energy_ad(const MeshData &x_ref,
                             const MeshData &x_def,
                             int f,
                             const ShellEnergyParams &params,
                             std::vector<LocalDof> &dofs) {
    const int dim = 18;
    const int i = x_ref.F(f, 0);
    const int j = x_ref.F(f, 1);
    const int k = x_ref.F(f, 2);
    const std::array<int, 3> face = {i, j, k};

    dofs.clear();
    dofs.reserve(static_cast<size_t>(dim));
    for (int v : face) {
        for (int c = 0; c < 3; ++c) {
            dofs.push_back(LocalDof{true, v, c});
        }
    }
    for (int v : face) {
        for (int c = 0; c < 3; ++c) {
            dofs.push_back(LocalDof{false, v, c});
        }
    }

    const ADVec3 r0 = make_ad_vec3(x_ref, i, dim, 0);
    const ADVec3 r1 = make_ad_vec3(x_ref, j, dim, 3);
    const ADVec3 r2 = make_ad_vec3(x_ref, k, dim, 6);
    const ADVec3 d0 = make_ad_vec3(x_def, i, dim, 9);
    const ADVec3 d1 = make_ad_vec3(x_def, j, dim, 12);
    const ADVec3 d2 = make_ad_vec3(x_def, k, dim, 15);

    const ADVec3 re1 = r1 - r0;
    const ADVec3 re2 = r2 - r0;
    const ADVec3 de1 = d1 - d0;
    const ADVec3 de2 = d2 - d0;

    const ADMat2 I_ref = {
        dot_ad(re1, re1), dot_ad(re1, re2),
        dot_ad(re2, re1), dot_ad(re2, re2)};
    const ADMat2 I_def = {
        dot_ad(de1, de1), dot_ad(de1, de2),
        dot_ad(de2, de1), dot_ad(de2, de2)};
    const ADMat2 A = mat2_mul(mat2_inverse(I_ref), I_def);
    const AD2 detA = ad_clamped_det(mat2_det(A), params.det_smoothing);
    const AD2 density =
        0.5 * params.mu * mat2_trace(A) +
        0.25 * params.lambda * detA -
        0.25 * (2.0 * params.mu + params.lambda) * ad_log(detA) -
        params.mu - 0.25 * params.lambda;
    const AD2 area = triangle_area_ad(r0, r1, r2);
    return params.thickness * area * density;
}

void scatter_local_gradient(const std::vector<LocalDof> &dofs,
                            const Eigen::VectorXd &grad,
                            Eigen::MatrixXd &grad_ref,
                            Eigen::MatrixXd &grad_def) {
    for (int i = 0; i < static_cast<int>(dofs.size()); ++i) {
        const LocalDof &d = dofs[static_cast<size_t>(i)];
        Eigen::MatrixXd &target = d.ref ? grad_ref : grad_def;
        target(d.vertex, d.coord) += grad(i);
    }
}

void scatter_local_hessian(
    const std::vector<LocalDof> &dofs,
    const Eigen::MatrixXd &hess,
    std::vector<Eigen::Triplet<double>> &rr,
    std::vector<Eigen::Triplet<double>> &rd,
    std::vector<Eigen::Triplet<double>> &dr,
    std::vector<Eigen::Triplet<double>> &dd,
    double drop_tol) {
    for (int i = 0; i < static_cast<int>(dofs.size()); ++i) {
        const LocalDof &di = dofs[static_cast<size_t>(i)];
        const int row = 3 * di.vertex + di.coord;
        for (int j = 0; j < static_cast<int>(dofs.size()); ++j) {
            const double value = hess(i, j);
            if (!std::isfinite(value) || std::abs(value) <= drop_tol) {
                continue;
            }
            const LocalDof &dj = dofs[static_cast<size_t>(j)];
            const int col = 3 * dj.vertex + dj.coord;
            if (di.ref && dj.ref) {
                rr.emplace_back(row, col, value);
            } else if (di.ref && !dj.ref) {
                rd.emplace_back(row, col, value);
            } else if (!di.ref && dj.ref) {
                dr.emplace_back(row, col, value);
            } else {
                dd.emplace_back(row, col, value);
            }
        }
    }
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

void accumulate_bending_gradient_def_fd(const MeshData &x_ref,
                                        const MeshData &x_def,
                                        const ShellEnergyParams &params,
                                        const std::vector<BendingEdge> &edges,
                                        Eigen::MatrixXd &grad_def) {
    const double eps = params.bending_fd_eps;
    if (!(eps > 0.0)) return;

    auto wb_ref_def = [&](const MeshData &xr, const MeshData &xd) -> double {
        return bending_energy_only(xr, xd, params, edges);
    };

    MeshData xd = x_def;
    for (int v = 0; v < x_def.n_vertices(); ++v) {
        for (int c = 0; c < 3; ++c) {
            const double h = eps * (1.0 + std::abs(x_def.V(v, c)));

            xd.V(v, c) += h;
            const double ep = wb_ref_def(x_ref, xd);
            xd.V(v, c) -= 2.0 * h;
            const double em = wb_ref_def(x_ref, xd);
            xd.V(v, c) += h;
            grad_def(v, c) += (ep - em) / (2.0 * h);
        }
    }
}

void accumulate_bending_gradient_closed_form(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params,
    const std::vector<BendingEdge> &edges,
    Eigen::MatrixXd &grad_ref,
    Eigen::MatrixXd &grad_def) {
    for (const BendingEdge &e : edges) {
        scatter_bending_hessian_closed_form(
            x_ref, x_def, e, params,
            &grad_ref, &grad_def,
            nullptr, nullptr, nullptr, nullptr, 0.0);
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
    if (params.use_analytical_bending_gradient) {
        accumulate_bending_gradient_closed_form(
            x_ref, x_def, params, edges, out.grad_ref, out.grad_def);
    } else {
        accumulate_bending_gradient_fd(
            x_ref, x_def, params, edges, out.grad_ref, out.grad_def);
    }

    out.energy.total = out.energy.membrane + out.energy.bending;
    return out;
}

Eigen::MatrixXd shell_energy_def_gradient(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params) {
    validate_mesh_pair(x_ref, x_def);

    Eigen::MatrixXd grad_ref_dummy =
        Eigen::MatrixXd::Zero(x_ref.n_vertices(), 3);
    Eigen::MatrixXd grad_def =
        Eigen::MatrixXd::Zero(x_def.n_vertices(), 3);

    const std::vector<TriFrame> frames = build_frames(x_ref, x_def);
    accumulate_membrane_gradient(
        x_ref, x_def, frames, params, grad_ref_dummy, grad_def);

    const std::vector<BendingEdge> edges = build_interior_edges(x_ref);
    if (params.use_analytical_bending_gradient) {
        accumulate_bending_gradient_closed_form(
            x_ref, x_def, params, edges, grad_ref_dummy, grad_def);
    } else {
        accumulate_bending_gradient_def_fd(
            x_ref, x_def, params, edges, grad_def);
    }

    return grad_def;
}

ShellEnergyHessianResult shell_energy_hessian(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params) {
    validate_mesh_pair(x_ref, x_def);

    const int ndof = 3 * x_ref.n_vertices();
    constexpr double drop_tol = 1e-12;
    std::vector<Eigen::Triplet<double>> rr;
    std::vector<Eigen::Triplet<double>> rd;
    std::vector<Eigen::Triplet<double>> dr;
    std::vector<Eigen::Triplet<double>> dd;
    rr.reserve(static_cast<size_t>(x_ref.n_faces()) * 81);
    rd.reserve(static_cast<size_t>(x_ref.n_faces()) * 81);
    dr.reserve(static_cast<size_t>(x_ref.n_faces()) * 81);
    dd.reserve(static_cast<size_t>(x_ref.n_faces()) * 81);

    for (int f = 0; f < x_ref.n_faces(); ++f) {
        std::vector<LocalDof> dofs;
        const AD2 local =
            membrane_local_energy_ad(x_ref, x_def, f, params, dofs);
        scatter_local_hessian(
            dofs, local.hess, rr, rd, dr, dd, drop_tol);
    }

    const std::vector<BendingEdge> edges = build_interior_edges(x_ref);
    for (const BendingEdge &e : edges) {
        scatter_bending_hessian_closed_form(
            x_ref, x_def, e, params,
            nullptr, nullptr, &rr, &rd, &dr, &dd, drop_tol);
    }

    ShellEnergyHessianResult out;
    out.ref_ref.resize(ndof, ndof);
    out.ref_def.resize(ndof, ndof);
    out.def_ref.resize(ndof, ndof);
    out.def_def.resize(ndof, ndof);
    auto sum_duplicates = [](double a, double b) { return a + b; };
    out.ref_ref.setFromTriplets(rr.begin(), rr.end(), sum_duplicates);
    out.ref_def.setFromTriplets(rd.begin(), rd.end(), sum_duplicates);
    out.def_ref.setFromTriplets(dr.begin(), dr.end(), sum_duplicates);
    out.def_def.setFromTriplets(dd.begin(), dd.end(), sum_duplicates);
    out.ref_ref.makeCompressed();
    out.ref_def.makeCompressed();
    out.def_ref.makeCompressed();
    out.def_def.makeCompressed();
    return out;
}

Eigen::SparseMatrix<double> shell_energy_def_hessian(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params) {
    return shell_energy_hessian(x_ref, x_def, params).def_def;
}

} // namespace rsh
