#include "HsPreconditioner.h"
#include "DeterministicReduction.h"
#include "Constraints.h"
#include "FaceGeom.h"
#include "BVH.h"
#include "BCT.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/IterativeSolvers>

namespace rsh {

// Forward declaration for traits specialization
struct HsOperator;
struct HsRightPreconditionedOperator;
struct HsLeftPreconditionedOperator;

} // namespace rsh

namespace Eigen {
namespace internal {
template <>
struct traits<rsh::HsOperator> : public traits<Eigen::MatrixXd> {};
template <>
struct traits<rsh::HsRightPreconditionedOperator> : public traits<Eigen::MatrixXd> {};
template <>
struct traits<rsh::HsLeftPreconditionedOperator> : public traits<Eigen::MatrixXd> {};
} // namespace internal
} // namespace Eigen

namespace rsh {

namespace {

double robust_cotangent(const Eigen::Vector3d &u, const Eigen::Vector3d &v) {
    const double cross_norm = u.cross(v).norm();
    const double scale = u.norm() * v.norm();
    if (!std::isfinite(cross_norm) || !std::isfinite(scale) || scale <= 0.0) {
        return 0.0;
    }

    constexpr double kRelEps = 1e-14;
    if (cross_norm <= kRelEps * scale) {
        return 0.0;
    }

    const double cot = u.dot(v) / cross_norm;
    return std::isfinite(cot) ? cot : 0.0;
}

Eigen::Vector3d face_scalar_gradient(const MeshData &mesh,
                                     const FaceGeom &g,
                                     const Eigen::VectorXd &x,
                                     int f) {
    const int i0 = mesh.F(f, 0);
    const int i1 = mesh.F(f, 1);
    const int i2 = mesh.F(f, 2);

    const Eigen::Vector3d v0 = mesh.V.row(i0).transpose();
    const Eigen::Vector3d v1 = mesh.V.row(i1).transpose();
    const Eigen::Vector3d v2 = mesh.V.row(i2).transpose();

    Eigen::Vector3d E0;
    Eigen::Vector3d E1;
    Eigen::Vector3d E2;
    opposite_edges(v0, v1, v2, E0, E1, E2);

    const Eigen::Vector3d n = g.N.row(f).transpose();
    const double inv_2a = 1.0 / (2.0 * g.A(f));
    return inv_2a *
        (x(i0) * n.cross(E0) +
         x(i1) * n.cross(E1) +
         x(i2) * n.cross(E2));
}

void scatter_face_gradient_adjoint(const MeshData &mesh,
                                   const FaceGeom &g,
                                   int f,
                                   const Eigen::Vector3d &face_dual,
                                   Eigen::VectorXd &y) {
    const int i0 = mesh.F(f, 0);
    const int i1 = mesh.F(f, 1);
    const int i2 = mesh.F(f, 2);

    const Eigen::Vector3d v0 = mesh.V.row(i0).transpose();
    const Eigen::Vector3d v1 = mesh.V.row(i1).transpose();
    const Eigen::Vector3d v2 = mesh.V.row(i2).transpose();

    Eigen::Vector3d E0;
    Eigen::Vector3d E1;
    Eigen::Vector3d E2;
    opposite_edges(v0, v1, v2, E0, E1, E2);

    const Eigen::Vector3d n = g.N.row(f).transpose();
    const double inv_2a = 1.0 / (2.0 * g.A(f));
    y(i0) += face_dual.dot(inv_2a * n.cross(E0));
    y(i1) += face_dual.dot(inv_2a * n.cross(E1));
    y(i2) += face_dual.dot(inv_2a * n.cross(E2));
}

double face_average(const MeshData &mesh,
                    const Eigen::VectorXd &x,
                    int f) {
    return (x(mesh.F(f, 0)) +
            x(mesh.F(f, 1)) +
            x(mesh.F(f, 2))) / 3.0;
}

Eigen::VectorXd flatten_vertex_field(const Eigen::MatrixXd &x) {
    if (x.cols() != 3) {
        throw std::runtime_error("flatten_vertex_field: input must have three columns");
    }
    Eigen::VectorXd out(3 * x.rows());
    for (int i = 0; i < x.rows(); ++i) {
        for (int c = 0; c < 3; ++c) {
            out(3 * i + c) = x(i, c);
        }
    }
    return out;
}

Eigen::MatrixXd unflatten_vertex_field(const Eigen::VectorXd &x, int nv) {
    if (x.size() != 3 * nv) {
        throw std::runtime_error("unflatten_vertex_field: size mismatch");
    }
    Eigen::MatrixXd out(nv, 3);
    for (int i = 0; i < nv; ++i) {
        for (int c = 0; c < 3; ++c) {
            out(i, c) = x(3 * i + c);
        }
    }
    return out;
}

Eigen::Vector3d face_vector_average(const MeshData &mesh,
                                    const Eigen::MatrixXd &x,
                                    int f) {
    return (x.row(mesh.F(f, 0)).transpose() +
            x.row(mesh.F(f, 1)).transpose() +
            x.row(mesh.F(f, 2)).transpose()) / 3.0;
}

void face_basis_gradients(const MeshData &mesh,
                          const FaceGeom &g,
                          int f,
                          Eigen::Vector3d grads[3]) {
    const int i0 = mesh.F(f, 0);
    const int i1 = mesh.F(f, 1);
    const int i2 = mesh.F(f, 2);

    const Eigen::Vector3d v0 = mesh.V.row(i0).transpose();
    const Eigen::Vector3d v1 = mesh.V.row(i1).transpose();
    const Eigen::Vector3d v2 = mesh.V.row(i2).transpose();

    Eigen::Vector3d E0;
    Eigen::Vector3d E1;
    Eigen::Vector3d E2;
    opposite_edges(v0, v1, v2, E0, E1, E2);

    const Eigen::Vector3d n = g.N.row(f).transpose();
    const double inv_2a = 1.0 / (2.0 * g.A(f));
    grads[0] = inv_2a * n.cross(E0);
    grads[1] = inv_2a * n.cross(E1);
    grads[2] = inv_2a * n.cross(E2);
}

Eigen::Matrix3d face_vector_gradient(const MeshData &mesh,
                                     const FaceGeom &g,
                                     const Eigen::MatrixXd &x,
                                     int f) {
    Eigen::Vector3d grads[3];
    face_basis_gradients(mesh, g, f, grads);

    Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
    for (int c = 0; c < 3; ++c) {
        const int vi = mesh.F(f, c);
        const Eigen::Vector3d value = x.row(vi).transpose();
        out += value * grads[c].transpose();
    }
    return out;
}

void scatter_face_matrix_adjoint(const MeshData &mesh,
                                 const FaceGeom &g,
                                 int f,
                                 const Eigen::Matrix3d &face_dual,
                                 Eigen::MatrixXd &y) {
    Eigen::Vector3d grads[3];
    face_basis_gradients(mesh, g, f, grads);
    for (int c = 0; c < 3; ++c) {
        const int vi = mesh.F(f, c);
        y.row(vi) += (face_dual * grads[c]).transpose();
    }
}

double b0_kernel_sym_faces(const FaceGeom &g,
                           int a,
                           int b,
                           double s) {
    const Eigen::Vector3d e = g.C.row(a) - g.C.row(b);
    const double r2 = e.squaredNorm();
    if (r2 == 0.0) return 0.0;

    const Eigen::Vector3d na = g.N.row(a).transpose();
    const Eigen::Vector3d nb = g.N.row(b).transpose();
    const double pa = na.dot(e);
    const double pb = nb.dot(-e);
    return (pa * pa + pb * pb) / std::pow(r2, s + 2.0);
}

double b0_kernel_sym_clusters(const BVHNode &U,
                              const BVHNode &V,
                              double s) {
    if (U.area <= 0.0 || V.area <= 0.0) return 0.0;

    const Eigen::Vector3d e = U.centroid - V.centroid;
    const double r2 = e.squaredNorm();
    if (r2 == 0.0) return 0.0;

    const Eigen::Vector3d nU = U.normal_sum / U.area;
    const Eigen::Vector3d nV = V.normal_sum / V.area;
    const double pU = nU.dot(e);
    const double pV = nV.dot(-e);
    return (pU * pU + pV * pV) / std::pow(r2, s + 2.0);
}

void project_constant_mode(Eigen::VectorXd &x) {
    if (x.size() == 0) return;
    x.array() -= x.mean();
}

void project_mass_constant_mode(const HsOperators &hs, Eigen::VectorXd &x) {
    if (x.size() == 0) return;
    const double total_mass = hs.mass_diag.sum();
    if (total_mass <= 0.0 || hs.mass_diag.size() != x.size()) {
        project_constant_mode(x);
        return;
    }
    const double mean = (hs.mass_diag.array() * x.array()).sum() / total_mass;
    x.array() -= mean;
}

Eigen::VectorXd apply_lumped_mass(const HsOperators &hs,
                                  const Eigen::VectorXd &x) {
    if (hs.mass_diag.size() != x.size()) {
        throw std::runtime_error("apply_lumped_mass: size mismatch");
    }
    return hs.mass_diag.array() * x.array();
}

bool is_flat_vector_field(const HsOperators &hs, const Eigen::VectorXd &x) {
    return hs.mass_diag.size() > 0 && x.size() == 3 * hs.mass_diag.size();
}

Eigen::VectorXd apply_lumped_mass_general(const HsOperators &hs,
                                          const Eigen::VectorXd &x) {
    if (is_flat_vector_field(hs, x)) {
        Eigen::VectorXd out(x.size());
        for (int i = 0; i < hs.mass_diag.size(); ++i) {
            for (int c = 0; c < 3; ++c) {
                out(3 * i + c) = hs.mass_diag(i) * x(3 * i + c);
            }
        }
        return out;
    }
    return apply_lumped_mass(hs, x);
}

Eigen::SparseMatrix<double> h1_metric_matrix(const HsOperators &hs) {
    Eigen::SparseMatrix<double> h1 = hs.L;
    if (hs.M_full.rows() == hs.L.rows() && hs.M_full.cols() == hs.L.cols()) {
        h1 += hs.M_full;
    } else {
        h1 += hs.M;
    }
    h1.makeCompressed();
    return h1;
}

Eigen::SparseMatrix<double> lifted_laplacian(const HsOperators &hs) {
    constexpr double kLift = 1e-12;
    Eigen::SparseMatrix<double> lifted = hs.L;
    lifted += kLift * hs.M;
    lifted.makeCompressed();
    return lifted;
}

void apply_hs_constraints(Eigen::MatrixXd &field,
                          const HsConstraints &constraints) {
    if (constraints.pin_barycenter) {
        project_barycenter(field);
    }
    if (constraints.pin_mask != nullptr) {
        apply_pin_mask(field, *constraints.pin_mask);
    }
}

} // namespace

// Matrix-free vector-valued operator for A = sigma * (B + B_0), where B is
// the high-order Frobenius D_f(delta f) term from RSu Eq. 7 / Eq. 12 and
// B_0 is the low-order TPE-modulated term from RSu Eq. 8. The Eigen
// interface is a flattened 3-vector field, ordered as
// (v0.x, v0.y, v0.z, v1.x, ...).
struct HsOperator : public Eigen::EigenBase<HsOperator> {
    const MeshData& mesh;
    const HsPreconditionerParams& params;
    const HsOperators& hs;
    const FaceGeom& g;
    const BVH& bvh;
    const BlockPairs& bp;

    HsOperator(const MeshData& mesh, const HsPreconditionerParams& params,
               const HsOperators& hs, const FaceGeom& g, const BVH& bvh, const BlockPairs& bp)
        : mesh(mesh), params(params), hs(hs), g(g), bvh(bvh), bp(bp) {}

    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    typedef int Index;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return 3 * mesh.n_vertices(); }
    Index cols() const { return 3 * mesh.n_vertices(); }

    template<typename Rhs>
    Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& x) const {
        Eigen::VectorXd flat = x.derived();
        if (flat.size() != rows()) {
            throw std::runtime_error("HsOperator::operator*: size mismatch");
        }
        return flatten_vertex_field(apply(unflatten_vertex_field(flat, mesh.n_vertices())));
    }

    Eigen::MatrixXd apply(const Eigen::MatrixXd &x) const {
        const int nv = mesh.n_vertices();
        const int nf = mesh.n_faces();
        const double power = params.s;
        if (x.rows() != nv || x.cols() != 3) {
            throw std::runtime_error("HsOperator::apply: input must be n_vertices x 3");
        }

        // 1. Per-face D_f delta f matrices for B and face averages for B_0.
        std::vector<Eigen::Matrix3d> grad_face(static_cast<size_t>(nf));
        std::vector<Eigen::Vector3d> face_x(static_cast<size_t>(nf));
        for (int i = 0; i < nf; ++i) {
            grad_face[static_cast<size_t>(i)] =
                face_vector_gradient(mesh, g, x, i);
            face_x[static_cast<size_t>(i)] = face_vector_average(mesh, x, i);
        }

        // 2. Pre-aggregate Q_U = sum_{a in U} A_a D_f delta f(a).
        std::vector<Eigen::Matrix3d> Q(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Matrix3d::Zero());
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    sum += g.A(f) * grad_face[static_cast<size_t>(f)];
                }
                Q[i] = sum;
            } else {
                Q[i] = Q[node.left] + Q[node.right];
            }
        }

        // B_0 vector aggregate Q0_U = sum_{a in U} A_a x_a.
        std::vector<Eigen::Vector3d> Q0(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Vector3d::Zero());
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    sum += g.A(f) * face_x[static_cast<size_t>(f)];
                }
                Q0[static_cast<size_t>(i)] = sum;
            } else {
                Q0[static_cast<size_t>(i)] =
                    Q0[static_cast<size_t>(node.left)] +
                    Q0[static_cast<size_t>(node.right)];
            }
        }

        // 3. Hierarchical integrals. The BCT enumerates ordered face pairs.
        // B has a symmetric kernel, so the directed target gets the factor 2.
        // B_0 symmetrizes the asymmetric projector kernel explicitly as
        // K(S,T)+K(T,S), so it does not get an additional factor 2.
        std::vector<double> SumA(bvh.n_nodes(), 0.0);
        std::vector<Eigen::Matrix3d> SumQ(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Matrix3d::Zero());
        std::vector<double> SumA0(bvh.n_nodes(), 0.0);
        std::vector<Eigen::Vector3d> SumQ0(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Vector3d::Zero());
        const int lanes = canonical_reduction_lanes();
        std::vector<std::vector<double>> tls_SumA(
            static_cast<size_t>(lanes),
            std::vector<double>(static_cast<size_t>(bvh.n_nodes()), 0.0));
        std::vector<std::vector<Eigen::Matrix3d>> tls_SumQ(
            static_cast<size_t>(lanes),
            std::vector<Eigen::Matrix3d>(
                static_cast<size_t>(bvh.n_nodes()), Eigen::Matrix3d::Zero()));
        std::vector<std::vector<double>> tls_SumA0(
            static_cast<size_t>(lanes),
            std::vector<double>(static_cast<size_t>(bvh.n_nodes()), 0.0));
        std::vector<std::vector<Eigen::Vector3d>> tls_SumQ0(
            static_cast<size_t>(lanes),
            std::vector<Eigen::Vector3d>(
                static_cast<size_t>(bvh.n_nodes()), Eigen::Vector3d::Zero()));
        const int n_admissible = static_cast<int>(bp.admissible.size());
#pragma omp parallel for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            const IndexRange range =
                canonical_static_range(n_admissible, lane, lanes);
            for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
                const auto& cp = bp.admissible[static_cast<size_t>(pair_idx)];
                const BVHNode& U = bvh.nodes[cp.u];
                const BVHNode& V = bvh.nodes[cp.v];
                double r2 = (U.centroid - V.centroid).squaredNorm();
                if (r2 == 0.0) continue;
                double K = 1.0 / std::pow(r2, power);
                
                // Interaction U <- V (covers directed pair)
                tls_SumA[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K * V.area;
                tls_SumQ[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K * Q[cp.v];

                const double K0 = b0_kernel_sym_clusters(U, V, power);
                tls_SumA0[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K0 * V.area;
                tls_SumQ0[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K0 * Q0[static_cast<size_t>(cp.v)];
            }
        }
        for (int tid = 0; tid < lanes; ++tid) {
            for (int i = 0; i < bvh.n_nodes(); ++i) {
                const size_t si = static_cast<size_t>(i);
                SumA[si] += tls_SumA[static_cast<size_t>(tid)][si];
                SumQ[si] += tls_SumQ[static_cast<size_t>(tid)][si];
                SumA0[si] += tls_SumA0[static_cast<size_t>(tid)][si];
                SumQ0[si] += tls_SumQ0[static_cast<size_t>(tid)][si];
            }
        }

        // Propagate cluster sums down the tree.
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (!node.is_leaf()) {
                SumA[node.left] += SumA[i];
                SumQ[node.left] += SumQ[i];
                SumA0[node.left] += SumA0[i];
                SumQ0[node.left] += SumQ0[i];
                SumA[node.right] += SumA[i];
                SumQ[node.right] += SumQ[i];
                SumA0[node.right] += SumA0[i];
                SumQ0[node.right] += SumQ0[i];
            }
        }

        std::vector<Eigen::Matrix3d> face_dual(
            static_cast<size_t>(nf), Eigen::Matrix3d::Zero());
        std::vector<Eigen::Vector3d> face_dual0(
            static_cast<size_t>(nf), Eigen::Vector3d::Zero());
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    face_dual[static_cast<size_t>(f)] +=
                        2.0 * g.A(f) *
                        (grad_face[static_cast<size_t>(f)] * SumA[i] - SumQ[i]);
                    face_dual0[static_cast<size_t>(f)] +=
                        g.A(f) *
                        (face_x[static_cast<size_t>(f)] * SumA0[i] - SumQ0[i]);
                }
            }
        }

        // Near-field exact summation
        std::vector<std::vector<Eigen::Matrix3d>> tls_face_dual(
            static_cast<size_t>(lanes),
            std::vector<Eigen::Matrix3d>(
                static_cast<size_t>(nf), Eigen::Matrix3d::Zero()));
        std::vector<std::vector<Eigen::Vector3d>> tls_face_dual0(
            static_cast<size_t>(lanes),
            std::vector<Eigen::Vector3d>(
                static_cast<size_t>(nf), Eigen::Vector3d::Zero()));
        const int n_near = static_cast<int>(bp.near_field.size());
#pragma omp parallel for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            const IndexRange range = canonical_static_range(n_near, lane, lanes);
            for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
                const auto& cp = bp.near_field[static_cast<size_t>(pair_idx)];
                const BVHNode& U = bvh.nodes[cp.u];
                const BVHNode& V = bvh.nodes[cp.v];
                const bool self = (cp.u == cp.v);
                for (int i = U.face_start; i < U.face_end; ++i) {
                    int a = bvh.face_indices[i];
                    for (int j = V.face_start; j < V.face_end; ++j) {
                        int b = bvh.face_indices[j];
                        if (self && a == b) continue;
                        double r2 = (g.C.row(a) - g.C.row(b)).squaredNorm();
                        if (r2 == 0.0) continue;
                        double K = 1.0 / std::pow(r2, power);
                        tls_face_dual[static_cast<size_t>(lane)]
                            [static_cast<size_t>(a)] +=
                            2.0 * g.A(a) * g.A(b) * K *
                            (grad_face[static_cast<size_t>(a)] -
                             grad_face[static_cast<size_t>(b)]);

                        const double K0 = b0_kernel_sym_faces(g, a, b, power);
                        tls_face_dual0[static_cast<size_t>(lane)]
                            [static_cast<size_t>(a)] +=
                            g.A(a) * g.A(b) * K0 *
                            (face_x[static_cast<size_t>(a)] -
                             face_x[static_cast<size_t>(b)]);
                    }
                }
            }
        }
        for (int tid = 0; tid < lanes; ++tid) {
            for (int f = 0; f < nf; ++f) {
                const size_t sf = static_cast<size_t>(f);
                face_dual[sf] +=
                    tls_face_dual[static_cast<size_t>(tid)][sf];
                face_dual0[sf] +=
                    tls_face_dual0[static_cast<size_t>(tid)][sf];
            }
        }

        // 4. Combine: y = sigma * (B x + B_0 x).
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(nv, 3);
        for (int f = 0; f < nf; ++f) {
            scatter_face_matrix_adjoint(
                mesh,
                g,
                f,
                params.sigma * face_dual[static_cast<size_t>(f)],
                y);
            const Eigen::Vector3d b0_vertex =
                params.sigma * face_dual0[static_cast<size_t>(f)] / 3.0;
            y.row(mesh.F(f, 0)) += b0_vertex.transpose();
            y.row(mesh.F(f, 1)) += b0_vertex.transpose();
            y.row(mesh.F(f, 2)) += b0_vertex.transpose();
        }

        return y;
    }
};

// Matrix-free scalar fractional Laplacian from RSu Eq. 6. The parameter
// sigma_scalar gives an operator of order 2*sigma_scalar on a surface:
//
//   <L^sigma u, v> = int int (u(x)-u(y))(v(x)-v(y))
//                    / |f(x)-f(y)|^(2*sigma_scalar + 2).
//
// Unit 2B.3 uses sigma_scalar = 2 - s as the middle forward apply in the
// sandwich preconditioner Abar^{-1}.
struct ScalarFractionalLaplacian {
    const MeshData& mesh;
    const FaceGeom& g;
    const BVH& bvh;
    const BlockPairs& bp;
    double sigma_scalar = 1.0;

    ScalarFractionalLaplacian(const MeshData& mesh,
                              const FaceGeom& g,
                              const BVH& bvh,
                              const BlockPairs& bp,
                              double sigma_scalar)
        : mesh(mesh), g(g), bvh(bvh), bp(bp), sigma_scalar(sigma_scalar) {}

    template<typename Rhs>
    Eigen::VectorXd apply(const Eigen::MatrixBase<Rhs>& x) const {
        const int nf = mesh.n_faces();
        const int nv = mesh.n_vertices();
        const double power = sigma_scalar + 1.0;

        Eigen::VectorXd face_x(nf);
        for (int f = 0; f < nf; ++f) {
            face_x(f) = face_average(mesh, x, f);
        }

        std::vector<double> Q(static_cast<size_t>(bvh.n_nodes()), 0.0);
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                double sum = 0.0;
                for (int j = node.face_start; j < node.face_end; ++j) {
                    const int f = bvh.face_indices[j];
                    sum += g.A(f) * face_x(f);
                }
                Q[static_cast<size_t>(i)] = sum;
            } else {
                Q[static_cast<size_t>(i)] =
                    Q[static_cast<size_t>(node.left)] +
                    Q[static_cast<size_t>(node.right)];
            }
        }

        std::vector<double> SumA(bvh.n_nodes(), 0.0);
        std::vector<double> SumQ(bvh.n_nodes(), 0.0);
        const int lanes = canonical_reduction_lanes();
        std::vector<std::vector<double>> tls_SumA(
            static_cast<size_t>(lanes),
            std::vector<double>(static_cast<size_t>(bvh.n_nodes()), 0.0));
        std::vector<std::vector<double>> tls_SumQ(
            static_cast<size_t>(lanes),
            std::vector<double>(static_cast<size_t>(bvh.n_nodes()), 0.0));
        const int n_admissible = static_cast<int>(bp.admissible.size());
#pragma omp parallel for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            const IndexRange range =
                canonical_static_range(n_admissible, lane, lanes);
            for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
                const auto& cp = bp.admissible[static_cast<size_t>(pair_idx)];
                const BVHNode& U = bvh.nodes[cp.u];
                const BVHNode& V = bvh.nodes[cp.v];
                const double r2 = (U.centroid - V.centroid).squaredNorm();
                if (r2 == 0.0) continue;
                const double K = 1.0 / std::pow(r2, power);
                tls_SumA[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K * V.area;
                tls_SumQ[static_cast<size_t>(lane)][static_cast<size_t>(cp.u)] +=
                    K * Q[static_cast<size_t>(cp.v)];
            }
        }
        for (int tid = 0; tid < lanes; ++tid) {
            for (int i = 0; i < bvh.n_nodes(); ++i) {
                const size_t si = static_cast<size_t>(i);
                SumA[si] += tls_SumA[static_cast<size_t>(tid)][si];
                SumQ[si] += tls_SumQ[static_cast<size_t>(tid)][si];
            }
        }

        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (!node.is_leaf()) {
                SumA[node.left] += SumA[i];
                SumQ[node.left] += SumQ[i];
                SumA[node.right] += SumA[i];
                SumQ[node.right] += SumQ[i];
            }
        }

        Eigen::VectorXd face_dual = Eigen::VectorXd::Zero(nf);
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                for (int j = node.face_start; j < node.face_end; ++j) {
                    const int f = bvh.face_indices[j];
                    face_dual(f) +=
                        2.0 * g.A(f) * (face_x(f) * SumA[i] - SumQ[i]);
                }
            }
        }

        std::vector<Eigen::VectorXd> tls_face_dual;
        tls_face_dual.reserve(static_cast<size_t>(lanes));
        for (int tid = 0; tid < lanes; ++tid) {
            tls_face_dual.push_back(Eigen::VectorXd::Zero(nf));
        }
        const int n_near = static_cast<int>(bp.near_field.size());
#pragma omp parallel for schedule(static)
        for (int lane = 0; lane < lanes; ++lane) {
            Eigen::VectorXd &face_dual_local =
                tls_face_dual[static_cast<size_t>(lane)];
            const IndexRange range = canonical_static_range(n_near, lane, lanes);
            for (int pair_idx = range.begin; pair_idx < range.end; ++pair_idx) {
                const auto& cp = bp.near_field[static_cast<size_t>(pair_idx)];
                const BVHNode& U = bvh.nodes[cp.u];
                const BVHNode& V = bvh.nodes[cp.v];
                const bool self = (cp.u == cp.v);
                for (int i = U.face_start; i < U.face_end; ++i) {
                    const int a = bvh.face_indices[i];
                    for (int j = V.face_start; j < V.face_end; ++j) {
                        const int b = bvh.face_indices[j];
                        if (self && a == b) continue;
                        const double r2 = (g.C.row(a) - g.C.row(b)).squaredNorm();
                        if (r2 == 0.0) continue;
                        const double K = 1.0 / std::pow(r2, power);
                        face_dual_local(a) +=
                            2.0 * g.A(a) * g.A(b) * K * (face_x(a) - face_x(b));
                    }
                }
            }
        }
        for (int tid = 0; tid < lanes; ++tid) {
            face_dual += tls_face_dual[static_cast<size_t>(tid)];
        }

        Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
        for (int f = 0; f < nf; ++f) {
            const double vertex_value = face_dual(f) / 3.0;
            y(mesh.F(f, 0)) += vertex_value;
            y(mesh.F(f, 1)) += vertex_value;
            y(mesh.F(f, 2)) += vertex_value;
        }
        return y;
    }
};

enum class HsLaplacianInverseMode {
    RawStiffness,
    LaplaceBeltrami,
    H1Metric
};

enum class HsConstantProjectionMode {
    Algebraic,
    MassWeighted,
    None
};

void project_constant_mode_general(const HsOperators &hs,
                                   HsConstantProjectionMode mode,
                                   Eigen::VectorXd &x) {
    if (mode == HsConstantProjectionMode::None || x.size() == 0) {
        return;
    }
    if (!is_flat_vector_field(hs, x)) {
        if (mode == HsConstantProjectionMode::MassWeighted) {
            project_mass_constant_mode(hs, x);
        } else {
            project_constant_mode(x);
        }
        return;
    }

    Eigen::MatrixXd field = unflatten_vertex_field(x, hs.mass_diag.size());
    for (int c = 0; c < 3; ++c) {
        Eigen::VectorXd col = field.col(c);
        if (mode == HsConstantProjectionMode::MassWeighted) {
            project_mass_constant_mode(hs, col);
        } else {
            project_constant_mode(col);
        }
        field.col(c) = col;
    }
    x = flatten_vertex_field(field);
}

Eigen::VectorXd solve_laplacian_general(
    const HsOperators &hs,
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &factor,
    const Eigen::VectorXd &rhs) {
    if (!is_flat_vector_field(hs, rhs)) {
        return factor.solve(rhs);
    }

    Eigen::MatrixXd rhs_field = unflatten_vertex_field(rhs, hs.mass_diag.size());
    Eigen::MatrixXd out(rhs_field.rows(), 3);
    for (int c = 0; c < 3; ++c) {
        out.col(c) = factor.solve(rhs_field.col(c));
    }
    return flatten_vertex_field(out);
}

Eigen::VectorXd apply_middle_general(const HsOperators &hs,
                                     const ScalarFractionalLaplacian &middle,
                                     const Eigen::VectorXd &x) {
    if (!is_flat_vector_field(hs, x)) {
        return middle.apply(x);
    }

    Eigen::MatrixXd field = unflatten_vertex_field(x, hs.mass_diag.size());
    Eigen::MatrixXd out(field.rows(), 3);
    for (int c = 0; c < 3; ++c) {
        out.col(c) = middle.apply(field.col(c));
    }
    return flatten_vertex_field(out);
}

struct HsSandwichPreconditioner {
    const HsOperators& hs;
    const ScalarFractionalLaplacian& middle;
    HsLaplacianInverseMode inverse_mode = HsLaplacianInverseMode::RawStiffness;
    HsConstantProjectionMode projection_mode = HsConstantProjectionMode::Algebraic;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> laplacian_factor;
    Eigen::ComputationInfo factor_info = Eigen::InvalidInput;

    HsSandwichPreconditioner(const HsOperators& hs,
                             const ScalarFractionalLaplacian& middle,
                             HsLaplacianInverseMode inverse_mode =
                                 HsLaplacianInverseMode::RawStiffness,
                             HsConstantProjectionMode projection_mode =
                                 HsConstantProjectionMode::Algebraic)
        : hs(hs),
          middle(middle),
          inverse_mode(inverse_mode),
          projection_mode(projection_mode) {
        const Eigen::SparseMatrix<double> system =
            inverse_mode == HsLaplacianInverseMode::H1Metric
                ? h1_metric_matrix(hs)
                : lifted_laplacian(hs);
        laplacian_factor.compute(system);
        factor_info = laplacian_factor.info();
    }

    void project(Eigen::VectorXd &x) const {
        project_constant_mode_general(hs, projection_mode, x);
    }

    Eigen::VectorXd laplace_inverse(Eigen::VectorXd rhs) const {
        if (inverse_mode == HsLaplacianInverseMode::LaplaceBeltrami) {
            rhs = apply_lumped_mass_general(hs, rhs);
        }
        // The lifted factor can solve a nonzero-sum RHS, but projecting the
        // RHS keeps the solve aligned with the cotan nullspace we intend to
        // quotient out.
        project(rhs);
        Eigen::VectorXd out = solve_laplacian_general(hs, laplacian_factor, rhs);
        project(out);
        return out;
    }

    template<typename Rhs>
    Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& r) const {
        if (factor_info != Eigen::Success) {
            throw std::runtime_error(
                "HsSandwichPreconditioner: lifted cotan factorization failed");
        }

        Eigen::VectorXd rhs = r;
        project(rhs);

        Eigen::VectorXd w1 = laplace_inverse(rhs);

        Eigen::VectorXd w2 = apply_middle_general(hs, middle, w1);
        project(w2);

        Eigen::VectorXd out = laplace_inverse(w2);
        project(out);
        return out;
    }

    Eigen::ComputationInfo info() const { return factor_info; }
};

struct HsRightPreconditionedOperator
    : public Eigen::EigenBase<HsRightPreconditionedOperator> {
    const HsOperator& op;
    const HsSandwichPreconditioner& preconditioner;

    HsRightPreconditionedOperator(const HsOperator& op,
                                  const HsSandwichPreconditioner& preconditioner)
        : op(op), preconditioner(preconditioner) {}

    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    typedef int Index;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return op.rows(); }
    Index cols() const { return op.cols(); }

    template<typename Rhs>
    Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& z) const {
        return op * preconditioner.solve(z);
    }
};

struct HsLeftPreconditionedOperator
    : public Eigen::EigenBase<HsLeftPreconditionedOperator> {
    const HsOperator& op;
    const HsSandwichPreconditioner& preconditioner;

    HsLeftPreconditionedOperator(const HsOperator& op,
                                 const HsSandwichPreconditioner& preconditioner)
        : op(op), preconditioner(preconditioner) {}

    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    typedef int Index;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return op.rows(); }
    Index cols() const { return op.cols(); }

    template<typename Rhs>
    Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return preconditioner.solve(op * x);
    }
};

HsOperators build_hs_operators(const MeshData &mesh) {
    HsOperators out;
    const int nv = mesh.n_vertices();
    const int nf = mesh.n_faces();

    out.mass_diag = Eigen::VectorXd::Zero(nv);

    std::vector<Eigen::Triplet<double>> l_triplets;
    l_triplets.reserve(static_cast<size_t>(12 * std::max(0, nf)));

    auto add_edge_weight = [&](int a, int b, double cot) {
        const double w = 0.5 * cot;
        if (!std::isfinite(w) || w == 0.0) return;
        l_triplets.emplace_back(a, a, w);
        l_triplets.emplace_back(b, b, w);
        l_triplets.emplace_back(a, b, -w);
        l_triplets.emplace_back(b, a, -w);
    };

    for (int f = 0; f < nf; ++f) {
        const int i = mesh.F(f, 0);
        const int j = mesh.F(f, 1);
        const int k = mesh.F(f, 2);

        const Eigen::Vector3d vi = mesh.V.row(i).transpose();
        const Eigen::Vector3d vj = mesh.V.row(j).transpose();
        const Eigen::Vector3d vk = mesh.V.row(k).transpose();

        const double area = 0.5 * (vj - vi).cross(vk - vi).norm();
        const double lump = area / 3.0;
        out.mass_diag(i) += lump;
        out.mass_diag(j) += lump;
        out.mass_diag(k) += lump;

        const double cot_i = robust_cotangent(vj - vi, vk - vi);
        const double cot_j = robust_cotangent(vk - vj, vi - vj);
        const double cot_k = robust_cotangent(vi - vk, vj - vk);

        add_edge_weight(j, k, cot_i);
        add_edge_weight(k, i, cot_j);
        add_edge_weight(i, j, cot_k);
    }

    out.L.resize(nv, nv);
    out.L.setFromTriplets(
        l_triplets.begin(), l_triplets.end(),
        [](double a, double b) { return a + b; });
    out.L.makeCompressed();

    std::vector<Eigen::Triplet<double>> m_triplets;
    m_triplets.reserve(static_cast<size_t>(std::max(0, nv)));
    for (int i = 0; i < nv; ++i) {
        const double mii = out.mass_diag(i);
        if (mii != 0.0) {
            m_triplets.emplace_back(i, i, mii);
        }
    }

    out.M.resize(nv, nv);
    out.M.setFromTriplets(
        m_triplets.begin(), m_triplets.end(),
        [](double a, double b) { return a + b; });
    out.M.makeCompressed();

    std::vector<Eigen::Triplet<double>> m_full_triplets;
    m_full_triplets.reserve(static_cast<size_t>(9 * std::max(0, nf)));
    for (int f = 0; f < nf; ++f) {
        const int i = mesh.F(f, 0);
        const int j = mesh.F(f, 1);
        const int k = mesh.F(f, 2);
        const Eigen::Vector3d vi = mesh.V.row(i).transpose();
        const Eigen::Vector3d vj = mesh.V.row(j).transpose();
        const Eigen::Vector3d vk = mesh.V.row(k).transpose();
        const double area = 0.5 * (vj - vi).cross(vk - vi).norm();
        const int ids[3] = {i, j, k};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double val = area * (a == b ? 1.0 / 6.0 : 1.0 / 12.0);
                m_full_triplets.emplace_back(ids[a], ids[b], val);
            }
        }
    }

    out.M_full.resize(nv, nv);
    out.M_full.setFromTriplets(
        m_full_triplets.begin(), m_full_triplets.end(),
        [](double a, double b) { return a + b; });
    out.M_full.makeCompressed();

    return out;
}

Eigen::MatrixXd hs_apply_operator(
    const MeshData &mesh,
    const Eigen::MatrixXd &field,
    const HsPreconditionerParams &params) {
    if (field.rows() != mesh.n_vertices() || field.cols() != 3) {
        throw std::runtime_error(
            "hs_apply_operator: field must have shape (n_vertices x 3)");
    }

    const HsOperators hs = build_hs_operators(mesh);
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, params.theta);
    const HsOperator op(mesh, params, hs, g, bvh, bp);
    return op.apply(field);
}

HsDirectionResult hs_preconditioned_direction(
    const MeshData &mesh,
    const Eigen::MatrixXd &gradient,
    const HsPreconditionerParams &params,
    const HsConstraints &constraints) {
    const int nv = mesh.n_vertices();
    if (nv <= 0) {
        throw std::runtime_error("hs_preconditioned_direction: mesh has no vertices");
    }
    if (gradient.rows() != nv || gradient.cols() != 3) {
        throw std::runtime_error(
            "hs_preconditioned_direction: gradient must have shape (n_vertices x 3)");
    }
    if (!gradient.allFinite()) {
        throw std::runtime_error(
            "hs_preconditioned_direction: gradient contains NaN/Inf");
    }
    if (!std::isfinite(params.s) ||
        !std::isfinite(params.sigma) ||
        !std::isfinite(params.mass_weight) ||
        !std::isfinite(params.theta)) {
        throw std::runtime_error(
            "hs_preconditioned_direction: s, sigma, mass_weight, and theta must be finite");
    }
    if (params.s < 1.0 || params.s >= 2.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: s must be in [1, 2) for the RSu sandwich");
    }
    if (params.sigma < 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: sigma must be non-negative");
    }
    if (params.mass_weight < 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: mass_weight must be non-negative");
    }
    if (params.theta < 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: theta must be non-negative");
    }
    if (constraints.pin_mask != nullptr &&
        static_cast<int>(constraints.pin_mask->size()) != nv) {
        throw std::runtime_error(
            "hs_preconditioned_direction: pin mask size must match vertex count");
    }

    Eigen::MatrixXd constrained_gradient = gradient;
    apply_hs_constraints(constrained_gradient, constraints);

    const HsOperators hs = build_hs_operators(mesh);
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, params.theta);

    HsOperator op(mesh, params, hs, g, bvh, bp);
    ScalarFractionalLaplacian middle(mesh, g, bvh, bp, 2.0 - params.s);
    HsSandwichPreconditioner sandwich(
        hs,
        middle,
        HsLaplacianInverseMode::H1Metric,
        HsConstantProjectionMode::None);
    if (sandwich.info() != Eigen::Success) {
        throw std::runtime_error(
            "hs_preconditioned_direction: lifted cotan factorization failed");
    }

    HsLeftPreconditionedOperator left_op(op, sandwich);
    Eigen::GMRES<HsLeftPreconditionedOperator, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(left_op);
    gmres.setTolerance(1e-5);
    gmres.setMaxIterations(100);

    HsDirectionResult out;
    out.direction = Eigen::MatrixXd::Zero(nv, 3);
    out.max_gmres_iterations = 0;
    out.max_gmres_error = 0.0;
    out.used_identity_fallback = false;

    Eigen::VectorXd g_flat = flatten_vertex_field(constrained_gradient);
    sandwich.project(g_flat);
    const Eigen::VectorXd rhs_left = sandwich.solve(g_flat);
    Eigen::VectorXd d_flat = gmres.solve(rhs_left);
    sandwich.project(d_flat);
    out.direction = unflatten_vertex_field(d_flat, nv);
    apply_hs_constraints(out.direction, constraints);
    out.max_gmres_iterations = static_cast<int>(gmres.iterations());
    out.max_gmres_error = gmres.error();

    auto compute_g_dot_dir = [&]() {
        return (constrained_gradient.array() * out.direction.array()).sum();
    };

    out.g_dot_dir = out.direction.allFinite()
        ? compute_g_dot_dir()
        : std::numeric_limits<double>::quiet_NaN();

    if (!out.direction.allFinite() ||
        !std::isfinite(out.g_dot_dir) ||
        out.g_dot_dir <= 0.0) {
        std::cerr
            << "hs_preconditioned_direction: sandwich GMRES produced a "
            << "non-descent or invalid direction (g_dot_dir = "
            << out.g_dot_dir
            << "); falling back to identity-preconditioned GMRES\n";

        Eigen::GMRES<HsOperator, Eigen::IdentityPreconditioner> identity_gmres;
        identity_gmres.compute(op);
        identity_gmres.setTolerance(1e-6);
        identity_gmres.setMaxIterations(100);

        out.direction.setZero();
        out.max_gmres_iterations = 0;
        out.max_gmres_error = 0.0;
        out.used_identity_fallback = true;

        Eigen::VectorXd rhs_flat = flatten_vertex_field(constrained_gradient);
        sandwich.project(rhs_flat);
        Eigen::VectorXd fallback_flat = identity_gmres.solve(rhs_flat);
        sandwich.project(fallback_flat);
        out.direction = unflatten_vertex_field(fallback_flat, nv);
        apply_hs_constraints(out.direction, constraints);
        out.max_gmres_iterations = static_cast<int>(identity_gmres.iterations());
        out.max_gmres_error = identity_gmres.error();

        if (!out.direction.allFinite()) {
            throw std::runtime_error(
                "hs_preconditioned_direction: fallback direction contains NaN/Inf");
        }
        out.g_dot_dir = compute_g_dot_dir();
    }
    if (!std::isfinite(out.g_dot_dir)) {
        throw std::runtime_error("hs_preconditioned_direction: g_dot_dir is not finite");
    }
    if (out.g_dot_dir <= 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: g_dot_dir must be positive after fallback, got " +
            std::to_string(out.g_dot_dir));
    }
    return out;
}

} // namespace rsh
