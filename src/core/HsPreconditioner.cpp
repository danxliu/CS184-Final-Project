#include "HsPreconditioner.h"
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

} // namespace rsh

namespace Eigen {
namespace internal {
template <>
struct traits<rsh::HsOperator> : public traits<Eigen::MatrixXd> {};
template <>
struct traits<rsh::HsRightPreconditionedOperator> : public traits<Eigen::MatrixXd> {};
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

Eigen::SparseMatrix<double> lifted_laplacian(const HsOperators &hs) {
    constexpr double kLift = 1e-12;
    Eigen::SparseMatrix<double> lifted = hs.L;
    lifted += kLift * hs.M;
    lifted.makeCompressed();
    return lifted;
}

} // namespace

// Matrix-free operator for A = sigma * (B + B_0), where B is the
// gradient-based high-order term from RSu Eq. 7 / Eq. 12 and B_0 is the
// low-order TPE-modulated term from RSu Eq. 8.
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

    Index rows() const { return mesh.n_vertices(); }
    Index cols() const { return mesh.n_vertices(); }

    template<typename Rhs>
    Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& x) const {
        const int nv = mesh.n_vertices();
        const int nf = mesh.n_faces();
        const double power = params.s;

        // 1. Per-face gradients D_f x for B and face barycenter values for B_0.
        std::vector<Eigen::Vector3d> grad_face(static_cast<size_t>(nf));
        Eigen::VectorXd face_x(nf);
        for (int i = 0; i < nf; ++i) {
            grad_face[static_cast<size_t>(i)] =
                face_scalar_gradient(mesh, g, x, i);
            face_x(i) = face_average(mesh, x, i);
        }

        // 2. Pre-aggregate Q_U = sum_{a in U} A_a D_f x(a).
        std::vector<Eigen::Vector3d> Q(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Vector3d::Zero());
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    sum += g.A(f) * grad_face[static_cast<size_t>(f)];
                }
                Q[i] = sum;
            } else {
                Q[i] = Q[node.left] + Q[node.right];
            }
        }

        // B_0 scalar aggregate Q0_U = sum_{a in U} A_a x_a.
        std::vector<double> Q0(static_cast<size_t>(bvh.n_nodes()), 0.0);
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                double sum = 0.0;
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    sum += g.A(f) * face_x(f);
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
        std::vector<Eigen::Vector3d> SumQ(
            static_cast<size_t>(bvh.n_nodes()), Eigen::Vector3d::Zero());
        std::vector<double> SumA0(bvh.n_nodes(), 0.0);
        std::vector<double> SumQ0(bvh.n_nodes(), 0.0);
        for (const auto& cp : bp.admissible) {
            const BVHNode& U = bvh.nodes[cp.u];
            const BVHNode& V = bvh.nodes[cp.v];
            double r2 = (U.centroid - V.centroid).squaredNorm();
            if (r2 == 0.0) continue;
            double K = 1.0 / std::pow(r2, power);
            
            // Interaction U <- V (covers directed pair)
            SumA[cp.u] += K * V.area;
            SumQ[cp.u] += K * Q[cp.v];

            const double K0 = b0_kernel_sym_clusters(U, V, power);
            SumA0[cp.u] += K0 * V.area;
            SumQ0[cp.u] += K0 * Q0[static_cast<size_t>(cp.v)];
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

        std::vector<Eigen::Vector3d> face_dual(
            static_cast<size_t>(nf), Eigen::Vector3d::Zero());
        Eigen::VectorXd face_dual0 = Eigen::VectorXd::Zero(nf);
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    face_dual[static_cast<size_t>(f)] +=
                        2.0 * g.A(f) *
                        (grad_face[static_cast<size_t>(f)] * SumA[i] - SumQ[i]);
                    face_dual0(f) +=
                        g.A(f) * (face_x(f) * SumA0[i] - SumQ0[i]);
                }
            }
        }

        // Near-field exact summation
        for (const auto& cp : bp.near_field) {
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
                    face_dual[static_cast<size_t>(a)] +=
                        2.0 * g.A(a) * g.A(b) * K *
                        (grad_face[static_cast<size_t>(a)] -
                         grad_face[static_cast<size_t>(b)]);

                    const double K0 = b0_kernel_sym_faces(g, a, b, power);
                    face_dual0(a) +=
                        g.A(a) * g.A(b) * K0 * (face_x(a) - face_x(b));
                }
            }
        }

        // 4. Combine: y = sigma * (B x + B_0 x).
        Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
        for (int f = 0; f < nf; ++f) {
            scatter_face_gradient_adjoint(
                mesh,
                g,
                f,
                params.sigma * face_dual[static_cast<size_t>(f)],
                y);
            const double b0_vertex = params.sigma * face_dual0(f) / 3.0;
            y(mesh.F(f, 0)) += b0_vertex;
            y(mesh.F(f, 1)) += b0_vertex;
            y(mesh.F(f, 2)) += b0_vertex;
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
        for (const auto& cp : bp.admissible) {
            const BVHNode& U = bvh.nodes[cp.u];
            const BVHNode& V = bvh.nodes[cp.v];
            const double r2 = (U.centroid - V.centroid).squaredNorm();
            if (r2 == 0.0) continue;
            const double K = 1.0 / std::pow(r2, power);
            SumA[cp.u] += K * V.area;
            SumQ[cp.u] += K * Q[static_cast<size_t>(cp.v)];
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

        for (const auto& cp : bp.near_field) {
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
                    face_dual(a) +=
                        2.0 * g.A(a) * g.A(b) * K * (face_x(a) - face_x(b));
                }
            }
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
    LaplaceBeltrami
};

enum class HsConstantProjectionMode {
    Algebraic,
    MassWeighted
};

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
        const Eigen::SparseMatrix<double> lifted = lifted_laplacian(hs);
        laplacian_factor.compute(lifted);
        factor_info = laplacian_factor.info();
    }

    void project(Eigen::VectorXd &x) const {
        if (projection_mode == HsConstantProjectionMode::MassWeighted) {
            project_mass_constant_mode(hs, x);
        } else {
            project_constant_mode(x);
        }
    }

    Eigen::VectorXd laplace_inverse(Eigen::VectorXd rhs) const {
        if (inverse_mode == HsLaplacianInverseMode::LaplaceBeltrami) {
            rhs = apply_lumped_mass(hs, rhs);
        }
        // The lifted factor can solve a nonzero-sum RHS, but projecting the
        // RHS keeps the solve aligned with the cotan nullspace we intend to
        // quotient out.
        project(rhs);
        Eigen::VectorXd out = laplacian_factor.solve(rhs);
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

        Eigen::VectorXd w2 = middle.apply(w1);
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

    return out;
}

HsDirectionResult hs_preconditioned_direction(
    const MeshData &mesh,
    const Eigen::MatrixXd &gradient,
    const HsPreconditionerParams &params) {
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

    const HsOperators hs = build_hs_operators(mesh);
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, params.theta);

    HsOperator op(mesh, params, hs, g, bvh, bp);
    ScalarFractionalLaplacian middle(mesh, g, bvh, bp, 2.0 - params.s);
    HsSandwichPreconditioner sandwich(hs, middle);
    if (sandwich.info() != Eigen::Success) {
        throw std::runtime_error(
            "hs_preconditioned_direction: lifted cotan factorization failed");
    }

    HsRightPreconditionedOperator right_op(op, sandwich);
    Eigen::GMRES<HsRightPreconditionedOperator, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(right_op);
    gmres.setTolerance(1e-6);
    gmres.setMaxIterations(100);

    HsDirectionResult out;
    out.direction = Eigen::MatrixXd::Zero(nv, 3);
    out.max_gmres_iterations = 0;
    out.max_gmres_error = 0.0;
    out.used_identity_fallback = false;

    for (int j = 0; j < 3; ++j) {
        Eigen::VectorXd g_col = gradient.col(j);
        sandwich.project(g_col);

        const Eigen::VectorXd z_col = gmres.solve(g_col);
        out.direction.col(j) = sandwich.solve(z_col);
        out.max_gmres_iterations =
            std::max(out.max_gmres_iterations,
                     static_cast<int>(gmres.iterations()));
        out.max_gmres_error = std::max(out.max_gmres_error, gmres.error());

        if (gmres.info() != Eigen::Success) {
            // Fallback or warning could be added here
        }

        // Project out the chosen constant mode of A.
        Eigen::VectorXd d_col = out.direction.col(j);
        sandwich.project(d_col);
        out.direction.col(j) = d_col;
    }

    auto compute_g_dot_dir = [&]() {
        return (gradient.array() * out.direction.array()).sum();
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

        for (int j = 0; j < 3; ++j) {
            Eigen::VectorXd g_col = gradient.col(j);
            sandwich.project(g_col);

            Eigen::VectorXd d_col = identity_gmres.solve(g_col);
            sandwich.project(d_col);
            out.direction.col(j) = d_col;

            out.max_gmres_iterations =
                std::max(out.max_gmres_iterations,
                         static_cast<int>(identity_gmres.iterations()));
            out.max_gmres_error =
                std::max(out.max_gmres_error, identity_gmres.error());
        }

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
