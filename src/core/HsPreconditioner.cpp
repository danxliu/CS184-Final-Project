#include "HsPreconditioner.h"
#include "FaceGeom.h"
#include "BVH.h"
#include "BCT.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

namespace rsh {

// Forward declaration for traits specialization
struct HsOperator;

} // namespace rsh

namespace Eigen {
namespace internal {
template <>
struct traits<rsh::HsOperator> : public traits<Eigen::MatrixXd> {};
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

} // namespace

// Matrix-free operator for A = params.mass_weight * M + params.sigma * L^s
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
        const double power = params.s + 1.0;
        const double eps = 1e-2 * mesh.L0 * mesh.L0;

        // 1. Face values: x_a = mean(x_i for i in face a)
        Eigen::VectorXd face_x(nf);
        for (int i = 0; i < nf; ++i) {
            face_x(i) = (x(mesh.F(i, 0)) + x(mesh.F(i, 1)) + x(mesh.F(i, 2))) / 3.0;
        }

        // 2. Pre-aggregate Q_U = sum_{a in U} A_a x_a
        std::vector<double> Q(bvh.n_nodes(), 0.0);
        for (int i = bvh.n_nodes() - 1; i >= 0; --i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                double sum = 0.0;
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    sum += g.A(f) * face_x(f);
                }
                Q[i] = sum;
            } else {
                Q[i] = Q[node.left] + Q[node.right];
            }
        }

        // 3. Hierarchical integral for L^s x
        std::vector<double> SumA(bvh.n_nodes(), 0.0);
        std::vector<double> SumQ(bvh.n_nodes(), 0.0);
        for (const auto& cp : bp.admissible) {
            const BVHNode& U = bvh.nodes[cp.u];
            const BVHNode& V = bvh.nodes[cp.v];
            double r2 = (U.centroid - V.centroid).squaredNorm();
            double K = 1.0 / std::pow(r2 + eps, power);
            
            // Interaction U <- V (covers directed pair)
            SumA[cp.u] += K * V.area;
            SumQ[cp.u] += K * Q[cp.v];
        }

        // Propagate SumA/SumQ down the tree
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (!node.is_leaf()) {
                SumA[node.left] += SumA[i];
                SumQ[node.left] += SumQ[i];
                SumA[node.right] += SumA[i];
                SumQ[node.right] += SumQ[i];
            }
        }

        Eigen::VectorXd z = Eigen::VectorXd::Zero(nf);
        for (int i = 0; i < bvh.n_nodes(); ++i) {
            const BVHNode& node = bvh.nodes[i];
            if (node.is_leaf()) {
                for (int j = node.face_start; j < node.face_end; ++j) {
                    int f = bvh.face_indices[j];
                    z(f) = g.A(f) * (face_x(f) * SumA[i] - SumQ[i]);
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
                    double K = g.A(b) / std::pow(r2 + eps, power);
                    z(a) += g.A(a) * K * (face_x(a) - face_x(b));
                }
            }
        }

        // 4. Combine: y = params.mass_weight * M x + params.sigma * L^s x
        Eigen::VectorXd y = Eigen::VectorXd::Zero(nv);
        for (int i = 0; i < nv; ++i) {
            y(i) = params.mass_weight * hs.mass_diag(i) * x(i);
        }

        for (int f = 0; f < nf; ++f) {
            double val = params.sigma * z(f) / 3.0;
            y(mesh.F(f, 0)) += val;
            y(mesh.F(f, 1)) += val;
            y(mesh.F(f, 2)) += val;
        }

        return y;
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
    if (!std::isfinite(params.sigma) || !std::isfinite(params.mass_weight)) {
        throw std::runtime_error(
            "hs_preconditioned_direction: sigma and mass_weight must be finite");
    }
    if (params.sigma < 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: sigma must be non-negative");
    }
    if (params.mass_weight <= 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: mass_weight must be positive");
    }

    const HsOperators hs = build_hs_operators(mesh);
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, 0.5);

    HsOperator op(mesh, params, hs, g, bvh, bp);
    Eigen::GMRES<HsOperator, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(op);
    gmres.setTolerance(1e-6);
    gmres.setMaxIterations(100);

    HsDirectionResult out;
    out.direction = Eigen::MatrixXd::Zero(nv, 3);

    for (int j = 0; j < 3; ++j) {
        Eigen::VectorXd g_col = gradient.col(j);
        out.direction.col(j) = gmres.solve(g_col);

        if (gmres.info() != Eigen::Success) {
            // Fallback or warning could be added here
        }

        // Project out translation (ensure area-weighted mean is zero)
        const double total_mass = hs.mass_diag.sum();
        if (total_mass > 0.0) {
            double mean = (hs.mass_diag.array() * out.direction.col(j).array()).sum() / total_mass;
            out.direction.col(j).array() -= mean;
        }
    }

    if (!out.direction.allFinite()) {
        throw std::runtime_error(
            "hs_preconditioned_direction: direction contains NaN/Inf");
    }

    out.g_dot_dir = (gradient.array() * out.direction.array()).sum();
    if (!std::isfinite(out.g_dot_dir)) {
        throw std::runtime_error("hs_preconditioned_direction: g_dot_dir is not finite");
    }
    if (out.g_dot_dir <= 0.0) {
        throw std::runtime_error(
            "hs_preconditioned_direction: g_dot_dir must be positive, got " +
            std::to_string(out.g_dot_dir));
    }
    return out;
}

} // namespace rsh
