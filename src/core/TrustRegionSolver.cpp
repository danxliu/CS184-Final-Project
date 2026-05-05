#include "TrustRegionSolver.h"

#include "PathEnergy.h"
#include "ShellEnergy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/SparseCholesky>

namespace rsh {

namespace {

int interior_dim(const std::vector<MeshData> &frames, const TrustRegionParams& tr_params) {
    const int n = static_cast<int>(frames.size()) - 1;
    if (n < 1) return 0;
    int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    if (opt_frames < 1) return 0;
    int num_free = 0;
    if (tr_params.free_vertices.empty()) {
        num_free = frames[0].n_vertices();
    } else {
        for (bool is_free : tr_params.free_vertices) {
            if (is_free) num_free++;
        }
    }
    return std::max(0, opt_frames * num_free * 3);
}

Eigen::VectorXd pack_interior_frames(const std::vector<MeshData> &frames, const TrustRegionParams& tr_params) {
    const int n = static_cast<int>(frames.size()) - 1;
    const int nv = frames[0].n_vertices();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(interior_dim(frames, tr_params));
    int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    int off = 0;
    for (int k = 1; k <= opt_frames; ++k) {
        for (int i = 0; i < nv; ++i) {
            if (tr_params.free_vertices.empty() || tr_params.free_vertices[i]) {
                z.segment<3>(off) = frames[k].V.row(i).transpose();
                off += 3;
            }
        }
    }
    return z;
}

void unpack_interior_frames(const Eigen::VectorXd &z, std::vector<MeshData> &frames, const TrustRegionParams& tr_params) {
    const int n = static_cast<int>(frames.size()) - 1;
    const int nv = frames[0].n_vertices();
    int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    int off = 0;
    for (int k = 1; k <= opt_frames; ++k) {
        for (int i = 0; i < nv; ++i) {
            if (tr_params.free_vertices.empty() || tr_params.free_vertices[i]) {
                frames[k].V.row(i) = z.segment<3>(off).transpose();
                off += 3;
            }
        }
    }
}

Eigen::VectorXd pack_interior_gradient(const PathEnergyGradientResult &g, const TrustRegionParams& tr_params) {
    const int n = static_cast<int>(g.grad_frames.size()) - 1;
    const int nv = static_cast<int>(g.grad_frames[0].rows());
    int num_free = 0;
    if (tr_params.free_vertices.empty()) {
        num_free = nv;
    } else {
        for (bool is_free : tr_params.free_vertices) if (is_free) num_free++;
    }
    int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    Eigen::VectorXd out = Eigen::VectorXd::Zero(std::max(0, opt_frames * num_free * 3));
    int off = 0;
    for (int k = 1; k <= opt_frames; ++k) {
        for (int i = 0; i < nv; ++i) {
            if (tr_params.free_vertices.empty() || tr_params.free_vertices[i]) {
                out.segment<3>(off) = g.grad_frames[k].row(i).transpose();
                off += 3;
            }
        }
    }
    return out;
}

double norm2(const Eigen::VectorXd &x) { return x.squaredNorm(); }

std::vector<int> free_vertex_indices(int nv,
                                     const TrustRegionParams &tr_params) {
    if (!tr_params.free_vertices.empty() &&
        static_cast<int>(tr_params.free_vertices.size()) != nv) {
        throw std::runtime_error(
            "trust-region free_vertices size must match vertex count");
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(nv));
    for (int i = 0; i < nv; ++i) {
        if (tr_params.free_vertices.empty() || tr_params.free_vertices[i]) {
            out.push_back(i);
        }
    }
    return out;
}

std::vector<int> global_dof_to_free_map(int nv,
                                        const std::vector<int> &free_vertices) {
    std::vector<int> map(static_cast<size_t>(3 * nv), -1);
    for (int slot = 0; slot < static_cast<int>(free_vertices.size()); ++slot) {
        const int vi = free_vertices[static_cast<size_t>(slot)];
        for (int c = 0; c < 3; ++c) {
            map[static_cast<size_t>(3 * vi + c)] = 3 * slot + c;
        }
    }
    return map;
}

int optimized_frame_start(int frame,
                          int block_dim,
                          int n,
                          const TrustRegionParams &tr_params) {
    const int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    if (frame < 1 || frame > opt_frames) {
        return -1;
    }
    return (frame - 1) * block_dim;
}

int interior_dof_index(int frame,
                       int global_dof,
                       int block_dim,
                       int n,
                       const std::vector<int> &dof_to_free,
                       const TrustRegionParams &tr_params) {
    const int frame_start =
        optimized_frame_start(frame, block_dim, n, tr_params);
    if (frame_start < 0 ||
        global_dof < 0 ||
        global_dof >= static_cast<int>(dof_to_free.size())) {
        return -1;
    }
    const int local = dof_to_free[static_cast<size_t>(global_dof)];
    if (local < 0) {
        return -1;
    }
    return frame_start + local;
}

void scatter_sparse_block(
    const Eigen::SparseMatrix<double> &block,
    int row_frame,
    int col_frame,
    double scale,
    int block_dim,
    int n,
    const std::vector<int> &dof_to_free,
    const TrustRegionParams &tr_params,
    std::vector<Eigen::Triplet<double>> &triplets) {
    for (int outer = 0; outer < block.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(block, outer); it;
             ++it) {
            const int row = interior_dof_index(
                row_frame, it.row(), block_dim, n, dof_to_free, tr_params);
            const int col = interior_dof_index(
                col_frame, it.col(), block_dim, n, dof_to_free, tr_params);
            if (row >= 0 && col >= 0) {
                triplets.emplace_back(row, col, scale * it.value());
            }
        }
    }
}

Eigen::SparseMatrix<double>
extract_free_block(const Eigen::SparseMatrix<double> &full,
                   const std::vector<int> &dof_to_free,
                   int block_dim,
                   double scale) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(full.nonZeros()));
    for (int outer = 0; outer < full.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(full, outer); it;
             ++it) {
            const int row = dof_to_free[static_cast<size_t>(it.row())];
            const int col = dof_to_free[static_cast<size_t>(it.col())];
            if (row >= 0 && col >= 0) {
                triplets.emplace_back(row, col, scale * it.value());
            }
        }
    }
    Eigen::SparseMatrix<double> out(block_dim, block_dim);
    out.setFromTriplets(
        triplets.begin(), triplets.end(),
        [](double a, double b) { return a + b; });
    out.makeCompressed();
    return out;
}

Eigen::SparseMatrix<double>
build_shell_path_hessian(const std::vector<MeshData> &frames,
                         const PathEnergyParams &energy_params,
                         const TrustRegionParams &tr_params) {
    const int n = static_cast<int>(frames.size()) - 1;
    const int nv = frames[0].n_vertices();
    const std::vector<int> free_vertices =
        free_vertex_indices(nv, tr_params);
    const int block_dim = static_cast<int>(free_vertices.size()) * 3;
    const int dim = interior_dim(frames, tr_params);
    const std::vector<int> dof_to_free =
        global_dof_to_free_map(nv, free_vertices);
    std::vector<Eigen::Triplet<double>> triplets;
    const double scale = static_cast<double>(n);

    for (int k = 1; k <= n; ++k) {
        const ShellEnergyHessianResult H =
            shell_energy_hessian(frames[static_cast<size_t>(k - 1)],
                                 frames[static_cast<size_t>(k)],
                                 energy_params.shell);
        scatter_sparse_block(H.ref_ref, k - 1, k - 1, scale, block_dim, n,
                             dof_to_free, tr_params, triplets);
        scatter_sparse_block(H.ref_def, k - 1, k, scale, block_dim, n,
                             dof_to_free, tr_params, triplets);
        scatter_sparse_block(H.def_ref, k, k - 1, scale, block_dim, n,
                             dof_to_free, tr_params, triplets);
        scatter_sparse_block(H.def_def, k, k, scale, block_dim, n,
                             dof_to_free, tr_params, triplets);
    }

    Eigen::SparseMatrix<double> H(dim, dim);
    H.setFromTriplets(
        triplets.begin(), triplets.end(),
        [](double a, double b) { return a + b; });
    H.makeCompressed();
    Eigen::SparseMatrix<double> Ht = H.transpose();
    Eigen::SparseMatrix<double> sym = 0.5 * (H + Ht);
    sym.makeCompressed();
    return sym;
}

struct FrameBlockPreconditioner {
    int start = 0;
    int dim = 0;
    bool valid = false;
    double diagonal_shift = 0.0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
};

struct BlockDiagonalPreconditioner {
    int dim = 0;
    bool enabled = false;
    std::vector<std::unique_ptr<FrameBlockPreconditioner>> blocks;

    Eigen::VectorXd apply_inverse(const Eigen::VectorXd &r) const {
        if (!enabled || r.size() != dim) {
            return r;
        }

        Eigen::VectorXd z = r;
        for (const auto &block_ptr : blocks) {
            const FrameBlockPreconditioner &block = *block_ptr;
            if (!block.valid || block.dim <= 0) {
                continue;
            }
            const Eigen::VectorXd rhs = r.segment(block.start, block.dim);
            Eigen::VectorXd solved = block.factor.solve(rhs);
            if (block.factor.info() != Eigen::Success ||
                solved.size() != block.dim ||
                !solved.allFinite() ||
                rhs.dot(solved) <= 0.0) {
                continue;
            }
            z.segment(block.start, block.dim) = solved;
        }
        return z;
    }
};

std::unique_ptr<BlockDiagonalPreconditioner>
build_block_diagonal_preconditioner(const std::vector<MeshData> &frames,
                                    const PathEnergyParams &energy_params,
                                    const TrustRegionParams &tr_params) {
    auto precond = std::make_unique<BlockDiagonalPreconditioner>();
    precond->dim = interior_dim(frames, tr_params);
    precond->enabled = false;
    if (!tr_params.use_block_diagonal_preconditioner ||
        precond->dim == 0 ||
        !(tr_params.block_preconditioner_regularization > 0.0)) {
        return precond;
    }

    const int n = static_cast<int>(frames.size()) - 1;
    const int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    const int nv = frames[0].n_vertices();
    const std::vector<int> free_vertices =
        free_vertex_indices(nv, tr_params);
    const int block_dim = static_cast<int>(free_vertices.size()) * 3;
    const std::vector<int> dof_to_free =
        global_dof_to_free_map(nv, free_vertices);
    if (block_dim == 0 ||
        block_dim > tr_params.block_preconditioner_max_block_dofs) {
        if (block_dim > tr_params.block_preconditioner_max_block_dofs) {
            std::cout
                << "[TrustRegion] Skipping RS block preconditioner: block dim "
                << block_dim << " exceeds limit "
                << tr_params.block_preconditioner_max_block_dofs << "\n";
        }
        return precond;
    }

    const double path_scale = static_cast<double>(n);
    int start = 0;
    precond->blocks.reserve(static_cast<size_t>(std::max(0, opt_frames)));

    for (int k = 1; k <= opt_frames; ++k) {
        const MeshData &rest = frames[static_cast<size_t>(k)];
        const Eigen::SparseMatrix<double> Hdef =
            shell_energy_def_hessian(rest, rest, energy_params.shell);
        const Eigen::SparseMatrix<double> H =
            extract_free_block(Hdef, dof_to_free, block_dim, path_scale);
        Eigen::SparseMatrix<double> Ht = H.transpose();
        Eigen::SparseMatrix<double> sym = 0.5 * (H + Ht);
        sym.makeCompressed();

        double diag_scale = 1.0;
        for (int i = 0; i < block_dim; ++i) {
            diag_scale = std::max(diag_scale, std::abs(sym.coeff(i, i)));
        }

        auto block = std::make_unique<FrameBlockPreconditioner>();
        block->start = start;
        block->dim = block_dim;

        double shift = std::max(
            tr_params.block_preconditioner_regularization * diag_scale,
            tr_params.block_preconditioner_regularization_floor);
        bool factored = false;
        for (int attempt = 0; attempt < 8 && !factored; ++attempt) {
            Eigen::SparseMatrix<double> shifted = sym;
            std::vector<Eigen::Triplet<double>> diag;
            diag.reserve(static_cast<size_t>(block_dim));
            for (int i = 0; i < block_dim; ++i) {
                diag.emplace_back(i, i, shift);
            }
            Eigen::SparseMatrix<double> D(block_dim, block_dim);
            D.setFromTriplets(diag.begin(), diag.end());
            shifted += D;
            shifted.makeCompressed();

            block->factor.compute(shifted);
            if (block->factor.info() == Eigen::Success) {
                Eigen::VectorXd probe =
                    block->factor.solve(Eigen::VectorXd::Ones(block_dim));
                factored =
                    block->factor.info() == Eigen::Success &&
                    probe.allFinite() &&
                    probe.dot(Eigen::VectorXd::Ones(block_dim)) > 0.0;
            }
            if (!factored) {
                shift *= 10.0;
            }
        }

        block->valid = factored;
        block->diagonal_shift = shift;
        if (!block->valid) {
            std::cout
                << "[TrustRegion] RS block preconditioner frame " << k
                << " factorization failed; using identity for this block\n";
        }
        precond->blocks.push_back(std::move(block));
        start += block_dim;
    }

    precond->enabled = true;
    std::cout << "[TrustRegion] Built RS block-diagonal CG preconditioner: "
              << precond->blocks.size() << " blocks, block_dim="
              << block_dim << "\n";
    return precond;
}

struct ModelEval {
    double energy = 0.0;
    Eigen::VectorXd grad;
    std::vector<double> phi_per_frame;
    std::vector<Eigen::MatrixXd> grad_phi_per_frame;
};

ModelEval eval_model(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &energy_params,
    const TrustRegionParams& tr_params,
    const std::vector<PathEnergyFrameCache> &frame_cache) {
    ModelEval out;
    const PathEnergyGradientResult pr =
        path_energy_with_gradient(frames, energy_params, &frame_cache);
    out.energy = pr.energy.terms.total;
    out.grad = pack_interior_gradient(pr, tr_params);
    out.phi_per_frame = pr.energy.phi_per_frame;
    out.grad_phi_per_frame = pr.grad_phi_per_frame;
    return out;
}

Eigen::VectorXd hvp(const std::vector<MeshData> &frames,
                    const PathEnergyParams &energy_params,
                    const TrustRegionParams& tr_params,
                    const Eigen::SparseMatrix<double> &shell_hessian,
                    const std::vector<PathEnergyFrameCache> &frame_cache,
                    const ModelEval &cur,
                    const Eigen::VectorXd &v) {
    if (v.size() == 0) return Eigen::VectorXd();
    Eigen::VectorXd Hs = shell_hessian * v;

    const int n = static_cast<int>(frames.size()) - 1;
    const double scale = static_cast<double>(n);
    const int nv = frames[0].n_vertices();
    int opt_frames = tr_params.optimize_end_frame ? n : (n - 1);
    
    std::vector<double> w(static_cast<size_t>(n + 1), 0.0);
    std::vector<Eigen::MatrixXd> v_frames(static_cast<size_t>(n + 1), Eigen::MatrixXd::Zero(nv, 3));
    int off = 0;
    for (int k = 1; k <= opt_frames; ++k) {
        for (int i = 0; i < nv; ++i) {
            if (tr_params.free_vertices.empty() || tr_params.free_vertices[i]) {
                v_frames[static_cast<size_t>(k)].row(i) = v.segment<3>(off).transpose();
                off += 3;
            }
        }
    }
    
    for (int k = 1; k <= n; ++k) {
        const int km1 = k - 1;
        double dot_km1 = 0.0;
        double dot_k = 0.0;
        if (km1 > 0 && km1 <= opt_frames) {
            dot_km1 = (cur.grad_phi_per_frame[static_cast<size_t>(km1)].array() * v_frames[static_cast<size_t>(km1)].array()).sum();
        }
        if (k > 0 && k <= opt_frames) {
            dot_k = (cur.grad_phi_per_frame[static_cast<size_t>(k)].array() * v_frames[static_cast<size_t>(k)].array()).sum();
        }
        w[static_cast<size_t>(k)] = dot_km1 - dot_k;
    }
    
    std::vector<Eigen::MatrixXd> h_gn_frames(static_cast<size_t>(n + 1), Eigen::MatrixXd::Zero(nv, 3));
    for (int i = 1; i <= opt_frames; ++i) {
        double w_next = (i < n) ? w[static_cast<size_t>(i + 1)] : 0.0;
        double w_curr = w[static_cast<size_t>(i)];
        h_gn_frames[static_cast<size_t>(i)] = cur.grad_phi_per_frame[static_cast<size_t>(i)] * (w_next - w_curr);
    }
    
    Eigen::VectorXd h_gn_vec = Eigen::VectorXd::Zero(v.size());
    off = 0;
    for (int i = 1; i <= opt_frames; ++i) {
        for (int j = 0; j < nv; ++j) {
            if (tr_params.free_vertices.empty() || tr_params.free_vertices[j]) {
                h_gn_vec.segment<3>(off) = h_gn_frames[static_cast<size_t>(i)].row(j).transpose();
                off += 3;
            }
        }
    }
    
    Hs += scale * energy_params.graph_beta * 2.0 * h_gn_vec;

    if (tr_params.use_graph_residual_hessian &&
        energy_params.graph_beta != 0.0 &&
        tr_params.graph_residual_hvp_fd_eps > 0.0) {
        double x_norm_sq = 0.0;
        for (int k = 1; k <= opt_frames; ++k) {
            x_norm_sq += frames[static_cast<size_t>(k)].V.squaredNorm();
        }
        const double v_norm = v.norm();
        if (v_norm > 0.0) {
            const double h =
                tr_params.graph_residual_hvp_fd_eps *
                (1.0 + std::sqrt(x_norm_sq)) / v_norm;
            if (h > 0.0 && std::isfinite(h)) {
                std::vector<MeshData> plus = frames;
                std::vector<MeshData> minus = frames;
                for (int k = 1; k <= opt_frames; ++k) {
                    plus[static_cast<size_t>(k)].V +=
                        h * v_frames[static_cast<size_t>(k)];
                    minus[static_cast<size_t>(k)].V -=
                        h * v_frames[static_cast<size_t>(k)];
                }
                const PathEnergyGradientResult gp =
                    path_energy_with_gradient(plus, energy_params,
                                              &frame_cache);
                const PathEnergyGradientResult gm =
                    path_energy_with_gradient(minus, energy_params,
                                              &frame_cache);

                std::vector<double> dphi(static_cast<size_t>(n + 1), 0.0);
                for (int k = 1; k <= n; ++k) {
                    dphi[static_cast<size_t>(k)] =
                        cur.phi_per_frame[static_cast<size_t>(k - 1)] -
                        cur.phi_per_frame[static_cast<size_t>(k)];
                }

                std::vector<Eigen::MatrixXd> h_res_frames(
                    static_cast<size_t>(n + 1),
                    Eigen::MatrixXd::Zero(nv, 3));
                const double inv_2h = 0.5 / h;
                for (int i = 1; i <= opt_frames; ++i) {
                    const double d_next =
                        (i < n) ? dphi[static_cast<size_t>(i + 1)] : 0.0;
                    const double d_curr = dphi[static_cast<size_t>(i)];
                    const double residual_coeff = d_next - d_curr;
                    if (residual_coeff == 0.0) {
                        continue;
                    }
                    h_res_frames[static_cast<size_t>(i)] =
                        residual_coeff * inv_2h *
                        (gp.grad_phi_per_frame[static_cast<size_t>(i)] -
                         gm.grad_phi_per_frame[static_cast<size_t>(i)]);
                }

                Eigen::VectorXd h_res_vec = Eigen::VectorXd::Zero(v.size());
                off = 0;
                for (int i = 1; i <= opt_frames; ++i) {
                    for (int j = 0; j < nv; ++j) {
                        if (tr_params.free_vertices.empty() ||
                            tr_params.free_vertices[j]) {
                            h_res_vec.segment<3>(off) =
                                h_res_frames[static_cast<size_t>(i)]
                                    .row(j)
                                    .transpose();
                            off += 3;
                        }
                    }
                }
                Hs += scale * energy_params.graph_beta * 2.0 * h_res_vec;
            }
        }
    }

    if (energy_params.rigid_translation_weight > 0.0 ||
        energy_params.rigid_rotation_weight > 0.0) {
        std::vector<Eigen::MatrixXd> h_rigid_frames(
            static_cast<size_t>(n + 1), Eigen::MatrixXd::Zero(nv, 3));

        if (energy_params.rigid_translation_weight > 0.0) {
            const double coeff =
                scale * energy_params.rigid_translation_weight * 2.0 /
                static_cast<double>(nv);
            for (int k = 1; k <= n; ++k) {
                const Eigen::Vector3d dprev =
                    v_frames[static_cast<size_t>(k - 1)]
                        .colwise()
                        .mean()
                        .transpose();
                const Eigen::Vector3d dnext =
                    v_frames[static_cast<size_t>(k)]
                        .colwise()
                        .mean()
                        .transpose();
                const Eigen::RowVector3d row =
                    (coeff * (dnext - dprev)).transpose();
                for (int i = 0; i < nv; ++i) {
                    h_rigid_frames[static_cast<size_t>(k)].row(i) += row;
                    h_rigid_frames[static_cast<size_t>(k - 1)].row(i) -= row;
                }
            }
        }

        if (energy_params.rigid_rotation_weight > 0.0) {
            // HVP for E_rot = w·‖L‖² with L = Σ_v b_v × a_v (RS Eq. 27).
            // δL = Σ_v (db × a + b × da). Then
            //   H_a v = 2w · (δL × b + L × db),
            //   H_b v = 2w · (da × L + a × δL).
            const double coeff =
                scale * energy_params.rigid_rotation_weight * 2.0;
            for (int k = 1; k <= n; ++k) {
                const MeshData &prev = frames[static_cast<size_t>(k - 1)];
                const MeshData &next = frames[static_cast<size_t>(k)];
                const Eigen::MatrixXd &vprev =
                    v_frames[static_cast<size_t>(k - 1)];
                const Eigen::MatrixXd &vnext =
                    v_frames[static_cast<size_t>(k)];
                Eigen::Vector3d L = Eigen::Vector3d::Zero();
                Eigen::Vector3d dL = Eigen::Vector3d::Zero();
                for (int i = 0; i < nv; ++i) {
                    const Eigen::Vector3d a = prev.V.row(i).transpose();
                    const Eigen::Vector3d b = next.V.row(i).transpose();
                    const Eigen::Vector3d da = vprev.row(i).transpose();
                    const Eigen::Vector3d db = vnext.row(i).transpose();
                    L += b.cross(a);
                    dL += db.cross(a) + b.cross(da);
                }
                for (int i = 0; i < nv; ++i) {
                    const Eigen::Vector3d a = prev.V.row(i).transpose();
                    const Eigen::Vector3d b = next.V.row(i).transpose();
                    const Eigen::Vector3d da = vprev.row(i).transpose();
                    const Eigen::Vector3d db = vnext.row(i).transpose();
                    const Eigen::Vector3d ha =
                        coeff * (dL.cross(b) + L.cross(db));
                    const Eigen::Vector3d hb =
                        coeff * (da.cross(L) + a.cross(dL));
                    h_rigid_frames[static_cast<size_t>(k - 1)].row(i) +=
                        ha.transpose();
                    h_rigid_frames[static_cast<size_t>(k)].row(i) +=
                        hb.transpose();
                }
            }
        }

        Eigen::VectorXd h_rigid_vec = Eigen::VectorXd::Zero(v.size());
        off = 0;
        for (int i = 1; i <= opt_frames; ++i) {
            for (int j = 0; j < nv; ++j) {
                if (tr_params.free_vertices.empty() ||
                    tr_params.free_vertices[j]) {
                    h_rigid_vec.segment<3>(off) =
                        h_rigid_frames[static_cast<size_t>(i)]
                            .row(j)
                            .transpose();
                    off += 3;
                }
            }
        }
        Hs += h_rigid_vec;
    }
    return Hs;
}

Eigen::VectorXd steihaug_cg(const std::vector<MeshData> &frames,
                            const PathEnergyParams &energy_params,
                            const TrustRegionParams& tr_params,
                            const Eigen::SparseMatrix<double> &shell_hessian,
                            const std::vector<PathEnergyFrameCache> &frame_cache,
                            const ModelEval &cur,
                            const Eigen::VectorXd &g,
                            const BlockDiagonalPreconditioner *precond,
                            double radius) {
    const int dim = g.size();
    Eigen::VectorXd s = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd r = g;
    if (dim == 0) return s;
    const double r0 = r.norm();
    if (r0 < tr_params.grad_tol) return s;

    auto apply_precond = [&](const Eigen::VectorXd &q) {
        if (precond == nullptr) {
            return q;
        }
        return precond->apply_inverse(q);
    };

    Eigen::VectorXd z = apply_precond(r);
    double rz = r.dot(z);
    if (!std::isfinite(rz) || rz <= 0.0 || !z.allFinite()) {
        std::cout
            << "  [steihaug_cg] Block preconditioner produced invalid "
               "initial residual; falling back to identity CG\n";
        z = r;
        rz = r.dot(r);
    }
    const double rz0 = std::max(rz, 0.0);
    Eigen::VectorXd d = -z;
    if (d.norm() == 0.0) return s;

    auto tau_to_boundary = [&](const Eigen::VectorXd &s0, const Eigen::VectorXd &d0) {
        const double a = d0.squaredNorm();
        if (!(a > 0.0)) return 0.0;
        const double b = 2.0 * s0.dot(d0);
        const double c = s0.squaredNorm() - radius * radius;
        const double disc = std::max(0.0, b * b - 4.0 * a * c);
        return (-b + std::sqrt(disc)) / (2.0 * a);
    };

    for (int it = 0; it < tr_params.max_cg_iters; ++it) {
        std::cout << "  [steihaug_cg] Iteration " << it + 1 << "/" << tr_params.max_cg_iters << "..." << std::endl;
        const Eigen::VectorXd Hd =
            hvp(frames, energy_params, tr_params, shell_hessian, frame_cache,
                cur, d);
        const double dHd = d.dot(Hd);
        if (dHd <= 0.0) {
            const double tau = tau_to_boundary(s, d);
            std::cout << "  [steihaug_cg] Finished (negative curvature). tau: " << tau << std::endl;
            return s + tau * d;
        }
        const double alpha = rz / dHd;
        const Eigen::VectorXd s_next = s + alpha * d;
        if (s_next.norm() >= radius) {
            const double tau = tau_to_boundary(s, d);
            std::cout << "  [steihaug_cg] Finished (reached boundary). tau: " << tau << std::endl;
            return s + tau * d;
        }
        s = s_next;
        const Eigen::VectorXd r_next = r + alpha * Hd;
        Eigen::VectorXd z_next = apply_precond(r_next);
        double rz_next = r_next.dot(z_next);
        if (!std::isfinite(rz_next) || rz_next <= 0.0 ||
            !z_next.allFinite()) {
            z_next = r_next;
            rz_next = r_next.dot(r_next);
        }
        const double precond_res =
            std::sqrt(std::max(0.0, rz_next));
        const double raw_res = r_next.norm();
        const double raw_rel = raw_res / std::max(1.0, r0);
        const double precond_rel =
            precond_res /
            std::max(1.0, std::sqrt(std::max(0.0, rz0)));
        if (raw_rel <= tr_params.cg_tol ||
            (precond_rel <= tr_params.cg_tol &&
             raw_rel <= std::sqrt(tr_params.cg_tol))) {
            std::cout << "  [steihaug_cg] Finished (converged). raw residual: "
                      << raw_res << ", raw rel: " << raw_rel
                      << ", precond rel: " << precond_rel << std::endl;
            return s;
        }
        const double beta = rz_next / rz;
        d = -z_next + beta * d;
        r = r_next;
        z = z_next;
        rz = rz_next;
    }
    std::cout << "  [steihaug_cg] Finished (max iterations reached)." << std::endl;
    return s;
}

} // namespace

TrustRegionResult interpolate_geodesic_trust_region(
    const std::vector<MeshData> &initial_frames,
    const PathEnergyParams &energy_params,
    const TrustRegionParams &tr_params) {
    if (initial_frames.size() < 2) {
        throw std::runtime_error(
            "interpolate_geodesic_trust_region: need at least 2 frames");
    }
    TrustRegionResult out;
    out.frames = initial_frames;
    const int dim = interior_dim(out.frames, tr_params);
    if (dim == 0) {
        out.converged = true;
        out.accepted_energy.push_back(path_energy(out.frames, energy_params).terms.total);
        return out;
    }

    Eigen::VectorXd x = pack_interior_frames(out.frames, tr_params);
    double radius = std::max(1e-10, tr_params.initial_radius);
    // Freeze the hierarchical TPE partition for the local trust-region model.
    // Rebuild only after an accepted step; trial rho checks must use the same
    // local approximation as the current gradient/HVP.
    std::vector<PathEnergyFrameCache> model_cache =
        build_path_energy_frame_cache(out.frames, energy_params);
    ModelEval cur =
        eval_model(out.frames, energy_params, tr_params, model_cache);
    out.accepted_energy.push_back(cur.energy);
    std::unique_ptr<BlockDiagonalPreconditioner> block_precond;
    Eigen::SparseMatrix<double> shell_hessian;
    bool block_precond_dirty = true;

    for (int it = 0; it < tr_params.max_iters; ++it) {
        std::cout << "[TrustRegion] Starting iteration " << it + 1 << "/" << tr_params.max_iters << "..." << std::endl;
        auto it_start = std::chrono::high_resolution_clock::now();
        out.outer_iterations = it + 1;
        const double gnorm = cur.grad.norm();
        if (gnorm < tr_params.grad_tol) {
            out.converged = true;
            if (tr_params.iteration_callback) {
                TrustRegionIterationInfo info;
                info.iteration = it + 1;
                info.energy = cur.energy;
                info.trial_energy = cur.energy;
                info.grad_norm = gnorm;
                info.radius = radius;
                info.converged = true;
                tr_params.iteration_callback(info, out.frames);
            }
            break;
        }

        if (block_precond_dirty) {
            shell_hessian =
                build_shell_path_hessian(out.frames, energy_params, tr_params);
            block_precond =
                build_block_diagonal_preconditioner(
                    out.frames, energy_params, tr_params);
            block_precond_dirty = false;
        }

        const Eigen::VectorXd s =
            steihaug_cg(out.frames, energy_params, tr_params, shell_hessian,
                        model_cache, cur, cur.grad, block_precond.get(),
                        radius);
        if (s.norm() == 0.0) {
            radius *= 0.5;
            if (radius < 1e-12) break;
            continue;
        }

        const Eigen::VectorXd Hs =
            hvp(out.frames, energy_params, tr_params, shell_hessian,
                model_cache, cur, s);
        const double g_dot_s = cur.grad.dot(s);
        const double sHs = s.dot(Hs);

        bool accepted = false;
        std::vector<MeshData> accepted_trial;
        Eigen::VectorXd accepted_x_trial;
        double accepted_alpha = 0.0;
        double accepted_step_norm = s.norm();
        double e_trial = std::numeric_limits<double>::infinity();
        double rho = -1.0;
        const int max_backtracks =
            std::max(0, tr_params.max_step_backtracks);
        const double shrink =
            (tr_params.step_backtrack_shrink > 0.0 &&
             tr_params.step_backtrack_shrink < 1.0)
                ? tr_params.step_backtrack_shrink
                : 0.5;
        double alpha = 1.0;
        for (int bt = 0; bt <= max_backtracks; ++bt) {
            const Eigen::VectorXd s_trial = alpha * s;
            std::vector<MeshData> trial = out.frames;
            const Eigen::VectorXd x_trial = x + s_trial;
            unpack_interior_frames(x_trial, trial, tr_params);
            const double model_e_trial =
                path_energy(trial, energy_params, &model_cache).terms.total;
            e_trial =
                tr_params.use_rebuilt_acceptance_energy
                    ? path_energy(trial, energy_params).terms.total
                    : model_e_trial;

            const double pred =
                -(alpha * g_dot_s + 0.5 * alpha * alpha * sHs);
            const double ared = cur.energy - e_trial;
            rho = (pred > 0.0) ? (ared / pred) : -1.0;
            if (rho > tr_params.accept_eta && std::isfinite(e_trial)) {
                accepted = true;
                accepted_trial = std::move(trial);
                accepted_x_trial = x_trial;
                accepted_alpha = alpha;
                accepted_step_norm = s_trial.norm();
                break;
            }
            alpha *= shrink;
        }

        if (accepted) {
            accepted = true;
            out.frames = std::move(accepted_trial);
            x = accepted_x_trial;
            model_cache =
                build_path_energy_frame_cache(out.frames, energy_params);
            cur = eval_model(
                out.frames, energy_params, tr_params, model_cache);
            out.accepted_steps += 1;
            out.accepted_energy.push_back(cur.energy);
            block_precond_dirty = true;
            if (accepted_alpha < 1.0) {
                radius = std::max(1e-12, accepted_step_norm);
            } else if (rho > 0.75 &&
                       std::abs(accepted_step_norm - radius) < 1e-8) {
                radius = std::min(tr_params.max_radius, 2.0 * radius);
            } else if (rho < 0.25) {
                radius *= 0.5;
            }
        } else {
            radius *= 0.5;
        }

        auto it_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = it_end - it_start;
        std::cout << "[TrustRegion] Iteration " << it + 1 << "/" << tr_params.max_iters 
                  << " finished in " << diff.count() << " seconds (Accepted step: " 
                  << (accepted ? "Yes" : "No") << ")\n";

        if (tr_params.iteration_callback) {
            TrustRegionIterationInfo info;
            info.iteration = it + 1;
            info.energy = cur.energy;
            info.trial_energy = e_trial;
            info.grad_norm = gnorm;
            info.radius = radius;
            info.step_norm = accepted ? accepted_step_norm : s.norm();
            info.rho = rho;
            info.accepted = accepted;
            info.converged = false;
            tr_params.iteration_callback(info, out.frames);
        }

        if (radius < 1e-12) break;
    }
    return out;
}

} // namespace rsh
