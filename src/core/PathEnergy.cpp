#include "PathEnergy.h"

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "Obstacle.h"
#include "SurfaceBarrier.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace rsh {

namespace {

struct FrameEval {
    FaceGeom g;
    BVH bvh;
    BlockPairs bp;
    TpeAdaptiveCache cache;
    double tpe_phi = 0.0;
    double obstacle_phi = 0.0;
    double phi = 0.0;
    Eigen::MatrixXd grad_phi;
};

void validate_frames(const std::vector<MeshData> &frames) {
    if (frames.empty()) {
        throw std::runtime_error("path_energy: need at least 1 frame");
    }
    const int nv = frames.front().n_vertices();
    const int nf = frames.front().n_faces();
    const Eigen::MatrixXi &F0 = frames.front().F;
    for (size_t k = 1; k < frames.size(); ++k) {
        if (frames[k].n_vertices() != nv || frames[k].n_faces() != nf) {
            throw std::runtime_error("path_energy: all frames must share topology");
        }
        if ((frames[k].F.rows() != F0.rows()) ||
            ((frames[k].F - F0).array() != 0).any()) {
            throw std::runtime_error("path_energy: frame connectivity mismatch");
        }
    }
}

std::vector<FrameEval> build_frame_eval(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params,
    bool with_gradient,
    const std::vector<PathEnergyFrameCache> *frame_cache) {
    std::vector<FrameEval> out(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[PathEnergy] Evaluating " << (with_gradient ? "energy+grad" : "energy") 
                  << " for frame " << k+1 << " of " << frames.size() << "..." << std::flush;
        FrameEval &fe = out[k];
        fe.g = compute_face_geom(frames[k]);
        if (with_gradient) {
            fe.grad_phi =
                Eigen::MatrixXd::Zero(frames[k].n_vertices(), 3);
        }
        if (params.self_tpe_weight != 0.0) {
            Eigen::MatrixXd grad_tpe;
            if (frame_cache != nullptr) {
                const PathEnergyFrameCache &fc = frame_cache->at(k);
                fe.bvh = fc.bvh;
                update_bvh_aggregates(fe.bvh, fe.g);
                fe.bp = fc.bp;
                fe.cache = fc.adaptive_cache;
                if (params.tpe_adaptive.enabled && fc.has_adaptive) {
                    fe.tpe_phi = tpe_energy_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                        params.tpe_alpha, &fe.cache);
                    if (with_gradient) {
                        grad_tpe = tpe_gradient_bh(
                            frames[k], fe.g, fe.bvh, fe.bp,
                            params.tpe_adaptive, params.tpe_alpha,
                            &fe.cache);
                    }
                } else {
                    fe.tpe_phi =
                        tpe_energy_bh(fe.g, fe.bvh, fe.bp,
                                      params.tpe_alpha);
                    if (with_gradient) {
                        grad_tpe = tpe_gradient_bh(
                            frames[k], fe.g, fe.bvh, fe.bp,
                            params.tpe_alpha);
                    }
                }
            } else {
                fe.bvh = build_bvh(frames[k], fe.g);
                fe.bp = build_bct_self(fe.bvh, params.tpe_theta);
                if (params.tpe_adaptive.enabled) {
                    fe.cache = build_tpe_adaptive_cache(
                        frames[k], fe.g, fe.bvh, fe.bp,
                        params.tpe_adaptive);
                    fe.tpe_phi = tpe_energy_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                        params.tpe_alpha, &fe.cache);
                    if (with_gradient) {
                        grad_tpe = tpe_gradient_bh(
                            frames[k], fe.g, fe.bvh, fe.bp,
                            params.tpe_adaptive, params.tpe_alpha,
                            &fe.cache);
                    }
                } else {
                    fe.tpe_phi =
                        tpe_energy_bh(fe.g, fe.bvh, fe.bp,
                                      params.tpe_alpha);
                    if (with_gradient) {
                        grad_tpe = tpe_gradient_bh(
                            frames[k], fe.g, fe.bvh, fe.bp,
                            params.tpe_alpha);
                    }
                }
            }
            fe.phi += params.self_tpe_weight * fe.tpe_phi;
            if (with_gradient) {
                fe.grad_phi += params.self_tpe_weight * grad_tpe;
            }
        }
        if (params.tpe_barrier_mesh != nullptr &&
            params.tpe_barrier_weight != 0.0) {
            const bool use_barrier_cache =
                frame_cache != nullptr &&
                frame_cache->at(k).has_barrier_cache;
            const SurfaceBarrierCache *barrier_cache =
                use_barrier_cache ? &frame_cache->at(k).barrier_cache
                                  : nullptr;
            const double barrier_phi =
                (barrier_cache != nullptr)
                    ? surface_tpe_barrier_energy_bh(
                          frames[k], *params.tpe_barrier_mesh,
                          *barrier_cache, params.tpe_alpha)
                    : surface_tpe_barrier_energy_bh(
                          frames[k], *params.tpe_barrier_mesh,
                          params.tpe_adaptive,
                          params.tpe_alpha, params.tpe_theta);
            fe.obstacle_phi += barrier_phi;
            fe.phi += params.tpe_barrier_weight * barrier_phi;
            if (with_gradient) {
                fe.grad_phi +=
                    params.tpe_barrier_weight *
                    ((barrier_cache != nullptr)
                         ? surface_tpe_barrier_gradient_bh(
                               frames[k], *params.tpe_barrier_mesh,
                               *barrier_cache, params.tpe_alpha)
                         : surface_tpe_barrier_gradient_bh(
                               frames[k], *params.tpe_barrier_mesh,
                               params.tpe_adaptive,
                               params.tpe_alpha, params.tpe_theta));
            }
        }
        if (params.obstacle != nullptr && params.obstacle_weight != 0.0) {
            const double sdf_phi = obstacle_energy(frames[k], *params.obstacle);
            fe.obstacle_phi += sdf_phi;
            fe.phi += params.obstacle_weight * sdf_phi;
            if (with_gradient) {
                fe.grad_phi += params.obstacle_weight *
                               obstacle_gradient(frames[k], *params.obstacle);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        std::cout << " done in " << diff.count() << " seconds\n";
    }
    return out;
}

Eigen::Vector3d barycenter(const MeshData &mesh) {
    return mesh.V.colwise().mean().transpose();
}

double rigid_segment_energy(const MeshData &prev,
                            const MeshData &next,
                            const PathEnergyParams &params) {
    double e = 0.0;
    if (params.rigid_translation_weight > 0.0) {
        const Eigen::Vector3d dc = barycenter(next) - barycenter(prev);
        e += params.rigid_translation_weight * dc.squaredNorm();
    }
    if (params.rigid_rotation_weight > 0.0) {
        // RS Eq. 27: ‖Σ_v (x_v^k − x_v^{k-1}) × x_v^{k-1}‖². Since a × a = 0,
        // this collapses to ‖Σ_v b_v × a_v‖² with a = prev, b = next — a
        // single net angular-momentum-like vector, then squared.
        Eigen::Vector3d L = Eigen::Vector3d::Zero();
        for (int i = 0; i < prev.n_vertices(); ++i) {
            const Eigen::Vector3d a = prev.V.row(i).transpose();
            const Eigen::Vector3d b = next.V.row(i).transpose();
            L += b.cross(a);
        }
        e += params.rigid_rotation_weight * L.squaredNorm();
    }
    return e;
}

void add_rigid_segment_gradient(const MeshData &prev,
                                const MeshData &next,
                                const PathEnergyParams &params,
                                double scale,
                                Eigen::MatrixXd &grad_prev,
                                Eigen::MatrixXd &grad_next) {
    const int nv = prev.n_vertices();
    if (params.rigid_translation_weight > 0.0) {
        const Eigen::Vector3d dc = barycenter(next) - barycenter(prev);
        const Eigen::RowVector3d g_next =
            (scale * params.rigid_translation_weight * 2.0 /
             static_cast<double>(nv)) *
            dc.transpose();
        for (int i = 0; i < nv; ++i) {
            grad_next.row(i) += g_next;
            grad_prev.row(i) -= g_next;
        }
    }

    if (params.rigid_rotation_weight > 0.0) {
        const double w = scale * params.rigid_rotation_weight;
        // L = Σ_v b_v × a_v is a single 3-vector shared across all vertices.
        Eigen::Vector3d L = Eigen::Vector3d::Zero();
        for (int i = 0; i < nv; ++i) {
            const Eigen::Vector3d a = prev.V.row(i).transpose();
            const Eigen::Vector3d b = next.V.row(i).transpose();
            L += b.cross(a);
        }
        // ∇_{a_v} ‖L‖² = 2 (L × b_v); ∇_{b_v} ‖L‖² = 2 (a_v × L).
        const double two_w = 2.0 * w;
        for (int i = 0; i < nv; ++i) {
            const Eigen::Vector3d a = prev.V.row(i).transpose();
            const Eigen::Vector3d b = next.V.row(i).transpose();
            grad_prev.row(i) += (two_w * L.cross(b)).transpose();
            grad_next.row(i) += (two_w * a.cross(L)).transpose();
        }
    }
}

} // namespace

std::vector<PathEnergyFrameCache> build_path_energy_frame_cache(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params) {
    validate_frames(frames);
    std::vector<PathEnergyFrameCache> out(frames.size());
    const bool need_self_tpe = params.self_tpe_weight != 0.0;
    const bool need_barrier =
        params.tpe_barrier_mesh != nullptr &&
        params.tpe_barrier_weight != 0.0;
    if (!need_self_tpe && !need_barrier) {
        return out;
    }
    for (size_t k = 0; k < frames.size(); ++k) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[PathEnergy] Building cache for frame " << k+1 << " of " << frames.size() << "..." << std::flush;
        const FaceGeom g = compute_face_geom(frames[k]);
        if (need_self_tpe) {
            out[k].bvh = build_bvh(frames[k], g);
            out[k].bp = build_bct_self(out[k].bvh, params.tpe_theta);
            if (params.tpe_adaptive.enabled) {
                out[k].adaptive_cache = build_tpe_adaptive_cache(
                    frames[k], g, out[k].bvh, out[k].bp,
                    params.tpe_adaptive);
                out[k].has_adaptive = true;
            }
        }
        if (need_barrier) {
            out[k].barrier_cache = build_surface_tpe_barrier_cache(
                frames[k], *params.tpe_barrier_mesh, params.tpe_theta,
                params.tpe_adaptive);
            out[k].has_barrier_cache = true;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        std::cout << " done in " << diff.count() << " seconds\n";
    }
    return out;
}

PathEnergyResult path_energy(const std::vector<MeshData> &frames,
                             const PathEnergyParams &params,
                             const std::vector<PathEnergyFrameCache> *frame_cache) {
    validate_frames(frames);
    if (frame_cache != nullptr && frame_cache->size() != frames.size()) {
        throw std::runtime_error("path_energy: frame_cache size mismatch");
    }
    const int num_frames = static_cast<int>(frames.size());
    const int n = num_frames - 1;
    const double scale = static_cast<double>(n);

    const std::vector<FrameEval> fe =
        build_frame_eval(frames, params, false, frame_cache);
    PathEnergyResult out;
    out.phi_per_frame.resize(frames.size(), 0.0);
    for (size_t k = 0; k < fe.size(); ++k) out.phi_per_frame[k] = fe[k].phi;

    if (n >= 1) {
        if (params.obstacle != nullptr || params.tpe_barrier_mesh != nullptr) {
            for (const FrameEval &frame : fe) {
                out.terms.obstacle_sum += frame.obstacle_phi;
            }
        }
        for (int k = 1; k <= n; ++k) {
            const ShellEnergyValue w =
                shell_energy(frames[static_cast<size_t>(k - 1)],
                             frames[static_cast<size_t>(k)], params.shell);
            const double dphi = fe[static_cast<size_t>(k - 1)].phi - fe[static_cast<size_t>(k)].phi;
            out.terms.shell_sum += w.total;
            out.terms.repulsive_sum += params.graph_beta * dphi * dphi;
            out.terms.rigid_sum += rigid_segment_energy(
                frames[static_cast<size_t>(k - 1)],
                frames[static_cast<size_t>(k)], params);
        }
        out.terms.total = scale *
            (out.terms.shell_sum + out.terms.repulsive_sum +
             out.terms.rigid_sum);
    }
    return out;
}

PathEnergyGradientResult path_energy_with_gradient(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params,
    const std::vector<PathEnergyFrameCache> *frame_cache) {
    validate_frames(frames);
    if (frame_cache != nullptr && frame_cache->size() != frames.size()) {
        throw std::runtime_error("path_energy_with_gradient: frame_cache size mismatch");
    }
    const int num_frames = static_cast<int>(frames.size());
    const int n = num_frames - 1;
    const int nv = frames.front().n_vertices();
    const double scale = static_cast<double>(n);

    const std::vector<FrameEval> fe =
        build_frame_eval(frames, params, true, frame_cache);

    PathEnergyGradientResult out;
    out.grad_frames.assign(frames.size(), Eigen::MatrixXd::Zero(nv, 3));
    out.grad_phi_per_frame.assign(frames.size(), Eigen::MatrixXd::Zero(nv, 3));
    out.energy.phi_per_frame.resize(frames.size(), 0.0);
    for (size_t k = 0; k < fe.size(); ++k) {
        out.energy.phi_per_frame[k] = fe[k].phi;
        out.grad_phi_per_frame[k] = fe[k].grad_phi;
    }

    if (n >= 1) {
        if (params.obstacle != nullptr || params.tpe_barrier_mesh != nullptr) {
            for (const FrameEval &frame : fe) {
                out.energy.terms.obstacle_sum += frame.obstacle_phi;
            }
        }
        for (int k = 1; k <= n; ++k) {
            const int km1 = k - 1;
            const ShellEnergyGradientResult sw =
                shell_energy_with_gradient(frames[static_cast<size_t>(km1)],
                                           frames[static_cast<size_t>(k)],
                                           params.shell);
            out.energy.terms.shell_sum += sw.energy.total;
            out.grad_frames[static_cast<size_t>(km1)] += scale * sw.grad_ref;
            out.grad_frames[static_cast<size_t>(k)] += scale * sw.grad_def;

            const double dphi = fe[static_cast<size_t>(km1)].phi - fe[static_cast<size_t>(k)].phi;
            out.energy.terms.repulsive_sum +=
                params.graph_beta * dphi * dphi;
            out.grad_frames[static_cast<size_t>(km1)] +=
                scale * params.graph_beta * (2.0 * dphi) *
                fe[static_cast<size_t>(km1)].grad_phi;
            out.grad_frames[static_cast<size_t>(k)] +=
                scale * params.graph_beta * (-2.0 * dphi) *
                fe[static_cast<size_t>(k)].grad_phi;

            out.energy.terms.rigid_sum += rigid_segment_energy(
                frames[static_cast<size_t>(km1)],
                frames[static_cast<size_t>(k)], params);
            add_rigid_segment_gradient(
                frames[static_cast<size_t>(km1)],
                frames[static_cast<size_t>(k)], params, scale,
                out.grad_frames[static_cast<size_t>(km1)],
                out.grad_frames[static_cast<size_t>(k)]);
        }

        out.energy.terms.total = scale *
            (out.energy.terms.shell_sum +
             out.energy.terms.repulsive_sum +
             out.energy.terms.rigid_sum);
    }
    return out;
}

} // namespace rsh
