#include "PathEnergy.h"

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"

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
        if (frame_cache != nullptr) {
            const PathEnergyFrameCache &fc = frame_cache->at(k);
            fe.bvh = fc.bvh;
            update_bvh_aggregates(fe.bvh, fe.g);
            fe.bp = fc.bp;
            fe.cache = fc.adaptive_cache;
            if (params.tpe_adaptive.enabled && fc.has_adaptive) {
                fe.phi = tpe_energy_bh(
                    frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                    params.tpe_alpha, &fe.cache);
                if (with_gradient) {
                    fe.grad_phi = tpe_gradient_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                        params.tpe_alpha, &fe.cache);
                }
            } else {
                fe.phi = tpe_energy_bh(fe.g, fe.bvh, fe.bp, params.tpe_alpha);
                if (with_gradient) {
                    fe.grad_phi = tpe_gradient_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_alpha);
                }
            }
        } else {
            fe.bvh = build_bvh(frames[k], fe.g);
            fe.bp = build_bct_self(fe.bvh, params.tpe_theta);
            if (params.tpe_adaptive.enabled) {
                fe.cache = build_tpe_adaptive_cache(
                    frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive);
                fe.phi = tpe_energy_bh(
                    frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                    params.tpe_alpha, &fe.cache);
                if (with_gradient) {
                    fe.grad_phi = tpe_gradient_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_adaptive,
                        params.tpe_alpha, &fe.cache);
                }
            } else {
                fe.phi = tpe_energy_bh(fe.g, fe.bvh, fe.bp, params.tpe_alpha);
                if (with_gradient) {
                    fe.grad_phi = tpe_gradient_bh(
                        frames[k], fe.g, fe.bvh, fe.bp, params.tpe_alpha);
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        std::cout << " done in " << diff.count() << " seconds\n";
    }
    return out;
}

} // namespace

std::vector<PathEnergyFrameCache> build_path_energy_frame_cache(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params) {
    validate_frames(frames);
    std::vector<PathEnergyFrameCache> out(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[PathEnergy] Building cache for frame " << k+1 << " of " << frames.size() << "..." << std::flush;
        const FaceGeom g = compute_face_geom(frames[k]);
        out[k].bvh = build_bvh(frames[k], g);
        out[k].bp = build_bct_self(out[k].bvh, params.tpe_theta);
        if (params.tpe_adaptive.enabled) {
            out[k].adaptive_cache = build_tpe_adaptive_cache(
                frames[k], g, out[k].bvh, out[k].bp, params.tpe_adaptive);
            out[k].has_adaptive = true;
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
        for (int k = 1; k <= n; ++k) {
            const ShellEnergyValue w =
                shell_energy(frames[static_cast<size_t>(k - 1)],
                             frames[static_cast<size_t>(k)], params.shell);
            const double dphi = fe[static_cast<size_t>(k - 1)].phi - fe[static_cast<size_t>(k)].phi;
            out.terms.shell_sum += w.total;
            out.terms.repulsive_sum += dphi * dphi;
        }
        out.terms.total = scale * (out.terms.shell_sum + out.terms.repulsive_sum);
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
            out.energy.terms.repulsive_sum += dphi * dphi;
            out.grad_frames[static_cast<size_t>(km1)] +=
                scale * (2.0 * dphi) * fe[static_cast<size_t>(km1)].grad_phi;
            out.grad_frames[static_cast<size_t>(k)] +=
                scale * (-2.0 * dphi) * fe[static_cast<size_t>(k)].grad_phi;
        }

        out.energy.terms.total =
            scale * (out.energy.terms.shell_sum + out.energy.terms.repulsive_sum);
    }
    return out;
}

} // namespace rsh
