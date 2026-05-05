#ifndef PATH_ENERGY_H
#define PATH_ENERGY_H

#include "MeshData.h"
#include "ShellEnergy.h"
#include "SurfaceBarrier.h"
#include "TPE.h"

#include <Eigen/Dense>
#include <vector>

namespace rsh {

class Obstacle;

struct PathEnergyParams {
    ShellEnergyParams shell;
    double tpe_alpha = 6.0;
    double tpe_theta = 0.5;
    // RS Eq. 5 graph-metric strength beta. The graph contribution to each
    // path segment is graph_beta * (Phi_total(x_{k-1}) - Phi_total(x_k))^2.
    double graph_beta = 1.0;
    double self_tpe_weight = 1.0;
    TpeAdaptiveParams tpe_adaptive;
    // Optional fixed obstacle surface. When non-null, a symmetrized
    // surface-to-surface tangent-point barrier Phi_barrier is added to the
    // graph-manifold potential:
    //   Phi_total = TPE + tpe_barrier_weight * Phi_barrier.
    // This matches the RS Fig. 4 "barrier with a hole" treatment more closely
    // than a vertex SDF penalty.
    const MeshData *tpe_barrier_mesh = nullptr;
    double tpe_barrier_weight = 1.0;
    // Optional analytic obstacle. When non-null, its vertex barrier is added
    // to the graph-manifold potential:
    //   Phi_total = TPE + obstacle_weight * Phi_obstacle.
    // It is not an ordinary additive force term in the path energy.
    const Obstacle *obstacle = nullptr;
    double obstacle_weight = 1.0;
    // RS Section 7.4.3 rigid-motion terms. The shell metric factors out rigid
    // motions, so translated endpoints need these terms to avoid zero-cost
    // teleporting paths.
    double rigid_translation_weight = 0.0;
    double rigid_rotation_weight = 0.0;
};

struct PathEnergyTermBreakdown {
    double total = 0.0;
    double shell_sum = 0.0;
    double repulsive_sum = 0.0;
    double obstacle_sum = 0.0;
    double rigid_sum = 0.0;
};

struct PathEnergyResult {
    PathEnergyTermBreakdown terms;
    // Per-frame total graph potential Phi_total, not raw TPE.
    std::vector<double> phi_per_frame;
};

struct PathEnergyGradientResult {
    PathEnergyResult energy;
    std::vector<Eigen::MatrixXd> grad_frames;  // dE / d x_k
    // dPhi_total / d x_k.
    std::vector<Eigen::MatrixXd> grad_phi_per_frame;
};

struct PathEnergyFrameCache {
    BVH bvh;
    BlockPairs bp;
    TpeAdaptiveCache adaptive_cache;
    bool has_adaptive = false;
    SurfaceBarrierCache barrier_cache;
    bool has_barrier_cache = false;
};

// Build frozen per-frame hierarchy/partition caches. These can be reused while
// evaluating energy/gradient on nearby geometry to avoid admissibility flips.
std::vector<PathEnergyFrameCache> build_path_energy_frame_cache(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams());

// Discrete path energy (RS Eq. 6 + Eq. 17 approximation):
// E_hat(x0..xn) = n * sum_k [
//   W_c(x_{k-1}, x_k) + beta (Phi(x_{k-1}) - Phi(x_k))^2
// ].
PathEnergyResult path_energy(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams(),
    const std::vector<PathEnergyFrameCache> *frame_cache = nullptr);

// Energy + per-frame gradients.
PathEnergyGradientResult path_energy_with_gradient(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams(),
    const std::vector<PathEnergyFrameCache> *frame_cache = nullptr);

} // namespace rsh

#endif
