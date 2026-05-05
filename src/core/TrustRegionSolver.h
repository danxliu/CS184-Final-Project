#ifndef TRUST_REGION_SOLVER_H
#define TRUST_REGION_SOLVER_H

#include "PathEnergy.h"

#include <functional>
#include <vector>

namespace rsh {

struct TrustRegionIterationInfo {
    int iteration = 0;
    double energy = 0.0;
    double trial_energy = 0.0;
    double grad_norm = 0.0;
    double radius = 0.0;
    double step_norm = 0.0;
    double rho = 0.0;
    bool accepted = false;
    bool converged = false;
};

struct TrustRegionParams {
    int max_iters = 200;
    int max_cg_iters = 50;
    double grad_tol = 1e-6;
    double cg_tol = 1e-4;
    double initial_radius = 1e-2;
    double max_radius = 1.0;
    double accept_eta = 0.1;
    int max_step_backtracks = 8;
    double step_backtrack_shrink = 0.5;
    // RS Section 6.1 uses the Gauss-Newton graph Hessian near a minimizer.
    // When the graph residual is still large, include the omitted residual
    // Hessian term by finite-differencing dPhi in HVPs. This is slower but
    // matches the discrete objective more closely during diagnostics.
    bool use_graph_residual_hessian = false;
    double graph_residual_hvp_fd_eps = 1e-6;
    // Use a freshly rebuilt hierarchy/adaptive quadrature for actual trial
    // acceptance. The local model still freezes partitions for derivatives,
    // but large steps can otherwise miss newly singular near-contact terms.
    bool use_rebuilt_acceptance_energy = true;
    bool use_block_diagonal_preconditioner = true;
    int block_preconditioner_max_block_dofs = 1200;
    double block_preconditioner_regularization = 1e-6;
    // Lift the elastic shell Hessian's rigid-motion nullspace. If this floor
    // is too small, the block preconditioner over-amplifies whole-frame rigid
    // translations and the ball-through-tube solve prefers temporal jumps over
    // shell deformation.
    double block_preconditioner_regularization_floor = 1e-3;
    std::vector<bool> free_vertices;
    bool optimize_end_frame = false;
    std::function<void(const TrustRegionIterationInfo &,
                       const std::vector<MeshData> &)> iteration_callback;
};

struct TrustRegionResult {
    std::vector<MeshData> frames;
    std::vector<double> accepted_energy;
    int outer_iterations = 0;
    int accepted_steps = 0;
    bool converged = false;
};

// Interpolate a geodesic-like trajectory by minimizing the discrete path energy
// with fixed endpoints using a trust-region method and Steihaug CG subproblems.
TrustRegionResult interpolate_geodesic_trust_region(
    const std::vector<MeshData> &initial_frames,
    const PathEnergyParams &energy_params = PathEnergyParams(),
    const TrustRegionParams &tr_params = TrustRegionParams());

} // namespace rsh

#endif
