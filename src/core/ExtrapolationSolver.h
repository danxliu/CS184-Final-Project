#ifndef EXTRAPOLATION_SOLVER_H
#define EXTRAPOLATION_SOLVER_H

#include "PathEnergy.h"

#include <vector>

namespace rsh {

struct ExtrapolationParams {
    int max_newton_iters = 20;
    int max_gmres_iters = 50;
    double newton_tol = 1e-5;
    double gmres_tol = 1e-4;
    double fd_eps = 1e-5;
    int armijo_max_steps = 10;
    double armijo_c = 1e-4;
};

struct ExtrapolationResult {
    MeshData next_frame;
    bool converged = false;
    int newton_iters = 0;
};

ExtrapolationResult extrapolate_geodesic(
    const MeshData &x_km1,
    const MeshData &x_k,
    const PathEnergyParams &energy_params = PathEnergyParams(),
    const ExtrapolationParams &params = ExtrapolationParams());

} // namespace rsh

#endif
