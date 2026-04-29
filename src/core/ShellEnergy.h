#ifndef SHELL_ENERGY_H
#define SHELL_ENERGY_H

#include "MeshData.h"

#include <Eigen/Dense>

namespace rsh {

struct ShellEnergyParams {
    // Shell thickness (paper notation: delta).
    double thickness = 1.0;
    // Lamé parameters for membrane energy.
    double lambda = 1.0;
    double mu = 1.0;
    // Smoothing used to keep det terms finite around degeneracy.
    double det_smoothing = 1e-8;
    // Clamp theta to pi - eps before tan(theta / 2) to keep barrier finite.
    double angle_clamp_eps = 1e-8;
    // Step size for finite-difference bending gradients.
    double bending_fd_eps = 1e-6;
    // If false, use (theta_tilde - theta)^2 instead of tan-based barrier.
    bool use_tan_bending = true;
};

struct ShellEnergyValue {
    double total = 0.0;
    double membrane = 0.0;
    double bending = 0.0;
};

struct ShellEnergyGradientResult {
    ShellEnergyValue energy;
    Eigen::MatrixXd grad_ref;  // dW_c / dx
    Eigen::MatrixXd grad_def;  // dW_c / d\tilde{x}
};

// Discrete shell energy W_c(x, x_tilde) from RS Section 3.1:
//   W_c = W_membrane + W_bending.
ShellEnergyValue shell_energy(const MeshData &x_ref,
                              const MeshData &x_def,
                              const ShellEnergyParams &params =
                                  ShellEnergyParams());

// Energy + gradients wrt both reference and deformed positions.
// Membrane gradient is analytical; bending gradient is finite-difference.
ShellEnergyGradientResult shell_energy_with_gradient(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params = ShellEnergyParams());

} // namespace rsh

#endif
