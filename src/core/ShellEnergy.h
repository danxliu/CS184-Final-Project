#ifndef SHELL_ENERGY_H
#define SHELL_ENERGY_H

#include "MeshData.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

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
    // Step size used only when analytical bending gradients are disabled.
    double bending_fd_eps = 1e-6;
    // If false, use (theta_tilde - theta)^2 instead of tan-based barrier.
    bool use_tan_bending = true;
    // If true, use closed-form local formulas for bending gradients.
    bool use_analytical_bending_gradient = true;
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

struct ShellEnergyHessianResult {
    // Sparse Hessian blocks of W_c(x_ref, x_def). Each block is 3|V| x 3|V|
    // with vertex-major, xyz-minor ordering.
    Eigen::SparseMatrix<double> ref_ref;
    Eigen::SparseMatrix<double> ref_def;
    Eigen::SparseMatrix<double> def_ref;
    Eigen::SparseMatrix<double> def_def;
};

// Discrete shell energy W_c(x, x_tilde) from RS Section 3.1:
//   W_c = W_membrane + W_bending.
ShellEnergyValue shell_energy(const MeshData &x_ref,
                              const MeshData &x_def,
                              const ShellEnergyParams &params =
                                  ShellEnergyParams());

// Energy + gradients wrt both reference and deformed positions.
// Membrane and bending gradients are analytical by default.
ShellEnergyGradientResult shell_energy_with_gradient(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params = ShellEnergyParams());

// Gradient with respect to the deformed positions only. This is useful for
// Hessian-vector and preconditioner probes where the reference mesh is fixed.
Eigen::MatrixXd shell_energy_def_gradient(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params = ShellEnergyParams());

// Exact local Hessian assembled from per-face membrane and per-edge bending
// terms. Bending uses closed-form hinge-angle derivatives; membrane still uses
// local second-order differentiation of the neo-Hookean triangle formula.
ShellEnergyHessianResult shell_energy_hessian(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params = ShellEnergyParams());

// Convenience block for d^2 W_c / d x_def^2.
Eigen::SparseMatrix<double> shell_energy_def_hessian(
    const MeshData &x_ref,
    const MeshData &x_def,
    const ShellEnergyParams &params = ShellEnergyParams());

} // namespace rsh

#endif
