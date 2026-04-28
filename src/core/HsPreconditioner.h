#ifndef HSPRECONDITIONER_H
#define HSPRECONDITIONER_H

#include "MeshData.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace rsh {

struct HsOperators {
    Eigen::SparseMatrix<double> L;
    Eigen::SparseMatrix<double> M;
    Eigen::VectorXd mass_diag;
};

struct HsPreconditionerParams {
    double s = 1.0;
    double sigma = 1.0;
    double mass_weight = 1.0;
};

struct HsDirectionResult {
    Eigen::MatrixXd direction;
    double g_dot_dir = 0.0;
};

HsOperators build_hs_operators(const MeshData &mesh);

HsDirectionResult hs_preconditioned_direction(
    const MeshData &mesh,
    const Eigen::MatrixXd &gradient,
    const HsPreconditionerParams &params = HsPreconditionerParams());

} // namespace rsh

#endif
