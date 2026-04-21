#ifndef GRADCHECK_H
#define GRADCHECK_H

#include <Eigen/Dense>
#include <functional>

namespace rsh {

struct GradCheckResult {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    int worst_index = -1;
    Eigen::VectorXd analytical;
    Eigen::VectorXd numerical;

    bool pass(double tol = 1e-5) const { return max_rel_err < tol; }
};

// Central-differences gradient check.
// f(x) evaluates scalar energy; grad_f(x) returns analytical gradient.
GradCheckResult finite_diff_gradient_check(
    const std::function<double(const Eigen::VectorXd &)> &f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    const Eigen::VectorXd &x,
    double eps = 1e-5);

} // namespace rsh

#endif
