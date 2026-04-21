#include "GradCheck.h"

#include <algorithm>
#include <cmath>

namespace rsh {

GradCheckResult finite_diff_gradient_check(
    const std::function<double(const Eigen::VectorXd &)> &f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    const Eigen::VectorXd &x,
    double eps) {
    GradCheckResult r;
    r.analytical = grad_f(x);
    r.numerical = Eigen::VectorXd::Zero(x.size());

    Eigen::VectorXd xp = x;
    for (int i = 0; i < x.size(); ++i) {
        const double h = eps * (1.0 + std::abs(x[i]));
        const double xi = x[i];
        xp[i] = xi + h;
        const double fp = f(xp);
        xp[i] = xi - h;
        const double fm = f(xp);
        xp[i] = xi;
        r.numerical[i] = (fp - fm) / (2.0 * h);
    }

    for (int i = 0; i < x.size(); ++i) {
        const double a = r.analytical[i];
        const double n = r.numerical[i];
        const double abs_err = std::abs(a - n);
        const double scale = std::max({1.0, std::abs(a), std::abs(n)});
        const double rel_err = abs_err / scale;
        if (rel_err > r.max_rel_err) {
            r.max_rel_err = rel_err;
            r.worst_index = i;
        }
        r.max_abs_err = std::max(r.max_abs_err, abs_err);
    }
    return r;
}

} // namespace rsh
