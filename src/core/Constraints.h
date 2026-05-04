#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <Eigen/Dense>

#include <vector>

namespace rsh {

// Project a per-vertex 3-vector field so each component has zero arithmetic
// mean across vertices. This is the tangent-space projection for a fixed
// barycenter constraint.
void project_barycenter(Eigen::MatrixXd &d);

// Zero rows in a per-vertex 3-vector field for pinned vertices.
void apply_pin_mask(Eigen::MatrixXd &d, const std::vector<bool> &pin);

} // namespace rsh

#endif
