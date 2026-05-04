#ifndef SURFACE_BARRIER_H
#define SURFACE_BARRIER_H

#include "MeshData.h"

#include <Eigen/Dense>

namespace rsh {

// Fixed-surface tangent-point barrier used for RS-style obstacles. This is the
// cross term between a moving shell and a fixed obstacle surface, symmetrized
// over both source normals as in the ordered TPE sum on a union of surfaces.
double surface_tpe_barrier_energy(const MeshData &mesh,
                                  const MeshData &barrier,
                                  double alpha = 6.0);

Eigen::MatrixXd surface_tpe_barrier_gradient(const MeshData &mesh,
                                             const MeshData &barrier,
                                             double alpha = 6.0);

} // namespace rsh

#endif
