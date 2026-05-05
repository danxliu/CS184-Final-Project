#ifndef SURFACE_BARRIER_H
#define SURFACE_BARRIER_H

#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include "TPE.h"

#include <Eigen/Dense>
#include <vector>

namespace rsh {

struct SurfaceBarrierCache {
    FaceGeom barrier_geom;
    BVH moving_bvh;
    BVH barrier_bvh;
    BlockPairs bp;
    TpeAdaptiveParams adaptive;
    std::vector<TpeNearFieldTerm> near_terms;
    double theta = 0.5;
    bool has_adaptive = false;
};

// Fixed-surface tangent-point barrier used for RS-style obstacles. This is the
// cross term between a moving shell and a fixed obstacle surface, symmetrized
// over both source normals as in the ordered TPE sum on a union of surfaces.
double surface_tpe_barrier_energy(const MeshData &mesh,
                                  const MeshData &barrier,
                                  double alpha = 6.0);

Eigen::MatrixXd surface_tpe_barrier_gradient(const MeshData &mesh,
                                             const MeshData &barrier,
                                             double alpha = 6.0);

// Hierarchical variant matching Repulsor's unsymmetric mesh/obstacle BCT
// structure. The cache freezes the moving/tree obstacle partition at the local
// trust-region iterate; callers may update the moving vertex positions and the
// aggregate values will be refreshed without rebuilding admissibility.
SurfaceBarrierCache build_surface_tpe_barrier_cache(
    const MeshData &mesh,
    const MeshData &barrier,
    double theta = 0.5,
    const TpeAdaptiveParams &adaptive = TpeAdaptiveParams());

double surface_tpe_barrier_energy_bh(
    const MeshData &mesh,
    const MeshData &barrier,
    const SurfaceBarrierCache &cache,
    double alpha = 6.0);

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(
    const MeshData &mesh,
    const MeshData &barrier,
    const SurfaceBarrierCache &cache,
    double alpha = 6.0);

double surface_tpe_barrier_energy_bh(const MeshData &mesh,
                                     const MeshData &barrier,
                                     double alpha = 6.0,
                                     double theta = 0.5);

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(const MeshData &mesh,
                                                const MeshData &barrier,
                                                double alpha = 6.0,
                                                double theta = 0.5);

double surface_tpe_barrier_energy_bh(const MeshData &mesh,
                                     const MeshData &barrier,
                                     const TpeAdaptiveParams &adaptive,
                                     double alpha = 6.0,
                                     double theta = 0.5);

Eigen::MatrixXd surface_tpe_barrier_gradient_bh(const MeshData &mesh,
                                                const MeshData &barrier,
                                                const TpeAdaptiveParams &adaptive,
                                                double alpha = 6.0,
                                                double theta = 0.5);

} // namespace rsh

#endif
