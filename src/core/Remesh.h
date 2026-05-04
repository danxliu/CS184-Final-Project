#ifndef REMESH_H
#define REMESH_H

#include "MeshData.h"

namespace rsh {

// Repulsor-style edge-length pass: collapse edges shorter than 4/5 L0, then
// split edges longer than 4/3 L0 subject to local topology and valence guards.
MeshData remesh_split_collapse(const MeshData &mesh, int max_faces = 0);

// Flip non-Delaunay interior edges until quiescence or max_passes.
MeshData remesh_delaunay_flip(const MeshData &mesh, int max_passes = 10);

// Tangentially smooth non-boundary vertices toward incident simplex centers
// using an area-weighted tangent projector, matching Repulsor's remesher.
MeshData remesh_tangential_smooth(const MeshData &mesh,
                                  double rho = 0.5,
                                  int n_iters = 5);

// Full remeshing primitive: Repulsor-style edge-length pass, Delaunay flip,
// then tangential smoothing.
MeshData remesh_full(const MeshData &mesh, int max_faces = 0);

} // namespace rsh

#endif
