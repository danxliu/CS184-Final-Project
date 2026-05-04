#ifndef REMESH_H
#define REMESH_H

#include "MeshData.h"

namespace rsh {

// One RSu-style remeshing pass: split edges longer than 4/3 L0, then collapse
// edges shorter than 4/5 L0 subject to local foldover and valence guards.
MeshData remesh_split_collapse(const MeshData &mesh);

// Flip non-Delaunay interior edges until quiescence or max_passes.
MeshData remesh_delaunay_flip(const MeshData &mesh, int max_passes = 10);

// Tangentially smooth non-boundary vertices toward their area-weighted
// circumcenter targets.
MeshData remesh_tangential_smooth(const MeshData &mesh,
                                  double rho = 0.5,
                                  int n_iters = 5);

// Full RSu remeshing primitive: split, collapse, Delaunay flip, smooth.
MeshData remesh_full(const MeshData &mesh);

} // namespace rsh

#endif
