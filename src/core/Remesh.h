#ifndef REMESH_H
#define REMESH_H

#include "MeshData.h"

namespace rsh {

// One RSu-style remeshing pass: split edges longer than 4/3 L0, then collapse
// edges shorter than 4/5 L0 subject to local foldover and valence guards.
MeshData remesh_split_collapse(const MeshData &mesh);

} // namespace rsh

#endif
