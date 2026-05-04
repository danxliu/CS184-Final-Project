#ifndef TESTMESHES_H
#define TESTMESHES_H

#include "MeshData.h"

namespace rsh {

MeshData make_icosphere(int subdivisions = 2);
MeshData make_torus(double R = 1.0, double r = 0.3, int nu = 40, int nv = 20);
MeshData make_n_torus(int genus,
                      double R = 1.0,
                      double r = 0.3,
                      int nu_per_torus = 18,
                      int nv = 10);

} // namespace rsh

#endif
