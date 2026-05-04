#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include "TPE.h"

#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <mesh.obj>\n", argv[0]);
        return 1;
    }
    rsh::MeshData mesh = rsh::MeshData::load_obj(argv[1]);
    rsh::FaceGeom g = rsh::compute_face_geom(mesh);
    const double alpha = 6.0;
    const double e_brute = rsh::tpe_energy_brute(mesh, alpha);
    std::printf("brute=%.6f\n", e_brute);
    std::printf("%-9s %-9s %-15s %-12s %-12s %-12s\n",
                "leaf", "theta", "energy", "rel_err%", "n_admis", "n_near");
    for (int leaf : {8, 4, 2, 1}) {
        for (double theta : {0.5, 0.25, 0.1}) {
            rsh::BVH bvh = rsh::build_bvh(mesh, g, leaf);
            rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);
            const double e = rsh::tpe_energy_bh(g, bvh, bp, alpha);
            std::printf("%-9d %-9.2f %-15.6f %-12.4f %-12zu %-12zu\n",
                        leaf, theta, e, 100.0 * (e - e_brute) / e_brute,
                        bp.admissible.size(), bp.near_field.size());
        }
    }
    return 0;
}
