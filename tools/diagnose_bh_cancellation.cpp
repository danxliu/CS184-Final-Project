#include "BCT.h"
#include "BVH.h"
#include "FaceGeom.h"
#include "MeshData.h"
#include "TPE.h"

#include <algorithm>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <mesh.obj> [theta=0.5]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    const double theta = (argc >= 3) ? std::stod(argv[2]) : 0.5;

    rsh::MeshData mesh = rsh::MeshData::load_obj(path);
    rsh::FaceGeom g = rsh::compute_face_geom(mesh);
    rsh::BVH bvh = rsh::build_bvh(mesh, g);
    rsh::BlockPairs bp = rsh::build_bct_self(bvh, theta);

    std::vector<double> coh;
    coh.reserve(bp.admissible.size() * 2);
    for (const auto &cp : bp.admissible) {
        for (int u : {cp.u, cp.v}) {
            const auto &node = bvh.nodes[u];
            const double mag = node.normal_sum.norm();
            const double c = (node.area > 0.0) ? mag / node.area : 0.0;
            coh.push_back(c);
        }
    }
    std::sort(coh.begin(), coh.end());
    auto pct = [&](double p) {
        const size_t n = coh.size();
        if (n == 0) return 0.0;
        size_t i = static_cast<size_t>(p * (n - 1));
        return coh[i];
    };

    int n_below_05 = 0, n_below_02 = 0, n_below_01 = 0, n_below_001 = 0;
    for (double c : coh) {
        if (c < 0.5)  ++n_below_05;
        if (c < 0.2)  ++n_below_02;
        if (c < 0.1)  ++n_below_01;
        if (c < 0.01) ++n_below_001;
    }

    std::printf("mesh=%s theta=%.2f\n", path.c_str(), theta);
    std::printf("admissible_pairs=%zu (each contributes 2 cluster slots)\n",
                bp.admissible.size());
    std::printf("near_field_pairs=%zu\n", bp.near_field.size());
    std::printf("|n_U|/a_U distribution (1.0 = perfectly aligned, 0 = full cancellation):\n");
    std::printf("  min=%.6f  p10=%.6f  p25=%.6f  p50=%.6f  p75=%.6f  p90=%.6f  max=%.6f\n",
                coh.empty() ? 0.0 : coh.front(),
                pct(0.10), pct(0.25), pct(0.50),
                pct(0.75), pct(0.90),
                coh.empty() ? 0.0 : coh.back());
    std::printf("  fraction with coh<0.5 : %.3f (%d/%zu)\n",
                coh.empty() ? 0.0 : double(n_below_05) / coh.size(), n_below_05, coh.size());
    std::printf("  fraction with coh<0.2 : %.3f (%d/%zu)\n",
                coh.empty() ? 0.0 : double(n_below_02) / coh.size(), n_below_02, coh.size());
    std::printf("  fraction with coh<0.1 : %.3f (%d/%zu)\n",
                coh.empty() ? 0.0 : double(n_below_01) / coh.size(), n_below_01, coh.size());
    std::printf("  fraction with coh<0.01: %.3f (%d/%zu)\n",
                coh.empty() ? 0.0 : double(n_below_001) / coh.size(), n_below_001, coh.size());

    const double e_brute = rsh::tpe_energy_brute(mesh, 6.0);
    const double e_bh = rsh::tpe_energy_bh(mesh, 6.0, theta);
    std::printf("brute=%.6f  bh@theta=%.2f=%.6f  rel_err=%.4f%%\n",
                e_brute, theta, e_bh, 100.0 * (e_bh - e_brute) / e_brute);

    return 0;
}
