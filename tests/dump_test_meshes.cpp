#include "MeshData.h"
#include "TestMeshes.h"

#include <iostream>

using namespace rsh;

int main() {
    for (int k = 0; k <= 3; ++k) {
        MeshData m = make_icosphere(k);
        const std::string path = "assets/icosphere_" + std::to_string(k) + ".obj";
        m.save_obj(path);
        std::cout << "wrote " << path << "  (V=" << m.n_vertices() << ", F=" << m.n_faces() << ")\n";
    }
    {
        MeshData m = make_torus(1.0, 0.3, 40, 20);
        m.save_obj("assets/torus.obj");
        std::cout << "wrote assets/torus.obj  (V=" << m.n_vertices() << ", F=" << m.n_faces() << ")\n";
    }
    {
        MeshData m = make_torus(1.0, 0.4, 12, 8);
        m.save_obj("assets/torus_12x8.obj");
        std::cout << "wrote assets/torus_12x8.obj  (V=" << m.n_vertices()
                  << ", F=" << m.n_faces() << ")\n";
    }
    return 0;
}
