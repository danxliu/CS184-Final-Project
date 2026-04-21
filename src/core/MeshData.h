#ifndef MESHDATA_H
#define MESHDATA_H

#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <string>

namespace rsh {

struct DoubleTraits : public OpenMesh::DefaultTraits {
    using Point = OpenMesh::Vec3d;
    using Normal = OpenMesh::Vec3d;
};
using OMesh = OpenMesh::TriMesh_ArrayKernelT<DoubleTraits>;

struct MeshData {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd N;
    double L0 = 0.0;

    int n_vertices() const { return static_cast<int>(V.rows()); }
    int n_faces() const { return static_cast<int>(F.rows()); }

    static MeshData from_openmesh(const OMesh &mesh);
    OMesh to_openmesh() const;

    static MeshData load_obj(const std::string &filename);
    void save_obj(const std::string &filename) const;

    void normalize();
    double compute_L0() const;
    Eigen::Vector3d centroid() const;
    double bbox_diagonal() const;
};

} // namespace rsh

#endif
