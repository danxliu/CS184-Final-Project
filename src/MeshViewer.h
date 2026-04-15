#ifndef MESHVIEWER_H
#define MESHVIEWER_H

#include <nanogui/nanogui.h>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <string>

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

class MeshViewer : public nanogui::Screen {
public:
    explicit MeshViewer(const std::string& filename);
    void draw_contents() override;

private:
    std::vector<float> m_vertices;
    std::vector<float> m_normals;
    std::vector<uint32_t> m_indices;
    size_t m_index_count;
    nanogui::ref<nanogui::RenderPass> m_render_pass;
    nanogui::ref<nanogui::Shader> m_shader;

    std::string read_file(const std::string& filename);
    void get_mesh_data(Mesh &mesh);
};

#endif // MESHVIEWER_H
