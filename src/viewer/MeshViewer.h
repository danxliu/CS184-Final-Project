#ifndef MESHVIEWER_H
#define MESHVIEWER_H

#include <nanogui/nanogui.h>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <array>
#include <string>
#include <vector>

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

class MeshViewer : public nanogui::Screen {
public:
    explicit MeshViewer(const std::string& filename);
    void draw_contents() override;
    bool keyboard_event(int key, int scancode, int action, int modifiers) override;
    bool mouse_button_event(const nanogui::Vector2i &p, int button, bool down, int modifiers) override;
    bool mouse_motion_event_f(const nanogui::Vector2f &p, const nanogui::Vector2f &rel,
                              int button, int modifiers) override;
    bool scroll_event(const nanogui::Vector2i &p, const nanogui::Vector2f &rel) override;

private:
    std::vector<float> m_vertices;
    std::vector<float> m_normals;
    std::vector<uint32_t> m_indices;
    size_t m_index_count;
    nanogui::ref<nanogui::RenderPass> m_render_pass;
    nanogui::ref<nanogui::Shader> m_shader;

    std::vector<std::string> m_frames;
    int m_frame_idx = 0;
    bool m_playing = false;
    float m_fps = 24.0f;
    double m_last_advance_time = -1.0;
    nanogui::Label *m_frame_label = nullptr;
    nanogui::Slider *m_frame_slider = nullptr;
    nanogui::Button *m_play_button = nullptr;

    std::string read_file(const std::string& filename);
    void read_mesh(const std::string& filename);
    bool load_frame_sequence(const std::string& dir);
    void load_frame(int idx);
    void upload_mesh_buffers();
    void build_playback_ui();
    nanogui::Vector3f camera_forward() const;
    nanogui::Vector3f camera_right() const;
    bool handle_keyboard_motion(float delta_time);
    void orbit_camera(const nanogui::Vector2f &rel);
    void pan_camera(const nanogui::Vector2f &rel);
    void zoom_camera(float scroll_delta);

    float m_fov = 45.0f;
    nanogui::Color m_background_color = nanogui::Color(0.2f, 0.25f, 0.3f, 1.0f);
    nanogui::Color m_object_color = nanogui::Color(0.85f, 0.75f, 0.5f, 1.0f);
    nanogui::Vector3f m_light_pos = nanogui::Vector3f(2.0f, 5.0f, 5.0f);
    nanogui::Vector3f m_camera_eye = nanogui::Vector3f(0.0f, 0.0f, 5.0f);
    nanogui::Vector3f m_camera_target = nanogui::Vector3f(0.0f, 0.0f, 0.0f);
    nanogui::Vector3f m_world_up = nanogui::Vector3f(0.0f, 1.0f, 0.0f);
    std::array<bool, 512> m_keys{};
    bool m_left_mouse_down = false;
    bool m_right_mouse_down = false;
    float m_move_speed = 2.0f;
    float m_orbit_sensitivity = 0.005f;
    float m_pan_sensitivity = 1.0f;
    float m_zoom_sensitivity = 0.15f;
    float m_min_camera_distance = 0.2f;
    float m_max_camera_distance = 100.0f;
    double m_last_frame_time = -1.0;
};

#endif // MESHVIEWER_H
