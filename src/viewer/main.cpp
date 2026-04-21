#include <nanogui/nanogui.h>
#include <iostream>
#include "MeshViewer.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh.obj | frame_dir>" << std::endl;
        std::cerr << "  Pass a directory of frame_XXXX.obj files for playback.\n";
        return -1;
    }

    nanogui::init();
    {
        nanogui::ref<MeshViewer> app = new MeshViewer(argv[1]);
        app->draw_all();
        app->set_visible(true);
        nanogui::run();
    }
    nanogui::shutdown();

    return 0;
}
