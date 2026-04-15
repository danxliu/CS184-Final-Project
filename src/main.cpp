#include <nanogui/nanogui.h>
#include <iostream>

using namespace nanogui;

int main() {
    nanogui::init();

    Screen *screen = new Screen(Vector2i(500, 700), "NanoGUI test");
    screen->set_visible(true);

    nanogui::run();

    nanogui::shutdown();

    return 0;
}
