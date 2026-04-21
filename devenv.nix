{ pkgs, ... }:

let
  # Runtime shared libraries the viewer dlopens at startup. These have to be on
  # LD_LIBRARY_PATH because NixOS doesn't expose system libraries via standard
  # paths.
  runtimeLibs = with pkgs; [
    libGL
    libxkbcommon
    wayland
    xorg.libX11
    xorg.libXcursor
    xorg.libXext
    xorg.libXi
    xorg.libXinerama
    xorg.libXrandr
  ];
in {
  languages.cplusplus.enable = true;

  packages = with pkgs; [
    # Build tools
    cmake
    ninja
    pkg-config
    gcc
    clang-tools

    # C++ libraries the project links against
    eigen
    openmesh
    (suitesparse.override { enableCuda = false; })

    # Viewer-side platform deps (windowing, GL dispatch, IPC)
    dbus
    libGL
    libGLU
    libxkbcommon
    wayland
    wayland-protocols
    wayland-scanner
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    xorg.libXinerama
    xorg.libXrandr
    xorg.libXxf86vm
  ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

  enterShell = ''
    cmake --version
  '';
}
