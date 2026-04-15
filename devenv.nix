{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  env.GREET = "devenv";
  packages = with pkgs; [
    git
    cmake
    clang-tools
    pkg-config
    ninja
    gcc
    eigen
    openmesh
    dbus
    wayland-scanner
    wayland
    wayland-protocols
    libxkbcommon

    (suitesparse.override {
      enableCuda = false;
    })

    # OpenGL & X11
    libGL
    libGLU
    xorg.libX11
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi
    xorg.libXxf86vm
  ];

  languages.cplusplus.enable = true;

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
    with pkgs;
    [
      libGL
      xorg.libX11
      xorg.libXext
      xorg.libXinerama
      xorg.libXcursor
      xorg.libXrandr
      xorg.libXi
      wayland
      libxkbcommon
    ]
  );

  enterShell = ''
    cmake --version
    git --version
  '';
}
