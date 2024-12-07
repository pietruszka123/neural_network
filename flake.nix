{
  description = "A platformer written in Rust using Bevy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      projectName = "bevy-platformer";
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          vscode-langservers-extracted
          rustup
          pkg-config
          kaggle
        ];
        buildInputs = with pkgs; [
          alsa-lib.dev

          libevdev
          linuxKernel.packages.linux_6_6.perf
          udev.dev
          cargo-flamegraph
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.vulkan-loader
          pkgs.libGL
          pkgs.libxkbcommon
          pkgs.wayland
          pkgs.xorg.libX11
          pkgs.xorg.libXcursor
          pkgs.xorg.libXi
          pkgs.xorg.libXrandr

        ];

        shellHook = ''
          printf '\x1b[36m\x1b[1m\x1b[4mTime to develop ${projectName}!\x1b[0m\n\n'
        '';
      };
    };
}
