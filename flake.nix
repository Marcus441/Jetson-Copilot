{
  description = "Jetson C++ flake with g2o + Ceres";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pythonPkgs = pkgs.python313Packages;
      in {
        devShells.default = pkgs.mkShell {
          name = "jetson-python";

          packages = with pythonPkgs; [
            fastapi
            llama-cpp-python
            python
            requests
            torch
            transformers
            uvicorn
          ];

          shellHook = ''
            echo "╔═══════════════════════════════════════╗"
            echo "║    Jetson Python Development Shell    ║"
            echo "╚═══════════════════════════════════════╝"
            echo -n " Python: "
            python3 --version
            echo -n " System: "
            uname -m
          '';
        };
      }
    );
}
