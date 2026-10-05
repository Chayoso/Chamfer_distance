{
  description = "Reproduction of 'On the Structural Failure of Chamfer Distance in 3D Shape Optimization'";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      common = ps: with ps; [ numpy scipy matplotlib trimesh rtree pyyaml tqdm pytest ];
      pythonCpu = pkgs.python3.withPackages (ps: common ps ++ [ ps.torch ]);
      # GPU shell: Nix python + libs, torch comes from a prebuilt CUDA wheel in .venv-gpu
      # (nixpkgs' CUDA torch needs hours of local compiles for uncached packages).
      pythonGpu = pkgs.python3.withPackages common;
    in
    {
      # default: CPU-only, prebuilt from cache.nixos.org
      devShells.${system} = {
        default = pkgs.mkShell { packages = [ pythonCpu ]; };

        # nix develop .#gpu : CUDA wheel venv (scripts/setup_gpu_env.sh) + PRIME render-offload.
        gpu = pkgs.mkShell {
          packages = [ pythonGpu pkgs.uv ];
          shellHook = ''
            export __NV_PRIME_RENDER_OFFLOAD=1
            export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0
            export __GLX_VENDOR_LIBRARY_NAME=nvidia
            export __VK_LAYER_NV_optimus=NVIDIA_only
            export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib   # triton can not find libcuda via /sbin/ldconfig on NixOS
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
            [ -d .venv-gpu ] && export PYTHONPATH="$PWD/.venv-gpu/lib/python3.14/site-packages:$PYTHONPATH"
          '';
        };
      };
    };
}
