# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{
  description = "OpenShell development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crate2nix = {
      url = "github:nix-community/crate2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      crate2nix,
      flake-utils,
      nixpkgs,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (import rust-overlay)
            (import ./nix/overlay.nix)
          ];
        };
        lib = pkgs.lib;
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        shellPackages = with pkgs; [
          rustToolchain
          # Required to find packages
          pkg-config
          # Required for bindgen generation.
          llvmPackages.libclang
          # system dependency for openshell-prover
          z3
        ];
        shellEnv = {
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };
        generatedCargoNix = crate2nix.tools.${system}.generatedCargoNix {
          name = "openshell";
          src = ./.;
          cargo = rustToolchain;
        };
        cargoNix = pkgs.callPackage generatedCargoNix {
          defaultCrateOverrides = pkgs.defaultCrateOverrides // {
            "openshell-core" = prev: {
              src = pkgs.runCommand "openshell-core-src" { } ''
                mkdir -p "$out/crates" "$out/proto"
                cp -R ${prev.src} "$out/crates/openshell-core"
                cp -R ${./proto}/. "$out/proto/"
              '';
              workspace_member = "crates/openshell-core";
            };
            "openshell-providers" = prev: {
              src = pkgs.runCommand "openshell-providers-src" { } ''
                mkdir -p "$out/crates" "$out/providers"
                cp -R ${prev.src} "$out/crates/openshell-providers"
                cp -R ${./providers}/. "$out/providers/"
              '';
              workspace_member = "crates/openshell-providers";
            };
            "protobuf-src" = _: {
              postConfigure = ''
                build_dir="$(pwd)/target/build/protobuf-src.out/install"
                install_dir="$lib/lib/protobuf-src.out/install"

                export INSTALL_DIR="$install_dir"

                substituteInPlace target/env \
                  --replace "$build_dir" "$install_dir"
              '';
            };
            "z3-sys" = _: {
              nativeBuildInputs = with pkgs; [
                pkg-config
                llvmPackages.libclang
              ];
              buildInputs = with pkgs; [
                z3
              ];
              LIBCLANG_PATH = "${pkgs.lib.getLib pkgs.llvmPackages.libclang}/lib";
            };
          };
          buildRustCrateForPkgs =
            pkgs:
            pkgs.buildRustCrate.override {
              rustc = rustToolchain;
              cargo = rustToolchain;
            };
        };
        releaseCrates = lib.mapAttrs' (
          name: crate: lib.nameValuePair "${name}-release" crate.build
        ) cargoNix.workspaceMembers;
        vmRuntimeCompressed = pkgs.callPackage ./nix/vm-runtime.nix {
          openshellSandbox = releaseCrates."openshell-sandbox-release";
        };
        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
        };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = shellPackages;

            env = shellEnv;
          };
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          vm = pkgs.mkShell {
            packages =
              shellPackages
              ++ (with pkgs; [
                e2fsprogs
                nftables
                qemu
                zstd
              ]);

            env = shellEnv // {
              OPENSHELL_VM_RUNTIME_COMPRESSED_DIR = "${vmRuntimeCompressed}";
            };
          };
        };

        packages = {
          all = cargoNix.allWorkspaceMembers;
        }
        // releaseCrates
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          inherit vmRuntimeCompressed;
        };

        formatter = treefmtEval.config.build.wrapper;
      }
    );
}
