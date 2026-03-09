#!/usr/bin/env python3
"""
Setup script for PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

Defaults:
- Optimization enabled, OpenMP enabled, and Eigen debug disabled.
- Optional precision toggle: set env DIFFMPM_DOUBLE=1 to build with double precision
  (defines DIFFMPM_USE_DOUBLE; you also need to make pch.h/types honor this macro).
"""

"""
Setup script for DiffMPM Python bindings (release-focused).

- Adds /fp:precise and /arch:AVX2 for consistency with the C++ project build.
- Supports DIFFMPM_DOUBLE=1 for double precision.
- Supports DIFFMPM_DETERMINISTIC=1 for reproducible builds (disables LTO).
"""

import os, sys
from pathlib import Path
from setuptools import setup

# Try to import PyTorch's CppExtension for better compatibility
try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    TORCH_AVAILABLE = True
except ImportError:
    from pybind11.setup_helpers import Pybind11Extension as CppExtension
    from pybind11.setup_helpers import build_ext as BuildExtension
    TORCH_AVAILABLE = False

project_root = Path(__file__).parent
include_dir = project_root / "include"
diffmpm_lib_dir = project_root / "DiffMPMLib3D"

is_win = sys.platform == "win32"

# --- Configuration ---

extra_compile_args = []
extra_link_args = []
define_macros = [("_USE_MATH_DEFINES", None), ("NDEBUG", None), ("EIGEN_NO_DEBUG", None)]

# Diagnostics toggle: export DIFFMPM_DIAGNOSTICS=1 to enable
diagnostics = False # os.environ.get("DIFFMPM_DIAGNOSTICS", "0") == "1"
diagnostics = False
deterministic = False  # ✅ Default to performance mode (LTO enabled)
if diagnostics:
    define_macros.append(("DIAGNOSTICS", None))
    # Prefer reproducible build (disable LTO) when diagnostics are on
    deterministic = True
    print("[setup.py] DIAGNOSTICS enabled: defining -DDIAGNOSTICS and disabling LTO")

# PyTorch integration: Auto-detect or force via environment variable
# Set DIFFMPM_WITH_TORCH=0 to explicitly DISABLE torch support
force_disable = os.environ.get("DIFFMPM_WITH_TORCH", "auto") == "0"

if force_disable or not TORCH_AVAILABLE:
    print("[setup.py] INFO: PyTorch integration DISABLED")
    if force_disable:
        print("[setup.py]       (explicitly disabled via DIFFMPM_WITH_TORCH=0)")
    else:
        print("[setup.py]       (PyTorch not found)")
    with_torch = False
else:
    # PyTorch is available and not disabled
    import torch
    with_torch = True
    define_macros.append(("DIFFMPM_WITH_TORCH", None))
    print(f"[setup.py] OK: PyTorch integration ENABLED (torch {torch.__version__})")
    print(f"[setup.py]     Using torch.utils.cpp_extension.CppExtension for compatibility")


# Precision toggle
if os.environ.get("DIFFMPM_DOUBLE", "0") == "1":
    define_macros.append(("DIFFMPM_USE_DOUBLE", None))

if is_win:
    extra_compile_args = [
        "/std:c++17",
        "/Ox",
        "/MP",
        "/openmp",
        "/D_OPENMP",
        "/utf-8",
        "/EHsc",
        "/DNOMINMAX",
        "/permissive-",
        "/fp:fast",  # Use precise floating-point model, disable fast-math
        "/arch:AVX2",   # Specify AVX2 instruction set
    ]
    
    if not deterministic:
        # Enable Link-Time Optimization (LTO) only in performance mode
        extra_compile_args += ["/GL"]
        extra_link_args += ["/openmp", "/LTCG"]
    else:
        # For deterministic builds, add flags that reduce sources of variance
        define_macros += [("EIGEN_DONT_PARALLELIZE", None)]

else: # Linux/macOS
    extra_compile_args = [
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-fno-fast-math",      # Equivalent to /fp:precise
        "-ffp-contract=off",
        "-mavx2",              # Equivalent to /arch:AVX2
    ]
    extra_link_args = ["-fopenmp"]
    
    if not deterministic:
        if os.environ.get("NO_LTO", "0") != "1":
            extra_compile_args += ["-flto"]
            extra_link_args += ["-flto"]
    else:
        define_macros += [("EIGEN_DONT_PARALLELIZE", None)]

# Add debug symbols when diagnostics are enabled
if diagnostics:
    if is_win:
        extra_compile_args += ["/Zi"]
        extra_link_args    += ["/DEBUG"]
    else:
        extra_compile_args += ["-g"]

sources = [
    "bind/bind.cpp",
    str(diffmpm_lib_dir / "CompGraph.cpp"),
    str(diffmpm_lib_dir / "E2ESession.cpp"),  # ✅ Add session mode!
    str(diffmpm_lib_dir / "ForwardSimulation.cpp"),
    str(diffmpm_lib_dir / "BackPropagation.cpp"),
    str(diffmpm_lib_dir / "Elasticity.cpp"),
    str(diffmpm_lib_dir / "PointCloud.cpp"),
    str(diffmpm_lib_dir / "Grid.cpp"),
    str(diffmpm_lib_dir / "GridNode.cpp"),
    str(diffmpm_lib_dir / "MaterialPoint.cpp"),
    str(diffmpm_lib_dir / "GeometryLoading.cpp"),
    str(diffmpm_lib_dir / "Tensor3x3x3x3.cpp"),
    str(diffmpm_lib_dir / "SphereUnionSurfacing.cpp"),
]

include_dirs = [
    str(include_dir),
    str(include_dir / "Eigen"),
    str(include_dir / "glm"),
    str(include_dir / "igl"),
    str(include_dir / "cereal"),
    str(include_dir / "json"),
    str(include_dir / "happly"),
    str(include_dir / "stb"),
    str(include_dir / "qrsvd"),
    str(include_dir / "autodiff"),
    str(include_dir / "args"),
    str(include_dir / "sse2neon"),
    str(include_dir / "unsupported"),
    str(diffmpm_lib_dir),
]

# Build extension module
# When PyTorch is available, CppExtension automatically handles torch libraries
ext_modules = [
    CppExtension(
        name="diffmpm_bindings",
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="diffmpm",
    version="3.0.0",
    author="Changyong Song",
    description="Python bindings for PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing",
    long_description=(project_root / "README.md").read_text(encoding="utf-8") if (project_root / "README.md").exists() else "",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.26.4",
        "pybind11>=2.10.0",
        "pyyaml>=6.0"
    ],
)
