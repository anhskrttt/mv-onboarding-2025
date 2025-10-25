# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
    ROCM_HOME,
    TORCH_LIB_PATH,
)

# NOTE: This is extension name, please do not change
library_name = "extension_cpp"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

os.environ["CC"] = f"{ROCM_HOME}/bin/hipcc"
os.environ["CXX"] = f"{ROCM_HOME}/bin/hipcc"
os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_HIP", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources=[
                "extension_cpp/csrc/muladd.cpp", # CPU implementation
                # TODO(Step 2): Add HIP source files here
                
            ],
            library_dirs=[f"{ROCM_HOME}/lib", TORCH_LIB_PATH],
            runtime_library_dirs=[f"{ROCM_HOME}/lib", TORCH_LIB_PATH],
            extra_compile_args=[
                "-O3",
                "-DNDEBUG",
                "-std=c++17",
                "--offload-arch=gfx90a",
                "-D__HIP_PLATFORM_AMD__=1",
                "-DUSE_ROCM",
            ],
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
