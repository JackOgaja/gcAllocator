# MIT License
# 
# Copyright (c) 2025 Jack Ogaja
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
from setuptools import setup, find_packages

def get_extensions():
    """Build the C++ and CUDA extensions with lazy torch import"""
    
    # CRITICAL: Import torch only when building extensions
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        raise ImportError(
            "PyTorch is required to build gcAllocator. "
            "Please install PyTorch first: pip install torch"
        )
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Building CPU-only version.")
    
    sources = [
        'gcAllocator/src/gc_allocator_core.cpp',
        'gcAllocator/src/allocator_stats_instrument.cpp',
        'gcAllocator/src/retry_strategy.cpp',  
        'gcAllocator/src/python_bindings.cpp',
    ]
    
    include_dirs = [
        'gcAllocator/src',
    ]
    
    define_macros = []
    
    # Add version macros for PyTorch compatibility
    torch_version = torch.__version__.split('.')
    define_macros.append(('TORCH_VERSION_MAJOR', torch_version[0]))
    define_macros.append(('TORCH_VERSION_MINOR', torch_version[1]))
    
    extra_compile_args = {
        'cxx': ['-std=c++17', '-O3'],
        'nvcc': ['-O3']
    }
    
    # Add debug symbols if requested
    if os.getenv('DEBUG_BUILD', '0') == '1':
        extra_compile_args['cxx'].append('-g')
        extra_compile_args['nvcc'].append('-g')
        define_macros.append(('DEBUG_BUILD', 1))
    
    ext = CUDAExtension(
        name='gcAllocator.gc_allocator_core',
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        libraries=['cudart', 'c10_cuda', 'torch'],
    )
    
    return [ext]

# Import BuildExtension lazily
try:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass = {'build_ext': BuildExtension}
except ImportError:
    # Fallback if torch not available yet
    from setuptools.command.build_ext import build_ext
    cmdclass = {'build_ext': build_ext}

setup(
    name='gcAllocator',
    version='0.1.0',
    author='Jack Ogaja',
    description='Graceful CUDA Allocator for PyTorch with OOM handling',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
    install_requires=[
        'torch>=1.9.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'isort',
            'flake8',
        ],
    },
    python_requires='>=3.7',
)