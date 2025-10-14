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
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check for CUDA availability
import torch
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Building CPU-only version.")
    
def get_extensions():
    """Build the C++ and CUDA extensions"""
    
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

setup(
    name='gcAllocator',
    version='0.1.0',
    author='Jack Ogaja',
    description='Graceful CUDA Allocator for PyTorch with OOM handling',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
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
    package_data={"gcallocator": ["*.py"]},
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Artificial Intelligence :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
