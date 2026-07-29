<div align="center">
  <img width="500px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/KernelTuner-logo.png"/>
</div>

---
[![Build Status](https://github.com/KernelTuner/kerneltuner.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/KernelTuner/kerneltuner.jl/actions/workflows/Test.yml)
[![CodeCov Badge](https://codecov.io/gh/KernelTuner/kernel_tuner/branch/master/graph/badge.svg)](https://codecov.io/gh/KernelTuner/kernel_tuner)
[![SonarCloud Badge](https://sonarcloud.io/api/project_badges/measure?project=KernelTuner_kernel_tuner&metric=alert_status)](https://sonarcloud.io/dashboard?id=KernelTuner_kernel_tuner)
[![FairSoftware Badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
---

**KernelTuner.jl** brings Kernel Tuner's wide array of auto-tuning capabilities to the Julia ecosystem, providing the first full-fledged auto-tuning framework for Julia. For more information and [documentation](https://kerneltuner.github.io/kernel_tuner/stable/index.html), please see the [main Kernel Tuner repository](https://github.com/KernelTuner/kernel_tuner).

## GPU kernel writing

Place any tunable parameters as final arguments in the kernel using the syntax: `::Val{TILE} = Val(16)`: see [`matmulkernel.jl`](examples/matmulkernel.jl) in [examples](examples). This ensures Kernel Tuner works while being minimally invasive and allows your kernel to be ran separately as well. 

## Input parameters

As per [`example.jl`](examples/example.jl) in [examples](examples), the following input parameters need to be passed to `tune_kernel`:
- The kernel name, `"kernel!"` and its location "kernelfile.jl"
- Threadblock dimensions: each threadblock dimension is a parameter that can be tuned: `["BLOCKDIM_X", ]` 
- Grid dimensions: determines the number of blocks, e.g. `("BLOCKDIM_X*NUMBLOCKS_X",)` (combinations of parameters are allowed)
- Tunable parameters, e.g. `tune_params = [("BLOCKDIM_X", [16, 32]),("NUMBLOCKS_X", [16, 32])]`
- Optional: strings of restrictions on the tunable parameters, search strategies, and many more (see [Kernel Tuner documentation](https://kerneltuner.github.io/kernel_tuner/stable/index.html))

Then run the as follows: `tune_kernel("kernel!",kernelfile.jl", ["BLOCKDIM_X", ], ("BLOCKDIM_X*NUMBLOCKS_X",) , [("BLOCKDIM_X", [16, 32]),("NUMBLOCKS_X", [16, 32])])` 

## Notes
- The CPU backend is always available, but not selected by default. On devices with GPU backends, the GPU backend is selected by default. To use the CPU backend, you must pass `compiler_options=["CPU"]`. 

## Requirements

Running this package requires:
- Julia >= 1.11
- KernelAbstractions (most recently tested with latest master, #a8022b2)
- Either CUDA, AMDGPU, OneAPI or Metal packages
- PythonCall and CondaPkg

## Development & Testing

To develop and test running the package locally, install the main Kernel Tuner repository alongside this one, create a Python environment, run `pip install -e .` in it, and run `pytest` to make sure it is installed correctly. 
Following this, `cd` to the local path of this repository, in `CondaPkg.toml` comment the version and set the correct local path instead, and run `julia --project=.` followed by `] test]` to test the Julia installation.  

## Examples
Given the following simple example:

```
using KernelAbstractions

# Define the kernel
@kernel function vector_add!(C, A, B, n)
    i = @index(Global)
    if i <= n
        @inbounds C[i] = A[i] + B[i]
    end
end

# Set up the arguments for the kernel
backend = CPU()     # or your GPU backend of choice
size = Int32(10000000)
rsize = size ÷ 4  # Repeat the base array to reach the desired size
a = repeat(Float32[1, 2, 3, 4], outer=rsize)
b = repeat(Float32[10, 20, 30, 40], outer=rsize)
c = zeros(Float32, size)

# Run the kernel
kernel! = vector_add!(backend, 128)
kernel!(c, a, b, size, ndrange=size)
synchronize(backend)
println("Kernel completed. First 10 results: ", c[1:10])
```

It can also be run directly via Kernel Tuner as follows:
```
using KernelTuner

# Set up the arguments for the kernel
c = zeros(Float32, size)
arguments = [c, a, b, size]

# Run the kernel
results = KernelTuner.run_kernel(
    "vector_add!",
    kernel_code,    # can be either a string or the path to the file
    (size,),
    arguments,
    [],
    lang="Julia",
    compiler_options=["CPU"],
)
println("Kernel completed. First 10 results: ", results[1][1:10])
```

The interesting part of course is tuning. 
This can be done as follows:
```
tune_params = [
    ("block_size_x", [32, 64, 128, 256, 512])
]

# Tune the kernel
results = KernelTuner.tune_kernel(
    "vector_add!",
    kernel_code,
    (size,),
    [c, a, b, size],
    tune_params,
    grid_div_x=1, grid_div_y=1,
    block_size_names=["block_size_x"],
    lang="Julia",
    compiler_options=["CPU"],
)
```

This is a simple example, much more extensive functionality is available. 
For this example, you might want to add `answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],` for output verification. 
See the [Kernel Tuner documentation](https://kerneltuner.github.io/kernel_tuner/stable/index.html) for more information. 
