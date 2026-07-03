<div align="center">
  <img width="500px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/KernelTuner-logo.png"/>
</div>

---
[![Build Status](https://github.com/KernelTuner/kerneltuner.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/KernelTuner/kerneltuner.jl/actions/workflows/Test.yml)
[![CodeCov Badge](https://codecov.io/gh/KernelTuner/kernel_tuner/branch/master/graph/badge.svg)](https://codecov.io/gh/KernelTuner/kernel_tuner)
[![SonarCloud Badge](https://sonarcloud.io/api/project_badges/measure?project=KernelTuner_kernel_tuner&metric=alert_status)](https://sonarcloud.io/dashboard?id=KernelTuner_kernel_tuner)
[![FairSoftware Badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
---

**KernelTuner.jl** brings Kernel Tuner's wide array of auto-tuning capabilities to the Julia ecosystem with this minimal wrapper implementation, providing the first full-fledged auto-tuning framework for Julia. For more information and [documentation](https://kerneltuner.github.io/kernel_tuner/stable/index.html), please see the [main Kernel Tuner repository](https://github.com/KernelTuner/kernel_tuner).

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

## Requirements

Running this package requires:
- KernelAbstractions (most recently tested with latest master, #a8022b2)
- Either CUDA, AMDGPU, OneAPI or Metal packages
- PythonCall and CondaPkg

## Development & Testing

To develop and test running the package locally, install the main Kernel Tuner repository alongside this one, and in `CondaPkg.toml` comment the PIP version and set the correct local path instead. 