# Consider the the matmul kernel that you would execute as follows on NVIDIA:

using KernelAbstractions
using CUDA
backend = CUDABackend()
N, M, R = 1024, 1024, 1024
input1 = rand(Float32, N, R)
input2 = rand(Float32, R, M)
output = zeros(Float32, N, M)
TILEDIM = 16
coalesced_matmul_kernel_v2!(backend, (TILEDIM,TILEDIM))((output, input1, input2, N, R, M, Val(TILEDIM)), ndrange=(N, M))
KernelAbstractions.synchronize(backend)

# You can now tune the TILEDIM parameter using KernelTuner.jl:

input_args=[ input1, input2, output, N, R, M]
tune_params = [("TILESIZE", [16, 32]),]
restrictions = ["TILESIZE<=32", "TILESIZE>=16"]
tune_kernel("brdkernel_large_v2!", "matmulkernel.jl", ["TILESIZE", "TILESIZE"],("N","M"),  input_args, tune_params; restrictions=restrictions)


# In certain cases, your input parameters are not static, but depend on the tunable parameters.
# Let us pretend you only want to matmul the first 2TILESIZEx2TILESIZE elements of the input matrices, you can pass this as follows:

input_args=[ parameterized_input("TILESIZE", [16, 32], f(ts)=view(input1, 1:ts, 1:ts)), 
            parameterized_input("TILESIZE", [16, 32], f(ts)=view(input2, 1:ts, 1:ts)), 
            parameterized_input("TILESIZE", [16, 32], f(ts)=view(output, 1:ts, 1:ts)), 
            N, R, M]
tune_kernel("brdkernel_large_v2!", "matmulkernel.jl", ["TILESIZE", "TILESIZE"],("N","M"),  input_args, tune_params; restrictions=restrictions)



