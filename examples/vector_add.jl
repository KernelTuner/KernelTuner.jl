# Simple vector add kernel

using KernelAbstractions

@kernel function vector_add!(C, A, B, n)
    i = @index(Global)
    if i <= n
        @inbounds C[i] = A[i] + B[i]
    end
end

backend = CPU()
size = Int32(10000000)
rsize = size ÷ 4  # Repeat the base array to reach the desired size
a = repeat(Float32[1.11, 2.22, 3.33, 4.44], outer=rsize)
b = repeat(Float32[10, 20, 30, 40], outer=rsize)
c = zeros(Float32, size)

kernel! = vector_add!(backend, 128)
event = kernel!(c, a, b, size, ndrange=size)
synchronize(backend)
println("Kernel completed. First 10 results: ", c[1:10])
println(typeof(c))

# Execution via Kernel tuner

using KernelTuner

# Defining a simple vector add kernel
kernel_code = """
using KernelAbstractions

@kernel function vector_add!(C, A, B, n)
    i = @index(Global)
    if i <= n
        @inbounds C[i] = A[i] + B[i]
    end
end
"""

# Set up the arguments for the kernel
size = Int32(10000000)
rsize = size ÷ 4  # Repeat the base array to reach the desired size
a = repeat(Float32[1, 2, 3, 4], outer=rsize)
b = repeat(Float32[10, 20, 30, 40], outer=rsize)
c = zeros(Float32, size)
arguments = [c, a, b, size]

# Run the kernel
results = KernelTuner.run_kernel(
    "vector_add!",
    kernel_code,
    (size,),
    arguments,
    [],
    lang="Julia",
    compiler_options=["CPU"],
)
