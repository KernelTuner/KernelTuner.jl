using KernelTuner
using Test
using PythonCall

# Uncomment for debugging
# using CondaPkg
# # Print the status to see if kernel-tuner is listed as 'installed'
# CondaPkg.status()
# # Manually force a resolution if it's empty
# CondaPkg.resolve()

@testset "KernelTuner.jl" begin

    @testset "Environment Setup" begin
        # Verify the Python module was actually loaded
        @test string(pytype(KernelTuner.kt[])) == "<class 'module'>"  # Check if it's a Python module
    end

    @testset "Basic API" begin
        # Test a simple function call
        ver = string(KernelTuner.kt[].__version__)
        @test ver isa String
    end

    # TODO write actual test
    # @testset "Data Conversion (NumPy/Julia)" begin
    #     # Testing that a Julia array goes in and a result comes out
    #     data = Float32[1.0, 2.0, 3.0]

    #     # Test a wrapped function
    #     # PythonCall will wrap the Julia array as a NumPy-compatible object
    #     result = KernelTuner.tune_kernel(data)

    #     @test result isa Py  # Or whatever type you expect back
    # end
end
