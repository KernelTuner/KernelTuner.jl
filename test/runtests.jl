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
end
