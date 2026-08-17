using KernelTuner
using Test
using PythonCall    # for the first test

# Uncomment for debugging
# using CondaPkg
# # Print the status to see if kernel-tuner is listed as 'installed'
# CondaPkg.status()
# # Manually force a resolution if it's empty
# CondaPkg.resolve()

# set default test arguments (override via command line argument)
config = Dict(
    :backend => "default", # choose from default (auto-detected), CPU, and the available GPU backends
    :optimization_algorithms => ["random_sample", "genetic_algorithm", "simulated_annealing"], # choose from `strategy_map` in interface.py
    :CI => false, # whether this test is running on the CI (skips heavy / GPU required tests)
)
for arg in ARGS
    if startswith(arg, "--backend=")
        config[:backend] = split(arg, "=")[2]
    elseif arg == "--optalgs="
        config[:optimization_algorithms] = split(split(arg, "=")[2], ",")
    elseif arg == "--CI"
        config[:CI] = true
    end
end
compiler_options=[]
if config[:backend] == "default"
    @info "Testing with default backend (auto-detected)"
else
    @info "Testing with backend: $(config[:backend])"
    compiler_options = [config[:backend]]
end


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

    @testset "Run a kernel" begin
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

        # Verify the code string was created successfully
        @test !isempty(kernel_code)

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
            compiler_options=compiler_options,
        )

        # Verify the results
        answer = repeat(Float32[11, 22, 33, 44], outer=rsize)
        @test length(results) == length(arguments)
        @test length(results[1]) == length(arguments[1]) == length(answer)
        @test results[1] == answer
    end

    @testset "Tune a kernel" begin
        # Defining a simple vector add kernel
        kernel_code = """
        using KernelAbstractions

        @kernel function vector_add!(C, A, B, n, block_size_x)
            i = @index(Global)
            if i <= n
                @inbounds C[i] = A[i] + B[i]
            end
        end
        """

        # Set up the arguments and tunable parameters
        size = Int32(10000000)
        rsize = size ÷ 4  # Repeat the base array to reach the desired size
        a = repeat(Float32[1, 2, 3, 4], outer=rsize)
        b = repeat(Float32[10, 20, 30, 40], outer=rsize)
        c = zeros(Float32, size)
        tune_params = [
            ("block_size_x", [32, 64, 128, 256])
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
            answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],
            compiler_options=compiler_options,
        )

        # Verify the results
        @test results[1] isa Tuple
        @test length(results[1]) == length(tune_params[1][2])  # number of configurations tested
        @test results[2] isa Dict
        @test "best_config" in keys(results[2])
    end

    @testset "Tune a tunable kernel" begin
        # Defining a vector add kernel with work per thread parameter
        kernel_code = """
        using KernelAbstractions

        @kernel function vector_add!(C, A, B, n, block_size_x,
                             ::Val{WORK_PER_THREAD}) where {WORK_PER_THREAD}
            tid = @index(Global)
            first_i = (tid - 1) * WORK_PER_THREAD + 1

            for offset in 0:(WORK_PER_THREAD - 1)
                i = first_i + offset
                if i <= n
                    @inbounds C[i] = A[i] + B[i]
                end
            end
        end
        """

        # Set up the arguments and tunable parameters
        size = Int32(10000000)
        rsize = size ÷ 4  # Repeat the base array to reach the desired size
        a = repeat(Float32[1, 2, 3, 4], outer=rsize)
        b = repeat(Float32[10, 20, 30, 40], outer=rsize)
        c = zeros(Float32, size)
        tune_params = [
            ("block_size_x", [32, 64, 128, 256, 512]),
            ("work_per_thread", [1, 2, 4, 8]),
        ]

        # Tune the kernel
        results = KernelTuner.tune_kernel(
            "vector_add!",
            kernel_code,
            (size,),
            [c, a, b, size],
            tune_params,
            grid_div_x=["block_size_x", "work_per_thread"],
            block_size_names=["block_size_x"],
            lang="Julia",
            answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],
            compiler_options=compiler_options,
        )

        # Verify the results
        @test results[1] isa Tuple
        @test length(results[1]) == length(tune_params[1][2]) * length(tune_params[2][2])  # number of configurations tested
        @test results[2] isa Dict
        @test "best_config" in keys(results[2])
    end

    @testset "Test the optimization algorithms" begin
        for alg in config[:optimization_algorithms]
            @info "Testing optimization algorithm: $alg"

            # Defining a vector add kernel with work per thread parameter
            kernel_code = """
            using KernelAbstractions

            @kernel function vector_add!(C, A, B, n, block_size_x,
                                ::Val{WORK_PER_THREAD}) where {WORK_PER_THREAD}
                tid = @index(Global)
                first_i = (tid - 1) * WORK_PER_THREAD + 1

                for offset in 0:(WORK_PER_THREAD - 1)
                    i = first_i + offset
                    if i <= n
                        @inbounds C[i] = A[i] + B[i]
                    end
                end
            end
            """

            # Set up the arguments and tunable parameters
            size = Int32(10000000)
            rsize = size ÷ 4  # Repeat the base array to reach the desired size
            a = repeat(Float32[1, 2, 3, 4], outer=rsize)
            b = repeat(Float32[10, 20, 30, 40], outer=rsize)
            c = zeros(Float32, size)
            tune_params = [
                ("block_size_x", [32, 64, 128, 256, 512]),
                ("work_per_thread", [1, 2, 4, 8]),
            ]
            max_fevals = (length(tune_params[1][2]) * length(tune_params[2][2])) - 1

            # Tune the kernel
            results = KernelTuner.tune_kernel(
                "vector_add!",
                kernel_code,
                (size,),
                [c, a, b, size],
                tune_params,
                grid_div_x=["block_size_x", "work_per_thread"],
                block_size_names=["block_size_x"],
                lang="Julia",
                answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],
                compiler_options=compiler_options,
                strategy=alg,
                strategy_options=[("max_fevals", max_fevals)]
            )

            # Verify the results
            @test results[1] isa Tuple
            @test length(results[1]) <= max_fevals # number of configurations tested
            @test results[2] isa Dict
            @test "best_config" in keys(results[2])
        end
    end

    # this is an intensive GPU test, don't run on CI or with CPU backend
    if !config[:CI] && config[:backend] != "CPU"
        @testset "Full integration test with the unmqr_tsmqr kernel" begin
            include("integration_test.jl")
            run_integration_test(compiler_options)
            run_integration_test(compiler_options, insert_bad_data=true)
        end
    end
end
