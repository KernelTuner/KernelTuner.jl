using KernelTuner
using Test
using PythonCall    # for the first test
using KernelTuner: get_observer

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
    :minimal => false, # only runs the lightest tests (import, simple run_kernel, simple tune_kernel)
)
for arg in ARGS
    if startswith(arg, "--backend=")
        config[:backend] = split(arg, "=")[2]
    elseif arg == "--optalgs="
        config[:optimization_algorithms] = split(split(arg, "=")[2], ",")
    elseif arg == "--CI"
        config[:CI] = true
    elseif arg == "--minimal"
        config[:minimal] = true
    end
end
compiler_options=[]
if config[:backend] == "default"
    @info "Testing with default backend (auto-detected)"
else
    @info "Testing with backend: $(config[:backend])"
    compiler_options = [config[:backend]]
end

if count(f -> startswith(f, "temp_") && endswith(f, ".jl") && isfile(f), readdir()) > 0
    @warn "Found temporary files from a previous test run. Deleting now."
    rm.(filter(f -> startswith(f, "temp_") && endswith(f, ".jl") && isfile(f), readdir()); force=true)
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
        @test results isa Vector
        @test results[1] isa Vector{Float32}
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
            block_size_names=["block_size_x"],
            lang="Julia",
            answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],
            compiler_options=compiler_options,
        )

        # Verify the results
        @test results[1] isa Vector
        @test length(results[1]) == length(tune_params[1][2])  # number of configurations tested
        @test results[1][1] isa Dict
        @test results[2] isa Dict
        @test "best_config" in keys(results[2])
    end

    @testset "Interrupts" begin
        # Kernel Tuner stops with a KeyboardInterrupt when interrupted, which is raised as an InterruptException in Julia
        namespace = pydict()
        pyexec("""
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        def failed(*args, **kwargs):
            raise ValueError("not an interrupt")
        """, namespace)
        interrupted, failed = namespace["interrupted"], namespace["failed"]
        @test_throws InterruptException KernelTuner.call_kernel_tuner(interrupted, 1; a=2)
        @test_throws PyException KernelTuner.call_kernel_tuner(failed)
        @test pyconvert(Int, KernelTuner.call_kernel_tuner(pybuiltins.len, pylist([1, 2]))) == 2
    end

    @testset "Julia-native metrics and AccuracyObserver" begin
        # metrics are converted to an ordered Python dict, Julia functions receive a Julia Dict
        metrics = KernelTuner.convert_metrics([
            "GFLOP/s" => "1.0 / (time / 1e3)",
            "work-items" => p -> p["block_size_x"] * p["block_size_y"],
            "double" => p -> 2 * p["work-items"],
        ])
        @test pyconvert(Vector{String}, pylist(metrics.keys())) == ["GFLOP/s", "work-items", "double"]
        params = pydict(Dict("time" => 2.0, "block_size_x" => 32, "block_size_y" => 4, "times" => pylist([1.0, 3.0])))
        params = pyimport("kernel_tuner.util").process_metrics(params, metrics)
        @test pyconvert(Int, params["work-items"]) == 128
        @test pyconvert(Int, params["double"]) == 256
        @test pyconvert(Float64, params["GFLOP/s"]) ≈ 500.0
        @test KernelTuner.params_to_julia(params)["times"] == [1.0, 3.0]
        @test_throws ErrorException KernelTuner.convert_metrics(["bad" => 42])

        # AccuracyObserver with a named metric and with a Julia function
        np = pyimport("numpy")
        answers, outputs = pylist([np.array([1.0, 2.0]), nothing]), pylist([np.array([1.0, 4.0]), np.array([0.0, 0.0])])
        obs = AccuracyObserver("absolute error")
        obs.process_output(answers, outputs)
        @test pyconvert(Float64, obs.get_results()["error"]) ≈ 1.0   # mean absolute error of [0, 2]
        obs = AccuracyObserver((a, o) -> (@assert a isa Array && o isa Array; maximum(abs.(a .- o))); key="max_error")
        obs.process_output(answers, outputs)
        @test pyconvert(Float64, obs.get_results()["max_error"]) ≈ 2.0
    end

    if !config[:minimal]
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

            # set up cache
            cache_file = "julia_test_cache.json"
            cache_filepath = joinpath(pwd(), cache_file)
            rm(cache_filepath; force=true)  # clear cache to avoid using old results
            @test isfile(cache_filepath) == false

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

            # set up observers for energy measurement
            observers = []
            energy_observer_fields = []
            try
                energy_observer, energy_observer_fields = get_energy_observer_for_backend(config[:backend])
                observers = [energy_observer]
            catch e
                @error "Failed to create energy observer for backend $config[:backend]: $e"
            end

            # Tune the kernel
            results = KernelTuner.tune_kernel(
                "vector_add!",
                kernel_code,
                (size,),
                [c, a, b, size],
                tune_params,
                observers=observers,
                grid_div_x=["block_size_x", "work_per_thread"],
                block_size_names=["block_size_x"],
                lang="Julia",
                answer=[repeat(Float32[11, 22, 33, 44], outer=rsize), nothing, nothing, nothing],
                compiler_options=compiler_options,
                cache=cache_file,
            )

            # Verify the results
            @test results[1] isa Vector
            @test length(results[1]) == length(tune_params[1][2]) * length(tune_params[2][2])  # number of configurations tested
            @test results[1][1] isa Dict
            @test results[2] isa Dict
            @test "best_config" in keys(results[2])

            # Check the contents of the results
            for result in results[1]
                # check that the tunable parameter keys are present in the result
                for tp in tune_params
                    @test tp[1] in keys(result)
                    @test result[tp[1]] isa Int   # integer parameters stay integers on their way back from Python
                end
                @test result["times"] isa Vector{Float64}

                # check that the expected observer fields are present
                if haskey(result, "time") && result["time"] !== nothing
                    @test result["time"] > 0
                    @test result["time"] < Inf
                    for field in energy_observer_fields
                        @test field in keys(result)
                    end
                end
            end

            # check and delete the cachefile
            @test isfile(cache_filepath) == true
            rm(cache_filepath; force=true)
        end
    end

    if !config[:minimal]
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
                @test results[1] isa Vector
                @test length(results[1]) <= max_fevals # number of configurations tested
                @test results[1][1] isa Dict
                @test results[2] isa Dict
                @test "best_config" in keys(results[2])
            end
        end
    end

    # this is an intensive GPU test, don't run on CI or with CPU backend
    if !config[:minimal] && !config[:CI] && config[:backend] != "CPU"
        @testset "Full integration test with the unmqr_tsmqr kernel" begin
            include("integration_test.jl")
            run_integration_test(compiler_options)
            run_integration_test(compiler_options, insert_bad_data=true)
        end
    end
end
