"""Integration test for tuning a Julia kernel from Julia using Kernel Tuner. Based on test3.jl in NextLATuner."""

import KernelTuner as kt
using KernelTuner: Tunable
using KernelTuner: detect_julia_gpu_backends
using KernelTuner: get_energy_observer_for_backend

# === Kernel script starts below ===

using KernelAbstractions
using Random

API = detect_julia_gpu_backends()[1]
backend = CPU()  # set up argument arrays on host, Kernel Tuner copies them to the GPU

# settings for this test
GlobalProgramRNG = MersenneTwister(1234)
N = 4096    # large enough for many tiles per column, small enough to keep host-side data handling cheap

count_temp_files() = count(f -> startswith(f, "temp_") && endswith(f, ".jl") && isfile(f), readdir())

function run_integration_test(compiler_options)
    # set the tunable parameters and restrictions
    TILESIZE = [32, 64]
    tune_params = [
        ("TILESIZE", TILESIZE),
        ("TILEHEIGHT", [64, 128]),
        ("TILESIZEMUL", [8, 16]),
        ("QRSPLIT", [2, 4]),
        ("FACTORQR", [32]),
        ("THREADFACTOR", [4, 8]),
        ("THREADFACTOR2", [2, 4, 8]),
        # ("THREADFACTOR3", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
    ]
    push!(tune_params, ("TILEBLOCK", unique([ts ÷ qs for ts in TILESIZE for qs in tune_params[4][2] if ts % qs == 0])))   # This is effectively a single-value tunable parameter, necessary to avoid calculating TILEBLOCK in the kernel
    restrictions = [
        "QRSPLIT<=TILEHEIGHT",
        "TILEHEIGHT>=TILESIZE",
        "FACTORQR==TILEHEIGHT/QRSPLIT",
        "THREADFACTOR==(1 if (TILESIZEMUL*QRSPLIT>TILEHEIGHT) else (TILEHEIGHT//(TILESIZEMUL*QRSPLIT)))",
        "THREADFACTOR2==(1 if (TILESIZEMUL*QRSPLIT>TILESIZE) else (TILESIZE//(TILESIZEMUL*QRSPLIT)))",
        # "THREADFACTOR3==(1 if (QRSPLIT>=TILESIZE) else (TILESIZE//QRSPLIT))",
        "TILESIZEMUL*QRSPLIT<=832", # Metal GPU limit of 832 threads per block dimension
        "TILESIZE/QRSPLIT==TILEBLOCK",
    ]
    # the same restrictions in Julia, to determine the valid configurations independently of Kernel Tuner
    function is_valid(p)
        TS, TH, TSM, QR = p["TILESIZE"], p["TILEHEIGHT"], p["TILESIZEMUL"], p["QRSPLIT"]
        return QR <= TH && TH >= TS && p["FACTORQR"] == TH / QR &&
               p["THREADFACTOR"] == (TSM * QR > TH ? 1 : TH ÷ (TSM * QR)) &&
               p["THREADFACTOR2"] == (TSM * QR > TS ? 1 : TS ÷ (TSM * QR)) &&
               TSM * QR <= 832 && TS / QR == p["TILEBLOCK"]
    end
    names = first.(tune_params)
    valid_configs = [Dict(zip(names, values)) for values in Iterators.product(last.(tune_params)...) if is_valid(Dict(zip(names, values)))]
    @test !isempty(valid_configs)

    # prepare data arrays (Kernel Tuner copies the arguments to the GPU, so the views can share the same parent arrays)
    eltype = Float32
    size_i = N
    A = randn!(GlobalProgramRNG, KernelAbstractions.zeros(backend, eltype, size_i, size_i))
    # full Tau array from which tunable views will be created, ensures that the same randomly generated data is used for all tunable configurations
    # scaled to the magnitude of real reflector coefficients (tau = 2 / (1 + norm(v)^2)), as the repeated updates overflow to NaN with unscaled values
    Tau_full = eltype(0.01) .* randn!(GlobalProgramRNG, KernelAbstractions.zeros(backend, eltype, maximum(TILESIZE), Int(size_i / minimum(TILESIZE))))
    Tau = TS -> view(Tau_full, 1:TS, 1:Int(size_i/TS))
    args_list = [
        Dict(TS => view(A, 1:512, (TS+1):size(A, 2)) for TS in TILESIZE),
        Dict(TS => view(A, (TS+1):size(A, 1), (TS+1):size(A, 2)) for TS in TILESIZE),
        Dict(TS => view(A, (TS+1):size(A, 1), 1:TS) for TS in TILESIZE),
        Dict(TS => view(Tau(TS), 1:TS, 2:size(Tau(TS), 2)) for TS in TILESIZE),
    ]
    args = [Tunable("TILESIZE", arg) for arg in args_list]
    problem_size = ("int(ceil(($N - TILESIZE)/ TILESIZEMUL))*TILESIZEMUL", "QRSPLIT")  # problem size (ndrange)

    # run the kernel once for each (QRSPLIT, TILEHEIGHT, TILESIZE) combination that occurs in the search space,
    # as these parameters affect the outcome, to get the answers for tuning and to check that it runs without errors
    answers = Dict{Int,Dict{Int,Dict{Int,Any}}}()  # QRSPLIT => TILEHEIGHT => TILESIZE => output of the first argument
    for config in valid_configs
        qs, th, ts = config["QRSPLIT"], config["TILEHEIGHT"], config["TILESIZE"]
        haskey(get(get(answers, qs, Dict()), th, Dict()), ts) && continue  # the other parameters don't affect the outcome
        result = kt.run_kernel(
            "applyQorQt_unsafe_kernel2_2d_fused_A!",
            "test/unmqr_tsmqr.jl",
            problem_size,
            [arg[ts] for arg in args_list],
            [(name, config[name]) for name in names],
            block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
            lang="Julia",
            compiler_options=compiler_options,
        )
        Ar = result[1]
        @test size(Ar) == size(args_list[1][ts])
        @test !any(isnan, Ar)     # NaNs would be considered equal during verification, hiding differences
        @test Ar != args_list[1][ts]  # sanity check to make sure the kernel did something useful
        get!(get!(answers, qs, Dict()), th, Dict())[ts] = Ar
    end
    # nested Tunable answers, optionally with deliberately invalid data to check that verification fails
    make_answer(f=identity) = Tunable("QRSPLIT", Dict(qs => Tunable("TILEHEIGHT", Dict(th => Tunable("TILESIZE", Dict(
        ts => [f(Ar), nothing, nothing, nothing] for (ts, Ar) in a_th)) for (th, a_th) in a_qs)) for (qs, a_qs) in answers))

    @testset "Tune with valid answers" begin
        observers = []
        energy_observer_fields = []
        try
            energy_observer, energy_observer_fields = get_energy_observer_for_backend(API)
            observers = [energy_observer]
        catch e
            @error "Failed to create energy observer for backend $API: $e"
        end

        # set up cache
        cache_file = "julia_integration_test_cache.json"
        cache_filepath = joinpath(pwd(), cache_file)
        rm(cache_filepath; force=true)  # clear cache to avoid using old results
        @test isfile(cache_filepath) == false

        results = kt.tune_kernel(
            "applyQorQt_unsafe_kernel2_2d_fused_A!",
            "test/unmqr_tsmqr.jl",
            problem_size,
            args,
            tune_params,
            block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
            lang="Julia",
            observers=observers,
            compiler_options=compiler_options,
            restrictions=restrictions,
            iterations=7,
            verbose=true,
            answer=make_answer(),
            cache=cache_file,
        )
        @test results[1] isa Vector
        @test results[1][1] isa Dict
        @test results[2] isa Dict
        @test "best_config" in keys(results[2])

        # check for expected results: all valid configurations are tested and pass verification
        @test length(results[1]) == length(valid_configs)
        for result in results[1]
            # check that the tunable parameter keys are present in the result
            for tp in tune_params
                @test tp[1] in keys(result)
            end
            @test is_valid(result)

            # check that the expected observer fields are present
            @test result["time"] isa Real
            @test 0 < result["time"] < Inf
            for field in energy_observer_fields
                @test field in keys(result)
            end
        end

        # check the cachefile
        @test isfile(cache_filepath) == true
        # rm(cache_filepath; force=true)
    end

    @testset "Tune with invalid answers" begin
        # check if there are no temporary files left from other runs
        num_temp_files_start = count_temp_files()
        bad_answer = make_answer(Ar -> (Ar = copy(Ar); Ar[1, 1] = 1234.0; Ar))  # check if this indeed gives a verification failure
        try
            kt.tune_kernel(
                "applyQorQt_unsafe_kernel2_2d_fused_A!",
                "test/unmqr_tsmqr.jl",
                problem_size,
                args,
                tune_params,
                block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
                lang="Julia",
                compiler_options=compiler_options,
                restrictions=restrictions,
                iterations=7,
                verbose=false,
                answer=bad_answer,
            )
            @test false # fail if no error is thrown
        catch e
            @test nameof(typeof(e)) == :PyException
            # @test occursin("Kernel result verification failed for", string(e))    # doesn't work for large arrays because string write buffer crashes
            println("Test of invalid data passed with expected exception, automatically deleting temporary file. Caught expected exception: ", e)
            if num_temp_files_start == 0
                if count_temp_files() == 1
                    rm.(filter(f -> startswith(f, "temp_") && endswith(f, ".jl") && isfile(f), readdir()); force=true)
                else
                    @warn "Temporary file was not deleted after the test, and there were no temporary files before the test. Please check if the temporary file is still present and delete it manually."
                end
            end
        end
    end
end
