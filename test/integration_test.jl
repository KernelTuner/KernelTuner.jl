"""Integration test for tuning a Julia kernel from Julia using Kernel Tuner. Based on test3.jl in NextLATuner."""

import KernelTuner as kt
using KernelTuner: Tunable
using KernelTuner: detect_julia_gpu_backends

# === Kernel script starts below ===

using KernelAbstractions
using Random

API = detect_julia_gpu_backends()[1]
backend = CPU()  # set up argument arrays on host to avoid GPU out-of-memory errors

# settings for this test
GlobalProgramRNG = MersenneTwister(1234)
N = 8192 * 2

function run_integration_test(compiler_options; insert_bad_data=false)
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

    # prepare data arrays
    eltype = Float32
    size_i = N
    A = randn!(GlobalProgramRNG, KernelAbstractions.zeros(backend, eltype, size_i, size_i))
    Tau_full = randn!(GlobalProgramRNG, KernelAbstractions.zeros(backend, eltype, maximum(TILESIZE), Int(size_i / minimum(TILESIZE))))  # full Tau array from which tunable views will be created, ensures that the same randomly generated data is used for all tunable configurations
    Tau = TS -> view(Tau_full, 1:TS, 1:Int(size_i/TS))
    args_list = [
        Dict(TS => view(deepcopy(A), 1:512, (TS+1):size(A, 2)) for TS in TILESIZE),
        Dict(TS => view(deepcopy(A), (TS+1):size(A, 1), (TS+1):size(A, 2)) for TS in TILESIZE),
        Dict(TS => view(deepcopy(A), (TS+1):size(A, 1), 1:TS) for TS in TILESIZE),
        Dict(TS => view(deepcopy(Tau(TS)), 1:TS, 2:size(Tau(TS), 2)) for TS in TILESIZE),
    ]   # TODO check if we can do without the copies
    args = [
        Tunable("TILESIZE", deepcopy(args_list[1])),
        Tunable("TILESIZE", deepcopy(args_list[2])),
        Tunable("TILESIZE", deepcopy(args_list[3])),
        Tunable("TILESIZE", deepcopy(args_list[4])),
    ]
    problem_size = ("int(ceil(($N - TILESIZE)/ TILESIZEMUL))*TILESIZEMUL", "QRSPLIT")  # problem size (ndrange)

    # run the kernel once for each tilesize to get the answer for tuning, and to check that it runs without errors before tuning
    answers = Dict{Int,Any}()
    for qs in tune_params[4][2]  # iterate over QRSPLIT values
        answers_th = Dict{Int,Any}()
        for th in tune_params[2][2]  # iterate over TILEHEIGHT values
            answers_ts = Dict{Int,Any}()
            for ts in TILESIZE
                params = [(tp[1], tp[2][1]) for tp in tune_params]  # select parameter values that don't affect the outcome
                params[1] = ("TILESIZE", ts)  # override TILESIZE parameter as it does affect the outcome
                params[2] = ("TILEHEIGHT", th)  # override TILEHEIGHT parameter as it does affect the outcome
                params[4] = ("QRSPLIT", qs)  # override QRSPLIT parameter as it does affect the outcome
                params[end] = ("TILEBLOCK", ts ÷ qs)  # override TILEBLOCK parameter as it does affect the outcome and is dependent on TILESIZE and QRSPLIT
                result = kt.run_kernel(
                    "applyQorQt_unsafe_kernel2_2d_fused_A!",
                    "test/unmqr_tsmqr.jl",
                    problem_size,
                    [deepcopy(arg[ts]) for arg in args_list],
                    params,
                    block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
                    grid_div_x=1, grid_div_y=1,                       # prevents division of the grid
                    lang="Julia",
                    compiler_options=compiler_options,
                )
                Ar = result[1]
                @test !all(isnan.(Ar))    # check if the answer is not only NaNs
                @test Ar != A  # sanity check to make sure the kernel did something useful
                if insert_bad_data
                    Ar[1, 1] = 1234.0 # check if this indeed gives a verification failure
                end
                answers_ts[ts] = [Ar, nothing, nothing, nothing]
            end
            answers_th[th] = Tunable("TILESIZE", answers_ts)
        end
        answers[qs] = Tunable("TILEHEIGHT", answers_th)
    end
    answer = Tunable("QRSPLIT", answers)

    # tune the kernel
    if insert_bad_data
        try
            kt.tune_kernel(
                "applyQorQt_unsafe_kernel2_2d_fused_A!",
                "test/unmqr_tsmqr.jl",
                problem_size,
                args,
                tune_params,
                block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
                grid_div_x=1, grid_div_y=1,                       # prevents division of the grid
                lang="Julia",
                compiler_options=compiler_options,
                restrictions=restrictions,
                iterations=7,
                verbose=true,
                answer=answer,
            )
            @test false # fail if no error is thrown
        catch e
            @test nameof(typeof(e)) == :PyException
            # @test occursin("Kernel result verification failed for", string(e))    # doesn't work for large arrays because string write buffer crashes
        end
    else
        results = kt.tune_kernel(
            "applyQorQt_unsafe_kernel2_2d_fused_A!",
            "test/unmqr_tsmqr.jl",
            problem_size,
            args,
            tune_params,
            block_size_names=["TILESIZEMUL", "QRSPLIT"],      # block size (workgroup) dimensions
            grid_div_x=1, grid_div_y=1,                       # prevents division of the grid
            lang="Julia",
            compiler_options=compiler_options,
            restrictions=restrictions,
            iterations=7,
            verbose=true,
            answer=answer,
        )
        @test results[1] isa Tuple
        @test results[2] isa Dict
        @test "best_config" in keys(results[2])
    end
end
