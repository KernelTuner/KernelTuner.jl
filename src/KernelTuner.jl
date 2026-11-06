module KernelTuner

using PythonCall
using CondaPkg

# Initialize a global reference for the Python module
const kt = Ref{Py}(Py(nothing))

function __init__()
    # This runs when the user types `using KernelTuner`
    try
        kt[] = pyimport("kernel_tuner")
    catch e
        sys = pyimport("sys")
        @info "Python Version" PythonCall.pystr(sys.version)
        @info "Python Executable" PythonCall.pystr(sys.executable)
        @info "Python Path" PythonCall.pystr(sys.path)
        @error "Failed to import the kernel_tuner Python module. Make sure it is installed in your Python environment and available in the Python path." exception = (e, catch_backtrace())
    end
end

# PythonCall converts Julia numbers to plain Python `int`/`float`, which come back into Julia as Int64/Float64
# (e.g. Float32 → float → Float64, which Metal cannot compile). Wrapping them as NumPy scalars of the matching
# width preserves the exact Julia type on its way through Kernel Tuner.
const numpy_scalar_types = Dict{DataType,String}(
    Float16 => "float16", Float32 => "float32", Float64 => "float64",
    Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64",
    UInt8 => "uint8", UInt16 => "uint16", UInt32 => "uint32", UInt64 => "uint64",
)  # Bool is left out on purpose: Python's `bool` already round-trips to Bool, `numpy.bool_` does not

to_numpy_scalar(x) = x
function to_numpy_scalar(x::Number)
    name = get(numpy_scalar_types, typeof(x), nothing)
    name === nothing && return x
    return getproperty(pyimport("numpy"), Symbol(name))(x)
end

# Apply `to_numpy_scalar` to the kernel arguments, passed either as 4th positional argument or as `arguments` keyword.
function convert_kernel_arguments(args, kwargs)
    if length(args) >= 4
        args = (args[1:3]..., map(to_numpy_scalar, args[4]), args[5:end]...)
    end
    if haskey(kwargs, :arguments)
        kwargs = merge(NamedTuple(kwargs), (arguments=map(to_numpy_scalar, kwargs[:arguments]),))
    end
    return args, kwargs
end

# Convert a value coming from Python into its Julia counterpart (numbers, strings, arrays, wrapped Julia objects).
# PythonCall may already have wrapped the value (e.g. as `PyArray` or `PyDict`), so those are handled as well.
from_python(x::Py) = from_python(pyconvert(Any, x))
from_python(x::PythonCall.PyArray) = Array(x)
from_python(x::PythonCall.PyList) = [from_python(v) for v in x]
from_python(x::PythonCall.PyDict) = params_to_julia(x)
from_python(x) = x

# A Python dict with string keys (e.g. the parameters and results of a configuration) as a Julia `Dict{String,Any}`.
# The values are converted one by one, as `pyconvert(Dict, ...)` would turn integer parameters into Float64.
params_to_julia(params) = Dict{String,Any}(pyconvert(String, pystr(k)) => from_python(v) for (k, v) in Py(params).items())

"""
    convert_metrics(metrics)

Converts Julia-style user-defined metrics to the ordered Python `dict` that Kernel Tuner expects.
`metrics` is a vector of `name => metric` pairs (or any other `AbstractDict`), where each metric is either
- a string expression in terms of the tunable parameters and earlier results, e.g. `"1e-6 * n / (time / 1e3)"`, or
- a Julia function that takes the configuration as a `Dict{String,Any}`, e.g. `p -> p["block_size_x"] * p["block_size_y"]`.

Metrics are computed in order, so later metrics can use earlier ones. Use a vector of pairs to keep that order,
as the order of a `Dict` is not defined.
"""
function convert_metrics(metrics::Union{AbstractVector{<:Pair},AbstractDict})
    py_metrics = pydict()
    for (name, metric) in metrics
        if metric isa AbstractString
            py_metrics[string(name)] = metric
        elseif metric isa Base.Callable
            py_metrics[string(name)] = params -> metric(params_to_julia(params))
        else
            error("Metric \"$name\" should be a string expression or a function, got $(typeof(metric))")
        end
    end
    return py_metrics
end
convert_metrics(metrics) = metrics    # already a Python dict (or `nothing`)

# Kernel Tuner stops tuning with a KeyboardInterrupt when it is interrupted (e.g. Ctrl-C or interrupting a notebook cell),
# which is raised in Julia as an InterruptException, like any other interrupt.
is_keyboard_interrupt(e) = e isa PyException && pyissubclass(e.t, pybuiltins.KeyboardInterrupt)

function call_kernel_tuner(f, args...; kwargs...)
    try
        return f(args...; kwargs...)
    catch e
        is_keyboard_interrupt(e) && throw(InterruptException())
        rethrow()
    end
end

# wrap the Python functions
export tune_kernel
function tune_kernel(args...; kwargs...)
    args, kwargs = convert_kernel_arguments(args, kwargs)
    if haskey(kwargs, :metrics)
        kwargs = merge(NamedTuple(kwargs), (metrics=convert_metrics(kwargs[:metrics]),))
    end
    results, env = call_kernel_tuner(kt[].tune_kernel, args...; kwargs...)
    return Dict{String,Any}[params_to_julia(r) for r in results], params_to_julia(env)
end

export run_kernel
function run_kernel(args...; kwargs...)
    py_args, py_kwargs = convert_kernel_arguments(args, kwargs)
    result = call_kernel_tuner(kt[].run_kernel, py_args...; py_kwargs...)
    result_converted = Vector{Any}(undef, length(result))
    # convert the processed resulting arguments back to their Julia types
    for i in 1:length(result)
        try
            result_converted[i] = pyconvert(typeof(args[4][i]), result[i-1])
        catch e
            fallback = pyconvert(Any, result[i-1])
            if fallback isa PythonCall.PyArray
                result_converted[i] = pyconvert(Array, result[i-1])
            else
                result_converted[i] = fallback
            end
        end
    end
    return result_converted
end

export detect_julia_gpu_backends
function detect_julia_gpu_backends()
    return pyconvert(Vector{String}, pyimport("kernel_tuner.backends.julia_helper").detect_julia_gpu_backends())
end

export Tunable, AccuracyObserver
function Tunable(name::String, value)
    return kt[].accuracy.Tunable(name, value)
end

"""
    AccuracyObserver(metric="RMSE"; key="error", atol=1e-8)

Observer that compares the output of each configuration to the `answer` passed to `tune_kernel`, and stores
the largest error over all output arguments under `key` in the results. When this observer is used, the default
output verification is skipped, so configurations with a large error are recorded instead of stopping the tuning.

`metric` is either the name of an error metric known to Kernel Tuner (such as `"RMSE"`, `"MAE"`, `"absolute error"`
or `"relative error"`; `atol` prevents division by zero in relative metrics), or a Julia function
`(answer, output) -> error` that receives both as Julia arrays.
"""
function AccuracyObserver(metric::Union{AbstractString,Base.Callable}="RMSE"; key::AbstractString="error", atol::Real=1e-8)
    py_metric = metric isa AbstractString ? metric : (answer, output) -> metric(from_python(answer), from_python(output))
    return pyimport("kernel_tuner.accuracy").AccuracyObserver(py_metric, key; atol=atol)
end

export get_observer
function get_observer(observer_type::String, args...; kwargs...)
    local obs

    # lazy-load the observer class based on the observer_type
    if observer_type == "JuliaRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.julia").JuliaRuntimeObserver
    elseif observer_type == "MetalObserver"
        obs = pyimport("kernel_tuner.observers.metal").MetalObserver
    elseif observer_type == "MetalContinuousObserver"
        obs = pyimport("kernel_tuner.observers.metal").MetalContinuousObserver
    elseif observer_type == "AMDSMIObserver"
        obs = pyimport("kernel_tuner.observers.amd").AMDSMIObserver
    elseif observer_type == "AMDSMIContinuousObserver"
        obs = pyimport("kernel_tuner.observers.amd").AMDSMIContinuousObserver
    elseif observer_type == "CompilerRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.compiler").CompilerRuntimeObserver
    elseif observer_type == "CupyRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.cupy").CupyRuntimeObserver
    elseif observer_type == "HipRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.hip").HipRuntimeObserver
    elseif observer_type == "NCUObserver"
        obs = pyimport("kernel_tuner.observers.ncu").NCUObserver
    elseif observer_type == "NVMLObserver"
        obs = pyimport("kernel_tuner.observers.nvml").NVMLObserver
    elseif observer_type == "CudaRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.nvcuda").CudaRuntimeObserver
    elseif observer_type == "OpenCLObserver"
        obs = pyimport("kernel_tuner.observers.opencl").OpenCLObserver
    elseif observer_type == "PMTObserver"
        obs = pyimport("kernel_tuner.observers.pmt").PMTObserver
    elseif observer_type == "PowerSensorObserver"
        obs = pyimport("kernel_tuner.observers.powersensor").PowerSensorObserver
    elseif observer_type == "PyCudaRuntimeObserver"
        obs = pyimport("kernel_tuner.observers.pycuda").PyCudaRuntimeObserver
    elseif observer_type == "RegisterObserver"
        obs = pyimport("kernel_tuner.observers.register").RegisterObserver
    elseif observer_type == "TegraObserver"
        obs = pyimport("kernel_tuner.observers.tegra").TegraObserver
    else
        error("Unknown observer type: $observer_type")
    end

    # instantiate the observer with the provided arguments
    return obs(args...; kwargs...)
end

export get_energy_observer_for_backend
function get_energy_observer_for_backend(backend::String, args...; kwargs...)::Tuple{Any,Vector{String}}
    local obs
    backend = uppercase(backend)
    if backend == "DEFAULT" || backend === nothing
        backend = detect_julia_gpu_backends()[1]  # get the first available backend
    end

    # get the correct observer based on the backend
    if backend == "METAL"
        obs = get_observer("MetalObserver", args...; kwargs...)
    elseif backend == "AMDGPU"
        obs = get_observer("AMDSMIObserver", args...; kwargs...)
    elseif backend == "CUDA"
        CondaPkg.add_pip("pynvml"; version=">=11.4.1")   # TODO see if we can move to CUDA.jl.NVML instead
        if length(args) == 0 && !haskey(kwargs, :observables)
            # set default observables if none are provided
            args = [["power_readings", "nvml_power", "nvml_energy", "core_freq", "mem_freq", "temperature"]]
        end
        obs = get_observer("NVMLObserver", args...; kwargs...)
    else
        error("No observer available for backend: $backend")
    end

    # get the selected observables from the observer
    observables = pyconvert(Vector{String}, obs.observables)
    if hasproperty(obs, :prefix) && obs.prefix !== nothing
        prefix = pyconvert(Union{String,Nothing}, obs.prefix)
        if prefix !== nothing && prefix != ""
            observables = [prefix * "_" * ob for ob in observables]
        end
    end

    return obs, observables
end

end # module KernelTuner