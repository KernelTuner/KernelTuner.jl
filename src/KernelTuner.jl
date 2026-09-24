module KernelTuner

using PythonCall
using CondaPkg

# Initialize a global reference for the Python module
const kt = Ref{Py}(Py(nothing))

function __init__()
    # OS check
    if Sys.iswindows()
        error("KernelTuner.jl only supports macOS and Linux. Windows is not supported.")
    end

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

# wrap the Python functions
export tune_kernel
function tune_kernel(args...; kwargs...)
    res = kt[].tune_kernel(args...; kwargs...)
    c_res = pyconvert(Tuple{Vector{Dict},Dict}, res)
    if length(res[1]) > 0
        c_res[2]["best_config"] = pyconvert(Dict, c_res[2]["best_config"])
    end
    return c_res
end

export run_kernel
function run_kernel(args...; kwargs...)
    result = kt[].run_kernel(args...; kwargs...)
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

export Tunable
function Tunable(name::String, value)
    return kt[].accuracy.Tunable(name, value)
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
        obs = get_observer("NVMLObserver", args...; kwargs...)
    else
        error("No observer available for backend: $backend")
    end

    # get the selected observables from the observer
    observables = pyconvert(Vector{String}, obs.observables)
    if hasproperty(obs, :prefix)
        prefix = pyconvert(String, obs.prefix)
        if prefix != ""
            observables = [prefix * "_" * ob for ob in observables]
        end
    end

    return obs, observables
end

end # module KernelTuner