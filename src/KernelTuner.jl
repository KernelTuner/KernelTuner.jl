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
    return pyconvert(Tuple{Tuple,Dict}, res)
end

export run_kernel
function run_kernel(args...; kwargs...)
    result = kt[].run_kernel(args...; kwargs...)
    return pyconvert(Vector{Any}, result)  # convert to a Julia vector of Any
end

export detect_julia_gpu_backends
function detect_julia_gpu_backends()
    return pyconvert(Vector{String}, pyimport("kernel_tuner.backends.julia_helper").detect_julia_gpu_backends())
end

export Tunable
function Tunable(name::String, value)
    return kt[].accuracy.Tunable(name, value)
end

end # module KernelTuner