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

# Example: Wrap a Python function to make it feel like Julia
export tune_kernel
function tune_kernel(args...; kwargs...)
    return kt[].tune_kernel(args...; kwargs...)
end

end # module KernelTuner