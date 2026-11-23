# Visualization of tuning results.
# Reading cache files and computing Pareto fronts works without extra dependencies; the plotting functions are
# declared here and implemented in the `KernelTunerPlotsExt` package extension, which loads with `using Plots`.

export read_tuning_cache, pareto_front
export plot_tuning_heatmap, plot_tuning_progress, plot_time_energy, with_live_tuning_plot

const plotting_functions = Symbol[:plot_tuning_heatmap, :plot_tuning_progress, :plot_time_energy, :with_live_tuning_plot]

"""
    plot_tuning_heatmap(results, x_param, y_param; objective="time", higher_is_better=false, annotate=true)

Heatmap of `objective` over two tunable parameters, with `results` as returned by [`tune_kernel`](@ref) (or the
configurations returned by [`read_tuning_cache`](@ref)). When other tunable parameters vary as well, each cell
shows the best value over them. Failed configurations are left out. Returns the plot.

Requires Plots.jl to be loaded (`using Plots`).
"""
function plot_tuning_heatmap end

"""
    plot_tuning_progress(cachefile::AbstractString; objective="time", higher_is_better=false)
    plot_tuning_progress(configs::AbstractVector; param_keys=String[], objective="time", higher_is_better=false)

Scatterplot of `objective` for every configuration in order of evaluation, with the best value found so far,
from a Kernel Tuner cache file or from the `results` returned by [`tune_kernel`](@ref). Pass the names of the
tunable parameters as `param_keys` to show the best configuration in the title (read from the cache file otherwise).
Returns the plot.

Requires Plots.jl to be loaded (`using Plots`).
"""
function plot_tuning_progress end

"""
    plot_time_energy(cachefile::AbstractString; energy_key="energy")
    plot_time_energy(configs::AbstractVector; param_keys=String[], energy_key="energy", time_key="time")

Scatterplot of time versus energy of every configuration, highlighting the fastest configuration, the configuration
with the lowest energy, and the Pareto front between them. Use `energy_key` to select the energy observable
(e.g. `"nvml_energy"`, see [`get_energy_observer_for_backend`](@ref)). Returns the plot.

Requires Plots.jl to be loaded (`using Plots`).
"""
function plot_time_energy end

"""
    with_live_tuning_plot(f, cachefile; interval=0.5, plotter=plot_tuning_progress, kwargs...)

Runs `f` (typically a call to [`tune_kernel`](@ref) with `cache=cachefile`) while a plot of the configurations
evaluated so far is redrawn in place every `interval` seconds, and returns the result of `f`:

```julia
results, env = with_live_tuning_plot("vector_add_cache.json") do
    tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params; cache="vector_add_cache.json")
end
```

`plotter` draws the plot from the cache file, e.g. [`plot_tuning_progress`](@ref) or [`plot_time_energy`](@ref);
the other keyword arguments are passed on to it. Live updates need a Jupyter notebook (IJulia) and Julia started with
more than one thread (e.g. `julia --threads=auto`); otherwise the plot is shown once tuning is done.

Requires Plots.jl to be loaded (`using Plots`).
"""
function with_live_tuning_plot end

# Point users to Plots.jl when they call a plotting function without having loaded it
function plotting_error_hint(io, exc, argtypes, kwargs)
    if exc.f isa Function && nameof(exc.f) in plotting_functions && parentmodule(exc.f) === (@__MODULE__) &&
       isnothing(Base.get_extension((@__MODULE__), :KernelTunerPlotsExt))
        print(io, "\n$(nameof(exc.f)) requires Plots.jl: install it and load it with `using Plots`.")
    end
end

"""
    pareto_front(a, b)

Indices of the Pareto-optimal points when minimizing both `a` and `b`: the points for which no other point has both a
lower `a` and a lower `b`, ordered by `a`.
"""
function pareto_front(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || throw(DimensionMismatch("a and b should have the same length"))
    front, lowest_b = Int[], Inf
    for i in sortperm(a)
        if b[i] < lowest_b
            push!(front, i)
            lowest_b = b[i]
        end
    end
    return front
end

"""
    read_tuning_cache(cachefile) -> (param_keys, configs)

Reads a Kernel Tuner cache file, also while tuning is still writing to it, and returns the names of the tunable
parameters and the evaluated configurations (each a `Dict{String,Any}`, in order of evaluation). Failed
configurations store an error string (such as `"InvalidConfig"`) instead of a number in their results.
Returns empty vectors if the file does not exist (yet).
"""
function read_tuning_cache(cachefile::AbstractString)
    # Kernel Tuner appends ".json" to the cache file name if it is missing
    path = !isfile(cachefile) && isfile(cachefile * ".json") ? cachefile * ".json" : cachefile
    text = isfile(path) ? read(path, String) : ""
    # While tuning, the cache file is not valid JSON yet (it is only closed at the end), but Kernel Tuner writes a header
    # followed by one line per evaluated configuration, so we parse those parts separately.
    r = findfirst("\"cache\": {", text)
    isnothing(r) && return String[], Dict{String,Any}[]
    param_keys = try
        String.(parse_json(text[1:last(r)] * "}}")["tune_params_keys"])
    catch
        String[]
    end
    configs = Dict{String,Any}[]
    for line in eachsplit(text[(last(r)+1):end], '\n')
        m = match(r"^\s*\"[^\"]*\":\s*(\{.*\}),?\s*$", line)
        isnothing(m) && continue
        entry = m.captures[1]
        config = try
            parse_json(entry)
        catch
            try   # once tuning is done, the last line also closes the cache: `{...}}`
                endswith(entry, "}}") ? parse_json(chop(entry)) : nothing
            catch   # a half-written line, it will be complete on the next read
                nothing
            end
        end
        config isa Dict && push!(configs, config)
    end
    return param_keys, configs
end

# A minimal JSON parser for cache files. It is used instead of a JSON package to avoid the dependency, because it must
# not call into Python (it runs on another thread while Python is tuning), and because Python writes non-standard
# `NaN` and `Infinity` values. Objects become `Dict{String,Any}`, arrays `Vector{Any}`, and `null` becomes `nothing`.
function parse_json(text::AbstractString)
    s = String(text)
    value, i = parse_json_value(s, skip_whitespace(s, firstindex(s)))
    skip_whitespace(s, i) > ncodeunits(s) || error("Unexpected trailing characters in JSON at position $i")
    return value
end

function skip_whitespace(s::String, i::Int)
    while i <= ncodeunits(s) && codeunit(s, i) in (UInt8(' '), UInt8('\n'), UInt8('\r'), UInt8('\t'))
        i += 1
    end
    return i
end

function parse_json_value(s::String, i::Int)
    i <= ncodeunits(s) || error("Unexpected end of JSON")
    c = s[i]
    if c == '{'
        obj = Dict{String,Any}()
        i = skip_whitespace(s, i + 1)
        i <= ncodeunits(s) && s[i] == '}' && return obj, i + 1
        while true
            i <= ncodeunits(s) && s[i] == '"' || error("Expected a string key in JSON at position $i")
            key, i = parse_json_string(s, i)
            i = skip_whitespace(s, i)
            i <= ncodeunits(s) && s[i] == ':' || error("Expected ':' in JSON at position $i")
            obj[key], i = parse_json_value(s, skip_whitespace(s, i + 1))
            i = skip_whitespace(s, i)
            i <= ncodeunits(s) || error("Unexpected end of JSON")
            s[i] == '}' && return obj, i + 1
            s[i] == ',' || error("Expected ',' or '}' in JSON at position $i")
            i = skip_whitespace(s, i + 1)
        end
    elseif c == '['
        arr = Any[]
        i = skip_whitespace(s, i + 1)
        i <= ncodeunits(s) && s[i] == ']' && return arr, i + 1
        while true
            value, i = parse_json_value(s, i)
            push!(arr, value)
            i = skip_whitespace(s, i)
            i <= ncodeunits(s) || error("Unexpected end of JSON")
            s[i] == ']' && return arr, i + 1
            s[i] == ',' || error("Expected ',' or ']' in JSON at position $i")
            i = skip_whitespace(s, i + 1)
        end
    elseif c == '"'
        return parse_json_string(s, i)
    end
    for (literal, value) in (("true", true), ("false", false), ("null", nothing),
        ("NaN", NaN), ("Infinity", Inf), ("-Infinity", -Inf))
        startswith(SubString(s, i), literal) && return value, i + ncodeunits(literal)
    end
    m = match(r"\G-?\d+(\.\d+)?([eE][+-]?\d+)?", s, i)
    (isnothing(m) || m.offset != i) && error("Invalid JSON value at position $i")
    number = isnothing(m.captures[1]) && isnothing(m.captures[2]) ? parse(Int, m.match) : parse(Float64, m.match)
    return number, i + ncodeunits(m.match)
end

function parse_json_string(s::String, i::Int)
    io = IOBuffer()
    i += 1  # opening quote
    while i <= ncodeunits(s)
        c = s[i]
        if c == '"'
            return String(take!(io)), i + 1
        elseif c == '\\'
            e = s[i+1]
            if e == 'u'
                code = parse(UInt16, s[i+2:i+5]; base=16)
                i += 6
                # characters outside the Basic Multilingual Plane are written as a UTF-16 surrogate pair
                if 0xd800 <= code < 0xdc00 && startswith(SubString(s, i), "\\u")
                    low = parse(UInt16, s[i+2:i+5]; base=16)
                    print(io, Char(0x10000 + ((UInt32(code) - 0xd800) << 10) + (low - 0xdc00)))
                    i += 6
                else
                    print(io, Char(code))
                end
                continue
            end
            print(io, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), e, e))
            i += 2
        else
            print(io, c)
            i = nextind(s, i)
        end
    end
    error("Unterminated string in JSON")
end
