# Plotting of tuning results, loaded automatically when Plots.jl is loaded alongside KernelTuner.
# See `src/visualization.jl` for the documentation of these functions.
module KernelTunerPlotsExt

using KernelTuner
using KernelTuner: read_tuning_cache, pareto_front
using Plots
using Base64: base64encode

# Make names like "block_size_x" readable, but keep names like "GB/s" as they are
pretty(name) = name == lowercase(name) ? titlecase(replace(name, "_" => " ")) : name

# The objective as a finite number, or `nothing` for failed configurations (which store an error string instead)
measured(config, key) = (v = get(config, key, nothing); v isa Real && isfinite(v) ? Float64(v) : nothing)

# Green is good, red is bad
colormap(higher_is_better) = higher_is_better ? cgrad(:RdYlGn) : reverse(cgrad(:RdYlGn))

function KernelTuner.plot_tuning_heatmap(results::AbstractVector, x_param::AbstractString, y_param::AbstractString;
    objective::AbstractString="time", higher_is_better::Bool=false, annotate::Bool=true, kwargs...)
    valid = filter(r -> haskey(r, x_param) && haskey(r, y_param) && !isnothing(measured(r, objective)), results)
    isempty(valid) && error("No configurations with a valid \"$objective\" and parameters \"$x_param\" and \"$y_param\" to plot")

    # The parameter values are placed at evenly spaced positions, so that e.g. powers of two are spread out evenly
    xs, ys = sort(unique([r[x_param] for r in valid])), sort(unique([r[y_param] for r in valid]))
    better(a, b) = isnan(b) || (higher_is_better ? a > b : a < b)
    values = fill(NaN, length(ys), length(xs))
    for r in valid
        i, j, v = findfirst(isequal(r[y_param]), ys), findfirst(isequal(r[x_param]), xs), measured(r, objective)
        better(v, values[i, j]) && (values[i, j] = v)    # the best value over the other tunable parameters
    end

    p = heatmap(1:length(xs), 1:length(ys), values;
        xlabel=pretty(x_param), ylabel=pretty(y_param), title="Tuning Heatmap: " * pretty(objective),
        color=colormap(higher_is_better),
        colorbar_title=" \n" * pretty(objective) * (higher_is_better ? " (higher is better)" : " (lower is better)"),
        colorbar=true, right_margin=3Plots.mm, xticks=(1:length(xs), string.(xs)), yticks=(1:length(ys), string.(ys)),
        dpi=300, grid=false, kwargs...)

    if annotate
        # The middle of this color scheme is yellow, so use black text in the middle and white text at the extremes
        lo, hi = extrema(filter(!isnan, values))
        for i in axes(values, 1), j in axes(values, 2)
            v = values[i, j]
            isnan(v) && continue
            color = lo + 0.25 * (hi - lo) < v < lo + 0.75 * (hi - lo) ? :black : :white
            annotate!(p, j, i, text(string(round(v, sigdigits=3)), 10, color, :center))
        end
    end
    return p
end

function KernelTuner.plot_tuning_progress(cachefile::AbstractString; kwargs...)
    param_keys, configs = read_tuning_cache(cachefile)
    return KernelTuner.plot_tuning_progress(configs; param_keys=param_keys, kwargs...)
end

function KernelTuner.plot_tuning_progress(configs::AbstractVector; param_keys=String[],
    objective::AbstractString="time", higher_is_better::Bool=false, kwargs...)
    values = [measured(c, objective) for c in configs]
    valid = findall(!isnothing, values)
    y = Float64[values[i] for i in valid]
    label = objective == "time" ? "Time (ms)" : pretty(objective)
    title = "$(length(configs)) configurations evaluated ($(length(configs) - length(valid)) failed)"
    if !isempty(y)
        i = higher_is_better ? argmax(y) : argmin(y)
        title *= "\nbest: $(round(y[i], sigdigits=4))$(objective == "time" ? " ms" : "")"
        params = filter(k -> haskey(configs[valid[i]], k), param_keys)
        isempty(params) || (title *= " with " * join(["$k=$(configs[valid[i]][k])" for k in params], ", "))
    end
    log_scale = !isempty(y) && minimum(y) > 0 && maximum(y) > 10 * minimum(y)   # log scale for wide ranges
    # color by value, following the y-axis scale so outliers don't wash out the rest
    p = scatter(valid, y; label="configuration", markersize=5, markerstrokewidth=0.3, markerstrokecolor=:gray40, alpha=0.8,
        marker_z=isempty(y) ? nothing : (log_scale ? log10.(y) : y), color=colormap(higher_is_better), colorbar=false,
        xlabel="Configuration (in order of evaluation)", ylabel=label * (higher_is_better ? ", higher is better" : ", lower is better"),
        yscale=log_scale ? :log10 : :identity, xlims=(0, max(length(configs), 1) + 1), title=title, titlefontsize=10,
        size=(800, 600), left_margin=3Plots.mm, bottom_margin=3Plots.mm, legend=:topright, dpi=200, kwargs...)
    if !isempty(y)
        plot!(p, valid, higher_is_better ? accumulate(max, y) : accumulate(min, y);
            seriestype=:steppost, linewidth=2, color=:black, label="best so far")
    end
    return p
end

function KernelTuner.plot_time_energy(cachefile::AbstractString; kwargs...)
    param_keys, configs = read_tuning_cache(cachefile)
    return KernelTuner.plot_time_energy(configs; param_keys=param_keys, kwargs...)
end

function KernelTuner.plot_time_energy(configs::AbstractVector; param_keys=String[],
    energy_key::AbstractString="energy", time_key::AbstractString="time", kwargs...)
    positive(c, k) = (v = measured(c, k); !isnothing(v) && v > 0)
    valid = findall(c -> positive(c, time_key) && positive(c, energy_key), configs)
    t = Float64[configs[i][time_key] for i in valid]
    e = Float64[configs[i][energy_key] for i in valid]
    title = "$(length(configs)) configurations evaluated ($(length(configs) - length(valid)) failed)"
    wide(v) = !isempty(v) && maximum(v) > 10 * minimum(v)   # log scale for wide ranges
    p = scatter(t, e; label="configuration", markersize=5, markerstrokewidth=0.3, markerstrokecolor=:gray40, alpha=0.8,
        marker_z=isempty(e) ? nothing : (wide(e) ? log10.(e) : e), color=colormap(false), colorbar=false,
        xlabel="Time (ms), lower is better", ylabel="Energy (J), lower is better",
        xscale=wide(t) ? :log10 : :identity, yscale=wide(e) ? :log10 : :identity,
        size=(800, 600), left_margin=3Plots.mm, bottom_margin=3Plots.mm, legend=:topleft, dpi=200, titlefontsize=10)
    if !isempty(t)
        function config_string(c)
            params = filter(k -> haskey(c, k), param_keys)
            return isempty(params) ? "" : " with " * join(["$k=$(c[k])" for k in params], ", ")
        end
        fastest, lowest = argmin(t), argmin(e)
        title *= "\nfastest: $(round(t[fastest], sigdigits=3)) ms, $(round(e[fastest], sigdigits=3)) J$(config_string(configs[valid[fastest]]))"
        title *= "\nlowest energy: $(round(t[lowest], sigdigits=3)) ms, $(round(e[lowest], sigdigits=3)) J$(config_string(configs[valid[lowest]]))"
        front = pareto_front(t, e)
        plot!(p, t[front], e[front]; seriestype=:steppost, linewidth=2, color=:black, label="Pareto front")
        scatter!(p, [t[fastest]], [e[fastest]]; marker=:star5, markersize=10, color=:dodgerblue, label="fastest")
        scatter!(p, [t[lowest]], [e[lowest]]; marker=:diamond, markersize=8, color=:magenta, label="lowest energy")
    end
    plot!(p; title=title, kwargs...)
    return p
end

# The running IJulia kernel, if this is a Jupyter notebook with an IJulia version whose internals we know
function ijulia_kernel()
    isdefined(Main, :IJulia) || return nothing
    IJ = Main.IJulia
    all(name -> isdefined(IJ, name), (:_default_kernel, :send_ipython, :msg_pub)) || return nothing
    kernel = IJ._default_kernel
    return isnothing(kernel) || !hasproperty(kernel, :execute_msg) || !hasproperty(kernel, :publish) ? nothing : kernel
end

function KernelTuner.with_live_tuning_plot(f, cachefile::AbstractString; interval::Real=0.5,
    plotter=KernelTuner.plot_tuning_progress, kwargs...)
    render() = plotter(cachefile; kwargs...)
    other_threads = Threads.nthreads(:default) - (Threads.threadpool() == :default ? 1 : 0)
    kernel = ijulia_kernel()
    if other_threads < 1 || isnothing(kernel)
        @warn "Live plotting needs a Jupyter notebook and Julia started with more than one thread (e.g. `--threads=auto`); plotting once tuning is done."
        result = f()
        display(render())
        return result
    end

    # Jupyter replaces an output in place when it receives `update_display_data` with the same `display_id`
    IJ, parent, display_id = Main.IJulia, kernel.execute_msg, string(rand(UInt64), base=16)
    function send(msg_type)
        png = base64encode(io -> show(io, MIME("image/png"), render()))
        content = Dict("data" => Dict("image/png" => png, "text/plain" => "Live tuning plot"),
            "metadata" => Dict(), "transient" => Dict("display_id" => display_id))
        IJ.send_ipython(kernel.publish[], kernel, IJ.msg_pub(parent, msg_type, content))
    end
    send("display_data")    # on the main thread, so the plotting code is compiled before the poller starts

    done = Threads.Atomic{Bool}(false)
    # Runs on another thread while the main thread is busy inside Kernel Tuner. It must not call into Python.
    poller = Threads.@spawn begin
        last_size = -1
        while !done[]
            size = isfile(cachefile) ? filesize(cachefile) : filesize(cachefile * ".json")
            if size != last_size
                last_size = size
                try
                    send("update_display_data")
                catch e
                    @error "Live plot update failed" exception = e
                end
            end
            sleep(interval)
        end
    end
    try
        return f()
    finally
        done[] = true
        wait(poller)
        send("update_display_data")   # final state, including the last configurations
    end
end

end # module KernelTunerPlotsExt
