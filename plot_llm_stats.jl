using ArgParse
using CairoMakie
using CSV
using DataFrames
using JSON
using Statistics

const ROOT = dirname(@__FILE__)

const AUDIENCE_COLORS = Dict(
    "cs" => "#0072B2",
    "doctor" => "#D55E00",
    "patient" => "#009E73",
)

const LLM_COLORS = Dict(
    "gemini" => "#CC79A7",
    "openai" => "#0072B2",
    "anthropic" => "#E69F00",
)

const LLM_LABELS = Dict(
    "gemini" => "Gemini 3.1 Pro preview",
    "openai" => "GPT 5",
    "anthropic" => "Sonnet 4.6",
)

const ACTIVE_LLM_DIRS = ("gemini", "openai", "anthropic")

function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--aggregated_root"
        arg_type = String
        default = joinpath(ROOT, "aggregated")
        "--stats_file"
        arg_type = String
        default = ""
        "--output_dir"
        arg_type = String
        default = joinpath(ROOT, "plots")
        "--score_plot"
        arg_type = String
        default = "density"
        "--program_length_plot"
        arg_type = String
        default = "hist"
    end
    return s
end

function make_theme()
    return Theme(
        fonts = (;
            regular = "CMU Serif",
            bold = "CMU Serif Bold",
            italic = "CMU Serif Italic",
            bold_italic = "CMU Serif Bold Italic",
        ),
        fontsize = 18,
        Axis = (
            xlabelsize = 24,
            ylabelsize = 24,
            xticklabelsize = 18,
            yticklabelsize = 18,
            titlealign = :left,
            titlesize = 22,
        ),
        Legend = (
            labelsize = 18,
            titlesize = 20,
        ),
    )
end

function load_score_values(path::String)
    parsed = JSON.parsefile(path)
    payloads = parsed isa AbstractVector ? parsed : Any[parsed]
    vals = Float64[]
    for payload in payloads
        for entry in payload["scores"]
            push!(vals, Float64(entry["score"]))
        end
    end
    return vals
end

function ensure_color(dict::Dict{String, <:Any}, key::String)
    return get(dict, key, Makie.wong_colors()[mod1(length(dict) + hash(key), length(Makie.wong_colors()))])
end

function llm_label(key::String)
    return get(LLM_LABELS, key, key)
end

function draw_distribution!(ax, vals::Vector{Float64}, label::String, color; mode::String)
    if mode == "density"
        density!(ax, vals; color = (color, 0.25), strokecolor = color, strokewidth = 3, label = label)
    elseif mode == "hist"
        hist!(ax, vals; bins = 1:11, color = (color, 0.35), strokecolor = color, strokewidth = 1.5, normalization = :pdf, label = label)
    else
        error("Unknown --score_plot=$mode. Use density or hist.")
    end
end

function save_global_audience_plot(aggregated_root::String, output_dir::String, gram::String; mode::String)
    audience_files = Dict(
        "cs" => joinpath(aggregated_root, "global", "cs_$(gram)_avg.json"),
        "doctor" => joinpath(aggregated_root, "global", "doctor_$(gram)_avg.json"),
        "patient" => joinpath(aggregated_root, "global", "patient_$(gram)_avg.json"),
    )

    fig = Figure(size = (1200, 700))
    ax = Axis(fig[1, 1], title = "Global $(uppercasefirst(gram)) score distribution by audience", xlabel = "Score", ylabel = mode == "density" ? "Density" : "Probability density")
    for audience in ("cs", "doctor", "patient")
        path = audience_files[audience]
        @assert isfile(path) "Missing aggregated file: $path"
        vals = load_score_values(path)
        draw_distribution!(ax, vals, audience, AUDIENCE_COLORS[audience]; mode = mode)
    end
    Legend(fig[2, 1], ax, orientation = :horizontal, framevisible = false)
    xlims!(ax, 1, 10)
    save(joinpath(output_dir, "global_$(gram)_audiences_$(mode).pdf"), fig)
end

function discover_llm_dirs(aggregated_root::String)
    dirs = filter(name -> isdir(joinpath(aggregated_root, name)) && (name in ACTIVE_LLM_DIRS), readdir(aggregated_root))
    sort!(dirs)
    return dirs
end

function save_per_audience_llm_plots(aggregated_root::String, output_dir::String, llms::Vector{String}; mode::String)
    for audience in ("cs", "doctor", "patient")
        for gram in ("unigram", "bigram")
            fig = Figure(size = (1200, 700))
            ax = Axis(fig[1, 1], title = "$(uppercasefirst(audience)) $(uppercasefirst(gram)) score distribution by LLM", xlabel = "Score", ylabel = mode == "density" ? "Density" : "Probability density")
            plotted = 0
            for llm in llms
                path = joinpath(aggregated_root, llm, "$(audience)_$(gram)_avg.json")
                isfile(path) || continue
                vals = load_score_values(path)
                draw_distribution!(ax, vals, llm_label(llm), ensure_color(LLM_COLORS, llm); mode = mode)
                plotted += 1
            end
            plotted == 0 && error("No LLM aggregated files found for audience=$audience gram=$gram")
            Legend(fig[2, 1], ax, orientation = :horizontal, framevisible = false)
            xlims!(ax, 1, 10)
            save(joinpath(output_dir, "$(audience)_$(gram)_llms_$(mode).pdf"), fig)
        end
    end
end

function save_program_length_plot(stats_file::String, output_dir::String; mode::String)
    isempty(stats_file) && return nothing
    @assert isfile(stats_file) "Missing stats file: $stats_file"
    df = CSV.read(stats_file, DataFrame)
    @assert "n_steps" in names(df) "Expected n_steps column in stats file: $stats_file"
    vals = Float64.(df.n_steps)

    fig = Figure(size = (1200, 700))
    ax = Axis(fig[1, 1], title = "Program length distribution", xlabel = "Unique active nodes (n_steps)", ylabel = mode == "hist" ? "Count" : "Density")
    if mode == "hist"
        bins = minimum(vals):maximum(vals)
        hist!(ax, vals; bins = bins, color = ("#4C72B0", 0.45), strokecolor = "#2F3E75", strokewidth = 1.5)
    elseif mode == "density"
        density!(ax, vals; color = ("#4C72B0", 0.3), strokecolor = "#2F3E75", strokewidth = 3)
    else
        error("Unknown --program_length_plot=$mode. Use hist or density.")
    end
    save(joinpath(output_dir, "program_length_$(mode).pdf"), fig)
end

function main()
    args = parse_args(build_parser())
    aggregated_root = abspath(args["aggregated_root"])
    output_dir = abspath(args["output_dir"])
    mkpath(output_dir)

    CairoMakie.activate!(type = "pdf")
    with_theme(make_theme()) do
        llms = discover_llm_dirs(aggregated_root)
        save_global_audience_plot(aggregated_root, output_dir, "unigram"; mode = args["score_plot"])
        save_global_audience_plot(aggregated_root, output_dir, "bigram"; mode = args["score_plot"])
        save_per_audience_llm_plots(aggregated_root, output_dir, llms; mode = args["score_plot"])
        save_program_length_plot(args["stats_file"], output_dir; mode = args["program_length_plot"])
    end
end

main()
