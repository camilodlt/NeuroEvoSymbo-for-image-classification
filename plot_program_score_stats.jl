using ArgParse
using CairoMakie
using CSV
using DataFrames

const ROOT = dirname(@__FILE__)

const AUDIENCE_COLORS = Dict(
    "cs" => "#0072B2",
    "doctor" => "#D55E00",
    "patient" => "#009E73",
)

function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--stats_file"
        arg_type = String
        required = true
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

function draw_distribution!(ax, vals::Vector{Float64}, label::String, color; mode::String)
    if mode == "density"
        density!(ax, vals; color = (color, 0.25), strokecolor = color, strokewidth = 3, label = label)
    elseif mode == "hist"
        hist!(ax, vals; bins = 1:11, color = (color, 0.35), strokecolor = color, strokewidth = 1.5, normalization = :pdf, label = label)
    else
        error("Unknown --score_plot=$mode. Use density or hist.")
    end
end

function program_score_column(reducer::String, audience::String, gram::String)
    @assert reducer in ("mean", "min")
    @assert audience in ("cs", "doctor", "patient")
    @assert gram in ("unigram", "bigram")
    suffix = gram == "unigram" ? "_unigram_avg_single" : "_bigram_avg_composition1"
    return Symbol("$(reducer)_$(audience)$(suffix)")
end

function save_program_score_plot(df::DataFrame, output_dir::String, gram::String, reducer::String; mode::String)
    fig = Figure(size = (1200, 700))
    title = "Program $(reducer) $(gram) score distribution by audience"
    ylabel = mode == "density" ? "Density" : "Probability density"
    ax = Axis(fig[1, 1], title = title, xlabel = "Program score", ylabel = ylabel)

    for audience in ("cs", "doctor", "patient")
        col = program_score_column(reducer, audience, gram)
        @assert String(col) in names(df) "Expected $(col) in stats file"
        vals = Float64.(df[!, col])
        draw_distribution!(ax, vals, audience, AUDIENCE_COLORS[audience]; mode = mode)
    end
    Legend(fig[2, 1], ax, orientation = :horizontal, framevisible = false)
    xlims!(ax, 1, 10)
    save(joinpath(output_dir, "program_scores_$(gram)_$(reducer)_audiences_$(mode).pdf"), fig)
end

function save_program_length_plot(df::DataFrame, output_dir::String; mode::String)
    @assert "n_steps" in names(df) "Expected n_steps column in stats file"
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
    stats_file = abspath(args["stats_file"])
    output_dir = abspath(args["output_dir"])
    @assert isfile(stats_file) "Missing stats file: $stats_file"
    mkpath(output_dir)

    df = CSV.read(stats_file, DataFrame)

    CairoMakie.activate!(type = "pdf")
    with_theme(make_theme()) do
        save_program_score_plot(df, output_dir, "unigram", "mean"; mode = args["score_plot"])
        save_program_score_plot(df, output_dir, "unigram", "min"; mode = args["score_plot"])
        save_program_score_plot(df, output_dir, "bigram", "mean"; mode = args["score_plot"])
        save_program_score_plot(df, output_dir, "bigram", "min"; mode = args["score_plot"])
        save_program_length_plot(df, output_dir; mode = args["program_length_plot"])
    end
end

main()
