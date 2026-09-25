using ArgParse
using CSV
using DataFrames
using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "ml_utils.jl"))

const AUDIENCE_LINE_COLORS = Dict(
    "cs" => "#0072B2",
    "doctor" => "#D55E00",
    "patient" => "#009E73",
    "fi_only" => "#222222",
)

const AUDIENCE_LABELS = Dict(
    "cs" => "CS",
    "doctor" => "Doctor",
    "patient" => "Patient",
    "fi_only" => "Feature importance only",
)

function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--trial_id"
        arg_type = String
        "--output_dir"
        arg_type = String
        "--use_only_bests"
        arg_type = Bool
        "--use_really_all"
        arg_type = Bool
        default = false
        "--algorithm"
        arg_type = String
        default = "lr"
        "--gram"
        arg_type = String
        default = "unigram"
        "--reducer"
        arg_type = String
        default = "mean"
        "--save_name"
        arg_type = String
        default = ""
        "--resnet_test_bacc"
        arg_type = Float64
        default = NaN
        "--mage_test_bacc"
        arg_type = Float64
        default = NaN
    end
    return s
end

function overlay_output_path(parsed_args, suffix::String)
    base_dir = joinpath(parsed_args["output_dir"], parsed_args["trial_id"], "plots", suffix, "global", parsed_args["reducer"])
    mkpath(base_dir)
    if !isempty(strip(parsed_args["save_name"]))
        filename = basename(parsed_args["save_name"])
        return joinpath(base_dir, splitext(filename)[2] == "" ? filename * ".pdf" : filename)
    end
    return joinpath(
        base_dir,
        "$(parsed_args["trial_id"])_$(parsed_args["algorithm"])_global_$(parsed_args["reducer"])_$(parsed_args["gram"])_test_overlay.pdf",
    )
end

function find_one_curve(dir::String, pattern::Regex)
    isdir(dir) || error("Missing curve directory: $dir")
    matches = sort(filter(p -> occursin(pattern, basename(p)), readdir(dir; join = true)))
    length(matches) == 1 || error("Expected exactly one curve CSV in $dir matching $pattern, found $(length(matches)): $(matches)")
    return matches[1]
end

function curve_paths(parsed_args, suffix::String)
    root = joinpath(parsed_args["output_dir"], parsed_args["trial_id"], "plots", suffix)
    algorithm = parsed_args["algorithm"]
    gram = parsed_args["gram"]
    reducer = parsed_args["reducer"]

    paths = Dict{String, String}()
    paths["fi_only"] = find_one_curve(
        joinpath(root, "fi_only", "none"),
        Regex("_$(algorithm)_fi_only_curve\\.csv\$"),
    )
    for audience in ("cs", "doctor", "patient")
        paths[audience] = find_one_curve(
            joinpath(root, audience, reducer),
            Regex("_$(algorithm)_$(audience)_$(reducer)_$(gram)_curve\\.csv\$"),
        )
    end
    return paths
end

function load_curve(path::String)
    df = CSV.read(path, DataFrame)
    @assert "k" in names(df)
    @assert "test_selected" in names(df)
    return (
        k = Float64.(df.k),
        test = Float64.(df.test_selected),
        path = path,
    )
end

function plot_overlay(curves::AbstractDict{String, <:NamedTuple}, outpath::String, parsed_args, suffix::String)
    all_vals = Float64[]
    for key in ("fi_only", "cs", "doctor", "patient")
        append!(all_vals, filter(isfinite, curves[key].test))
    end

    CairoMakie.with_theme(make_font_theme()) do
        fig = CairoMakie.Figure(size = (900, 620))
        ax = CairoMakie.Axis(
            fig[1, 1],
            xlabel = L"\mathrm{Programs\ used}",
            ylabel = L"\mathrm{Balanced\ accuracy}",
            xgridvisible = false,
            ygridvisible = true,
            xlabelsize = 30,
            ylabelsize = 30,
            xticklabelsize = 22,
            yticklabelsize = 22,
            titlealign = :left,
        )
        ax.yticks = ytick_values_for_bacc(all_vals)

        line_styles = Dict(
            "fi_only" => :solid,
            "cs" => :solid,
            "doctor" => :solid,
            "patient" => :solid,
        )
        line_widths = Dict(
            "fi_only" => 5.8,
            "cs" => 4.8,
            "doctor" => 4.8,
            "patient" => 4.8,
        )

        for key in ("fi_only", "cs", "doctor", "patient")
            CairoMakie.lines!(
                ax,
                curves[key].k,
                curves[key].test;
                color = AUDIENCE_LINE_COLORS[key],
                linewidth = line_widths[key],
                linestyle = line_styles[key],
            )
        end

        if has_metric(parsed_args["resnet_test_bacc"])
            CairoMakie.hlines!(ax, [parsed_args["resnet_test_bacc"]], color = (:black, 0.9), linewidth = 3.6, linestyle = :dot)
        end
        if has_metric(parsed_args["mage_test_bacc"])
            CairoMakie.hlines!(ax, [parsed_args["mage_test_bacc"]], color = (:black, 0.9), linewidth = 3.6, linestyle = :dash)
        end

        model_elements = [
            CairoMakie.LineElement(color = AUDIENCE_LINE_COLORS["fi_only"], linewidth = line_widths["fi_only"]),
            CairoMakie.LineElement(color = AUDIENCE_LINE_COLORS["cs"], linewidth = line_widths["cs"]),
            CairoMakie.LineElement(color = AUDIENCE_LINE_COLORS["doctor"], linewidth = line_widths["doctor"]),
            CairoMakie.LineElement(color = AUDIENCE_LINE_COLORS["patient"], linewidth = line_widths["patient"]),
            CairoMakie.LineElement(color = :black, linewidth = 3, linestyle = :dot),
            CairoMakie.LineElement(color = :black, linewidth = 3, linestyle = :dash),
        ]
        model_labels = [
            AUDIENCE_LABELS["fi_only"],
            AUDIENCE_LABELS["cs"],
            AUDIENCE_LABELS["doctor"],
            AUDIENCE_LABELS["patient"],
            "ResNet18",
            "MAGE Alone",
        ]
        CairoMakie.axislegend(
            ax,
            model_elements,
            model_labels;
            position = :rb,
            orientation = :vertical,
            framevisible = true,
            backgroundcolor = (:white, 0.9),
            labelsize = 22,
            patchsize = (52, 22),
            padding = (14, 14, 14, 14),
        )

        CairoMakie.save(outpath, fig)
    end
    @info "Saved global overlay plot" outpath suffix algorithm = parsed_args["algorithm"] gram = parsed_args["gram"] reducer = parsed_args["reducer"]
    return outpath
end

parsed_args = parse_args(build_parser())
@show parsed_args

@assert parsed_args["algorithm"] in ("lr", "rf", "svc") "Argument --algorithm must be one of lr, rf, svc"
@assert parsed_args["gram"] in ("unigram", "bigram") "Argument --gram must be unigram or bigram"
@assert parsed_args["reducer"] in ("mean", "min") "Argument --reducer must be mean or min"

suffix = "$(parsed_args["use_only_bests"])_$(parsed_args["use_really_all"])"
paths = curve_paths(parsed_args, suffix)
@info "Resolved overlay curve paths" paths
curves = Dict(k => load_curve(v) for (k, v) in paths)

ordered_keys = ("fi_only", "cs", "doctor", "patient")
reference_k = curves["fi_only"].k
for key in ordered_keys
    @assert curves[key].k == reference_k "All overlay curves must share the same k grid"
end

outpath = overlay_output_path(parsed_args, suffix)
plot_overlay(curves, outpath, parsed_args, suffix)
println(outpath)
