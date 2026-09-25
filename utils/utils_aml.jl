import StatsBase: sample
import Flux
using StatisticalMeasures
using CategoricalDistributions
using ErrorTypes
using Dates
import Statistics
using MetaGraphsNext
using MLUtils
using Base.Threads
using TimerOutputs
using Base.Threads
using StatsBase

################################
# MODELS PREDICTIONS ###########
################################

abstract type CLSTYPE end
struct BINARYCLS <: CLSTYPE end
struct MULTICLS <: CLSTYPE end

"""
On a single thread. Batch >> pop subset.
So it's looped (batch -> pop)
"""
abstract type ABSTRACT_POP_VS_SAMPLE <: UTCGP.BatchEndpoint end
abstract type ABSTRACT_IND_VS_SAMPLE end

abstract type ABSTRACT_IND_VS_SAMPLE_REG <: ABSTRACT_IND_VS_SAMPLE end
abstract type ABSTRACT_IND_VS_SAMPLE_CLS <: ABSTRACT_IND_VS_SAMPLE end

abstract type ABSTRACT_IND_VS_SAMPLES end

struct IndVsSampleRegCorr <: ABSTRACT_IND_VS_SAMPLE_REG
    pred::Float64
    truth::Float64
    time::Float64
end
struct IndVsSampleRegRMSE <: ABSTRACT_IND_VS_SAMPLE_REG
    pred::Float64
    truth::Float64
    time::Float64
end

"""
ONE IND vs ONE sample. 

The purpose is to have, at the end, a matrix of IndVsSample.
"""
struct IndVsSampleNBACC <: ABSTRACT_IND_VS_SAMPLE_CLS
    pred::Int
    prob_preds::Vector{Float64}
    truth::Int
    time::Float64
end
struct IndVsSampleNF1 <: ABSTRACT_IND_VS_SAMPLE_CLS
    pred::Int
    prob_preds::Vector{Float64}
    truth::Int
    time::Float64
end

"""
Transforms predictions of a single individual for a single instance into a 
IndVsSample. 

- Nan are replaced by 0.
- `argmax` will give the final pred. 
- `Flux.softmax` is used to the probabilities.


Also saves the time that is passed as argument.
This time is normally used to penalize long running individuals.
"""
function transform_preds_to_IndVsSample(T, preds::Vector{Float64}, gt::Int, time::Float64)
    replace_nan_in_preds!(preds)
    pred = binary_decision_from_preds(preds)
    prob_preds = prob_decision_from_preds(preds)
    return T(pred, prob_preds, gt, time)
end

"""
Individual vs several Samples.

This struct is useful to calculate metrics. 
Metric calculations are usually done after all instances have been predicted by a single individual.
"""
struct IndVsSamples{T} <: ABSTRACT_IND_VS_SAMPLES where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
    n_classes::Int
    ind_vs_samples::Vector{T}
    other::Dict
    function IndVsSamples{T}(n_classes::Int, ind_vs_samples::Vector{T}, other::Dict) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        return new{T}(n_classes, ind_vs_samples, other)
    end
    function IndVsSamples{T}(n_classes::Int, ind_vs_samples::Vector{T}) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        return new{T}(n_classes, ind_vs_samples, Dict())
    end

    """
    Received a vector of raw predictions done by a single individual. 
    Each raw prediction and the corresponding ground truth are packed in a IndVsSample. 
    """
    function (s::IndVsSamples{T})(
            ind_preds_per_sample::Vector{<:Vector{<:Number}},
            samples_truth::Vector{Int},
            times::Vector{Float64}
        ) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        nsamples = length(ind_preds_per_sample)
        ind_vs_samples = Vector{T}(undef, nsamples)
        @assert nsamples == length(samples_truth) == length(times)
        for (i, (ind_predictions_for_sample, gt, time)) in enumerate(zip(ind_preds_per_sample, samples_truth, times))
            ind_vs_sample = transform_preds_to_IndVsSample(T, ind_predictions_for_sample, gt, time)
            ind_vs_samples[i] = ind_vs_sample
        end
        return IndVsSamples{T}(
            s.n_classes,
            ind_vs_samples,
        )
    end
end

"""
Population vs One sample.

Normally in Multi Threaded computation, we pass copies of all programs to threads and each thread handles a partition of the dataset. 
For each instance in the subset of instances that the thread has, the results of the computation is stored in a PopVsSampleCLS. 
"""
struct PopVsSampleCLS{T} <: ABSTRACT_POP_VS_SAMPLE where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
    n_classes::Int
    inds_vs_sample::Vector{T}
    other::Dict
    function PopVsSampleCLS{T}(n_classes::Int, inds_vs_sample::Vector{T}, other::Dict) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        return new{T}(n_classes, inds_vs_sample, other)
    end
    function PopVsSampleCLS{T}(n_classes::Int, inds_vs_sample::Vector{T}) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        return new{T}(n_classes, inds_vs_sample, Dict())
    end
    function PopVsSampleCLS{T}(n_classes::Int) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        return new{T}(n_classes, T[], Dict())
    end

    """ 
    Receives a vector with raw predictions done for a single instance by all individuals.
    Raw predictions are turned into IndVsSample.

    Returns another PopVsSampleCLS.
    """
    function (a::PopVsSampleCLS{T})(
            pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ].
            truth::Int, # one obs
            times::Vector{Float64}
        ) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
        n = length(pop_preds)
        @assert n == length(times)
        inds_vs_sample = Vector{T}(undef, n)
        for (i, (ind_predictions, time)) in enumerate(zip(pop_preds, times))
            ind_vs_sample = transform_preds_to_IndVsSample(T, ind_predictions, truth, time)
            inds_vs_sample[i] = ind_vs_sample
        end
        return PopVsSampleCLS{T}(
            a.n_classes,
            inds_vs_sample,
        )
    end
end

# REGRESSION ##############################################

abstract type ABSTRACT_POP_VS_SAMPLE_REG <: ABSTRACT_POP_VS_SAMPLE end

"""
Holding metrics Pop vs Sample where the metric:
    - is 1 - |cor|
    - rmse
"""
struct PopVsSampleREG{T} <: ABSTRACT_POP_VS_SAMPLE_REG where {T <: Union{IndVsSampleRegCorr, IndVsSampleRegRMSE}}
    inds_vs_sample::Vector{T}
    other::Dict
    function PopVsSampleREG{T}(inds_vs_sample::Vector{T}, other::Dict) where {T <: Union{IndVsSampleRegCorr, IndVsSampleRegRMSE}}
        return new(inds_vs_sample, other)
    end
    function PopVsSampleREG{T}(inds_vs_sample::Vector{T}) where {T <: Union{IndVsSampleRegCorr, IndVsSampleRegRMSE}}
        return new(inds_vs_sample, Dict())
    end
    function PopVsSampleREG{T}() where {T <: Union{IndVsSampleRegCorr, IndVsSampleRegRMSE}}
        return new(Vector{T}(), Dict())
    end
end

function (a::PopVsSampleREG{T})(
        pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ].
        truth::Number, # one obs
        times::Vector{Float64}
    ) where {T <: Union{IndVsSampleRegCorr, IndVsSampleRegRMSE}}
    n = length(pop_preds)
    @assert n == length(times)
    inds_vs_sample = Vector{T}(undef, n)
    for (i, outs) in enumerate(pop_preds)
        inds_vs_sample[i] = T(outs[1], truth, times[i])
    end
    return PopVsSampleREG{T}(
        inds_vs_sample
    )
end

get_ind_vs_sample_type(::PopVsSampleCLS{IndVsSampleNBACC}) = IndVsSampleNBACC
get_ind_vs_sample_type(::PopVsSampleCLS{IndVsSampleNF1}) = IndVsSampleNF1
get_ind_vs_sample_type(::PopVsSampleREG{T}) where {T} = T
get_metric_of_interest(::Type{IndVsSampleNBACC}) = :nbacc
get_metric_of_interest(::Type{IndVsSampleNF1}) = :nf1
get_metric_of_interest(::Type{<:ABSTRACT_IND_VS_SAMPLE_REG}) = :fitness

"""
Gets the probability predictions for each instance.
"""
function extract_prob_preds(ind_vs_samples::ABSTRACT_IND_VS_SAMPLES)
    return map(x -> x.prob_preds, ind_vs_samples.ind_vs_samples)
end

"""
Gets the argmax decision for each instance.
"""
function extract_preds(ind_vs_samples::ABSTRACT_IND_VS_SAMPLES)
    return map(x -> x.pred, ind_vs_samples.ind_vs_samples)
end

"""
Gets the ground truth for each instance.
"""
function extract_gts(ind_vs_samples::ABSTRACT_IND_VS_SAMPLES)
    return map(x -> x.truth, ind_vs_samples.ind_vs_samples)
end

"""
Returns the levels : all classes. 
ex : [1,2] for binary classification
"""
function get_levels(n_classes::Int)
    return collect(1:n_classes)
end

"""
Returns the levels : all classes.

The upper bound is given by the `n_classes` parameter in IndVsSamples.
ex : [1,2] for binary classification
"""
function get_levels(v::ABSTRACT_IND_VS_SAMPLES)
    return get_levels(v.n_classes)
end

########################
# LOSS UTILITIES #######
########################

# Handle Preds directly
"""
Replaces nan by 0.0 in place.
"""
function replace_nan_in_preds!(preds::Vector{Float64})
    return replace!(preds, NaN => 0.0)
end

"""
Returns the argmax position
"""
function binary_decision_from_preds(preds)
    return argmax(preds)
end

"""
Returns the softmax of all predictions
"""
function prob_decision_from_preds(preds::Vector{Float64})
    return s = Flux.softmax(preds)
end

# function confidence_penalty(preds::Vector{Float64})
#     @assert length(preds) == 2
#     0.01 * abs(preds[1] - preds[2]) # penalty if too confident about one class
# end

"""
Returns a `UnivariateFinite` for the predictions.

- `preds` are supposed to be probabilities for each class.
- `names` correspond to the labels of each class.
"""
function prob_decisions_to_pdf(preds::Vector{Float64}, names::Vector{Int})
    return UnivariateFinite(names, preds, pool = missing)
end

#----# FN TO OPTIM #----#

"""
This will be the final function that is optimized.
It uses the global parameter `ERR_WEIGHT` which is the coefficient for 
the error score and `NERR_WEIGHT` is the coefficient for the nll loss.
"""
function calc_loss(err::Float64, nll_loss::Float64)
    global ERR_WEIGHT, NERR_WEIGHT
    return ERR_WEIGHT * err + NERR_WEIGHT * nll_loss
end

#----# Calculate All Metrics for individual #----#

"""
Calculates all metrics for a single individual.

The only input needed is a `IndVsSamples` struct, that is, the predictions done by a 
single individual for all the samples.
"""
function calc_binary_metrics(ind_vs_samples::IndVsSamples{T}) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
    levels = get_levels(ind_vs_samples)
    p_ŷ = extract_prob_preds(ind_vs_samples)
    ŷ = extract_preds(ind_vs_samples)
    y = extract_gts(ind_vs_samples)
    pmfs = prob_decisions_to_pdf.(p_ŷ, Ref(levels))
    acc = StatisticalMeasures.accuracy(ŷ, y)
    err = 1 - acc
    tpr = StatisticalMeasures.TruePositiveRate(levels = levels)(ŷ, y) # hit rate
    fpr = StatisticalMeasures.FalsePositiveRate(levels = levels)(ŷ, y) # fallout
    tnr = 1 - fpr
    ppv = StatisticalMeasures.PositivePredictiveValue(levels = levels)(ŷ, y)
    f1 = StatisticalMeasures.FScore(levels = levels)(ŷ, y)
    nf1 = 1.0 - f1
    auc = StatisticalMeasures.AreaUnderCurve()(pmfs, y)
    bacc = StatisticalMeasures.BalancedAccuracy()(ŷ, y)
    berr = 1 - bacc
    nll_loss = StatisticalMeasures.LogLoss()(pmfs, y)

    if get_metric_of_interest(T) == :nbacc
        neg_metric = berr
    elseif get_metric_of_interest(T) == :nf1
        # @info nf1 nll_loss calc_loss(nf1, nll_loss)
        neg_metric = nf1
    end
    loss = calc_loss(neg_metric, nll_loss)
    return Dict{Symbol, Float64}(
        :acc => acc,
        :err => err,
        :tpr => tpr,
        :fpr => fpr,
        :tnr => tnr,
        :ppv => ppv,
        :f1 => f1,
        :nf1 => nf1,
        :auc => auc,
        :berr => berr,
        :bacc => bacc,
        :nbacc => -bacc,
        :nll_loss => nll_loss,
        :loss => loss
    )
end

"""
Calculates selected metrics (nll_loss, berr, bacc, loss) for a single individual in a multi-class setting.

The only input needed is a `IndVsSamples` struct, that is, the predictions done by a 
single individual for all the samples.
"""
function calc_multi_metrics(ind_vs_samples::IndVsSamples{T}) where {T <: ABSTRACT_IND_VS_SAMPLE_CLS}
    levels = get_levels(ind_vs_samples)
    p_ŷ = extract_prob_preds(ind_vs_samples)
    y = extract_gts(ind_vs_samples)
    pmfs = prob_decisions_to_pdf.(p_ŷ, Ref(levels))
    bacc = StatisticalMeasures.BalancedAccuracy()(extract_preds(ind_vs_samples), y)
    berr = 1 - bacc
    nll_loss = StatisticalMeasures.LogLoss()(pmfs, y)
    @assert get_metric_of_interest(T) == :nbacc "No f1 in multi metrics"
    loss = calc_loss(berr, nll_loss)
    return Dict{Symbol, Float64}(
        :berr => berr,
        :bacc => bacc,
        :nbacc => -bacc,
        :nll_loss => nll_loss,
        :loss => loss
    )
end

function calc_metrics(ind_vs_samples::IndVsSamples, cls_type::BINARYCLS)
    return calc_binary_metrics(ind_vs_samples)
end
function calc_metrics(ind_vs_samples::IndVsSamples, cls_type::MULTICLS)
    return calc_multi_metrics(ind_vs_samples)
end

"""
Considers the Vector{IndVsSample} as all predictions done by a single individual for all samples ⇒ IndVsSamples 

The vector of IndVsSamples comes from indexing the Matrix{IndVsSample}(ind, samples) (a row). 
"""
function calculate_metrics_ind_vs_samples(ind_vs_samples::Vector{C}, n_classes::Int, other::Dict = Dict()) where {C <: ABSTRACT_IND_VS_SAMPLE_CLS}
    inds_vs_samples_ = IndVsSamples{C}(n_classes, ind_vs_samples, other)
    cls_type = length(get_levels(inds_vs_samples_)) == 2 ? BINARYCLS() : MULTICLS()
    metrics_dict = calc_metrics(inds_vs_samples_, cls_type)
    times = [holder.time for holder in ind_vs_samples]
    metrics_dict[:time] = mean(times)
    metrics_dict[:fitness] = metrics_dict[:loss]
    return metrics_dict

end

function calculate_metrics_ind_vs_samples(ind_vs_samples::Vector{IndVsSampleRegRMSE}, other::Dict = Dict())
    preds_ind = [holder.pred for holder in ind_vs_samples]
    truths = [holder.truth for holder in ind_vs_samples]
    times = [holder.time for holder in ind_vs_samples]

    std(truths) < 0.0001 ? println("WARN low min std !! ") : nothing

    fitness = StatisticalMeasures.RootMeanSquaredError()(preds_ind, truths)
    fitness = isnan(fitness) ? 100 : fitness
    fitness = isinf(fitness) ? 100 : fitness

    return Dict(
        :fitness => fitness,
        :time => mean(times)
    )
end

function calculate_metrics_ind_vs_samples(ind_vs_samples::Vector{IndVsSampleRegCorr}, other::Dict = Dict())
    preds_ind = [holder.pred for holder in ind_vs_samples]
    truths = [holder.truth for holder in ind_vs_samples]
    times = [holder.time for holder in ind_vs_samples]

    min_std = std(truths) * 0.01
    min_std < 0.0001 ? println("WARN low min std !! ") : nothing
    min_std = clamp(min_std, 0.00001, min_std)
    if std(preds_ind) <= min_std # cor would give wrong result otherwise
        fitness = 100
    else
        # r = cor(preds_ind, truths)
        r = corspearman(preds_ind, truths)
        fitness = 1 - abs(r)
        fitness = isnan(fitness) ? 100 : fitness
        fitness = isinf(fitness) ? 100 : fitness
    end
    return Dict(
        :fitness => fitness,
        :time => mean(times)
    )
end

"""
Gets the inner IndsVsSample vector from a PopVsSampleCLS. 
Returns an Option since initialized structs might have an empty vector, which means they do not hold any data yet.
"""
function Main.get_endpoint_results(container::PopVsSampleCLS{C})::Option{Vector{C}} where {C <: ABSTRACT_IND_VS_SAMPLE_CLS}
    if length(container.inds_vs_sample) == 0
        return ErrorTypes.none
    end
    return some(container.inds_vs_sample)
end
function Main.get_endpoint_results(container::PopVsSampleREG{T})::Option{Vector{T}} where {T}
    length(container.inds_vs_sample) == 0 && return ErrorTypes.none
    return some(container.inds_vs_sample)
end

"""
From a Matrix Pop x Samples, calculates the metrics for each individual.

For all rows (individuals), it calls `calculate_metrics_ind_vs_samples`. 

Returns a vector of Dict{Symbol, Any}.
"""
function calculate_metrics_pop_vs_samples(preds::Matrix{C}, endpoint) where {C <: ABSTRACT_IND_VS_SAMPLE_CLS}
    n_classes = endpoint.n_classes
    other = endpoint.other
    # Calculate sample wise
    n_individuals = size(preds, 1)
    Metrics_per_individual = Vector{Dict{Symbol, Any}}(undef, n_individuals)
    for ind_idx in 1:size(preds, 1)
        Metrics_per_individual[ind_idx] = calculate_metrics_ind_vs_samples(
            preds[ind_idx, :],
            n_classes,
            other
        )
    end
    return Metrics_per_individual
end

function calculate_metrics_pop_vs_samples(
        M::Matrix{T},
        endpoint
    ) where {T <: ABSTRACT_IND_VS_SAMPLE_REG}
    n_inds, n_samples = size(M)

    # compute correlation loss per ind
    losses_per_ind = zeros(n_inds)
    times_per_ind = zeros(n_inds)
    for i in 1:n_inds
        ind_vs_samples = M[i, :]
        metrics = calculate_metrics_ind_vs_samples(ind_vs_samples)
        losses_per_ind[i] = metrics[:fitness]
        times_per_ind[i] = metrics[:time]
    end

    metrics = Vector{Dict}(undef, n_inds)
    for i in 1:n_inds
        metrics[i] = Dict(
            :loss => losses_per_ind[i],
            :fitness => losses_per_ind[i],
            :time => times_per_ind[i],
        )
    end
    return metrics
end

#############################
# ACC CALLBACK ##############
#############################
struct acc_callback <: UTCGP.AbstractCallable
    X_test
    Y_test
    subset
end

function run_individual_against_a_dataset(program::UTCGP.IndividualPrograms, xs, ys, model_arch, meta_library, TaskType, endpoint)
    n = length(xs)
    @assert n == length(ys)
    indices = 1:n
    nt = Threads.nthreads()
    M_ind_vs_samples = Vector{TaskType}(undef, n)
    n_samples_per_thread = Base.ceil(Int, n / nt)
    tasks = []
    for ith_x in Iterators.partition(collect(indices), n_samples_per_thread) # paralellize over samples
        slot_for_metrics = @view M_ind_vs_samples[ith_x]
        batch_x = xs[ith_x]
        batch_y = ys[ith_x]
        t = Threads.@spawn begin
            evaluate_batch_single_individual!(
                batch_x, batch_y, slot_for_metrics,
                deepcopy(program), model_arch, meta_library,
                TaskType, endpoint
            )
        end
        push!(tasks, t)
    end
    fetch.(tasks)

    # Calculate metrics
    metrics = calculate_metrics_pop_vs_samples(
        reshape(
            M_ind_vs_samples,
            (1, size(M_ind_vs_samples)[1])
        )
        , endpoint
    )
    return metrics[1]
end

function evaluate_batch_single_individual!(
        xs::Vector,
        ys::Vector,
        metrics_slot::SubArray,
        mage::UTCGP.IndividualPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary,
        tasktype::Type{<:ABSTRACT_IND_VS_SAMPLE},
        endpoint_callback
    )
    @debug "Started eval at Thread $(Threads.threadid())"
    @assert length(metrics_slot) == length(xs) == length(ys) "Incorrect length for Thread slot"
    pop_of_one = UTCGP.PopulationPrograms([mage])
    for (sample_idx, (x, y)) in enumerate(zip(xs, ys))
        UTCGP.reset_programs!.(pop_of_one)
        UTCGP.replace_shared_inputs!(
            pop_of_one,
            x,
        )
        outputs, times = UTCGP.evaluate_population_programs_with_time(
            pop_of_one,
            model_arch,
            meta_library,
        )
        outputs = INTER_ACT[].(outputs)

        # Endpoint results
        pop_of_one_vs_sample = endpoint_callback(outputs, y, times)

        vec_of_ind_vs_sample =
            Main.get_endpoint_results(pop_of_one_vs_sample)

        if is_error(vec_of_ind_vs_sample)
            @error "PopVsSample was empty. Maybe not called before with predictions?"
            throw(ErrorException())
        end
        metrics_slot[sample_idx] = unwrap(vec_of_ind_vs_sample)[1] # col_idx refers to the idx of the sample in the view. : because it's all the inds
    end
    return @debug "Ended eval at Thread $(Threads.threadid())"
end

# ###########################
# # CHECKPOINT ##############
# ###########################
# """
# Saves the passed genome to a file.
# """
function save_payload(
        best_genome,
        folder::String,
        name::String = "best_genome.pickle"
    )
    payload = Dict()
    payload["best_genome"] = deepcopy(best_genome)
    genome_path = joinpath(folder, name)
    return open(genome_path, "w") do io
        @info "Writing payload to $genome_path"
        write(io, UTCGP.general_serializer(payload))
    end
end
struct checkpoint <: UTCGP.AbstractCallable
    every::Int
    folder::String
end
function (c::checkpoint)(
        iteration,
        genome
    )
    return if iteration % c.every == 0
        save_payload(genome, c.folder, "checkpoint_$iteration.pickle")
    end
end

mutable struct jsonTrackerGA <: UTCGP.AbstractCallable
    tracker::UTCGP.jsonTracker
    acc_callback::acc_callback
    label::String
    test_losses::Union{Nothing, Vector}
    best_ind #::Union{MNModel, UTCGP.UTGenome, Nothing}
    best_loss::Float64
    checkpoint
end

function unique_active_node_count(ind_programs::UTCGP.IndividualPrograms)
    return length(unique(node.id for node in UTCGP.get_active_nodes(ind_programs)))
end

"""
Track Metrics for MAGE

Runs the best program against data.
Minimizes.
Stores metrics over time. 
"""
function (jtga::jsonTrackerGA)(
        ind_performances::Union{Vector{<:Number}, Vector{Vector{<:Number}}},
        population::Population,
        generation::Int,
        run_config::AbstractRunConf,
        model_architecture::modelArchitecture,
        node_config::nodeConfig,
        meta_library::MetaLibrary,
        shared_inputs::SharedInput,
        programs::UTCGP.PopulationPrograms,
        best_loss::Union{Float64, Vector{Float64}},
        best_program::Union{UTCGP.IndividualPrograms, Vector{UTCGP.IndividualPrograms}},
        elite_idx::Union{Int, Vector{Int}},
        Batch::SubArray;
        extras::Dict = Dict()
    )
    TASKTYPE = extras[:task_type]
    METRIC_OF_INTEREST = extras[:metric_of_interest]
    ENDPOINT = extras[:endpoint]
    best_ind_idx = elite_idx[1]
    best_prog = best_program[1]
    best_ind_fitness = ind_performances[best_ind_idx]
    best_ind = population[best_ind_idx]
    current_elite_unique_active_nodes = unique_active_node_count(best_prog)
    best_so_far_unique_active_nodes = if isnothing(jtga.best_ind)
        current_elite_unique_active_nodes
    else
        best_so_far_program = UTCGP.decode_with_output_nodes(
            jtga.best_ind,
            meta_library,
            model_architecture,
            shared_inputs,
        )
        unique_active_node_count(best_so_far_program)
    end
    @info "Best index : $best_ind_idx & fitness : $best_ind_fitness"
    @warn "Val active-node counts" generation best_so_far_unique_active_nodes current_elite_unique_active_nodes
    return if !isnothing(jtga.test_losses)
        metrics = run_individual_against_a_dataset(best_prog, jtga.acc_callback.X_test, jtga.acc_callback.Y_test, model_architecture, meta_library, TASKTYPE, endpoint)
        fitness = metrics[METRIC_OF_INTEREST]
        @warn "JTT $(jtga.label) Val metric ($METRIC_OF_INTEREST) : $(fitness). BEST $(jtga.best_loss)"
        payload = Dict{Any, Any}(metrics...)
        payload[:data] = jtga.label
        payload[:iteration] = generation
        ind = deepcopy(best_ind)
        jtga.checkpoint(generation, ind) # the val tracker writes checkpoint
        push!(jtga.test_losses, fitness)
        if fitness <= jtga.best_loss
            # we have a new best
            jtga.best_ind = ind
            jtga.checkpoint(0, jtga.best_ind) # always rewrite the best ind
        end
        jtga.best_loss = minimum(jtga.test_losses)
        best_metric = jtga.best_loss
        write(jtga.tracker.file, JSON.json(payload), "\n")
        flush(jtga.tracker.file)
        @info "Best VAL METRIC :$(jtga.best_loss)"
    else
        metrics = run_individual_against_a_dataset(best_prog, jtga.acc_callback.xs, jtga.acc_callback.ys, model_architecture, meta_library, TASKTYPE)
        @warn "JTT $(jtga.label) METRIC: $(metrics[METRIC_OF_INTEREST])"
    end
end

# #######################
# # FITTER ##############
# #######################

# # MEAN BATCH #
# """
# """

function fit_ga_meanbatch_mt(
        X::Any,
        Y::Union{Any, Nothing},
        shared_inputs::UTCGP.SharedInput,
        initial_pop::Vector{UTCGP.UTGenome},
        model_architecture::modelArchitecture,
        node_config::nodeConfig,
        run_config::RunConfGA,
        meta_library::MetaLibrary,
        # Callbacks before training
        pre_callbacks::UTCGP.Optional_FN,
        # Callbacks before step (before looping through data)
        population_callbacks::UTCGP.Mandatory_FN,
        mutation_callbacks::UTCGP.Mandatory_FN,
        output_mutation_callbacks::UTCGP.Mandatory_FN,
        decoding_callbacks::UTCGP.Mandatory_FN,
        # Callbacks per step (while looping through data)
        endpoint_callback::ABSTRACT_POP_VS_SAMPLE,
        final_step_callbacks::UTCGP.Optional_FN,
        # Callbacks after step ::
        elite_selection_callbacks::UTCGP.Mandatory_FN,
        epoch_callbacks::UTCGP.Optional_FN,
        early_stop_callbacks::UTCGP.Optional_FN,
        last_callback::UTCGP.Optional_FN,
        ; use_cma::Bool = false,
        cma_at::Int = 0,
        cma_sigma::Float64 = 5.0
    ) # Tuple{UTGenome, IndividualPrograms, GenerationLossTracker}::

    global TIMEPENALTY

    IND_VS_SAMPLE_TYPE = get_ind_vs_sample_type(endpoint_callback)
    METRIC_OF_INTEREST = get_metric_of_interest(IND_VS_SAMPLE_TYPE)

    @warn IND_VS_SAMPLE_TYPE
    @warn METRIC_OF_INTEREST

    population = Population(initial_pop)
    early_stop = false; best_programs = nothing
    elite_idx = Int[i for i in 1:length(population)]
    ind_performances = Float64[1.0 for i in 1:length(population)]
    # DL
    BatchSize = X.batch_size
    TrainSize = length(X)

    # PRE CALLBACKS
    UTCGP._make_pre_callbacks_calls(pre_callbacks)
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    x0 = nothing
    cma = nothing
    cma_vals = nothing
    if use_cma
        cma_nodes = get_cma_nodes(population[1], cma_at)
        x0 = map(x -> x.value[], cma_nodes)
        s = rand(1:100) # TODO
        cma = create_cma_es(x0, cma_sigma, Dict("seed" => s); popsize = run_conf.n_new)
        @info "Init cma with $x0, seed $s and popsize $(run_conf.n_new)"
    end

    for iteration in 1:run_config.generations
        early_stop ? break : nothing
        @warn "Iteration : $iteration of $(run_config.generations)"
        # Population
        ga_pop_args = GA_POP_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            ind_performances,
            elite_idx
        )
        population, time_pop =
            @unwrap_or UTCGP._make_ga_population(ga_pop_args, population_callbacks) throw(
            "Could not unwrap make_population",
        )

        # Program mutations ---
        ga_mutation_args = GA_MUTATION_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
        )
        population, time_mut =
            @unwrap_or UTCGP._make_ga_mutations!(ga_mutation_args, mutation_callbacks) throw(
            "Could not unwrap make_ga_mutations",
        )
        if use_cma
            # mutate on non elite
            from = run_conf.n_elite + 1
            to = length(population)
            cma_vals = mutate_cma!(population.pop[from:to], cma, cma_at)
            if isdefined(Main, :Infiltrator)
                Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
            end
            @info "Mutated CMA values"
        end

        # Output mutations ---
        # population, time_out_mut = @unwrap_or _make_ga_output_mutations!(
        #     ga_mutation_args,
        #     output_mutation_callbacks,
        # ) throw("Could not unwrap make_ga_output_mutations")

        # Genotype to Phenotype mapping ---
        population_programs, time_pop_prog = UTCGP._make_decoding(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
            decoding_callbacks,
        )

        # Storage for all evaluations (pop * X)
        # Parallelized by spawning individuals in the full dataset
        @warn "Multi Threaded Graphs evals"
        UTCGP.reset_programs!(population_programs)
        n_pop = length(population)
        n_threads = Threads.nthreads()
        n_samples = TrainSize
        batch, _ = sample(X, n_samples) # sample a mini batch
        ALL_BATCHS = batch
        M_individual_loss_tracker = Matrix{IND_VS_SAMPLE_TYPE}(undef, n_pop, n_samples) # storage for all losses
        inds_per_thread = ceil(Int, n_pop / nt) # at least 1
        tasks = []
        for ith_inds in Iterators.partition(1:n_pop, inds_per_thread) # partition the population in threads
            t = Threads.@spawn begin
                @info "Spawn training $ith_inds on Thread : $(threadid()) $(now()) "
                store_col = @view M_individual_loss_tracker[ith_inds, :] # where to place the results (in the score matrix)
                data = @view batch[:] # all the batch
                progs = UTCGP.PopulationPrograms(population_programs[collect(ith_inds)]) # subset of pop
                @timeit_debug to "ga_fit_mt. evalbatch" _eval_batch_on_population(
                    data,
                    store_col,
                    progs,
                    model_architecture,
                    meta_library,
                    endpoint, # pop vs sample
                )
            end
            push!(tasks, t)
        end
        @debug "Waiting on all training batches. $(now()) "
        fetch.(tasks)
        @debug "All workers are done with their batches. All fitness were fetched. $(now())"

        Metrics_per_individual = calculate_metrics_pop_vs_samples(M_individual_loss_tracker, endpoint)
        ind_performances = map(x -> x[:loss], Metrics_per_individual)
        ind_mean_time_for_batch = map(x -> x[:time], Metrics_per_individual)
        mean_times = ind_mean_time_for_batch ./ TIMEPENALTY
        ind_performances .+= mean_times

        # CMA tell
        if use_cma
            from = run_conf.n_elite + 1
            to = length(population)
            tell(cma, cma_vals, ind_performances[from:to])
            log_add(cma)
            disp(cma)
            @info "Updated CMA with fitness values"
        end

        # final step call...
        if !isnothing(final_step_callbacks)
            for final_step_callback in final_step_callbacks
                UTCGP.get_fn_from_symbol(final_step_callback)()
            end
        end

        # Selection
        # ind_performances = resolve_ind_loss_tracker(M_individual_loss_tracker)
        # Elite selection callbacks
        ga_selection_args = GA_SELECTION_ARGS(
            ind_performances,
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            population_programs,
        )

        elite_idx, time_elite = @unwrap_or UTCGP._make_ga_elite_selection(
            ga_selection_args,
            elite_selection_callbacks,
        ) throw("Could not unwrap make_ga_selection")
        @show elite_idx

        elite_program = population_programs[elite_idx[1]]
        if maximum(length.(elite_program.programs)) <= 100
            println(elite_program)
        end

        elite_fitnesses = ind_performances[elite_idx]
        elite_best_fitness = minimum(skipmissing(elite_fitnesses))
        elite_best_ftiness_idx = argmin(elite_fitnesses)
        elite_avg_fitness = mean(skipmissing(elite_fitnesses))
        elite_std_fitness = std(filter(!isnan, ind_performances))
        best_programs = population_programs[elite_idx]
        elite_unique_active_nodes = map(best_programs) do ind_prog
            active_nodes = UTCGP.get_active_nodes(ind_prog)
            return length(unique(node.id for node in active_nodes))
        end
        @info "Elite unique active nodes" generation = iteration counts = elite_unique_active_nodes

        try
            histogram(ind_performances) |> println
        catch e
            @show e
            @error "Could not drawn histogram"
        end

        # Subset Based on Elite IDX---
        # old_pop = deepcopy(population.pop[elite_idx])
        # empty!(population.pop)
        # push!(population.pop, old_pop...)
        # ind_performances = ind_performances[elite_idx]

        # EPOCH CALLBACK
        if !isnothing(epoch_callbacks)
            UTCGP._make_epoch_callbacks_calls(
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
                view(ALL_BATCHS, :),
                epoch_callbacks;
                extras = Dict(:task_type => IND_VS_SAMPLE_TYPE, :metric_of_interest => METRIC_OF_INTEREST, :endpoint => endpoint)
            )
        end

        # if use_cma && iteration % 10 == 0
        # plot_and_save_cma(cma)
        # if isdefined(Main, :Infiltrator)
        # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
        # end
        # end

        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, elite_best_fitness)
        println(
            "Iteration $iteration.
            Best fitness: $(round(elite_best_fitness, digits = 10)) at index $elite_best_ftiness_idx
            Elite mean fitness : $(round(elite_avg_fitness, digits = 10)). Std: $(round(elite_std_fitness)) at indices : $(elite_idx)",
        )

        # EARLY STOP CALLBACK # TODO
        EMPTY_TRACKER = UTCGP.IndividualLossTrackerMT(length(population), TrainSize)
        if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
            early_stop_args = UTCGP.GA_EARLYSTOP_ARGS(
                M_gen_loss_tracker,
                EMPTY_TRACKER,
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
            )
            early_stop =
                UTCGP._make_ga_early_stop_callbacks_calls(early_stop_args, early_stop_callbacks) # true if any
        end

        if early_stop
            g = run_config.generations
            @warn "Early returning at iteration : $iteration from $g total iterations"
            if !isnothing(last_callback)
                last_callback(
                    ind_performances,
                    population,
                    iteration,
                    run_config,
                    model_architecture,
                    node_config,
                    meta_library,
                    population_programs,
                    elite_fitnesses,
                    best_programs,
                    elite_idx,
                )
            end
            # UTCGP.show_program(program)
            return tuple(population.pop, best_programs, M_gen_loss_tracker)
        end
        gct = @elapsed GC.gc(true)
        @warn "Running GC at the end of iteration. GC time : $gct"
    end
    return (population.pop, best_programs, M_gen_loss_tracker)
end


"""
Evaluates all programs in `non_shared_pop_programs` against each element in the `batch`, one by one.

Hence creating, for each instance, a Vec{IndVsSample} with length equal to the number of programs.
The execution time for each individual is also captured by `UTCGP.evaluate_population_programs_with_time`.

Once a Vec{IndVsSample} is obtained, it will replace a column in the view of the Matrix ind x Samples.
"""
function _eval_batch_on_population(
        batch::SubArray,
        store_view::SubArray{T},
        non_shared_pop_programs::UTCGP.PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary, endpoint_callback::Union{Type{<:UTCGP.BatchEndpoint}, <:UTCGP.BatchEndpoint}
    ) where {T <: ABSTRACT_IND_VS_SAMPLE}
    global to
    tid = Threads.threadid()
    @info "Started Batch fit at Thread $(tid). $(now())"
    non_shared_pop_programs = deepcopy(non_shared_pop_programs)
    @timeit_debug to "fit_mt. Thread eval loop" for (col_idx, (x, y)) in enumerate(batch) # Iterate over DATA because in a single Thread data >> pop
        @timeit_debug to "fit_mt. Reset Progs" UTCGP.reset_programs!(
            non_shared_pop_programs,
        )
        # append input nodes to pop
        @timeit_debug to "fit_mt. replace inputs" UTCGP.replace_shared_inputs!(
            non_shared_pop_programs,
            x,
        ) # update
        @timeit_debug to "fit_mt. Eval pop" time_eval =
            @elapsed outputs, times = UTCGP.evaluate_population_programs_with_time(
            non_shared_pop_programs,
            model_arch,
            meta_library,
        )
        outputs = INTER_ACT[].(outputs)

        # Endpoint results
        @timeit_debug to "fit_mt. Endpoint" pop_vs_sample = endpoint_callback(outputs, y, times)

        @timeit_debug to "fit_mt. get results" vec_of_ind_vs_sample =
            Main.get_endpoint_results(pop_vs_sample)

        if is_error(vec_of_ind_vs_sample)
            @error "PopVsSample was empty. Maybe not called before with predictions?"
            throw(ErrorException())
        end
        store_view[:, col_idx] = unwrap(vec_of_ind_vs_sample) # col_idx refers to the idx of the sample in the view. : because it's all the inds
    end
    return @info "Ended Batch fit at Thread $(tid). $(now())"
end


# MCTS
"""
"""

# function fit_MCTS_meanbatch_mt(
#         graph,
#         X::Any,
#         Y::Union{Any, Nothing},
#         n_repetitions::Int,
#         shared_inputs::UTCGP.SharedInput,
#         genome::UTCGP.UTGenome,
#         model_architecture::modelArchitecture,
#         node_config::nodeConfig,
#         run_config::RunConfGA,
#         meta_library::MetaLibrary,
#         # Callbacks before training
#         pre_callbacks::UTCGP.Optional_FN,
#         # Callbacks before step (before looping through data)
#         population_callbacks::UTCGP.Mandatory_FN,
#         mutation_callbacks::UTCGP.Mandatory_FN,
#         output_mutation_callbacks::UTCGP.Mandatory_FN,
#         decoding_callbacks::UTCGP.Mandatory_FN,
#         # Callbacks per step (while looping through data)
#         endpoint_callback::Union{Type{<:UTCGP.BatchEndpoint}, <:UTCGP.BatchEndpoint},
#         final_step_callbacks::UTCGP.Optional_FN,
#         # Callbacks after step ::
#         elite_selection_callbacks::UTCGP.Mandatory_FN,
#         epoch_callbacks::UTCGP.Optional_FN,
#         early_stop_callbacks::UTCGP.Optional_FN,
#         last_callback::UTCGP.Optional_FN,
#     ) # Tuple{UTGenome, IndividualPrograms, GenerationLossTracker}::

#     local early_stop, best_programs, elite_idx, population, ind_performances, =
#         UTCGP._ga_init_params(genome, run_config)
#     # DL
#     BatchSize = X.batch_size
#     TrainSize = length(X)

#     # PRE CALLBACKS
#     UTCGP._make_pre_callbacks_calls(pre_callbacks)
#     M_gen_loss_tracker = UTCGP.GenerationLossTracker()

#     # add the root node
#     root_node = MCTSNodeLabel(:root_node)
#     graph.g[root_node] = MCTSNode(0, 0.0, MCTSProgram(population[1]))
#     @info "Graph size after pushing root node : $(length(graph.g))"

#     for iteration in 1:run_config.generations
#         early_stop ? break : nothing
#         @warn "Iteration : $iteration of $(run_config.generations)"

#         # Select a branch to expand
#         node_label = MAGE_MCTS.select(graph.g, root_node)
#         @info "Node Label is : $(node_label)"

#         # if iteration == 2
#         #     break
#         # end

#         # Expand based on branching factor
#         new_node, new_label = expand(
#             graph, node_label, 2,
#             Dict(
#                 "run_config" => run_config,
#                 "model_architecture" => model_architecture,
#                 "meta_library" => meta_library,
#                 "shared_inputs" => shared_inputs
#             )
#         ) # returns a node

#         # Rollout
#         rollout_result_label, reward = rollout(
#             graph, new_label, endpoint, Dict(
#                 "run_config" => run_config,
#                 "model_architecture" => model_architecture,
#                 "meta_library" => meta_library,
#                 "shared_inputs" => shared_inputs,
#                 "TRAIN_SIZE" => TrainSize,
#                 "X" => X,
#                 "BATCH_SIZE" => BatchSize,
#                 "endpoint" => endpoint
#             )
#         )

#         # Update
#         backpropagate(graph.g, rollout_result_label, reward)


#         # EPOCH CALLBACK
#         if !isnothing(epoch_callbacks)
#             last_leaf = graph.g[rollout_result_label]
#             individual = last_leaf.program.code
#             decoded_individual = UTCGP.decode_with_output_nodes(individual, meta_library, model_architecture, shared_inputs)

#             UTCGP._make_epoch_callbacks_calls(
#                 [reward * -1.0],
#                 UTCGP.Population([individual]),
#                 iteration,
#                 run_config,
#                 model_architecture,
#                 node_config,
#                 meta_library,
#                 shared_inputs,
#                 UTCGP.PopulationPrograms([decoded_individual]),
#                 [reward * -1.0],
#                 [decoded_individual],
#                 1,
#                 view([], :),
#                 epoch_callbacks,
#             )
#         end

#         # store iteration loss/fitness
#         # UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, elite_best_fitness)
#         # println(
#         #     "Iteration $iteration.
#         #     Best fitness: $(round(elite_best_fitness, digits = 10)) at index $elite_best_ftiness_idx
#         #     Elite mean fitness : $(round(elite_avg_fitness, digits = 10)). Std: $(round(elite_std_fitness)) at indices : $(elite_idx)",
#         # )

#         # EARLY STOP CALLBACK # TODO
#         # EMPTY_TRACKER = UTCGP.IndividualLossTrackerMT(pop_size, TrainSize * n_repetitions)
#         # if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
#         #     early_stop_args = UTCGP.GA_EARLYSTOP_ARGS(
#         #         M_gen_loss_tracker,
#         #         EMPTY_TRACKER,
#         #         ind_performances,
#         #         population,
#         #         iteration,
#         #         run_config,
#         #         model_architecture,
#         #         node_config,
#         #         meta_library,
#         #         shared_inputs,
#         #         population_programs,
#         #         elite_fitnesses,
#         #         best_programs,
#         #         elite_idx,
#         #     )
#         #     early_stop =
#         #         UTCGP._make_ga_early_stop_callbacks_calls(early_stop_args, early_stop_callbacks) # true if any
#         # end

#         # if early_stop
#         #     g = run_config.generations
#         #     @warn "Early returning at iteration : $iteration from $g total iterations"
#         #     if !isnothing(last_callback)
#         #         last_callback(
#         #             ind_performances,
#         #             population,
#         #             iteration,
#         #             run_config,
#         #             model_architecture,
#         #             node_config,
#         #             meta_library,
#         #             population_programs,
#         #             elite_fitnesses,
#         #             best_programs,
#         #             elite_idx,
#         #         )
#         #     end
#         #     # UTCGP.show_program(program)
#         #     return tuple(genome, best_programs, M_gen_loss_tracker)
#         # end
#         @info "Number of nodes in the graph : $(labels(graph.g) |> length)"
#         gct = @elapsed GC.gc(true)
#         @warn "Running GC at the end of iteration. GC time : $gct"

#     end
#     return (genome, best_programs, M_gen_loss_tracker)
# end
