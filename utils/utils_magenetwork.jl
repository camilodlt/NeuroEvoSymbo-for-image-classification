using MAGENetwork: MAGEProgramInstance
using UTCGP: _eval_batch_on_pop
import Pkg
import UUIDs
using FileIO
using Base.Threads
using UTCGP
import PNGFiles
using Images
using Flux: sigmoid
using StatsBase: sample, Weights
using UnicodePlots
using ThreadPools
using ErrorTypes
using DataFlowTasks
using MAGENetwork
using Logging

function reset_random_weights_model!(model::MNModel)
    for layer in model.mnsequence.mnlayers
        for mage_and_surrogate in layer.programs
            if mage_and_surrogate.surrogate isa MAGENetwork.NNSurrogateModel
                mage_and_surrogate.surrogate.model = MAGENetwork.reinit_model(mage_and_surrogate.surrogate.model)
                @info "Reset"
            end
        end
    end
    return
end

function create_layers(trial, initial_img_type, extra_types, lib_mixed, lib_float, n_inputs, n_nodes = 4, n_outputs = 2)
    global NODESLAST, L1_OUT_SIZE
    n_layers = trial["n_layers"]
    n_layers_img = trial["n_layers_img"]
    if n_layers_img >= n_layers
        n_layers_img = n_layers - 1
    end

    # HANDLE OUTS
    n_outs = [trial["l$(i)_out_size"] for i in 1:n_layers]

    @info "Layers : $n_layers"
    @info "Layers img : $n_layers_img"

    layers = []
    current_output_size = nothing

    for layer_idx in 1:n_layers
        is_last_layer = layer_idx == n_layers
        # Determine layer type based on position
        if layer_idx <= n_layers_img
            layer_type = layer_idx == 1 ? :mixed : :image
        else
            layer_type = :float
        end

        # Suggest output size for non-final layers
        if layer_idx < n_layers
            n_progs = trial["l$(layer_idx)_size"]
            output_size = n_progs * n_outs[layer_idx] # TODO also index trial
        else
            n_progs = n_outputs
            output_size = n_outputs  # Final layer has fixed output size
        end

        @info "Layer : $layer_idx of type : $layer_type. Prev out size $current_output_size"

        # OUTS
        current_output_dim = n_outs[layer_idx]
        if layer_idx < n_layers_img # TODO does not handle all cases
            out_types = [initial_img_type for i in 1:current_output_dim]
            out_indices = [1 for i in 1:current_output_dim]
        elseif layer_idx == n_layers_img # we have to ret floats
            out_types = [Float64 for i in 1:current_output_dim]
            out_indices = [4 for i in 1:current_output_dim]
        elseif layer_idx == n_layers
            out_types = [Float64]
            out_indices = [1]
        else
            # it can return img
            out_types = [Float64 for i in 1:current_output_dim]
            out_indices = [1 for i in 1:current_output_dim]
        end

        #Create model architecture based on layer type
        ma, lib = if layer_type == :mixed
            # Mixed input (images + floats)
            modelArchitecture( # TODO
                    [[initial_img_type for _ in 1:n_inputs]...], # [[initial_img_type for _ in 1:7]..., [Float64 for _ in 1:7]...],
                    [[1 for _ in 1:n_inputs]...], # [[1 for _ in 1:7]..., [2 for _ in 1:7]...],
                    [initial_img_type, extra_types..., Float64],
                    out_types,
                    out_indices
                ), lib_mixed
        elseif layer_type == :image
            # Image-to-image processing
            modelArchitecture(
                    [initial_img_type for _ in 1:current_output_size],
                    [1 for _ in 1:current_output_size],
                    [initial_img_type, extra_types..., Float64],
                    out_types,
                    out_indices
                ), lib_mixed
        else
            @info "Float with ins: $(current_output_size). Outs : 1"
            # Float processing
            modelArchitecture(
                    [Float64 for _ in 1:current_output_size],
                    [1 for _ in 1:current_output_size],
                    [Float64],
                    out_types,
                    out_indices
                ), lib_float
        end

        current_output_size = output_size
        pnc = prenodeConfig(n_nodes, 3)
        if is_last_layer
            @info "Using $(NODESLAST) nodes for the last module"
            pnc = prenodeConfig(NODESLAST, 3)
        end
        layer = MNLayer(n_progs, ma, lib, pnc)
        push!(layers, layer)
    end

    return MNModel(MNSequence(layers...))
end

###################################################
# EPOCH CALLBACK ##################################
###################################################

"""
"""
struct MNGA_EPOCHCALLBACK_ARGS <: UTCGP.Abstract_GA_POP_ARGS
    population::MAGENetwork.AbstractMNPopulation
    generation::Int
    run_config::UTCGP.AbstractRunConf
    fitnesses::Vector{Float64}
    elite_idx::Vector{Int}
    extras::Dict

    function MNGA_EPOCHCALLBACK_ARGS(
            population::MAGENetwork.AbstractMNPopulation,
            generation::Int,
            run_config::UTCGP.AbstractRunConf,
            fitnesses::Vector{Float64},
            elite_idx::Vector{Int},
            extras::Dict = Dict(),
        )
        return new(
            population,
            generation,
            run_config,
            fitnesses,
            elite_idx,
            extras
        )
    end
end

MAGENetwork._T_args_has_mnpop(::MNGA_EPOCHCALLBACK_ARGS) = MAGENetwork.HAS_MNPOP

function call_epoch_callbacks!(args::MNGA_EPOCHCALLBACK_ARGS, callbacks, moreargs...)
    t = []
    for callback in callbacks
        t_e = @elapsed callback(args)
        push!(t, t_e)
    end
    tt = mean(t)
    return @info "Epoch Callbacks done, Time $tt"
end

###################################################
# FIT  ############################################
###################################################

# function loss_func(output, labels)
#     # Convert the integer label to one-hot encoding
#     # Assuming labels are 1-indexed (1 or 2)
#     one_hot_gt = zeros(Float32, 2)
#     one_hot_gt[labels] = 1.0

#     # Compute cross-entropy on raw logits
#     return Flux.logitcrossentropy(output, one_hot_gt)
# end

function (pd::PlateauDetector)(args::MNGA_EPOCHCALLBACK_ARGS)
    best_fitness = minimum(args.fitnesses)
    push!(pd.history, best_fitness)

    # Check if we have enough history
    if length(pd.history) < pd.patience + 1
        pd.detected = false
        return
    end

    # Check for improvement over patience window
    # @show pd.history
    recent_history = pd.history[(end - pd.patience + 1):end]
    # @show recent_history
    best_recent = minimum(recent_history)
    # @show best_recent
    best_prior = pd.history[end - pd.patience]
    # @show best_prior

    # Check if improvement is below threshold
    improvement = best_prior - best_recent
    # @show improvement
    pd.detected = improvement < pd.min_improvement

    return if pd.detected
        println("Plateau detected! Best fitness: $best_fitness compared to patience window : $recent_history")
    end
end

function ml_from_vbundles(vector_of_bundles::Vector{<:Vector{<:UTCGP.FunctionBundle}})
    ml = MetaLibrary(map(x -> Library(x), vector_of_bundles))
    return ml
end

function create_initial_magenet_population(
        backend::MAGENetwork.AbstractBackend,
        n_elite, Parsed_args, Type2Dimg_intensity, type2dimg_tuple, ml, ml_float, valx, N_NODES, NCLASSES;
        type_of_module::Symbol = :small,
        use_surrogate::Bool = true
    )
    @info "Using this type of modules : $type_of_module"
    initial_pop = MNModel[]
    for i in 1:n_elite
        mn_model = create_layers(Parsed_args, Type2Dimg_intensity, type2dimg_tuple, ml, ml_float, length(valx[1]), N_NODES, NCLASSES)
        init_mn!(mn_model, backend; size = type_of_module, use_surrogate = use_surrogate)
        push!(initial_pop, mn_model)
    end
    initial_pop = MNPopulation(initial_pop)
    for ind in initial_pop
        @info ind
    end
    return initial_pop
end

function setup_output_folder_and_checkpointer(outputdir, id, seed, rootdir)
    folder = joinpath(outputdir, id, string(seed))
    isdir(folder) || mkpath(folder)
    checkpointer = checkpoint(1, folder)
    json_path = joinpath(rootdir, folder, id) * ".json"
    f = open(json_path, "w", lock = true)
    return folder, checkpointer, f
end

function setup_metric_trackers(parsed_args, f, allx, ally, checkpointer)
    metric_tracker = UTCGP.jsonTracker(parsed_args, f)
    train_tracker = jsonTrackerGA(
        metric_tracker,
        acc_callback([], [], "train"), "Train", nothing, nothing, 0.0, nothing
    )
    val_tracker = jsonTrackerGA(
        metric_tracker,
        acc_callback(allx, ally, "Val"), "Val", [], nothing, 0.0, checkpointer
    )
    return metric_tracker, train_tracker, val_tracker
end

function setup_surrogate_explorer(epochs_surr, k, device, plateau_patience)
    plateau_detector = PlateauDetector(plateau_patience)
    surrogate_explorer = SurrogateExplorer(
        plateau_detector;
        surrogate_epochs = epochs_surr, strategy = :loop, k = k, device = device, variants_per_elite = 1000
    )
    return plateau_detector, surrogate_explorer
end

function fit_rf_head(args::NamedTuple)
    population = args.population
    n = length(population)
    n_features = population[1][1] |> length
    trainx, trainy = args.train_data.xs, args.train_data.ys
    valx, valy = args.val_data.xs, args.val_data.ys
    n_train = length(trainx)
    n_val = length(valx)
    MAGENetwork._reset_mnmodel!.(population)

    nt = nthreads()
    inds_per_thread = ceil(Int, n / nt)
    partition_inds_per_thread = Iterators.partition(1:n, inds_per_thread)
    fitness_per_ind = Vector{Float64}(undef, n) # where to store the val bacc per ind
    preds_train_per_ind = Vector{Matrix{Float64}}(undef, n)
    preds_val_per_ind = Vector{Matrix{Float64}}(undef, n)

    tasks = []
    for inds_idx in partition_inds_per_thread
        t = Base.Threads.@spawn begin
            ind_features_train = Matrix{Float64}(undef, n_train, n_features)
            ind_features_val = Matrix{Float64}(undef, n_val, n_features)
            for ind_idx in inds_idx
                ind = population[ind_idx]
                for (i, (x, y)) in enumerate(zip(trainx, trainy)) # run ind in train data
                    MAGENetwork._reset_mnmodel!(ind)
                    outputs = forward(ind, x, true)
                    ind_features_train[i, :] .= outputs
                end
                for (i, (x, y)) in enumerate(zip(valx, valy)) # run ind on val data
                    MAGENetwork._reset_mnmodel!(ind)
                    outputs = forward(ind, x, true)
                    ind_features_val[i, :] .= outputs
                end
                preds_train_per_ind[ind_idx] = ind_features_train
                preds_val_per_ind[ind_idx] = ind_features_val
            end
        end
        push!(tasks, t)
    end
    fetch.(tasks)

    RandomForestClassifierSKLEARN = MLJ.@load RandomForestClassifier pkg = MLJScikitLearnInterface

    for ind_idx in 1:n
        train_mat = preds_train_per_ind[ind_idx]
        val_mat = preds_val_per_ind[ind_idx]
        train_mat = DataFrame(train_mat, :auto)
        val_mat = DataFrame(val_mat, :auto)
        MLJ.coerce!(train_mat, MLJ.Continuous => MLJ.Continuous)
        MLJ.coerce!(train_mat, MLJ.Count => MLJ.Continuous)
        MLJ.coerce!(val_mat, MLJ.Continuous => MLJ.Continuous)
        MLJ.coerce!(val_mat, MLJ.Count => MLJ.Continuous)
        # Train the RF
        model = RandomForestClassifierSKLEARN(
            n_estimators = 30, max_depth = 10, class_weight = "balanced",
            min_samples_split = 10, max_samples = 0.8, n_jobs = -1,
            random_state = 123
        )
        mach = MLJ.machine(model, train_mat, MLJ.categorical(trainy)) |> MLJ.fit!
        # Predict on train
        # y_hat = predict_mode(mach, train_mat)
        # train_bacc = StatisticalMeasures.balanced_accuracy(y_hat, trainy)
        # Predict on val
        y_hat = MLJ.predict_mode(mach, val_mat)
        val_bacc = StatisticalMeasures.balanced_accuracy(y_hat, valy)
        fitness_per_ind[ind_idx] = -val_bacc
    end
    return fitness_per_ind

end


function fit_ga_network_batch(
        trainxy::Any, # DATALOADER
        valxy::Any, # DATALOADER
        population_MNModel::MNPopulation,
        rc::MNRunConf,
        head_callback::UTCGP.Optional_FN,
        population_callbacks::UTCGP.Mandatory_FN,
        mutation_callbacks::UTCGP.Mandatory_FN,
        decoding_callbacks::UTCGP.Mandatory_FN,
        tracker_callback,
        early_stop_callbacks::UTCGP.Optional_FN,
    )
    early_stop = false
    best_programs = nothing
    elite_idx = nothing
    ind_performances = [1.0 for i in 1:rc.n_elite]
    full_population = deepcopy(population_MNModel)

    # PRE CALLBACKS
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    for iteration in 1:rc.gens
        start_time = time()
        early_stop ? break : nothing
        @warn "Iteration : $iteration"
        # Population
        ga_pop_args = MNGA_POP_ARGS(
            full_population,
            iteration,
            rc,
            ind_performances,
        )
        population, time_pop = MAGENetwork._call_mn_population_callbacks!(ga_pop_args, population_callbacks)

        # Program mutations ---
        ga_mutation_args = MNGA_MUTATION_ARGS(
            population,
            iteration,
            rc,
            ind_performances
        )
        population, time_mut = with_logger(NullLogger()) do
            MAGENetwork._call_mn_mutation_callbacks!(ga_mutation_args, mutation_callbacks)
        end


        # Genotype to Phenotype mapping ---
        decoding_args = MNGA_DECODING_ARGS(
            population, iteration, rc, ind_performances
        )
        _, time_pop_prog = MAGENetwork._call_mn_decoding_callbacks!(decoding_args, decoding_callbacks)

        ind_performances = head_callback[1](
            (
                population = population,
                train_data = trainxy,
                val_data = valxy,
            )
        )
        @show minimum(ind_performances) argmin(ind_performances)

        # Selection
        elite_idx = sortperm(ind_performances)[1:rc.n_elite]
        @show elite_idx
        elite_fitnesses = ind_performances[elite_idx]
        elite_best_fitness = minimum(skipmissing(elite_fitnesses))
        elite_best_ftiness_idx = argmin(elite_fitnesses)
        elite_avg_fitness = mean(skipmissing(elite_fitnesses))
        elite_std_fitness = std(filter(!isnan, ind_performances))
        # genome = deepcopy(population[elite_idx])
        try
            histogram(ind_performances) |> println
        catch e
            @error "Could not drawn histogram"
        end

        elite_models = [population.pop[elite_idx]...]
        # full_population = [population.pop[elite_idx]...]
        # ind_performances = ind_performances[elite_idx]
        full_population = population # so that tournament selection is ok. First pop is truncated then TS

        try
            histogram(elite_fitnesses) |> println
        catch e
            @error "Could not drawn histogram"
        end

        @show length(full_population.pop)

        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, elite_best_fitness)
        println(
            "Iteration $iteration. 
            Best fitness: $(round(elite_best_fitness, digits = 10)) at index $elite_best_ftiness_idx 
            Elite mean fitness : $(round(elite_avg_fitness, digits = 10)). Std: $(round(elite_std_fitness)) at indices : $(elite_idx)",
        )

        if !isnothing(tracker_callback)
            tracker_callback(
                (
                    iteration = iteration,
                    ind_performances = ind_performances,
                    best_ind = elite_models[1],
                )
            )
        end

        # EARLY STOP
        if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
            early_stop = early_stop_callbacks[1]()
        end

        if early_stop
            @warn "Early returning at iteration : $iteration"
            return full_population, M_gen_loss_tracker
        end

    end
    return full_population, M_gen_loss_tracker
end

function (se::SurrogateExplorer)(args::MNGA_EPOCHCALLBACK_ARGS)
    global STO_WARMUP, N_CLASSES
    # Update plateau detector
    se.plateau_detector(args)
    generation = args.generation
    if generation < STO_WARMUP
        return
    end
    endpoint = args.extras["endpoint"]
    only_first = args.extras["only_first"]
    n_nn_batches_train = args.extras["n_nn_batches_train"]
    n_nn_batches_val = args.extras["n_nn_batches_val"]
    val_tracker = args.extras["val_tracker"]
    transformations = args.extras["transformations"]

    best_ind_so_far = val_tracker.best_ind

    # ARGS FOR FITTING MAGE TO NN
    generations_mage = args.extras["generations_mage"]
    lambda_mage = args.extras["lambda_mage"]
    trainsize_mage = args.extras["trainsize_mage"]

    if only_first
        @warn "USING BEST VAL INDIVIDUAL"
        first = best_ind_so_far
        MAGENetwork.set_parent!(first, first)
        empty!(args.population.pop)
        push!(args.population.pop, first)
    end

    @info objectid.(args.population.pop)

    @show se.plateau_detector.detected
    @show se.is_exploring

    # if se.plateau_detector.detected # && !se.is_exploring
    se.is_exploring = true

    # Get batch data
    data = args.extras["all_data"]
    val_data = args.extras["val_data"]
    @show args.elite_idx

    # PHASE 1: Sequential Training for all elite individuals
    # ----------------------------------------------------
    @info "Starting sequential training phase for all elite individuals"

    # Sample data for training
    # examples = get_augmented_data(data, 1:(MAGENetwork.BS[] * n_nn_batches_train)) #data is dataloder so already samples from the whole dataset
    examples = data[1:(MAGENetwork.BS[] * n_nn_batches_train)] #data is dataloder so already samples from the whole dataset
    # shuffle!(examples)
    inputs_nn = [x[1] for x in examples]
    labels_nn = [x[2][1] for x in examples]
    println(histogram(labels_nn, nbins = length(unique(labels_nn))))
    program_data_nn = MAGENetwork.MAGEProgramInstance[]
    for (x, y) in zip(inputs_nn, labels_nn)
        push!(program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
    end

    # val_examples = get_augmented_data(val_data, 1:(MAGENetwork.BS[] * n_nn_batches_val))
    val_examples = val_data[1:(MAGENetwork.BS[] * n_nn_batches_val)] #data is dataloder so already samples from the whole dataset
    # shuffle!(val_examples)
    val_inputs_nn = [x[1] for x in val_examples]
    val_labels_nn = [x[2][1] for x in val_examples]
    println(histogram(val_labels_nn, nbins = length(unique(val_labels_nn))))
    val_program_data_nn = MAGENetwork.MAGEProgramInstance[]
    for (x, y) in zip(val_inputs_nn, val_labels_nn)
        push!(val_program_data_nn, MAGENetwork.MAGEProgramInstance(x, y))
    end

    for elite_model in args.population
        @show objectid(elite_model)
        MAGENetwork.turn_on!(elite_model)
        # Collect training data and perform sequential training
        training_data = MAGENetwork.collect_training_data(elite_model, inputs_nn)
        val_data = MAGENetwork.collect_training_data(elite_model, val_inputs_nn)
        MAGENetwork.sequential_training!(
            elite_model, training_data, val_data;
            epochs = se.surrogate_epochs, device = se.device, train = generation != 1
        )
    end

    @info "ALL MODELS TRAINED"
    # Extract elite population
    elite_population = args.population # population is already elite [args.population[idx] for idx in args.elite_idx]
    @info "Length of population in SE: $(length(elite_population))"

    # Apply selected strategy
    local new_elite_population

    @info objectid.(elite_population)
    @info objectid(elite_population[1])

    if se.strategy == :joint
        # Strategy 1: Joint training
        new_elite_population, nextgen = MAGENetwork.strategy_one(
            elite_population,
            program_data_nn, labels_nn,
            program_data_mage, labels_mage,
            endpoint;
            k = se.k,
            variants_per_elite = se.variants_per_elite,
            epochs = se.surrogate_epochs,
            device = se.device,
        )
        # Replace elite models in population
        mn_pop = args.extras["mn_pop"]
        mn_pop_fitnesses = args.extras["mn_pop_fitnesses"]
        # @assert length(mn_pop) == length(new_elite_population)
        empty!(mn_pop)
        empty!(mn_pop_fitnesses)
        for new in nextgen.full_new_pop
            insert!(mn_pop, 1, new)
            insert!(mn_pop_fitnesses, 1, 0.0)
        end
        @show size(mn_pop) size(mn_pop_fitnesses)
        # for i in 1:length(mn_pop)
        #     @info "Replacing individual $i"
        #     mn_pop.pop[i] = new_elite_population[i]
        # end
    elseif se.strategy == :sequential
        # Strategy 2: Sequential freezing
        new_elite_population = MAGENetwork.strategy_two(
            elite_population, program_data_nn, labels_nn;
            k = se.k,
            variants_per_elite = se.variants_per_elite,
            epochs = se.surrogate_epochs,
            device = se.device
        )
    elseif se.strategy == :loop
        # Strategy 2: Sequential freezing
        train_align_data = [MAGENetwork.collect_training_data(elite, inputs_nn) for elite in elite_population]
        val_align_data = [MAGENetwork.collect_training_data(elite, val_inputs_nn) for elite in elite_population]
        @show length(program_data_nn)
        @show length(train_align_data)
        @show length(val_align_data)
        next_gen = MAGENetwork.strategy_three(
            elite_population,
            program_data_nn, val_program_data_nn, # GT instances
            train_align_data, val_align_data, # Intermediary Outputs for alignment
            val_program_data_nn;
            k = se.k,
            device = se.device,
            epochs = generations_mage,
            trainsize_mage = trainsize_mage,
            lambda_mage = lambda_mage,
            n_classes = N_CLASSES,
            transformations = transformations
        )
        mn_pop = args.extras["mn_pop"]
        mn_pop_fitnesses = args.extras["mn_pop_fitnesses"]
        # @assert length(mn_pop) == length(new_elite_population)
        empty!(mn_pop)
        push!(mn_pop, next_gen.full_new_pop...)
        empty!(mn_pop_fitnesses)
        push!(mn_pop_fitnesses, next_gen.full_new_pop_fitnesses...)
        @show size(mn_pop) size(mn_pop_fitnesses)

        @info "Out from Surrogate Trainer: $(MAGENetwork.get_model_uuid.(mn_pop)). Fitnesses : $(mn_pop_fitnesses).(OLD, NEW)"
    else
        @error "Unknown strategy: $(se.strategy)"
    end

    # Reset exploration state
    se.is_exploring = false
    se.plateau_detector.detected = false
    # end
    return nothing
end


function (a::acc_callback)(
        args::MNGA_EPOCHCALLBACK_ARGS
    )
    return a(MAGENetwork._T_args_has_mnpop(args), args)
end

function (a::acc_callback)(
        ::Type{MAGENetwork.HAS_MNPOP},
        args::MNGA_EPOCHCALLBACK_ARGS
    )
    """
    Use either passed batch of the data in the struct
    """
    function _get_xs_ys(batch, holder_struct, idx, use_batch::Bool)
        if use_batch
            subBatch = batch[idx]
            xs = [i[1] for i in subBatch]
            ys = [i[2] for i in subBatch]
            return (xs, ys)
        else
            xs, ys = a.X_test[idx], a.Y_test[idx]
            return (xs, ys)
        end
    end

    """
    Takes the program at index argmin of the losses.
    """
    function get_best_ind(best_models, best_losses)
        MAGENetwork._reset_mnmodel!.(best_models)
        best_model = best_models[1]
        @assert argmin(best_losses) == 1
        return best_model
    end

    global N_CLASSES
    batch = args.extras["ALL_BATCHS"]
    nt = Threads.nthreads()
    use_batch = length(a.X_test) == 0
    use_batch ? n = length(batch) : n = length(a.X_test)
    best_model = get_best_ind(args.population, args.fitnesses)
    M_ind_vs_samples = Vector{IndVsSample}(undef, n)
    indices = collect(1:n)
    BATCH_SIZE = ceil(Int, n / nt)
    @info "Running acc_callback"
    for ith_x in Iterators.partition(indices, BATCH_SIZE)
        xs, ys = _get_xs_ys(batch, a, ith_x, use_batch)
        slot_for_metrics = @view M_ind_vs_samples[ith_x]
        let batch_x = xs
            batch_y = ys
            res = @dspawn begin
                @W slot_for_metrics
                t = evaluate_batch_single_individual!(
                    batch_x, batch_y, slot_for_metrics,
                    deepcopy(best_model)
                )
            end
        end
    end
    final_task = @dspawn @R(view(M_ind_vs_samples, :)) label = "result"
    fetch(final_task)

    # Calculate metrics
    metrics = calculate_metrics_ind_vs_samples(
        M_ind_vs_samples,
        N_CLASSES,
        Dict(),
    )
    @show metrics
    return metrics
end

function evaluate_batch_single_individual!(
        xs::Vector,
        ys::Vector,
        metrics_slot::SubArray,
        model::MNModel,
    )
    @debug "Started eval at Thread $(Threads.threadid())"
    @assert length(metrics_slot) == length(xs) == length(ys) "Incorrect length for Thread slot"
    for (sample_idx, (x, y)) in enumerate(zip(xs, ys))
        MAGENetwork._reset_mnmodel!(model)
        time = @elapsed outs = forward(model, x)
        outs = identity.(outs)
        gt = y[1]
        metrics_slot[sample_idx] = transform_preds_to_IndVsSample(outs, gt, time)
    end
    return @debug "Ended eval at Thread $(Threads.threadid())"
end

"""
Track Metrics for MAGENET
"""
function (jtga::jsonTrackerGA)(
        args::MNGA_EPOCHCALLBACK_ARGS
    )
    return if !isnothing(jtga.test_losses)
        pop_passed_filter = args.population
        for idx in 1:length(pop_passed_filter)
            ind_to_be_tested = pop_passed_filter[idx]
            metrics = jtga.acc_callback(
                MNGA_EPOCHCALLBACK_ARGS(
                    MNPopulation([ind_to_be_tested]),
                    args.generation,
                    args.run_config,
                    [1.0], [1], args.extras
                )
            )
            @warn "JTT $(jtga.label) Balanced Accuracy: $(metrics[:bacc])"
            balanced_acc = metrics[:bacc]
            payload = Dict{Any, Any}(metrics...)
            payload[:data] = jtga.label
            payload[:iteration] = args.generation
            ind = deepcopy(ind_to_be_tested)
            jtga.checkpoint(args.generation, ind) # the val tracker writes checkpoint
            push!(jtga.test_losses, balanced_acc)
            if balanced_acc >= jtga.best_loss
                # we have a new best
                jtga.best_ind = ind
            end
            jtga.best_loss = maximum(jtga.test_losses)
            best_metric = jtga.best_loss
            MAGENetwork.REQ[] = max(0.5, best_metric + 1.0)
            @show MAGENetwork.REQ[]
            write(jtga.tracker.file, JSON.json(payload), "\n")
            flush(jtga.tracker.file)
        end
        @info "Best VAL BACC :$(jtga.best_loss)"
    else
        metrics = jtga.acc_callback(args)
        @warn "JTT $(jtga.label) Balanced Accuracy: $(metrics[:bacc])"
    end
end


#############################
# TRANSFORMATIONS           #
#############################

# flipper = FlipX{2}()
# change_brightness = AdjustBrightness(0.2)
# change_contrast = AdjustContrast(0.2)
# function apply_transformation(x::T, f) where {T <: SizedImage{SIZE, PT}} where {SIZE, PT}
#     IT = UTCGP._get_image_type(T)
#     x_ = itemdata(apply(f, Image(reinterpret(x.img))))
#     s = size(x_)
#     x_matrix = reshape(x_, s)
#     if size(x) !== size(x_)
#         x_matrix = imresize(x_matrix, size(x))
#     end
#     return SImageND(PT.(UTCGP.image2D_morph.cast(IT, x_matrix)), SIZE)
# end
# function apply_fn(x::T, f) where {T <: SizedImage{SIZE, PT}} where {SIZE, PT}
#     IT = UTCGP._get_image_type(T)
#     x_ = f(reinterpret(x.img))
#     if size(x) !== size(x_)
#         x_ = imresize(x_, size(x))
#     end
#     return SImageND(PT.(UTCGP.image2D_morph.cast(IT, x_)), SIZE)
# end

# tr_flip(x, args...) = apply_transformation(x, flipper)
# tr_crop(x, args...) = apply_transformation(x, CenterCrop(size(x) .- 5))
# tr_brightness(x, args...) = apply_transformation(x, change_brightness)
# tr_contrast(x, args...) = apply_transformation(x, change_contrast)
# rotate1_(x, args...) = reshape(ImageTransformations.imrotate(x, π / 2), size(x))
# rotate2_(x, args...) = reshape(ImageTransformations.imrotate(x, π), size(x))
# tr_rotate1(x, args...) = apply_fn(x, rotate1_)
# tr_rotate2(x, args...) = apply_fn(x, rotate2_)

# function normalize(x, img_idx, args...)
#     m = if img_idx == 1
#         0.49118972f0
#     elseif img_idx == 2
#         0.48195773f0
#     else
#         0.44618067f0
#     end
#     s = if img_idx == 1
#         0.24600388f0
#     elseif img_idx == 2
#         0.24112508f0
#     else
#         0.25764066f0
#     end

#     return SImageND(IntensityPixel{Float32}.((x .- m) ./ s))
# end

# transformations = Function[
#     normalize,
#     tr_flip,
#     tr_crop,
#     tr_brightness,
#     tr_contrast,
#     # tr_rotate1,
#     # tr_rotate2,
# ]
#
