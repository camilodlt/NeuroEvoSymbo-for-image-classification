function ml_from_vbundles(vector_of_bundles::Vector{<:Vector{<:UTCGP.FunctionBundle}})
    ml = MetaLibrary(map(x -> Library(x), vector_of_bundles))
    return ml
end

struct es <: UTCGP.AbstractCallable
    init_time
end

function (e::es)(args...)
    global HLIMIT
    new_time = time()
    elapsed_s = (new_time - e.init_time)
    elapsed_m = elapsed_s / 60
    elapsed_h = elapsed_m / 60
    @info "TIME ELAPSED $elapsed_h"
    if elapsed_h > HLIMIT #|| elapsed_h + t > HLIMIT
        @info "Early Stopping because of time limit. Elapsed : $elapsed_h. Limit $(HLIMIT)"
        # @info "Early Stopping because of time limit. Elapsed : $elapsed_h. Time gen :$t. Limit $(HLIMIT)"
        return true
    else
        @info "Time passed since start : $elapsed_m (minutes)"
        return false
    end
end

function fix_all_output_nodes!(ut_genome::UTGenome)
    for (ith_out_node, output_node) in enumerate(ut_genome.output_nodes)
        to_node = output_node[2].highest_bound + 1 - ith_out_node
        set_node_element_value!(
            output_node[2],
            to_node
        )
        set_node_freeze_state(output_node[2])
        set_node_freeze_state(output_node[1])
        set_node_freeze_state(output_node[3])
        println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        println("Output Node material : $(node_to_vector(output_node))")
    end
    return
end
