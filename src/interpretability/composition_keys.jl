"""
Utilities to extract single-function and level-1 composition keys from decoded UTCGP programs.

Unigram key per node:
- `caller`

Canonical composition key per node:
- `libK::caller` when the node has no upstream function dependencies
- `libK::caller(dep1, dep2, ...)` when some arguments come from upstream CGP nodes

Direct program inputs are ignored in dependency lists. Function identity in composition
space is library-qualified, so the same caller name in two different libraries is treated
as different.
"""

function _operation_body(program)
    return [op for op in program if !(op.calling_node isa UTCGP.OutputNode)]
end

function _unwrap_operation_input_node(program, op_input)
    maybe_node = UTCGP._extract_input_node_from_operationInput(program.program_inputs, op_input)
    try
        return unwrap(maybe_node)
    catch
        return nothing
    end
end

function _push_step_key!(dict::Dict{Int, Vector{String}}, lib::Int, key::String)
    push!(get!(dict, lib, String[]), key)
    return nothing
end

_libqual(lib::Int, caller::AbstractString) = "lib$(lib)::$(caller)"

function _unique_dict_values(dict::Dict{Int, Vector{String}})
    out = Dict{Int, Vector{String}}()
    for (k, v) in dict
        out[k] = unique(v)
    end
    return out
end

"""
    extract_program_keys(program)

Extracts keys/stats for one decoded UTCGP program.

Returns a NamedTuple with:
- `single_steps_by_lib::Dict{Int,Vector{String}}`
- `composition_steps_by_lib::Dict{Int,Vector{String}}`
- `all_steps_by_lib::Dict{Int,Vector{String}}`
- `single_unique_by_lib::Dict{Int,Vector{String}}`
- `composition_unique_by_lib::Dict{Int,Vector{String}}`
- `all_unique_by_lib::Dict{Int,Vector{String}}`
- `n_steps::Int`
- `n_fn_by_lib::Dict{Int,Int}`
"""
function extract_program_keys(program)
    ops = _operation_body(program)

    single_steps_by_lib = Dict{Int, Vector{String}}()
    composition_steps_by_lib = Dict{Int, Vector{String}}()
    all_steps_by_lib = Dict{Int, Vector{String}}()
    n_fn_by_lib = Dict{Int, Int}()
    seen_node_ids = Set{String}()

    # Maps CGP node ids to library-qualified caller names for immediate dependency resolution.
    nodeid_to_caller = Dict{String, String}()

    for op in ops
        node_id = String(op.calling_node.id)
        if node_id in seen_node_ids
            continue
        end
        push!(seen_node_ids, node_id)

        caller = String(op.fn.name)
        lib = Int(op.calling_node.y_position)
        qualified_caller = _libqual(lib, caller)

        _push_step_key!(single_steps_by_lib, lib, caller)
        n_fn_by_lib[lib] = get(n_fn_by_lib, lib, 0) + 1

        deps = String[]
        for op_input in op.inputs
            input_node = _unwrap_operation_input_node(program, op_input)
            if isnothing(input_node)
                continue
            end
            if input_node isa UTCGP.CGPNode
                dep = get(nodeid_to_caller, String(input_node.id), nothing)
                !isnothing(dep) && push!(deps, dep)
            end
        end

        comp_key = isempty(deps) ? qualified_caller : string(qualified_caller, "(", join(deps, ", "), ")")
        _push_step_key!(composition_steps_by_lib, lib, comp_key)
        _push_step_key!(all_steps_by_lib, lib, comp_key)

        nodeid_to_caller[node_id] = qualified_caller
    end

    single_unique_by_lib = _unique_dict_values(single_steps_by_lib)
    composition_unique_by_lib = _unique_dict_values(composition_steps_by_lib)
    all_unique_by_lib = _unique_dict_values(all_steps_by_lib)

    return (
        single_steps_by_lib = single_steps_by_lib,
        composition_steps_by_lib = composition_steps_by_lib,
        all_steps_by_lib = all_steps_by_lib,
        single_unique_by_lib = single_unique_by_lib,
        composition_unique_by_lib = composition_unique_by_lib,
        all_unique_by_lib = all_unique_by_lib,
        n_steps = length(seen_node_ids),
        n_fn_by_lib = n_fn_by_lib,
    )
end
