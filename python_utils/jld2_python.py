from juliacall import Main as jl
jl.seval("using JLD2")

def read_jld2_dataset(path):
    print(f"Reading {path}")
    data = jl.load(path)
    xs = data["single_stored_object"].xs
    ys = data["single_stored_object"].ys
    extras = data["single_stored_object"].extras
    return xs, ys, extras

# Example usage
# from jld2_python import *
# path = "../datasets_pickle/pcam_rgb_train.jld2"
# xs, ys, extras = read_jld2_dataset(path)
