using DataFrames
using PythonCall

"""Fit one sklearn StandardScaler on train only, then apply it to train, validation, and test."""
function maybe_standard_scale_prediction_matrices(train_mat, val_mat, test_mat; normalize::Bool)
    if !normalize
        @info "Skipping StandardScaler normalization"
        return train_mat, val_mat, test_mat
    end

    @info "Applying sklearn StandardScaler normalization fitted on train only"
    sklearn_preprocessing = pyimport("sklearn.preprocessing")
    scaler = sklearn_preprocessing.StandardScaler()

    train_names = names(train_mat)
    val_names = names(val_mat)
    test_names = isnothing(test_mat) ? nothing : names(test_mat)

    train_array = Matrix{Float64}(train_mat)
    val_array = Matrix{Float64}(val_mat)
    test_array = isnothing(test_mat) ? nothing : Matrix{Float64}(test_mat)

    scaler.fit(train_array)

    train_scaled = pyconvert(Matrix{Float64}, scaler.transform(train_array))
    val_scaled = pyconvert(Matrix{Float64}, scaler.transform(val_array))
    test_scaled = isnothing(test_array) ? nothing : pyconvert(Matrix{Float64}, scaler.transform(test_array))

    train_scaled_df = DataFrame(train_scaled, train_names; copycols = false)
    val_scaled_df = DataFrame(val_scaled, val_names; copycols = false)
    test_scaled_df = isnothing(test_scaled) ? nothing : DataFrame(test_scaled, test_names; copycols = false)

    @info "StandardScaler normalization complete" train_size = size(train_scaled_df) val_size = size(val_scaled_df) test_size = isnothing(test_scaled_df) ? nothing : size(test_scaled_df)
    return train_scaled_df, val_scaled_df, test_scaled_df
end
