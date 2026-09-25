using PoissonRandom
using OrderedCollections, FileIO
using Random, Distributions, HDF5

# --- GAUSSIAN NOISE
function add_noise(imgs::Vector, dist, rng)
    return add_noise.(imgs, Ref(dist), Ref(rng))
end
function add_noise(channel::SImageND, dist, rng)
    noise = rand(rng, dist, size(channel))
    return round.( # closest int
        UInt8,
        clamp.(
            Float64.( # float
                reinterpret.( # UInt8
                    reinterpret(channel.img) # N0f8
                )
            ) + noise
            , 0.0, 255.0
        )
    )
end

# --- POISSON
function sample_poisson_noise(x::Float64, photons, normalized, rng)
    n = normalized + eps(Float64)
    lambda = (x / n * photons)
    sampled = pois_rand(rng, lambda) * n / photons
    return round(UInt8, clamp01nan(sampled) * 255)
end
function sample_poisson_noise(x::AbstractArray{T, 2}, photons, rng) where {T}
    normalized = maximum(x)
    return map(p -> sample_poisson_noise(p, photons, normalized, rng), x)
end
function sample_poisson_noise(v::Vector{<:AbstractArray{T, 2}}, photons, rng) where {T}
    new_imgs = [similar(x, UInt8) for x in v]
    for (x, new_img) in zip(v, new_imgs)
        new_img .= sample_poisson_noise(x, photons, rng)
    end
    return new_imgs
end

# --- BRIGHTNESS
function add_brightness(x::AbstractArray{T, 2}, amount) where {T}
    return round.(UInt8, clamp01nan.(x .+ amount) .* 255)
end
function add_brightness(v::Vector{<:AbstractArray{T, 2}}, amount) where {T}
    new_imgs = [similar(x, UInt8) for x in v]
    for (x, new_img) in zip(v, new_imgs)
        new_img .= add_brightness(x, amount)
    end
    return new_imgs
end
