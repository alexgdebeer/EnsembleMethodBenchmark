abstract type Localiser end
abstract type Inflator end

struct IdentityLocaliser <: Localiser end
struct IdentityInflator <: Inflator end
struct FisherLocaliser <: Localiser end

mutable struct ShuffleLocaliser <: Localiser

    n_shuffle::Int 
    P::Union{AbstractMatrix, Nothing}
    
    function ShuffleLocaliser(n_shuffle=50)
        return new(n_shuffle, nothing)
    end

end

struct BootstrapLocaliser <: Localiser

    n_boot::Int 
    σ::Real 
    tol::Real

    function BootstrapLocaliser(n_boot=50, σ=0.6, tol=1e-8) 
        return new(n_boot, σ, tol)
    end

end

struct PowerLocaliser <: Localiser
    α::Real 
    PowerLocaliser(α=0.4) = new(α)
end

struct AdaptiveInflator <: Inflator
    n_dummy_params::Int 
    AdaptiveInflator(n_dummy_params=50) = new(n_dummy_params)
end

function compute_Δs(xs::AbstractMatrix)

    N = size(xs, 2)
    Δs = (xs .- mean(xs, dims=2)) ./ √(N-1)
    return Δs

end

function compute_covs(
    ηs::AbstractMatrix, 
    Gs::AbstractMatrix
)

    Δη = compute_Δs(ηs)
    ΔG = compute_Δs(Gs)

    C_ηG = Δη * ΔG'
    C_GG = ΔG * ΔG'

    return C_ηG, C_GG

end

function compute_cors(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix
)

    V_η = Diagonal(std(ηs, dims=2)[:])
    V_G = Diagonal(std(Gs, dims=2)[:])

    C_ηG, C_GG = compute_covs(ηs, Gs)
    R_ηG = inv(V_η) * C_ηG * inv(V_G)
    R_GG = inv(V_G) * C_GG * inv(V_G)

    return R_ηG, R_GG

end

"""Shuffles a list of indices from 1 to N, ensuring that none of them
end up where they started."""
function get_shuffled_inds(N::Int)

    inds = shuffle(1:N)
    for i ∈ 1:N 
        if inds[i] == i 
            inds[(i%N)+1], inds[i] = inds[i], inds[(i%N)+1]
        end
    end

    return inds

end

function gaspari_cohn(z::Real)
    if 0 ≤ z ≤ 1 
        return -(1/4)z^5 + (1/2)z^4 + (5/8)z^3 - (5/3)z^2 + 1
    elseif 1 < z ≤ 2 
        return (1/12)z^5 - (1/2)z^4 + (5/8)z^3 + (5/3)z^2 - 5z + 4 - (2/3)z^-1
    else
        return 0.0
    end
end

function save_results(results::Dict, fname::AbstractString)

    for (k, v) in pairs(results)
        h5write(fname, k, v)
    end

end