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

function get_shuffled_inds(N::Int)

    inds = shuffle(1:N)

    # Ensure none of the indices end up where they started
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

"""Carries out the localisation procedure outlined by Luo and Bhakta 
(2020)."""
function localise(
    localiser::ShuffleLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    K::AbstractMatrix
)

    if localiser.P !== nothing 
        return localiser.P .* K
    end

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    R_ηG = compute_cors(ηs, Gs)[1]

    P = zeros(Nη, NG)
    R_ηGs = zeros(Nη, NG, localiser.n_shuffle)

    for i ∈ 1:localiser.n_shuffle
        inds = get_shuffled_inds(Ne)
        ηs_shuffled = ηs[:, inds]
        R_ηGs[:, :, i] = compute_cors(ηs_shuffled, Gs)[1]
    end

    σs_e = median(abs.(R_ηGs), dims=3) ./ 0.6745

    for i ∈ 1:Nη, j ∈ 1:NG
        z = (1 - abs(R_ηG[i, j])) / (1 - σs_e[i, j])
        P[i, j] = gaspari_cohn(z)
    end

    localiser.P = P
    return P .* K

end

"""Carries out the localisation procedure outlined by Flowerdew 
(2014)."""
function localise(
    localiser::FisherLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    K::AbstractMatrix
)

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    R_ηG = compute_cors(ηs, Gs)[1]
    P = zeros(size(R_ηG))

    for i ∈ 1:Nη, j ∈ 1:NG 
        ρ_ij = R_ηG[i, j]
        s = log((1+ρ_ij) / (1-ρ_ij)) / 2
        σ_s = (tanh(s + √(Ne-3)^-1) - tanh(s - √(Ne-3)^-1)) / 2
        P[i, j] = ρ_ij^2 / (ρ_ij^2 + σ_s^2)
    end

    return P .* K

end

function generate_dummy_params(
    inflator::AdaptiveInflator, 
    Ne::Int
)

    dummy_params = rand(Normal(), (inflator.n_dummy_params, Ne))
    for r ∈ eachrow(dummy_params)
        μ, σ = mean(r), std(r)
        r = (r .- μ) ./ σ
    end
    return dummy_params

end

function save_results(results::Dict, fname::AbstractString)

    for (k, v) in pairs(results)
        h5write(fname, k, v)
    end

end