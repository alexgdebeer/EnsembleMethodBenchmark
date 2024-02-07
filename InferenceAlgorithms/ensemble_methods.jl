abstract type Localiser end
abstract type Inflator end

struct IdentityLocaliser <: Localiser end
struct IdentityInflator <: Inflator end

struct ShuffleLocaliser <: Localiser
    n_shuffle::Int 
    ShuffleLocaliser(; n_shuffle=100) = new(n_shuffle)
end

struct BootstrapLocaliser <: Localiser

    n_boot::Int 
    σ::Real 
    type::Symbol

    function BootstrapLocaliser(; 
        n_boot=100, 
        σ=0.6, 
        type=:unregularised
    )

        if type ∉ [:regularised, :unregularised]
            error("Invalid type passed in.")
        end

        return new(n_boot, σ, type)
    end

end

struct AdaptiveInflator <: Inflator
    n_dummy_params::Int 
    AdaptiveInflator(; n_dummy_params=50) = new(n_dummy_params)
end

function tsvd(A::AbstractMatrix; e::Real=0.99)

    U, Λ, V = svd(A)
    minimum(Λ) < 0 && error(minimum(Λ))
    λ_cum = cumsum(Λ)

    for i ∈ 1:length(Λ)
        if λ_cum[i] / λ_cum[end] ≥ e 
            return U[:, 1:i], Λ[1:i], V[:, 1:i]
        end
    end

end

function compute_Δs(xs::AbstractMatrix)

    N = size(xs, 2)
    Δs = (xs .- mean(xs, dims=2)) ./ √(N-1)
    return Δs

end

function compute_covs(
    θs::AbstractMatrix, 
    Gs::AbstractMatrix
)

    Δθ = compute_Δs(θs)
    ΔG = compute_Δs(Gs)

    C_θG = Δθ * ΔG'
    C_GG = ΔG * ΔG'

    return C_θG, C_GG

end

function compute_cors(
    θs::AbstractMatrix,
    Gs::AbstractMatrix
)

    V_θ = Diagonal(std(θs, dims=2)[:])
    V_G = Diagonal(std(Gs, dims=2)[:])

    C_θG, C_GG = compute_covs(θs, Gs)
    R_θG = inv(V_θ) * C_θG * inv(V_G)
    R_GG = inv(V_G) * C_GG * inv(V_G)

    return R_θG, R_GG

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

"""Carries out the localisation procedure outlined by Luo and Bhakta 
(2020)."""
function localise(
    localiser::ShuffleLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    K::AbstractMatrix
)

    Nθ, Ne = size(θs)
    NG, Ne = size(Gs)

    R_θG = compute_cors(θs, Gs)[1]

    P = zeros(Nθ, NG)
    R_θGs = zeros(Nθ, NG, localiser.n_shuffle)

    for i ∈ 1:localiser.n_shuffle
        inds = get_shuffled_inds(Ne)
        θs_shuffled = θs[:, inds]
        R_θGs[:, :, i] = compute_cors(θs_shuffled, Gs)[1]
    end

    for i ∈ 1:Nθ, j ∈ 1:NG
        σ_ij = std(R_θGs[i, j, :])
        if σ_ij ≤ abs(R_θG[i, j])
            P[i, j] = 1
        end
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

function run_ensemble(
    θs::AbstractMatrix, 
    F::Function, 
    G::Function, 
    pr::MaternField
)

    us = hcat([transform(pr, θ_i) for θ_i ∈ eachcol(θs)]...)
    Fs = hcat([F(u_i) for u_i ∈ eachcol(us)]...)
    Gs = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    return us, Fs, Gs

end

function save_results(results::Dict, fname::AbstractString)

    for (k, v) in pairs(results)
        h5write(fname, k, v)
    end

end