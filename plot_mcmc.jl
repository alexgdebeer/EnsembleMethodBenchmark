using HDF5
using Statistics

n_batches = 5000
n_chains = 4

function get_logpost(
    j::Int,
    n_batches::Int,
    logpost_name::AbstractString
)

    f = h5open("data/mcmc/chain_$j.h5", "r")
    τs = reduce(vcat, [f["$(logpost_name)_$b"][:] for b ∈ 0:(n_batches-1)])
    close(f)

    return τs

end

function get_param(
    i::Int,
    j::Int,
    n_batches::Int,
    param_name::AbstractString
)

    f = h5open("data/mcmc/chain_$j.h5", "r")
    ηs = reduce(vcat, [f["$(param_name)_$b"][i, :] for b ∈ 0:(n_batches-1)])
    close(f)

    return ηs

end

function get_estimand(
    n_batches::Int,
    n_chains::Int,
    estimand_type::Symbol,
    estimand_name::AbstractString;
    estimand_num::Int=1
)

    xs = []

    for j ∈ 1:n_chains

        if estimand_type == :param 
            push!(xs, get_param(estimand_num, j, n_batches, estimand_name))
        elseif estimand_type == :logpost 
            push!(xs, get_logpost(j, n_batches, estimand_name))
        end
        
    end

    return hcat(xs...)

end

function compute_psrf(
    θs::AbstractMatrix
)

    n, m = size(θs)
    
    μ = mean(θs)
    μ_js = [mean(c) for c ∈ eachcol(θs)]
    s_js = [1 / (n-1) * sum((c .- μ_js[j]).^2) for (j, c) ∈ enumerate(eachcol(θs))]

    B = n / (m-1) * sum((μ_js .- μ).^2)
    W = 1 / m * sum(s_js)

    varp = (n-1)/n * W + 1/n * B
    psrf = sqrt(varp / W)

    return psrf

end

# τs = get_estimand(n_batches, n_chains, :logpost, "τs")
# # Remove warm-up iterations
# τs = τs[10_001:end, :]
# psrf = compute_psrf(τs)

# ηs = get_estimand(n_batches, n_chains, :param, "ηs", estimand_num=1)
# ηs = ηs[10_001:end, :]

# psrf = @time compute_psrf(ηs)

f = h5open("data/mcmc/chain_1.h5", "r")
ηs = reduce(hcat, [f["ηs_$b"][:, :] for b ∈ 0:(n_batches-1)])
close(f)

# # Compute overall mean and mean of each chain 
# μ = mean(τs)
# μ_js = [mean(c) for c ∈ eachcol(τs)]

# nτ = length(τs)
# nτ_j = length(τs[:, 1])

# s_js = [1 / (nτ-1) * sum((c .- μ_js[j]).^2) for (j, c) ∈ enumerate(eachcol(τs))]

# B = nτ_j / (N_CHAINS-1) * sum((μ_js .- μ).^2)
# W = 1 / N_CHAINS * sum(s_js)

# varp = (nτ-1)/nτ * W + 1/nτ * B 
# psrf = sqrt(varp / W)

# # ηs = reduce(hcat, [f["ηs_$i"][:, end] for i ∈ 10:(N_BATCHES-1)])
# # θs = reduce(hcat, [transform(pr, η) for η ∈ eachcol(ηs)])
# # μ_post = mean(θs, dims=2)
