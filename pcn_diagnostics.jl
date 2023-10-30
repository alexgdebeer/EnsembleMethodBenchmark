using HDF5
using Statistics

include("setup.jl")

CHAIN_LENGTH = 500_000 
WARMUP_LENGTH = 100 # TODO: EDIT
BATCH_LENGTH = 100

BATCH_INC = 10
BATCH_INDS = BATCH_INC:BATCH_INC:BATCH_LENGTH

N_CHAINS = 4
N_BATCHES = CHAIN_LENGTH ÷ BATCH_LENGTH
N_SAMPLES = length(BATCH_INDS) * N_BATCHES

DATA_FOLDER = "data/mcmc"

function compute_psrf(
    θs::AbstractMatrix
)::Real

    # Split each chain in half 
    θs = reshape(θs, :, 2N_CHAINS)
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

ηs = zeros(pr.Nη, N_SAMPLES, N_CHAINS)
θs = zeros(pr.Nθ, N_SAMPLES, N_CHAINS)
τs = zeros(CHAIN_LENGTH, N_CHAINS)

for i ∈ 1:N_CHAINS

    f = h5open("$(DATA_FOLDER)/chain_$i.h5", "r")
    ηs[:, :, i] = reduce(hcat, [f["ηs_$b"][:, BATCH_INDS] for b ∈ 1:N_BATCHES])
    θs[:, :, i] = reduce(hcat, [f["θs_$b"][:, BATCH_INDS] for b ∈ 1:N_BATCHES])
    τs[:, i] = reduce(vcat, [f["τs_$b"][:] for b ∈ 1:N_BATCHES])
    close(f)
    @info "Finished reading data from chain $i."

end

psrfs_η = @time [compute_psrf(ηs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nη]
psrfs_θ = @time [compute_psrf(θs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nθ]

μ_post = mean(reshape(θs, ))