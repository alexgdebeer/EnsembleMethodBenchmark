using HDF5
using Statistics

include("setup.jl")

CHAIN_LENGTH = 500_000 
WARMUP_LENGTH = 5000
BATCH_LENGTH = 100

BATCH_INC = 10
BATCH_INDS = BATCH_INC:BATCH_INC:BATCH_LENGTH

N_CHAINS = 4
N_BATCHES = CHAIN_LENGTH ÷ BATCH_LENGTH
N_SAMPLES = length(BATCH_INDS) * N_BATCHES

DATA_FOLDER = "data/pcn"
RESULTS_FNAME = "data/pcn/pcn.h5"

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

NF = n_wells * grid_c.nt

ηs = zeros(pr.Nη, N_SAMPLES, N_CHAINS)
θs = zeros(pr.Nθ, N_SAMPLES, N_CHAINS)
Fs = zeros(NF, N_SAMPLES, N_CHAINS)
τs = zeros(CHAIN_LENGTH, N_CHAINS)

for i ∈ 1:N_CHAINS

    f = h5open("$(DATA_FOLDER)/chain_$i.h5", "r")
    ηs[:, :, i] = reduce(hcat, [f["ηs_$b"][:, BATCH_INDS] for b ∈ 1:N_BATCHES])
    θs[:, :, i] = reduce(hcat, [f["θs_$b"][:, BATCH_INDS] for b ∈ 1:N_BATCHES])
    Fs[:, :, i] = reduce(hcat, [f["Fs_$b"][:, BATCH_INDS] for b ∈ 1:N_BATCHES])
    τs[:, i] = reduce(vcat, [f["τs_$b"][:] for b ∈ 1:N_BATCHES])
    close(f)
    @info "Finished reading data from chain $i."

end

psrfs_η = @time [compute_psrf(ηs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nη]
psrfs_θ = @time [compute_psrf(θs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nθ]

Fs = reshape(Fs, NF, :, 1)
Fs = dropdims(Fs, dims=3)

μ_post_η = vec(mean(reshape(ηs[:, WARMUP_LENGTH+1:end, :], pr.Nη, :, 1), dims=2))
μ_post = reshape(transform(pr, μ_post_η), grid_c.nx, grid_c.nx)
σ_post = reshape(std(reshape(θs[:, WARMUP_LENGTH+1:end, :], pr.Nθ, :, 1), dims=2), grid_c.nx, grid_c.nx)

θis = vec(θs[3200, WARMUP_LENGTH+1:end, :])
ls = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ vec(ηs[end, WARMUP_LENGTH+1:end, :])]

h5write(RESULTS_FNAME, "Fs", Fs[:, 200:200:end]) # TODO: tidy
h5write(RESULTS_FNAME, "μ", μ_post)
h5write(RESULTS_FNAME, "σ", σ_post)
h5write(RESULTS_FNAME, "θi", θis)
h5write(RESULTS_FNAME, "l", ls)