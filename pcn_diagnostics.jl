using HDF5
using Statistics

include("setup.jl")

N_CHAINS = 5
N_BATCHES = 20_000
N_SAMPLES_PER_BATCH = 1
N_SAMPLES = N_BATCHES * N_SAMPLES_PER_BATCH
WARMUP_LENGTH = 500

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
    s_js = [1 / (n-1) * sum((c .- μ_js[j]).^2) 
            for (j, c) ∈ enumerate(eachcol(θs))]

    B = n / (m-1) * sum((μ_js .- μ).^2)
    W = 1 / m * sum(s_js)

    varp = (n-1)/n * W + 1/n * B
    psrf = sqrt(varp / W)

    return psrf

end

NF = n_wells * grid_c.nt

θs = zeros(pr.Nθ, N_SAMPLES, N_CHAINS)
us = zeros(pr.Nu, N_SAMPLES, N_CHAINS)
Fs = zeros(NF, N_SAMPLES, N_CHAINS)
τs = zeros(100N_BATCHES, N_CHAINS)

for i ∈ 1:N_CHAINS

    f = h5open("$(DATA_FOLDER)/chain_$i.h5", "r")
    θs[:, :, i] = reduce(hcat, [f["ηs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    us[:, :, i] = reduce(hcat, [f["θs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    Fs[:, :, i] = reduce(hcat, [f["Fs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    τs[:, i] = reduce(vcat, [f["τs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    close(f)
    @info "Finished reading data from chain $i."

end

psrfs_θ = [compute_psrf(θs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nθ]

μ_post_θ = mean(reshape(θs[:, WARMUP_LENGTH+1:end, :], pr.Nθ, :, 1), dims=2)
μ_post = reshape(transform(pr, vec(μ_post_θ)), grid_c.nx, grid_c.nx)

σ_post = std(reshape(us[:, WARMUP_LENGTH+1:end, :], pr.Nu, :, 1), dims=2)
σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

trace_1 = τs[1:10:end, :]
trace_2 = θs[2109, :, :]
trace_3 = θs[end, :, :]

h5write(RESULTS_FNAME, "trace_1", trace_1)
h5write(RESULTS_FNAME, "trace_2", trace_2)
h5write(RESULTS_FNAME, "trace_3", trace_3)
h5write(RESULTS_FNAME, "mean", μ_post)
h5write(RESULTS_FNAME, "stds", σ_post)
h5write(RESULTS_FNAME, "samples_1", us[:, 5_000, :])
h5write(RESULTS_FNAME, "samples_2", us[:, 10_000, :])
