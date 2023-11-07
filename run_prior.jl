include("setup.jl")

RESULTS_FNAME = "data/prior/prior.h5"
N_SAMPLES = 1000

ηs = rand(pr, N_SAMPLES)
θs = hcat([transform(pr, ηi) for ηi ∈ eachcol(ηs)]...)
Fs = hcat([F(θi) for θi ∈ eachcol(θs)]...)
Gs = hcat([G(Fi) for Fi ∈ eachcol(Fs)]...)

μ_η = mean(ηs, dims=2)
μ_pri = reshape(transform(pr, μ_η), grid_c.nx, grid_c.nx)
σ_pri = reshape(std(θs, dims=2), grid_c.nx, grid_c.nx)

θis = θs[3200, :]
ls = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[end, :]]

h5write(RESULTS_FNAME, "ηs", ηs)
h5write(RESULTS_FNAME, "θs", θs)
h5write(RESULTS_FNAME, "Fs", model_r.B_wells * Fs)
h5write(RESULTS_FNAME, "Gs", Gs)
h5write(RESULTS_FNAME, "μ", μ_pri)
h5write(RESULTS_FNAME, "σ", σ_pri)
h5write(RESULTS_FNAME, "θi", θis)
h5write(RESULTS_FNAME, "l", ls)