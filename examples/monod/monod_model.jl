"""Defines a simple MONOD model, as used in Bardsley et al. (2014)."""

module MONODModel

using LinearAlgebra
using Random; Random.seed!(0)

include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const PLOTS_DIR = "plots/monod"

const XS = 1:400

# Define the factor vector (x) and observations (y)
const XS_O = [28, 55, 83, 110, 138, 225, 375]
const YS_O = [0.053, 0.060, 0.112, 0.105, 0.099, 0.122, 0.125]
const IS_O = indexin(XS_O, XS)
const N_IS = length(IS_O)

# Define the prior 
const μ_π = [0.15, 50.0]
const σs_π = [0.2, 50.0]
const Γ_π = diagm(σs_π.^2)
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)

# Define the likelihood
const σ_ϵ = 0.012
const Γ_ϵ = σ_ϵ^2 * Matrix(I, N_IS, N_IS)
const L = SimIntensiveInference.GaussianLikelihood(MONODModel.YS_O, MONODModel.Γ_ϵ)

# Define the model, and the mapping between the outputs and observations
const f(θs) = ((θs[1]*XS) ./ (θs[2].+XS))[:,:]'
const g(ys) = vec(ys[:, IS_O])

# Define a function that returns the modelled y value corresponding to a given x 
const H(θs, xs) = f(θs)[XS.==xs]

# Compute the true posterior on a grid
const N_PTS = 500
const θ1S = collect(range(0, 0.3, N_PTS))
const θ2S = collect(range(-50, 200, N_PTS))
const d(θs) = SimIntensiveInference.density(π, θs) * SimIntensiveInference.density(L, g(f(θs)))
const POST_JOINT, POST_MARG_θ1, POST_MARG_θ2 = Plotting.density_grid(θ1S, θ2S, d)

if abspath(PROGRAM_FILE) == @__FILE__

    Plotting.plot_monod_obs(XS, YS_O, "$(PLOTS_DIR)/observations.pdf")

    Plotting.plot_density_grid(
        θ1S, θ2S, POST_JOINT, POST_MARG_θ1, POST_MARG_θ2, 
        "Posterior Density",
        "$(PLOTS_DIR)/posterior_density.pdf"
    )

end

end