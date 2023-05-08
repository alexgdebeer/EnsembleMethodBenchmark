"""Defines a simple linear model to test things on."""

module LinearModel 

using Distributions
using LinearAlgebra
using Random; Random.seed!(0)
using Statistics

include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const PLOTS_DIR = "plots/linear_model"

# Define time parameters
const T_0 = 0
const T_1 = 5
const ΔT = 0.1
const TS = T_0:ΔT:T_1
const N_TS = length(TS)

"""Mapping between parameters and all modelled outputs."""
function f(θs)::AbstractMatrix
    return (θs[1] .+ θs[2]*TS)[:, :]'
end

"""Mapping that extracts modelled outputs corresponding to observations."""
function g(ys::AbstractMatrix)::AbstractVector 
    return vec(ys[:, IS_O])
end

"""Mapping that extrapolates the line forward to a new state."""
function a(θ; t_0::Real=T_0, t_1::Real=T_1)::AbstractMatrix
    return (θ[1] .+ (θ[2] .* vec(0.0:ΔT:(t_1-t_0))))[:, :]'
end

"""Mapping that extracts the modelled observations corresponding to a given state."""
function b(θs::AbstractVector, us::AbstractVector)::AbstractVector 
    return us
end

# Define true model parameters and outputs
const θS_T = [1.0, 1.0]
const YS_T = f(θS_T)

# Generate the observations
const σ_ϵ = 1.0
const N_IS_O = 8
const IS_O = sort(randperm(N_TS)[1:N_IS_O])
const TS_O = TS[IS_O]
const YS_O = YS_T[IS_O] + rand(Normal(0.0, σ_ϵ), N_IS_O)

# Define the prior
const μ_π = [0.0, 0.0]
const σ_π = 3.0
const Γ_π = σ_π^2 * Matrix(I, 2, 2)
const π = SimIntensiveInference.GaussianPrior(μ_π, Γ_π)

# Define the likelihood
const μ_ϵ = zeros(N_IS_O)
const μ_L = vec(YS_O)
const Γ_ϵ = σ_ϵ^2 * Matrix(I, N_IS_O, N_IS_O)
const L = SimIntensiveInference.GaussianLikelihood(μ_L, Γ_ϵ)

# Compute the properties of the true posterior 
const N_PTS = 200
const θ1S = collect(range(-2, 4, N_PTS))
const θ2S = collect(range(-2, 4, N_PTS))
const d(θs) = SimIntensiveInference.density(π, θs) * SimIntensiveInference.density(L, g(f(θs)))
const POST_JOINT, POST_MARG_θ1, POST_MARG_θ2 = Plotting.density_grid(θ1S, θ2S, d)

if abspath(PROGRAM_FILE) == @__FILE__

    Plotting.plot_linear_model(
        TS, YS_T, TS_O, YS_O, 
        "$(PLOTS_DIR)/linear_model.pdf"
    )

    Plotting.plot_density_grid(
        θ1S, θ2S, POST_JOINT, POST_MARG_θ1, POST_MARG_θ2, 
        "Posterior Density",
        "$(PLOTS_DIR)/posterior_density.pdf"
    )

end

end