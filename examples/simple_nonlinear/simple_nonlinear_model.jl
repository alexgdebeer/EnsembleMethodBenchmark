"""Defines the nonlinear model with a single parameter and observation used by 
Chen and Oliver (2013)."""

module SimpleNonlinear

using Distributions 
using LinearAlgebra
using Random; Random.seed!(0)
using Statistics 

include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const PLOTS_DIR = "plots/simple_nonlinear"

"""Mapping between parameters and all modelled outputs."""
function f(θs)::AbstractMatrix 
    return [7/12*θs[1]^3 - 7/2*θs[1]^2 + 8θs[1]][:,:]
end

"""Mapping that returns a vector of the modelled outputs corresponding to 
observations."""
function g(ys::AbstractMatrix)::AbstractVector 
    return vec(ys)
end

# Define true model parameter and output
const θS_T = [6]
const YS_T = f(θS_T)

# Generate the observation
const σ_ϵ = 4.0
const YS_O = YS_T + rand(Normal(0.0, σ_ϵ), 1)

# Define the prior 
const μ_π = [-2.0]
const σ_π = 1.0
const Γ_π = σ_π^2 * Matrix(I, 1, 1)
const π = SimIntensiveInference.GaussianPrior(μ_π, Γ_π)

# Define the likelihood
const μ_ϵ = [0]
const μ_L = vec(YS_O) 
const Γ_ϵ = σ_ϵ^2 * Matrix(I, 1, 1)
const L = SimIntensiveInference.GaussianLikelihood(μ_L, Γ_ϵ)

# Compute the properties of the true posterior 
const θ_MIN, Δθ, θ_MAX = -5, 0.05, 8
const θS = θ_MIN:Δθ:θ_MAX 
const d(θs) = SimIntensiveInference.density(π, θs) * SimIntensiveInference.density(L, g(f(θs)))
const θ_PRIOR= [SimIntensiveInference.density(π, [θ]) for θ ∈ θS]
const θ_POST_UNNORMALISED = [d([θ]) for θ ∈ θS]
const θ_POST = θ_POST_UNNORMALISED ./ (sum(θ_POST_UNNORMALISED)*Δθ)

import PyPlot 
PyPlot.plot(θS, θ_PRIOR, c="tab:blue")
PyPlot.plot(θS, θ_POST, c="tab:orange")
PyPlot.axvline(μ_π[1], c="tab:blue", ls="--")
PyPlot.axvline(θS_T[1], c="tab:orange", ls="--")
PyPlot.savefig("test.pdf")

end