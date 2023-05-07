"""Defines the Lotka-Volterra model, as used in Toni et al. (2009)."""

module LVModel

using DifferentialEquations
using Distributions
using LinearAlgebra
using Random; Random.seed!(0)
using Statistics

include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const PLOTS_DIR = "plots/lotka_volterra"

# Define time parameters
const T_0 = 0
const T_1 = 15
const ΔT = 0.1
const TS = T_0:ΔT:T_1
const N_TS = length(TS)

"""Runs the forward model. Returns the results as a 2*n array, where n is the 
number of timesteps."""
function f(
    θs; 
    u_0::AbstractVector=Y_0, 
    t_0::Real=T_0,
    t_1::Real=T_1
)::Matrix

    """Returns derivative of y at a given time."""
    function dydt(ys::Vector, θs, t::Real)::Vector
        a, b = θs
        return [a*ys[1]-ys[1]*ys[2], b*ys[1]*ys[2]-ys[2]]
    end

    prob = ODEProblem(dydt, u_0, (t_0, t_1), θs)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t_0:ΔT:t_1)
    return reduce(hcat, sol.u)

end


"""Returns a vector of the model outputs at a specified set of time indices."""
function h(ys::AbstractMatrix; is::AbstractVector=IS_O)::AbstractVector
    return reduce(vcat, ys[:, is])
end


"""Maps from the parameters / states to the modelled observations."""
function b(θ::AbstractVector, u::AbstractVector)::AbstractVector
    return u
end

# Define true model initial conditions, parameters and outputs
const Y_0 = [1.0, 0.5]
const θS_T = [1.0, 1.0]
const YS_T = f(θS_T)

# Generate the observations
const σ_ϵ = 0.25
const N_IS_O = 15
const IS_O = sort(randperm(N_TS)[1:N_IS_O])
const TS_O = TS[IS_O]
const YS_O = YS_T[:, IS_O] + rand(Normal(0.0, σ_ϵ), size(YS_T[:, IS_O]))

# Define the prior
const μ_π = [0.0, 0.0]
const σ_π = 3.0
const Γ_π = σ_π^2 * Matrix(I, 2, 2)
const π = SimIntensiveInference.GaussianPrior(μ_π, Γ_π)

# Define the likelihood
const μ_ϵ = zeros(2N_IS_O)
const μ_L = reduce(vcat, YS_O)
const Γ_ϵ = σ_ϵ^2 * Matrix(I, 2N_IS_O, 2N_IS_O)
const L = SimIntensiveInference.GaussianLikelihood(μ_L, Γ_ϵ)

# Compute the properties of the true posterior 
const N_PTS = 200
const AS = collect(range(0.5, 1.5, N_PTS))
const BS = collect(range(0.5, 1.5, N_PTS))
const d(θs) = SimIntensiveInference.density(π, θs) * SimIntensiveInference.density(L, h(f(θs)))
const POST_JOINT, POST_MARG_A, POST_MARG_B = Plotting.density_grid(AS, BS, d)

if abspath(PROGRAM_FILE) == @__FILE__

    Plotting.plot_lv_system(TS, YS_T, TS_O, YS_O, "$(PLOTS_DIR)/lv_system.pdf")

    Plotting.plot_density_grid(
        AS, BS, POST_JOINT, POST_MARG_A, POST_MARG_B, 
        "Posterior Density",
        "$(PLOTS_DIR)/posterior_density.pdf"
    )

end

end