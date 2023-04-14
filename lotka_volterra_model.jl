module LVModel


using DifferentialEquations
using Distributions
using LinearAlgebra
using Random; Random.seed!(0)
using Statistics

include("lotka_volterra_plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")


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
    y_0::AbstractVector=Y_0, 
    t_0::Real=T_0,
    t_1::Real=T_1
)::Matrix

    """Returns derivative of x at a given time."""
    function dydt(ys::Vector, θs, t::Real)::Vector
        a, b = θs
        return [a*ys[1]-ys[1]*ys[2], b*ys[1]*ys[2]-ys[2]]
    end

    prob = ODEProblem(dydt, y_0, (t_0, t_1), θs)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t_0:ΔT:t_1)
    return reduce(hcat, sol.u)

end


"""Provides a mapping from a complete set of model outputs to the observations."""
function g(ys::Matrix)::Vector 
    return reduce(vcat, ys[:, LVModel.IS_O])
end


"""Returns the sum of squared distances between the model outputs and the 
observations."""
function Δ(x_a, x_b)::Real
    return sum((x_a .- x_b).^2)
end


"""Returns the maximum absolute difference between the model outputs and the 
observations."""
function Δ_u(x_a, x_b)::Real
    return maximum(abs.(x_a.-x_b))
end


"""Generates the posterior joint and marginal densities on a grid of parameter 
values."""
function post_density(
    as::Vector, 
    bs::Vector, 
    f::Function, 
    g::Function, 
    π::SimIntensiveInference.AbstractPrior,
    L::SimIntensiveInference.AbstractLikelihood
)::Tuple

    area(x, y) = 0.5sum((x[i+1]-x[i]) * (y[i+1]+y[i]) for i ∈ 1:(length(x)-1))

    joint_density = hcat([
        [
            SimIntensiveInference.density(π, [a, b]) * 
            SimIntensiveInference.density(L, g(f([a, b]))) for b ∈ bs
        ] for a ∈ as
    ]...)

    marg_a = dropdims(sum(joint_density, dims=1), dims=1)
    marg_b = dropdims(sum(joint_density, dims=2), dims=2)
    marg_a ./= area(as, marg_a)
    marg_b ./= area(bs, marg_b)

    return joint_density, marg_a, marg_b

end


# Define true model initial conditions, parameters and outputs
const Y_0 = [1.0, 0.5]
const θS_T = [1.0, 1.0]
const YS_T = f(θS_T)

# Generate observations
const σ_ϵ = 0.25
const N_IS_O = 15
const IS_O = sort(randperm(N_TS)[1:N_IS_O])
const TS_O = TS[IS_O]
const YS_O = YS_T[:, IS_O] + rand(Normal(0.0, σ_ϵ), size(YS_T[:, IS_O]))

# Define a prior
const μ_π = [0, 0]
const σ_π = 3.0
const Γ_π = σ_π^2 * Matrix(I, 2, 2)
const π = SimIntensiveInference.GaussianPrior(μ_π, Γ_π)

# Define an error model
const μ_ϵ = zeros(2LVModel.N_IS_O)
const Γ_ϵ = σ_ϵ^2 .* Matrix(I, 2LVModel.N_IS_O, 2LVModel.N_IS_O)
const e = SimIntensiveInference.GaussianError(μ_ϵ, Γ_ϵ)

# Define the likelihood 
const μ_L = reduce(vcat, LVModel.YS_O) 
const L = SimIntensiveInference.GaussianLikelihood(μ_L, Γ_ϵ)

# Compute the properties of the true posterior 
const N_PTS = 200
const AS = collect(range(0.6, 1.4, N_PTS))
const BS = collect(range(0.6, 1.4, N_PTS))
const POST_JOINT, POST_MARG_A, POST_MARG_B = post_density(AS, BS, f, g, π, L)


if abspath(PROGRAM_FILE) == @__FILE__

    LVModelPlotting.plot_lv_system(TS, YS_T, TS_O, YS_O)

    LVModelPlotting.plot_density_grid(
        AS, BS, POST_JOINT, POST_MARG_A, POST_MARG_B, 
        "Posterior Density",
        "$(LVModel.PLOTS_DIR)/posterior_density.pdf"
    )

end


end