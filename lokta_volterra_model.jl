"""Defines some functions that form a Lokta Volterra ODE model."""

module LVModel


using DifferentialEquations
using Distributions
using LinearAlgebra
using Random
using Statistics


Random.seed!(0)


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


"""Provides a mapping from the complete model outputs to the outputs that 
correspond to the observations."""
function H(ys::Matrix, θs::Vector)
    return ys
end


"""Defines a parameter / state to measurement mapping (for this model, just 
returns the state.)"""
function g(θ, u)
    return u
end


"""Returns the sum of squared distances between the model outputs and the 
observations."""
function Δ(x_a, x_b)
    return sum((x_a .- x_b).^2)
end


"""Returns the maximum absolute difference between the model outputs and the 
observations."""
function Δ_u(x_a, x_b)
    return max(abs.(x_a-x_b)...)
end


end