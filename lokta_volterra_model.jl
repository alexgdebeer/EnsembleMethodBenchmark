"""Defines some functions that form a Lokta Volterra ODE model."""

module LVModel


export f, generate_data, generate_obs_operator, Δ, Δ_u


using DifferentialEquations
using Distributions
using LinearAlgebra
using Random
using Statistics


# Define time parameters
const T_0 = 0
const T_1 = 15
const ΔT = 0.1
const TS = T_0:ΔT:T_1
const N_STEPS = length(TS)

# Define model parameters
const X_0 = [1.0, 0.5]
const θs_t = [1.0, 1.0]

# Define 
const N_DATA = 30
const σ_ϵ = 0.25

Random.seed!(16)


"""Runs the forward model, and returns the results as a column vector, with the 
x measurements followed by the y measurements."""
function f(
    θs; 
    x_0::AbstractVector=X_0, 
    t_0::Real=T_0,
    t_1::Real=T_1
)::Vector

    """Returns derivative of x at a given time."""
    function dydt(ys::Vector, θs, t::Real)::Vector
        x, y = ys; a, b = θs
        return [a*x-x*y, b*x*y-y]
    end

    prob = ODEProblem(dydt, x_0, (t_0, t_1), θs)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t_0:ΔT:t_1)

    # Return everything as a single row, with the x observations followed by 
    # the y observations
    println(sol.u)
    return vec(hcat(sol.u...)')

end


"""Defines a parameter / state to measurement mapping (for this model, just 
returns the state.)"""
function g(θ, u)
    return u
end


"""Generates the data for the model."""
function generate_data(θs_t::Vector, σ_ϵ::Real)

    # Define a set of true values for the parameters and observations
    xs_t = f(θs_t)

    # Choose a random set of times to generate data at
    is_o = sort(randperm(N_STEPS)[1:N_DATA])

    # Generate the set of observed data
    ts_o = repeat(TS[is_o], 2)
    xs_o = vec(vcat(xs_t[is_o, :], xs_t[is_o .+ N_STEPS, :]))

    # Add independent Gaussian noise to the data
    xs_o .+= rand(Distributions.Normal(0.0, σ_ϵ), 2N_DATA)

    return TS, xs_t, is_o, ts_o, xs_o

end


"""Generates the observation operator."""
function generate_obs_operator(is_obs::Vector)::Matrix

    n = length(is_obs)

    G_filled = zeros(n, N_STEPS)
    G_blank = zeros(n, N_STEPS)

    for (i, i_obs) ∈ enumerate(is_obs)
        G_filled[i, i_obs] = 1.0
    end

    return [G_filled G_blank; G_blank G_filled]

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