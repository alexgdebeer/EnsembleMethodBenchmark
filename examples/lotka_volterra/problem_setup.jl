"""Defines the Lotka-Volterra model, as used in Toni et al. (2009)."""

using DifferentialEquations
using Distributions
using LinearAlgebra
using Random
using Statistics

include("../../plotting.jl")

Random.seed!(0)

plots_dir = "plots/lotka_volterra"

# Define time parameters
t_0 = 0.0
t_1 = 15
Δt = 0.1
ts = t_0:Δt:t_1
n_ts = length(ts)

"""Runs the forward model. Returns the results as a 2*n array, where n is the 
number of timesteps."""
function f(
    θs; 
    u_0::AbstractVector=y_0, 
    t_0::Real=t_0,
    t_1::Real=t_1
)::Matrix

    """Returns derivative of y at a given time."""
    function dydt(ys::Vector, θs, t::Real)::Vector
        a, b = θs
        return [a*ys[1]-ys[1]*ys[2], b*ys[1]*ys[2]-ys[2]]
    end

    prob = ODEProblem(dydt, u_0, (t_0, t_1), θs)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t_0:Δt:t_1)
    return reduce(hcat, sol.u)

end


"""Returns a vector of the model outputs at a specified set of time indices."""
function g(ys::AbstractMatrix; is::AbstractVector=is_o)::AbstractVector
    return reduce(vcat, ys[:, is])
end


"""Maps from the parameters / states to the modelled observations."""
function b(θ::AbstractVector, u::AbstractVector)::AbstractVector
    return u
end

# Define true model initial conditions, parameters and outputs
y_0 = [1.0, 0.5]
θs_t = [1.0, 1.0]
ys_t = f(θs_t)

# Generate the observations
σ_ϵ = 0.25
n_is_o = 15
is_o = sort(randperm(n_ts)[1:n_is_o])
ts_o = ts[is_o]
ys_o = ys_t[:, is_o] + rand(Normal(0.0, σ_ϵ), size(ys_t[:, is_o]))

# Define the prior
σ_π = 3.0
Γ_π = σ_π^2 * Matrix(I, 2, 2)
π = MvNormal(Γ_π)

# Define the likelihood
μ_L = reduce(vcat, ys_o)
Γ_ϵ = σ_ϵ^2 * Matrix(I, 2n_is_o, 2n_is_o)
L = MvNormal(μ_L, Γ_ϵ)

# Compute the properties of the true posterior 
n_pts = 100
as = collect(range(0.5, 1.5, n_pts))
bs = collect(range(0.5, 1.5, n_pts))
d(θs) = pdf(π, θs) * pdf(L, g(f(θs)))
post_joint, post_marg_a, post_marg_b = density_grid(as, bs, d)

if abspath(PROGRAM_FILE) == @__FILE__

    plot_lv_system(ts, ys_t, ts_o, ys_o, "$(PLOTS_DIR)/lv_system.pdf")

    plot_density_grid(
        as, bs, post_joint, post_marg_a, post_marg_b, 
        "Posterior Density",
        "$(PLOTS_DIR)/posterior_density.pdf"
    )

end