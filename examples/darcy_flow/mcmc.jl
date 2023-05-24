using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random

using DarcyFlow
using SimIntensiveInference

Random.seed!(16)

# ----------------
# Data generation 
# ----------------

# Define a fine grid
xmin, Δx, xmax = 0.0, 0.05, 1.0
ymin, Δy, ymax = 0.0, 0.05, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_x = length(xs)
n_y = length(ys)

# Set up boundary conditions
x0 = DarcyFlow.BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = DarcyFlow.BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = DarcyFlow.BoundaryCondition(:y0, :neumann, (x, y) -> -2.0)
y1 = DarcyFlow.BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
bcs = Dict(:y0 => y0, :y1 => y1, :x0 => x0, :x1 => x1)

# Define a permeability distribution
Γ_p = @time DarcyFlow.exp_squared_cov(1.0, 0.3, xs, ys)
p_dist = MvLogNormal(MvNormal(Γ_p))

# Define the observation locations
x_locs = 0.1:0.2:0.9
y_locs = 0.25:0.1:0.95

n_obs = length(x_locs) * length(y_locs)

# Define the observation noise
σ_ϵ = 0.01
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

# Generate a set of observations
ps_true, xs_o, ys_o, us_o = DarcyFlow.generate_data(xs, ys, bcs, p_dist, x_locs, y_locs, ϵ_dist)

# ----------------
# MCMC
# ----------------

# Define a mapping from a vector of permeabilities to the observations
function f(logps)::AbstractMatrix

    # Form permeability interpolation object 
    ps = exp.(logps)
    p = interpolate((xs, ys), reshape(ps, n_x, n_y), Gridded(Linear()))

    # Generate and solve the steady-state system of equations
    A, b = DarcyFlow.generate_grid(xs, ys, p, bcs)
    sol = solve(LinearProblem(A, b))

    return reshape(sol.u, n_x, n_y)

end

# Define a mapping from the full output of the model to the observations
function g(us::AbstractMatrix)::AbstractVector 

    us = interpolate((xs, ys), reshape(us, n_x, n_y), Gridded(Linear()))
    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end

# Form prior distribution 
Γ_π = DarcyFlow.exp_squared_cov(1.0, 0.3, xs, ys)
π = MvNormal(Γ_π)

# Form likelihood 
Γ_L = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_L)

# Form perurbation kernel
Γ_K = 0.0005 * Γ_π
K = MvNormal(Γ_K)

N = 1_000

#ps, us = SimIntensiveInference.run_mcmc(f, g, π, L, K, N)

p_map, ps = SimIntensiveInference.run_rml(f, g, π, L, N)