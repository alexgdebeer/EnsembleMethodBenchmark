using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random

using JLD2

using DarcyFlow
using SimIntensiveInference

Random.seed!(16)

# Define parameters of fine and coarse grids
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

Δx_f, Δy_f = 0.01, 0.01
Δx_c, Δy_c = 0.02, 0.02

xs_f = xmin:Δx_f:xmax
ys_f = ymin:Δy_f:ymax

xs_c = xmin:Δx_c:xmax
ys_c = ymin:Δy_c:ymax

nx_f = length(xs_f)
ny_f = length(ys_f)

nx_c = length(xs_c)
ny_c = length(ys_c)

# Set up boundary conditions
bcs = Dict(
    :x0 => DarcyFlow.BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => DarcyFlow.BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => DarcyFlow.BoundaryCondition(:y0, :neumann, (x, y) -> -2.0), 
    :y1 => DarcyFlow.BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

# Define a permeability distribution
Γ_p = DarcyFlow.exp_squared_cov(1.0, 0.25, xs_f, ys_f)
p_dist = MvLogNormal(MvNormal(Γ_p))

# Define the observation locations
x_locs = 0.1:0.2:0.9
y_locs = 0.25:0.1:0.95

n_obs = length(x_locs) * length(y_locs)

# Define the observation noise
σ_ϵ = 0.1
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

# Generate a set of observations
ps_true, xs_o, ys_o, us_o = DarcyFlow.generate_data(
    xs_f, ys_f, bcs, p_dist, x_locs, y_locs, ϵ_dist
)

# Define a mapping from a vector of permeabilities to the observations
function f(logps)::AbstractMatrix

    # Form permeability interpolation object 
    ps = reshape(exp.(logps), nx_c, ny_c)
    p = interpolate((xs_c, ys_c), ps, Gridded(Linear()))

    # Generate and solve the steady-state system of equations
    A, b = DarcyFlow.generate_grid(xs_c, ys_c, p, bcs)
    sol = solve(LinearProblem(A, b))

    return reshape(sol.u, nx_c, ny_c)

end

# Define a mapping from the full output of the model to the observations
function g(us::AbstractMatrix)::AbstractVector 

    us = interpolate((xs_c, ys_c), reshape(us, nx_c, ny_c), Gridded(Linear()))
    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end

# Form prior distribution 
Γ_π = DarcyFlow.exp_squared_cov(1.0, 0.2, xs_c, ys_c)
π = MvNormal(Γ_π)

# Form likelihood 
Γ_L = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_L)

# Form perurbation kernel
Γ_K = (0.03)^2 * Γ_π
K = MvNormal(Γ_K)

N = 100_000

ps, us = @time SimIntensiveInference.run_mcmc(f, g, π, L, K, N, n_chains=8)

@save "mcmc_results.jld2" ps us