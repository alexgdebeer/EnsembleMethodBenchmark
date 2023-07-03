using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random: seed!
using SimIntensiveInference

include("setup/setup.jl")

seed!(16)

xmin, Δx, xmax = 0.0, 0.02, 1.0
ymin, Δy, ymax = 0.0, 0.02, 1.0

grid = SteadyStateGrid(xmin:Δx:xmax, ymin:Δy:ymax)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

# ----------------
# Prior setup
# ----------------

σ, γx, γy = 1.0, 0.2, 0.2
k = ARDExpSquaredKernel(σ, γx, γy)

μ = 0.0
p = GaussianPrior(μ, k, grid.xs, grid.ys)

# ----------------
# Data generation
# ----------------

# Sample the true set of parameters from the prior
θs_t = rand(p)
logps_t = reshape(θs_t, grid.nx, grid.ny)
ps_t = exp.(logps_t)
us_t = solve(grid, ps_t, bcs)

# Define the observation locations
x_locs = 0.1:0.2:0.9
y_locs = 0.3:0.2:0.9
n_obs = length(x_locs) * length(y_locs)

# Define the distribution of the observation noise
σ_ϵ_t = 0.02
Γ_ϵ = σ_ϵ_t^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

xs_o, ys_o, us_o = generate_data(grid, us_t, x_locs, y_locs, ϵ_dist)

# ----------------
# Likelihood setup 
# ----------------

σ_ϵ = 0.02
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_ϵ)

# Define mapping between vector of log permeabilities and matrix of pressures
function f(
    θs::AbstractVector
)::AbstractMatrix

    ps = reshape(exp.(θs), grid.nx, grid.ny)
    us = solve(grid, ps, bcs)

    return us

end

# Define mapping from a matrix of the model pressure field to the observations
function g(
    us::AbstractMatrix
)::AbstractVector 

    us = interpolate((grid.xs, grid.ys), us, Gridded(Linear()))
    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end