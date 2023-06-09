using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random

using DarcyFlow

Random.seed!(16)

# Define boundaries of domain
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

# Generate grids
Δx_f, Δy_f = 0.05, 0.05
Δx_c, Δy_c = 0.05, 0.05

g_f = construct_grid(xmin:Δx_f:xmax, ymin:Δy_f:ymax)
g_c = construct_grid(xmin:Δx_c:xmax, ymin:Δy_c:ymax)

# Define boundary conditions
bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

# ----------------
# Generate data using fine grid
# ----------------

# Define the distribution the true (log) permeability field will be drawn from
σ_t, γ_t = 1.0, 0.20
Γ_p = DarcyFlow.exp_squared_cov(σ_t, γ_t, g_f.xs, g_f.ys)
logp_dist = MvNormal(Γ_p)

# Define the observation locations
x_locs = 0.1:0.2:0.9
y_locs = 0.25:0.1:0.95
n_obs = length(x_locs) * length(y_locs)

# Define the distribution of the observation noise
σ_ϵ_t = 0.02
Γ_ϵ = σ_ϵ_t^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

# Generate a set of observations
logps_t, xs_o, ys_o, us_o = DarcyFlow.generate_data(
    g_f, x_locs, y_locs, bcs, logp_dist, ϵ_dist
)

# ----------------
# Define inversion parameters
# ----------------

# Define prior
σ, γ = σ_t, γ_t
Γ_π = DarcyFlow.exp_squared_cov(σ, γ, g_c.xs, g_c.ys)
π = MvNormal(Γ_π)

# Define likelihood
σ_ϵ = σ_ϵ_t
Γ_L = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_L)

# Define mapping between vector of log permeabilities and matrix of pressures
function f(
    logps::AbstractVector, 
    grid::Grid, 
    bcs::Dict{Symbol, BoundaryCondition}
)::AbstractMatrix

    ps = reshape(exp.(logps), grid.nx, grid.ny)

    # Generate and solve the steady-state system of equations
    A = construct_A(grid, ps, bcs)
    b = construct_b(grid, ps, bcs)

    us = reshape(solve(LinearProblem(A, b)), grid.nx, grid.ny)

    return us

end

# Define mapping from a matrix of the model pressure field to the observations
function g(
    us::AbstractMatrix,
    grid::Grid,
    xs_o::AbstractVector,
    ys_o::AbstractVector
)::AbstractVector 

    us = interpolate((grid.xs, grid.ys), us, Gridded(Linear()))
    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end

f_args = (g_c, bcs)
g_args = (g_c, xs_o, ys_o)