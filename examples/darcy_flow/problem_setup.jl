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

# Define boundary conditions
bcs = Dict(
    :x0 => DarcyFlow.BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => DarcyFlow.BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => DarcyFlow.BoundaryCondition(:y0, :neumann, (x, y) -> -2.0), 
    :y1 => DarcyFlow.BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

# ----------------
# Generate data using fine grid
# ----------------

# Define fine grid parameters
Δx_f, Δy_f = 0.05, 0.05

xs_f = xmin:Δx_f:xmax
ys_f = ymin:Δy_f:ymax

nx_f = length(xs_f)
ny_f = length(ys_f)

# Define the distribution the true (log) permeability field will be drawn from
σ_t, γ_t = 1.0, 0.25
Γ_p = DarcyFlow.exp_squared_cov(σ_t, γ_t, xs_f, ys_f)
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
    xs_f, ys_f, x_locs, y_locs, bcs, logp_dist, ϵ_dist
)

# ----------------
# Define inversion parameters
# ----------------

# Define coarse grid parameters
Δx_c, Δy_c = 0.05, 0.05

xs_c = xmin:Δx_c:xmax
ys_c = ymin:Δy_c:ymax

nx_c = length(xs_c)
ny_c = length(ys_c)

# Generate grid and b vector
grid = DarcyFlow.construct_grid(xs_c, ys_c)
b = DarcyFlow.construct_b(grid, bcs)

# Define prior distribution 
σ, γ = σ_t, γ_t
Γ_π = DarcyFlow.exp_squared_cov(σ, γ, xs_c, ys_c)
π = MvNormal(Γ_π)

# Define likelihood
σ_ϵ = σ_ϵ_t
Γ_L = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_L)

# Define mapping between vector of log permeabilities and matrix of pressures
function f(logps::AbstractVector)::AbstractMatrix

    # Form permeability interpolation object 
    logps = reshape(logps, nx_c, ny_c)
    ps = interpolate((xs_c, ys_c), exp.(logps), Gridded(Linear()))

    # Generate and solve the steady-state system of equations
    A = DarcyFlow.construct_A(grid, ps, bcs)
    us = reshape(solve(LinearProblem(A, b)), nx_c, ny_c)

    return us

end

# Define mapping from a matrix of the model pressure field to the observations
function g(us::AbstractMatrix)::AbstractVector 

    us = reshape(us, nx_c, ny_c)
    us = interpolate((xs_c, ys_c), us, Gridded(Linear()))

    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end