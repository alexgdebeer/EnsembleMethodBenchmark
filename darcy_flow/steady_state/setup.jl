using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random: seed!
using SimIntensiveInference

include("../setup/setup.jl")

seed!(16)

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

Δx_f, Δy_f = 0.01, 0.01
Δx_c, Δy_c = 0.02, 0.02

grid_f = SteadyStateGrid(xmin:Δx_f:xmax, ymin:Δy_f:ymax)
grid_c = SteadyStateGrid(xmin:Δx_c:xmax, ymin:Δy_c:ymax)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

q(x, y) = 0

# ----------------
# Prior setup
# ----------------

m_bnds = [-0.3, 0.3]
c_bnds = [0.2, 0.8]
a_bnds = [0.05, 0.2]
p_bnds = [0.4, 1.0]
w_bnds = [0.05, 0.2]

μ_o = 0.0
μ_i = 2.0

k_o = ExpSquaredKernel(0.25, 0.1)
k_i = ExpKernel(0.25, 0.1)

p = ChannelPrior(
    m_bnds, c_bnds, a_bnds, p_bnds, w_bnds,
    μ_o, μ_i, k_o, k_i, grid_c.xs, grid_c.ys
)

# ----------------
# Data generation
# ----------------

# Sample the true set of parameters from a scaled version of the prior
θs_t_dist = ChannelPrior(
    m_bnds, c_bnds, a_bnds, p_bnds, w_bnds,
    μ_o, μ_i, k_o, k_i, grid_f.xs, grid_f.ys
)
θs_t = rand(θs_t_dist)
logps_t = reshape(get_perms(θs_t_dist, θs_t), grid_f.nx, grid_f.ny)
ps_t = exp.(logps_t)
us_t = solve(grid_f, ps_t, bcs, q)

# Define the observation locations
x_locs = 0.1:0.2:0.9
y_locs = 0.1:0.2:0.9
n_obs = length(x_locs) * length(y_locs)

# Define the distribution of the observation noise
σ_ϵ_t = 0.02
Γ_ϵ = σ_ϵ_t^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

xs_o, ys_o, us_o = generate_data(grid_f, us_t, x_locs, y_locs, ϵ_dist)

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

    logps = get_perms(p, θs)
    ps = reshape(exp.(logps), grid_c.nx, grid_c.ny)
    us = solve(grid_c, ps, bcs, q)

    return us

end

# Define mapping from a matrix of the model pressure field to the observations
function g(
    us::AbstractMatrix
)::AbstractVector 

    us = interpolate((grid_c.xs, grid_c.ys), us, Gridded(Linear()))
    return [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)]

end