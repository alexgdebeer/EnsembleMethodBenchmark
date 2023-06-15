using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random

include("finite_differences.jl")
include("data_generation.jl")

Random.seed!(16)

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tmax = 20.0

Δx, Δy = 0.02, 0.02
Δt = 0.05

g = construct_grid(xmin:Δx:xmax, ymin:Δy:ymax, tmax, Δt)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> 1.0)
)

σ, γ = 1.0, 0.10
Γ_p = exp_squared_cov(σ, γ, g.xs, g.ys)
logp_dist = MvNormal(Γ_p)

logps = sample_perms(logp_dist, g.nx, g.ny)
ps = exp.(logps)

us = solve_system(g, ps, bcs)

g_ss = construct_grid(xmin:Δx:xmax, ymin:Δy:ymax)
A = construct_A(g_ss, ps, bcs)
b = construct_b(g_ss, ps, bcs)

us_ss = solve(LinearProblem(A, b))
us_ss = reshape(us_ss, g_ss.nx, g_ss.ny)

println(maximum(abs.(us_ss-us[:,:,end])))
