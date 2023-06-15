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
tmax = 1.0

Δx, Δy = 0.01, 0.01
Δt = 0.01

g = construct_grid(xmin:Δx:xmax, ymin:Δy:ymax, tmax, Δt)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

σ, γ = 1.0, 0.10
Γ_p = exp_squared_cov(σ, γ, g.xs, g.ys)
logp_dist = MvNormal(Γ_p)

logps = sample_perms(logp_dist, g.nx, g.ny)
ps = exp.(logps)

u0 = ones(g.nx, g.ny)
u0 = vec(u0)

us = zeros(g.nx, g.ny, g.nt+1)
us[:,:,1] = u0

A = construct_A(g, ps, bcs)

for t ∈ 1:g.nt 

    b = construct_b(g, ps, bcs, us[:,:,t])
    u = solve(LinearProblem(A, b))
    us[:,:,t+1] = reshape(u[g.nu+1:end], g.nx, g.ny)

end

println("Done!")
