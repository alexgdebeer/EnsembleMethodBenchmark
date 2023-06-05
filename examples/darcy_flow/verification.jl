using DifferentialEquations
using DomainSets
using MethodOfLines
using ModelingToolkit
using NonlinearSolve

using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Random
using Statistics

using DarcyFlow

# Random.seed!(0)

xmin, Δx, xmax = 0.0, 0.05, 1.0
ymin, Δy, ymax = 0.0, 0.05, 1.0
tmin, tmax = 0.0, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

grid = DarcyFlow.construct_grid(xs, ys)

# ----------------
# Define the permeability field 
# ----------------

# Generate a permeability distribution
Γ = DarcyFlow.exp_squared_cov(0.8, 0.1, grid.xs, grid.ys)
d = MvNormal(Γ)

ps = interpolate(
    (xs, ys), 
    exp.(DarcyFlow.sample_perms(d, grid.nx, grid.ny)), 
    Gridded(Linear())
)

p(x, y) = ps(x, y)
#p(x, y) = 2.0 # x^2+y+1.0

# ----------------
# Solve using MethodOfLines.jl 
# ----------------

ModelingToolkit.@register_symbolic p(x, y)

ModelingToolkit.@parameters x y t
ModelingToolkit.@variables u(..)

∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂t = ModelingToolkit.Differential(t)

eq = ∂x(p(x, y)*∂x(u(x, y, t))) + ∂y(p(x, y)*∂y(u(x, y, t))) ~ ∂t(u(x, y, t))

domains = [
    x ∈ Interval(xmin, xmax), 
    y ∈ Interval(ymin, ymax),
    t ∈ Interval(tmin, tmax)
]

mol_bcs = [
    u(x, y, tmin) ~ 0.0,
    ∂x(u(grid.xmin, y, t)) ~ 0.0, 
    ∂x(u(grid.xmax, y, t)) ~ 0.0, 
    ∂y(u(x, grid.ymin, t)) ~ -2.0, 
    u(x, grid.ymax, t) ~ 0.0
]

discretization = MOLFiniteDifference([x => grid.Δx, y => grid.Δy], t)

darcy_pde = ModelingToolkit.PDESystem(
    [eq], mol_bcs, domains, 
    [x, y, t], [u(x, y, t)],
    name=:darcy_pde
)

prob = discretize(darcy_pde, discretization)
steadystateprob = SteadyStateProblem(prob)
state = solve(steadystateprob, DynamicSS(Rodas5()))

state.retcode != ReturnCode.Success && error("$(state.retcode)")
us_mol = reshape(state.u, (grid.nx-2, grid.ny-2))

# ----------------
# Solve with DarcyFlow.jl
# ----------------

fd_bcs = Dict(
    :x0 => DarcyFlow.BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => DarcyFlow.BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => DarcyFlow.BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => DarcyFlow.BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

A = DarcyFlow.construct_A(grid, ps.coefs, fd_bcs)
b = DarcyFlow.construct_b(grid, ps.coefs, fd_bcs)
us_fd = reshape(solve(LinearProblem(A, b)), grid.nx, grid.ny)

println(maximum(abs.(us_fd[2:end-1, 2:end-1] - us_mol) ./ us_mol))
# us_fd = reverse(us_fd, dims=1)