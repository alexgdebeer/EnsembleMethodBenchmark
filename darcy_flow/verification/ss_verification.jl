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

using SimIntensiveInference

include("../setup/setup.jl")

# Random.seed!(0)

xmin, Δx, xmax = 0.0, 0.05, 1.0
ymin, Δy, ymax = 0.0, 0.05, 1.0
tmin, tmax = 0.0, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

grid = SteadyStateGrid(xs, ys)

# ----------------
# Define the permeability field 
# ----------------

# Generate a permeability distribution
k = ARDExpSquaredKernel(0.8, 0.1, 0.1)
prior = GaussianPrior(0.0, grid.xs, grid.ys, k)

ps = interpolate(
    (xs, ys), 
    exp.(reshape(rand(prior), grid.nx, grid.ny)), 
    Gridded(Linear())
)

p(x, y) = ps(x, y)

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
    p(x,y) * ∂x(u(grid.xmin, y, t)) ~ 0.0, 
    p(x,y) * ∂x(u(grid.xmax, y, t)) ~ 0.0, 
    p(x,y) * ∂y(u(x, grid.ymin, t)) ~ -2.0, 
    u(x, grid.ymax, t) ~ 0.0
]

darcy_pde = ModelingToolkit.PDESystem(
    [eq], mol_bcs, domains, 
    [x, y, t], [u(x, y, t)],
    name=:darcy_pde
)

discretization = MOLFiniteDifference([x => grid.Δx, y => grid.Δy], t)
prob = MethodOfLines.discretize(darcy_pde, discretization)
steadystateprob = SteadyStateProblem(prob)
state = solve(steadystateprob, DynamicSS(Rodas5()))

state.retcode != ReturnCode.Success && error("$(state.retcode)")
us_mol = reshape(state.u, (grid.nx-2, grid.ny-2))

# ----------------
# Solve with finite differences code
# ----------------

fd_bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 2.0), 
    :y1 => BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
)

us_fd = solve(grid, ps.coefs, fd_bcs)

println(maximum(abs.(us_fd[2:end-1, 2:end-1] - us_mol) ./ us_mol))