"""Solves for the steady-state pressure profile of a 2D permeability field."""

using DifferentialEquations
using DomainSets
using MethodOfLines
using ModelingToolkit
using NonlinearSolve

using Distributions 
using LinearAlgebra
using Random; Random.seed!(0)

using Plots

Plots.gr()
Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/darcy_flow"

const TITLE_SIZE = 20
const LABEL_SIZE = 16

function exp_squared_cov(σ, γ)

    # Generate vectors of x and y coordinates
    cxs = vec([x for _ ∈ pys, x ∈ pxs])
    cys = vec([y for y ∈ pys, _ ∈ pxs])

    # Generate a matrix of distances between each set of coordinates
    ds = (cxs .- cxs').^2 + (cys .- cys').^2

    Γ = σ^2 * exp.(-(1/2γ^2).*ds) + 1e-6I

    return Γ

end

function sample_logperms(d)
    return reshape(rand(d), length(pxs), length(pxs))
end

Δx = 0.10
Δy = 0.10

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tmin, tmax = 0.0, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

pxs = xmin:0.01:xmax
pys = ymin:0.01:ymax

# Generate some permeability parameters
Γ = exp_squared_cov(1.0, 0.1)
d = MvNormal(Γ)

ModelingToolkit.@register_symbolic k(x,y)
ModelingToolkit.@register_symbolic v(x)

ModelingToolkit.@parameters x y t
ModelingToolkit.@variables u(..)

∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂t = ModelingToolkit.Differential(t)

eq = ∂x(k(x, y)*∂x(u(x, y, t))) + ∂y(k(x, y)*∂y(u(x, y, t))) ~ ∂t(u(x, y, t))

# Define flux on top boundary 
v(x) = 2.0

# Define function to return permeability at a given point
function k(x, y)
    i, j = findmin(abs.(pxs.-x))[2], findmin(abs.(pys.-y))[2]
    return ks[i, j]
end

# Constant pressure on the bottom, Neumann conditions on all other sides
bcs = [
    u(x, y, tmin) ~ 0.0,
    ∂x(u(xmin, y, t)) ~ 0.0, 
    ∂x(u(xmax, y, t)) ~ 0.0, 
    ∂y(u(x, ymax, t)) ~ v(x), 
    u(x, ymin, t) ~ 0.0
]

domains = [
    x ∈ DomainSets.Interval(xmin, xmax), 
    y ∈ DomainSets.Interval(ymin, ymax),
    t ∈ DomainSets.Interval(tmin, tmax)
]

@named darcy_pde = ModelingToolkit.PDESystem(
    [eq], bcs, domains, 
    [x, y, t], [u(x, y, t)]
)

discretization = MethodOfLines.MOLFiniteDifference([x => Δx, y => Δy], t)

ks = exp.(sample_logperms(d))
prob = @time MethodOfLines.discretize(darcy_pde, discretization)
steadystateprob = SteadyStateProblem(prob)
state = @time solve(steadystateprob, DynamicSS(Rodas5()))

heatmap(pxs, pys, logperms, cmap=:viridis)
heatmap(reshape(state, (9, 9)))