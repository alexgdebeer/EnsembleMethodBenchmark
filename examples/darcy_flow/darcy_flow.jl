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

global ks

"""Generates an exponential-squared covariance matrix with a given standard 
deviation and length-scale."""
function exp_squared_cov(σ, γ)

    # Generate vectors of x and y coordinates
    cxs = vec([x for _ ∈ pys, x ∈ pxs])
    cys = vec([y for y ∈ pys, _ ∈ pxs])

    # Generate a matrix of distances between each set of coordinates
    ds = (cxs .- cxs').^2 + (cys .- cys').^2

    Γ = σ^2 * exp.(-(1/2γ^2).*ds) + 1e-6I

    return Γ

end

"""Samples a set of log-normally distributed permeabilities."""
function sample_perms(d)
    return exp.(reshape(rand(d), length(pxs), length(pxs)))
end

"""Returns the value of the permeability field at a given point."""
function k(x, y)
    i, j = findmin(abs.(pxs.-x))[2], findmin(abs.(pys.-y))[2]
    return ks[i, j]
end

xmin, Δx, xmax = 0.0, 0.05, 1.0
ymin, Δy, ymax = 0.0, 0.05, 1.0
tmin, tmax = 0.0, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)

pxs = xmin:0.01:xmax
pys = ymin:0.01:ymax

ModelingToolkit.@register_symbolic k(x,y)
ModelingToolkit.@register_symbolic v(x)

function solve_darcy_pde()

    ModelingToolkit.@parameters x y t
    ModelingToolkit.@variables u(..)

    ∂x = ModelingToolkit.Differential(x)
    ∂y = ModelingToolkit.Differential(y)
    ∂t = ModelingToolkit.Differential(t)

    # Generate some permeability parameters
    Γ = exp_squared_cov(1.0, 0.1)
    d = MvNormal(Γ)

    # Define flux on top boundary 
    v(x) = 2.0

    eq = ∂x(k(x, y)*∂x(u(x, y, t))) + ∂y(k(x, y)*∂y(u(x, y, t))) ~ ∂t(u(x, y, t))

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

    global ks = sample_perms(d)

    @named darcy_pde = ModelingToolkit.PDESystem(
        [eq], bcs, domains, 
        [x, y, t], [u(x, y, t)]
    )

    discretization = MOLFiniteDifference([x => Δx, y => Δy], t)
    prob = @time discretize(darcy_pde, discretization)
    steadystateprob = SteadyStateProblem(prob)
    state = @time solve(steadystateprob, DynamicSS(Rodas5()))

    state.retcode != ReturnCode.Success && error("$(state.retcode)")

    ps = reshape(state.u, (length(xs)-2, length(ys)-2))

    return ps

end

ps_1 = solve_darcy_pde()
ps_2 = solve_darcy_pde()
ps_3 = solve_darcy_pde()

heatmap(pxs, pys, log.(ks), cmap=:viridis)
heatmap(xs[2:end-1], ys[2:end-1], ps_3)