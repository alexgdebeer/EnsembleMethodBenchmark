"""Solves for the steady-state pressure profile of a 2D permeability field."""

using DifferentialEquations
using DomainSets
using MethodOfLines
using ModelingToolkit
using NonlinearSolve

using Distributions
using Interpolations
using LinearAlgebra
using Statistics
using Random; Random.seed!(0)

import PyPlot

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

# using Plots

# Plots.gr()
# Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/darcy_flow"

const TITLE_SIZE = 20
const LABEL_SIZE = 14

xmin, Δx, xmax = 0.0, 0.05, 1.0
ymin, Δy, ymax = 0.0, 0.05, 1.0
tmin, tmax = 0.0, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

pxs = xmin:0.01:xmax
pys = ymin:0.01:ymax

n_xs = length(xs)
n_ys = length(ys)

n_pxs = length(pxs)
n_pys = length(pys)

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

# Define function to sample a set of permeabilities
sample_perms(d) = exp.(reshape(rand(d), length(pxs), length(pxs)))

# Define the permeability field and the flux through the top boundary
k(x, y) = ks(x, y)
v(x) = 2.0

ModelingToolkit.@register_symbolic k(x, y)

# Generate a permeability distribution
Γ = exp_squared_cov(0.8, 0.1)
d = MvNormal(Γ)

ModelingToolkit.@parameters x y t
ModelingToolkit.@variables u(..)

∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂t = ModelingToolkit.Differential(t)

eq = ∂x(k(x, y)*∂x(u(x, y, t))) + ∂y(k(x, y)*∂y(u(x, y, t))) ~ ∂t(u(x, y, t))

domains = [
    x ∈ DomainSets.Interval(xmin, xmax), 
    y ∈ DomainSets.Interval(ymin, ymax),
    t ∈ DomainSets.Interval(tmin, tmax)
]

# Constant pressure on the bottom, Neumann conditions on all other sides
bcs = [
    u(x, y, tmin) ~ 0.0,
    ∂x(u(xmin, y, t)) ~ 0.0, 
    ∂x(u(xmax, y, t)) ~ 0.0, 
    ∂y(u(x, ymax, t)) ~ v(x), 
    u(x, ymin, t) ~ 0.0
]

discretization = MOLFiniteDifference([x => Δx, y => Δy], t)

function solve_darcy_pde()

    # Sample a set of permeabilities and form an interpolation object
    global ks = interpolate((pxs, pys), sample_perms(d), Gridded(Linear()))

    darcy_pde = ModelingToolkit.PDESystem(
        [eq], bcs, domains, 
        [x, y, t], [u(x, y, t)],
        name=:darcy_pde
    )

    prob = @time discretize(darcy_pde, discretization)
    steadystateprob = SteadyStateProblem(prob)
    state = @time solve(steadystateprob, DynamicSS(Rodas5()))

    state.retcode != ReturnCode.Success && error("$(state.retcode)")
    ps = reshape(state.u, (n_xs-2, n_ys-2))

    return ks.coefs, ps

end

perms = Array{Float64}(undef, n_pxs, n_pys, 3)
pressures = Array{Float64}(undef, n_xs-2, n_ys-2, 3)

for i ∈ 1:3
    perms[:,:,i], pressures[:,:,i] = solve_darcy_pde()
end

logperms = log.(perms)
logperm_min = minimum(logperms)
logperm_max = maximum(logperms)
pressure_min = minimum(pressures)
pressure_max = maximum(pressures)

fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

for col ∈ 1:3 

    m1 = ax[1, col].pcolormesh(
        pxs, pys, rotr90(logperms[:, :, col]), 
        cmap=:viridis, vmin=logperm_min, vmax=logperm_max
    )
    
    m2 = ax[2, col].pcolormesh(
        xs[2:end-1], ys[2:end-1], rotr90(pressures[:, :, col]), 
        cmap=:coolwarm, vmin=pressure_min, vmax=pressure_max
    )

    for row ∈ 1:2
        ax[row, col].set_box_aspect(1)
        ax[row, col].set_xticks([0, 1])
        ax[row, col].set_yticks([0, 1])
    end

    PyPlot.colorbar(m1, fraction=0.046, pad=0.04, ax=ax[1, col])
    PyPlot.colorbar(m2, fraction=0.046, pad=0.04, ax=ax[2, col])

end

ax[1, 1].set_ylabel("ln(Permeability)", fontsize=LABEL_SIZE)
ax[2, 1].set_ylabel("Pressure", fontsize=LABEL_SIZE)

PyPlot.suptitle("ln(Permeability) and Pressure Fields", fontsize=TITLE_SIZE)
PyPlot.tight_layout()
PyPlot.savefig("$PLOTS_DIR/darcy_flow.pdf")

# heatmap(pxs, pys, rotl90(log.(ks_1)), cmap=:viridis)
# heatmap(xs[2:end-1], ys[2:end-1], rotl90(ps_3))