"""Solves for the steady-state pressure profile of a 2D permeability field."""

using DifferentialEquations
using DomainSets
using MethodOfLines
using ModelingToolkit
using NonlinearSolve

using Distributions 
using LinearAlgebra

using LaTeXStrings
using Plots

Plots.gr()
Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/darcy_flow"

const TITLE_SIZE = 20
const LABEL_SIZE = 16

const Δx = 0.05
const Δy = 0.05

const xmin, xmax = 0.0, 1.0
const ymin, ymax = 0.0, 1.0

const xs = xmin:Δx:xmax
const ys = ymin:Δy:ymax

const pxs = 0.0:0.01:1.0
const pys = 0.0:0.01:1.0


function exp_squared_cov(σ, γ)

    # Generate lists of x and y coordinates
    cxs = vec([x for _ ∈ pys, x ∈ pxs])
    cys = vec([y for y ∈ pys, _ ∈ pxs])

    # Generate a matrix of distances between each set of coordinates
    dxs = cxs .- cxs'
    dys = cys .- cys'
    ds = dxs.^2 + dys.^2

    # Add a small perturbation to the covariance matrix to ensure it is 
    # positive definite
    Γ = σ^2 * exp.(-(1/2γ^2).*ds) + 1e-6I

    return Γ

end

function sample_logperms(d)

    return reshape(rand(d), length(pxs), length(pxs))

end

# Generate some permeability parameters
Γ = exp_squared_cov(1.0, 0.1)
d = MvNormal(Γ)

ModelingToolkit.@register_symbolic k(x,y)
ModelingToolkit.@register_symbolic v(x)

ModelingToolkit.@parameters x y
ModelingToolkit.@variables u(..)

∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)

# Define flux on top boundary 
v(x) = 2.0

eqs = [-∂x(k(x,y)*∂x(u(x, y))) - ∂y(k(x,y)*∂y(u(x, y))) ~ 0]

# Constant pressure on the top, Neumann conditions on all other sides
bcs = [
    ∂x(u(xmin, y)) ~ 0.0, 
    ∂x(u(xmax, y)) ~ 0.0, 
    ∂y(u(x, ymax)) ~ v(x), 
    ∂y(u(x, ymin)) ~ 0.0
]

# Define function to return permeability at a given point
function k(x, y)
    i, j = findmin(abs.(pxs.-x))[2], findmin(abs.(pys.-y))[2]
    return ks[i, j]
end

domains = [
    x ∈ DomainSets.Interval(xmin, xmax), 
    y ∈ DomainSets.Interval(ymin, ymax)
]

@named pde_sys = ModelingToolkit.PDESystem(
    eqs, bcs, domains, 
    [x, y], [u(x, y)]
)

ks = sample_logperms(d)

Plots.heatmap(pxs, pys, ks)
Plots.savefig("$PLOTS_DIR/permeabilities.pdf")
ks = exp.(ks)

# bcs = [
#     u(xmin, y) ~ 0.0, 
#     u(xmax, y) ~ 0.0, 
#     u(x, ymax) ~ 0.0,
#     u(x, ymin) ~ 0.0
# ]

discretization = MethodOfLines.MOLFiniteDifference(
    [x => Δx, y => Δy], nothing, 
    approx_order=2
)

@info "Discretising..."
prob = @time MethodOfLines.discretize(pde_sys, discretization)

println(prob.u0)

@info "Solving..."
sol = @time NonlinearSolve.solve(prob, NewtonRaphson())

solu = sol[u(x,y)]
contourf(xs, ys, solu)
savefig("$PLOTS_DIR/pressures.pdf")

println(solu)

# newprob = @time remake(prob, p=[0.8])
# sol = @time NonlinearSolve.solve(newprob, NewtonRaphson(), progress=true, progress_steps=1)