"""Solves the heat equation in two dimensions."""

using DifferentialEquations
using DomainSets
using MethodOfLines
using ModelingToolkit
using NonlinearSolve

using LaTeXStrings
using Plots

Plots.gr()
Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/advection_diffusion"

const TITLE_SIZE = 20
const LABEL_SIZE = 16
const SMALL_SIZE = 8

const Δx = 1.0
const Δy = 1.0

const xmin, xmax = -10.0, 10.0
const ymin, ymax = 0.0, 10.0

const xs = xmin:Δx:xmax
const ys = ymin:Δy:ymax

ModelingToolkit.@parameters x y
ModelingToolkit.@variables u(..) vxu(..) vyu(..)

∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂x² = ModelingToolkit.Differential(x)^2
∂y² = ModelingToolkit.Differential(y)^2

# Define variable velocity field
function vx(x, y)
    v = 4*((y-ymin)/(ymax-ymin) - 0.5)
    return x > 0 ? v : -v
end

function vy(x, y)
    return x > 0 ? -4(x/xmax-0.5) : -4(x/xmin-0.5)
end

k(x) = 1.0
heat_source(x) = abs(x) < 3 ? -2.0 : -0.5

ModelingToolkit.@register_symbolic vx(x, y)
ModelingToolkit.@register_symbolic vy(x, y)

ModelingToolkit.@register_symbolic heat_source(x)
ModelingToolkit.@register_symbolic k(x)

eqs = [
    vxu(x,y) ~ vx(x,y)*u(x,y),
    vyu(x,y) ~ vy(x,y)*u(x,y),
    ∂x(vxu(x,y)) + ∂y(vyu(x,y)) ~ ∂x(∂x(u(x,y))) + ∂y(∂y(u(x,y)))
]

bcs = [
    u(xmin, y) ~ 0.0, 
    u(xmax, y) ~ 0.0,
    ∂y(u(x, ymin)) ~ heat_source(x),
    u(x, ymax) ~ 0.0
]

domains = [
    x ∈ DomainSets.Interval(xmin, xmax), 
    y ∈ DomainSets.Interval(ymin, ymax)
]

@named pde_sys = ModelingToolkit.PDESystem(
    eqs, bcs, domains, 
    [x, y], [u(x, y), vxu(x, y), vyu(x, y)]
)

discretization = MethodOfLines.MOLFiniteDifference(
    [x => Δx, y => Δy], nothing, 
    # advection_scheme=MethodOfLines.WENOScheme()
)

println("Discretising...")
prob = @time MethodOfLines.discretize(pde_sys, discretization)

println("Solving...")
sol = @time NonlinearSolve.solve(prob, DifferentialEquations.NewtonRaphson())

println(sol)

#steadystateprob = SteadyStateProblem(prob)
#steadystate = solve(steadystateprob, DynamicSS(Tsit5()))

# sol = @time DifferentialEquations.solve(prob, saveat=Δt)

# solu = sol[u(t, x, y)]
# eqn_latex = L"\dot{u} + \nabla \cdot (vu) = \Delta u"

# anim = @animate for i = 1:length(ts)
    
#     contourf(
#         xs, ys, solu[i,:,:]', 
#         xlims=extrema(xs), ylims=extrema(ys), clim=extrema(solu),
#         linewidth=0, c=:coolwarm, aspect_ratio=:equal, 
#         colorbar_title=L"u(x,y,t)", size=(500,280)
#     )

#     title!(eqn_latex, fontsize=TITLE_SIZE)
#     xlabel!(L"x", fontsize=LABEL_SIZE)
#     ylabel!(L"y", fontsize=LABEL_SIZE)

# end
 
# gif(anim, "$PLOTS_DIR/advection_diffusion_variable_v.gif", fps=5)