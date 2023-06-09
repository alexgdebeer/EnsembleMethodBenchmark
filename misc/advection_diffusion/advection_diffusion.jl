"""Solves the heat equation in two dimensions."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using Distributions

using LaTeXStrings
using Plots

Plots.gr()
Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/advection_diffusion"

const TITLE_SIZE = 20
const LABEL_SIZE = 16
const SMALL_SIZE = 8

const Δt = 0.25
const Δx = 1.0
const Δy = 1.0

const tmin, tmax = 0.0, 2.0
const xmin, xmax = -10.0, 10.0
const ymin, ymax = -10.0, 10.0

const ts = tmin:Δt:tmax
const xs = xmin:Δx:xmax
const ys = ymin:Δy:ymax

ModelingToolkit.@parameters t x y
ModelingToolkit.@variables u(..) 
ModelingToolkit.@constants c=500.0 ρ=8000.0

∂t = ModelingToolkit.Differential(t)
∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂x² = ModelingToolkit.Differential(x)^2
∂y² = ModelingToolkit.Differential(y)^2

k(x) = 1.0
ModelingToolkit.@register_symbolic k(x)

vx(x, y) = 2.0
vy(x, y) = 2.0

ModelingToolkit.@register_symbolic vx(x, y)
ModelingToolkit.@register_symbolic vy(x, y)

eqs = [
    ∂t(u(t,x,y)) ~ ∂x²(u(t,x,y)) + ∂y²(u(t,x,y))
]

ic(x, y) = x^2 + y^2 ≤ π^2 ? 0.25(cos(x)+1)*(cos(y)+1) : 0.0
ModelingToolkit.@register_symbolic ic(x, y)

bcs = [
    u(tmin, x, y) ~ ic(x, y),
    u(t, xmin, y) ~ 0.0, 
    u(t, xmax, y) ~ 0.0,
    u(t, x, ymin) ~ 0.0,
    u(t, x, ymax) ~ 0.0
]

domains = [
    t ∈ DomainSets.Interval(tmin, tmax), 
    x ∈ DomainSets.Interval(xmin, xmax),
    y ∈ DomainSets.Interval(ymin, ymax)
]

@named pde_sys = ModelingToolkit.PDESystem(
    eqs, bcs, domains, 
    [t, x, y], [u(t, x, y)]
)

discretization = MethodOfLines.MOLFiniteDifference(
    [x => Δx, y => Δy], t, 
    advection_scheme=MethodOfLines.WENOScheme()
)

println("Discretising...")
prob = MethodOfLines.discretize(pde_sys, discretization)

println("Solving...")
sol = @time solve(prob, Tsit5(), saveat=Δt, progress=true)

solu = sol[u(t, x, y)]

maxu = maximum(solu[1,:,:])
eqn_latex = L"\dot{u} + v \cdot \nabla u = \Delta u"

anim = @animate for i = 1:length(ts)
    
    contourf(
        xs, ys, solu[i,:,:], 
        xlims=extrema(xs), ylims=extrema(ys), clim=extrema(solu),
        linewidth=0, c=:coolwarm, aspect_ratio=:equal, 
        colorbar_title=L"u(x,y,t)"
    )

    title!(eqn_latex, fontsize=TITLE_SIZE)
    xlabel!(L"x", fontsize=LABEL_SIZE)
    ylabel!(L"y", fontsize=LABEL_SIZE)

end
 
gif(anim, "$PLOTS_DIR/advection_diffusion.gif", fps=5)