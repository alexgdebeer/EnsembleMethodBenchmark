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

const tmin, tmax = 0.0, 15.0
const xmin, xmax = -10.0, 10.0
const ymin, ymax = 0.0, 10.0

const ts = tmin:Δt:tmax
const xs = xmin:Δx:xmax
const ys = ymin:Δy:ymax

ModelingToolkit.@parameters t x y
ModelingToolkit.@variables u(..) vxu(..) vyu(..)

∂t = ModelingToolkit.Differential(t)
∂x = ModelingToolkit.Differential(x)
∂y = ModelingToolkit.Differential(y)
∂x² = ModelingToolkit.Differential(x)^2
∂y² = ModelingToolkit.Differential(y)^2

k(x) = 1.0
ModelingToolkit.@register_symbolic k(x)

function vx(x, y)

    v = 4*((y-ymin)/(ymax-ymin) - 0.5)
    return x > 0 ? v : -v

end

function vy(x, y)

    if x > 0
        return -4(x/xmax-0.5)
    else 
        return -4(x/xmin-0.5)
    end

end

ModelingToolkit.@register_symbolic vx(x, y)
ModelingToolkit.@register_symbolic vy(x, y)

eqs = [
    vxu(t,x,y) ~ vx(x,y)*u(t,x,y),
    vyu(t,x,y) ~ vy(x,y)*u(t,x,y),
    ∂t(u(t,x,y)) + ∂x(vxu(t,x,y)) + ∂y(vyu(t,x,y)) ~ ∂x(∂x(u(t,x,y))) + ∂y(∂y(u(t,x,y)))
]

heat_source(x) = abs(x) < 3 ? 1.0 : 0.0
ModelingToolkit.@register_symbolic heat_source(x)

bcs = [
    u(tmin, x, y) ~ 0.0,
    u(t, xmin, y) ~ 0.0, 
    u(t, xmax, y) ~ 0.0,
    u(t, x, ymin) ~ heat_source(x),
    u(t, x, ymax) ~ 0.0
]

domains = [
    t ∈ DomainSets.Interval(tmin, tmax), 
    x ∈ DomainSets.Interval(xmin, xmax),
    y ∈ DomainSets.Interval(ymin, ymax)
]

@named pde_sys = ModelingToolkit.PDESystem(
    eqs, bcs, domains, 
    [t, x, y], [u(t, x, y), vxu(t, x, y), vyu(t, x, y)]
)

discretization = MethodOfLines.MOLFiniteDifference(
    [x => Δx, y => Δy], t, 
    advection_scheme=MethodOfLines.WENOScheme()
)

println("Discretising...")
prob = @time MethodOfLines.discretize(pde_sys, discretization)

println("Solving...")
sol = @time solve(prob, Tsit5(), saveat=Δt)

solu = sol[u(t, x, y)]
eqn_latex = L"\dot{u} + \nabla \cdot (vu) = \Delta u"

anim = @animate for i = 1:length(ts)
    
    contourf(
        xs, ys, solu[i,:,:]', 
        xlims=extrema(xs), ylims=extrema(ys), clim=extrema(solu),
        linewidth=0, c=:coolwarm, aspect_ratio=:equal, 
        colorbar_title=L"u(x,y,t)", size=(500,280)
    )

    title!(eqn_latex, fontsize=TITLE_SIZE)
    xlabel!(L"x", fontsize=LABEL_SIZE)
    ylabel!(L"y", fontsize=LABEL_SIZE)

end
 
gif(anim, "$PLOTS_DIR/advection_diffusion_variable_v.gif", fps=5)