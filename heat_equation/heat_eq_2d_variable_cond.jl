"""Solves the heat equation in one dimension."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using Distributions

using LaTeXStrings
using PyPlot

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")

Δt = 20.0
Δx = 0.05
Δy = 0.05

tmin, tmax = 0.0, 100.0
xmin, xmax = 0.0, 0.5
ymin, ymax = 0.0, 0.5

ts = tmin:Δt:tmax
xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

@parameters t x y
@variables u(..) vxu(..) vyu(..)
@constants c=500.0 ρ=8000.0

∂t = Differential(t)
∂x = Differential(x)
∂y = Differential(y)
∂x² = Differential(x)^2
∂y² = Differential(y)^2

k(x) = 200.0#x ≤ 0.5xmax ? 50.0 : 200.0
@register_symbolic k(x)

vx(x, y) = x
@register_symbolic vx(x, y)

vy(x, y) = 1.0
@register_symbolic vy(x, y)

eqs = [
    vxu(t,x,y) ~ vx(x,y)*u(t,x,y),
    vyu(t,x,y) ~ vy(x,y)*u(t,x,y),
    c*ρ*∂t(u(t,x,y)) ~ ∂x(k(x) * ∂x(u(t,x,y))) + ∂y(k(x) * ∂y(u(t,x,y))) + ∂x(vxu(t,x,y)) + ∂y(vyu(t,x,y))
]

bcs = [
    u(tmin, x, y) ~ 300.0,
    u(t, xmin, y) ~ 500.0, 
    u(t, xmax, y) ~ 500.0,
    u(t, x, ymin) ~ 500.0,
    u(t, x, ymax) ~ 500.0
]

domains = [
    t ∈ Interval(tmin, tmax), 
    x ∈ Interval(xmin, xmax),
    y ∈ Interval(ymin, ymax)
]

@named pde_sys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), vxu(t,x,y), vyu(t,x,y)])

discretization = MOLFiniteDifference([x => Δx, y => Δy], t)
prob = discretize(pde_sys, discretization)

println("Solving...")
sol = solve(prob, Tsit5(), saveat=Δt)
println("Solved.")

solu = sol[u(t, x, y)]

# Hack?
solu[:,1,1] .= 500.0
solu[:,1,end] .= 500.0
solu[:,end,1] .= 500.0
solu[:,end,end] .= 500.0

println(size(solu))

for (i, t) ∈ enumerate(ts)

    println(solu[i,:,:])

    PyPlot.contourf(xs, ys, solu[i,:,:], cmap="coolwarm")
    
    PyPlot.title("Time: $(Int(round(t))) s")
    PyPlot.xlabel(L"x"*" (m)", fontsize=16)
    PyPlot.ylabel(L"y"*" (m)", fontsize=16)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/time_$(i).pdf")
    PyPlot.clf()

end

# for (i, t) ∈ enumerate(ts)
#     PyPlot.plot(xs, solu[i,:,:], label=L"t ="*" $(Int(round(t))) s") 
# end

# PyPlot.title("Solution to 1D heat equation (Dirichlet BCs)", fontsize=20)
# PyPlot.xlabel(L"x"*" (m)", fontsize=16)
# PyPlot.ylabel(L"u(x, t)"*" (K)", fontsize=16)
# PyPlot.legend(loc="lower right")

# PyPlot.tight_layout()
# PyPlot.savefig("plots/heat_eq_2d_variable_cond.pdf")