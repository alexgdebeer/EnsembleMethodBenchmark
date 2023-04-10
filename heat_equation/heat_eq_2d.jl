"""Solves the heat equation in two dimensions."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using PyPlot

Δt = 0.1
Δx = 0.05
Δy = 0.05

tmin, tmax = 0.0, 1.0
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

ts = tmin:Δt:tmax
xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

@parameters t x y
@variables u(..)

∂t = Differential(t)
∂x² = Differential(x)^2
∂y² = Differential(y)^2

eq = ∂t(u(t, x, y)) ~ ∂x²(u(t, x, y)) + ∂y²(u(t, x, y))

bcs = [
    u(tmin, x, y) ~ 0.0, #sin(π*x/2),
    u(t, xmin, y) ~ 0.0,
    u(t, xmax, y) ~ 1.0,
    u(t, x, ymin) ~ x,
    u(t, x, ymax) ~ x
]

domains = [
    t ∈ Interval(tmin, tmax), 
    x ∈ Interval(xmin, xmax),
    y ∈ Interval(ymin, ymax)
]

@named pde_sys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

discretization = MOLFiniteDifference([x => Δx, y => Δy], t)

prob = discretize(pde_sys, discretization)

sol = solve(prob, Tsit5(), saveat=Δt)
solu = sol[u(t, x, y)]

println(solu[1,:,:])

for (i, t) ∈ enumerate(ts)

    PyPlot.contourf(xs, ys, solu[i,:,:], cmap="coolwarm")
    
    PyPlot.title("Time: $(t)")
    PyPlot.xlabel("x")
    PyPlot.ylabel("y")

    PyPlot.tight_layout()
    PyPlot.savefig("plots/time_$(i).pdf")
    PyPlot.clf()

end