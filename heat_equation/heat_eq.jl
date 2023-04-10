"""Solves the heat equation in one dimension."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using PyPlot

Δt = 0.1
Δx = 0.01

tmin, tmax = 0.0, 1.0
xmin, xmax = 0.0, 1.0

ts = tmin:Δt:tmax
xs = xmin:Δx:xmax

@parameters t x
@parameters c ρ
@variables u(..)

∂t = Differential(t)
∂x² = Differential(x)^2

eq = c*ρ*∂t(u(t, x)) ~ ∂x²(u(t, x))

bcs = [
    u(tmin, x) ~ 20.0, #sin(π*x/2.0), 
    u(t, xmin) ~ 20.0, 
    u(t, xmax) ~ 40.0
]

domains = [t ∈ Interval(tmin, tmax), x ∈ Interval(xmin, xmax)]

@named pde_sys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [c=>1.0, ρ=>1.3])

discretization = MOLFiniteDifference([x => Δx], t)
prob = discretize(pde_sys, discretization)

sol = solve(prob, Tsit5(), saveat=Δt)
solu = sol[u(t, x)]

for (i, t) ∈ enumerate(ts)

    PyPlot.plot(xs, solu[i,:]) 

end

PyPlot.title("Solution to 1D heat equation (Dirichlet BCs)")
PyPlot.xlabel("x")
PyPlot.ylabel("u")

PyPlot.tight_layout()
PyPlot.savefig("plots/heat_eq_1d.pdf")