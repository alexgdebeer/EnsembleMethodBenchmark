"""Solves the heat equation in one dimension."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using LaTeXStrings
using PyPlot

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")

Δt = 200.0
Δx = 0.01

tmin, tmax = 0.0, 1000.0
xmin, xmax = 0.0, 0.5

ts = tmin:Δt:tmax
xs = xmin:Δx:xmax

@parameters t x
@variables u(..)
@constants c=500.0 ρ=8000.0

∂t = Differential(t)
∂x = Differential(x)
∂x² = Differential(x)^2

k(x) = x ≤ 0.5xmax ? 50.0 : 200.0
@register_symbolic k(x)

eq = c*ρ*∂t(u(t, x)) ~ ∂x(k(x) * ∂x(u(t, x)))

# Starts at 300 K
bcs = [
    u(tmin, x) ~ 300.0,
    u(t, xmin) ~ 500.0, 
    u(t, xmax) ~ 500.0
]

domains = [t ∈ Interval(tmin, tmax), x ∈ Interval(xmin, xmax)]

@named pde_sys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

discretization = MOLFiniteDifference([x => Δx], t)
prob = discretize(pde_sys, discretization)

println("Solving...")
sol = solve(prob, Tsit5(), saveat=Δt)
println("Solved.")

solu = sol[u(t, x)]

for (i, t) ∈ enumerate(ts)
    PyPlot.plot(xs, solu[i,:], label=L"t ="*" $(Int(round(t))) s") 
end

PyPlot.title("Solution to 1D heat equation (Dirichlet BCs)", fontsize=20)
PyPlot.xlabel(L"x"*" (m)", fontsize=16)
PyPlot.ylabel(L"u(x, t)"*" (K)", fontsize=16)
PyPlot.legend(loc="lower right")

PyPlot.tight_layout()
PyPlot.savefig("plots/heat_eq_1d_variable_cond.pdf")