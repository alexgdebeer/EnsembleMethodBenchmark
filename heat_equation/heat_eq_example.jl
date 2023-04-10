using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
∂t = Differential(t)
∂x² = Differential(x)^2

# 1D PDE and boundary conditions
eq = ∂t(u(t, x)) ~ ∂x²(u(t, x))
bcs = [u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, 1) ~ exp(-t) * cos(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
Δx = 0.1
discretization = MOLFiniteDifference([x => Δx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat=0.2)

println(sol[u(t, x)])