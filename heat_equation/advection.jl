"""Solves an advection equation."""

using DomainSets
using MethodOfLines
using ModelingToolkit
using OrdinaryDiffEq

using LaTeXStrings
import PyPlot

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")

const Δt = 0.5
const Δx = 0.1

const tmin, tmax = 0.0, 10.0
const xmin, xmax = -10.0, 10.0

const ts = tmin:Δt:tmax
const xs = xmin:Δx:xmax

@parameters t x
@variables u(..)

∂t = Differential(t)
∂x = Differential(x)
∂x² = Differential(x)^2

eq = ∂t(u(t,x)) ~ ∂x²(u(t,x)) - ∂x(u(t,x))

ic(x) = -π ≤ x ≤ π ? cos(x)+1 : 0
@register_symbolic ic(x)

bcs = [
    u(tmin, x) ~ ic(x),
    u(t, xmin) ~ 0.0, 
    u(t, xmax) ~ 0.0,
]

domains = [
    t ∈ Interval(tmin, tmax), 
    x ∈ Interval(xmin, xmax)
]

@named pde_sys = ModelingToolkit.PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

discretization = MOLFiniteDifference([x => Δx], t)
prob = MethodOfLines.discretize(pde_sys, discretization)

println("Solving...")
sol = solve(prob, Tsit5(), saveat=Δt)
println("Solved.")

solu = sol[u(t, x)]

for (i, us) ∈ enumerate(eachrow(solu))

    PyPlot.plot(xs, us)
    PyPlot.savefig("$(i).pdf")
    PyPlot.clf()

end

# anim = @animate for us ∈ eachrow(solu)
#     plot(xs, us, xlim=(xmin,xmax), ylim=(-1,2))
# end
# gif(anim, "test.gif", fps = 15)