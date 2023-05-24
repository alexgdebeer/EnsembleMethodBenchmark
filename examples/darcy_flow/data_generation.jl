using LaTeXStrings
using Plots
using PyPlot 

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

include("darcy_flow_ss.jl")

Random.seed!(16)

# Define a fine grid
xmin, Δx, xmax = 0.0, 0.01, 1.0
ymin, Δy, ymax = 0.0, 0.01, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# Set up boundary conditions
x0 = BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = BoundaryCondition(:y0, :neumann, (x, y) -> -2.0)
y1 = BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)
bcs = Dict(:y0 => y0, :y1 => y1, :x0 => x0, :x1 => x1)

# Define a permeability distribution
Γ = @time exp_squared_cov(1.0, 0.2, xs, ys)
d = MvNormal(Γ)

# Sample a set of permeabilities
p = interpolate((xs, ys), sample_perms(d, n_xs, n_ys), Gridded(Linear()))

A, b = generate_grid(xs, ys, Δx, Δy, p, bcs)

prob = LinearProblem(A, b)
sol = solve(prob)

ps = log.(p.coefs)
us = interpolate((xs, ys), reshape(sol.u, n_xs, n_ys), Gridded(Linear()))

# Define a set of observations
x_locs = 0.1:0.2:0.9
y_locs = 0.25:0.1:0.95

xs_o = vec([x for _ ∈ y_locs, x ∈ x_locs])
ys_o = vec([y for y ∈ y_locs, _ ∈ x_locs])
n_obs = length(us_o)

σ_ϵ = 0.1
us_o = [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)] + rand(Normal(0.0, σ_ϵ), n_obs)

# Plots.heatmap(xs, ys, us.coefs', cmap=:coolwarm)
# Plots.scatter!(xs_o, ys_o)