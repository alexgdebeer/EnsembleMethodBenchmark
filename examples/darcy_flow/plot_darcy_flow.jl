using LaTeXStrings
using Plots
using PyPlot 

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

include("darcy_flow_ss.jl")

Random.seed!(16)

# Define the grid dimensions
xmin, Δx, xmax = 0.0, 0.01, 1.0
ymin, Δy, ymax = 0.0, 0.01, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# Define permeability distribution
Γ = exp_squared_cov(0.8, 0.1, xs, ys)
d = MvNormal(Γ)

# Set up boundary conditions
x0 = BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = BoundaryCondition(:y0, :neumann, (x, y) -> -2.0)
y1 = BoundaryCondition(:y1, :dirichlet, (x, y) -> 0.0)

# Define a mapping between boundary condition names and objects
bcs = Dict(:y0 => y0, :y1 => y1, :x0 => x0, :x1 => x1)

n_sims = 3

us = zeros(n_xs, n_ys, n_sims)
ps = zeros(n_xs, n_ys, n_sims)

@time for i ∈ 1:n_sims 

    # Generate a permeability field
    p = interpolate((xs, ys), sample_perms(d), Gridded(Linear()))

    A, b = generate_grid(xs, ys, Δx, Δy, p, bcs)

    prob = LinearProblem(A, b)
    sol = solve(prob)

    ps[:,:,i] = reshape(log.(p.coefs), n_xs, n_ys)
    us[:,:,i] = reshape(sol.u, n_xs, n_ys)

end

p_min = minimum(ps)
p_max = maximum(ps)
u_min = minimum(us)
u_max = maximum(us)

fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

for col ∈ 1:3

    m1 = ax[1, col].pcolormesh(
        xs, ys, rotr90(ps[:, :, col]), 
        cmap=:viridis, vmin=p_min, vmax=p_max
    )
    
    m2 = ax[2, col].pcolormesh(
        xs, ys, rotr90(us[:, :, col]), 
        cmap=:coolwarm, vmin=u_min, vmax=u_max
    )

    for row ∈ 1:2
        ax[row, col].set_box_aspect(1)
        ax[row, col].set_xticks([0, 1])
        ax[row, col].set_yticks([0, 1])
    end

    PyPlot.colorbar(m1, fraction=0.046, pad=0.04, ax=ax[1, col])
    PyPlot.colorbar(m2, fraction=0.046, pad=0.04, ax=ax[2, col])

end

ax[1, 1].set_ylabel("ln(Permeability)", fontsize=14)
ax[2, 1].set_ylabel("Pressure", fontsize=14)

PyPlot.suptitle("ln(Permeability) and Pressure Fields", fontsize=20)
PyPlot.tight_layout()
PyPlot.savefig("plots/darcy_flow/variable_permeability.pdf")

# heatmap(xs, ys, rotr90(us[:,:,1]))

# fig, ax = PyPlot.subplots(figsize=(5, 4))

# m = ax.pcolormesh(
#     xs, ys, rotr90(reshape(sol.u, n_xs, n_ys)), 
#     cmap=:coolwarm
# )

# ax.set_box_aspect(1)
# ax.set_xticks([0, 1])
# ax.set_yticks([0, 1])

# PyPlot.colorbar(m, ax=ax)

# PyPlot.title("Pressure profile (constant permeability)\n", fontsize=14)
# PyPlot.xlabel(L"x", fontsize=12)
# PyPlot.ylabel(L"y", fontsize=12)

# PyPlot.tight_layout()
# PyPlot.savefig("plots/darcy_flow/uniform_permeability.pdf")