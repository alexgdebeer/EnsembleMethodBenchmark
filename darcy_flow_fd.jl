using Distributions
using Interpolations
using LinearAlgebra
using LinearSolve
using Plots
using PyPlot
using Random
using SparseArrays

Random.seed!(64)

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

struct BoundaryCondition
    name::Symbol
    type::Symbol
    func::Function
end

function in_corner(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] && y ∈ [ymin, ymax]
end

function on_boundary(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] || y ∈ [ymin, ymax]
end

function get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)

    x == xmin && return bcs[:x0]
    x == xmax && return bcs[:x1]
    y == ymin && return bcs[:y0]
    y == ymax && return bcs[:y1]

    error("Point ($x, $y) is not on a boundary.")

end

function add_corner_point!(rs, cs, vs, i)

    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)

    if bc.type == :dirichlet 
        add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)
    elseif bc.type == :neumann 
        add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)
    end

end

function add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)

    b[i] = bc.func(x, y)
    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)

    b[i] = bc.func(x, y)

    push!(rs, i, i, i)

    if bc.name == :y0 
        push!(cs, i, i+n_xs, i+2nxs)
        push!(vs, -3.0 / 2Δy, 4.0 / 2Δy, -1.0 / 2Δy)
    elseif bc.name == :y1 
        push!(cs, i-2n_xs, i-n_xs, i)
        push!(vs, -3.0 / 2Δy, 4.0 / 2Δy, -1.0 / 2Δy)
    elseif bc.name == :x0 
        push!(cs, i, i+1, i+2)
        push!(vs, -3.0 / 2Δx, 4.0 / 2Δx, -1.0 / 2Δx)
    elseif bc.name == :x1 
        push!(cs, i-2, i-1, i)
        push!(vs, -3.0 / 2Δx, 4.0 / 2Δx, -1.0 / 2Δx)
    end

end

function add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+n_xs, i-n_xs)
    
    push!(
        vs,
        -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
        (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y)) / Δx^2,
        (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y)) / Δx^2,
        (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y)) / Δy^2,
        (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y)) / Δy^2
    )

end

function generate_grid(xs, ys, Δx, Δy, p, bcs)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    n_xs = length(xs)
    n_ys = length(ys)
    n_us = n_xs * n_ys

    # Initalise components of A
    rs = Int64[]
    cs = Int64[]
    vs = Float64[]

    # TODO: make b sparse?
    b = zeros(n_us)

    for i ∈ 1:n_us 

        # Find the x and y coordinates of the current point
        x = xs[(i-1)%n_xs+1] 
        y = ys[Int(ceil(i/n_xs))]

        if in_corner(x, y, xmin, xmax, ymin, ymax)

            add_corner_point!(rs, cs, vs, i)

        elseif on_boundary(x, y, xmin, xmax, ymin, ymax)

            bc = get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)
            add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)
        
        end

    end

    A = sparse(rs, cs, vs, n_us, n_us)

    return A, b

end

function exp_squared_cov(σ, γ, xs, ys)

    # Generate vectors of x and y coordinates
    cxs = vec([x for _ ∈ ys, x ∈ xs])
    cys = vec([y for y ∈ ys, _ ∈ xs])

    # Generate a matrix of distances between each set of coordinates
    ds = (cxs .- cxs').^2 + (cys .- cys').^2

    Γ = σ^2 * exp.(-(1/2γ^2).*ds) + 1e-6I

    return Γ

end

function sample_perms(d) 
    return exp.(reshape(rand(d), length(xs), length(ys)))
end

# Define the grid dimensions
xmin, Δx, xmax = 0.0, 0.01, 1.0
ymin, Δy, ymax = 0.0, 0.01, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# Define permeability distribution
Γ = exp_squared_cov(1.0, 0.2, xs, ys)
d = MvNormal(Γ)

# Set up boundary conditions
x0 = BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = BoundaryCondition(:y0, :dirichlet, (x, y) -> 0.0)
y1 = BoundaryCondition(:y1, :neumann, (x, y) -> 2.0)

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
PyPlot.savefig("test.pdf")

heatmap(xs, ys, rotr90(us[:,:,1]))