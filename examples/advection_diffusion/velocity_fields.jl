using Statistics

using Plots
Plots.gr()

# const xmin, Δx, xmax = -10.0, 2.0, 10.0
# const ymin, Δy, ymax = 0.0, 2.0, 20.0

# xs = xmin:Δx:xmax 
# ys = ymin:Δy:ymax 

# n_xs = length(xs)
# n_ys = length(ys)

# xs = xs' .* ones(n_xs) 
# ys = ones(n_ys)' .* ys

# us = zeros(n_xs, n_ys)

# vs = collect(range(2, 0, n_ys)) .* ones(n_xs)'
# vs[1:Int(ceil(n_ys/2)), :] .= 1.0

# println(vs[:,1])

# Plots.quiver(
#     vec(xs), vec(ys),
#     quiver=(vec(us), vec(vs))
# )

# Plots.xlabel!("x")
# Plots.ylabel!("y")

# Plots.savefig("quiver.pdf")

const xmin, xmax, n_xs = -10.0, 10.0, 20
const ymin, ymax, n_ys = 0.0, 10.0, 20

xs = collect(range(xmin, xmax, n_xs))
ys = collect(range(ymin, ymax, n_ys))

xs = xs' .* ones(n_ys) 
ys = ys .* ones(n_xs)'

# Build up a single circular field 

us = ones(Int(n_xs/2))' .* -collect(range(-1, 1, n_ys))
vs = ones(n_ys) .* collect(range(-1, 1, Int(n_xs/2)))'

us = 0.5.*[us -us]
vs = 0.5.*[vs -vs]

Plots.quiver(
    vec(xs), vec(ys),
    quiver=(vec(us), vec(vs)),
    aspect_ratio=:equal
)

Plots.xlims!(xmin, xmax)
Plots.ylims!(ymin, ymax)

Plots.xlabel!("x")
Plots.ylabel!("y")

Plots.savefig("quiver.pdf")