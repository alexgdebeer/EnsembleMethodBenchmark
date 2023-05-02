using LaTeXStrings
using Plots
using Statistics

Plots.gr()
Plots.default(fontfamily="Computer Modern")

const PLOTS_DIR = "plots/advection_diffusion"

const TITLE_SIZE = 20
const LABEL_SIZE = 16
const SMALL_SIZE = 8

const xmin, xmax, n_xs = -10.0, 10.0, 10
const ymin, ymax, n_ys = 0.0, 10.0, 10

xs = collect(range(xmin, xmax, n_xs))
ys = collect(range(ymin, ymax, n_ys))

xs = xs' .* ones(n_ys) 
ys = ys .* ones(n_xs)'

# Build up a single circular field 

us = ones(Int(n_xs/2))' .* -collect(range(-1, 1, n_ys))
vs = ones(n_ys) .* collect(range(-1, 1, Int(n_xs/2)))'

us = 1.0.*[us -us]
vs = 1.0.*[vs -vs]

Plots.quiver(
    vec(xs), vec(ys),
    quiver=(vec(us), vec(vs)),
    aspect_ratio=:equal, arrow = :closed
)

Plots.xlims!(xmin, xmax)
Plots.ylims!(ymin, ymax)

Plots.title!("Velocity Field", fontsize=TITLE_SIZE)
Plots.xlabel!(L"x", fontsize=LABEL_SIZE)
Plots.ylabel!(L"y", fontsize=LABEL_SIZE)

Plots.savefig("$PLOTS_DIR/velocity_field.pdf")