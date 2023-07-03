using Distributions
using Interpolations

function generate_data(
    g::SteadyStateGrid,
    us::AbstractMatrix,
    x_locs::AbstractVector, 
    y_locs::AbstractVector, 
    ϵ_dist::Distribution
)

    us = interpolate((g.xs, g.ys), us, Gridded(Linear()))

    xs_o = [x for x ∈ x_locs for _ ∈ y_locs]
    ys_o = [y for _ ∈ x_locs for y ∈ y_locs]
    us_o = [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)] + rand(ϵ_dist)

    return xs_o, ys_o, us_o 

end