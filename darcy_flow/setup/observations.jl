using SparseArrays

function build_observation_operator(
    g::Grid,
    x_obs::AbstractVector,
    y_obs::AbstractVector
)

    function get_cell_index(g::Grid, xi::Int, yi::Int)
        return xi + g.nx * yi # TODO: check this...
    end
        
    n_obs = length(x_obs)

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_obs, y_obs))

        ix0 = findfirst(g.xs.>x) - 1
        iy0 = findfirst(g.xs.>y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [get_cell_index(g, i...) for i ∈ inds]

        Z = (x1-x0) * (y1-y0)

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / Z, 
            (x-x0) * (y1-y) / Z, 
            (x1-x) * (y-y0) / Z, 
            (x-x0) * (y-y0) / Z
        )

    end

    return sparse(is, js, vs, n_obs, g.nx^2)

end