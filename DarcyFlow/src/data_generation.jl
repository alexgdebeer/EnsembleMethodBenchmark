function exp_squared_cov(
    σ::Real, 
    γ::Real, 
    xs::AbstractVector, 
    ys::AbstractVector
)::AbstractMatrix

    # Generate vectors of x and y coordinates
    cxs = [x for _ ∈ ys for x ∈ xs]
    cys = [y for y ∈ ys for _ ∈ xs]

    # Generate a matrix of distances between each set of coordinates
    ds = (cxs .- cxs').^2 + (cys .- cys').^2

    Γ = σ^2 * exp.(-(1/2γ^2) * ds) + 1e-6I

    return Γ

end

function sample_perms(
    d::Distribution, 
    nx::Int, 
    ny::Int
)::AbstractMatrix

    return reshape(rand(d), nx, ny)

end

function generate_data(
    g::Grid,
    x_locs::AbstractVector, 
    y_locs::AbstractVector, 
    bcs::Dict{Symbol, BoundaryCondition}, 
    logp_dist::Distribution, 
    ϵ_dist::Distribution
)

    # Sample a permeability field
    logps = sample_perms(logp_dist, g.nx, g.ny)
    ps = exp.(logps)
    
    # Generate the steady-state problem
    g = construct_grid(g.xs, g.ys)
    A = construct_A(g, ps, bcs)
    b = construct_b(g, ps, bcs)

    # Solve the steady-state problem
    sol = solve(LinearProblem(A, b))
    us = reshape(sol.u, g.nx, g.ny)
    us = interpolate((g.xs, g.ys), us, Gridded(Linear()))

    # Form a set of observations
    xs_o = [x for x ∈ x_locs for _ ∈ y_locs]
    ys_o = [y for _ ∈ x_locs for y ∈ y_locs]
    us_o = [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)] + rand(ϵ_dist)

    return logps, xs_o, ys_o, us_o 

end