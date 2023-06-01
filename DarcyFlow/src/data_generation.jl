function exp_squared_cov(
    σ::Real, 
    γ::Real, 
    xs::AbstractVector, 
    ys::AbstractVector
)

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
    n_x::Int, 
    n_y::Int
)::AbstractMatrix

    return reshape(rand(d), n_x, n_y)

end

function generate_data(
    xs::AbstractVector, 
    ys::AbstractVector, 
    x_locs::AbstractVector, 
    y_locs::AbstractVector, 
    bcs::Dict{Symbol, BoundaryCondition}, 
    p_dist::Distribution, 
    ϵ_dist::Distribution
)

    n_x = length(xs)
    n_y = length(ys)

    # Sample a permeability field
    p = interpolate((xs, ys), sample_perms(p_dist, n_x, n_y), Gridded(Linear()))
    
    # Generate and solve the corresponding steady-state problem
    g = construct_grid(xs, ys)
    A = generate_A(g, p, bcs)
    b = generate_b(g, bcs)
    sol = solve(LinearProblem(A, b))

    us = interpolate((xs, ys), reshape(sol.u, n_x, n_y), Gridded(Linear()))

    # Form a set of observations
    xs_o = [x for x ∈ x_locs for _ ∈ y_locs]
    ys_o = [y for _ ∈ x_locs for y ∈ y_locs]
    us_o = [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)] + rand(ϵ_dist)

    return log.(p.coefs), xs_o, ys_o, us_o 

end