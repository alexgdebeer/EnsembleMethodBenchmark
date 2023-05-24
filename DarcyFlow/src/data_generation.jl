function generate_data(xs, ys, bcs, p_dist, x_locs, y_locs, ϵ_dist)

    n_x = length(xs)
    n_y = length(ys)

    # Sample a permeability field
    p = interpolate((xs, ys), sample_perms(p_dist, n_x, n_y), Gridded(Linear()))
    
    # Generate and solve the corresponding steady-state problem
    A, b = generate_grid(xs, ys, p, bcs)
    sol = solve(LinearProblem(A, b))

    us = interpolate((xs, ys), reshape(sol.u, n_x, n_y), Gridded(Linear()))

    # Form a set of observations
    xs_o = vec([x for _ ∈ y_locs, x ∈ x_locs])
    ys_o = vec([y for y ∈ y_locs, _ ∈ x_locs])
    us_o = [us(x, y) for (x, y) ∈ zip(xs_o, ys_o)] + rand(ϵ_dist)

    return log.(p.coefs), xs_o, ys_o, us_o 

end