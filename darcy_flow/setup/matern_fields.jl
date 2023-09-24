using Distributions
using LinearSolve
using SparseArrays
using SpecialFunctions: gamma

struct MaternField

    g::Grid

    mu::Real
    σ_bounds::Tuple{Real, Real}
    l_bounds::Tuple{Real, Real}
    
    I::AbstractMatrix
    X::AbstractMatrix 
    Y::AbstractMatrix
    
    D::AbstractMatrix
    N::AbstractMatrix 

    corner_pts::AbstractVector 
    boundary_pts::AbstractVector

    Nθ::Int 

    function MaternField(
        g::Grid, 
        mu::Real,
        σ_bounds::Tuple{Real, Real},
        l_bounds::Tuple{Real, Real}
    )::MaternField

        return new(
            g, mu, σ_bounds, l_bounds,
            build_fd_matrices(g)..., g.nu + 2
        )

    end

end

function gauss_to_unif(
    x::Real, 
    lb::Real, 
    ub::Real
)::Real

    return lb + (ub - lb) * cdf(Normal(), x)

end

function fill_corners!(
    X::AbstractMatrix
)::Nothing

    X[1, 1] = (X[1, 2] + X[2, 1]) / 2
    X[end, 1] = (X[end, 2] + X[end-1, 1]) / 2
    X[1, end] = (X[1, end-1] + X[2, end]) / 2
    X[end, end] = (X[end, end-1] + X[end-1, end]) / 2
    return nothing

end

function transform(
    f::MaternField, 
    W::AbstractVector
)::AbstractMatrix

    σ = gauss_to_unif(W[1], f.σ_bounds...)
    l = gauss_to_unif(W[2], f.l_bounds...)

    d = 2
    ν = 2-d/2
    α = σ^2 * 2^d * π^(d/2) * gamma(ν+d/2) / gamma(ν)
    λ = 1.42l

    A = f.I - l^2*(f.X+f.Y) + f.D + λ*f.N
    b = √((α*l^2)/(f.g.Δx*f.g.Δy)) * W[3:end]
    b[[f.corner_pts..., f.boundary_pts...]] .= 0

    X = solve(LinearProblem(A, b)).u
    X = reshape(X, f.g.nx, f.g.ny)
    fill_corners!(X)
    return f.mu .+ X

end

function Base.rand(
    f::MaternField, n::Int=1
)::AbstractMatrix
    
    return rand(Normal(), f.Nθ, n)

end

function get_coordinates(
    i::Int,
    g::Grid
)::Tuple{Real, Real}

    x = g.xs[(i-1)%g.nx+1] 
    y = g.ys[Int(ceil(i/g.nx))]
    return x, y

end

function in_corner(
    x::Real, 
    y::Real, 
    g::Grid
)::Bool

    return x ∈ [g.xmin, g.xmax] && y ∈ [g.ymin, g.ymax]

end

function on_boundary(
    x::Real, 
    y::Real, 
    g::Grid
)::Bool

    return x ∈ [g.xmin, g.xmax] || y ∈ [g.ymin, g.ymax]

end

function get_point_types(
    g::Grid
)::Tuple{AbstractVector, AbstractVector, AbstractVector}

    corner_pts = Int[]
    boundary_pts = Int[]
    interior_pts = Int[]

    for i ∈ 1:g.nu

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g) 
            push!(corner_pts, i)
        elseif on_boundary(x, y, g)
            push!(boundary_pts, i)
        else
            push!(interior_pts, i)
        end

    end

    return corner_pts, boundary_pts, interior_pts

end

function get_boundary(
    x::Real, 
    y::Real, 
    g::Grid
)::Symbol

    x == g.xmin && return :x0
    x == g.xmax && return :x1
    y == g.ymin && return :y0
    y == g.ymax && return :y1

    error("Point ($x, $y) is not on a boundary.")

end

function add_corner_points!(
    D_i::AbstractVector,
    D_j::AbstractVector,
    D_v::AbstractVector,
    corner_pts::AbstractVector
)::Nothing

    for i ∈ corner_pts
        push!(D_i, i)
        push!(D_j, i)
        push!(D_v, 1.0)
    end

    return

end

function add_interior_points!(
    I_i::AbstractVector,
    I_j::AbstractVector,
    I_v::AbstractVector,
    X_i::AbstractVector,
    X_j::AbstractVector,
    X_v::AbstractVector, 
    Y_i::AbstractVector,
    Y_j::AbstractVector,
    Y_v::AbstractVector, 
    interior_pts::AbstractVector, 
    g::Grid
)::Nothing

    for i ∈ interior_pts

        push!(I_i, i)
        push!(I_j, i)
        push!(I_v, 1.0)

        push!(X_i, i, i, i)
        push!(X_j, i-1, i, i+1)
        push!(X_v, 1.0 / g.Δx^2, -2.0 / g.Δx^2, 1.0 / g.Δx^2)

        push!(Y_i, i, i, i)
        push!(Y_j, i-g.nx, i, i+g.nx)
        push!(Y_v, 1.0 / g.Δy^2, -2.0 / g.Δy^2, 1.0 / g.Δy^2)

    end

    return

end

function add_boundary_points!(
    D_i::AbstractVector,
    D_j::AbstractVector,
    D_v::AbstractVector,
    N_i::AbstractVector,
    N_j::AbstractVector,
    N_v::AbstractVector,
    boundary_pts::AbstractVector,
    g::Grid
)::Nothing

    for i ∈ boundary_pts

        x, y = get_coordinates(i, g)
        boundary = get_boundary(x, y, g)

        push!(D_i, i)
        push!(D_j, i)
        push!(D_v, 1.0)

        push!(N_i, i, i, i)

        if boundary == :x0
            push!(N_j, i, i+1, i+2)
            push!(N_v, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
        elseif boundary == :x1
            push!(N_j, i, i-1, i-2)
            push!(N_v, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
        elseif boundary == :y0
            push!(N_j, i, i+g.nx, i+2g.nx)
            push!(N_v, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
        elseif boundary == :y1 
            push!(N_j, i, i-g.nx, i-2g.nx)
            push!(N_v, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
        end

    end

    return

end

function build_fd_matrices(
    g::Grid
)

    I_i, I_j, I_v = Int[], Int[], Float64[]
    X_i, X_j, X_v = Int[], Int[], Float64[]
    Y_i, Y_j, Y_v = Int[], Int[], Float64[]
    
    D_i, D_j, D_v = Int[], Int[], Float64[]
    N_i, N_j, N_v = Int[], Int[], Float64[]

    corner_pts, boundary_pts, interior_pts = get_point_types(g)

    add_corner_points!(
        D_i, D_j, D_v, 
        corner_pts
    )

    add_interior_points!(
        I_i, I_j, I_v, 
        X_i, X_j, X_v, 
        Y_i, Y_j, Y_v, 
        interior_pts, g
    )

    add_boundary_points!(
        D_i, D_j, D_v,
        N_i, N_j, N_v,
        boundary_pts, g
    )

    I = sparse(I_i, I_j, I_v, g.nu, g.nu)
    X = sparse(X_i, X_j, X_v, g.nu, g.nu)
    Y = sparse(Y_i, Y_j, Y_v, g.nu, g.nu)
    
    D = sparse(D_i, D_j, D_v, g.nu, g.nu)
    N = sparse(N_i, N_j, N_v, g.nu, g.nu)

    return I, X, Y, D, N, corner_pts, boundary_pts

end