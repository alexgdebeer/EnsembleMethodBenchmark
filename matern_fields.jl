using Distributions
using LinearSolve 
using SparseArrays
using SpecialFunctions: gamma

include("darcy_flow/setup/structs.jl")

function gauss_to_unif(x::Real, lb::Real, ub::Real)
    return lb + (ub - lb) * cdf(Normal(), x)
end

struct MaternField

    g::Grid

    mu::Real
    σ_bounds::Tuple{Real, Real}
    l_bounds::Tuple{Real, Real}
    
    I::AbstractMatrix
    X::AbstractMatrix 
    Y::AbstractMatrix
    
    D::AbstractMatrix
    Nx::AbstractMatrix 
    Ny::AbstractMatrix

    function MaternField(
        g::Grid, 
        mu::Real,
        σ_bounds::Tuple{Real, Real},
        l_bounds::Tuple{Real, Real}
    )::MaternField

        return new(
            g, mu, σ_bounds, l_bounds,
            build_fd_matrices(g)...
        )

    end

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

    corner_pts, boundary_pts, _ = get_point_types(f.g)

    A = f.I - l^2*f.X - l^2*f.Y + f.D + λ*(f.Nx+f.Ny)
    b = √((α*l^2)/(f.g.Δx*f.g.Δy)) * W[3:end]
    b[[corner_pts..., boundary_pts...]] .= 0

    X = solve(LinearProblem(A, b)).u

    X = reshape(X, f.g.nx, f.g.ny)

    # Fill in corners
    X[1, 1] = (X[1, 2] + X[2, 1]) / 2
    X[end, 1] = (X[end, 2] + X[end-1, 1]) / 2
    X[1, end] = (X[1, end-1] + X[2, end]) / 2
    X[end, end] = (X[end, end-1] + X[end-1, end]) / 2

    return f.mu .+ X

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

function get_point_types(g::Grid)

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

function build_fd_matrices(g::Grid)

    I_i, I_j, I_v = Int[], Int[], Float64[]
    X_i, X_j, X_v = Int[], Int[], Float64[]
    Y_i, Y_j, Y_v = Int[], Int[], Float64[]
    
    D_i, D_j, D_v = Int[], Int[], Float64[]
    Nx_i, Nx_j, Nx_v = Int[], Int[], Float64[]
    Ny_i, Ny_j, Ny_v = Int[], Int[], Float64[]

    corner_pts, boundary_pts, interior_pts = get_point_types(g)

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

    for i ∈ corner_pts

        push!(D_i, i)
        push!(D_j, i)
        push!(D_v, 1.0)

    end

    for i ∈ boundary_pts

        x, y = get_coordinates(i, g)
        boundary = get_boundary(x, y, g)

        push!(D_i, i)
        push!(D_j, i)
        if boundary ∈ (:x0, :y0)
            push!(D_v, 1.0)
        else
            push!(D_v, -1.0)
        end

        if boundary == :x0
            push!(Nx_i, i, i, i)
            push!(Nx_j, i, i+1, i+2)
            push!(Nx_v, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
        elseif boundary == :x1
            push!(Nx_i, i, i, i)
            push!(Nx_j, i, i-1, i-2)
            push!(Nx_v, -3.0 / 2g.Δx, 4.0 / 2g.Δx, -1.0 / 2g.Δx)
        elseif boundary == :y0
            push!(Ny_i, i, i, i)
            push!(Ny_j, i, i+g.nx, i+2g.nx)
            push!(Ny_v, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
        elseif boundary == :y1 
            push!(Ny_i, i, i, i)
            push!(Ny_j, i, i-g.nx, i-2g.nx)
            push!(Ny_v, -3.0 / 2g.Δy, 4.0 / 2g.Δy, -1.0 / 2g.Δy)
        end

    end

    I = sparse(I_i, I_j, I_v, g.nu, g.nu)
    X = sparse(X_i, X_j, X_v, g.nu, g.nu)
    Y = sparse(Y_i, Y_j, Y_v, g.nu, g.nu)
    
    D = sparse(D_i, D_j, D_v, g.nu, g.nu)
    Nx = sparse(Nx_i, Nx_j, Nx_v, g.nu, g.nu)
    Ny = sparse(Ny_i, Ny_j, Ny_v, g.nu, g.nu)

    return I, X, Y, D, Nx, Ny

end

xmin, xmax = 0, 1000
ymin, ymax = 0, 1000
Δx, Δy = 20, 20

mu = -14
σ_bounds = (1.0, 2.0)
l_bounds = (100, 300)

g = SteadyStateGrid(xmin:Δx:xmax, ymin:Δy:ymax)
f = MaternField(g, mu, σ_bounds, l_bounds)

W = rand(Normal(), g.nu + 2)
X = transform(f, W)

heatmap(rotl90(X), aspect_ratio=:equal, cmap=:turbo)