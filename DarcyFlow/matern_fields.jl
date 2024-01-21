using Distributions
using LinearAlgebra
using LinearSolve
using SparseArrays
using SpecialFunctions: gamma

# Gradients of basis functions
∇s = [-1.0 1.0 0.0; -1.0 0.0 1.0]

# Transformation matrix inverses
Tinvs = Dict(
    :lower => [
        [ 1.0 -1.0;  0.0  1.0], 
        [ 0.0  1.0; -1.0  0.0], 
        [-1.0  0.0;  1.0 -1.0]
    ],
    :upper => [
        [-1.0  1.0;  1.0  0.0], 
        [ 1.0  0.0;  0.0 -1.0], 
        [ 0.0 -1.0; -1.0  1.0]
    ]
)

function gauss_to_unif(
    x::Real, 
    lb::Real, 
    ub::Real
)::Real

    return lb + (ub - lb) * cdf(Normal(), x)

end

function build_fem_matrices(g::Grid)

    grid_is = reshape(1:g.nx^2, g.nx, g.nx)
    
    elements = []
    for j ∈ 1:(g.nx-1), i ∈ 1:(g.nx-1)
        push!(elements, [grid_is[i, j], grid_is[i+1, j], grid_is[i+1, j+1]])
        push!(elements, [grid_is[i, j], grid_is[i, j+1], grid_is[i+1, j+1]])
    end
    elements = hcat(elements...)
    
    facets_x0 = hcat([[(i-1)*g.nx+1, i*g.nx+1] for i ∈ 1:(g.nx-1)]...)
    facets_x1 = hcat([[i*g.nx, (i+1)*g.nx] for i ∈ 1:(g.nx-1)]...)
    facets_y0 = hcat([[i, i+1] for i ∈ 1:(g.nx-1)]...)
    facets_y1 = hcat([[i, i+1] for i ∈ (g.nx^2-g.nx+1):(g.nx^2-1)]...)
    
    boundary_facets = hcat(facets_x0, facets_x1, facets_y0, facets_y1)
    
    M_i, M_j, M_v = Int[], Int[], Float64[]
    K_i, K_j, K_v = Int[], Int[], Float64[]
    N_i, N_j, N_v = Int[], Int[], Float64[]
    
    for (n, e) ∈ enumerate(eachcol(elements))
    
        for i ∈ 1:3
    
            element_type = n % 2 == 1 ? :lower : :upper
            Tinv = 1/g.Δx * Tinvs[element_type][i]
    
            for j ∈ 1:3
    
                push!(M_i, e[i])
                push!(M_j, e[j])
                i == j && push!(M_v, g.Δx^2/12)
                i != j && push!(M_v, g.Δx^2/24)
    
                push!(K_i, e[i])
                push!(K_j, e[j])
                push!(K_v, 1/2 * g.Δx^2 * ∇s[:, 1]' * Tinv * Tinv' * ∇s[:, (j-i+3)%3+1])
    
            end
    
        end
    
    end
    
    for (fi, fj) ∈ eachcol(boundary_facets)
    
        push!(N_i, fi, fj, fi, fj)
        push!(N_j, fi, fj, fj, fi)
        push!(N_v, g.Δx/3, g.Δx/3, g.Δx/6, g.Δx/6)
    
    end
    
    M = sparse(M_i, M_j, M_v, g.nx^2, g.nx^2)
    K = sparse(K_i, K_j, K_v, g.nx^2, g.nx^2)
    N = sparse(N_i, N_j, N_v, g.nx^2, g.nx^2)
    
    chol = cholesky(Hermitian(M))
    P = sparse(1:g.nx^2, chol.p, ones(g.nx^2))
    L = P' * sparse(chol.L)

    return M, K, N, L

end

struct MaternField

    μ::AbstractVector
    σ_bounds::Tuple
    l_bounds::Tuple

    Δσ::Real 
    Δl::Real

    M::AbstractMatrix 
    K::AbstractMatrix 
    N::AbstractMatrix
    L::AbstractMatrix

    Nθ::Int
    Nu::Int 
    Nω::Int

    function MaternField(
        g::Grid,
        μ::Real,
        σ_bounds::Tuple,
        l_bounds::Tuple
    )

        Δσ = σ_bounds[2] - σ_bounds[1]
        Δl = l_bounds[2] - l_bounds[1]
        
        μ = fill(μ, g.nx^2)
        return new(
            μ, σ_bounds, l_bounds, Δσ, Δl,
            build_fem_matrices(g)...,  
            g.nx^2+2, g.nx^2, 2
        )

    end

end

function Base.rand(
    mf::MaternField, 
    n::Int=1
)::AbstractMatrix   
    return rand(Normal(), mf.Nθ, n)
end

function _transform(
    mf::MaternField,
    θ::AbstractVecOrMat
)

    ξ..., ξ_σ, ξ_l = θ

    σ = gauss_to_unif(ξ_σ, mf.σ_bounds...)
    l = gauss_to_unif(ξ_l, mf.l_bounds...)

    α = σ^2 * (4π * gamma(2)) / gamma(1)

    A = mf.M + l^2 * mf.K + l / 1.42 * mf.N
    b = √(α) * l * mf.L * ξ + A * mf.μ

    return solve(LinearProblem(A, b)).u

end