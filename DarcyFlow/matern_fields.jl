using Distributions
using LinearAlgebra
using SpecialFunctions

struct MaternFieldKL

    μ::Real
    Γ::AbstractMatrix

    A::AbstractMatrix
    λ::AbstractVector 
    V::AbstractMatrix

    n_modes::Int

    function MaternFieldKL(
        g::Grid, 
        μ::Real,
        σ::Real, 
        l::Real, 
        ν::Real;
        n_modes::Int=g.nx^2
    )

        # Form covariance matrix 
        dxs = g.cxs .- g.cxs'
        dys = g.cys .- g.cys'
        Δxs = (dxs.^2 + dys.^2) .^ 0.5 + 1e-8I # Hack

        @info "Building covariance matrix..."
        Γ = σ^2 * (2.0^(1-ν) / gamma(ν)) .* (Δxs/l).^ν .* besselk.(ν, Δxs/l)

        @info "Computing eigendecomposition..."
        N = size(Γ, 1)
        decomp = eigen(Symmetric(Γ), N-n_modes+1:N)

        λ, V = decomp.values, decomp.vectors
        A = V * diagm(sqrt.(λ))

        return new(μ, Γ, A, λ, V, n_modes)

    end

end

function transform(
    f::MaternFieldKL,
    θs::AbstractVecOrMat 
)::AbstractVecOrMat

    return f.μ .+ f.A * θs

end

function Base.rand(
    f::MaternFieldKL, 
    n::Int=1
)::AbstractMatrix

    return rand(Normal(), f.n_modes, n)

end