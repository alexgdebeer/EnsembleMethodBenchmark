using Distributions
using LinearAlgebra
using SpecialFunctions

struct MaternFieldKL

    μ::AbstractVector
    Γ::AbstractMatrix
    Γ_inv::AbstractMatrix
    d::MvNormal

    function MaternFieldKL(
        g::Grid, 
        μ::Real,
        σ::Real, 
        l::Real, 
        ν::Real
    )

        # Form covariance matrix 
        dxs = g.cxs .- g.cxs'
        dys = g.cys .- g.cys'
        Δxs = (dxs.^2 + dys.^2) .^ 0.5 + 1e-8I # Hack

        μ = fill(μ, g.nx^2)
        @info "Building covariance matrix..."
        Γ = σ^2 * (2.0^(1-ν) / gamma(ν)) .* (Δxs/l).^ν .* besselk.(ν, Δxs/l)

        d = MvNormal(μ, Γ)
        return new(μ, Γ, inv(Γ), d)

    end

end

function Base.rand(
    f::MaternFieldKL, 
    n::Int=1
)::AbstractMatrix

    return rand(f.d, n)

end