export GaussianPrior, UniformPrior, GaussianError, 
GaussianPerturbationKernel, UniformPerturbationKernel, 
GaussianAcceptanceKernel


abstract type AbstractPrior end

abstract type AbstractErrorModel end
abstract type AbstractLikelihood end

abstract type AbstractPerturbationKernel end
abstract type AbstractAcceptanceKernel end


struct GaussianPrior <: AbstractPrior 

    μ::AbstractVector
    Σ::AbstractMatrix

end


struct UniformPrior <: AbstractPrior

    lbs::AbstractVector
    ubs::AbstractVector

end


struct GaussianError <: AbstractErrorModel

    μ::AbstractVector
    Σ::AbstractMatrix

end


struct GaussianLikelihood <: AbstractLikelihood

    μ::AbstractVector
    Σ::AbstractMatrix

end


struct GaussianPerturbationKernel <: AbstractPerturbationKernel

    Σ::AbstractMatrix

end


struct UniformPerturbationKernel <: AbstractPerturbationKernel
    
    lbs::AbstractVector
    ubs::AbstractVector

end


struct GaussianAcceptanceKernel <: AbstractAcceptanceKernel

    Σ::AbstractMatrix 
    c::Real

end

GaussianAcceptanceKernel(Σ) = GaussianAcceptanceKernel(
    Σ, 
    Distributions.pdf(Distributions.MvNormal(Σ), zeros(size(Σ, 1)))
)


struct UniformAcceptanceKernel <: AbstractAcceptanceKernel

    lbs::AbstractVector 
    ubs::AbstractVector

end


function density(π::GaussianPrior, θ::AbstractVector)::Real

    d = Distributions.MvNormal(π.μ, π.Σ)
    return Distributions.pdf(d, θ)

end


function density(π::UniformPrior, θ::AbstractVector)::Real

    d = prod(1 / (ub - lb) for (lb, ub) ∈ zip(π.lbs, π.ubs))
    return all(π.lbs .≤ θ .≤ π.ubs) ? d : 0.0

end


function density(L::GaussianLikelihood, θ::AbstractVector)::Real 

    d = Distributions.MvNormal(L.μ, L.Σ)
    return Distributions.pdf(d, θ)

end


function density(
    κ::GaussianPerturbationKernel, 
    θ⁺::AbstractVector, 
    θ::AbstractVector
)::Real

    d = Distributions.MvNormal(θ⁺, κ.Σ)
    return Distributions.pdf(d, θ)

end


function density(
    κ::UniformPerturbationKernel, 
    θ⁺::AbstractVector, 
    θ::AbstractVector
)::Real

    d = prod(1 / (ub-lb) for (lb, ub) ∈ zip(κ.lbs, κ.ubs))
    return all(κ.lbs .≤ (θ .- θ⁺) .≤ κ.ubs) ? d : 0.0

end


function density(K::GaussianAcceptanceKernel, θ::Vector)::Real 

    d = Distributions.MvNormal(K.Σ)
    return Distributions.pdf(d, θ)

end



function sample(π::GaussianPrior; n::Int = -1)::Union{AbstractVector, Real}

    d = Distributions.MvNormal(π.μ, π.Σ)
    return n == -1 ? rand(d) : [vec(c) for c ∈ eachcol(rand(d, n))]

end


function sample(π::UniformPrior; n::Int = -1)::Union{AbstractVector, Real}

    n_params = length(π.lbs)
    s = [π.lbs .+ (π.ubs .- π.lbs) .* rand(n_params) for _ ∈ 1:abs(n)]

    return n == -1 ? s[1] : s

end


function add_noise!(y::AbstractVector, n::GaussianError)::Nothing

    e = Distributions.MvNormal(n.μ, n.Σ)
    y .+= reshape(rand(e), size(y))

    return nothing

end


function perturb(
    κ::UniformPerturbationKernel, 
    θ::AbstractVector, 
    π::AbstractPrior
)::AbstractVector

    n_θs = length(κ.lbs)

    p = κ.lbs .+ (κ.ubs .- κ.lbs) .* rand(n_θs)

    while density(π, θ .+ p) ≤ 1e-16
        p = κ.lbs .+ (κ.ubs .- κ.lbs) .* rand(n_θs)
    end

    return θ .+ p

end


function perturb(
    κ::GaussianPerturbationKernel, 
    θ::AbstractVector, 
    π::AbstractPrior
)::AbstractVector

    d = Distributions.MvNormal(θ, κ.Σ)

    p = rand(d)

    while density(π, p) ≤ 1e-16
        p = rand(d)
    end

    return p

end


function sample_from_population(
    θs::AbstractVector, 
    weights::AbstractVector
)::AbstractVector

    return θs[findfirst(cumsum(weights) .≥ rand())]

end