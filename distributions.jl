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


"""Evaluates the prior density of a Gaussian prior at particle θ."""
function density(π::GaussianPrior, θ::AbstractVector)::Real

    d = Distributions.MvNormal(π.μ, π.Σ)
    return Distributions.pdf(d, θ)

end


"""Evaluates the prior density of a uniform prior at particle θ."""
function density(π::UniformPrior, θ::AbstractVector)::Real

    d = prod(1 / (ub - lb) for (lb, ub) ∈ zip(π.lbs, π.ubs))
    return all(π.lbs .≤ θ .≤ π.ubs) ? d : 0.0

end


"""Evaluates the likelihood at particle θ."""
function density(L::GaussianLikelihood, θ::AbstractVector)::Real 

    d = Distributions.MvNormal(L.μ, L.Σ)
    return Distributions.pdf(d, θ)

end


"""Evaluates the density of kernel κ centred at particle θ⁺, at particle θ.
TODO: this is probably wrong if there are bounds on the prior."""
function density(
    κ::GaussianKernel, 
    θ⁺::AbstractVector, 
    θ::AbstractVector
)::Real

    d = Distributions.MvNormal(θ⁺, κ.Σ)
    return Distributions.pdf(d, θ)

end


"""Evaluates the density of kernel κ, centred at θ⁺, at θ.
TODO: this is probably wrong if there are bounds on the prior."""
function density(
    κ::UniformKernel, 
    θ⁺::AbstractVector, 
    θ::AbstractVector
)::Real

    d = prod(1 / (ub-lb) for (lb, ub) ∈ zip(κ.lbs, κ.ubs))
    return all(κ.lbs .≤ (θ .- θ⁺) .≤ κ.ubs) ? d : 0.0

end


"""Evaluates the density of the error function."""
function density(
    e::GaussianError,
    θ::AbstractVector
)::Real



end


"""Generates samples from a Gaussian prior."""
function sample(π::GaussianPrior; n::Int = -1)::Union{AbstractVector, Real}

    d = Distributions.MvNormal(π.μ, π.Σ)
    return n == -1 ? rand(d) : [vec(c) for c ∈ eachcol(rand(d, n))]

end


"""Generates samples from a uniform prior."""
function sample(π::UniformPrior; n::Int = -1)::Union{AbstractVector, Real}

    n_params = length(π.lbs)
    s = [π.lbs .+ (π.ubs .- π.lbs) .* rand(n_params) for _ ∈ 1:abs(n)]

    return n == -1 ? s[1] : s

end


"""Adds Gaussian noise to a set of model outputs y."""
function add_noise!(y::AbstractVector, n::GaussianError)::Nothing

    e = Distributions.MvNormal(n.μ, n.Σ)
    y .+= reshape(rand(e), size(y))

    return nothing

end


"""Perturbs a particle, ensuring that the perturbed particle has a positive 
prior probability."""
function perturb(
    κ::UniformKernel, 
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

"""Perturbs a particle, ensuring that the perturbed particle has a positive 
prior probability."""
function perturb(
    κ::GaussianKernel, 
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


"""Samples from a given population θ with normalised weights w."""
function sample_from_population(
    θs::AbstractVector, 
    weights::AbstractVector
)::AbstractVector

    return θs[findfirst(cumsum(weights) .≥ rand())]

end