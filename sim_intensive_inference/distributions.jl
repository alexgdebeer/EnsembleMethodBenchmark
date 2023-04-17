export GaussianPrior


abstract type AbstractPrior end

abstract type AbstractErrorModel end
abstract type AbstractLikelihood end

abstract type AbstractPerturbationKernel end
abstract type AbstractAdaptivePerturbationKernel <: AbstractPerturbationKernel end
abstract type AbstractStaticPerturbationKernel <: AbstractPerturbationKernel end

abstract type AbstractAcceptanceKernel end


struct GaussianPrior <: AbstractPrior 
    μ::AbstractVector
    Σ::AbstractMatrix
end


struct GaussianError <: AbstractErrorModel
    μ::AbstractVector
    Σ::AbstractMatrix
end


struct GaussianLikelihood <: AbstractLikelihood
    μ::AbstractVector
    Σ::AbstractMatrix
end


struct UniformLikelihood <: AbstractLikelihood
    lbs::AbstractVector
    ubs::AbstractVector
end


struct StaticGaussianKernel <: AbstractStaticPerturbationKernel
    Σ::AbstractMatrix
end


mutable struct ComponentwiseGaussianKernel <: AbstractAdaptivePerturbationKernel
    Σ::AbstractMatrix 
end

ComponentwiseGaussianKernel() = ComponentwiseGaussianKernel(Matrix(undef, 0, 0))


mutable struct MultivariateGaussianKernel <: AbstractAdaptivePerturbationKernel
    Σ::AbstractMatrix
end

MultivariateGaussianKernel() = MultivariateGaussianKernel(Matrix(undef, 0, 0))


struct GaussianAcceptanceKernel <: AbstractAcceptanceKernel
    Σ::AbstractMatrix 
    c::Real
end

GaussianAcceptanceKernel(Σ) = GaussianAcceptanceKernel(
    Σ, Distributions.pdf(Distributions.MvNormal(Σ), zeros(size(Σ, 1))))


struct UniformAcceptanceKernel <: AbstractAcceptanceKernel
    lbs::AbstractVector 
    ubs::AbstractVector
    c::Real
end

UniformAcceptanceKernel(lbs, ubs) = UniformAcceptanceKernel(
    lbs, ubs, prod(1 / (ub - lb) for (lb, ub) ∈ zip(lbs, ubs)))


function density(π::GaussianPrior, θ::AbstractVector)::Real

    d = Distributions.MvNormal(π.μ, π.Σ)
    return Distributions.pdf(d, θ)

end


function density(L::GaussianLikelihood, y::AbstractVector)::Real 

    d = Distributions.MvNormal(L.μ, L.Σ)
    return Distributions.pdf(d, y)

end


function density(L::UniformLikelihood, y::AbstractVector)::Real 

    d = prod(1 / (ub - lb) for (lb, ub) ∈ zip(L.lbs, L.ubs))
    return all(L.lbs .≤ y .≤ L.ubs) ? d : 0.0

end


function density(
    K::AbstractPerturbationKernel, 
    θ⁺::AbstractVector, 
    θ::AbstractVector
)::Real

    d = Distributions.MvNormal(θ⁺, K.Σ)
    return Distributions.pdf(d, θ)

end


function density(K::GaussianAcceptanceKernel, θ::AbstractVector)::Real

    d = Distributions.MvNormal(K.Σ)
    return Distributions.pdf(d, θ)

end


function density(L::UniformAcceptanceKernel, θ::AbstractVector)::Real 

    d = prod(1 / (ub - lb) for (lb, ub) ∈ zip(L.lbs, L.ubs))
    return all(L.lbs .≤ θ .≤ L.ubs) ? d : 0.0

end


function sample(π::GaussianPrior; n::Int = -1)::Union{AbstractVector, Real}

    d = Distributions.MvNormal(π.μ, π.Σ)
    return n == -1 ? rand(d) : Vector{Float64}[eachcol(rand(d, n))...]

end


function sample(L::GaussianLikelihood)::AbstractVector

    d = Distributions.MvNormal(L.μ, L.Σ)
    return rand(d)

end


function add_noise!(y::AbstractVector, n::GaussianError)::Nothing

    e = Distributions.MvNormal(n.μ, n.Σ)
    y .+= rand(e)
    return nothing

end


function perturb(
    K::AbstractPerturbationKernel, 
    θ::AbstractVector, 
    π::AbstractPrior
)::AbstractVector

    d = Distributions.MvNormal(θ, K.Σ)

    p = rand(d)

    while density(π, p) ≤ 1e-16
        p = rand(d)
    end

    return p

end


function sample_from_population(
    θs::AbstractVector, 
    ws::AbstractVector
)::AbstractVector

    return θs[findfirst(cumsum(ws) .+ 1e-8 .≥ rand())]

end


function resample_population(
    θs::AbstractVector,
    ys::AbstractVector,
    ws::AbstractVector
)::Tuple

    is = [findfirst(cumsum(ws) .+ 1e-8 .≥ rand()) for _ ∈ 1:length(θs)]
    return θs[is], ys[is]

end


function resample_population(
    θs::AbstractVector,
    ws::AbstractVector;
    N::Int=length(θs)
)::AbstractVector

    cs = cumsum(ws)
    is = [findfirst(cs .≥ rand()) for _ ∈ 1:N]
    return θs[is]

end


function update!(
    K::ComponentwiseGaussianKernel, 
    θs::AbstractDict, 
    ws::AbstractDict, 
    ds::AbstractDict,
    t::Int,
    εₜ::Real
)::Nothing

    n_σs = length(θs[t-1][1])
    vars = zeros(n_σs)

    θ̂s = θs[t-1][ds[t-1] .≤ εₜ]
    ŵs = ws[t-1][ds[t-1] .≤ εₜ] / sum(ws[t-1][ds[t-1] .≤ εₜ])

    for (θ, w) ∈ zip(θs[t-1], ws[t-1])
        for (θ̂, ŵ) ∈ zip(θ̂s, ŵs)
            vars += ŵ*w * (θ̂-θ).^2
        end
    end
    
    K.Σ = LinearAlgebra.diagm(vars)

    return nothing

end


function update!(
    K::MultivariateGaussianKernel, 
    θs::AbstractDict, 
    ws::AbstractDict, 
    ds::AbstractDict,
    t::Int,
    εₜ::Real
)::Nothing

    n_σs = length(θs[t-1][1])
    Σₜ = zeros(n_σs, n_σs)

    θ̂s = θs[t-1][ds[t-1] .≤ εₜ]
    ŵs = ws[t-1][ds[t-1] .≤ εₜ] / sum(ws[t-1][ds[t-1] .≤ εₜ])

    for (θ, w) ∈ zip(θs[t-1], ws[t-1])
        for (θ̂, ŵ) ∈ zip(θ̂s, ŵs)
            Σₜ += ŵ*w * (θ̂-θ)*(θ̂-θ)'
        end
    end

    K.Σ = LinearAlgebra.Hermitian(Σₜ)

    return nothing

end


function update!(
    K::ComponentwiseGaussianKernel,
    θs::AbstractDict,
    ws::AbstractDict,
    t::Int
)::Nothing

    n = length(θs[t-1])
    θs_res = [sample_from_population(θs[t-1], ws[t-1]) for _ ∈ 1:n]
    
    vars = Statistics.var(θs_res)
    K.Σ = LinearAlgebra.diagm(vars)

    return nothing

end