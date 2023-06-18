using Distributions
using LinearAlgebra
using Plots

abstract type KernelFunction end

struct ExpSquaredKernel <: KernelFunction

    σ::Real 
    γ::Real

end

struct ExpKernel <: KernelFunction

    σ::Real 
    γ::Real

end

struct ARDExpSquaredKernel <: KernelFunction

    σ::Real
    γs::AbstractVector

end

struct ARDExpKernel <: KernelFunction

    σ::Real
    γs::AbstractVector

end

function generate_cov(
    k::ARDExpSquaredKernel,
    xs::AbstractVector,
    ys::AbstractVector;
    θ::Real=0
)

    # Generate vectors of x and y coordinates
    cxs = [x for _ ∈ ys for x ∈ xs]
    cys = [y for y ∈ ys for _ ∈ xs]

    # TEMPORARY
    m = 0.2
    c = 0.4

    for i ∈ eachindex(cxs)
        if cxs[i] ≥ m*cys[i]+c 
            cys[i] += 0.1
        end
    end

    if θ !== 0
        R = [cos(θ) -sin(θ); sin(θ)  cos(θ)]
        cxs, cys = eachrow(R * [cxs'; cys'])
    end

    # Generate a matrix of scaled distances
    ds = (cxs .- cxs').^2 ./ k.γs[1].^2 + 
         (cys .- cys').^2 ./ k.γs[2].^2

    Γ = k.σ^2 * (exp.(-0.5ds) + 1.0e-8I)

    return Γ

end

xs = 0.0:0.01:1.0
ys = 0.0:0.01:1.0

nx = length(xs)
ny = length(ys)

σ = 1.0
γs = [5.0, 0.1]
k = ARDExpSquaredKernel(σ, γs)

Γ = generate_cov(k, xs, ys, θ=0)#rand()*π/4)

ks = rand(MvNormal(Γ))
ks = reshape(ks, nx, ny)

levels = -5:0.4:5

for i ∈ 1:nx 
    for j ∈ 1:ny 
        diffs = abs.(ks[i,j].-levels)
        ks[i,j] = levels[findall(diffs .== minimum(diffs))][1]
    end
end

heatmap(ks', cmap=:viridis)