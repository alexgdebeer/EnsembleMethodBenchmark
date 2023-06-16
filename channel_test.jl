using Distributions
using LinearAlgebra
using Plots

# TODO: add logpdf function
# Could just sample the full domain for all three sections and treat the 
# permeabilities that aren't actually used as auxilliary variables

abstract type AbstractPrior end

struct ChannelPrior <: AbstractPrior 

    # Gradient of underlying line, width, amplitude, period, centre
    α_dist::Uniform
    w_dist::Uniform
    a_dist::Uniform
    p_dist::Uniform
    c_dist::Uniform

    cxs::AbstractVector
    cys::AbstractVector
    Nθ::Int

    μ_o::AbstractVector
    μ_i::AbstractVector

    Γ_o::AbstractMatrix
    Γ_i::AbstractMatrix

    function ChannelPrior(
        α_bnds::AbstractVector,
        w_bnds::AbstractVector,
        a_bnds::AbstractVector,
        p_bnds::AbstractVector,
        c_bnds::AbstractVector,
        xs::AbstractVector,
        ys::AbstractVector,
        μ_o::Real,
        μ_i::Real,
        σ_o::Real,
        σ_i::Real,
        γ_o::Real,
        γ_i::Real
    )

        α_dist = Uniform(α_bnds...)
        w_dist = Uniform(w_bnds...)
        a_dist = Uniform(a_bnds...)
        p_dist = Uniform(p_bnds...)
        c_dist = Uniform(c_bnds...)

        Nθ = length(xs) * length(ys)

        cxs = [x for _ ∈ ys for x ∈ xs]
        cys = [y for y ∈ ys for _ ∈ xs] 
        ds = (cxs .- cxs').^2 + (cys .- cys').^2

        μ_o = fill(μ_o, Nθ)
        μ_i = fill(μ_i, Nθ)

        Γ_o = σ_o^2 * exp.(-(1/2γ_o^2) * ds) + 1.0e-8I
        Γ_i = σ_i^2 * exp.(-(1/γ_i) * sqrt.(ds)) + 1.0e-8I

        return new(
            α_dist, w_dist, a_dist, p_dist, c_dist, 
            cxs, cys, Nθ,
            μ_o, μ_i, Γ_o, Γ_i
        )

    end
    
end

function channel_bounds(
    x::Real, α::Real, w::Real, a::Real, p::Real, c::Real
)::Tuple

    centre = a * sin((2π/p) * x) + α*x + c 
    return centre - w, centre + w

end

function sample_channel(
    d::ChannelPrior
)::AbstractVector

    θs = [
        rand(d.α_dist),
        rand(d.w_dist),
        rand(d.a_dist),
        rand(d.p_dist),
        rand(d.c_dist)
    ]

    return θs

end

function sample_perms(
    d::ChannelPrior,
    channel_ps::AbstractVector 
)::AbstractVector

    μ = zeros(d.Nθ)
    Γ = zeros(d.Nθ, d.Nθ)

    is_a = Int[]
    is_w = Int[]
    is_b = Int[]

    for (i, (x, y)) ∈ enumerate(zip(d.cxs, d.cys))
        
        ymin, ymax = channel_bounds(x, channel_ps...)
        
        if y > ymax 
            push!(is_a, i)
        elseif y < ymin 
            push!(is_b, i)
        else
            push!(is_w, i)
        end

    end

    μ[is_a] = p.μ_o[is_a]
    μ[is_b] = p.μ_o[is_b]
    μ[is_w] = p.μ_i[is_w]

    Γ[is_a, is_a] = p.Γ_o[is_a, is_a]
    Γ[is_b, is_b] = p.Γ_o[is_b, is_b]
    Γ[is_w, is_w] = p.Γ_i[is_w, is_w]

    return rand(MvNormal(μ, Γ))

end

function Base.rand(
    d::ChannelPrior,
    n::Int=1
)::AbstractVecOrMat

    θs = zeros(p.Nθ+5, n)

    for i ∈ 1:n

        channel_ps = sample_channel(d)
        perms = sample_perms(d, channel_ps)
        θs[:,i] = [channel_ps; perms]

    end

    n == 1 && return vec(θs)
    return θs

end

α_bnds = [-0.5, 0.5]
w_bnds = [0.1, 0.2]
a_bnds = [-0.2, 0.2]
p_bnds = [0.25, 0.75]
c_bnds = [0.4, 0.6]
xs = 0:0.02:1
ys = 0:0.02:1
μ_o = 2.0
μ_i = -1.0
σ_o = 0.5
σ_i = 0.5
γ_o = 0.1
γ_i = 0.1

p = ChannelPrior(
    α_bnds, w_bnds, a_bnds, p_bnds, c_bnds,
    xs, ys, μ_o, μ_i, σ_o, σ_i, γ_o, γ_i
)

θs = reshape(rand(p)[6:end], length(xs), length(ys))
# @time rand(p, 10)

using Plots
heatmap(θs', cmap=:viridis)