using Distributions
using LinearAlgebra
using Plots

abstract type AbstractPrior end

struct ChannelPrior <: AbstractPrior 

    α_dist::Uniform
    w_dist::Uniform
    a_dist::Uniform
    p_dist::Uniform
    cx_dist::Uniform
    cy_dist::Uniform

    cxs::AbstractVector
    cys::AbstractVector
    Nθ::Int

    μ_o::AbstractVector
    μ_i::AbstractVector
    Γ_o::AbstractMatrix
    Γ_i::AbstractMatrix

    # Logpdf: return something proportional to pdf

    function ChannelPrior(
        α_bnds::AbstractVector,
        w_bnds::AbstractVector,
        a_bnds::AbstractVector,
        p_bnds::AbstractVector,
        cx_bnds::AbstractVector,
        cy_bnds::AbstractVector,
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
        cx_dist = Uniform(cx_bnds...)
        cy_dist = Uniform(cy_bnds...)

        Nθ = length(xs) * length(ys)

        # Generate squared distance matrix between each coordinate
        cxs = [x for _ ∈ ys for x ∈ xs]
        cys = [y for y ∈ ys for _ ∈ xs] 
        ds = (cxs .- cxs').^2 + (cys .- cys').^2

        μ_o = fill(μ_o, Nθ)
        μ_i = fill(μ_i, Nθ)

        Γ_o = σ_o^2 * exp.(-(1/2γ_o^2) * ds) + 1.0e-8I
        Γ_i = σ_i^2 * exp.(-(1/γ_i) * sqrt.(ds)) + 1.0e-8I

        return new(
            α_dist, w_dist, a_dist, p_dist, cx_dist, cy_dist, 
            cxs, cys, Nθ,
            μ_o, μ_i, Γ_o, Γ_i
        )

    end
    
end

# TODO: the Nθ stuff currently makes no sense and needs cleaning up
function Base.rand(
    d::ChannelPrior,
    n::Int=1
)

    function channel_bounds(
        x::Real, α::Real, w::Real, a::Real, p::Real, cx::Real, cy::Real
    )

        centre = a * sin((2π/p) * (x-cx)) + α*(x-cx) + cy 
        return centre - w, centre + w

    end

    θs = Matrix(undef, p.Nθ+6, n)

    for i ∈ 1:n

        θs[1,i] = rand(d.α_dist)
        θs[2,i] = rand(d.w_dist)
        θs[3,i] = rand(d.a_dist)
        θs[4,i] = rand(d.p_dist)
        θs[5,i] = rand(d.cx_dist)
        θs[6,i] = rand(d.cy_dist)

        μ = zeros(d.Nθ)
        Γ = zeros(d.Nθ, d.Nθ)

        is_a = Int[]
        is_w = Int[]
        is_b = Int[]

        for (j, (x, y)) ∈ enumerate(zip(d.cxs, d.cys))
            
            ymin, ymax = channel_bounds(x, θs[1:6, i]...)
            
            if y > ymax 
                push!(is_a, j)
            elseif y < ymin 
                push!(is_b, j)
            else
                push!(is_w, j)
            end

        end

        μ[is_a] = p.μ_o[is_a]
        μ[is_b] = p.μ_o[is_b]
        μ[is_w] = p.μ_i[is_w]

        Γ[is_a, is_a] = p.Γ_o[is_a, is_a]
        Γ[is_b, is_b] = p.Γ_o[is_b, is_b]
        Γ[is_w, is_w] = p.Γ_i[is_w, is_w]

        @time display(cholesky(p.Γ_o[is_b, is_b]))
        @time display(cholesky(Γ))
        d = @time MvNormal(μ, Γ)
        θs[7:end,i] = @time rand(d)

    end

    return θs

end

α_bnds = [-0.8, 0.8]
w_bnds = [0.1, 0.2]
a_bnds = [0.1, 0.2]
p_bnds = [0.25, 0.75]
cx_bnds = [0.3, 0.7]
cy_bnds = [0.3, 0.7]
xs = 0:0.01:1
ys = 0:0.01:1
μ_o = 2.0
μ_i = -1.0
σ_o = 0.5
σ_i = 0.5
γ_o = 0.1
γ_i = 0.1

p = ChannelPrior(
    α_bnds, w_bnds, a_bnds, p_bnds, cx_bnds, cy_bnds,
    xs, ys, μ_o, μ_i, σ_o, σ_i, γ_o, γ_i
)

θs = reshape(rand(p)[7:end, :], length(xs), length(ys))

using Plots
heatmap(θs', cmap=:viridis)



# # Gradient of underlying line, width, amplitude, period, centre
# α_dist = Uniform(-0.8, 0.8)
# w_dist = Uniform(0.1, 0.2)
# a_dist = Uniform(0.1, 0.2)
# p_dist = Uniform(0.25, 0.75)
# cx_dist = Uniform(0.3, 0.7)
# cy_dist = Uniform(0.3, 0.7)

# # Centre
# α = rand(α_dist)
# w = rand(w_dist)
# a = rand(a_dist)
# p = rand(p_dist)
# cx = rand(cx_dist)
# cy = rand(cy_dist)

# xs = 0.0:0.01:1.0
# ys = 0.0:0.01:1.0

# nx = length(xs)
# ny = length(ys)
# nu = nx * ny

# # Generate vectors of x and y coordinates
# cxs = [x for _ ∈ ys for x ∈ xs]
# cys = [y for y ∈ ys for _ ∈ xs]

# is_a = Int[]
# is_w = Int[]
# is_b = Int[]

# @time for (i, (x, y)) ∈ enumerate(zip(cxs, cys))
    
#     ymin, ymax = channel_coords(x)
    
#     y < ymin && push!(is_b, i)
#     y > ymax && push!(is_a, i)
#     ymin ≤ y ≤ ymax && push!(is_w, i)

# end

# cxs_a = cxs[is_a]
# cys_a = cys[is_a]
# cxs_w = cxs[is_w]
# cys_w = cys[is_w]
# cxs_b = cxs[is_b]
# cys_b = cys[is_b]

# ds_a = (cxs_a .- cxs_a').^2 + (cys_a .- cys_a').^2
# ds_w = (cxs_w .- cxs_w').^2 + (cys_w .- cys_w').^2
# ds_b = (cxs_b .- cxs_b').^2 + (cys_b .- cys_b').^2

# μ_a = ones(length(is_a)) * 2.0
# μ_w = ones(length(is_w)) * -1.0
# μ_b = ones(length(is_b)) * 2.0

# Γ_a = 1.0^2 * exp.(-(1/(2*0.1)^2) * ds_a) + 1e-6I
# Γ_w = 0.5^2 * exp.(-(1/0.1) * sqrt.(ds_w)) + 1e-6I
# Γ_b = 1.0^2 * exp.(-(1/(2*0.1)^2) * ds_b) + 1e-6I

# μ = zeros(nu)
# Γ = zeros(nu, nu)

# μ[is_a] = μ_a
# μ[is_w] = μ_w 
# μ[is_b] = μ_b

# @time Γ[is_a, is_a] = Γ_a
# Γ[is_w, is_w] = Γ_w 
# Γ[is_b, is_b] = Γ_b

# d = MvNormal(μ, Γ)

# us = @time rand(d)

# heatmap(reshape(us, nx, ny)')