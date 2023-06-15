using Distributions
using LinearAlgebra
using Plots

# Gradient of underlying line, width, amplitude, period, centre
α_dist = Uniform(-0.8, 0.8)
w_dist = Uniform(0.1, 0.2)
a_dist = Uniform(0.1, 0.2)
p_dist = Uniform(0.25, 0.75)
c_dist = MvNormal([0.5, 0.5], [0.1^2 0.05^2; 0.05^2 0.1^2])

# Centre
α = rand(α_dist)
w = rand(w_dist)
a = rand(a_dist)
p = rand(p_dist)
cx, cy = rand(c_dist)

xs = 0.0:0.01:1.0
ys = 0.0:0.01:1.0

nx = length(xs)
ny = length(ys)
nu = nx * ny

function channel_coords(
    xs::Union{AbstractVector, <:Real}
)
    centre = a * sin.((2π/p).*(xs.-cx)) .+ α * (xs .- cx) .+ cy 
    return centre - w, centre + w

end

# Generate vectors of x and y coordinates
cxs = [x for _ ∈ ys for x ∈ xs]
cys = [y for y ∈ ys for _ ∈ xs]

is_a = Int[]
is_w = Int[]
is_b = Int[]

@time for (i, (x, y)) ∈ enumerate(zip(cxs, cys))
    
    ymin, ymax = channel_coords(x)
    
    y < ymin && push!(is_b, i)
    y > ymax && push!(is_a, i)
    ymin ≤ y ≤ ymax && push!(is_w, i)

end

cxs_a = cxs[is_a]
cys_a = cys[is_a]
cxs_w = cxs[is_w]
cys_w = cys[is_w]
cxs_b = cxs[is_b]
cys_b = cys[is_b]

ds_a = (cxs_a .- cxs_a').^2 + (cys_a .- cys_a').^2
ds_w = (cxs_w .- cxs_w').^2 + (cys_w .- cys_w').^2
ds_b = (cxs_b .- cxs_b').^2 + (cys_b .- cys_b').^2

μ_a = ones(length(is_a)) * 2.0
μ_w = ones(length(is_w)) * -1.0
μ_b = ones(length(is_b)) * 2.0

Γ_a = 1.0^2 * exp.(-(1/(2*0.1)^2) * ds_a) + 1e-6I
Γ_w = 0.5^2 * exp.(-(1/0.1) * sqrt.(ds_w)) + 1e-6I
Γ_b = 1.0^2 * exp.(-(1/(2*0.1)^2) * ds_b) + 1e-6I

μ = zeros(nu)
Γ = zeros(nu, nu)

μ[is_a] = μ_a
μ[is_w] = μ_w 
μ[is_b] = μ_b

@time Γ[is_a, is_a] = Γ_a
Γ[is_w, is_w] = Γ_w 
Γ[is_b, is_b] = Γ_b

d = MvNormal(μ, Γ)

us = @time rand(d)

heatmap(reshape(us, nx, ny)')