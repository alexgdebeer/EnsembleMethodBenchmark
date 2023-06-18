using Plots
using SimIntensiveInference

m_bnds = [-0.3, 0.3]
c_bnds = [0.4, 0.6]
θ_bnds = [-π/4, π/4]

xs = 0.0:0.02:1.0
ys = 0.0:0.02:1.0

nx = length(xs)
ny = length(ys)

σ = 1.0
γx = 5.0
γy = 0.2
k = ARDExpSquaredKernel(σ, γx, γy)

μ = 0.0

p = FaultPrior(m_bnds, c_bnds, θ_bnds, μ, xs, ys, k)

ks = rand(p, 1)
ks = reshape(ks[4:end], nx, ny)

# levels = -5:0.4:5

# for i ∈ 1:nx 
#     for j ∈ 1:ny 
#         diffs = abs.(ks[i,j].-levels)
#         ks[i,j] = levels[findall(diffs .== minimum(diffs))][1]
#     end
# end

heatmap(ks', cmap=:viridis)