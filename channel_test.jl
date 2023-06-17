using Plots
using SimIntensiveInference

m_bnds = [-0.5, 0.5]
c_bnds = [0.4, 0.6]
a_bnds = [-0.2, 0.2]
p_bnds = [0.25, 0.75]
w_bnds = [0.1, 0.2]

xs = 0:0.02:1
ys = 0:0.02:1

μ_o = 2.0
μ_i = -1.0
σ_o = 0.5
σ_i = 0.5
γ_o = 0.1
γ_i = 0.1

p = ChannelPrior(
    m_bnds, c_bnds, a_bnds, p_bnds, w_bnds,
    xs, ys, μ_o, μ_i, σ_o, σ_i, γ_o, γ_i
)

θs = reshape(rand(p)[6:end], length(xs), length(ys))
# @time rand(p, 10)

using Plots 
heatmap(θs', cmap=:viridis)