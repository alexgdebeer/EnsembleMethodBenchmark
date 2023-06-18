using PyPlot
using SimIntensiveInference

function apply_levels!(ks, Δx)

    nx, ny = size(ks)

    kmin = floor(minimum(ks)) - 0.5Δx
    kmax = ceil(maximum(ks)) + 0.5Δx

    ls = kmin:Δx:kmax

    for i ∈ 1:nx 
        for j ∈ 1:ny 
            ds = abs.(ks[i,j].-ls)
            ks[i,j] = ls[findall(ds .== minimum(ds))][1]
        end
    end

    return

end

xs = 0:0.01:1
ys = 0:0.01:1

nx = length(xs)
ny = length(ys)

function gen_fault_samples()

    m_bnds = [-0.2, 0.2]
    c_bnds = [0.4, 0.6]
    θ_bnds = [-π/16, π/16]

    σ = 1.0
    γx = 15.0
    γy = 0.1
    k = ARDExpSquaredKernel(σ, γx, γy)

    μ = 0.0

    p = FaultPrior(m_bnds, c_bnds, θ_bnds, μ, xs, ys, k)

    ks = rand(p, 3)
    ks = [reshape(k[4:end], nx, ny) for k ∈ eachcol(ks)]

    Δx = 0.8

    for k ∈ ks
        apply_levels!(k, Δx)
    end

    return ks

end

function gen_channel_samples()

    m_bnds = [-0.3, 0.3]
    c_bnds = [0.3, 0.5]
    a_bnds = [0.1, 0.2]
    p_bnds = [0.4, 0.7]
    w_bnds = [0.1, 0.2]

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

    ks = rand(p, 3)
    ks = [reshape(k[6:end], nx, ny) for k ∈ eachcol(ks)]
    
    return ks

end

ks = [
    gen_fault_samples();
    gen_channel_samples()
]

fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

for i ∈ 1:2, j ∈ 1:3

    cmap = i == 1 ? :viridis : :magma

    ind = 3(i-1)+j
    ax[i, j].pcolormesh(xs, ys, ks[ind]', cmap=cmap)

    ax[i, j].set_box_aspect(1)
    ax[i, j].set_xticks([])
    ax[i, j].set_yticks([])

end

PyPlot.tight_layout()
PyPlot.savefig("plots/prior_samples.pdf")
PyPlot.clf()