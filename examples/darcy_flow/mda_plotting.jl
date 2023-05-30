using JLD2
using LaTeXStrings
using PyPlot
using Statistics

include("problem_setup.jl")

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

post_mean_vs_truth = true

μ_post = reshape(mean(ps[:,:,end], dims=2), nx_c, ny_c)

pmin = min(minimum(μ_post), minimum(ps_true))
pmax = max(maximum(μ_post), maximum(ps_true))

if post_mean_vs_truth

    fig, ax = PyPlot.subplots(1, 2, figsize=(6, 3))

    m_1 = ax[1].pcolormesh(
        xs_c, ys_c, ps_true', 
        cmap=:viridis, vmin=pmin, vmax=pmax
    )

    m_2 = ax[2].pcolormesh(
        xs_c, ys_c, μ_post',
        cmap=:viridis, vmin=pmin, vmax=pmax
    )

    ax[1].set_box_aspect(1)
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_title("Truth", fontsize=12)

    ax[2].set_box_aspect(1)
    ax[2].set_xticks([0, 1])
    ax[2].set_yticks([0, 1])
    ax[2].set_title("Posterior mean", fontsize=12)

    PyPlot.colorbar(m_1, fraction=0.046, pad=0.04, ax=ax[1])
    PyPlot.colorbar(m_2, fraction=0.046, pad=0.04, ax=ax[2])

    PyPlot.suptitle("ES-MDA: posterior mean vs truth", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/es/es_mda_post_mean_vs_truth.pdf")
    PyPlot.clf()

end