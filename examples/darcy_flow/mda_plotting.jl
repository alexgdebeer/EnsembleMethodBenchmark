using JLD2
using LaTeXStrings
using PyPlot
using Statistics

include("problem_setup.jl")

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

plot_μ_post = true
plot_σ_post = true

figheight = 3

# Calculate the mean and standard deviation of the (log-)permeabilities
μ_post = reshape(mean(logps[:,:,end], dims=2), nx_c, ny_c)
σ_post = reshape(std(logps[:,:,end], dims=2), nx_c, ny_c)

# Calculate limits for colourbars
logp_min = min(minimum(μ_post), minimum(logps_t))
logp_max = max(maximum(μ_post), maximum(logps_t))

if plot_μ_post

    fig, ax = PyPlot.subplots(1, 2, figsize=(6, figheight))

    m_1 = ax[1].pcolormesh(
        xs_c, ys_c, logps_t', 
        cmap=:viridis, vmin=logp_min, vmax=logp_max)

    m_2 = ax[2].pcolormesh(
        xs_c, ys_c, μ_post', 
        cmap=:viridis, vmin=logp_min, vmax=logp_max)

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

    PyPlot.suptitle("Posterior mean", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/es/mda_post_mean.pdf")
    PyPlot.clf()

end

if plot_σ_post

    PyPlot.figure(figsize=(4, figheight))
    PyPlot.axes().set_aspect("equal")

    PyPlot.pcolormesh(xs_c, ys_c, σ_post', cmap=:magma)
    PyPlot.colorbar()

    PyPlot.xticks(ticks=[0, 1])
    PyPlot.yticks(ticks=[0, 1])

    PyPlot.title("Posterior standard deviations", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/es/mda_post_stds.pdf")
    PyPlot.clf()

end