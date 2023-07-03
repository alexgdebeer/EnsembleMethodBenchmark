using JLD2
using LaTeXStrings
using PyPlot
using Statistics

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

const figheight = 3

function plot_μ_post(μ_post, logps_t, g_c, g_f, fname; title="Posterior mean")

    # Calculate limits for colourbars
    logp_min = min(minimum(μ_post), minimum(logps_t))
    logp_max = max(maximum(μ_post), maximum(logps_t))

    _, ax = PyPlot.subplots(1, 2, figsize=(6, figheight))

    m_1 = ax[1].pcolormesh(
        g_f.xs, g_f.ys, logps_t', 
        cmap=:viridis, vmin=logp_min, vmax=logp_max)

    m_2 = ax[2].pcolormesh(
        g_c.xs, g_c.ys, μ_post', 
        cmap=:viridis, vmin=logp_min, vmax=logp_max)

    ax[1].set_box_aspect(1)
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_title("Truth", fontsize=12)

    ax[2].set_box_aspect(1)
    ax[2].set_xticks([0, 1])
    ax[2].set_yticks([0, 1])
    ax[2].set_title(title, fontsize=12)

    PyPlot.colorbar(m_1, fraction=0.046, pad=0.04, ax=ax[1])
    PyPlot.colorbar(m_2, fraction=0.046, pad=0.04, ax=ax[2])

    PyPlot.suptitle(title, fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig(fname)
    PyPlot.clf()

end

function plot_σ_post(σ_post, g_c, fname)

    PyPlot.figure(figsize=(4, figheight))
    PyPlot.axes().set_aspect("equal")

    PyPlot.pcolormesh(g_c.xs, g_c.ys, σ_post', cmap=:magma)
    PyPlot.colorbar()

    PyPlot.xticks(ticks=[0, 1])
    PyPlot.yticks(ticks=[0, 1])

    PyPlot.title("Posterior standard deviations", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig(fname)
    PyPlot.clf()

end