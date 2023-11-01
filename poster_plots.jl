using HDF5
using LaTeXStrings
using PyCall
using PyPlot

@pyimport matplotlib.animation as anim

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

include("setup.jl")

FNAME_PRIOR = "data/prior/prior.h5"
FNAME_MCMC = "data/pcn/pcn.h5"
FNAME_EKI = "data/eki/eki.h5"
FNAME_ENRML = "data/enrml/enrml.h5"
FNAME_LAPLACE = "data/laplace/laplace.h5"

TITLE_SIZE = 14
LABEL_SIZE = 8
TEXT_SIZE = 10
TICK_SIZE = 6

POSTER_BACKGROUND = "gainsboro"
POSTER_GREEN = "forestgreen" # TODO: fix
POSTER_PURPLE = "rebeccapurple"

CMAP_PERMS = :viridis
CMAP_PRESSURES = :magma

WELL_TO_PLOT = 4

function plot_forward_problem(well_centres, θ_t, u_t)

    θ_t = reshape(θ_t, grid_f.nx, grid_f.nx)
    u_t = reshape(u_t, grid_f.nx, grid_f.nx, :) ./ 1.0e6

    u_well = model_f.B_wells * vec(u_t)
    inds = 1:length(u_well)
    u_well = u_well[findall(i -> i%9 == WELL_TO_PLOT, inds)]

    well_nums = [7, 4, 1, 8, 5, 2, 9, 6, 3]
    
    u_min, u_max = extrema(u_t)

    function set_reservoir_ticks!(ax)

        ax.set_xlim((0, 1000))
        ax.set_ylim((0, 1000))
        ax.set_xticks([])
        ax.set_yticks([])

    end

    function make_frame(i::Int)

        println(i)

        axes[2, 1].clear()
        axes[2, 2].clear()

        axes[2, 1].pcolormesh(grid_f.xs, grid_f.xs, u_t[:, :, i+1], 
                              cmap=CMAP_PRESSURES, vmin=u_min, vmax=u_max)
        
        set_reservoir_ticks!(axes[2, 1])
        axes[2, 1].set_title("Reservoir Pressure", fontsize=TITLE_SIZE)

        axes[2, 2].plot(grid_f.ts[1:i+1], u_well[1:i+1], color="cornflowerblue")

        axes[2, 2].set_ylim(0.98minimum(u_well), 20)
        axes[2, 2].set_xlim(0, grid_f.ts[end])

        axes[2, 2].set_xticks([])
        axes[2, 2].set_yticks([])

        # TODO: plot observations alongside this
        axes[2, 2].set_title("Pressure in Well 8", fontsize=TITLE_SIZE)
        
    end

    fig, axes = PyPlot.subplots(nrows=2, ncols=2, figsize=(5.75, 6))
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    # Plot well positions
    for (i, c) ∈ zip(well_nums, well_centres) 
        axes[1, 1].scatter(c[1], c[2], color="k", s=4)
        axes[1, 1].text(c[1], c[2]+40, "Well $i", ha="center", fontsize=TEXT_SIZE)
    end
    axes[1, 1].set_facecolor("lightskyblue")

    axes[1, 2].pcolormesh(grid_f.xs, grid_f.xs, θ_t, cmap=CMAP_PERMS)

    for ax ∈ axes
        ax.set_box_aspect(1)
    end

    set_reservoir_ticks!(axes[1, 1])
    set_reservoir_ticks!(axes[1, 2])

    axes[1, 1].set_title("Well Locations", fontsize=TITLE_SIZE)
    axes[1, 2].set_title("Reservoir Permeability", fontsize=TITLE_SIZE)

    PyPlot.tight_layout()

    ani = anim.FuncAnimation(fig, make_frame, frames=size(u_t, 3), 
                             interval=250)
    ani.save(filename="forward_problem.gif", writer="pillow", dpi=200)

end

BATCH_LENGTH = 100

BATCH_INC = 10
BATCH_INDS = BATCH_INC:BATCH_INC:BATCH_LENGTH

function plot_method_comparison(
    g::Grid, 
    pr::MaternField,
    n_wells::Int, 
    t_obs::AbstractVector,
    d_obs::AbstractVector
)

    μ_min, μ_max = extrema(θ_t)
    σ_min, σ_max = 0.13, 0.9
    u_min, u_max = 1.60e7, 2.05e7

    function get_well_pressures(
        Fs::AbstractMatrix, 
        well_num::Int
    )::AbstractMatrix

        inds = well_num:n_wells:size(Fs, 1)
        return Fs[inds, :]

    end

    function get_well_data(
        well_num::Int
    )

        inds = well_num:n_wells:length(d_obs)
        return d_obs[inds]
    
    end

    function plot_method_data!(
        fname::AbstractString, 
        i::Int, 
        well_a::Int=4,
        well_b::Int=7
    )

        f = h5open(fname, "r")

        μ = f["μ"][:, :]
        σ = f["σ"][:, :]
        us_wa = get_well_pressures(f["Fs"][:, :], well_a)[:, 1:100]
        us_wb = get_well_pressures(f["Fs"][:, :], well_b)[:, 1:100]

        axes[1, i].pcolormesh(g.xs, g.xs, μ, cmap=CMAP_PERMS, vmin=μ_min, vmax=μ_max)
        axes[2, i].pcolormesh(g.xs, g.xs, σ, cmap=CMAP_PERMS, vmin=σ_min, vmax=σ_max)

        # axes[3, i].axvline(80, color="darkgray", linestyle="--", zorder=0)
        axes[3, i].plot(g.ts, us_wa, color=POSTER_GREEN, linewidth=1, zorder=1, alpha=0.25)
        axes[3, i].scatter(t_obs, get_well_data(well_a), zorder=2, color="k", s=5)
        axes[3, i].set_ylim(u_min, u_max)

        # axes[4, i].axvline(80, color="darkgray", linestyle="--", zorder=0)
        axes[4, i].plot(g.ts, us_wb, color=POSTER_PURPLE, linewidth=1, zorder=1, alpha=0.25)
        axes[4, i].scatter(t_obs, get_well_data(well_b), zorder=2, color="k", s=5)
        axes[4, i].set_ylim(u_min, u_max)

        close(f)

    end

    fig, axes = PyPlot.subplots(nrows=4, ncols=5, figsize=(10, 8))
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    for ax ∈ axes
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
    end

    fnames = [FNAME_PRIOR, FNAME_MCMC, FNAME_LAPLACE, FNAME_EKI, FNAME_ENRML]
    for (i, fname) ∈ enumerate(fnames)
        plot_method_data!(fname, i)
    end

    axes[1, 1].set_title("Prior", fontsize=TITLE_SIZE)
    axes[1, 2].set_title("MCMC", fontsize=TITLE_SIZE)
    axes[1, 3].set_title("Laplace\nApproximation", fontsize=TITLE_SIZE)
    axes[1, 4].set_title("Ensemble Kalman\nInversion", fontsize=TITLE_SIZE)
    axes[1, 5].set_title("Ensemble Randomised\nMaximum Likelihood", fontsize=TITLE_SIZE)

    axes[1, 1].set_ylabel("Means", fontsize=TITLE_SIZE)
    axes[2, 1].set_ylabel("Standard\nDeviations", fontsize=TITLE_SIZE)
    axes[3, 1].set_ylabel("Predictions\nat Well 4", fontsize=TITLE_SIZE) # TODO: generalise
    axes[4, 1].set_ylabel("Predictions\nat Well 7", fontsize=TITLE_SIZE) # TODO: generalise

    PyPlot.tight_layout()
    PyPlot.savefig("plots/poster/comparison.png", dpi=300)

end

plot_forward_problem(well_centres, θ_t, u_t)
# plot_mcmc()
plot_method_comparison(grid_c, pr, n_wells, t_obs, d_obs)
# plot_marginals(grid_c)