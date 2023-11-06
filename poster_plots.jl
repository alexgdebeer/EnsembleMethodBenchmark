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
TEXT_SIZE = 8
TICK_SIZE = 6

POSTER_BACKGROUND = "gainsboro"
POSTER_GREEN = "forestgreen" # TODO: fix
POSTER_PURPLE = "rebeccapurple"
POSTER_BLUE = "lightskyblue"

CMAP_PERMS = :viridis
CMAP_PRESSURES = :magma

WELL_TO_PLOT = 4

WELL_NUMS = [7, 4, 1, 8, 5, 2, 9, 6, 3]

well_name(n) = raw"$\texttt{WELL"* "$n" * raw"}$"

function plot_forward_problem(g::Grid, well_centres, θ_t, u_t, d_obs, t_obs)

    u_min, u_max = 1.60e7, 2.05e7

    θ_t = reshape(θ_t, grid_f.nx, grid_f.nx)

    u_wells = model_f.B_wells * vec(u_t)
    u_well = u_wells[WELL_TO_PLOT:n_wells:length(u_wells)]
    u_well = [model_f.u0, u_well...]
    ts = [0.0, g.ts...]

    u_t = [fill(model_r.u0, g.nx^2)..., u_t...]
    u_t = reshape(u_t, grid_f.nx, grid_f.nx, :)

    get_well_obs(n) = d_obs[n:n_wells:length(d_obs)]

    function set_reservoir_ticks!(ax)

        ax.set_xlim((0, 1000))
        ax.set_ylim((0, 1000))
        ax.set_xticks([])
        ax.set_yticks([])

    end

    i_max = size(u_t, 3)

    function make_frame(i::Int)

        println(i)

        well_obs = get_well_obs(WELL_TO_PLOT)

        axes[2, 1].clear()
        axes[2, 2].clear()

        axes[2, 1].pcolormesh(grid_f.xs, grid_f.xs, u_t[:, :, min(i+1, i_max)], 
                              cmap=CMAP_PRESSURES, vmin=u_min, vmax=u_max)
        
        set_reservoir_ticks!(axes[2, 1])
        axes[2, 1].set_title("Reservoir Pressure", fontsize=TITLE_SIZE)

        axes[2, 2].plot(ts[1:min(i+1, i_max)], u_well[1:min(i+1, i_max)], color="k", linestyle="--", linewidth=1)
        if ts[min(i+1, i_max)] >= t_obs[1]
            axes[2, 2].scatter(t_obs[t_obs .<= ts[min(i+1, i_max)]], well_obs[t_obs .<= ts[min(i+1, i_max)]], color="k", s=5)
        end

        axes[2, 2].set_ylim(u_min, u_max)
        axes[2, 2].set_xlim(-5.0, ts[end]+5.0)
        axes[2, 2].set_xticks([])
        axes[2, 2].set_yticks([])

        axes[2, 2].set_title("Pressure in " * well_name(8), fontsize=TITLE_SIZE)
        
    end

    fig, axes = PyPlot.subplots(nrows=2, ncols=2, figsize=(4.5, 5))
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    # Plot well positions
    for (i, c) ∈ zip(WELL_NUMS, well_centres)
        axes[1, 1].scatter(c[1], c[2], s=10, facecolors="none", edgecolors="k")
        axes[1, 1].text(c[1], c[2]+40, well_name(i), ha="center", fontsize=TEXT_SIZE)
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

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ani = anim.FuncAnimation(fig, make_frame, frames=size(u_t, 3)+10, interval=250)
    ani.save(filename="plots/poster/forward_problem.apng", writer="pillow", dpi=500)

end

function plot_prior_samples(g::Grid, pr::MaternField)

    ηs = rand(pr, 4)
    ηs[end, :] = [-2, -0.5, 0.5, 2]
    θs = [transform(pr, η) for η ∈ eachcol(ηs)]
    θs = [reshape(θ, g.nx, g.nx) for θ ∈ θs]

    fig, axes = PyPlot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    for (i, ax) ∈ enumerate(axes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.pcolormesh(θs[i], cmap=CMAP_PERMS)
    end

    # PyPlot.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    PyPlot.savefig("plots/poster/prior_samples.png", dpi=300)

end

BATCH_LENGTH = 100

BATCH_INC = 10
BATCH_INDS = BATCH_INC:BATCH_INC:BATCH_LENGTH

function plot_method_comparison(
    g_f::Grid,
    g_c::Grid, 
    u_t::AbstractVector,
    model_f::Model,
    n_wells::Int, 
    t_obs::AbstractVector,
    d_obs::AbstractVector
)

    μ_min, μ_max = extrema(θ_t)
    σ_min, σ_max = 0.13, 0.9
    u_min, u_max = 1.60e7, 2.05e7

    ts_f = [0.0, g_f.ts...]
    ts_c = [0.0, g_c.ts...]

    u_t = model_f.B_wells * u_t

    function get_well_pressures(
        Fs::AbstractMatrix, 
        well_num::Int
    )::AbstractMatrix

        inds = well_num:n_wells:size(Fs, 1)
        us = model_f.u0 * ones(length(inds)+1, size(Fs, 2))
        us[2:end, :] = Fs[inds, :]
        return us

    end

    function get_well_pressures_true(
        well_num::Int
    )::AbstractVector

        inds = well_num:n_wells:length(u_t)
        return [model_f.u0, u_t[inds]...]

    end

    get_well_obs(n) = d_obs[n:n_wells:length(d_obs)]

    function plot_method_data!(
        fname::AbstractString, 
        i::Int, 
        well_a::Int=4,
        well_b::Int=7
    )::Nothing

        f = h5open(fname, "r")
        Fs = f["Fs"][:, :]
        μ = f["μ"][:, :]
        σ = f["σ"][:, :]
        close(f)

        NFs = size(Fs, 2)
        us_wa = get_well_pressures(Fs, well_a)[:, rand(1:NFs, 100)]
        us_wb = get_well_pressures(Fs, well_b)[:, rand(1:NFs, 100)]

        us_wa_obs = get_well_obs(well_a)
        us_wb_obs = get_well_obs(well_b)

        us_wa_true = get_well_pressures_true(well_a)
        us_wb_true = get_well_pressures_true(well_b)

        axes[1, i].pcolormesh(g_c.xs, g_c.xs, μ, cmap=CMAP_PERMS, vmin=μ_min, vmax=μ_max)
        axes[2, i].pcolormesh(g_c.xs, g_c.xs, σ, cmap=CMAP_PERMS, vmin=σ_min, vmax=σ_max)

        # axes[3, i].axvline(80, color="darkgray", linestyle="--", zorder=0)
        axes[3, i].plot(ts_c, us_wa, color=POSTER_BLUE, linewidth=1, zorder=1, alpha=0.25)
        axes[3, i].plot(ts_f, us_wa_true, color="k", linewidth=1, linestyle="--", zorder=2)
        axes[3, i].scatter(t_obs, us_wa_obs, zorder=3, color="k", s=5)
        axes[3, i].set_ylim(u_min, u_max)

        # axes[4, i].axvline(80, color="darkgray", linestyle="--", zorder=0)
        axes[4, i].plot(ts_c, us_wb, color=POSTER_BLUE, linewidth=1, zorder=1, alpha=0.25)
        axes[4, i].plot(ts_f, us_wb_true, color="k", linewidth=1, linestyle="--", zorder=2)
        axes[4, i].scatter(t_obs, us_wb_obs, zorder=3, color="k", s=5)
        axes[4, i].set_ylim(u_min, u_max)

        return nothing

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

    axes[1, 1].set_title("Prior\nSamples", fontsize=TITLE_SIZE)
    axes[1, 2].set_title("Markov Chain\nMonte Carlo", fontsize=TITLE_SIZE)
    axes[1, 3].set_title("Laplace\nApproximation", fontsize=TITLE_SIZE)
    axes[1, 4].set_title("Ensemble Kalman\nInversion", fontsize=TITLE_SIZE)
    axes[1, 5].set_title("Ensemble Randomised\nMaximum Likelihood", fontsize=TITLE_SIZE)

    axes[1, 1].set_ylabel("Means", fontsize=TITLE_SIZE)
    axes[2, 1].set_ylabel("Standard\nDeviations", fontsize=TITLE_SIZE)
    axes[3, 1].set_ylabel("Pressure in\n" * well_name(WELL_NUMS[4]), fontsize=TITLE_SIZE) # TODO: generalise
    axes[4, 1].set_ylabel("Pressure in\n" * well_name(WELL_NUMS[7]), fontsize=TITLE_SIZE) # TODO: generalise

    PyPlot.tight_layout()
    PyPlot.savefig("plots/poster/comparison.png", dpi=300)

end

function generate_colourbars()

    μ_min, μ_max = extrema(θ_t)
    σ_min, σ_max = 0.13, 0.9

    fig, axes = PyPlot.subplots(2, 1, figsize=(4, 5))
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    mesh_μ = axes[1].pcolormesh(zeros(100, 100), vmin=μ_min, vmax=μ_max, cmap=CMAP_PERMS)
    mesh_σ = axes[2].pcolormesh(zeros(100, 100), vmin=σ_min, vmax=σ_max, cmap=CMAP_PERMS)
    cbar_μ = PyPlot.colorbar(mesh_μ, ticks=[-32, -31, -30])
    cbar_σ = PyPlot.colorbar(mesh_σ)

    cbar_μ.ax.tick_params(labelsize=TEXT_SIZE) 
    cbar_σ.ax.tick_params(labelsize=TEXT_SIZE) 
    cbar_μ.ax.set_ylabel(L"ln(Permeability) (ln(m$^2$))")
    cbar_σ.ax.set_ylabel(L"ln(Permeability) (ln(m$^2$))")

    PyPlot.savefig("plots/poster/colourbars.png", dpi=500)

end

function generate_legend()

    fig, ax = PyPlot.subplots()
    fig.patch.set_facecolor(POSTER_BACKGROUND)

    ax.scatter(1:10, 1:10, color="k", label="Observations", s=10)
    ax.plot(1:10, 1:10, color="k", linestyle="--", label="True Pressures")
    ax.plot(1:10, 1:10, color=POSTER_BLUE, label="Modelled Pressures")

    PyPlot.legend(ncol=3, bbox_to_anchor=(0.95, 1.1), framealpha=0.0, frameon=false)

    PyPlot.savefig("plots/poster/legend.png", dpi=500)

end

plot_forward_problem(grid_f, well_centres, θ_t, u_t, d_obs, t_obs)
# plot_prior_samples(grid_c, pr)
# plot_method_comparison(grid_f, grid_c, u_t, model_f, n_wells, t_obs, d_obs)
# generate_colourbars()
# generate_legend()