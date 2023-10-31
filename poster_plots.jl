using HDF5
using LaTeXStrings
using PyCall
using PyPlot

@pyimport matplotlib.animation as anim

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

include("setup.jl")

TITLE_SIZE = 12
LABEL_SIZE = 8
TEXT_SIZE = 8
TICK_SIZE = 6

WELL_TO_PLOT = 4

function plot_forward_problem(well_centres, θ_t, u_t)

    θ_t = reshape(θ_t, grid_f.nx, grid_f.nx)
    u_t = reshape(u_t, grid_f.nx, grid_f.nx, :) ./ 1.0e6

    u_well = model_f.B_wells * vec(u_t)
    inds = 1:length(u_well)
    u_well = u_well[findall(i -> i%9 == WELL_TO_PLOT, inds)]
    
    u_min, u_max = extrema(u_t)

    function set_reservoir_ticks!(ax)
        
        ax.set_xlim((0, 1000))
        ax.set_ylim((0, 1000))
        
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        
        ax.set_xlabel(L"$x$ (m)", fontsize=LABEL_SIZE)
        ax.set_ylabel(L"$y$ (m)", fontsize=LABEL_SIZE)
    
    end

    function make_frame(i::Int)

        println(i)

        axes[2, 1].clear()
        axes[2, 2].clear()

        axes[2, 1].pcolormesh(grid_f.xs, grid_f.xs, u_t[:, :, i+1], 
                              cmap=:turbo, vmin=u_min, vmax=u_max)
        
        set_reservoir_ticks!(axes[2, 1])
        axes[2, 1].set_title("True Pressures", fontsize=TITLE_SIZE)

        axes[2, 2].plot(grid_f.ts[1:i+1], u_well[1:i+1], color="cornflowerblue")

        axes[2, 2].set_ylim(0.98minimum(u_well), 20)
        axes[2, 2].set_xlim(0, grid_f.ts[end])

        axes[2, 2].set_xticks([0, 40, 80, 120])
        axes[2, 2].set_yticks([17, 18, 19, 20])
        axes[2, 2].set_yticklabels(["17", "18", "19", "20"])

        axes[2, 2].set_title("Pressure in Well X", fontsize=TITLE_SIZE)
        axes[2, 2].set_xlabel("Time (days)", fontsize=LABEL_SIZE)
        axes[2, 2].set_ylabel("Pressure (MPa)", fontsize=LABEL_SIZE)
        
    end

    fig, axes = PyPlot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    fig.patch.set_facecolor("lightgrey")

    # Plot well positions
    # TODO: correct the numbering on this
    for (i, c) ∈ enumerate(well_centres) 
        axes[1, 1].scatter(c[1], c[2], color="k", s=4)
        axes[1, 1].text(c[1], c[2]+40, "Well $i", ha="center", fontsize=TEXT_SIZE)
    end
    axes[1, 1].set_facecolor("lightskyblue")

    axes[1, 2].pcolormesh(grid_f.xs, grid_f.xs, θ_t, cmap=:turbo)

    for ax ∈ axes
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        ax.set_box_aspect(1)
    end

    set_reservoir_ticks!(axes[1, 1])
    set_reservoir_ticks!(axes[1, 2])

    axes[1, 1].set_title("Well Locations", fontsize=TITLE_SIZE)
    axes[1, 2].set_title("True Reservoir Permeability", fontsize=TITLE_SIZE)

    PyPlot.tight_layout()

    ani = anim.FuncAnimation(fig, make_frame, frames=size(u_t, 3), 
                             interval=250)
    ani.save(filename="forward_problem.gif", writer="pillow", dpi=200)

end

BATCH_LENGTH = 100

BATCH_INC = 10
BATCH_INDS = BATCH_INC:BATCH_INC:BATCH_LENGTH

function plot_mcmc(n_batches::Int=500)

    function make_frame(i::Int)

        println(i)

        axes[1, 1].set_title("Sample $i")        
        sample_plot.set_data(samples[:, :, i+1])
        cm_plot.set_data(cms[:, :, i+1])
        # axes[1, 2].plot(ls[1:i+1], color="black")

    end

    fig, axes = PyPlot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    fig.patch.set_facecolor("lightgrey")

    axes[1, 2].set_title("Conditional Mean")

    # Read in the MCMC output
    f = h5open("data/mcmc/chain_1.h5", "r")
    θs = reduce(hcat, [f["θs_$i"][:, end] for i ∈ 1:n_batches])
    ls = reduce(vcat, [f["ηs_$i"][end, end] for i ∈ 1:n_batches])
    close(f)

    samples = θs[:, :]
    cms = reduce(hcat, [mean(θs[:, 1:i], dims=2) for i ∈ 1:size(samples, 2)])
    
    samples = reshape(samples, grid_c.nx, grid_c.nx, :)
    cms = reshape(cms, grid_c.nx, grid_c.nx, :)

    θ_min, θ_max = extrema(samples)
    sample_plot = axes[1, 1].imshow(samples[:, :, 1], cmap=:turbo, vmin=θ_min, vmax=θ_max)
    cm_plot = axes[1, 2].imshow(cms[:, :, 1], cmap=:turbo, vmin=θ_min, vmax=θ_max)

    # TODO: compute conditional means and standard deviations.

    n_frames = size(samples, 3)
    ani = anim.FuncAnimation(fig, make_frame, frames=n_frames, 
                             interval=250)

    ani.save(filename="mcmc.gif", writer="pillow", dpi=200, fps=5)


end

plot_forward_problem(well_centres, θ_t, u_t)
# plot_mcmc()