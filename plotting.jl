module Plotting


using Statistics
using LaTeXStrings
import PyPlot
import Seaborn


PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

const TITLE_SIZE = 20
const LABEL_SIZE = 16
const SMALL_SIZE = 8

const DARK_BLUE = "#4358CB"


"""Calculates the joint and marginal densities of two parameters on a grid of 
values."""
function density_grid(
    θ1s::Vector, 
    θ2s::Vector, 
    density::Function
)::Tuple

    area(x, y) = 0.5sum((x[i+1]-x[i])*(y[i+1]+y[i]) for i ∈ 1:(length(x)-1))

    joint = [density([θ1, θ2]) for θ2 ∈ θ2s, θ1 ∈ θ1s]

    marg_θ1 = vec(sum(joint, dims=1))
    marg_θ2 = vec(sum(joint, dims=2))
    marg_θ1 ./= area(θ1s, marg_θ1)
    marg_θ2 ./= area(θ2s, marg_θ2)

    return joint, marg_θ1, marg_θ2

end


"""Plots the outputs of the LV model run with the true values of the parameters,
and the noisy data."""
function plot_lv_system(
    ts::AbstractVector, 
    ys_t::AbstractMatrix, 
    ts_o::AbstractVector, 
    ys_o::AbstractMatrix,
    fname::AbstractString
)

    PyPlot.plot(ts, ys_t[1, :], label=L"x(t)")
    PyPlot.plot(ts, ys_t[2, :], label=L"y(t)")
    
    PyPlot.scatter(ts_o, ys_o[1, :], marker="o")
    PyPlot.scatter(ts_o, ys_o[2, :], marker="^")

    PyPlot.title("True LV System and Observations", fontsize=TITLE_SIZE)
    PyPlot.xlabel(L"t", fontsize=LABEL_SIZE)
    PyPlot.ylabel(L"x(t), y(t)", fontsize=LABEL_SIZE)
    PyPlot.legend(fontsize=LABEL_SIZE)

    PyPlot.savefig(fname)
    PyPlot.clf()

end


"""Plots the observations used for the MONOD model."""
function plot_monod_obs(
    xs::AbstractVector,
    ys_o::AbstractVector,
    fname::AbstractString
)

    PyPlot.scatter(xs, ys_o)
    
    PyPlot.title("MONOD Model Observations", fontsize=TITLE_SIZE)
    PyPlot.xlabel(L"x", fontsize=LABEL_SIZE)
    PyPlot.ylabel(L"y", fontsize=LABEL_SIZE)

    PyPlot.savefig(fname)
    PyPlot.clf()

end


"""Plots the outputs of the linear model run with the true values of the 
parameters, and the noisy data."""
function plot_linear_model(
    ts::AbstractVector,
    ys_t::AbstractVector,
    ts_o::AbstractVector,
    ys_o::AbstractVector,
    fname::AbstractString
)

    PyPlot.plot(ts, ys_t)
    PyPlot.scatter(ts_o, ys_o)

    PyPlot.title("True Linear Model and Observations", fontsize=TITLE_SIZE)
    PyPlot.xlabel(L"t", fontsize=LABEL_SIZE)
    PyPlot.ylabel(L"y(t)", fontsize=LABEL_SIZE)

    PyPlot.savefig(fname)
    PyPlot.clf()

end


"""Plots the joint posterior density of two parameters."""
function plot_density_grid(
    θ1_vals::AbstractVector, 
    θ2_vals::AbstractVector, 
    joint::AbstractMatrix, 
    marg_θ1::AbstractVector, 
    marg_θ2::AbstractVector, 
    title::AbstractString, 
    fname::AbstractString;
    θs_t::Union{AbstractVector,Nothing}=nothing
)

    # Initialise the grid for plotting
    g = Seaborn.JointGrid(xlim=extrema(θ1_vals), ylim=extrema(θ2_vals))

    # Plot the joint and marginal densities
    g.ax_joint.contourf(θ1_vals, θ2_vals, joint, cmap="coolwarm", levels=8)
    g.ax_marg_x.plot(θ1_vals, marg_θ1, c="tab:gray")
    g.ax_marg_y.plot(marg_θ2, θ2_vals, c="tab:gray")

    PyPlot.suptitle(title, fontsize=TITLE_SIZE)
    g.ax_joint.set_xlabel(L"\theta_{1}", fontsize=LABEL_SIZE)
    g.ax_joint.set_ylabel(L"\theta_{2}", fontsize=LABEL_SIZE)

    # Plot the true values of the parameters
    if θs_t !== nothing
        g.ax_joint.scatter(
            x=θs_t[1], y=θs_t[2], c="k", 
            marker="x", label="True parameters")
        g.ax_marg_x.axvline(x=θs_t[1], c="k")
        g.ax_marg_y.axhline(y=θs_t[2], c="k")
        g.ax_joint.legend(fontsize=SMALL_SIZE)
    end

    PyPlot.tight_layout()
    PyPlot.savefig(fname)
    PyPlot.clf()

end


"""Plots a set of samples from a posterior distribution."""
function plot_approx_posterior(
    θs_s, 
    θ1s::AbstractVector, 
    θ2s::AbstractVector, 
    marg_θ1::Vector, 
    marg_θ2::Vector, 
    title::AbstractString, 
    fname::AbstractString; 
    θs_t::Union{Nothing,AbstractVector}=nothing, 
    caption::Union{Nothing,AbstractString}=nothing
)

    θ1s_s = [θ[1] for θ ∈ θs_s]
    θ2s_s = [θ[2] for θ ∈ θs_s]

    g = Seaborn.JointGrid(xlim=extrema(θ1s), ylim=extrema(θ2s))

    # Plot the sampled values
    Seaborn.kdeplot(x=θ1s_s, y=θ2s_s, ax=g.ax_joint, fill=true, cmap="coolwarm", levels=9, bw_adjust=2.0)
    Seaborn.kdeplot(x=θ1s_s, ax=g.ax_marg_x, c=DARK_BLUE, label="Sampled density")
    Seaborn.kdeplot(y=θ2s_s, ax=g.ax_marg_y, c=DARK_BLUE)
    
    # Plot the true posterior marginals
    g.ax_marg_x.plot(θ1s, marg_θ1, c="tab:gray", ls="--", label="True posterior density")
    g.ax_marg_y.plot(marg_θ2, θ2s, c="tab:gray", ls="--")

    # Plot the true parameter values
    if θs_t !== nothing
        g.ax_joint.scatter(
            x=θs_t[1], y=θs_t[2], c="k", 
            marker="x", label="True parameters")
        g.ax_marg_x.axvline(x=θs_t[1], c="k", label="True parameters")
        g.ax_marg_y.axhline(y=θs_t[2], c="k")
        g.ax_joint.legend(fontsize=SMALL_SIZE)
    end

    g.ax_marg_x.set_title(title, fontsize=TITLE_SIZE)
    g.ax_joint.set_xlabel(L"\theta_{1}", fontsize=LABEL_SIZE)
    g.ax_joint.set_ylabel(L"\theta_{2}", fontsize=LABEL_SIZE)
    g.ax_marg_x.legend(fontsize=SMALL_SIZE, frameon=false)

    if caption !== nothing 
        PyPlot.gcf().supxlabel(caption, x=0.01, ha="left", fontsize=SMALL_SIZE)
    end

    g.ax_joint.set_facecolor(DARK_BLUE)

    PyPlot.tight_layout()
    PyPlot.savefig(fname)
    PyPlot.clf()

end


function plot_intermediate_distributions(
    θs, ws, εs, nrows, ncols, 
    title, save_path; caption = nothing
)

    fig, axes = PyPlot.subplots(nrows, ncols, sharey="row", figsize=(3*ncols, 3*nrows))

    T = length(εs)

    pmap = reshape(1:(nrows*ncols), (nrows, ncols))'

    for t ∈ 1:T

        θs_sample = [SimIntensiveInference.sample_from_population(θs[t], ws[t]) for _ ∈ 1:10_000]
        a, b = [θ[1] for θ ∈ θs_sample], [θ[2] for θ ∈ θs_sample]

        ax = axes[pmap[t]]
        ax.set_facecolor("#4358CB")
        ax.set_aspect("equal", adjustable="datalim")

        Seaborn.kdeplot(x=a, y=b, ax=ax, fill=true, cmap="coolwarm", levels=9, bw_adjust=2.0)
        ax.scatter(x=[1], y=[1], c="k", marker="x")

        ax.set_title("Iteration $(t) ("*L"$\varepsilon$ = "*"$(εs[t]))", fontsize=LABEL_SIZE)

    end

    for t ∈ (T+1):(nrows*ncols)
        axes[pmap[t]].set_axis_off()
    end

    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.supxlabel(L"a", fontsize=LABEL_SIZE)
    fig.supylabel(L"b", fontsize=LABEL_SIZE)

    if caption !== nothing 
        fig = PyPlot.gcf()
        fig.supxlabel(caption, x=0.01, ha="left", fontsize=10)
    end

    PyPlot.tight_layout()
    PyPlot.savefig(save_path)
    PyPlot.clf()

end


function plot_intermediate_distributions(
    θs, T, nrows, ncols, title, save_path; caption = nothing
)

    fig, axes = PyPlot.subplots(nrows, ncols, sharey="row", figsize=(3*ncols, 3*nrows))

    pmap = reshape(1:(nrows*ncols), (nrows, ncols))'

    for t ∈ 1:T

        a, b = [θ[1] for θ ∈ θs[t]], [θ[2] for θ ∈ θs[t]]

        ax = axes[pmap[t]]
        ax.set_facecolor("#4358CB")
        ax.set_aspect("equal", adjustable="datalim")

        Seaborn.kdeplot(x=a, y=b, ax=ax, fill=true, cmap="coolwarm", levels=9, bw_adjust=2.0)
        ax.scatter(x=[1], y=[1], c="k", marker="x")

        # Hack to clean up the axes
        ylim = ax.get_ylim()
        ax.set_ylim(max(ylim[1], -1.0), min(ylim[2], 2.0))

        ax.set_title("Iteration $(t)", fontsize=LABEL_SIZE)

    end

    for t ∈ (T+1):(nrows*ncols)
        axes[pmap[t]].set_axis_off()
    end

    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.supxlabel(L"a", fontsize=LABEL_SIZE)
    fig.supylabel(L"b", fontsize=LABEL_SIZE)

    if caption !== nothing 
        fig = PyPlot.gcf()
        fig.supxlabel(caption, x=0.01, ha="left", fontsize=10)
    end

    PyPlot.tight_layout()
    PyPlot.savefig(save_path)
    PyPlot.clf()

end


function plot_diagnostic_curves(θs, title, save_name)

    fig, ax = PyPlot.subplots(2)

    ax[1].plot([θ[1] for θ ∈ θs], lw=0.5)
    ax[2].plot([θ[2] for θ ∈ θs], lw=0.5)

    fig.suptitle(title, fontsize=TITLE_SIZE)
    ax[1].set_xlabel("Iteration", fontsize=LABEL_SIZE)
    ax[1].set_ylabel(L"a", fontsize=LABEL_SIZE)
    ax[2].set_xlabel("Iteration", fontsize=LABEL_SIZE)
    ax[2].set_ylabel(L"b", fontsize=LABEL_SIZE)

    PyPlot.tight_layout()
    PyPlot.savefig(save_name)
    PyPlot.clf()

end


function plot_autocorrelations(θs, ks, title, save_name)

    as, bs = [θ[1] for θ ∈ θs], [θ[2] for θ ∈ θs]
    
    ρs_as = [Statistics.cor(as[1:(end-k)], as[(k+1):end]) for k ∈ ks]
    ρs_bs = [Statistics.cor(bs[1:(end-k)], bs[(k+1):end]) for k ∈ ks]

    fig, ax = PyPlot.subplots(2)

    ax[1].stem(ks, ρs_as, markerfmt=" ", basefmt="k")
    ax[2].stem(ks, ρs_bs, markerfmt=" ", basefmt="k")

    fig.suptitle(title, fontsize=TITLE_SIZE)
    ax[1].set_xlabel(L"k", fontsize=LABEL_SIZE)
    ax[1].set_ylabel(L"\rho(a_{t}, a_{t+k})", fontsize=LABEL_SIZE)
    ax[2].set_xlabel(L"k", fontsize=LABEL_SIZE)
    ax[2].set_ylabel(L"\rho(b_{t}, b_{t+k})", fontsize=LABEL_SIZE)

    PyPlot.tight_layout()
    PyPlot.savefig(save_name)
    PyPlot.clf()

end


function plot_lv_posterior_predictions(
    ts::AbstractVector,
    ys::AbstractMatrix, 
    ts_o::AbstractVector,
    ys_o::AbstractMatrix,
    title::AbstractString,
    fname::AbstractString
)::Nothing

    # Extract the indices corresponding to the predator / prey predictions
    ns = vec(1:size(ys, 1))

    is_y1 = ns[ns .% 2 .== 1]
    is_y2 = ns[ns .% 2 .== 0]

    ys_y1 = ys[is_y1, :]
    ys_y2 = ys[is_y2, :]

    qs_y1 = reduce(hcat, [quantile(c, [0.025, 0.975]) for c ∈ eachcol(ys_y1)])
    qs_y2 = reduce(hcat, [quantile(c, [0.025, 0.975]) for c ∈ eachcol(ys_y2)])

    fig, ax = PyPlot.subplots(1, 2, figsize=(8, 4))
    
    # Plot the observations
    ax[1].scatter(ts_o, ys_o[1, :], c="k", zorder=3)
    ax[2].scatter(ts_o, ys_o[2, :], c="k", zorder=3)
    
    # Plot the model outputs
    ax[1].plot(ts, ys_y1', c="gray", alpha=0.8, zorder=1)
    ax[2].plot(ts, ys_y2', c="gray", alpha=0.8, zorder=1)

    # Plot the central 95% of each set of outputs
    ax[1].plot(ts, qs_y1', c="red", zorder=2)
    ax[2].plot(ts, qs_y2', c="red", zorder=2)

    ax[1].set_ylim(-0.1, 5)
    ax[2].set_ylim(-0.1, 5)

    PyPlot.suptitle(title, fontsize=TITLE_SIZE)
    ax[1].set_title(L"y_{1}", fontsize=LABEL_SIZE)
    ax[2].set_title(L"y_{2}", fontsize=LABEL_SIZE) 

    PyPlot.tight_layout()
    PyPlot.savefig(fname)

    return nothing

end


"""Plots the evolution of the states over time as found using the EnKF."""
function plot_enkf_states(us_e, N_e, ts, ys_t, ts_o, ys_o, fname)

    fig, ax = PyPlot.subplots(1, 2, figsize=(7, 4))

    # Extract the observations generated by the ensemble
    y1s = us_e[mod.(1:2N_e,2).==1, :]
    y2s = us_e[mod.(1:2N_e,2).==0, :]

    # Compute some quantiles of the ensemble
    y1_qs = mapslices(c -> quantile(c, [0.025, 0.975]), y1s, dims=1)
    y2_qs = mapslices(c -> quantile(c, [0.025, 0.975]), y2s, dims=1)

    # Plot the true model states
    ax[1].plot(ts, ys_t[1, :], c="k", ls="--", zorder=4)
    ax[2].plot(ts, ys_t[2, :], c="k", ls="--", zorder=4)

    # Plot the observations
    ax[1].scatter(ts_o, ys_o[1, :], color="k", marker="x", zorder=4)
    ax[2].scatter(ts_o, ys_o[2, :], color="k", marker="x", zorder=4)

    # Plot the ensemble
    ax[1].plot(ts, y1s', color="gray", alpha=0.5, zorder=2)
    ax[2].plot(ts, y2s', color="gray", alpha=0.5, zorder=2)

    # Plot the quantiles 
    ax[1].plot(ts, y1_qs', color="red", zorder=3)
    ax[2].plot(ts, y2_qs', color="red", zorder=3)

    ax[1].set_ylim((-1, 4))
    ax[2].set_ylim((-1, 4))

    ax[1].set_xlabel(L"t", fontsize=LABEL_SIZE)
    ax[2].set_xlabel(L"t", fontsize=LABEL_SIZE)
    ax[1].set_ylabel(L"y_{1}(t)", fontsize=LABEL_SIZE)
    ax[2].set_ylabel(L"y_{2}(t)", fontsize=LABEL_SIZE)

    fig.suptitle("EnKF: State Evolution", fontsize=TITLE_SIZE)

    PyPlot.tight_layout()
    PyPlot.savefig(fname)

end


"""Plots the evolution of the paramters over time as found using the EnKF."""
function plot_enkf_parameters(θs_e, N_e, fname)

    fig, ax = PyPlot.subplots(1, 2, figsize=(7, 4))

    # Extract the evolution of each parameter
    as = θs_e[mod.(1:2N_e,2).==1, :]
    bs = θs_e[mod.(1:2N_e,2).==0, :]

    # Compute some quantiles of the ensemble
    a_qs = mapslices(c -> quantile(c, [0.025, 0.975]), as, dims=1)
    b_qs = mapslices(c -> quantile(c, [0.025, 0.975]), bs, dims=1)

    # Plot the ensemble
    ax[1].plot(as', color="gray", alpha=0.5, zorder=2)
    ax[2].plot(bs', color="gray", alpha=0.5, zorder=2)

    # Plot the quantiles 
    ax[1].plot(a_qs', color="red", zorder=3)
    ax[2].plot(b_qs', color="red", zorder=3)

    ax[1].set_xlabel("Iteration", fontsize=LABEL_SIZE)
    ax[2].set_xlabel("Iteration", fontsize=LABEL_SIZE)
    ax[1].set_ylabel(L"a", fontsize=LABEL_SIZE)
    ax[2].set_ylabel(L"b", fontsize=LABEL_SIZE)

    fig.suptitle("EnKF: Parameter Evolution", fontsize=TITLE_SIZE)

    PyPlot.tight_layout()
    PyPlot.savefig(fname)

end


end