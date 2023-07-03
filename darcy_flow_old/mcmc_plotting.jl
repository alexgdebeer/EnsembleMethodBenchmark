using JLD2
using LaTeXStrings
using PyPlot
using Statistics

include("problem_setup.jl")

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

post_mean_vs_truth = false
post_stds = false
chain_means = false
trace_plots = false
marginal_plots = false 
pressure_predictions = true

n_warmup = 500_000
n_skip = 10
n_chains = 6
chain_length = 1_000_000

# Read in the MCMC results 
@load "mcmc_results_coarse.jld2" ps us

# Remove warm-up iterations and thin the chain if necessary
ps = ps[:,n_warmup+1:n_skip:end,:]
us = us[:,n_warmup+1:n_skip:end,:]

μ_post = reshape(mean(ps, dims=(2,3)), nx_c, ny_c)
    
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

    PyPlot.suptitle("Posterior mean vs truth", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/post_mean_vs_truth.pdf")
    PyPlot.clf()

end

if post_mean_preds_vs_truth

    # Generate the predictions corresponding to the true field
    p = interpolate((xs_f, ys_f), exp.(ps_true), Gridded(Linear()))
    A, b = DarcyFlow.generate_grid(xs_f, ys_f, p, bcs)
    sol = solve(LinearProblem(A, b))
    us_true = reshape(sol.u, nx_f, ny_f)

    # Generate the predictions corresponding to the posterior mean
    p = interpolate((xs_c, ys_c), exp.(μ_post), Gridded(Linear()))
    A, b = DarcyFlow.generate_grid(xs_c, ys_c, p, bcs)
    sol = solve(LinearProblem(A, b))
    us_μ_post = reshape(sol.u, nx_c, ny_c)

    umin = min(minimum(us_μ_post), minimum(us_true))
    umax = max(maximum(us_μ_post), maximum(us_true))

    fig, ax = PyPlot.subplots(1, 2, figsize=(6, 3))

    m_1 = ax[1].pcolormesh(
        xs_c, ys_c, us_true', 
        cmap=:coolwarm, vmin=umin, vmax=umax
    )

    m_2 = ax[2].pcolormesh(
        xs_c, ys_c, us_μ_post',
        cmap=:coolwarm, vmin=umin, vmax=umax
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

    PyPlot.suptitle("Posterior mean vs truth: predictions", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/post_mean_preds_vs_truth.pdf")
    PyPlot.clf()

end

if post_stds

    PyPlot.axes().set_aspect("equal")

    PyPlot.pcolormesh(
        xs_c, ys_c, reshape(std(ps, dims=(2,3)), nx_c, ny_c)',
        cmap=:magma
    )
    PyPlot.colorbar()

    PyPlot.xticks(ticks=[0, 1])
    PyPlot.yticks(ticks=[0, 1])

    PyPlot.title("Posterior standard deviations", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/post_stds.pdf")
    PyPlot.clf()

end

if chain_means

    fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

    for i ∈ 1:2, j ∈ 1:3

        chain = 3(i-1)+j

        μ_θ = reshape(mean(ps[:, :, chain], dims=2), nx_c, ny_c)

        m = ax[i, j].pcolormesh(
            xs_c, ys_c, μ_θ', 
            cmap=:viridis, vmin=pmin, vmax=pmax
        )

        ax[i, j].set_box_aspect(1)
        ax[i, j].set_xticks([0, 1])
        ax[i, j].set_yticks([0, 1])

        PyPlot.colorbar(m, fraction=0.046, pad=0.04, ax=ax[i, j])

        ax[i, j].set_title("Chain $chain", fontsize=12)
        # ax[i, j].set_xlabel(L"x", fontsize=12)
        # ax[i, j].set_ylabel(L"y", fontsize=12)

    end

    PyPlot.suptitle("Chain means", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/chain_means.pdf")
    PyPlot.clf()

end

# Select some indices from the permeability field to use
perm_is = rand(1:nx_c*ny_c, 6)

if marginal_plots

    fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

    for (n, i) ∈ enumerate(perm_is)

        # Find the corresponding subplot row/column
        r = n ≤ 3 ? 1 : 2
        c = n%3 + 1

        # Find the corresponding x and y coordinates
        x = xs_c[(i-1)%nx_c+1] 
        y = ys_c[Int(ceil(i/nx_c))]

        ax[r, c].hist(vec(ps[i,:,:]), bins=20, zorder=1, density=true)
        ax[r, c].axvline(vec(ps_true)[i], color="k", linestyle="--", zorder=2)

        ax[r, c].set_title(L"x"*" = $x, "*L"y"*" = $y")

    end

    PyPlot.suptitle("Marginal permeability distributions at random locations", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/marginals.pdf")
    PyPlot.clf()

end

if trace_plots

    fig, ax = PyPlot.subplots(2, 3, figsize=(8, 5))

    for (n, i) ∈ enumerate(perm_is)

        # Find the corresponding subplot row/column
        r = n ≤ 3 ? 1 : 2
        c = n%3 + 1

        # Find the corresponding x and y coordinates
        x = xs_c[(i-1)%nx_c+1] 
        y = ys_c[Int(ceil(i/nx_c))]

        for j ∈ 1:n_chains
            ax[r, c].plot(n_warmup+1:n_skip:chain_length, ps[i,:,j], zorder=1, linewidth=0.5)
            ax[r, c].axhline(vec(ps_true)[i], color="k", linestyle="--", zorder=2)
        end

        ax[r, c].set_title(L"x"*" = $x, "*L"y"*" = $y")
        ax[r, c].set_xlabel("Iteration number", fontsize=10)
        ax[r, c].set_ylabel("Permeability", fontsize=10)

    end

    PyPlot.suptitle("Trace plots for permeabilities at random locations", fontsize=20)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/mcmc/trace_plots.pdf")
    PyPlot.clf()

end

if posterior_predictions 

    PyPlot.figure(figsize=(5, 4))

    bp = PyPlot.boxplot(
        [vec(us[i,:,:]) for i ∈ 1:8], 
        flierprops=Dict(:marker=>".", :markersize=>2),
        zorder=1
    )

    for i ∈ 1:8
        x_mean = mean(bp["medians"][i].get_xdata())
        PyPlot.errorbar(
            x=x_mean, y=us_o[i], 
            yerr=σ_ϵ,
            color="tab:blue", marker=".", markersize=4,
            zorder=2
        )
    end

    PyPlot.xticks(1:8, y_locs)

    PyPlot.title("Posterior predictions "*L"(x=0.1)", fontsize=20)
    PyPlot.ylabel("Temperature", fontsize=12)
    PyPlot.xlabel(L"y"*" coordinate", fontsize=12)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/darcy_flow/posterior_predictions.pdf")
    PyPlot.clf()

end