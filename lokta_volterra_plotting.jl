"""Defines some functions for plotting the results of a number of 
simulation-intensive algorithms applied to a Lokta-Volterra model."""

using Distributions
using LinearAlgebra
using Random
using Statistics

using LaTeXStrings
import PyPlot
import Seaborn

include("lokta_volterra_model.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")


Random.seed!(16)

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")

# Define plotting font sizes
const TITLE_SIZE = 20
const LABEL_SIZE = 16
const SMALL_SIZE = 8
const TINY_SIZE = 5


const true_posterior = false

const RUN_SMC = false
const RUN_MCMC = false
const RUN_IBIS = false
const RUN_RML = false
const RUN_RTO = false
const RUN_ENKF = true

const RUN_ABC = false
const RUN_ABC_SMC = false
const RUN_ABC_MCMC = false


"""Plots the outputs of the model run with the true values of the parameters,
and the noisy data."""
function plot_lv_system(ts, ys_true, ts_obs, ys_obs)

    PyPlot.plot(ts, ys_true[1:LVModel.n_steps], label="\$x(t)\$")
    PyPlot.plot(ts, ys_true[(LVModel.n_steps+1):end], label="\$y(t)\$")
    
    PyPlot.scatter(ts_obs[1:8], ys_obs[1:8], marker="o")
    PyPlot.scatter(ts_obs[9:end], ys_obs[9:end], marker="^")

    PyPlot.title("True LV System and Observations", fontsize=TITLE_SIZE)
    PyPlot.xlabel(L"t", fontsize=LABEL_SIZE)
    PyPlot.ylabel(L"x(t), y(t)", fontsize=LABEL_SIZE)
    PyPlot.legend(fontsize=LABEL_SIZE)

    PyPlot.savefig("plots/lokta_volterra.pdf")
    PyPlot.clf()

end


"""Generates the posterior joint and marginal densities on a grid of parameter 
values."""
function generate_true_posterior(as, bs)

    joint_density = hcat([
        [
            SimIntensiveInference.density(π, [a, b]) * 
            SimIntensiveInference.density(L, G * LVModel.f([a, b])) for b ∈ bs
        ] for a ∈ as
    ]...)

    marg_a = dropdims(sum(joint_density, dims=1), dims=1)
    marg_b = dropdims(sum(joint_density, dims=2), dims=2)
    marg_a ./= area(as, marg_a)
    marg_b ./= area(bs, marg_b)

    return joint_density, marg_a, marg_b

end


function plot_true_posterior(as, bs, joint_density, marg_a, marg_b)

    g = Seaborn.JointGrid(xlim=(minimum(as), maximum(as)), ylim=(minimum(bs), maximum(bs)))

    g.ax_joint.contourf(as, bs, joint_density, cmap="coolwarm", levels=8)
    g.ax_joint.scatter(x=[1], y=[1], c="k", marker="x", label="True parameters")
    
    g.ax_marg_x.plot(as, marg_a, c="tab:gray")
    g.ax_marg_y.plot(marg_b, bs, c="tab:gray")

    g.ax_marg_x.axvline(x=1, c="k")
    g.ax_marg_y.axhline(y=1, c="k")

    PyPlot.suptitle("True posterior", fontsize=TITLE_SIZE)
    g.ax_joint.set_xlabel(L"a", fontsize=LABEL_SIZE)
    g.ax_joint.set_ylabel(L"b", fontsize=LABEL_SIZE)
    
    g.ax_joint.legend(fontsize=SMALL_SIZE)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/lokta_volterra/true_posterior.pdf")
    PyPlot.clf()

end


function plot_approx_posterior(
    θs_sampled,
    as, bs, post_marg_a, post_marg_b, 
    title, save_path; caption=nothing
)

    as_sampled = [θ[1] for θ ∈ θs_sampled]
    bs_sampled = [θ[2] for θ ∈ θs_sampled]

    g = Seaborn.JointGrid(xlim=(minimum(as), maximum(as)), ylim=(minimum(bs), maximum(bs)))

    Seaborn.kdeplot(x=as_sampled, y=bs_sampled, ax=g.ax_joint, fill=true, cmap="coolwarm", levels=9, bw_adjust=2.0)
    g.ax_joint.scatter(x=[1], y=[1], c="k", marker="x", label="True parameters")
    
    Seaborn.kdeplot(x=as_sampled, ax=g.ax_marg_x, c="#4358CB", label="Sampled density")
    Seaborn.kdeplot(y=bs_sampled, ax=g.ax_marg_y, c="#4358CB")

    g.ax_marg_x.axvline(x=1, c="k", label="True parameters")
    g.ax_marg_y.axhline(y=1, c="k")
    
    g.ax_marg_x.plot(as, post_marg_a, c="tab:gray", ls="--", label="True posterior density")
    g.ax_marg_y.plot(post_marg_b, bs, c="tab:gray", ls="--")

    g.ax_marg_x.set_title(title, fontsize=TITLE_SIZE)
    g.ax_joint.set_xlabel(L"a", fontsize=LABEL_SIZE)
    g.ax_joint.set_ylabel(L"b", fontsize=LABEL_SIZE)
    
    g.ax_joint.legend(fontsize=SMALL_SIZE)
    g.ax_marg_x.legend(fontsize=SMALL_SIZE, frameon=false, loc="lower right")

    if caption !== nothing 
        fig = PyPlot.gcf()
        fig.supxlabel(caption, x=0.01, ha="left", fontsize=SMALL_SIZE)
    end

    g.ax_joint.set_facecolor("#4358CB")

    PyPlot.tight_layout()
    PyPlot.savefig(save_path)
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


"""Calculates the area under a curve using the trapezoidal rule."""
function area(x, y)
    return 0.5 * sum((x[i+1]-x[i]) * (y[i+1]+y[i]) for i ∈ 1:(length(x)-1))
end


# Define the true parameters, and the standard deviation of the measurement 
# noise to be added
θs_true = [1.0, 1.0]
σₑ = 0.25

ts, ys_true, is_obs, ts_obs, ys_obs = LVModel.generate_data(θs_true, σₑ)
G = LVModel.generate_obs_operator(is_obs)

# Specify a Gaussian prior
μπ = zeros(2)
σπ = 3.0
Σπ = σπ^2 .* Matrix(I, 2, 2)
π = SimIntensiveInference.GaussianPrior(μπ, Σπ)

# Define an error model
μₑ = zeros(2LVModel.N_DATA)
Σₑ = σₑ^2 .* Matrix(I, 2LVModel.N_DATA, 2LVModel.N_DATA)
e = SimIntensiveInference.GaussianError(μₑ, Σₑ)

# Define the likelihood 
L = SimIntensiveInference.GaussianLikelihood(ys_obs, Σₑ)

n_points = 200
as = collect(range(0.6, 1.4, n_points))
bs = collect(range(0.6, 1.4, n_points))

post_joint, post_marg_a, post_marg_b = generate_true_posterior(as, bs)

# plot_lv_system(ts, ys_true, ts_obs, ys_obs)
# plot_true_posterior(as, bs, post_joint, post_marg_a, post_marg_b)


if RUN_ABC

    # Specify number of simulations to run and the proportion of runs to accept
    N = 100_000
    α = 0.005

    θs, ys, ds, is = SimIntensiveInference.run_abc(
        π, f, e, ys_obs, G, LVModel.Δ, N, α
    )

end


if RUN_SMC

    K = SimIntensiveInference.ComponentwiseGaussianKernel()

    # Define sequence of error distributions to evaluate 
    ms = [10, 8, 6, 5, 4, 3, 2, 1.5, 1.25, 1.15, 1.05, 1.0]
    Σs = [(m*σₑ)^2 * Matrix(I, 2n_data, 2n_data) for m ∈ ms]
    Es = [SimIntensiveInference.GaussianAcceptanceKernel(Σ) for Σ ∈ Σs]
    
    # Define number of iterations and number of particles
    T = length(ms)
    N = 5000

    θs, ys, ws = @time SimIntensiveInference.run_smc(
        π, f, ys_obs, G, K, T, N, Es
    )

    title = "SMC intermediate distributions"
    save_path = "plots/lokta_volterra/smc_intermediate_distributions.pdf"
    nrows, ncols = 3, 4
    plot_intermediate_distributions(θs, ws, ms, nrows, ncols, title, save_path)

    # Re-sample with replacement from the final population
    θs = [SimIntensiveInference.sample_from_population(θs[T], ws[T]) for _ ∈ 1:10N]

    title = "SMC posterior"
    save_path = "plots/lokta_volterra/smc_posterior.pdf"
    caption = "5000 samples from the true posterior."
    plot_posterior(θs, title, save_path, caption=caption)

end


if RUN_ABC_SMC

    K = SimIntensiveInference.MultivariateGaussianKernel()

    N = 1000
    εs = [20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.75, 1.5]
    T = length(εs)

    θs, ys, ws = @time SimIntensiveInference.run_abc_smc(
        π, f, e, ys_obs, G, d, K, T, N, εs
    )

    title = "ABC SMC intermediate distributions"
    save_path = "plots/lokta_volterra/abc_smc_intermediate_distributions.pdf"
    nrows, ncols = 3, 4
    plot_intermediate_distributions(θs, ws, εs, nrows, ncols, title, save_path)

    # Re-sample with replacement from the final population
    θs = [SimIntensiveInference.sample_from_population(θs[T], ws[T]) for _ ∈ 1:10_000]

    title = "ABC SMC posterior"
    save_path = "plots/lokta_volterra/abc_smc_posterior.pdf"
    caption = L"\noindent 1000 samples from approximate posterior $p(\theta | \rho(y_{\textrm{obs}}, y) \leq \varepsilon)$.\\ $\rho(y_{\textrm{obs}}, y)$ = sum of squared differences between noisy model outputs and observations.\\ $\varepsilon$ = 1.5."
    plot_posterior(θs, title, save_path, caption=caption)

end


if RUN_MCMC 

    # Define the likelihood
    L = SimIntensiveInference.GaussianLikelihood(xs_obs, Σₑ)

    # Define a perturbation kernel
    σₖ = 0.05
    Σₖ = σₖ^2 .* Matrix(1.0LinearAlgebra.I, 2, 2)
    K = SimIntensiveInference.StaticGaussianKernel(Σₖ)

    # Define number of simulations to run
    N = 100_000

    θs = @time SimIntensiveInference.run_mcmc(π, f, L, G, K, N)

    title = "MCMC posterior"
    save_name = "plots/lokta_volterra/mcmc_posterior.pdf"
    caption = "100,000 samples from the true posterior."
    plot_posterior(θs, title, save_name, caption=caption)

    title = "MCMC diagnostic curves"
    save_name = "plots/lokta_volterra/mcmc_diagnostic_curves.pdf"
    plot_diagnostic_curves(θs, title, save_name)

    title = "MCMC autocorrelations"
    save_name = "plots/lokta_volterra/mcmc_autocorrelations.pdf"
    ks = 0:10:500
    plot_autocorrelations(θs, ks, title, save_name)

end


if RUN_ABC_MCMC 

    σₖ = 0.05
    Σₖ = σₖ^2 .* Matrix(1.0LinearAlgebra.I, 2, 2)
    K = SimIntensiveInference.StaticGaussianKernel(Σₖ)

    N = 100_000
    ε = 1.5

    t1 = time()

    θs = SimIntensiveInference.run_abc_mcmc(π, f, e, xs_obs, G, d, K, N, ε)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

    title = "ABC MCMC posterior"
    save_name = "plots/lokta_volterra/abc_mcmc_posterior.pdf"
    caption = L"\noindent 100,000 samples from approximate posterior $p(\theta | \rho(y_{\textrm{obs}}, y) \leq \varepsilon)$.\\ $\rho(y_{\textrm{obs}}, y)$ = sum of squared differences between noisy model outputs and observations.\\ $\varepsilon$ = 1.5."
    plot_posterior(θs, title, save_name, caption=caption)

    title = "ABC MCMC diagnostic curves"
    save_name = "plots/lokta_volterra/abc_mcmc_diagnostic_curves.pdf"
    plot_diagnostic_curves(θs, title, save_name)

    title = "ABC MCMC autocorrelations"
    save_name = "plots/lokta_volterra/abc_mcmc_autocorrelations.pdf"
    ks = 0:10:500
    plot_autocorrelations(θs, ks, title, save_name)

end


if RUN_IBIS

    Gs = [generate_obs_operator(is_obs[1:i]) for i ∈ 1:n_data]
    y_batches = [vcat(xs_obs[1:i], xs_obs[(n_data+1):(n_data+i)]) for i ∈ 1:n_data]
    Ls = [
        SimIntensiveInference.GaussianLikelihood(y_batches[i], σₑ^2*Matrix(1.0LinearAlgebra.I, 2i, 2i)) 
            for i ∈ 1:n_data
    ]

    N = 10_000

    θs = SimIntensiveInference.run_ibis(π, f, Ls, y_batches, Gs, N)

    T = length(y_batches)

    title = "IBIS posterior"
    save_path = "plots/lokta_volterra/ibis_posterior.pdf"
    caption = "10,000 samples from the true posterior."
    plot_posterior(θs[T], title, save_path, caption=caption)

    title = "IBIS intermediate distributions"
    save_path = "plots/lokta_volterra/ibis_intermediate_distributions.pdf"
    nrows, ncols = 3, 3
    plot_intermediate_distributions(θs, T, nrows, ncols, title, save_path)

end


if RUN_RML

    N = 1000
    θ_MAP, θs = @time SimIntensiveInference.run_rml(LVModel.f, π, L, G, N)

    title = "RML posterior"
    save_path = "plots/lokta_volterra/rml_posterior.pdf"
    caption = "$N samples from an approximate posterior."

    plot_approx_posterior(
        θs,
        as, bs, post_marg_a, post_marg_b, 
        title, save_path; caption=caption
    )

end


if RUN_RTO

    N = 1_000
    θ_MAP, θs, ws = @time SimIntensiveInference.run_rto(LVModel.f, π, L, G, N)

    title = "RTO posterior"
    save_path = "plots/lokta_volterra/rto_posterior.pdf"
    caption = "$N samples from an approximate posterior."

    plot_approx_posterior(
        θs,
        as, bs, post_marg_a, post_marg_b, 
        title, save_path; caption=caption
    )

    θs_reweighted = [SimIntensiveInference.sample_from_population(θs, ws) for _ ∈ 1:10N]

    title = "Reweighted RTO posterior"
    save_path = "plots/lokta_volterra/rto_posterior_reweighted.pdf"
    caption = "$N weighted samples from the true posterior."

    plot_approx_posterior(
        θs_reweighted,
        as, bs, post_marg_a, post_marg_b,
        title, save_path, caption=caption
    )

end


if RUN_ENKF 

    ts_obs = ts_obs[1:LVModel.N_DATA]
    ys_obs = [[ys_obs[i], ys_obs[i+LVModel.N_DATA]] for i ∈ 1:LVModel.N_DATA]

    # Specify a Gaussian prior
    μ_p = [1.0, 0.5]
    σ_p = 0.1
    Σ_p = σ_p^2 .* Matrix(I, 2, 2)
    π_p = SimIntensiveInference.GaussianPrior(μ_p, Σ_p)

    N_e = 100

    SimIntensiveInference.run_enkf(
        LVModel.f, LVModel.g, ts_obs, ys_obs, σₑ, π, π_p, N_e
    )

end


# L_θ = LinearAlgebra.cholesky(inv(π.Σ)).U  
# L_ϵ = LinearAlgebra.cholesky(inv(L.Σ)).U

# # Define augmented system 
# f̃(θ) = [L_ϵ*G*f(θ); L_θ*θ]
# ỹ = [L_ϵ*L.μ; L_θ*π.μ]

# # Calculate the MAP estimate
# map_func(θ) = 0.5sum((f̃(θ)-ỹ).^2)
# res = Optim.optimize(map_func, [1.0, 1.0], Optim.NelderMead())
# θ_MAP = Optim.minimizer(res)

# J̃θ_MAP = ForwardDiff.jacobian(f̃, θ_MAP)
# Q = Matrix(LinearAlgebra.qr(J̃θ_MAP))
# LinearAlgebra.normalize!.(eachcol(Q))

# θs = []
# ws = []

# println("Beginning plotting...")
# n_points = 100
# as = collect(range(0.8, 1.2, n_points))
# bs = collect(range(0.8, 1.2, n_points))

# joint_density = hcat([
#     [
#         abs(LinearAlgebra.det(Q'*ForwardDiff.jacobian(f̃, [a,b]))) * exp(-0.5sum((Q'*(f̃([a,b])-ỹ)).^2)) for b ∈ bs
#     ] for a ∈ as
# ]...)

# PyPlot.contourf(as, bs, joint_density, cmap="coolwarm")
# PyPlot.gca().set_aspect("equal")
# PyPlot.savefig("plots/lokta_volterra/rto_density.pdf")