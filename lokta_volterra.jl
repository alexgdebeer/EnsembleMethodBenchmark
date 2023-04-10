"""A Lokta-Volterra model for testing a set of simulation-intensive algorithms."""

import DifferentialEquations
import Distributions
import LinearAlgebra
import Random
import Statistics

using LaTeXStrings
import PyPlot
import Seaborn

include("sim_intensive_inference/sim_intensive_inference.jl")


Random.seed!(16)

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")

# Define plotting font sizes
TITLE_SIZE = 20
LABEL_SIZE = 16
SMALL_SIZE = 8
TINY_SIZE = 5


# Define a set of t values to solve over
t_start = 0
t_stop = 15
n_steps = 151
ts = range(t_start, t_stop, n_steps)

# Define the number of parameters, and the number of times at which data is 
# generated
n_data = 8


"""Runs the forward model."""
function f(θs)

    """Returns derivative of x at a given time."""
    function dxdt(xs::Vector, θs, t::Real)::Vector
        
        x, y = xs
        a, b = θs

        return [a*x - x*y, b*x*y - y]

    end

    # Define initial condition
    x₀ = [1.0, 0.5]

    prob = DifferentialEquations.ODEProblem(dxdt, x₀, (t_start, t_stop), θs)
    
    sol = DifferentialEquations.solve(
        prob,
        DifferentialEquations.AutoTsit5(DifferentialEquations.Rosenbrock23()),
        tstops = ts, saveat = ts
    )

    # Return everything as a single row, with the x measurements followed by
    # the y measurements
    return vec(hcat(sol.u...)')

end


"""Generates the observation operator."""
function generate_obs_operator(is_obs)

    n = length(is_obs)

    G_filled = zeros(n, n_steps)
    G_blank = zeros(n, n_steps)

    for (i, i_obs) ∈ enumerate(is_obs)
        G_filled[i, i_obs] = 1.0
    end

    return [G_filled G_blank; G_blank G_filled]

end


"""Returns a distance measure between a set of outputs and the data."""
function d(x_a, x_b)
    return sum((x_a .- x_b).^2)
end


function Δ_uniform(x_a, x_b)
    return max(abs.(x_a-x_b)...)
end


"""Plots the true values of the model outputs and the noisy data."""
function plot_solution(ts, xs_true, ts_obs, xs_obs)

    PyPlot.plot(ts, xs_true[1:n_steps], label = "\$x(t)\$")
    PyPlot.plot(ts, xs_true[(n_steps+1):end], label = "\$y(t)\$")
    
    PyPlot.scatter(ts_obs[1:8], xs_obs[1:8], marker = "o")
    PyPlot.scatter(ts_obs[9:end], xs_obs[9:end], marker = "^")

    PyPlot.title("Underlying LV system and observations")
    PyPlot.xlabel("\$t\$")
    PyPlot.ylabel("\$x(t)\$, \$y(t)\$")
    PyPlot.legend()

    PyPlot.savefig("plots/lokta_volterra.pdf")
    PyPlot.clf()

end


function plot_posterior(θs, title, save_path; plot_uniform_posterior=false, caption=nothing)

    a, b = [θ[1] for θ ∈ θs], [θ[2] for θ ∈ θs]

    g = Seaborn.JointGrid(xlim=(0.6, 1.4), ylim=(0.6, 1.4))

    Seaborn.kdeplot(x=a, y=b, ax=g.ax_joint, fill=true, cmap="coolwarm", levels=9, bw_adjust=2.0)
    g.ax_joint.scatter(x=[1], y=[1], c="k", marker="x", label="True parameters")
    
    Seaborn.kdeplot(x=a, ax=g.ax_marg_x, c="#4358CB", label="Sampled density")
    Seaborn.kdeplot(y=b, ax=g.ax_marg_y, c="#4358CB")

    g.ax_marg_x.axvline(x=1, c="k", label="True parameters")
    g.ax_marg_y.axhline(y=1, c="k")
    
    g.ax_marg_x.plot(xs, marg_x, c="tab:gray", ls="--", label="True posterior density")
    g.ax_marg_y.plot(marg_y, ys, c="tab:gray", ls="--")

    if plot_uniform_posterior
        g.ax_marg_x.plot(xs, marg_x_uniform, c="tab:purple", ls="--", label="Density of posterior\nwith uniform likelihood")
        g.ax_marg_y.plot(marg_y_uniform, ys, c="tab:purple", ls="--")
    end

    g.ax_marg_x.set_title(title, fontsize=TITLE_SIZE)
    g.ax_joint.set_xlabel(L"a", fontsize=LABEL_SIZE)
    g.ax_joint.set_ylabel(L"b", fontsize=LABEL_SIZE)
    
    g.ax_joint.legend(fontsize=TINY_SIZE)
    g.ax_marg_x.legend(fontsize=TINY_SIZE, frameon=false, loc="lower right")

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


true_posterior = true
uniform_posterior = true

abc = false

smc = false
smc_uniform = false
abc_smc = false
abc_smc_uniform = false

mcmc = false
mcmc_uniform = false
abc_mcmc = false
abc_mcmc_uniform = false

ibis = true


# ----------------
# Data generation
# ----------------

# Define a set of true values for the parameters and observations
θs_true = [1.0, 1.0]
xs_true = f(θs_true)

# Choose a random set of times to generate data at
is_obs = sort(rand(1:n_steps, n_data))

ts_obs = repeat(ts[is_obs], 2)
xs_obs = vec(vcat(xs_true[is_obs, :], xs_true[is_obs .+ n_steps, :]))

# Add independent Gaussian noise to the data
σₑ = 0.25
dist = Distributions.Normal(0.0, σₑ)
xs_obs .+= rand(dist, 2*n_data)

plot_solution(ts, xs_true, ts_obs, xs_obs)

# ----------------
# General-purpose variables
# ----------------

G = generate_obs_operator(is_obs)

# Specify a Gaussian prior
μπ = zeros(2)
σπ = 3.0
Σπ = σπ^2 .* Matrix(LinearAlgebra.I, 2, 2)
π = SimIntensiveInference.GaussianPrior(μπ, Σπ)

# Specify an error model
μₑ = zeros(2 * n_data)
Σₑ = σₑ^2 .* Matrix(1.0 * LinearAlgebra.I, 2n_data, 2n_data)
e = SimIntensiveInference.GaussianError(μₑ, Σₑ)

# ----------------
# Algorithms
# ----------------

if true_posterior

    n_points = 200

    xs = range(0.6, 1.4, n_points)
    ys = range(0.6, 1.4, n_points)

    density = zeros(Float64, n_points, n_points)

    # Define the likelihood
    L = SimIntensiveInference.GaussianLikelihood(xs_obs, Σₑ)

    for (j, x) ∈ enumerate(xs)
        for (i, y) ∈ enumerate(ys)

            θ = [x, y]

            # Evaluate the posterior density at each set of parameters
            density[i, j] = SimIntensiveInference.density(π, θ) * 
                SimIntensiveInference.density(L, G * f(θ))
        
        end
    end

    # Estimate the marginal densities
    marg_x = dropdims(sum(density, dims=1), dims=1)
    marg_y = dropdims(sum(density, dims=2), dims=2)
    marg_x ./= area(xs, marg_x)
    marg_y ./= area(ys, marg_y)

    g = Seaborn.JointGrid(xlim=(0.6, 1.4), ylim=(0.6, 1.4))

    g.ax_joint.contourf(xs, ys, density, cmap="coolwarm", levels=8)
    g.ax_joint.scatter(x=[1], y=[1], c="k", marker="x", label="True parameters")
    
    g.ax_marg_x.plot(xs, marg_x, c="tab:gray")
    g.ax_marg_y.plot(marg_y, ys, c="tab:gray")

    g.ax_marg_x.axvline(x=1, c="k")
    g.ax_marg_y.axhline(y=1, c="k")

    PyPlot.suptitle("True posterior", fontsize=20)
    g.ax_joint.set_xlabel(L"a", fontsize=16)
    g.ax_joint.set_ylabel(L"b", fontsize=16)
    
    g.ax_joint.legend(fontsize=10)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/lokta_volterra/true_posterior.pdf")
    PyPlot.clf()

end


if uniform_posterior

    n_points = 200

    xs = range(0.6, 1.4, n_points)
    ys = range(0.6, 1.4, n_points)

    density = zeros(Float64, n_points, n_points)

    lbs = xs_obs .- 0.5
    ubs = xs_obs .+ 0.5
    L = SimIntensiveInference.UniformLikelihood(lbs, ubs)

    for (j, x) ∈ enumerate(xs)
        for (i, y) ∈ enumerate(ys)

            θ = [x, y]

            # Evaluate the posterior density at each set of parameters
            density[i, j] = SimIntensiveInference.density(π, θ) * 
                SimIntensiveInference.density(L, G * f(θ))
        
        end
    end

    # Estimate the marginal densities
    marg_x_uniform = dropdims(sum(density, dims=1), dims=1)
    marg_y_uniform = dropdims(sum(density, dims=2), dims=2)
    marg_x_uniform ./= area(xs, marg_x_uniform)
    marg_y_uniform ./= area(ys, marg_y_uniform)

    g = Seaborn.JointGrid(xlim=(0.6, 1.4), ylim=(0.6, 1.4))

    g.ax_joint.contourf(xs, ys, density, cmap="coolwarm", levels=8)
    g.ax_joint.scatter(x=[1], y=[1], c="k", marker="x", label="True parameters")
    
    g.ax_marg_x.plot(xs, marg_x_uniform, c="tab:purple")
    g.ax_marg_y.plot(marg_y_uniform, ys, c="tab:purple")

    g.ax_marg_x.axvline(x=1, c="k")
    g.ax_marg_y.axhline(y=1, c="k")

    PyPlot.suptitle("Posterior with uniform likelihood", fontsize=20)
    g.ax_joint.set_xlabel(L"a", fontsize=16)
    g.ax_joint.set_ylabel(L"b", fontsize=16)
    
    g.ax_joint.legend(fontsize=10)

    PyPlot.tight_layout()
    PyPlot.savefig("plots/lokta_volterra/uniform_posterior.pdf")
    PyPlot.clf()

end


if abc

    # Specify number of simulations to run and proportion of runs to accept
    N = 100_000
    α = 0.005

    θs, ys, ds, is = SimIntensiveInference.run_abc(
        π, f, e, xs_obs, G, d, N, α
    )

    plot_abc_posterior(θs, is)
    plot_abc_posterior_predictions(ts_obs, xs_obs, ts, ys, is, xs_true)

end


if smc

    K = SimIntensiveInference.ComponentwiseGaussianKernel()

    # Define sequence of error distributions to evaluate 
    ms = [10, 8, 6, 5, 4, 3, 2, 1.5, 1.25, 1.15, 1.05, 1.0]
    Σs = [(m*σₑ)^2 * Matrix(1.0LinearAlgebra.I, 2n_data, 2n_data) for m ∈ ms]
    Es = [SimIntensiveInference.GaussianAcceptanceKernel(Σ) for Σ ∈ Σs]
    
    T = length(ms)
    N = 5000

    t1 = time()

    #  TODO: fix ys (conflicts with another variable)
    θs, _, ws = SimIntensiveInference.run_smc(π, f, xs_obs, G, K, T, N, Es)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

    title = "SMC intermediate distributions"
    save_path = "plots/lokta_volterra/smc_intermediate_distributions.pdf"
    nrows, ncols = 3, 4
    plot_intermediate_distributions(θs, ws, ms, nrows, ncols, title, save_path)

    # Re-sample with replacement from the final population
    θs = [SimIntensiveInference.sample_from_population(θs[T], ws[T]) for _ ∈ 1:10_000]

    title = "SMC posterior"
    save_path = "plots/lokta_volterra/smc_posterior.pdf"
    caption = "5000 samples from the true posterior."
    plot_posterior(θs, title, save_path, caption=caption)

end


if smc_uniform

    K = SimIntensiveInference.ComponentwiseGaussianKernel()

    # Define sequence of error distributions to evaluate 
    δs = [5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.5]
    δs = [repeat([δ], 2n_data) for δ ∈ δs]

    Es = [SimIntensiveInference.UniformAcceptanceKernel(-δ, δ) for δ ∈ δs]
    
    T = length(δs)
    N = 1000

    #  TODO: fix ys (conflicts with another variable)
    θs, _, ws = SimIntensiveInference.run_smc_b(π, f, xs_obs, G, K, T, N, Es)

    # Re-sample with replacement from the final population
    θs = [SimIntensiveInference.sample_from_population(θs[T], ws[T]) for _ ∈ 1:10_000]

    title = "SMC posterior (uniform likelihood)"
    save_path = "plots/lokta_volterra/smc_posterior_uniform.pdf"
    caption = L"\noindent 1000 samples from posterior where the true Gaussian likelihood ($\mathcal{N}(y_{\textrm{obs}}, 0.25^{2}\textrm{I})$) has been replaced by a\\uniform likelihood ($\mathcal{U}(y_{\textrm{obs}}-0.5, y_{\textrm{obs}}+0.5)$)."
    plot_posterior(θs, title, save_path, plot_uniform_posterior=true, caption=caption)

end


if abc_smc

    K = SimIntensiveInference.MultivariateGaussianKernel()

    N = 1000
    εs = [20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.75, 1.5]
    T = length(εs)

    t1 = time()

    θs, _, _, ws = SimIntensiveInference.run_abc_smc(π, f, e, xs_obs, G, d, K, T, N, εs)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

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


if abc_smc_uniform

    K = SimIntensiveInference.MultivariateGaussianKernel()

    N = 1000
    εs = [5.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5]
    T = length(εs)

    t1 = time()

    θs, _, _, ws = SimIntensiveInference.run_abc_smc(π, f, e, xs_obs, G, Δ_uniform, K, T, N, εs)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

    # Re-sample with replacement from the final population
    θs = [SimIntensiveInference.sample_from_population(θs[T], ws[T]) for _ ∈ 1:10_000]

    title = "ABC SMC posterior (uniform distance)"
    save_path = "plots/lokta_volterra/abc_smc_posterior_uniform.pdf"
    caption = L"\noindent 1000 samples from approximate posterior $p(\theta | \rho(y_{\textrm{obs}}, y) \leq \varepsilon)$.\\ $\rho(y_{\textrm{obs}}, y) = \textrm{max}\{|y_{\textrm{obs}} - y|\}$, where $y$ denotes the (non-noisy) model output.\\ $\varepsilon$ = 0.5."
    plot_posterior(θs, title, save_path, plot_uniform_posterior=true, caption=caption)

end


if mcmc 

    # Define the likelihood
    L = SimIntensiveInference.GaussianLikelihood(xs_obs, Σₑ)

    # Define a perturbation kernel
    σκ = 0.05
    Σκ = σκ^2 .* Matrix(1.0LinearAlgebra.I, 2, 2)
    κ = SimIntensiveInference.StaticGaussianKernel(Σκ)

    # Define number of simulations to run
    N = 100_000

    t1 = time()

    θs = SimIntensiveInference.run_mcmc(π, f, L, G, κ, N)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

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


if mcmc_uniform

    # Define the likelihood 0.43 for same variance
    lbs = xs_obs .- 0.5
    ubs = xs_obs .+ 0.5
    L = SimIntensiveInference.UniformLikelihood(lbs, ubs)

    # Define a perturbation kernel
    σₖ = 0.05
    Σₖ = σₖ^2 .* Matrix(1.0LinearAlgebra.I, 2, 2)
    K = SimIntensiveInference.StaticGaussianKernel(Σₖ)

    # Define number of simulations to run
    N = 100_000

    θs = SimIntensiveInference.run_mcmc(π, f, L, G, K, N)

    title = "MCMC posterior (uniform likelihood)"
    save_name = "plots/lokta_volterra/mcmc_posterior_uniform.pdf"
    caption = L"\noindent 100,000 samples from posterior where the true Gaussian likelihood ($\mathcal{N}(y_{\textrm{obs}}, 0.25^{2}\textrm{I})$) has been replaced by a\\uniform likelihood ($\mathcal{U}(y_{\textrm{obs}}-0.5, y_{\textrm{obs}}+0.5)$)."
    plot_posterior(θs, title, save_name, plot_uniform_posterior=true, caption=caption)

end


if abc_mcmc 

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


if abc_mcmc_uniform

    σₖ = 0.05
    Σₖ = σₖ^2 .* Matrix(1.0LinearAlgebra.I, 2, 2)
    K = SimIntensiveInference.StaticGaussianKernel(Σₖ)

    N = 100_000
    ε = 0.5

    t1 = time()

    θs = SimIntensiveInference.run_abc_mcmc(π, f, e, xs_obs, G, Δ_uniform, K, N, ε)

    t2 = time()
    @info("Elapsed time: $((t2-t1)/60) mins.")

    title = "ABC MCMC posterior (uniform distance)"
    save_name = "plots/lokta_volterra/abc_mcmc_posterior_uniform.pdf"
    caption = L"\noindent 100,000 samples from approximate posterior $p(\theta | \rho(y_{\textrm{obs}}, y) \leq \varepsilon)$.\\ $\rho(y_{\textrm{obs}}, y) = \textrm{max}\{|y_{\textrm{obs}} - y|\}$, where $y$ denotes the (non-noisy) model output.\\ $\varepsilon$ = 0.5."
    plot_posterior(θs, title, save_name, plot_uniform_posterior=true, caption=caption)

end


if ibis

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