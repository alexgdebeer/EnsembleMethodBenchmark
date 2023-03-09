"""A Lokta-Volterra model for testing a set of simulation-intensive algorithms."""

import DifferentialEquations
import Distributions
import LinearAlgebra
import Random

import PyPlot
import Seaborn

include("sim_intensive_inference/sim_intensive_inference.jl")


Random.seed!(16)

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")


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

    G_filled = zeros(n_data, n_steps)
    G_blank = zeros(n_data, n_steps)

    for (i, i_obs) ∈ enumerate(is_obs)
        G_filled[i, i_obs] = 1.0
    end

    return [G_filled G_blank; G_blank G_filled]

end


"""Returns a distance measure between a set of outputs and the data."""
function d(x_a, x_b)
    return sum((x_a .- x_b).^2)
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


"""Plot the posterior found using ABC."""
function plot_abc_posterior(θs, is)

    fig, ax = PyPlot.subplots(1, 2)
    fig.suptitle("ABC: Prior vs Posterior", fontsize = 20)
    fig.set_size_inches(8, 4)

    Seaborn.kdeplot(
        x = [θ[1] for θ ∈ θs],
        y = [θ[2] for θ ∈ θs],
        cmap = "coolwarm", 
        fill = true,
        ax = ax[1]
    )

    Seaborn.kdeplot(
        x = [θ[1] for θ ∈ θs[is]],
        y = [θ[2] for θ ∈ θs[is]],
        cmap = "coolwarm", 
        fill = true,
        ax = ax[2]
    )

    ax[1].set_title("Prior")
    ax[2].set_title("Posterior")

    for i ∈ 1:2

        ax[i].scatter([1], [1], marker = "x", color = "k")
        ax[i].set_xlabel("\$a\$")
        ax[i].set_ylabel("\$b\$")

    end 

    PyPlot.tight_layout()
    PyPlot.savefig("plots/abc_posterior.pdf")
    PyPlot.clf()

end


"""Plot the predictions made using the parameters from the posterior."""
function plot_abc_posterior_predictions(ts_obs, ys_obs, ts, ys, is, ys_true)

    ts_obs_x, ts_obs_y = ts_obs[1:n_data], ts_obs[(n_data+1):end]
    ys_obs_x, ys_obs_y = ys_obs[1:n_data], ys_obs[(n_data+1):end]

    fig, ax = PyPlot.subplots(1, 2)
    fig.suptitle("ABC: Predictions", fontsize = 20)
    fig.set_size_inches(8, 4)

    ax[1].plot(ts, ys_true[1:n_steps], c = "k", zorder = 5)
    ax[2].plot(ts, ys_true[(n_steps+1):end], c = "k", zorder = 5)

    ax[1].scatter(ts_obs_x, ys_obs_x, marker = "o", color = "k", zorder = 4)
    ax[2].scatter(ts_obs_y, ys_obs_y, marker = "^", color = "k", zorder = 4)

    for y ∈ ys[is]

        ax[1].plot(ts, y[1:n_steps], c = "tab:green", zorder = 2)
        ax[2].plot(ts, y[(n_steps+1):end], c = "tab:green", zorder = 2)

    end

    for y ∈ ys[setdiff(1:10_000, is)]

        ax[1].plot(ts, y[1:n_steps], c = "tab:gray", zorder = 1, alpha = 0.5)
        ax[2].plot(ts, y[(n_steps+1):end], c = "tab:gray", zorder = 1, alpha = 0.5)

    end

    ax[1].set_ylim(0, 5.0)
    ax[2].set_ylim(0, 5.0)

    ax[1].set_title("\$x(t)\$")
    ax[2].set_title("\$y(t)\$")

    ax[1].set_xlabel("\$t\$")
    ax[2].set_xlabel("\$t\$")
    
    ax[1].set_ylabel("\$x(t)\$")
    ax[2].set_ylabel("\$y(t)\$")

    PyPlot.tight_layout()
    PyPlot.savefig("plots/abc_posterior_predictions.pdf")
    PyPlot.clf()

end


function plot_abc_smc_posterior(θs, is, ws)

    fig, ax = PyPlot.subplots(3, 2)
    fig.suptitle("ABC SMC: Prior vs Posterior", fontsize = 20)
    fig.set_size_inches(10, 12)

    # Plot the prior
    Seaborn.kdeplot(
        x = [θ[1] for θ ∈ θs[1]],
        y = [θ[2] for θ ∈ θs[1]],
        cmap = "coolwarm", 
        fill = true,
        ax = ax[1, 1]
    )

    ax[1, 1].scatter([1], [1], marker = "x", color = "k")

    ax[1, 1].set_title("Prior")
    ax[1, 1].set_xlabel("\$a\$")
    ax[1, 1].set_ylabel("\$b\$")
    
    for i ∈ 2:6

        r = floor(Int64, (i + 1) / 2)
        c = 2 - (i % 2)

        # Re-sample with the weights
        θs_r = [
            SimIntensiveInference.sample_from_population(
                θs[i-1][is[i-1]], ws[i-1]
            ) for _ ∈ 1:1000
        ]

        Seaborn.kdeplot(
            x = [θ[1] for θ ∈ θs_r],
            y = [θ[2] for θ ∈ θs_r],
            cmap = "coolwarm", 
            fill = true,
            ax = ax[r, c]
        )

        ax[r, c].scatter([1], [1], marker = "x", color = "k")

        ax[r, c].set_title("Population $(i-1)")
        ax[r, c].set_xlabel("\$a\$")
        ax[r, c].set_ylabel("\$b\$")

    end

    PyPlot.tight_layout()
    PyPlot.savefig("plots/abc_smc_posterior.pdf")
    PyPlot.clf()

end


function plot_abc_smc_posterior_predictions(ts_obs, ys_obs, ts, ys, is, ys_true)

    ts_obs_x, ts_obs_y = ts_obs[1:n_data], ts_obs[(n_data+1):end]
    ys_obs_x, ys_obs_y = ys_obs[1:n_data], ys_obs[(n_data+1):end]

    fig, ax = PyPlot.subplots(5, 2)
    fig.suptitle("ABC SMC: Predictions", fontsize = 20)
    fig.set_size_inches(8, 12)
    
    for i ∈ 1:5

        ax[i, 1].plot(ts, ys_true[1:n_steps], c = "k", zorder = 5)
        ax[i, 2].plot(ts, ys_true[(n_steps+1):end], c = "k", zorder = 5)

        ax[i, 1].scatter(ts_obs_x, ys_obs_x, marker = "o", color = "k", zorder = 4)
        ax[i, 2].scatter(ts_obs_y, ys_obs_y, marker = "^", color = "k", zorder = 4)

        for y ∈ ys[i][is[i]]

            ax[i, 1].plot(ts, y[1:n_steps], c = "tab:green", zorder = 2, alpha = 0.5)
            ax[i, 2].plot(ts, y[(n_steps+1):end], c = "tab:green", zorder = 2, alpha = 0.5)
    
        end
    
        for y ∈ ys[i][setdiff(1:end, is[i])]
    
            ax[i, 1].plot(ts, y[1:n_steps], c = "tab:gray", zorder = 1, alpha = 0.1)
            ax[i, 2].plot(ts, y[(n_steps+1):end], c = "tab:gray", zorder = 1, alpha = 0.1)
    
        end

        ax[i, 1].set_ylim(0, 5.0)
        ax[i, 2].set_ylim(0, 5.0)

        ax[i, 1].set_title("Population $(i)")
        ax[i, 2].set_title("Population $(i)")

        ax[i, 1].set_xlabel("\$t\$")
        ax[i, 2].set_xlabel("\$t\$")
        
        ax[i, 1].set_ylabel("\$x(t)\$")
        ax[i, 2].set_ylabel("\$y(t)\$")

    end

    PyPlot.tight_layout()
    PyPlot.savefig("plots/abc_smc_posterior_predictions.pdf")
    PyPlot.clf()

end


true_posterior = true
abc = false
probabilistic_abc = false
abc_smc = false
probabilistic_abc_mcmc = true


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
xs_obs .+= rand(dist, 2 * n_data)

plot_solution(ts, xs_true, ts_obs, xs_obs)

# ----------------
# General-purpose variables
# ----------------

G = generate_obs_operator(is_obs)

# Specify a Gaussian prior
μπ = zeros(2)
σπ = 3.0
Σπ = σπ ^ 2 .* Matrix(LinearAlgebra.I, 2, 2)
π = SimIntensiveInference.GaussianPrior(μπ, Σπ)

# Specify an error model
μₑ = zeros(2 * n_data)
Σₑ = σₑ^2 .* Matrix(1.0 * LinearAlgebra.I, 2n_data, 2n_data)
e = SimIntensiveInference.GaussianError(μₑ, Σₑ)

# ----------------
# Algorithms
# ----------------

if true_posterior

    n_points = 100

    xs = range(0.8, 1.2, n_points)
    ys = range(0.8, 1.2, n_points)

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

    PyPlot.contourf(xs, ys, density, cmap = "coolwarm")
    PyPlot.scatter([1], [1], c = "k", marker = "x", label = "True parameter values")

    PyPlot.gca().set_aspect("equal")

    PyPlot.title("True posterior density", fontsize = 20)
    PyPlot.xlabel("\$a\$", fontsize = 16)
    PyPlot.ylabel("\$b\$", fontsize = 16)
    PyPlot.legend(fontsize = 16)

    PyPlot.savefig("plots/lokta_volterra/true_posterior.pdf")
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


if probabilistic_abc

    N = 10_000_000

    # Define an acceptance kernel 
    K = SimIntensiveInference.GaussianAcceptanceKernel(16.0 .* Σₑ)

    SimIntensiveInference.run_probabilistic_abc(π, f, xs_obs, G, N, K)

end


if abc_smc

    σκ = 0.5
    Σκ = σκ .* Matrix(1.0LinearAlgebra.I, 2, 2)
    κ = SimIntensiveInference.GaussianPerturbationKernel(Σκ)

    T = 5
    n = 500
    α₁ = 0.05
    αs = [0.7, 0.5, 0.4, 0.2958]

    θs, ys, ds, is, ws = SimIntensiveInference.run_abc_smc(
        π, f, e, xs_obs, G, d, κ, T, n, α₁, αs
    )

    plot_abc_smc_posterior(θs, is, ws)
    plot_abc_smc_posterior_predictions(ts_obs, xs_obs, ts, ys, is, xs_true)
    # 4.78146474496098
    # 16.16298990060011

end


if probabilistic_abc_mcmc

    N = 1_000_000

    σκ = 0.1
    Σκ = σκ .* Matrix(1.0LinearAlgebra.I, 2, 2)
    κ = SimIntensiveInference.GaussianPerturbationKernel(Σκ)

    # Define an acceptance kernel 
    E = SimIntensiveInference.GaussianAcceptanceKernel(Σₑ)

    θs, ys = SimIntensiveInference.run_probabilistic_abc_mcmc(
        π,
        f,
        xs_obs, 
        G, 
        κ,
        E,
        N
    )

    inds = [i for i ∈ 1:N if i % 5000 == 0]

    Seaborn.kdeplot(
        x = [θ[1] for θ ∈ θs][inds],
        y = [θ[2] for θ ∈ θs][inds],
        cmap = "coolwarm", 
        fill = true
    )

    PyPlot.scatter([1], [1], c = "k", marker = "x", label = "True parameter values")

    PyPlot.xlim(0.8, 1.2)
    PyPlot.ylim(0.8, 1.2)
    PyPlot.gca().set_aspect("equal")

    PyPlot.title("Probabilistic ABC-MCMC posterior density", fontsize = 20)
    PyPlot.xlabel("\$a\$", fontsize = 16)
    PyPlot.ylabel("\$b\$", fontsize = 16)
    PyPlot.legend(fontsize = 16)

    PyPlot.savefig("plots/lokta_volterra/abc_mcmc_posterior.pdf")

end