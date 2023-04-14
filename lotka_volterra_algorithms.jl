"""Defines some functions for plotting the results of a number of 
simulation-intensive algorithms applied to a Lotka-Volterra model."""

using Distributions
using LinearAlgebra
using Random
using Statistics

include("lotka_volterra_model.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

const RUN_SMC = false
const RUN_MCMC = false
const RUN_IBIS = false

const RUN_ABC = false
const RUN_ABC_SMC = false
const RUN_ABC_MCMC = false


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