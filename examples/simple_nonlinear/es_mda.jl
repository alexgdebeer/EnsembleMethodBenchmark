"""Runs the ensemble smoother with multiple data assimilation."""

using LaTeXStrings

include("simple_nonlinear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const π = SimIntensiveInference.GaussianPrior(SimpleNonlinear.μ_π, SimpleNonlinear.Γ_π)
const αs = [16.0 for _ ∈ 1:16]
const N_e = 10_000

θs = SimIntensiveInference.run_es_mda(
    SimpleNonlinear.f, SimpleNonlinear.g, π, 
    vec(SimpleNonlinear.YS_O), SimpleNonlinear.σ_ϵ, 
    αs, N_e
)

Plotting.plot_nonlinear_approx_posterior(
    vec(θs), 
    SimpleNonlinear.θS, 
    SimpleNonlinear.POST, 
    SimpleNonlinear.θS_T[1],
    "ES-MDA Posterior",
    "$(SimpleNonlinear.PLOTS_DIR)/es/es_mda_posterior.pdf",
    lims=(0.5, 7.0),
    caption="Ensemble size: $N_e. Iterations: $(length(αs)). "*L"\sigma_\epsilon"*": $(SimpleNonlinear.σ_ϵ)."
)