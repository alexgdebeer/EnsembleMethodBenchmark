"""Runs RTO on the MONOD model."""

using LinearAlgebra

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")


# Define the prior and likelihood
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const L = SimIntensiveInference.GaussianLikelihood(MONODModel.YS_O, MONODModel.Γ_ϵ)

# Define the number of samples to draw
const N = 10_000


θ_MAP, Q, θs, ws = SimIntensiveInference.run_rto(
    MONODModel.f, MONODModel.g, π, L, N
)

Plotting.plot_approx_posterior(
    θs, 
    MONODModel.θ1S, MONODModel.θ2S, 
    MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
    "MONOD Model: Uncorrected RTO Posterior",
    "$(MONODModel.PLOTS_DIR)/rml_rto/rto_posterior_uncorrected.pdf",
    caption="$N uncorrected samples."
)

θs_r = SimIntensiveInference.resample_population(θs, ws, N=N)

Plotting.plot_approx_posterior(
    θs_r, 
    MONODModel.θ1S, MONODModel.θ2S, 
    MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
    "MONOD Model: Corrected RTO Posterior",
    "$(MONODModel.PLOTS_DIR)/rml_rto/rto_posterior_corrected.pdf",
    caption="$N re-weighted samples."
)