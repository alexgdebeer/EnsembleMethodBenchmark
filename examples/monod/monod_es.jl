"""Runs the ES on the MONOD model."""

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const N_e = 10_000

# Specify whether multiple data assimilation will occur, and if so, the α 
# values to use
const MDA = true
const αs = [16.0 for _ ∈ 1:16]

if MDA

    θs, ys = SimIntensiveInference.run_es_mda(
        MONODModel.f, MONODModel.g, π, 
        MONODModel.YS_O, MONODModel.σ_ϵ, 
        αs, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: ES-MDA Posterior",
        "$(MONODModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_monod_posterior_predictions(
        MONODModel.XS, ys[end], 
        MONODModel.XS_O, MONODModel.YS_O, 
        "MONOD: ES-MDA Posterior Predictions",
        "$(MONODModel.PLOTS_DIR)/es/es_mda_posterior_predictions.pdf"
    )

else

    θs = SimIntensiveInference.run_ensemble_smoother(
        MONODModel.f, 
        MONODModel.g,
        π,  
        MONODModel.YS_O, 
        MONODModel.σ_ϵ, 
        N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: ES Posterior",
        "$(MONODModel.PLOTS_DIR)/es/es_posterior.pdf";
        caption="Ensemble size: $N_e."
    )

end