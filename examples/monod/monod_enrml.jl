"""Runs EnRML on the MONOD model."""

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const β_0 = 0.5
const N_e = 100

θs, ys = SimIntensiveInference.run_batch_enrml(
    MONODModel.f, MONODModel.g, π, 
    MONODModel.YS_O, MONODModel.σ_ϵ, 
    β_0, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    MONODModel.θ1S, MONODModel.θ2S, 
    MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
    "MONOD: EnRML Posterior",
    "$(MONODModel.PLOTS_DIR)/enrml/enrml_posterior.pdf";
    caption="Ensemble size: $N_e."
)

Plotting.plot_monod_posterior_predictions(
    MONODModel.XS, ys, 
    MONODModel.XS_O, MONODModel.YS_O, 
    "MONOD: EnRML Posterior Predictions",
    "$(MONODModel.PLOTS_DIR)/enrml/enrml_posterior_predictions.pdf"
)