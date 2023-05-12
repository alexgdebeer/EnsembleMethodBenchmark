"""Runs the Levenberg-Marquardt iterative ensemble smoother on the MONOD model."""

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const γ = 10
const l_max = 20
const N_e = 1000

θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
    MONODModel.f, MONODModel.g, π, 
    MONODModel.YS_O, MONODModel.σ_ϵ,
    γ, l_max, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs[end]), 
    MONODModel.θ1S, MONODModel.θ2S, 
    MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
    "MONOD: LM-EnRML Posterior",
    "$(MONODModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
    caption="Ensemble size: $N_e."
)

Plotting.plot_monod_posterior_predictions(
    MONODModel.XS, ys[end],
    MONODModel.XS_O, MONODModel.YS_O,
    "MONOD: LM-EnRML Posterior Predictions",
    "$(MONODModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
)