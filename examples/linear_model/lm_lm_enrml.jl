"""Runs the Levenberg-Marquardt iterative ensemble smoother on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const γ = 10
const l_max = 20
const N_e = 10_000

θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
    LinearModel.f, LinearModel.g, π, 
    LinearModel.YS_O, LinearModel.σ_ϵ,
    γ, l_max, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs[end]), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "Linear Model: LM-EnRML Posterior",
    "$(LinearModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble size: $N_e."
)

Plotting.plot_lm_posterior_predictions(
    LinearModel.TS,
    ys[end],
    LinearModel.YS_T,
    LinearModel.TS_O,
    LinearModel.YS_O,
    "Linear Model: LM-EnRML Posterior Predictions",
    "$(LinearModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
)