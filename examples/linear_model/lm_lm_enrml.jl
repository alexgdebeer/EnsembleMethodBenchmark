"""Runs the Levenberg-Marquardt iterative ensemble smoother on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const γ = 10
const l_max = 20
const N_e = 1000

θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
    LinearModel.f, LinearModel.g, π, 
    LinearModel.YS_O, LinearModel.σ_ϵ,
    γ, l_max, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs[end]), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "Linear Model: LV-EnRML Posterior",
    "$(LinearModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble size: $N_e."
)