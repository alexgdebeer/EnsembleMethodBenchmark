"""Runs EnRML on the linear model."""

include("linear_model.jl")
include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const β_0 = 0.5
const N_e = 10_000

θs = SimIntensiveInference.run_batch_enrml(
    LinearModel.f, 
    LinearModel.g,
    π,  
    LinearModel.YS_O, 
    LinearModel.σ_ϵ, 
    β_0,
    N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "Linear Model: EnRML Posterior",
    "$(LinearModel.PLOTS_DIR)/enrml/enrml_posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble size: $N_e."
)