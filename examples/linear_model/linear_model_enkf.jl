"""Runs the EnKF for parameter estimation on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const N_e = 1000

θs = SimIntensiveInference.run_enkf(
    LinearModel.f, 
    π, [LinearModel.θS_T[1]],
    LinearModel.TS_O, LinearModel.YS_O[:,:]', 
    LinearModel.σ_ϵ, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "Linear Model: Final EnKF Posterior",
    "$(LinearModel.PLOTS_DIR)/enkf/true_enkf_posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble size: $N_e."
)
