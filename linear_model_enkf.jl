"""Runs the EnKF on the linear model."""

include("linear_model.jl")
include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior 
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)

# Define the ensemble size 
const N_e = 1_000

θs = SimIntensiveInference.run_enkf_simplified(
    LinearModel.f, 
    π, 
    LinearModel.TS_O, 
    LinearModel.YS_O, 
    LinearModel.σ_ϵ, 
    N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "EnKF Posterior",
    "$(LinearModel.PLOTS_DIR)/enkf/posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble with $N_e members."
)