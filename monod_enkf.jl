"""Runs the EnKF for parameter estimation on the MONOD model."""

include("monod_model.jl")
include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const N_e = 100

θs = SimIntensiveInference.run_enkf_simplified(
    MONODModel.H, π, 
    MONODModel.XS, MONODModel.YS_O[:,:]', 
    MONODModel.σ_ϵ, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    MONODModel.θ1S, MONODModel.θ2S, 
    MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
    "MONOD Model: Final EnKF Posterior",
    "$(MONODModel.PLOTS_DIR)/enkf/posterior.pdf",
    caption="Ensemble size: $N_e."
)