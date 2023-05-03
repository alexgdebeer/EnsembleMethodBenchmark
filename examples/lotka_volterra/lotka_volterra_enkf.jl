"""Runs the EnKF on the LV model."""

include("lotka_volterra_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)
const N_e = 100

θs = SimIntensiveInference.run_enkf(
    LVModel.f, 
    π, LVModel.Y_0,
    LVModel.TS_O, LVModel.YS_O, 
    LVModel.σ_ϵ, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B,
    "LV: Final EnKF Posterior",
    "$(LVModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
    θs_t=LVModel.θS_T,
    caption="Ensemble size: $N_e."
)
