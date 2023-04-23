"""Runs the EnKF for parameter estimation on the Lotka-Volterra model."""

include("lotka_volterra_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)
const N_e = 1000

const MDA = false
const αs = [57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0]

if MDA 

    θs = SimIntensiveInference.run_enkf_mda(
        LVModel.H, π, 
        LVModel.TS_O, LVModel.YS_O, 
        LVModel.σ_ϵ, αs, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
        "LV: Final EnKF (MDA) Posterior",
        "$(LVModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_enkf_simplified(
        LVModel.H, π, 
        LVModel.TS_O, LVModel.YS_O, 
        LVModel.σ_ϵ, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
        "LV: Final EnKF MDA Posterior",
        "$(LVModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

end