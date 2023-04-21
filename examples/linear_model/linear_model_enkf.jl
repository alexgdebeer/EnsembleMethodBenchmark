"""Runs the EnKF for parameter estimation on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const N_e = 10_000

const MDA = true
const αs = [9.333, 7.0, 4.0, 2.0]

if MDA 

    θs = SimIntensiveInference.run_enkf_mda(
        LinearModel.H, 
        π, 
        LinearModel.TS_O, 
        LinearModel.YS_O[:,:]', 
        LinearModel.σ_ϵ, 
        αs,
        N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: Final EnKF MDA Posterior",
        "$(LinearModel.PLOTS_DIR)/enkf/enkf_mda_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_enkf_simplified(
        LinearModel.H, 
        π, 
        LinearModel.TS_O, 
        LinearModel.YS_O[:,:]', 
        LinearModel.σ_ϵ, 
        N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: Final EnKF Posterior",
        "$(LinearModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

end