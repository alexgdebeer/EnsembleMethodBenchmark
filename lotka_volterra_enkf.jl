using Distributions
using LinearAlgebra
using Statistics

include("lotka_volterra_model.jl")
include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior parameters
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)

# Define observation noise and ensemble size
const σ_ϵ = LVModel.σ_ϵ
const N_e = 10_000

θs = SimIntensiveInference.run_enkf_simplified(
    LVModel.f, π, 
    LVModel.TS_O, LVModel.YS_O, 
    σ_ϵ, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
    "EnKF: Final Posterior",
    "$(LVModel.PLOTS_DIR)/enkf/test_post.pdf";
    θs_t=LVModel.θS_T,
    caption="Ensemble size: $N_e"
)