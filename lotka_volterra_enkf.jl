using Distributions
using LinearAlgebra
using Statistics

include("lotka_volterra_model.jl")
include("lotka_volterra_plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")


# Define parameters of the prior for the model state 
const μ_u = [0.75, 0.75]
const σ_u = 0.5
const Γ_u = σ_u^2 * Matrix(I, 2, 2)
const π_u = SimIntensiveInference.GaussianPrior(μ_u, Γ_u)

# Define parameters of the prior for the model parameters 
const μ_θ = [0.0, 0.0]
const σ_θ = 1.0
const Γ_θ = σ_θ^2 * Matrix(I, 2, 2)
const π_θ = SimIntensiveInference.GaussianPrior(μ_θ, Γ_θ)

# Define observation noise and ensemble size
const σ_ϵ = LVModel.σ_ϵ
const N_e = 100

# Define the mapping between the model state and the observations 
H(ys, θs) = ys


us_e, θs_e = @time SimIntensiveInference.run_enkf_params(
    LVModel.f, 
    H, 
    π_u, 
    π_θ,
    LVModel.TS_O, 
    LVModel.YS_O,
    LVModel.T_1,
    σ_ϵ, 
    N_e
)

LVModelPlotting.plot_enkf_states(
    us_e, N_e, LVModel.TS, LVModel.YS_T, LVModel.TS_O, LVModel.YS_O,
    "$(LVModel.PLOTS_DIR)/enkf/state_evolution.pdf"
)

LVModelPlotting.plot_enkf_parameters(
    θs_e, N_e,
    "$(LVModel.PLOTS_DIR)/enkf/parameter_evolution.pdf"
)

# Plot the final posterior
as_final = θs_e[mod.(1:2N_e,2).==1, end]
bs_final = θs_e[mod.(1:2N_e,2).==0, end]

LVModelPlotting.plot_approx_posterior(
    [[a, b] for (a, b) ∈ zip(as_final, bs_final)], 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
    "EnKF: Final Posterior",
    "$(LVModel.PLOTS_DIR)/enkf/parameter_posterior.pdf"
)