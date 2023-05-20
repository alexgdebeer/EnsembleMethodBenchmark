"""Runs the weighted ES-MDA algorithm, as described in Stordal and Elsheik 
(2015)."""

"""Runs the ES on the Lotka-Volterra model."""

using Revise

include("lv_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

Revise.track("examples/lotka_volterra/lv_model.jl")
Revise.track("plotting.jl")
Revise.track("sim_intensive_inference/sim_intensive_inference.jl")

import .LVModel
import .Plotting
import .SimIntensiveInference

using Distributions

# Define the prior and ensemble size
π = MvNormal(LVModel.μ_π, LVModel.Γ_π)
L = MvNormal(LVModel.μ_L, LVModel.Γ_ϵ)

N_e = 5000
αs = collect(60.1:-4:0.1)
αs *= sum(1 ./ αs)

θs, ys, ws = SimIntensiveInference.run_wes_mda(
    LVModel.f, LVModel.g, π, L, αs, N_e
)

inds = SimIntensiveInference.resample_ws(ws[end,:])

Plotting.plot_approx_posterior(
    eachcol(θs[:,inds,end]), 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B,
    "LV: WES-MDA Posterior",
    "$(LVModel.PLOTS_DIR)/es/wes_mda_posterior.pdf";
    θs_t=LVModel.θS_T,
    caption="Ensemble size: $N_e. Iterations: $(length(αs))."
)

# Plotting.plot_lv_posterior_predictions(
#     LVModel.TS, ys[end], LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
#     "LV: ES-MDA Posterior Predictions", 
#     "$(LVModel.PLOTS_DIR)/es/es_mda_posterior_predictions_modified_prior.pdf"
# )