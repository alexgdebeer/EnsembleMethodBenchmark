"""Runs the Levenberg-Marquardt iterative ensemble smoother."""

include("simple_nonlinear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

const π = SimIntensiveInference.GaussianPrior(SimpleNonlinear.μ_π, SimpleNonlinear.Γ_π)
const γ = 10
const l_max = 20
const N_e = 10_000

θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
    SimpleNonlinear.f, SimpleNonlinear.g, π, 
    vec(SimpleNonlinear.YS_O), SimpleNonlinear.σ_ϵ,
    γ, l_max, N_e
)

import PyPlot
PyPlot.hist(vec(θs[end]), density=true)
PyPlot.savefig("test2.pdf")