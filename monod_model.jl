"""Defines a simple MONOD model (Bardsley 2014)."""
module MONODModel


using LinearAlgebra

include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

const PLOTS_DIR = "plots/monod"

# Define the factor vector (x) and observations (y)
const XS = [28, 55, 83, 110, 138, 225, 375]
const YS_O = [0.053, 0.060, 0.112, 0.105, 0.099, 0.122, 0.125]
const N_IS = length(YS_O)

# Define the prior
const μ_π = [0.15, 50.0]
const σs_π = [0.2, 50.0]
const Γ_π = diagm(σs_π.^2)
const π = SimIntensiveInference.GaussianPrior(μ_π, Γ_π)

# Define the likelihood
const σ_ϵ = 0.012
const Γ_ϵ = σ_ϵ^2 * Matrix(I, N_IS, N_IS)
const L = SimIntensiveInference.GaussianLikelihood(YS_O, Γ_ϵ)

# Define the model, and the mapping between the outputs and observations (in 
# this case, they are the same
const f(θ) = (θ[1]*XS) ./ (θ[2].+XS)
const g(θ) = θ
const J(θ) = hcat(XS./(θ[2].+XS), -θ[1]*XS./(θ[2].+XS).^2)

"""Generates the posterior joint and marginal densities on a grid of parameter 
values."""
function post_density(
    θ1s::Vector, 
    θ2s::Vector, 
    f::Function, 
    g::Function, 
    π::SimIntensiveInference.AbstractPrior,
    L::SimIntensiveInference.AbstractLikelihood
)::Tuple

    area(x, y) = 0.5sum((x[i+1]-x[i]) * (y[i+1]+y[i]) for i ∈ 1:(length(x)-1))

    joint = [
        SimIntensiveInference.density(π, [θ1, θ2]) *
        SimIntensiveInference.density(L, g(f([θ1, θ2]))) for θ2 ∈ θ2s, θ1 ∈ θ1s
    ]

    marg_θ1 = vec(sum(joint, dims=1))
    marg_θ2 = vec(sum(joint, dims=2))
    marg_θ1 ./= area(θ1s, marg_θ1)
    marg_θ2 ./= area(θ2s, marg_θ2)

    return joint, marg_θ1, marg_θ2

end

# Compute the true posterior on a grid
const N_PTS = 500
const θ1S = collect(range(0.1, 0.25, N_PTS))
const θ2S = collect(range(0.0, 150, N_PTS))
const POST_JOINT, POST_MARG_θ1, POST_MARG_θ2 = post_density(θ1S, θ2S, f, g, π, L)


if abspath(PROGRAM_FILE) == @__FILE__

    Plotting.plot_density_grid(
        θ1S, θ2S, POST_JOINT, POST_MARG_θ1, POST_MARG_θ2, 
        "Posterior Density",
        "$(PLOTS_DIR)/posterior_density.pdf"
    )

end

end