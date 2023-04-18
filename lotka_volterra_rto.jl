"""Runs RML and RTO on the MONOD model."""

using Distributions
using ForwardDiff
using LinearAlgebra
using Statistics

include("lotka_volterra_model.jl")
include("plotting.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")


# Define the prior and likelihood
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)
const L = SimIntensiveInference.GaussianLikelihood(LVModel.μ_L, LVModel.Γ_ϵ)

# Define the number of samples to generate
const N = 10000

const RML = false 
const RTO = true


"""Returns the RTO density on a grid of parameter values."""
function rto_density(as, bs, f, g, π, L, Q)

    area(x, y) = 0.5sum((x[i+1]-x[i]) * (y[i+1]+y[i]) for i ∈ 1:(length(x)-1))

    # Define augmented system
    L_θ = cholesky(inv(π.Σ)).U  
    L_ϵ = cholesky(inv(L.Σ)).U
    f̃(θ) = [L_ϵ*g(f(θ)); L_θ*θ]
    ỹ = [L_ϵ*L.μ; L_θ*π.μ]

    # Compute the joint density on a grid of a and b values
    joint_density = reduce(hcat,
        [abs(det(Q'*ForwardDiff.jacobian(f̃, [a,b]))) * 
            exp(-0.5sum((Q'*(f̃([a,b])-ỹ)).^2)) for b ∈ bs] for a ∈ as)

    # Compute the marginal densities
    marg_a = vec(sum(joint_density, dims=1))
    marg_b = vec(sum(joint_density, dims=2))
    marg_a ./= area(as, marg_a)
    marg_b ./= area(bs, marg_b)

    return joint_density, marg_a, marg_b

end


if RML

    θ_MAP, θs = @time SimIntensiveInference.run_rml(
        LVModel.f, LVModel.g, π, L, N
    )

    Plotting.plot_approx_posterior(
        θs,
        LVModel.AS, LVModel.BS, LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
        "RML Posterior", 
        "$(LVModel.PLOTS_DIR)/rml_rto/rml_posterior.pdf"; 
        caption="$N samples computed using RML."
    )

end


if RTO

    θ_MAP, Q, θs, ws = @time SimIntensiveInference.run_rto(
        LVModel.f, LVModel.g, π, L, N
    )

    Plotting.plot_approx_posterior(
        θs,
        LVModel.AS, LVModel.BS, LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
        "RTO Posterior", 
        "$(LVModel.PLOTS_DIR)/rml_rto/rto_posterior.pdf"; 
        caption="$N samples from RTO density."
    )

    θs_reweighted = [SimIntensiveInference.sample_from_population(θs, ws) for _ ∈ 1:N]

    Plotting.plot_approx_posterior(
        θs_reweighted,
        LVModel.AS, LVModel.BS, LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
        "Reweighted RTO Posterior", 
        "$(LVModel.PLOTS_DIR)/rml_rto/rto_posterior_reweighted.pdf", 
        caption="$N re-weighted RTO samples."
    )

    const RTO_JOINT, RTO_MARG_A, RTO_MARG_B = rto_density(
        LVModel.AS, LVModel.BS,
        LVModel.f, LVModel.g, 
        π, L, Q
    )

    Plotting.plot_density_grid(
        LVModel.AS, LVModel.BS, 
        RTO_JOINT, RTO_MARG_A, RTO_MARG_B, 
        "RTO Density",
        "$(LVModel.PLOTS_DIR)/rml_rto/rto_density.pdf"
    )

end


