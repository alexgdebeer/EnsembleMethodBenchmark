"""Runs the ES on the Lotka-Volterra model."""

using SimIntensiveInference

include("problem_setup.jl")

N_e = 100

mda = true

if mda 

    αs = [16.0 for _ ∈ 1:16]

    θs, ys = SimIntensiveInference.run_es_mda(f, g, π, L, αs, N_e)

    plot_approx_posterior(
        eachcol(θs[:,:,end]), 
        as, bs, post_marg_a, post_marg_b,
        "LV: ES-MDA Posterior",
        "$(plots_dir)/es/es_mda_posterior.pdf";
        θs_t=θs_t,
        caption="Ensemble size: $N_e. Iterations: $(length(αs))."
    )

else

    θs, ys = SimIntensiveInference.run_es(f, g, π, L, N_e)

    plot_approx_posterior(
        eachcol(θs[:,:,end]), 
        as, bs, post_marg_a, post_marg_b,
        "LV: ES Posterior",
        "$(plots_dir)/es/es_posterior.pdf";
        θs_t=θs_t,
        caption="Ensemble size: $N_e."
    )

end