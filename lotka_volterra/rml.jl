using SimIntensiveInference

include("problem_setup.jl")

rml = false
rto = true

n = 10_000

if rml

    θ_map, θs = run_rml(f, g, π, L, n, x0=θs_t)

    plot_approx_posterior(
        eachcol(θs), 
        as, bs, post_marg_a, post_marg_b, 
        "RML Posterior",
        "$(plots_dir)/rml_rto/rml_posterior.pdf",
        θs_t=θs_t,
        caption="$n samples."
    )

end

if rto

    θ_map, Q, θs, ws = run_rto(f, g, π, L, n, x0=θs_t)

    plot_approx_posterior(
        eachcol(θs), 
        as, bs, post_marg_a, post_marg_b, 
        "Uncorrected RTO Posterior", 
        "$plots_dir/rml_rto/rto_posterior_uncorrected.pdf"; 
        θs_t=θs_t,
        caption="$n unweighted samples."
    )

    is = resample_ws(ws)
    θs_r = θs[:,is]

    plot_approx_posterior(
        eachcol(θs_r),
        as, bs, post_marg_a, post_marg_b, 
        "Corrected RTO Posterior", 
        "$plots_dir/rml_rto/rto_posterior_corrected.pdf";
        θs_t=θs_t,
        caption="$n reweighted samples."
    )

    # const RTO_JOINT, RTO_MARG_A, RTO_MARG_B = rto_density(
    #     LVModel.AS, LVModel.BS,
    #     LVModel.f, LVModel.g, 
    #     π, L, Q
    # )

    # Plotting.plot_density_grid(
    #     LVModel.AS, LVModel.BS, 
    #     RTO_JOINT, RTO_MARG_A, RTO_MARG_B, 
    #     "RTO Density",
    #     "$(LVModel.PLOTS_DIR)/rml_rto/rto_density.pdf";
    #     θs_t=LVModel.θS_T,
    # )

end

