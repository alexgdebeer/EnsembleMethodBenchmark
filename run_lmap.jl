include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 1000
n_trials = 10

results = Dict(100 => Dict(), 1000 => Dict())

for i ∈ 1:n_trials 
        
    θ0 = vec(rand(pr, 1))

    map, Γ_post, L_post, n_solves = compute_lmap(
        grid_c, model_r, pr, d_obs, θ0
    )

    for r ∈ keys(results)

        θs = map.θ .+ L_post * rand(Normal(), pr.Nθ, r)
        us = hcat([transform(pr, θi) for θi ∈ eachcol(θs)]...)
        Fs = model_r.B_wells * hcat([F(u_i) for u_i ∈ eachcol(us)]...)
        ls = [gauss_to_unif(ω_l, l_bounds...) for ω_l ∈ θs[end, :]]

        μ_post = reshape(Vector(map.u), grid_c.nx, grid_c.nx)
        σ_post = reshape(std(us, dims=2), grid_c.nx, grid_c.nx)

        results[r]["θs_$i"] = Matrix(θs)
        results[r]["us_$i"] = us
        results[r]["Fs_$i"] = Fs
        results[r]["ls_$i"] = ls
    
        results[r]["μ_post_$i"] = μ_post
        results[r]["σ_post_$i"] = σ_post
    
        results[r]["n_its_$i"] = 1
        results[r]["n_sims_$i"] = n_solves

    end

end

for r ∈ keys(results)
    fname = "data/lmap/lmap_$(r).h5"
    save_results(results[r], fname)
end