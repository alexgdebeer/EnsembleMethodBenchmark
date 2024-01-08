include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 1000
n_trials = 10

fname = "data/laplace/laplace_$Ne.h5"

results = Dict()

for i ∈ 1:n_trials 
    
    θ0 = vec(rand(pr, 1))

    map, Γ_post, L_post, n_solves = compute_laplace(
        grid_c, model_r, pr, d_obs, θ0
    ) # TODO: fix θ -- it is returned as a sparse vector

    θs = map.θ .+ L_post * rand(Normal(), pr.Nθ, Ne)
    us = hcat([transform(pr, θi) for θi ∈ eachcol(θs)]...)
    Fs = hcat([F(u_i) for u_i ∈ eachcol(us)]...)
    Gs = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    μ_post = reshape(Vector(map.u), grid_c.nx, grid_c.nx)
    σ_post = reshape(std(us, dims=2), grid_c.nx, grid_c.nx)

    results["θs_$i"] = Matrix(θs)
    results["us_$i"] = us
    results["Gs_$i"] = Gs
    
    results["μ_post_$i"] = μ_post
    results["σ_post_$i"] = σ_post
    
    results["n_its_$i"] = 1
    results["n_sims_$i"] = n_solves + Ne

end

save_results(results, fname)