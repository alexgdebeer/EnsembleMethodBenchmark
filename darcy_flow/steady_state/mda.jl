using SimIntensiveInference

include("setup.jl")

Ni = 8
Ne = 100

θs, us, αs = run_es_mda(f, g, p, L, Ni, Ne)#, α_method=:constant)

# logps = θs[1:end,:,:]
# μ_post = reshape(mean(logps[:,:,end], dims=2), grid.nx, grid.ny)

logps_pri = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,:,1]))
logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,:,end]))
μ_post = reshape(mean(logps_post, dims=2), grid_c.nx, grid_c.ny)
σ_post = reshape(std(logps_post, dims=2), grid_c.nx, grid_c.ny)

# plot(
#     heatmap(reshape(logps_pri[:,3], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_pri[:,4], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_pri[:,5], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_pri[:,6], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_pri[:,7], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_post[:,3], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_post[:,4], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_post[:,5], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_post[:,6], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     heatmap(reshape(logps_post[:,7], grid.nx, grid.ny)', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#     axis=([], false), layout=(2, 5), size=(1000, 400)
# )

# anim = @animate for i ∈ 1:Ni+10

#     logps = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,:,min(i, Ni+1)]))
#     μ = reshape(mean(logps, dims=2), grid.nx, grid.ny)
#     σ = reshape( std(logps, dims=2), grid.nx, grid.ny)

#     plot(
#         heatmap(logps_t', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none), 
#         heatmap(μ', cmap=:turbo, clim=(-2.0, 3.0), aspect_ratio=:equal, colorbar=:none),
#         heatmap(σ', aspect_ratio=:equal, colorbar=:none, clims=(0.0, 2.0)), 
#         axis=([], false), layout=(1, 3), size=(600, 200)
#     )

# end

# gif(anim, "anim.gif", fps=4)