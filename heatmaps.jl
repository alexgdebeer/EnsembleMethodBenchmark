using Plots

θ_t_reshaped = reshape(θ_t, grid_f.nx, grid_f.nx)
θ_map_reshaped = reshape(transform(pr, η_map), grid_c.nx, grid_c.nx)

heatmaps = [
    heatmap(grid_f.xs, grid_f.xs, θ_t_reshaped, cmap=:turbo, aspect_ratio=:equal, axis=([], false), title="Truth"),
    heatmap(grid_c.xs, grid_c.xs, θ_map_reshaped, cmap=:turbo, aspect_ratio=:equal, axis=([], false), title="MAP", clims=extrema(θ_t))
]

plot(heatmaps..., layout=(1, 2), colorbar=false, size=(450, 250))
#savefig(p, "plots/map_3.pdf")