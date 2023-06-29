using SimIntensiveInference
include("setup/setup.jl")

xmin, Δx, xmax = 0.0, 0.02, 1.0
ymin, Δy, ymax = 0.0, 0.02, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

tmax = 2.0
Δt = 0.01

grid = TimeVaryingGrid(xs, ys, tmax, Δt)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> 0.0)
)

# Define forcing function 
function f(x, y, t)

    if 0.1 ≤ x ≤ 0.2 && 0.1 ≤ y ≤ 0.2 && 0.0 ≤ t ≤ 0.2
        return -2.0
    elseif 0.1 ≤ x ≤ 0.2 && 0.8 ≤ y ≤ 0.9 && 0.2 ≤ t ≤ 0.4
        return -2.0
    elseif 0.8 ≤ x ≤ 0.9 && 0.1 ≤ y ≤ 0.2 && 0.4 ≤ t ≤ 0.6
        return -2.0
    elseif 0.8 ≤ x ≤ 0.9 && 0.8 ≤ y ≤ 0.9 && 0.6 ≤ t
        return -2.0
    elseif 0.4 ≤ x ≤ 0.6 && 0.4 ≤ y ≤ 0.6 
        return 0.5
    end

    return 0

end

σ, γx, γy = 1.0, 0.1, 0.1
k = ARDExpSquaredKernel(σ, γx, γy)

μ = 0.0
p = GaussianPrior(μ, k, grid.xs, grid.ys)

logps = reshape(rand(p), grid.nx, grid.ny)
ps = exp.(logps)

us = @time solve(grid, ps, bcs, f)

println("Done")