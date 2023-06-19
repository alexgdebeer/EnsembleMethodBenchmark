using SimIntensiveInference

include("steadystate_setup.jl")

αs = [16.0 for _ ∈ 1:16]
n = 1000

θs, us = run_es_mda(f, g, p, L, αs, n)

logps = θs[:,:,:]
μ_post = reshape(mean(logps[:,:,end], dims=2), grid.nx, grid.ny)