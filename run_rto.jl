include("DarcyFlow/DarcyFlow.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")
include("setup.jl")

n_samples = 100

samples, lnws = run_rto(grid_c, model_r, pr, d_obs, n_samples, verbose=false)
# samples_mcmc = run_rto_mcmc(samples, lnws)

