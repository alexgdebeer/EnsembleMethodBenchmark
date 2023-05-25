using Distributions
using PyPlot
using SimIntensiveInference

include("lv_model.jl")

# Define perturbation kernel 
Γ_K = 0.05^2 * Matrix(I, size(π.Σ))
K = MvNormal(Γ_K)

# Define length of chain and number of chains to run
N = 1_000_000
n_chains = 10

θs, ys = SimIntensiveInference.run_mcmc(
    f, g, π, L, K, N, n_chains=10
)

for i ∈ 1:n_chains
    PyPlot.plot(θs[1,:,i], θs[2,:,i], linewidth=0.2)
end

PyPlot.savefig("test.pdf")

