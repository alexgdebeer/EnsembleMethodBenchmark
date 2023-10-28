using HDF5

N_BATCHES = 5000

f = h5open("data/mcmc/chain_4.h5", "r")

ηs = reduce(hcat, [f["ηs_$i"][:, end] for i ∈ 10:(N_BATCHES-1)])

close(f)
println("Here")
θs = reduce(hcat, [transform(pr, η) for η ∈ eachcol(ηs)])
μ_post = mean(θs, dims=2)
