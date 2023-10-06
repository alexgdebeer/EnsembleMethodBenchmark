# TODO: make this into a Metropolis-within-Gibbs sampler?

function run_pcn(
    F::Function,
    G::Function,
    p,
    ys::AbstractVector,
    Γ::AbstractMatrix,
    NF::Int,
    Ni::Int,
    β::Real,
    verbose::Bool=true
)

    NG = length(ys)
    L = cholesky(inv(Γ)).U 

    θs = [rand(θ)]

    for i ∈ 1:Ni 

        sum((L * (x - y))^2)


end