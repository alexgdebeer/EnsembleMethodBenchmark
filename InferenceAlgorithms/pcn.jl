using Distributions
using Printf

# TODO: check this code!!

function run_chain(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    η0::AbstractVector,
    NF::Int,
    Ni::Int, 
    Nb::Int,
    β::Real,
    δ::Real,
    B_wells::AbstractMatrix, 
    n_chain::Int;
    verbose::Bool=true
)

    norm = Normal()
    NG = length(y)

    logpri(η) = -sum(η.^2)
    loglik(G) = -sum((L_e*(G+μ_e-y)).^2)
    logpost(η, G) = logpri(η) + loglik(G)

    ξs = Matrix{Float64}(undef, pr.Nθ, Nb)
    ωs = Matrix{Float64}(undef, pr.Nω, Nb)
    Fs = Matrix{Float64}(undef, NF, Nb)
    Gs = Matrix{Float64}(undef, NG, Nb)
    τs = Vector{Float64}(undef, Nb)

    α_ξ = 0
    α_ω = 0

    ξs[:, 1] = η0[1:pr.Nθ]
    ωs[:, 1] = η0[pr.Nθ+1:end]

    F_f = F(η0)
    Fs[:, 1] = B_wells * F_f
    Gs[:, 1] = G(F_f)
    τs[1] = logpost(η0, Gs[:, 1])

    t0 = time()
    for i ∈ 1:(Ni-1)

        ind_c = (i-1) % Nb + 1
        ind_p = i % Nb + 1

        ζ_ξ = rand(norm, pr.Nθ)
        ξ_p = √(1-β^2) * ξs[:, ind_c] + β*ζ_ξ

        η_p = vcat(ξ_p, ωs[:, ind_c])
        F_f = F(η_p)
        F_p = B_wells * F_f
        G_p = G(F_f)

        h = exp(loglik(G_p) - loglik(Gs[:, ind_c]))

        if h ≥ rand()
            α_ξ += 1
            ξs[:, ind_p] = ξ_p
            Fs[:, ind_p] = F_p
            Gs[:, ind_p] = G_p
        else
            ξs[:, ind_p] = ξs[:, ind_c]
            Fs[:, ind_p] = Fs[:, ind_c]
            Gs[:, ind_p] = Gs[:, ind_c]
        end

        ζ_ω = rand(norm, pr.Nω)
        ω_p = ωs[:, ind_c] + δ*ζ_ω

        η_p = vcat(ξs[:, ind_p], ω_p)
        F_f = F(η_p)
        F_p = B_wells * F_f
        G_p = G(F_f)

        h = exp((loglik(G_p) + logpri(ω_p)) - 
                (loglik(Gs[:, ind_p]) + logpri(ωs[:, ind_c])))

        if h ≥ rand()
            α_ω += 1
            ωs[:, ind_p] = ω_p
            Fs[:, ind_p] = F_p 
            Gs[:, ind_p] = G_p
        else
            ωs[:, ind_p] = ωs[:, ind_c]
        end

        η = vcat(ξs[:, ind_p], ωs[:, ind_p])
        τs[ind_p] = logpost(η, Gs[:, ind_p])

        if (i+1) % Nb == 0

            ηs = vcat(ξs, ωs)
            n_batch = i ÷ Nb

            h5write("data/mcmc/chain_$n_chain.h5", "ηs_$n_batch", ηs)
            h5write("data/mcmc/chain_$n_chain.h5", "Fs_$n_batch", Fs)
            h5write("data/mcmc/chain_$n_chain.h5", "Gs_$n_batch", Gs)
            h5write("data/mcmc/chain_$n_chain.h5", "τs_$n_batch", τs)

            if verbose

                t1 = time()
                time_per_it = (t1 - t0) / Nb
                t0 = t1

                @printf(
                    "%5.0i | %5.0e | %6.2f | %6.2f | %9.2e | %7.3f\n",
                    n_chain, i, α_ξ/i, α_ω/i, τs[ind_p], time_per_it
                )

            end

        end

    end

end

function run_pcn(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    NF::Int,
    Ni::Int,
    Nb::Int,
    Nc::Int,
    β::Real,
    δ::Real,
    B_wells::AbstractMatrix;
    verbose::Bool=true
)

    verbose && println("Chain | Iters | ξ acc. | ω acc. | logpost   | time (s)")

    Threads.@threads for n_chain ∈ 1:Nc

        η0 = vec(rand(pr))
        run_chain(
            F, G, pr, y, μ_e, L_e, η0, 
            NF, Ni, Nb, β, δ,
            B_wells, n_chain,
            verbose=verbose
        )

    end

end