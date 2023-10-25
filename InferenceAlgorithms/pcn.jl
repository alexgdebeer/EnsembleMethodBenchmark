using Distributions
using Printf

# function run_chain(
#     F::Function,
#     G::Function,
#     y::AbstractVector,
#     η_c::Vector,
#     μ_e::AbstractVector,
#     L_e::AbstractMatrix,
#     NF::Int,
#     NG::Int,
#     Ni::Int,
#     Nb::Int,
#     β::Real,
#     δ::Real;
#     verbose::Bool=false
# )

#     norm = Normal()

#     logpri(η) = -sum(η.^2)
#     loglik(g) = -sum((L_e*(g+μ_e-y)).^2)
#     logpost(η, g) = logpri(η) + loglik(g)

#     ξs = Matrix{Float64}(undef, pr.Nθ, Nb)
#     ωs = Matrix{Float64}(undef, pr.Nω, Nb)
#     Fs = Matrix{Float64}(undef, NF, Nb)
#     Gs = Matrix{Float64}(undef, NG, Nb)
#     τs = Vector{Float64}(undef, Ni, Nb)

#     α_ξ = 0
#     α_ω = 0

#     ξs[:, 1] = η_c[1:pr.Nθ]
#     ωs[:, 1] = η_c[pr.Nθ+1:end]
#     Fs[:, 1] = F(η_c)
#     Gs[:, 1] = G(Fs[:, 1])
#     τs[1] = logpost(η_c, G_c)

#     t0 = time()
#     for i ∈ 1:(Ni-1)

#         ind_c = (i-1) % Nb + 1
#         ind_p = i % Nb + 1

#         ζ_ξ = rand(norm, pr.Nθ)
#         ξ_p = √(1-β^2) * ξs[:, ind_c] + β*ζ_ξ

#         η_p = vcat(ξ_p, ω_c)
#         F_p = F(η_p)
#         G_p = G(F_p)

#         h = exp(loglik(G_p) - loglik(Gs[:, ind_c]))

#         if h ≥ rand()
#             α_ξ += 1
#             ξs[:, ind_p] = ξ_p
#             Fs[:, ind_p] = F_p
#             Gs[:, ind_p] = G_p
#         else
#             ξs[:, ind_p] = ξs[:, ind_c]
#             Fs[:, ind_p] = Fs[:, ind_c]
#             Gs[:, ind_p] = Gs[:, ind_c]
#         end

#         ζ_ω = rand(norm, pr.Nω)
#         ω_p = ω_c + δ*ζ_ω

#         η_p = vcat(ξs[:, ind_p], ω_p)
#         F_p = F(η_p)
#         G_p = G(F_p)

#         h = exp((loglik(G_p) + logpri(ω_p)) - 
#                 (loglik(Gs[:, ind_p]) + logpri(ωs[:, ind_p])))

#         if h ≥ rand()
#             α_ω += 1
#             ωs[:, ind_p] = ω_p
#             Fs[:, ind_p] = F_p 
#             Gs[:, ind_p] = G_p
#         else
#             ωs[:, ind_p] = ωs[:, ind_c]
#         end

#         η_c = vcat(ξ_c, ω_c)
#         τs[ind_p] = logpost(η_c, G_c)

#         if i % Nb == 0

#             h5write("data/mcmc/chain_$i.h5", "ηs_b$batch_num", ηs)
#             h5write("data/mcmc/chain_$i.h5", "Gs_b$batch_num", Gs)

#             if verbose

#                 t1 = time()
#                 time_per_it = (t1 - t0) / Nb
#                 t0 = t1
    
#                 @printf(
#                     "%5.0e | %6.2f | %6.2f | %9.2e | %6.2f\n",
#                     i, α_ξ/i, α_ω/i, τs[i], time_per_it
#                 )
    
#             end

#         end

#     end

#     return ηs, Fs, Gs, τs

# end

function save_chain(
    i::Int, 
    ηs::AbstractMatrix, 
    Fs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    τs::AbstractVector
)

    h5write("data/mcmc/chain_$i.h5", "ηs", ηs)
    h5write("data/mcmc/chain_$i.h5", "Fs", Fs)
    h5write("data/mcmc/chain_$i.h5", "Gs", Gs)
    h5write("data/mcmc/chain_$i.h5", "τs", τs)

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
    β::Real,
    δ::Real,
    chain_num::Int;
    verbose::Bool=true
)

    verbose && println("Iters | ξ acc. | ω acc. | logpost   | time (s)")

    NG = length(y)

    η_1 = rand(pr, 100)[:, chain_num]

    norm = Normal()

    logpri(η) = -sum(η.^2)
    loglik(g) = -sum((L_e*(g+μ_e-y)).^2)
    logpost(η, g) = logpri(η) + loglik(g)

    ξs = Matrix{Float64}(undef, pr.Nθ, Nb)
    ωs = Matrix{Float64}(undef, pr.Nω, Nb)
    Fs = Matrix{Float64}(undef, NF, Nb)
    Gs = Matrix{Float64}(undef, NG, Nb)
    τs = Vector{Float64}(undef, Nb)

    α_ξ = 0
    α_ω = 0

    ξs[:, 1] = η_1[1:pr.Nθ]
    ωs[:, 1] = η_1[pr.Nθ+1:end]
    Fs[:, 1] = F(η_1)
    Gs[:, 1] = G(Fs[:, 1])
    τs[1] = logpost(η_1, Gs[:, 1])

    t0 = time()
    for i ∈ 1:(Ni-1)

        ind_c = (i-1) % Nb + 1
        ind_p = i % Nb + 1

        ζ_ξ = rand(norm, pr.Nθ)
        ξ_p = √(1-β^2) * ξs[:, ind_c] + β*ζ_ξ

        η_p = vcat(ξ_p, ωs[:, ind_c])
        F_p = F(η_p)
        G_p = G(F_p)

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
        F_p = F(η_p)
        G_p = G(F_p)

        h = exp((loglik(G_p) + logpri(ω_p)) - 
                (loglik(Gs[:, ind_p]) + logpri(ωs[:, ind_p])))

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

        if i % Nb == 0

            ηs = vcat(ξs, ωs)
            n_batch = i ÷ Nb

            h5write("data/mcmc/chain_$chain_num.h5", "ηs_b$n_batch", ηs)
            h5write("data/mcmc/chain_$chain_num.h5", "Gs_b$n_batch", Gs)
            h5write("data/mcmc/chain_$chain_num.h5", "τs_b$n_batch", τs)

            if verbose

                t1 = time()
                time_per_it = (t1 - t0) / Nb
                t0 = t1
    
                @printf(
                    "%5.0e | %6.2f | %6.2f | %9.2e | %7.3f\n",
                    i, α_ξ/i, α_ω/i, τs[ind_p], time_per_it
                )
    
            end

        end

    end

end