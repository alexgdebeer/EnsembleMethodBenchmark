function run_chain(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    θ0::AbstractVector,
    NF::Int,
    NG::Int,
    Ni::Int, 
    Nb::Int,
    β::Real,
    δ::Real,
    B_wells::AbstractMatrix, 
    n_chain::Int;
    thin::Int=10,
    verbose::Bool=true
)

    u0 = transform(pr, θ0)
    norm = Normal()

    logpri(θ) = -sum(θ.^2)
    loglik(G) = -sum((L_e*(G+μ_e-d_obs)).^2)
    logpost(η, G) = logpri(η) + loglik(G)

    ξs = Matrix{Float64}(undef, pr.Nθ, Nb)
    ωs = Matrix{Float64}(undef, pr.Nω, Nb)
    us = Matrix{Float64}(undef, pr.Nu, Nb)
    Fs = Matrix{Float64}(undef, NF, Nb)
    Gs = Matrix{Float64}(undef, NG, Nb)
    τs = Vector{Float64}(undef, Nb)

    α_ξ = 0
    α_ω = 0

    ξs[:, 1] = θ0[1:pr.Nu]
    ωs[:, 1] = θ0[pr.Nu+1:end]
    us[:, 1] = u0

    F_f = F(u0)
    Fs[:, 1] = B_wells * F_f
    Gs[:, 1] = G(F_f)
    τs[1] = logpost(θ0, Gs[:, 1])

    t0 = time()
    n_chunk = 1
    for i ∈ 1:(Ni-1)

        ind_c = (i-1) % Nb + 1
        ind_p = i % Nb + 1

        ζ_ξ = rand(norm, pr.Nu)
        ξ_p = √(1-β^2) * ξs[:, ind_c] + β*ζ_ξ

        θ_p = vcat(ξ_p, ωs[:, ind_c])
        u_p = transform(pr, θ_p)

        F_f = F(u_p)
        F_p = B_wells * F_f
        G_p = G(F_f)

        h = exp(loglik(G_p) - loglik(Gs[:, ind_c]))

        if h ≥ rand()
            α_ξ += 1
            ξs[:, ind_p] = ξ_p
            us[:, ind_p] = u_p
            Fs[:, ind_p] = F_p
            Gs[:, ind_p] = G_p
        else
            ξs[:, ind_p] = ξs[:, ind_c]
            us[:, ind_p] = us[:, ind_c]
            Fs[:, ind_p] = Fs[:, ind_c]
            Gs[:, ind_p] = Gs[:, ind_c]
        end

        ζ_ω = rand(norm, pr.Nω)
        ω_p = ωs[:, ind_c] + δ*ζ_ω

        θ_p = vcat(ξs[:, ind_p], ω_p)
        u_p = transform(pr, θ_p)

        F_f = F(u_p)
        F_p = B_wells * F_f
        G_p = G(F_f)

        h = exp((loglik(G_p) + logpri(ω_p)) - 
                (loglik(Gs[:, ind_p]) + logpri(ωs[:, ind_c])))

        if h ≥ rand()
            α_ω += 1
            ωs[:, ind_p] = ω_p
            us[:, ind_p] = u_p
            Fs[:, ind_p] = F_p 
            Gs[:, ind_p] = G_p
        else
            ωs[:, ind_p] = ωs[:, ind_c]
        end

        θ = vcat(ξs[:, ind_p], ωs[:, ind_p])
        τs[ind_p] = logpost(θ, Gs[:, ind_p])

        if (i+1) % Nb == 0

            θs = vcat(ξs, ωs)

            h5write("data/pcn/chain_$n_chain.h5", "θs_$n_chunk", θs[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "us_$n_chunk", us[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "Fs_$n_chunk", Fs[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "Gs_$n_chunk", Gs[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "τs_$n_chunk", τs[:, 1:thin:end])

            n_chunk += 1

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

    return nothing

end

function run_pcn(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
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

    NG = length(d_obs)

    Threads.@threads for chain_num ∈ 1:Nc

        θ0 = vec(rand(pr))
        run_chain(
            F, G, pr, d_obs, μ_e, L_e, θ0, 
            NF, NG, Ni, Nb, β, δ,
            B_wells, chain_num,
            verbose=verbose
        )

    end

end