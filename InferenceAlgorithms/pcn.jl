using Distributions
using Printf

# TODO: account for burn-in when saving
# TODO: save this stuff to h5 file?
function run_chain(
    num::Int,
    F::Function,
    G::Function,
    y::AbstractVector,
    η_c::Vector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    NF::Int,
    NG::Int,
    Ni::Int,
    Ns::Int,
    β::Real,
    δ::Real;
    save_increment::Int=100,
    verbose::Bool=false
)

    norm = Normal()

    logpri(η) = -sum(η.^2)
    loglik(g) = -sum((L_e*(g+μ_e-y)).^2)
    logpost(η, g) = logpri(η) + loglik(g)

    ηs = Matrix{Float64}(undef, pr.Nη, Ns)
    Fs = Matrix{Float64}(undef, NF, Ns)
    Gs = Matrix{Float64}(undef, NG, Ns)
    τs = Vector{Float64}(undef, Ni)

    α_ξ = 0
    α_ω = 0

    ξ_c = η_c[1:pr.Nθ]
    ω_c = η_c[pr.Nθ+1:end]
    F_c = F(η_c)
    G_c = G(F_c)
    τs[1] = logpost(η_c, G_c)

    i_save = 0
    t0 = time()

    for i ∈ 2:Ni

        ζ_ξ = rand(norm, pr.Nθ)
        ξ_p = √(1-β^2) * ξ_c + β*ζ_ξ

        η_p = vcat(ξ_p, ω_c)
        F_p = F(η_p)
        G_p = G(F_p)

        h = exp(loglik(G_p) - loglik(G_c))

        if h ≥ rand()
            α_ξ += 1
            ξ_c = copy(ξ_p)
            F_c = copy(F_p)
            G_c = copy(G_p)
        end

        ζ_ω = rand(norm, pr.Nω)
        ω_p = ω_c + δ*ζ_ω

        η_p = vcat(ξ_c, ω_p)
        F_p = F(η_p)
        G_p = G(F_p)

        h = exp((loglik(G_p) + logpri(ω_p)) - 
                (loglik(G_c) + logpri(ω_c)))

        if h ≥ rand()
            α_ω += 1
            ω_c = copy(ω_p)
            F_c = copy(F_p)
            G_c = copy(G_p)
        end

        η_c = vcat(ξ_c, ω_c)
        τs[i] = logpost(η_c, G_c)

        if i % save_increment == 0
            i_save += 1
            ηs[:, i_save] = η_c
            Fs[:, i_save] = F_c 
            Gs[:, i_save] = G_c
        end

        if verbose && (i % 100 == 0)

            t1 = time()
            time_per_it = (t1 - t0) / 100
            t0 = t1

            @printf(
                "%5i | %5.0e | %6.2f | %6.2f | %9.2e | %6.2f\n",
                num, i, α_ξ/i, α_ω/i, τs[i], time_per_it
            )

        end

    end

    return ηs, Fs, Gs, τs

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
    Nc::Int,
    β::Real,
    δ::Real;
    save_increment::Int=100,
    verbose::Bool=true
)

    verbose && println("Chain | Iters | ξ acc. | ω acc. | logpost   | time (s)")

    NG = length(y)
    Ns = Ni ÷ save_increment

    # ηs_f = Array{Float64}(undef, pr.Nη, Ns, Nc)
    # Gs_f = Array{Float64}(undef, NG, Ns, Nc)
    # Fs_f = Array{Float64}(undef, NF, Ns, Nc)
    # τs_f = Matrix{Float64}(undef, Ni, Nc)

    η_cs = rand(pr, Nc)

    Threads.@threads for i ∈ 1:Nc

        run_chain(
            i, F, G, y, η_cs[:, i], μ_e, L_e, NF, NG, Ni, Ns, β, δ,
            save_increment=save_increment, verbose=verbose
        )
        
    end
    # ηs_f[:, :, i], Fs_f[:, :, i], Gs_f[:, :, i], τs_f[:, i] =
    # return ηs_f, Fs_f, Gs_f, τs_f

end