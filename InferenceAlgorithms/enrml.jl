function tsvd(A::AbstractMatrix; e::Real=0.99)

    U, Λ, V = svd(A)
    minimum(Λ) < 0 && error(minimum(Λ))
    λ_cum = cumsum(Λ)

    for i ∈ 1:length(Λ)
        if λ_cum[i] / λ_cum[end] ≥ e 
            return U[:, 1:i], Λ[1:i], V[:, 1:i]
        end
    end

end

function compute_S(Gs, ys, μ_e, C_e_invsqrt)
    φs = sum((C_e_invsqrt * (Gs .+ μ_e .- ys)).^2, dims=1)
    return mean(φs)
end

function compute_gain_enrml(
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    Ψ = Diagonal(ΛG ./ (λ .+ 1.0 .+ ΛG.^2))
    return Δη * VG * Ψ * UG' * C_e_invsqrt

end

"""Computes the EnRML gain matrix without applying any localisation."""
function compute_gain_enrml(
    localiser::IdentityLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δθ::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    return compute_gain_enrml(Δθ, UG, ΛG, VG, C_e_invsqrt, λ)

end

"""Computes the EnRML gain using the localisation method outlined by
Flowerdew (2015)."""
function compute_gain_enrml(
    localiser::FisherLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δθ::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    K = compute_gain_enrml(Δθ, UG, ΛG, VG, C_e_invsqrt, λ)
    return localise(localiser, θs, Gs, K)

end

"""Computes the EnRML gain using the localisation method outlined by 
Zhang and Oliver (2010)."""
function compute_gain_enrml(
    localiser::BootstrapLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δθ::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    Nθ, Ne = size(θs)
    NG, Ne = size(Gs)
    K = compute_gain_enrml(Δθ, UG, ΛG, VG, C_e_invsqrt, λ)
    Ks_boot = zeros(Nθ, NG, localiser.n_boot)

    for k ∈ 1:localiser.n_boot

        inds_res = rand(1:Ne, Ne)

        Δθ_res = compute_Δs(θs[:, inds_res])
        ΔG_res = compute_Δs(Gs[:, inds_res])
        UG_res, ΛG_res, VG_res = tsvd(ΔG_res)

        Ks_boot[:, :, k] = compute_gain_enrml(
            Δθ_res, UG_res, ΛG_res, VG_res, 
            C_e_invsqrt, λ
        )

    end

    var_Kis = mean((Ks_boot .- K).^2, dims=3)[:, :, 1]
    R² = var_Kis ./ (K.^2 .+ localiser.tol)
    P = 1 ./ (1 .+ R² * (1 + 1 / localiser.σ^2))

    return P .* K 

end

"""Computes the EnRML gain using a variant of the localisation method 
outlined by Luo and Bhakta (2020)."""
function compute_gain_enrml(
    localiser::ShuffleLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δθ::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    K = compute_gain_enrml(Δθ, UG, ΛG, VG, C_e_invsqrt, λ)
    return localise(localiser, θs, Gs, K)

end

"""Computes an EnRML update, following Chen and Oliver (2013)."""
function enrml_update(
    θs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    ys::AbstractMatrix, 
    μ_e::AbstractVector, 
    C_e_invsqrt::AbstractMatrix,
    θs_pr::AbstractMatrix,
    Uθ_pr::AbstractMatrix,
    Λθ_pr::AbstractVector,
    λ::Real,
    localiser::Localiser 
)

    Δθ = compute_Δs(θs)
    ΔG = compute_Δs(Gs)

    UG, ΛG, VG = tsvd(C_e_invsqrt * ΔG)
    K = compute_gain_enrml(localiser, θs, Gs, Δθ, UG, ΛG, VG, C_e_invsqrt, λ)

    # Calculate corrections based on prior deviations
    δθ_pr = Δθ * VG * inv((λ + 1)I + Diagonal(ΛG).^2) * VG' * Δθ' *
            Uθ_pr * Diagonal(1 ./ Λθ_pr.^2) * Uθ_pr' * (θs - θs_pr)

    # Calculate corrections based on fit to observations
    δθ_obs = K * (Gs .+ μ_e .- ys)

    return θs - δθ_pr - δθ_obs

end

"""Computes an EnRML update without applying any inflation."""
function enrml_update(
    θs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    ys::AbstractMatrix, 
    μ_e::AbstractVector, 
    C_e_invsqrt::AbstractMatrix,
    θs_pr::AbstractMatrix,
    Uθ_pr::AbstractMatrix,
    Λθ_pr::AbstractVector,
    λ::Real,
    localiser::Localiser,
    inflator::IdentityInflator
)

    return enrml_update(
        θs, Gs, ys, μ_e, C_e_invsqrt, 
        θs_pr, Uθ_pr, Λθ_pr, λ, localiser
    )

end

"""Updates the current ensemble using the adaptive inflation method 
outlined by Evensen (2009)."""
function enrml_update(
    θs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    ys::AbstractMatrix, 
    μ_e::AbstractVector, 
    C_e_invsqrt::AbstractMatrix,
    θs_pr::AbstractMatrix,
    Uθ_pr::AbstractMatrix,
    Λθ_pr::AbstractVector,
    λ::Real,
    localiser::Localiser,
    inflator::AdaptiveInflator
)

    Nθ, Ne = size(θs)
    dummy_params = generate_dummy_params(inflator, Ne)

    θs_pr_aug = [θs_pr; dummy_params]
    Uθ_pr_aug, Λθ_pr_aug, = tsvd(θs_pr_aug)

    θs_aug = enrml_update(
        [θs; dummy_params], Gs, ys, μ_e, C_e_invsqrt, 
        θs_pr_aug, Uθ_pr_aug, Λθ_pr_aug, λ, localiser
    )

    θs_new = θs_aug[1:Nθ, :]
    dummy_params = θs_aug[Nθ+1:end, :]
    ρ = 1 / mean(std(dummy_params, dims=2))
    @info "Inflation factor: $(ρ)."

    μ_θ = mean(θs_new, dims=2)
    return ρ * (θs_new .- μ_θ) .+ μ_θ

end

function run_enrml(
    F::Function, 
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    γ::Real=10,
    λ_min::Real=0.01,
    max_cuts::Int=5,
    max_its::Int=30,
    ΔS_min::Real=0.01,
    Δθ_min::Real=0.5,
    localiser::Localiser=IdentityLocaliser(),
    inflator::Inflator=IdentityInflator()
)

    println("It. | stat | ΔS       | Δη_max   | λ")

    C_e_invsqrt = √(inv(C_e))
    NG = length(y)

    θs = []
    us = []
    Fs = []
    Gs = []
    Ss = []
    λs = []

    ys = rand(MvNormal(y, C_e), Ne)

    θs_pr = rand(pr, Ne)
    us_pr, Fs_pr, Gs_pr = run_ensemble(θs_pr, F, G, pr)
    S_pr = compute_S(Gs_pr, ys, μ_e, C_e_invsqrt)
    λ = 10^floor(log10(S_pr / 2NG))

    push!(θs, θs_pr)
    push!(us, us_pr)
    push!(Fs, Fs_pr)
    push!(Gs, Gs_pr)
    push!(Ss, S_pr)
    push!(λs, λ)

    Δθ_pr = compute_Δs(θs_pr)
    Uθ_pr, Λθ_pr, = tsvd(Δθ_pr)

    i = 1
    en_ind = 1
    n_cuts = 0
    while i ≤ max_its

        θs_i = enrml_update(
            θs[en_ind], Gs[en_ind], ys, μ_e, C_e_invsqrt, 
            θs_pr, Uθ_pr, Λθ_pr, λ, localiser, inflator
        )

        us_i, Fs_i, Gs_i = run_ensemble(θs_i, F, G, pr)
        S_i = compute_S(Gs_i, ys, μ_e, C_e_invsqrt)

        push!(θs, θs_i)
        push!(us, us_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)
        push!(Ss, S_i)
        push!(λs, λ)
        i += 1

        if S_i ≤ Ss[en_ind]

            ΔS = 1 - S_i / Ss[en_ind]
            Δθ_max = maximum(abs.(θs_i - θs[en_ind]))

            en_ind = i
            n_cuts = 0
            λ = max(λ / γ, λ_min)

            @printf(
                "%2i  | acc. | %.2e | %.2e | %.2e \n", 
                i-1, ΔS, Δθ_max, λ
            )

            if (ΔS ≤ ΔS_min) && (Δθ_max ≤ Δθ_min)
                @info "Convergence criteria met."
                return θs, us, Fs, Gs, Ss, λs, en_ind
            end

        else 

            @printf(
                "%2i  | rej. | -------- | -------- | %.2e \n", 
                i-1, λ
            )

            n_cuts += 1
            λ *= γ

            if n_cuts == max_cuts
                @info "Terminating: $n_cuts consecutive cuts made."
                return θs, us, Fs, Gs, Ss, λs, en_ind
            end

        end

    end

    @info "Terminating: maximum number of iterations exceeded."
    return θs, us, Fs, Gs, Ss, λs, en_ind

end