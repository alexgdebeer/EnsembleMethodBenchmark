
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
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    return compute_gain_enrml(Δη, UG, ΛG, VG, C_e_invsqrt, λ)

end

"""Computes the EnRML gain using the localisation method outlined by
Flowerdew (2015)."""
function compute_gain_enrml(
    localiser::FisherLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    K = compute_gain_enrml(Δη, UG, ΛG, VG, C_e_invsqrt, λ)

    R_ηG = compute_cors(ηs, Gs)[1]
    P = zeros(size(R_ηG))

    for i ∈ 1:Nη
        for j ∈ 1:NG 
            ρ_ij = R_ηG[i, j]
            s = log((1+ρ_ij) / (1-ρ_ij)) / 2
            σ_s = (tanh(s + √(Ne-3)^-1) - tanh(s - √(Ne-3)^-1)) / 2
            P[i, j] = ρ_ij^2 / (ρ_ij^2 + σ_s^2)
        end
    end

    return P .* K

end

"""Computes the EnRML gain using the localisation method outlined by 
Zhang and Oliver (2010)."""
function compute_gain_enrml(
    localiser::BootstrapLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)
    K = compute_gain_enrml(Δη, UG, ΛG, VG, C_e_invsqrt, λ)
    Ks_boot = zeros(Nη, NG, localiser.n_boot)

    for k ∈ 1:localiser.n_boot

        inds_res = rand(1:Ne, Ne)

        Δη_res = compute_Δs(ηs[:, inds_res])
        ΔG_res = compute_Δs(Gs[:, inds_res])
        UG_res, ΛG_res, VG_res = tsvd(ΔG_res)

        Ks_boot[:, :, k] = compute_gain_enrml(
            Δη_res, UG_res, ΛG_res, VG_res, 
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
    localiser::BootstrapLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractVector,
    VG::AbstractMatrix,
    C_e_invsqrt::AbstractMatrix,
    λ::Real
)
    @warn "Check me please!!"

    K = compute_gain_enrml(Δη, UG, ΛG, VG, C_e_invsqrt, λ)

    if localiser.P !== nothing 
        return localiser.P .* K
    end

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    R_ηG = compute_cors(ηs, Gs)[1]

    P = zeros(Nη, NG)
    R_ηGs = zeros(Nη, NG, localiser.n_shuffle)

    for i ∈ 1:localiser.n_shuffle
        inds = get_shuffled_inds(Ne)
        ηs_shuffled = ηs[:, inds]
        R_ηGs[:, :, i] = compute_cors(ηs_shuffled, Gs)[1]
    end

    σs_e = median(abs.(R_ηGs), dims=3) ./ 0.6745

    for i ∈ 1:Nη
        for j ∈ 1:NG
            z = (1 - abs(R_ηG[i, j])) / (1 - σs_e[i, j])
            P[i, j] = gaspari_cohn(z)
        end
    end

    localiser.P = P
    return P .* K

end

function compute_S(Gs, ys, μ_e, C_e_invsqrt)
    φs = sum((C_e_invsqrt * (Gs .+ μ_e .- ys)).^2, dims=1)
    return mean(φs)
end

function enrml_update(
    ηs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    ys::AbstractMatrix, 
    μ_e::AbstractVector, 
    C_e_invsqrt::AbstractMatrix,
    ηs_pr::AbstractMatrix,
    Uη_pr::AbstractMatrix,
    Λη_pr::AbstractVector,
    λ::Real,
    localiser::Localiser 
)

    Δη = compute_Δs(ηs)
    ΔG = compute_Δs(Gs)

    UG, ΛG, VG = tsvd(C_e_invsqrt * ΔG)
    K = compute_gain_enrml(localiser, ηs, Gs, Δη, UG, ΛG, VG, C_e_invsqrt, λ)

    # Calculate corrections based on prior deviations
    δη_pr = Δη * VG * inv((λ + 1)I + Diagonal(ΛG).^2) * VG' * Δη' *
            Uη_pr * Diagonal(1 ./ Λη_pr.^2) * Uη_pr' * (ηs - ηs_pr)

    # Calculate corrections based on fit to observations
    δη_obs = K * (Gs .+ μ_e .- ys)

    return ηs - δη_pr - δη_obs

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
    Δη_min::Real=0.5,
    localiser::Localiser=IdentityLocaliser()
)

    println("It. | stat | ΔS       | Δη_max   | λ")

    C_e_invsqrt = √(inv(C_e))
    NG = length(y)

    ηs = []
    θs = []
    Fs = []
    Gs = []
    Ss = []
    λs = []

    ys = rand(MvNormal(y, C_e), Ne)

    ηs_pr = rand(pr, Ne)
    θs_pr, Fs_pr, Gs_pr = run_ensemble(ηs_pr, F, G, pr)
    S_pr = compute_S(Gs_pr, ys, μ_e, C_e_invsqrt)
    λ = 10^floor(log10(S_pr / 2NG))

    push!(ηs, ηs_pr)
    push!(θs, θs_pr)
    push!(Fs, Fs_pr)
    push!(Gs, Gs_pr)
    push!(Ss, S_pr)
    push!(λs, λ)

    Δη_pr = compute_Δs(ηs_pr)
    Uη_pr, Λη_pr, = tsvd(Δη_pr)

    i = 1
    en_ind = 1
    n_cuts = 0
    while i ≤ max_its

        ηs_i = enrml_update(
            ηs[en_ind], Gs[en_ind], ys, μ_e, C_e_invsqrt, 
            ηs_pr, Uη_pr, Λη_pr, λ, localiser
        )

        θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)
        S_i = compute_S(Gs_i, ys, μ_e, C_e_invsqrt)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)
        push!(Ss, S_i)
        push!(λs, λ)
        i += 1

        if S_i ≤ Ss[en_ind]

            ΔS = 1 - S_i / Ss[en_ind]
            Δη_max = maximum(abs.(ηs_i - ηs[en_ind]))

            en_ind = i
            n_cuts = 0
            λ = max(λ / γ, λ_min)

            @printf(
                "%2i  | acc. | %.2e | %.2e | %.2e \n", 
                i-1, ΔS, Δη_max, λ
            )

            if (ΔS ≤ ΔS_min) && (Δη_max ≤ Δη_min)
                @info "Convergence criteria met."
                return ηs, θs, Fs, Gs, Ss, λs, en_ind
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
                return ηs, θs, Fs, Gs, Ss, λs, en_ind
            end

        end

    end

    @info "Terminating: maximum number of iterations exceeded."
    return ηs, θs, Fs, Gs, Ss, λs, en_ind

end