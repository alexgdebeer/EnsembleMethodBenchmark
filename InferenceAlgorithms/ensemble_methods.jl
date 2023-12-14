TOL = 1e-8 # Convergence tolerance

abstract type Localiser end
abstract type Inflator end

struct IdentityLocaliser <: Localiser end
struct IdentityInflator <: Inflator end

struct FisherLocaliser <: Localiser end
struct LedoitWolfLocaliser <: Localiser end

mutable struct ShuffleLocaliser <: Localiser

    n_shuffle::Int 
    P::Union{AbstractMatrix, Nothing}
    
    function ShuffleLocaliser(n_shuffle=50)
        return new(n_shuffle, nothing)
    end

end

struct BootstrapLocaliser <: Localiser

    n_boot::Int 
    σ::Real 
    tol::Real

    function BootstrapLocaliser(n_boot=50, σ=0.6, tol=1e-8) 
        return new(n_boot, σ, tol)
    end

end

struct PowerLocaliser <: Localiser
    α::Real 
    PowerLocaliser(α=0.4) = new(α)
end

struct AdaptiveInflator <: Inflator
    n_dummy_params::Int 
    AdaptiveInflator(n_dummy_params=50) = new(n_dummy_params)
end

function compute_Δs(xs::AbstractMatrix)

    N = size(xs, 2)
    Δs = (xs .- mean(xs, dims=2)) ./ √(N-1)
    return Δs

end

function compute_covs(
    ηs::AbstractMatrix, 
    Gs::AbstractMatrix
)

    Δη = compute_Δs(ηs)
    ΔG = compute_Δs(Gs)

    C_ηG = Δη * ΔG'
    C_GG = ΔG * ΔG'

    return C_ηG, C_GG

end

function compute_cors(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix
)

    V_η = Diagonal(std(ηs, dims=2)[:])
    V_G = Diagonal(std(Gs, dims=2)[:])

    C_ηG, C_GG = compute_covs(ηs, Gs)
    R_ηG = inv(V_η) * C_ηG * inv(V_G)
    R_GG = inv(V_G) * C_GG * inv(V_G)

    return R_ηG, R_GG

end

function compute_gain_eki(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    C_ηG, C_GG = compute_covs(ηs, Gs)
    return C_ηG * inv(C_GG + α * C_e)

end

"""Computes the EKI gain without applying any localisation."""
function compute_gain_eki(
    localiser::IdentityLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    return compute_gain_eki(ηs, Gs, α, C_e)

end

"""Computes the EKI gain using the localisation scheme outlined by 
Zhang and Oliver (2010)."""
function compute_gain_eki(
    localiser::BootstrapLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)
    K = compute_gain_eki(ηs, Gs, α, C_e)
    Ks_boot = zeros(Nη, NG, localiser.n_boot)

    for k ∈ 1:localiser.n_boot 
        inds_res = rand(1:Ne, Ne)
        ηs_res = ηs[:, inds_res]
        Gs_res = Gs[:, inds_res]
        K_res = compute_gain_eki(ηs_res, Gs_res, α, C_e)
        Ks_boot[:, :, k] = K_res
    end

    var_Kis = mean((Ks_boot .- K).^2, dims=3)[:, :, 1]
    R² = var_Kis ./ (K.^2 .+ localiser.tol)
    P = 1 ./ (1 .+ R² * (1 + 1 / localiser.σ^2))

    return P .* K 

end

"""Computes the EKI gain using a variant of the localisation scheme 
outlined by Luo and Bhakta (2022)."""
function compute_gain_eki(
    localiser::ShuffleLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    function get_shuffled_inds(N::Int)

        inds = shuffle(1:N)
        for i ∈ 1:N 
            if inds[i] == i 
                inds[(i%N)+1], inds[i] = inds[i], inds[(i%N)+1]
            end
        end

        return inds

    end

    function gaspari_cohn(z::Real)
        if 0 ≤ z ≤ 1 
            return -(1/4)z^5 + (1/2)z^4 + (5/8)z^3 - (5/3)z^2 + 1
        elseif 1 < z ≤ 2 
            return (1/12)z^5 - (1/2)z^4 + (5/8)z^3 + (5/3)z^2 - 5z + 4 - (2/3)z^-1
        else
            return 0.0
        end
    end

    K = compute_gain_eki(ηs, Gs, α, C_e)

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

"""Computes the EKI gain using the localisation scheme outlined by 
Flowerdew (2015)."""
function compute_gain_eki(
    localiser::FisherLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    K = compute_gain_eki(ηs, Gs, α, C_e)

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

"""Computes the EKI gain using the localisation scheme outlined by 
Lee (2021)."""
function compute_gain_eki(
    localiser::PowerLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real, 
    C_e::AbstractMatrix
)

    V_η = Diagonal(std(ηs, dims=2)[:])
    V_G = Diagonal(std(Gs, dims=2)[:])

    R_ηG, R_GG = compute_cors(ηs, Gs)

    R_ηG_sec = R_ηG .* abs.(R_ηG) .^ localiser.α
    R_GG_sec = R_GG .* abs.(R_GG) .^ localiser.α

    C_ηG_sec = V_η * R_ηG_sec * V_G
    C_GG_sec = V_G * R_GG_sec * V_G

    return C_ηG_sec * inv(C_GG_sec + α * C_e)

end

using CovarianceEstimation

"""Computes the EKI gain after estimating the sample covariance matrix 
using the estimator outlined by Ledoit and Wolf (2004)."""
function compute_gain_eki(
    localiser::LedoitWolfLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    function lw_norm_sq(A, p)
        return sum(A.^2) / p
    end
    
    Ξs = [ηs; Gs]

    # ΔΞ = compute_Δs(Ξs)
    # S = ΔΞ * ΔΞ'
    

    Nη, Ne = size(ηs)
    # NΞ = size(Ξs, 1)

    # # Compute empirical covariance matrix
    # ΔΞ = compute_Δs(Ξs)
    # S = ΔΞ * ΔΞ'
    # display(S)

    # m = tr(S) / NΞ
    # d_sq = lw_norm_sq(S-m*I, NΞ)
    # b_sq = 0.0
    # for Ξ_k ∈ eachcol(Ξs)
    #     b_sq += lw_norm_sq(Ξ_k*Ξ_k' - S, NΞ) / Ne^2
    # end
    # b_sq = min(b_sq, d_sq)
    # a_sq = d_sq - b_sq

    # C_LW = (b_sq / d_sq) * m * I + (a_sq / d_sq) * S
    
    F = I
    κ   = n #(-1) if using correction
    γ   = n / n # appears to just be 1?
    Xc² = Xc.^2

    ΣS² = sumij2(S) # Sum of squares of all elements of S
    λ   = sumij(cov(Xc²)) / γ^2 - ΣS² # What is γ? Note the square here...
    λ  /= κ * (ΣS² - 2tr(S) + pvar)
    T(κ/wn)

    target = DiagonalUnitVariance()
    shrinkage = :ss # Ledoit-Wolf optimal shrinkage 
    method = LinearShrinkage(target, shrinkage)
    C_LW = cov(method, Matrix(Ξs'))

    C_ηG = C_LW[1:Nη, (Nη+1):end]
    C_GG = C_LW[(Nη+1):end, (Nη+1):end]

    # display(C_ηG)
    # display(S[1:Nη, (Nη+1):end])
    # display(C_GG)
    # display(S[(Nη+1):end, (Nη+1):end])

    return C_ηG * inv(C_GG + α * C_e)

end

function run_ensemble(
    ηs::AbstractMatrix, 
    F::Function, 
    G::Function, 
    pr::MaternField
)

    θs = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    Fs = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    Gs = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    return θs, Fs, Gs

end

"""Uses the standard EKI update to update the current ensemble."""
function eki_update(
    ηs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    α::Real, 
    y::AbstractVector, 
    μ_e::AbstractVector, 
    C_e::AbstractMatrix,
    localiser::Localiser
)

    ys = rand(MvNormal(y, α * C_e), Ne)
    K = compute_gain_eki(localiser, ηs, Gs, α, C_e)
    return ηs + K * (ys - Gs .- μ_e)

end

"""Updates the current ensemble without using any inflation."""
function eki_update(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    t::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    localiser::Localiser,
    inflator::IdentityInflator
)

    return eki_update(ηs, Gs, α, t, μ_e, C_e, localiser)

end

"""Updates the current ensemble using the adaptive localisation 
method outlined by Evensen (2009)."""
function eki_update(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    t::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    localiser::Localiser,
    inflator::AdaptiveInflator
)

    Nη, Ne = size(ηs)

    dummy_params = rand(Normal(), (inflator.n_dummy_params, Ne))
    for r ∈ eachrow(dummy_params)
        μ, σ = mean(r), std(r)
        r = (r .- μ) ./ σ
    end
    
    ηs_aug = eki_update(
        [ηs; dummy_params], Gs,
        α, t, μ_e, C_e, localiser
    )

    ηs_new = ηs_aug[1:Nη, :]
    dummy_params = ηs_aug[Nη+1:end, :]
    ρ = 1 / mean(std(dummy_params, dims=2))
    @info "Inflation factor: $(ρ)."

    μ_η = mean(ηs_new, dims=2)
    return ρ * (ηs_new .- μ_η) .+ μ_η

end

"""Computes the EKI inflation factor following Iglesias and Yang 
(2021)."""
function compute_α_dmc(
    t::Real, 
    Gs::AbstractMatrix, 
    μ_e::AbstractVector,
    y::AbstractVector, 
    NG::Int, 
    C_e_invsqrt::AbstractMatrix
)

    φs = 0.5 * sum((C_e_invsqrt * (Gs .+ μ_e .- y)).^2, dims=1)

    μ_φ = mean(φs)
    var_φ = var(φs)
    
    α_inv = max(NG / 2μ_φ, √(NG / 2var_φ))
    α_inv = min(α_inv, 1-t)

    return α_inv ^ -1

end

"""Runs the EKI-DMC algorithm (Iglesias and Yang, 2021)."""
function run_eki_dmc(
    F::Function,
    G::Function,
    pr::MaternField, 
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    localiser::Localiser=IdentityLocaliser(),
    inflator::Inflator=IdentityInflator()
)

    println("It. | t")

    C_e_invsqrt = √(inv(C_e))
    NG = length(y)

    ηs_i = rand(pr, Ne)
    θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)

    ηs = [ηs_i]
    θs = [θs_i]
    Fs = [Fs_i]
    Gs = [Gs_i]

    i = 0
    t = 0
    converged = false

    while !converged

        α_i = compute_α_dmc(t, Gs_i, μ_e, y, NG, C_e_invsqrt)
        t += α_i^-1
        if abs(t - 1.0) < TOL
            converged = true
        end

        ηs_i = eki_update(ηs_i, Gs_i, α_i, y, μ_e, C_e, localiser, inflator)
        θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        @printf "%2i  | %.2e \n" i t

    end

    return ηs, θs, Fs, Gs

end

# ----------------
# EnRML
# ----------------

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

function compute_S(Gs, ys, μ_e, C_e_invsqrt)
    # TODO: check whether perturbed observations should be in here...
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

function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt₀::Real=2.0,
    t_stop::Real=2.0
)

    println("It. | Δt       | t        | misfit ")

    NG = length(d_obs)

    ηs = []
    θs = []
    Fs = []
    Gs = []

    ηs_i = rand(pr, Ne)
    θs_i, Fs_i, Gs_i = @time run_ensemble(ηs_i, F, G, pr)

    push!(ηs, ηs_i)
    push!(θs, θs_i)
    # push!(Fs, Fs_i)
    push!(Gs, Gs_i)

    t = 0
    i = 1
    while true

        μ_G = mean(Gs_i, dims=2)
        μ_η = mean(ηs_i, dims=2)
       
        C_ηη = cov(ηs[end], dims=2) # TODO: localisation / sampling error correction?
        D = (1.0 / Ne) * (Gs_i .- μ_G)' * (C_e \ (Gs_i .+ μ_e .- d_obs))
        ζ = rand(MvNormal(C_ηη + 1e-8 * I), Ne)
        
        Δt = Δt₀ / (norm(D) + 1e-6)
        t += Δt

        μ_misfit = mean(abs.(Gs[end] .+ μ_e .- d_obs))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
        
        A_n = I + Δt * C_ηη
        B_n = ηs_i - 
            Δt * (ηs_i .- μ_η) * D +
            Δt * ((NG + 1) / Ne) * (ηs_i .- μ_η)

        ηs_i = @time (A_n \ B_n) + √(2 * Δt) * ζ
        θs_i, Fs_i, Gs_i = @time run_ensemble(ηs_i, F, G, pr)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        # push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        if t ≥ t_stop 
            return ηs, θs, Fs, Gs
        end

    end

end

function save_results(results::Dict, fname::AbstractString)

    for (k, v) in pairs(results)
        h5write(fname, k, v)
    end

end