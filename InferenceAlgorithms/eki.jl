CONV_TOL = 1e-8

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

"""Computes the EKI gain using the localisation method outlined by 
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

"""Computes the EKI gain using a variant of the localisation method 
outlined by Luo and Bhakta (2022)."""
function compute_gain_eki(
    localiser::ShuffleLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

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

    for i ∈ 1:Nη, j ∈ 1:NG
        z = (1 - abs(R_ηG[i, j])) / (1 - σs_e[i, j])
        P[i, j] = gaspari_cohn(z)
    end

    localiser.P = P
    return P .* K

end

"""Computes the EKI gain using the localisation method outlined by 
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

    for i ∈ 1:Nη, j ∈ 1:NG 
        ρ_ij = R_ηG[i, j]
        s = log((1+ρ_ij) / (1-ρ_ij)) / 2
        σ_s = (tanh(s + √(Ne-3)^-1) - tanh(s - √(Ne-3)^-1)) / 2
        P[i, j] = ρ_ij^2 / (ρ_ij^2 + σ_s^2)
    end

    return P .* K

end

"""Computes the EKI gain using the localisation method outlined by 
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
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    localiser::Localiser,
    inflator::IdentityInflator
)

    return eki_update(ηs, Gs, α, y, μ_e, C_e, localiser)

end

"""Updates the current ensemble using the adaptive localisation 
method outlined by Evensen (2009)."""
function eki_update(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    y::AbstractVector,
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
        α, y, μ_e, C_e, localiser
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
        if abs(t - 1.0) < CONV_TOL
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