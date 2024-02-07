CONV_TOL = 1e-8

function compute_gain_eki(
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    C_θG, C_GG = compute_covs(θs, Gs)
    return C_θG * inv(C_GG + α * C_e)

end

"""Computes the EKI gain without applying any localisation."""
function compute_gain_eki(
    ::IdentityLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    return compute_gain_eki(θs, Gs, α, C_e)

end

"""Computes the EKI gain using the localisation method outlined by 
Zhang and Oliver (2010)."""
function compute_gain_eki(
    localiser::BootstrapLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    Nθ, J = size(θs)
    NG, J = size(Gs)
    K = compute_gain_eki(θs, Gs, α, C_e)
    Ks_boot = zeros(Nθ, NG, localiser.n_boot)

    for k ∈ 1:localiser.n_boot 
        inds_res = rand(1:J, J)
        θs_res = θs[:, inds_res]
        Gs_res = Gs[:, inds_res]
        K_res = compute_gain_eki(θs_res, Gs_res, α, C_e)
        Ks_boot[:, :, k] = K_res
    end

    var_Kis = mean((Ks_boot .- K).^2, dims=3)[:, :, 1]
    R² = var_Kis ./ (K.^2)

    if localiser.type == :regularised
        Ψ = 1 ./ (1 .+ R² * (1 + 1 / localiser.σ^2))
    elseif localiser.type == :unregularised
        Ψ = (1 .- (R² ./ (localiser.n_boot - 1))) ./ (R² .+ 1)
        Ψ = max.(Ψ, 0.0)
    end

    return Ψ .* K 

end

"""Computes the EKI gain using a variant of the localisation method 
outlined by Luo and Bhakta (2022)."""
function compute_gain_eki(
    localiser::ShuffleLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    K = compute_gain_eki(θs, Gs, α, C_e)
    return localise(localiser, θs, Gs, K)

end

"""Computes the EKI gain using the localisation method outlined by 
Flowerdew (2015)."""
function compute_gain_eki(
    localiser::FisherLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    K = compute_gain_eki(θs, Gs, α, C_e)
    return localise(localiser, θs, Gs, K)

end

"""Computes the EKI gain using the localisation method outlined by 
Lee (2021)."""
function compute_gain_eki(
    localiser::PowerLocaliser,
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real, 
    C_e::AbstractMatrix
)

    V_θ = Diagonal(std(θs, dims=2)[:])
    V_G = Diagonal(std(Gs, dims=2)[:])

    R_θG, R_GG = compute_cors(θs, Gs)

    R_θG_sec = R_θG .* (abs.(R_θG) .^ localiser.α)
    R_GG_sec = R_GG .* (abs.(R_GG) .^ localiser.α)

    C_θG_sec = V_θ * R_θG_sec * V_G
    C_GG_sec = V_G * R_GG_sec * V_G

    return C_θG_sec * inv(C_GG_sec + α * C_e)

end

"""Uses the standard EKI update to update the current ensemble."""
function eki_update(
    θs::AbstractMatrix, 
    Gs::AbstractMatrix, 
    α::Real, 
    y::AbstractVector, 
    μ_e::AbstractVector, 
    C_e::AbstractMatrix,
    localiser::Localiser
)

    ys = rand(MvNormal(y, α * C_e), size(θs, 2))
    K = compute_gain_eki(localiser, θs, Gs, α, C_e)
    return θs + K * (ys - Gs .- μ_e)

end

"""Updates the current ensemble without using any inflation."""
function eki_update(
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    localiser::Localiser,
    ::IdentityInflator
)

    return eki_update(θs, Gs, α, y, μ_e, C_e, localiser)

end

"""Updates the current ensemble using the adaptive localisation 
method outlined by Evensen (2009)."""
function eki_update(
    θs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    localiser::Localiser,
    inflator::AdaptiveInflator
)

    Nθ, J = size(θs)
    dummy_params = generate_dummy_params(inflator, J)
    
    θs_aug = eki_update(
        [θs; dummy_params], Gs,
        α, y, μ_e, C_e, localiser
    )

    θs_new = θs_aug[1:Nθ, :]
    dummy_params = θs_aug[Nθ+1:end, :]
    ρ = 1 / mean(std(dummy_params, dims=2))
    @info "Inflation factor: $(ρ)."

    μ_θ = mean(θs_new, dims=2)
    return ρ * (θs_new .- μ_θ) .+ μ_θ

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
    J::Int;
    localiser::Localiser=IdentityLocaliser(),
    inflator::Inflator=IdentityInflator()
)

    println("It. | t")

    C_e_invsqrt = √(inv(C_e))
    NG = length(y)

    θs_i = rand(pr, J)
    us_i, Fs_i, Gs_i = run_ensemble(θs_i, F, G, pr)

    θs = [θs_i]
    us = [us_i]
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

        θs_i = eki_update(θs_i, Gs_i, α_i, y, μ_e, C_e, localiser, inflator)
        us_i, Fs_i, Gs_i = run_ensemble(θs_i, F, G, pr)

        push!(θs, θs_i)
        push!(us, us_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        @printf "%2i  | %.2e \n" i t

    end

    return θs, us, Fs, Gs

end