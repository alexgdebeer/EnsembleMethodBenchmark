TOL = 1e-8 # Convergence tolerance

abstract type Localiser end

struct IdentityLocaliser <: Localiser end
struct FisherLocaliser <: Localiser end

mutable struct CycleLocaliser <: Localiser

    num_cycle::Int 
    P # TODO: compute once and for all?
    
    function CycleLocaliser(num_cycle=50)
        return new(num_cycle, nothing)
    end

end

struct BootstrapLocaliser <: Localiser

    num_boot::Int 
    σ::Real 
    tol::Real

    function BootstrapLocaliser(num_boot=50, σ=0.6, tol=1e-8) 
        return new(num_boot, σ, tol)
    end

end

struct PowerLocaliser <: Localiser
    α::Real 
    PowerLocaliser(α=0.4) = new(α)
end

function compute_covs(
    ηs::AbstractMatrix, 
    Gs::AbstractMatrix
)

    Nη = size(ηs, 1)

    C = cov([ηs; Gs], dims=2)
    C_ηG = C[1:Nη, (Nη+1):end]
    C_GG = C[(Nη+1):end, (Nη+1):end]

    return C_ηG, C_GG

end

function compute_cors(
    ηs::AbstractMatrix,
    Gs::AbstractMatrix
)

    Nη = size(ηs, 1)

    R = cor([ηs; Gs], dims=2)
    R_ηG = R[1:Nη, (Nη+1):end]
    R_GG = R[(Nη+1):end, (Nη+1):end]

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
    Ks_boot = zeros(Nη, NG, localiser.num_boot)

    for k ∈ 1:localiser.num_boot 
        inds_res = rand(1:Ne, Ne)
        ηs_res = ηs[:, inds_res]
        Gs_res = Gs[:, inds_res]
        K_res = compute_gain_eki(ηs_res, Gs_res, α, C_e)
        Ks_boot[:, :, k] = K_res
    end

    var_Kis = mean((Ks_boot .- K).^2, dims=3)[:, :, 1]
    R² = var_Kis ./ (K.^2 .+ localiser.tol)
    P = 1 ./ (1 .+ R² * (1 + 1 / localiser.σ^2)) # TODO: tidy? this has no relation to α above

    return P .* K 

end

"""Computes the EKI gain using a variant of the localisation scheme 
outlined by Luo and Bhakta (2022)."""
function compute_gain_eki(
    localiser::CycleLocaliser,
    ηs::AbstractMatrix,
    Gs::AbstractMatrix,
    α::Real,
    C_e::AbstractMatrix
)

    function gaspari_cohn(z)
        if 0 ≤ z ≤ 1 
            return -(1/4)z^5 + (1/2)z^4 + (5/8)z^3 - (5/3)z^2 + 1
        elseif 1 < z ≤ 2 
            return (1/12)z^5 - (1/2)z^4 + (5/8)z^3 + (5/3)z^2 - 5z + 4 - (2/3)z^-1
        else
            return 0.0
        end
    end

    Nη, Ne = size(ηs)
    NG, Ne = size(Gs)

    K = compute_gain_eki(ηs, Gs, α, C_e)
    R_ηG = compute_cors(ηs, Gs)[1]

    P = zeros(Nη, NG)
    R_ηGs = zeros(Nη, NG, Ne-1)

    for i ∈ 1:min(localiser.num_cycle, Ne-1)
        ηs_cycled = circshift(ηs, (0, i))
        R_ηGs[:, :, i] = compute_cors(ηs_cycled, Gs)[1]
    end

    σs_e = median(abs.(R_ηGs), dims=3) ./ 0.6745

    for i ∈ 1:Nη
        for j ∈ 1:NG
            z = (1 - abs(R_ηG[i, j])) / (1 - σs_e[i, j])
            P[i, j] = gaspari_cohn(z)
        end
    end

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
            s = 0.5 * log((1+ρ_ij) / (1-ρ_ij))
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

    φs = 0.5 * sum((C_e_invsqrt * (y .- Gs .- μ_e)).^2, dims=1)

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
    Ne::Int,
    localiser::Localiser=IdentityLocaliser(),
    verbose::Bool=true
)

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

        ηs_i = eki_update(ηs_i, Gs_i, α_i, y, μ_e, C_e, localiser)
        θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        if verbose
            @info "Iteration $i complete. t = $t." # TODO: improve this printing...
        end

    end

    ηs = cat(ηs..., dims=3)
    θs = cat(θs..., dims=3)
    Fs = cat(Fs..., dims=3)
    Gs = cat(Gs..., dims=3)

    return ηs, θs, Fs, Gs

end

# ----------------
# Things to update 
# ----------------


function compute_enrml_gain(
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractMatrix,
    VG::AbstractMatrix,
    Γ_e_sqi::AbstractMatrix,
    λ::Real, 
)
    return Δη * VG * ΛG * inv((λ+1)I + ΛG.^2) * UG' * Γ_e_sqi
end

function compute_Δs(
    xs::AbstractMatrix, 
    N::Int
)::AbstractMatrix

    return (xs .- mean(xs, dims=2)) ./ √(N-1)

end

function tsvd(
    A::AbstractMatrix; 
    e::Real=0.99
)::Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}

    U, Λ, V = svd(A)
    minimum(Λ) < 0 && error(minimum(Λ))
    λ_cum = cumsum(Λ)

    for i ∈ 1:length(Λ)
        if λ_cum[i] / λ_cum[end] ≥ e 
            return U[:, 1:i], Λ[1:i], V[:, 1:i]
        end
    end

end

function run_enrml(
    F::Function, 
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    Ne::Int,
    NF::Int,
    i_max::Int;
    γ::Real=10,
    λ_min::Real=0.01,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    # TODO: make these functions available to both algorithms
    compute_θs(ηs) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    compute_Fs(θs) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    compute_Gs(Fs) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    """Returns the mean and standard deviations of the sum-of-squares data 
    misfit of each ensemble member."""
    # TODO: this can probably be combined with the equivalent function in EKI-DMC
    function compute_S(Gs, ds_p, Γ_e_sqi)
        ΔG_sums = sum((Γ_e_sqi * (Gs - ds_p)).^2, dims=1)
        return mean(ΔG_sums)
    end

    # Define convergence parameters
    ΔS_min = 0.01 # Minimum reduction in objective
    Δη_min = 0.5 # Greatest allowable change in a parameter of an ensemble member (prior standard deviations)

    NG = length(d_obs)

    ηs = zeros(pr.Nη, Ne, i_max+1)
    θs = zeros(pr.Nθ, Ne, i_max+1)
    Fs = zeros(NF, Ne, i_max+1)
    Gs = zeros(NG, Ne, i_max+1)
    Ss = zeros(i_max+1)
    λs = zeros(i_max+1)

    # Define set of indices of successful parameter sets
    inds = 1:Ne

    # Compute scaling matrices # TODO: edit
    Γ_e_sqi = sqrt(inv(Γ_e))

    ds_p = rand(MvNormal(d_obs, Γ_e), Ne)

    ηs[:, :, 1] = rand(pr, Ne)
    θs[:, :, 1] = compute_θs(ηs[:, :, 1])
    Fs[:, :, 1] = compute_Fs(θs[:, :, 1])
    Gs[:, :, 1] = compute_Gs(Fs[:, :, 1])

    Ss[1] = compute_S(Gs[:, :, 1], ds_p, Γ_e_sqi)
    λs[1] = 10^floor(log10(Ss[1] / 2NG))

    Δη_pri = compute_Δs(ηs[:, :, 1], length(inds))
    Uη_pri, Λη_pri, = tsvd(Δη_pri)

    if verbose
        println("It. | status   | ΔS       | Δη_max   | λ")
    end

    i = 1
    n_cuts = 0

    while i ≤ i_max

        Δη = compute_Δs(ηs[:, :, i], Ne)
        ΔG = compute_Δs(Gs[:, :, i], Ne)

        UG, ΛG, VG = tsvd(Γ_e_sqi * ΔG)
        K = compute_enrml_gain(Δη, UG, Diagonal(ΛG), VG, Γ_e_sqi, λs[i])

        # Calculate corrections based on prior deviations
        δη_pri = Δη * VG * inv((λs[i] + 1)I + Diagonal(ΛG).^2) * VG' * Δη' *
                 Uη_pri * Diagonal(1 ./ Λη_pri.^2) * Uη_pri' *
                 (ηs[:, :, i] - ηs[:, :, 1])

        # Calculate corrections based on fit to observations
        δη_obs = K * (Gs[:,:,i] .+ μ_e .- ds_p)

        ηs[:, :, i+1] = ηs[:, :, i] - δη_pri - δη_obs
        θs[:, :, i+1] = compute_θs(ηs[:, :, i+1])
        Fs[:, :, i+1] = compute_Fs(θs[:, :, i+1])
        Gs[:, :, i+1] = compute_Gs(Fs[:, :, i+1])

        Ss[i+1] = compute_S(Gs[:, :, i+1], ds_p, Γ_e_sqi)

        if Ss[i+1] ≤ Ss[i]

            n_cuts = 0
            ΔS = 1 - Ss[i+1] / Ss[i]
            Δη_max = maximum(abs.(ηs[:, :, i+1] - ηs[:, :, i]))

            # Check for convergence
            if (ΔS ≤ ΔS_min) && (Δη_max ≤ Δη_min) # TODO: change S to Φ potentially?
                @info "Convergence criteria met."
                return ηs[:, :, 1:i+1], θs[:, :, 1:i+1], Fs[:, :, 1:i+1], Gs[:, :, 1:i+1], Ss[1:i+1], λs[1:i]
            end
            
            i += 1
            λs[i] = max(λs[i-1] / γ, λ_min)

            if verbose
                @printf(
                    "%2i  | accepted | %.2e | %.2e | %.2e \n",
                    i-1, ΔS, Δη_max, λs[i]
                )
            end
             
        else 

            n_cuts += 1
            if n_cuts == 5 
                @info "Terminating algorithm: $n_cuts consecutive cuts to the step size."
                return ηs[:, :, 1:i], θs[:, :, 1:i], Fs[:, :, 1:i], Gs[:, :, 1:i], Ss[1:i], λs[1:i]
            end

            λs[i] *= γ
            
            if verbose
                @printf(
                    "%2i  | rejected | -------- | -------- | %.2e \n",
                    i-1, λs[i]
                )
            end

        end

    end

    return ηs[:, :, 1:i], θs[:, :, 1:i], Fs[:, :, 1:i], Gs[:, :, 1:i], Ss[1:i], λs[1:i-1]

end

# TODO: use these in EKI and EnRML
compute_θs(ηs, pr) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
compute_Fs(θs, F) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
compute_Gs(Fs, G) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

# TODO: read ALDI paper and do the generalised square root 
function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt₀::Real=10.0,
    t_stop::Real=2.0,
    verbose::Bool=true
)

    NG = length(d_obs)

    ηs = [rand(pr, Ne)]
    θs = [compute_θs(ηs[1], pr)]
    Fs = [compute_Fs(θs[1], F)]
    Gs = [compute_Gs(Fs[1], G)]

    verbose && println("It. | Δt       | t        | misfit ")

    t = 0
    i = 1
    while true

        Ḡ = mean(Gs[end], dims=2)
        η̄ = mean(ηs[end], dims=2)
       
        C_ηη = cov(ηs[end], dims=2, corrected=false)
        D = (1.0 / Ne) * (Gs[end] .- Ḡ)' * (C_e \ (Gs[end] .+ μ_e .- d_obs))
        ζ = rand(MvNormal(C_ηη + 1e-8 * I), Ne)
        
        Δt = Δt₀ / (norm(D) + 1e-6)
        t += Δt

        μ_misfit = mean(abs.(Gs[end] .+ μ_e .- d_obs))
        verbose && @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
        
        A_n = I + Δt * C_ηη
        B_n = ηs[end] - 
            Δt * (ηs[end] .- η̄) * D +
            Δt * ((NG + 1) / Ne) * (ηs[end] .- η̄)

        ηs_i = (A_n \ B_n) + √(2 * Δt) * ζ
    
        push!(ηs, ηs_i)
        push!(θs, compute_θs(ηs[end], pr))
        push!(Fs, compute_Fs(θs[end], F))
        push!(Gs, compute_Gs(Fs[end], G))

        i += 1
        t ≥ t_stop && break

    end

    return ηs, θs, Fs, Gs

end