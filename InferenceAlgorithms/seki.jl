"""Some algorithms from Huang et al (2022)."""

function compute_C_ν(
    C_e::AbstractMatrix,
    Nθ::Int,
    NG::Int,
    Δt::Real
)

    C_ν = Matrix(1.0I, NG+Nθ, NG+Nθ)
    C_ν[1:NG, 1:NG] .= C_e
    C_ν .*= (1 / Δt)

    return C_ν

end

function run_prediction(
    m::AbstractVecOrMat,
    θs::AbstractMatrix,
    Δt::Real
)

    m̂ = copy(m)
    θ̂s = m̂ .+ √(1/(1-Δt)) * (θs .- m̂)

    return m̂, θ̂s

end

function compute_covs(
    m̂::AbstractVecOrMat,
    θ̂s::AbstractMatrix,
    x̂s::AbstractMatrix,
    C_ν::AbstractMatrix,
    J::Int
)

    x̂ = mean(x̂s, dims=2)

    C_θx = (1 / (J-1)) * (θ̂s .- m̂) * (x̂s .- x̂)'
    C_xx = (1 / (J-1)) * (x̂s .- x̂) * (x̂s .- x̂)' + C_ν

    return C_θx, C_xx

end

function run_analysis_seki(
    m̂::AbstractVecOrMat, 
    θ̂s::AbstractMatrix, 
    x::AbstractVecOrMat,
    x̂s::AbstractMatrix, 
    C_ν::AbstractMatrix, 
    ν_dist::MvNormal,
    J::Int
)

    C_θx, C_xx = compute_covs(m̂, θ̂s, x̂s, C_ν, J)
    νs = rand(ν_dist, J)

    θs = θ̂s .+ C_θx * (C_xx \ (x .- x̂s .- νs))
    m = mean(θs, dims=2)

    return m, θs

end

function run_analysis_eaki(
    m̂::AbstractVecOrMat, 
    θ̂s::AbstractMatrix, 
    x::AbstractVecOrMat,
    x̂s::AbstractMatrix, 
    C_ν::AbstractMatrix, 
    C_ν_i::AbstractMatrix,
    J::Int
)

    x̂ = mean(x̂s, dims=2)

    Ẑ = (√(J-1))^-1 * (θ̂s .- m̂)
    Ŷ = (√(J-1))^-1 * (x̂s .- x̂)

    P, D̂_sqrt, V = svd(Ẑ)
    U, D, _ = svd(V' * ((I + Ŷ' * C_ν_i * Ŷ) \ V))

    D̂_sqrt = Diagonal(D̂_sqrt)
    D = Diagonal(D)

    A = P * D̂_sqrt * U * sqrt(D) * inv(D̂_sqrt) * P'

    C_θx, C_xx = compute_covs(m̂, θ̂s, x̂s, C_ν, J)

    m = m̂ + C_θx * (C_xx \ (x .- x̂))
    θs = m .+ A * (θ̂s .- m̂)

    return m, θs

end

function run_seki(
    F::Function,
    G::Function,
    pr::MaternField, 
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    J::Int;
    Δt::Real=0.5
)

    θs = rand(pr, J)
    m = mean(θs, dims=2)

    Nθ = length(m)
    NG = length(y)

    x = vcat(y - μ_e, zeros(Nθ))

    C_ν = compute_C_ν(C_e, Nθ, NG, Δt)
    ν_dist = MvNormal(C_ν)

    for i ∈ 1:10

        m̂, θ̂s = run_prediction(m, θs, Δt)

        _, _, Ĝs = run_ensemble(θ̂s, F, G, pr)
        x̂s = vcat(Ĝs, θ̂s)

        println(mean((Ĝs .- y)' * inv(C_e) * (Ĝs .- y)))

        m, θs = run_analysis_seki(m̂, θ̂s, x, x̂s, C_ν, ν_dist, J)

    end

    us, Fs, Gs = run_ensemble(θs, F, G, pr)
    return θs, us, Fs, Gs

end

function run_eaki(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    J::Int;
    Δt::Real=0.5
)

    θs = rand(pr, J)
    m = mean(θs, dims=2)

    Nθ = length(m)
    NG = length(y)

    x = vcat(y - μ_e, zeros(Nθ))

    C_ν = compute_C_ν(C_e, Nθ, NG, Δt)
    C_ν_i = inv(C_ν)

    for i ∈ 1:10
        
        # println(i)

        m̂, θ̂s = run_prediction(m, θs, Δt)

        # Analysis step 
        _, _, Ĝs = run_ensemble(θ̂s, F, G, pr)
        x̂s = vcat(Ĝs, θ̂s)

        println(mean((Ĝs .- y)' * inv(C_e) * (Ĝs .- y)))

        m, θs = run_analysis_eaki(m̂, θ̂s, x, x̂s, C_ν, C_ν_i, J)

    end

    us, Fs, Gs = run_ensemble(θs, F, G, pr)

    return θs, us, Fs, Gs

end

function run_etki(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int,
    Δt::Real=0.5
)

    θs = rand(pr, Ne)
    m = mean(θs, dims=2)

    Nθ = length(m)
    NG = length(y)

    x = vcat(y - μ_e, zeros(Nθ))

    C_ν = compute_C_ν(C_e, Nθ, NG, Δt)
    C_ν_i = inv(C_ν)

    for i ∈ 1:10
        
        println(i)
        
        m̂, θ̂s = run_prediction(m, θs, Δt)

        # Analysis step
        _, _, Ĝs = run_ensemble(θ̂s, F, G, pr)
        x̂s = vcat(Ĝs, θ̂s)

        println(mean((Ĝs .- y)' * inv(C_e) * (Ĝs .- y)))

        Z_p = (√(Ne-1))^-1 * (θs_p .- m_p)
        Y_p = (√(Ne-1))^-1 * (xs_p .- mean(xs_p, dims=2))
        
        P, Γ, _ = svd(Y_p' * Σ_ν_i * Y_p)
        T = P * sqrt(Diagonal(Γ) + I) * P'

        # θ = similar(θ_p) 
        # for j = 1:N_ens;  θ[j, :] .=  θ_p[j, :] - θ_p_mean;  end
        # # Z' = （Z_p * T)' = T' * Z_p
        # θ .= T' * θ 
        # for j = 1:N_ens;  θ[j, :] .+=  θ_mean;  end
        
        Δx_p = xs_p .- mean(xs_p, dims=2)
        C_θx = (1/(Ne-1)) * (θs_p .- m_p) * Δx_p'
        C_xx = (1/(Ne-1)) * Δx_p * Δx_p' + Σ_ν

        m = m_p + C_θx * (C_xx \ (x .- mean(xs_p)))
        Z = Z_p * T
        θs = (√(Ne-1) * Z) .+ m_p

    end

    us, Fs, Gs = run_ensemble(θs, F, G, pr)

    return θs, us, Fs, Gs

end