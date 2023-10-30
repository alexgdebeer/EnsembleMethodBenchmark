using LinearAlgebra
using Printf

function calculate_deviations(
    xs::AbstractMatrix, 
    N::Int
)::AbstractMatrix

    return (xs .- mean(xs, dims=2)) ./ √(N-1)

end

function run_eki_dmc(
    F::Function,
    G::Function,
    pr::MaternField, 
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    L_e::AbstractMatrix,
    Ne::Int,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    compute_θs(ηs) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    compute_Fs(θs) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    compute_Gs(Fs) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    NG = length(d_obs)

    ηs = [rand(pr, Ne)]
    θs = [compute_θs(ηs[1])]
    Fs = [compute_Fs(θs[1])]
    Gs = [compute_Gs(Fs[1])]
    αs = []

    t = 0
    converged = false

    while !converged

        # Generate new inflation factor
        φs = [0.5sum((L_e * (d_obs - Gs[end][:, i])).^2) for i ∈ 1:Ne]
        α_i = min(max(NG / 2mean(φs), √(NG / 2var(φs))), 1-t)^-1

        @info "Data-misfit mean: $(NG / 2mean(φs))"
        @info "Data-misfit variance: $(√(NG / 2var(φs)))"

        push!(αs, α_i) 
        t += α_i^-1
        if abs(t - 1.0) < 1e-8
            converged = true
        end 

        Δη = calculate_deviations(ηs[end], Ne)
        ΔG = calculate_deviations(Gs[end], Ne)

        C_GG = ΔG * ΔG'
        C_ηG = Δη * ΔG'

        # TODO: covariance localisation

        es = rand(MvNormal(αs[end] * Γ_e), Ne) # TODO: tidy up

        ηs_i = ηs[end] + C_ηG * inv(C_GG + αs[end] * Γ_e) * (d_obs .+ es - Gs[end] .- μ_e)
        θs_i = compute_θs(ηs_i)
        Fs_i = compute_Fs(θs_i)
        Gs_i = compute_Gs(Fs_i)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        if verbose
            @info "Iteration $(length(αs)) complete. t = $(t)."
        end

    end

    return ηs, θs, Fs, Gs, αs

end