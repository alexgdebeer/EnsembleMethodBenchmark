using LinearAlgebra

function run_ensemble(
    F::Function,
    G::Function,
    θs::AbstractArray,
    NF::Int,
    NG::Int,
    Ne::Int
)::Tuple{AbstractMatrix, AbstractMatrix}

    Fs = zeros(NF, Ne)
    Gs = zeros(NG, Ne)

    for i ∈ 1:Ne
        Fs[:,i] = @time F(θs[:,i])
        Gs[:,i] = G(Fs[:,i])
    end

    return Fs, Gs

end

function calculate_deviations(
    xs::AbstractMatrix, 
    N::Int
)::AbstractMatrix

    return (xs .- mean(xs, dims=2)) ./ √(N-1)

end

function run_eki_dmc(
    F::Function,
    G::Function,
    p, # TODO: define a suitable type for this
    ys::AbstractVector,
    Γ::AbstractMatrix,
    NF::Int,
    Ne::Int,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    NG = length(ys)
    L = cholesky(inv(Γ)).U # U vs L doesn't matter, matrix is diagonal

    θs = []
    Fs = []
    Gs = []
    αs = []

    θs_i = rand(p, Ne)
    Fs_i, Gs_i = run_ensemble(F, G, θs_i, NF, NG, Ne)

    push!(θs, θs_i)
    push!(Fs, Fs_i)
    push!(Gs, Gs_i)

    t = 0
    converged = false

    while !converged

        # Generate new inflation factor
        φs = [0.5sum((L * (ys - Gs[end][:, i])).^2) for i ∈ 1:Ne]
        α_i = min(max(NG / 2mean(φs), √(NG / 2var(φs))), 1-t)^-1
        push!(αs, α_i) 

        println("Q1: $(NG / 2mean(φs))")
        println("Q2: $(√(NG / 2var(φs)))")

        t += α_i^-1
        if abs(t - 1.0) < 1e-8
            converged = true
        end 

        Δθ = calculate_deviations(θs[end], Ne)
        ΔG = calculate_deviations(Gs[end], Ne)

        C_GG = ΔG * ΔG'
        C_θG = Δθ * ΔG'

        # TODO: covariance localisation

        ϵs = rand(MvNormal(αs[end]*Γ), Ne)

        θs_i = θs[end] + C_θG * inv(C_GG + αs[end]*Γ) * (ys .+ ϵs - Gs[end])
        Fs_i, Gs_i = run_ensemble(F, G, θs_i, NF, NG, Ne)

        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        if verbose
            @info "Iteration $(length(αs)) complete. t = $(t)."
        end

    end

    return θs, Fs, Gs, αs

end

include("setup.jl")

NF = grid_c.nx * grid_c.ny * (grid_c.nt+1)
Ne = 100

θs, Fs, Gs, αs = run_eki_dmc(F, G, p, us_o, Γ, NF, Ne)

# TODO: plot the other things...
logps = hcat([vec(transform(p, θ)) for θ in eachcol(θs[end])]...)

μ_post = reshape(mean(logps, dims=2), grid_c.nx, grid_c.ny)
σ_post = reshape(std(logps, dims=2), grid_c.nx, grid_c.ny)