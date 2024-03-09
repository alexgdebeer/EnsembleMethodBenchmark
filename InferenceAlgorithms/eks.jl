function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    J::Int;
    Δt₀::Real=2.0,
    t_stop::Real=2.0
)

    println("It. | Δt       | t        | misfit ")

    θs_i = rand(pr, J)
    us_i, Fs_i, Gs_i = run_ensemble(θs_i, F, G, pr)

    t = 0
    i = 1
    while true

        μ_G = mean(Gs_i, dims=2)
        μ_θ = mean(θs_i, dims=2)
       
        C_θθ = cov(θs_i, dims=2, corrected=false)
        D = (1.0 / J) * (Gs_i .- μ_G)' * (C_e \ (Gs_i .+ μ_e .- y))
        ζ = rand(MvNormal(C_θθ + 0.001 * Diagonal(diag(C_θθ))), J)
        
        Δt = Δt₀ / (norm(D) + 1e-8)
        t += Δt

        μ_misfit = mean(abs.(Gs_i .+ μ_e .- y))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
            
        # Prior mean is 0, prior covariance is identity
        θs_i = θs_i + Δt * (
            -(θs_i .- μ_θ) * D + 
            -C_θθ * θs_i + 
            ((pr.Nθ + 1) / J) * (θs_i .- μ_θ)
        ) + √(2 * Δt) * ζ

        us_i, Fs_i, Gs_i = run_ensemble(θs_i, F, G, pr)

        i += 1
        if t ≥ t_stop 
            return θs_i, us_i, Fs_i, Gs_i, i
        end

    end

end