"""Some algorithms from Huang et al (2022)."""

function run_seki(
    F::Function,
    G::Function,
    pr::MaternField, 
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt::Real=0.5
)

    # Sample intial ensemble from prior 
    θs = rand(pr, Ne)
    m = mean(θs, dims=2)

    Nθ = length(m) # TODO: clean up
    NG = length(y)

    x = vcat(y - μ_e, zeros(Nθ))

    Σ_ν = Matrix(1.0I, NG+Nθ, NG+Nθ)
    display(Σ_ν)
    Σ_ν[1:NG, 1:NG] .= C_e
    Σ_ν .*= (1 / Δt)

    for i ∈ 1:2
        
        print(i)

        # Prediction step 
        m_p = copy(m)
        θs_p = m_p .+ √(1/(1-Δt)) * (θs .- m)

        # Analysis step 
        us_p, Fs_p, Gs_p = run_ensemble(θs, F, G, pr)
        xs_p = vcat(Gs_p, zeros(Nθ, Ne))
        
        Δθ_p = θs_p .- mean(θs_p, dims=2)
        Δx_p = xs_p .- mean(xs_p, dims=2)

        C_θx = (Ne-1)^1 * Δθ_p * Δx_p'
        C_xx = (Ne-1)^1 * Δx_p * Δx_p' + Σ_ν
        
        νs = rand(MvNormal(Σ_ν), Ne)

        θs = θs_p .+ C_θx * inv(C_xx) * (x .- xs_p .- νs)

    end

    return θs

end