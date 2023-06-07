export calculate_map, run_rml, run_rto

function gauss_newton(
    f::Function, 
    x0::AbstractVector;
    β::Real=1.0,
    ny::Int=length(f(x0))
)

    # TODO: add a max_iters counter (more generally, tidy up covergence stuff)

    nx = length(x0)
    x_p = 10000 * ones(nx) 
    x = copy(x0)

    y_p = zeros(ny)
    y = zeros(ny)

    J = zeros(ny, nx)

    cfg = ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{20}())
    r = DiffResults.DiffResult(zeros(ny), zeros(ny, nx))

    while true

        r = ForwardDiff.jacobian!(r, f, x, cfg)
        y, J = DiffResults.value(r), DiffResults.jacobian(r)

        # g = ForwardDiff.gradient(x -> sum(f(x).^2), x)
        # n = norm(g)

        println(norm(x-x_p))

        norm(x-x_p) ≤ 1e-4 && break 

        x_p = copy(x)

        Δx = (J'*J + 1e-10I) \ J'*y
        x -= β*Δx

        y_p = copy(y)

    end

    return x, y, J

end

function calculate_map(
    f::Function, 
    g::Function,
    π::Distribution,
    L::Distribution,
    L_ϵ::AbstractMatrix,
    L_θ::AbstractMatrix;
    x0::Union{AbstractVector, Nothing}=nothing
)

    x0 = x0 === nothing ? π.μ : x0
    ny = length(L.μ) + length(π.μ)

    residuals(θ) = [L_ϵ * (g(f(θ)) - L.μ); L_θ * (θ - π.μ)]

    θ_map, y_map, J_map = gauss_newton(residuals, x0, ny=ny)

    # Extract and rescale the section of the Jacobian corresponding to the data
    J_map = inv(L_ϵ) * J_map[1:length(L.μ), :]

    return θ_map, y_map, J_map

end

function run_rml(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    N::Int;
    verbose::Bool=true
)

    N_θ = length(π.μ)
    θs = zeros(N_θ, N)
    
    evals = zeros(N)

    L_θ = cholesky(inv(π.Σ)).U
    L_ϵ = cholesky(inv(L.Σ)).U

    θ_map = calculate_map(f, g, π, L, L_θ, L_ϵ)

    for i ∈ 1:N

        θ_i = rand(π)
        y_i = rand(L)

        res = optimize(
            θ -> sum([L_ϵ*(g(f(θ))-y_i); L_θ*(θ-θ_i)].^2), 
            θ_map, 
            Newton(), 
            Options(show_trace=false, f_abstol=1e-10), 
            autodiff=:forward
        )

        if !Optim.converged(res)
            @warn "MAP estimate optimisation failed to converge."
        end

        θs[:,i] = Optim.minimizer(res)
        evals[i] = Optim.f_calls(res)

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    return θ_map, θs

end

function run_rto(
    f::Function,
    g::Function,
    π::GaussianPrior,
    L::GaussianLikelihood,
    N::Int;
    verbose::Bool=true
)

    L_θ = LinearAlgebra.cholesky(inv(π.Σ)).U  
    L_ϵ = LinearAlgebra.cholesky(inv(L.Σ)).U

    # Define augmented system 
    f̃(θ) = [L_ϵ*g(f(θ)); L_θ*θ]
    ỹ = [L_ϵ*L.μ; L_θ*π.μ]

    # Calculate the MAP estimate
    res = Optim.optimize(
        θ -> sum((f̃(θ).-ỹ).^2), [1.0, 1.0], 
        Optim.Newton(), 
        Optim.Options(show_trace=false), autodiff=:forward
    )
    
    !Optim.converged(res) && @warn "MAP estimate optimisation failed to converge."
    θ_MAP = Optim.minimizer(res)

    J̃θ_MAP = ForwardDiff.jacobian(f̃, θ_MAP)
    Q = Matrix(LinearAlgebra.qr(J̃θ_MAP).Q)

    θs = []
    ws = []
    evals = []

    for i ∈ 1:N

        # Sample an augmented vector
        ỹ_i = [L_ϵ*sample(L); L_θ*sample(π)]

        # Optimise to find the corresponding set of parameters
        res = Optim.optimize(
            θ -> sum((Q'*(f̃(θ).-ỹ_i)).^2), θ_MAP, 
            Optim.Newton(), 
            Optim.Options(show_trace=false), autodiff=:forward
        )

        !Optim.converged(res) && @warn "Optimisation failed to converge."
        Optim.minimum(res) > 1e-10 && @warn "Non-zero minimum: $(Optim.minimum(res))."
        θ = Optim.minimizer(res)

        J̃θ = ForwardDiff.jacobian(f̃, θ)

        # Compute the weight of the sample
        f̃θ = f̃(θ)
        w = exp(log(abs(LinearAlgebra.det(Q'*J̃θ))^-1) - 0.5sum((f̃θ-ỹ).^2) + 0.5sum((Q'*(f̃θ-ỹ)).^2))

        push!(θs, θ)
        push!(ws, w)
        push!(evals, Optim.f_calls(res))

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    # Normalise the weights
    ws ./= sum(ws)

    return θ_MAP, Q, θs, ws

end