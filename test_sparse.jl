using SparseArrays
using LinearAlgebra
using LinearSolve
using ForwardDiff

function func(b)

    t = typeof(b[1])

    vs = Vector{t}()
    for _ ∈ 1:3
        push!(vs, 1.0)
    end

    A = sparse([1, 2, 3], [1, 2, 3], vs)
    b = sparsevec([1, 2, 3], b)
    
    u = solve(LinearProblem(A, b))

    println(u)

    return sum(abs.(u)) 

end

ForwardDiff.gradient(func, [1.0, 1.0, 1.0])