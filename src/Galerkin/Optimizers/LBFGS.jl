export lbfgs

"""
    lbfgs(f, grad, x0; m=10, maxiter=100, tol=1e-6)

Simple L-BFGS optimizer.

Arguments
- f(x)::Real          objective
- grad(x)::Vector     gradient
- x0::Vector          initial point

Keyword args
- m       memory size
- maxiter iterations
- tol     gradient norm tolerance
"""
function lbfgs(f, grad, x0; m=10, maxiter=100, tol=1e-6)
    proj(x) = max.(x, 0.0)
    x = copy(x0)
    fx = f(x)
    println("Iteration 0 - Loss: $fx")
    g = copy(grad(x))

    S = Vector{Vector{Float64}}()  # s_k = x_{k+1}-x_k
    Y = Vector{Vector{Float64}}()  # y_k = g_{k+1}-g_k
    ρ = Float64[]

    function two_loop(g)
        q = copy(g)
        α = zeros(length(S))

        for i = length(S):-1:1
            α[i] = ρ[i] * dot(S[i], q)
            q .-= α[i] .* Y[i]
        end

        γ = isempty(Y) ? 1.0 : dot(S[end], Y[end]) / dot(Y[end], Y[end])
        r = γ .* q

        for i = 1:length(S)
            β = ρ[i] * dot(Y[i], r)
            r .+= (α[i] - β) .* S[i]
        end
        return -r
    end

    for k in 1:maxiter
        if norm(g) < tol
            break
        end

        p = two_loop(g)

        # simple backtracking line search
        α = 1.0

        while true
            x_trial = proj(x .+ α .* p)
            ft = f(x_trial)
            print("   Backtracking - Loss: $ft   ")
            dotp = dot(g, x_trial - x)
            print("Dot:  $(dotp)\n")
            if ft ≤ fx + 1e-4 * dotp
                break
            end
            α *= 0.5
            α < 1e-8 && break
        end
        x_new = proj(x .+ α .* p)
        fx = f(x_new)
        println("Iteration $k - Loss: $fx")
        g_new = copy(grad(x_new))

        s = x_new - x
        y = g_new - g

        if dot(s, y) > 1e-10
            push!(S, s)
            push!(Y, y)
            push!(ρ, 1 / dot(y, s))
            if length(S) > m
                popfirst!(S)
                popfirst!(Y)
                popfirst!(ρ)
            end
        end

        x, g = x_new, g_new
    end

    return x
end
