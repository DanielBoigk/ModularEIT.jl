export lbfgs

"""
    lbfgs(f, grad_f, x0; m=10, maxiter=100, tol=1e-6)

Simple L-BFGS optimizer for unconstrained minimization.

Arguments
- f(x)::Real         objective function to minimize
- grad_f(x)::Vector  gradient of objective (∇f)
- x0::Vector         initial point

Keyword args
- m       memory size for L-BFGS
- maxiter maximum iterations
- tol     gradient norm tolerance for convergence
"""
function lbfgs(f, descent_grad, x0; m=10, maxiter=100, tol=1e-6)
    proj(x) = max.(x, 0.0)
    x = proj(copy(x0))

    # Evaluate initial objective and descent direction
    fx = f(x)
    println("Iteration 0 - Loss: $fx")
    p_steepest = copy(descent_grad(x))  # Descent direction (negative gradient)

    S = Vector{Vector{Float64}}()  # s_k = x_{k+1}-x_k
    Y = Vector{Vector{Float64}}()  # y_k = descent_grad_{k+1}-descent_grad_k
    ρ = Float64[]

    function two_loop_direction(p_init)
        # Compute L-BFGS approximation: p_lbfgs = H * p_init
        # where p_init is the steepest descent direction and H ≈ (∇²f)⁻¹
        q = copy(p_init)
        α_vec = zeros(length(S))

        for i = length(S):-1:1
            α_vec[i] = ρ[i] * dot(S[i], q)
            q .-= α_vec[i] .* Y[i]
        end

        γ = isempty(Y) ? 1.0 : dot(S[end], Y[end]) / dot(Y[end], Y[end])
        r = γ .* q

        for i = 1:length(S)
            β = ρ[i] * dot(Y[i], r)
            r .+= (α_vec[i] - β) .* S[i]
        end
        return r  # Return H*p_init, the L-BFGS descent direction
    end

    for k in 1:maxiter
        p_norm = norm(p_steepest)
        if p_norm < tol
            println("Converged: descent direction norm = $p_norm")
            break
        end

        # Compute L-BFGS descent direction
        p = two_loop_direction(p_steepest)

        # Check descent condition (dot product should be positive)
        dotp = dot(p_steepest, p)

        if dotp <= 1e-14 * p_norm * norm(p)
            # Not a descent direction - use steepest descent instead
            p = copy(p_steepest)
            dotp = p_norm^2
        end

        # Backtracking line search with Armijo condition
        α = 1.0
        c1 = 1e-4  # Armijo condition parameter

        step_accepted = false
        x_new = copy(x)
        fx_new = fx

        for bt_iter in 1:50
            x_trial = proj(x .+ α .* p)
            ft = f(x_trial)

            # Armijo condition: f(x + α*p) ≤ f(x) - c1*α*(descent_dir · p)
            # We subtract because descent direction points downhill
            armijo_rhs = fx - c1 * α * dotp

            if ft ≤ armijo_rhs + 1e-14
                # Line search successful
                x_new = x_trial
                fx_new = ft
                step_accepted = true
                break
            end

            α *= 0.5
            if α < 1e-16
                break
            end
        end

        if !step_accepted
            # Emergency fallback: take tiny steepest descent step
            x_new = proj(x .+ 1e-10 .* p_steepest)
            fx_new = f(x_new)
        end

        println("Iteration $k - Loss: $fx_new")

        p_steepest_new = copy(descent_grad(x_new))

        s = x_new - x
        # y is the change in descent direction
        y = p_steepest_new - p_steepest

        # Update BFGS history if curvature condition is satisfied
        sy_dot = dot(s, y)
        if sy_dot > 1e-14 && norm(s) > 1e-14
            push!(S, copy(s))
            push!(Y, copy(y))
            push!(ρ, 1.0 / sy_dot)
            if length(S) > m
                popfirst!(S)
                popfirst!(Y)
                popfirst!(ρ)
            end
        end

        x = x_new
        fx = fx_new
        p_steepest = p_steepest_new
    end

    return x
end
