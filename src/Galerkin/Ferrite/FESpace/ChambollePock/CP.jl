export assemble_div_p

function assemble_div_p(p_nodal::Matrix, cellvalues::CellValues,
                         dh::DofHandler, ndims::Int)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    div_p = zeros(ndof(dh))

    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        re = zeros(n_basefuncs)

        # p at the dofs of this element: shape (n_basefuncs × ndims)
        p_e = p_nodal[dofs, :]

        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)

            # Interpolate ∇p at quadrature point → div(p) is trace of Jacobian
            div_p_q = 0.0
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)   # Vec{ndims}
                # p_e[i,:] is the dual vector at node i
                div_p_q += dot(p_e[i, :], ∇ϕᵢ)
            end

            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += div_p_q * ϕᵢ * dΩ
            end
        end
        div_p[dofs] .+= re
    end
    return div_p
end


function prox_TV(y::AbstractVector, cellvalues::CellValues, dh::DofHandler,
                 ndims::Int, λ::Float64, ρ::Float64;
                 σ::Float64=0.1,        # dual step size
                 max_iter::Int=200,
                 tol::Float64=1e-6)

    N = length(y)
    z = copy(y)                        # primal variable
    p = zeros(N, ndims)                # dual variable (vector field, per node)

    for k in 1:max_iter
        z_old = copy(z)

        # 1. Dual ascent: p ← p + σ ∇z
        grad = assemble_grad(z, cellvalues, dh, ndims)
        p .+= σ .* grad

        # 2. Project p onto unit ball pointwise: p[i,:] /= max(1, ||p[i,:]||)
        for i in 1:N
            n = norm(p[i, :])
            if n > 1.0
                p[i, :] ./= n
            end
        end

        # 3. Primal update: z ← (ρy + λ div(p)) / ρ
        #    Note: integration by parts flips sign: -∫div(p)z = ∫p·∇z
        #    So the update is z = y + (λ/ρ) * div(p)
        div_p = assemble_div_p(p, cellvalues, dh, ndims)
        z .= y .+ (λ / ρ) .* div_p

        # 4. Convergence check
        if norm(z - z_old) / (norm(z_old) + 1e-12) < tol
            @info "TV prox converged at iteration $k"
            break
        end
    end
    return z
end

function get_prox_TV(cellvalues, dh, ndims, λ, ρ; kwargs...)
    return (y) -> prox_TV(y, cellvalues, dh, ndims, λ, ρ; kwargs...)
end