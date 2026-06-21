export assemble_grad, assemble_div

# Returns a n × dim vector that contains gradient of coefficient vector.
function assemble_grad(z::AbstractVector, cellvalues::CellValues,dh::DofHandler, ndims::Int,N::Int)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    grad_z = zeros(N, ndims)

    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = z[dofs]
        re = zeros(n_basefuncs, ndims)

        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = function_gradient(cellvalues, q, ue)  # Vec{ndims}

            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                for d in 1:ndims
                    re[i, d] += ∇uh_q[d] * ϕᵢ * dΩ
                end
            end
        end
        grad_z[dofs, :] .+= re
    end
    return grad_z
end

function assemble_grad(z::AbstractVector, fe::FerriteFESpace)
    assemble_grad(z, fe.cellvalues, fe.dh, fe.dim, fe.n)
end

# I assume the div comes from the gradient method above.
function assemble_div(a::Matrix, cellvalues::CellValues, dh::DofHandler, ndims::Int, N::Int)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    div_a = zeros(N)

    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        re = zeros(n_basefuncs)

        # a at the dofs of this element: shape (n_basefuncs × ndims)
        a_e = a[dofs, :]

        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)

            # Interpolate ∇a at quadrature point → div(a) is trace of Jacobian
            div_a_q = 0.0
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)   # Vec{ndims}
                # a_e[i,:] is the dual vector at node i
                div_a_q += dot(a_e[i, :], ∇ϕᵢ)
            end

            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += div_a_q * ϕᵢ * dΩ
            end
        end
        div_a[dofs] .+= re
    end
    return div_a
end

function assemble_div(a::AbstractMatrix, fe::FerriteFESpace)
    assemble_div(a, fe.cellvalues, fe.dh, fe.dim, fe.n)
end


# Think about some test for that.