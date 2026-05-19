export assemble_grad

# Returns a n × dim vector that contains gradient of coefficient vector.
function assemble_grad(z::AbstractVector, cellvalues::CellValues,dh::DofHandler, ndims::Int)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    N = ndof(dh)
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