
using Ferrite
using SparseArrays

export calculate_bilinear_map!, calculate_bilinear_map
# Assemble the projection of ∇(u) ⋅ ∇(λ) onto the FE space.
# This computes rhs_i = ∫ (∇u ⋅ ∇λ) ϕ_i dΩ for each test function ϕ_i.
function calculate_bilinear_map!(rhs::AbstractVector, a::AbstractVector, b::AbstractVector, fe::FerriteFESpace, M)
    cellvalues = fe.cellvalues
    dh = fe.dh
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)
    ∇a_q = zero(Vec{2,Float64})
    ∇b_q = zero(Vec{2,Float64})
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)

        ae = a[dofs]
        be = b[dofs]

        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)

            fill!(∇a_q, 0.0)
            fill!(∇b_q, 0.0)

            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇a_q += ae[j] * ∇ϕⱼ
                ∇b_q += be[j] * ∇ϕⱼ
            end

            grad_dot_product = ∇a_q ⋅ ∇b_q

            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += grad_dot_product * ϕᵢ * dΩ
            end
        end
        assemble!(rhs, dofs, re)
    end

    return M \ rhs
end

function calculate_bilinear_map(a::AbstractVector, b::AbstractVector, fe::FerriteFESpace, M)
    rhs = zeros(fe.n)
    return calculate_bilinear_map!(rhs, a, b, fe, M)
end
