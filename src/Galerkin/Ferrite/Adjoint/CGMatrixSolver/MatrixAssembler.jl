# Write an allocation free matrix assembler:

mutable struct LMatrixAssembler
    n_basefuncs
    Le
    assembler
end
function LMatrixAssembler(fe::FerriteFESpace)

end

mutable struct MatrixAMG
    L # The matrix
    AMG # the Preconditioner
end

#=
n_basefuncs = getnbasefunctions(cellvalues)
Le = zeros(n_basefuncs, n_basefuncs)
assembler = start_assemble(L)
=#

function assemble_L!(L::AbstractMatrix, fe::FerriteFESpace, γ::AbstractVector, L_assem::LMatrixAssembler)
    cellvalues = fe.cellvalues
    dh = fe.dh
    fill!(L, 0.0)
    n_basefuncs = L_assem.n_basefuncs
    Le = L_assem.Le
    assembler = L_assem.assembler

    for cell in CellIterator(dh)
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            γe = γ[celldofs(cell)] # (Edit) Could be done more efficiently by copying into preallocated array
            σ = function_value(cellvalues, q, γe)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Le[i, j] += σ * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    return L
end
