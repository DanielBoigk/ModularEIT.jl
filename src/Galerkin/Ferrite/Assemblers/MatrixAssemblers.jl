using Ferrite
using SparseArrays
using LinearAlgebra

export assemble_M!, assemble_M
export assemble_K!, assemble_K
export assemble_L!, assemble_L

# This is the mass matrix: ∫(u*v)dΩ
# Used for calculating L² distance
# Also used for Tikhonov L² regularization
# Mainly we need this as a projector unto FE Space
function assemble_M!(M::AbstractMatrix, dh::DofHandler, cellvalues::CellValues)
    fill!(M, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Me, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    φⱼ = shape_value(cellvalues, q_point, j)
                    Me[i, j] += φᵢ * φⱼ * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Me)
    end
    return M, cholesky(M)
end

function assemble_M!(M::AbstractMatrix, fe::FerriteFESpace)
    cellvalues = fe.cellvalues
    dh = fe.dh
    assemble_M!(M, cellvalues, dh)
end

function assemble_M(fe::FerriteFESpace)
    M = allocate_matrix(fe.dh)
    assemble_M!(M, fe)
end

function assemble_M(dh, cellvalues)
    M = allocate_matrix(dh)
    assemble_M!(M, dh, cellvalues)
end

# This is: ∫(∇(u)⋅∇(v))dΩ the stiffness matrix without specified coefficients.
# This is used for calculatin H¹ -scalar product
# It is also used for Tikhonov H¹ regularization.
function assemble_K!(K::AbstractMatrix, dh::DofHandler, cellvalues::CellValues)
    fill!(K, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K, factorize(K) # Cholesky decomposition might fail
end
function assemble_K!(K::AbstractMatrix, fe::FerriteFESpace)
    cellvalues = fe.cellvalues
    dh = fe.dh
    assemble_K!(K, dh, cellvalues)
end

function assemble_K(fe::FerriteFESpace)
    K = allocate_matrix(fe.dh)
    assemble_K!(K, fe)
end
function assemble_K(dh::DofHandler, cellvalues::CellValues)
    K = allocate_matrix(dh)
    assemble_K!(K, dh, cellvalues)
end
function assemble_K!(K::AbstractMatrix, fe::FerriteFESpace, ϵ::Float64)
    cellvalues = fe.cellvalues
    dh = fe.dh
    fill!(K, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    K += ϵ * I
    return K, cholesky(K)
end
function assemble_K(fe::FerriteFESpace, ϵ::Float64)
    K = allocate_matrix(fe.dh)
    assemble_K!(K, fe, ϵ)
end


# This assembles L from function γ
function assemble_L!(L::AbstractMatrix, fe::FerriteFESpace, γ, ϵ=0.0)
    cellvalues = fe.cellvalues
    dh = fe.dh
    fill!(L, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Le = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(L)
    for cell in CellIterator(dh)
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            x = spatial_coordinate(cellvalues, q, getcoordinates(cell))
            σ = γ(x)
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
    if ϵ ≠ 0.0
        L += ϵ * I
    end
    return L
end

function assemble_L!(L::AbstractMatrix, fe::FerriteFESpace, γ::AbstractVector, ϵ::Float64)
    cellvalues = fe.cellvalues
    dh = fe.dh
    fill!(L, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Le = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(L)
    for cell in CellIterator(dh)
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            γe = γ[celldofs(cell)] # (Edit) Could be done more efficiently by copying into preallocated array
            σ = function_value(cellvalues, q_point, γe)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                #u = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Le[i, j] += σ * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    if ϵ ≠ 0.0
        L += ϵ * I
    end
    return L
end

function assemble_L(fe::FerriteFESpace, γ, ϵ::Float64)
    L = allocate_matrix(fe.dh)
    return assemble_L!(L, fe, γ, ϵ)
end


# This function is to get a stiffness matrix for dirichlet boundary conditions:
function to_dirichlet!(L::AbstractArray, fe::FerriteFESpace)
    apply!(L, fe.ch)
    L
end
function to_dirichlet(L::AbstractArray, fe::FerriteFESpace)
    Ld = copy(L)
    apply!(Ld, fe.ch)
    Ld
end
