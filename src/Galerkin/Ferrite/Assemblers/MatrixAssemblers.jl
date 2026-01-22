using Ferrite
using SparseArrays
using LinearAlgebra

export assemble_M!, assemble_M
export assemble_K!, assemble_K
export assemble_L!, assemble_L

export to_dirichlet, to_dirichlet!
export build_projection_matrix, assemble_coupling_mass, assemble_coupling_mass!

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

struct L_Assembler
    n_basefuncs::Int
    Le::Matrix{Float64}
end

# This assembles L from function γ
function assemble_L!(L::AbstractMatrix, fe::FerriteFESpace, γ, ϵ=1e-12)
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

function assemble_L!(L::AbstractMatrix, fe::FerriteFESpace, γ::AbstractVector, ϵ::Float64=1e-12)
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

function assemble_L(fe::FerriteFESpace, γ, ϵ::Float64=1e-12)
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



"""
    build_projection_matrix(fine_space::FerriteFESpace, coarse_space::FerriteFESpace)

Build L2 projection matrix P from fine space to coarse space.
For a function u_fine in the fine space, u_coarse = P * u_fine minimizes
‖u_fine - u_coarse‖_{L2}
"""
function build_projection_matrix(fine_space::FerriteFESpace, coarse_space::FerriteFESpace)
    # The projection satisfies: M_coarse * u_coarse = B * u_fine
    # where B is the "coupling" mass matrix between spaces

    n_coarse = coarse_space.n
    n_fine = fine_space.n

    # Assemble coupling mass matrix B
    B = assemble_coupling_mass(coarse_space, fine_space)

    # Solve: M_coarse * P = B
    # So: P = M_coarse \ B
    P = coarse_space.M_fac \ B

    return P
end

"""
    assemble_coupling_mass!(B::AbstractMatrix, coarse_space::FerriteFESpace, fine_space::FerriteFESpace)

Assemble the coupling mass matrix B where B[i,j] = ∫ φ_coarse[i] * φ_fine[j] dΩ
"""
function assemble_coupling_mass!(B::AbstractMatrix, coarse_space::FerriteFESpace, fine_space::FerriteFESpace) # Careful. This thing only works because it is assumed that the shape & placemants of the elements are exactly identical and only the order is different. This can not project from i.e. triangular to quadrilateral elements.
    fill!(B, 0.0)

    dh_coarse = coarse_space.dh
    dh_fine = fine_space.dh
    cv_fine = fine_space.cellvalues

    # Get interpolation for coarse space
    RefElem = typeof(coarse_space).parameters[1]
    ip_coarse = Lagrange{RefElem,coarse_space.order}()

    # Create cell values with fine quadrature rule but coarse interpolation
    cv_coarse = CellValues(fine_space.cellvalues.qr, ip_coarse)

    n_basefuncs_coarse = getnbasefunctions(cv_coarse)
    n_basefuncs_fine = getnbasefunctions(cv_fine)
    Be = zeros(n_basefuncs_coarse, n_basefuncs_fine)
    assembler = start_assemble(B)

    for cell in CellIterator(dh_fine)
        fill!(Be, 0)
        reinit!(cv_fine, cell)
        reinit!(cv_coarse, cell)

        # Assemble local coupling matrix
        for q_point in 1:getnquadpoints(cv_fine)
            dΩ = getdetJdV(cv_fine, q_point)
            for i in 1:n_basefuncs_coarse
                φ_coarse = shape_value(cv_coarse, q_point, i)
                for j in 1:n_basefuncs_fine
                    φ_fine = shape_value(cv_fine, q_point, j)
                    Be[i, j] += φ_coarse * φ_fine * dΩ
                end
            end
        end

        # Assemble into global matrix
        cellid = cell.cellid

        coarse_dofs = celldofs(dh_coarse, cellid)
        fine_dofs   = celldofs(dh_fine,   cellid)

        assemble!(assembler, coarse_dofs, fine_dofs, Be)
        #=for i in 1:length(coarse_dofs)
            I = coarse_dofs[i]
            for j in 1:length(fine_dofs)
                J = fine_dofs[j]
                B[I, J] += Be[i, j]
            end
        end=#

    end

    return B
end

function assemble_coupling_mass(coarse_space::FerriteFESpace, fine_space::FerriteFESpace)
    B = spzeros(coarse_space.n, fine_space.n)
    assemble_coupling_mass!(B, coarse_space, fine_space)
    return B
end
