using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Krylov
using Ferrite

export FerriteBlockMode
export residual!, jacobian!, residual_and_jacobian!

# Neumann problems are only determined up to an additive constant (the
# stiffness matrix L(σ) has the constant vector in its nullspace), so we fix
# the gauge by bordering the system with a Lagrange multiplier that forces
# the mean of the boundary trace to zero:
#
#   [ L(σ)  w ] [ u ]   [ g ]
#   [ w'    0 ] [ μ ] = [ 0 ]
#
# where w is the indicator of the boundary dofs lifted into the full space
# (so w'u = sum of the boundary dof values of u). This is symmetric but
# indefinite, hence MINRES rather than CG.

struct BlockLAssembler
    n_basefuncs::Int
    Le::Matrix{Float64}
end

function BlockLAssembler(fe::FerriteFESpace)
    n_basefuncs = getnbasefunctions(fe.cellvalues)
    return BlockLAssembler(n_basefuncs, zeros(n_basefuncs, n_basefuncs))
end

# Assembles L(σ) where σ lives on its own (possibly lower-order) FE space fe.dh_σ.
function assemble_L!(L::AbstractMatrix, ba::BlockLAssembler, fe::FerriteFESpace, σ::AbstractVector)
    cellvalues = fe.cellvalues
    cellvalues_σ = fe.cellvalues_σ
    dh = fe.dh
    dh_σ = fe.dh_σ
    n_basefuncs = ba.n_basefuncs
    Le = ba.Le

    fill!(L, 0.0)
    assembler = start_assemble(L)
    for (cell, cell_σ) in zip(CellIterator(dh), CellIterator(dh_σ))
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        reinit!(cellvalues_σ, cell_σ)
        σ_local = σ[celldofs(cell_σ)]
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            σ_f = function_value(cellvalues_σ, q, σ_local)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Le[i, j] += σ_f * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    return L
end

# ∂/∂σ_j ∫ σ ∇u⋅∇λ dΩ = ∫ φ_σ,j ∇u⋅∇λ dΩ, assembled directly (no mass-matrix
# projection needed since this already is the exact partial derivative).
function assemble_gradient!(grad::AbstractVector, fe::FerriteFESpace, λ::AbstractVector, u::AbstractVector)
    cellvalues = fe.cellvalues
    cellvalues_σ = fe.cellvalues_σ
    dh = fe.dh
    dh_σ = fe.dh_σ
    n_basefuncs_σ = getnbasefunctions(cellvalues_σ)

    fill!(grad, 0.0)
    for (cell, cell_σ) in zip(CellIterator(dh), CellIterator(dh_σ))
        reinit!(cellvalues, cell)
        reinit!(cellvalues_σ, cell_σ)
        λ_local = λ[celldofs(cell)]
        u_local = u[celldofs(cell)]
        dofs_σ = celldofs(cell_σ)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            s = (function_gradient(cellvalues, q, λ_local) ⋅ function_gradient(cellvalues, q, u_local)) * dΩ
            for a in 1:n_basefuncs_σ
                grad[dofs_σ[a]] += shape_value(cellvalues_σ, q, a) * s
            end
        end
    end
    return grad
end

function bordered(L::SparseMatrixCSC, w::AbstractVector)
    wcol = sparse(reshape(w, :, 1))
    return [L wcol; sparse(wcol') spzeros(eltype(L), 1, 1)]
end

mutable struct FerriteBlockMode
    ba::BlockLAssembler       # scratch for assembling L(σ)
    w::AbstractVector         # bordering vector: indicator of boundary dofs, lifted to full space
    L::AbstractMatrix         # scratch stiffness matrix, sparsity pattern preallocated
    Lb::AbstractMatrix        # bordered (saddle-point) system for the current σ
    F::AbstractMatrix         # target boundary data, m × num_modes
    G::AbstractMatrix         # lifted Neumann current patterns, n × num_modes
    U::AbstractMatrix         # cached state solution, n × num_modes
    Λ::AbstractMatrix         # cached adjoint solution, n × num_modes
    res::AbstractMatrix       # boundary residual U|∂Ω - F, m × num_modes
    grad::AbstractVector      # scratch gradient for a single mode, n_σ
    J::AbstractMatrix         # Jacobian/gradient stacked over modes, n_σ × num_modes
    num_modes::Int
end

function FerriteBlockMode(F_in::AbstractMatrix, G_in::AbstractMatrix, fe::FerriteFESpace)
    @assert size(F_in, 1) == fe.m "F must be boundary-restricted data (fe.m rows)"
    @assert size(G_in, 1) == fe.n "G must be lifted to the full space (fe.n rows)"
    @assert size(F_in, 2) == size(G_in, 2) "F and G must have the same number of modes"
    num_modes = size(G_in, 2)

    F = copy(F_in)
    G = copy(G_in)
    w = fe.up(ones(fe.m))

    ba = BlockLAssembler(fe)
    L = allocate_matrix(fe.dh)
    Lb = bordered(L, w)

    U = zeros(fe.n, num_modes)
    Λ = zeros(fe.n, num_modes)
    res = zeros(fe.m, num_modes)
    grad = zeros(fe.n_σ)
    J = zeros(fe.n_σ, num_modes)

    return FerriteBlockMode(ba, w, L, Lb, F, G, U, Λ, res, grad, J, num_modes)
end

# Solves AbX = RHS for all modes at once. `block=true` uses Krylov.jl's true
# block-MINRES (one shared block-Lanczos subspace, BLAS-3 mat-vecs); `block=false`
# loops IterativeSolvers.jl's single-RHS MINRES over the columns. Both give the
# same (up to tolerance) answer, block is generally faster for many modes since
# the columns share the same operator.
function solve_bordered!(X::AbstractMatrix, Lb::AbstractMatrix, RHS::AbstractMatrix, block::Bool)
    n = size(X, 1)
    if block
        Y, _ = Krylov.block_minres(Lb, RHS; atol=1e-10, rtol=1e-10)
        X .= view(Y, 1:n, :)
    else
        for c in 1:size(RHS, 2)
            x = zeros(size(Lb, 1))
            IterativeSolvers.minres!(x, Lb, view(RHS, :, c); abstol=1e-10, reltol=1e-10, maxiter=2 * size(Lb, 1), initially_zero=true)
            X[:, c] .= view(x, 1:n)
        end
    end
    return X
end

# Solves the mean-zero-gauged Neumann state equation for every mode at once
# (same L(σ), different right-hand sides).
function assemble_state!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false)
    assemble_L!(fbm.L, fbm.ba, fe, σ)
    fbm.Lb = bordered(fbm.L, fbm.w)
    RHS = vcat(fbm.G, zeros(1, fbm.num_modes))
    solve_bordered!(fbm.U, fbm.Lb, RHS, block)
    return fbm.U
end

"""
    residual!(fbm, fe, σ; block=false)

Solves the forward (state) problem for every current pattern with the given
conductivity `σ` and returns the boundary residual `U|∂Ω - F` (m × num_modes).
Must be called before `jacobian!` for the same `σ`.

`block=true` solves all modes in one shot with Krylov.jl's block-MINRES instead
of looping single-RHS MINRES calls; see `solve_bordered!`.
"""
function residual!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false)
    assemble_state!(fbm, fe, σ; block=block)
    for c in 1:fbm.num_modes
        fbm.res[:, c] .= fe.down(view(fbm.U, :, c)) .- view(fbm.F, :, c)
    end
    return fbm.res
end

"""
    jacobian!(fbm, fe, σ; block=false)

Solves the adjoint problem for every mode (reusing the state matrix and
solution assembled by the last `residual!` call) and returns the resulting
gradient of `‖U|∂Ω - F‖²` with respect to σ, stacked per mode
(n_σ × num_modes).

`block=true` solves all modes in one shot with Krylov.jl's block-MINRES instead
of looping single-RHS MINRES calls; see `solve_bordered!`.
"""
function jacobian!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false)
    RHS = zeros(fe.n, fbm.num_modes)
    for c in 1:fbm.num_modes
        RHS[:, c] .= fe.up(-2 .* view(fbm.res, :, c))
    end
    RHS = vcat(RHS, zeros(1, fbm.num_modes))
    solve_bordered!(fbm.Λ, fbm.Lb, RHS, block)
    for c in 1:fbm.num_modes
        assemble_gradient!(fbm.grad, fe, view(fbm.Λ, :, c), view(fbm.U, :, c))
        fbm.J[:, c] .= fbm.grad
    end
    return fbm.J
end

"""
    residual_and_jacobian!(fbm, fe, σ; block=false)

Convenience wrapper computing both the residual and the Jacobian for the
current conductivity `σ`.
"""
function residual_and_jacobian!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false)
    residual!(fbm, fe, σ; block=block)
    jacobian!(fbm, fe, σ; block=block)
    return fbm.res, fbm.J
end
