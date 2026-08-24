using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Krylov
using Ferrite

export FerriteBlockMode
export residual!, jacobian!, residual_and_jacobian!
export create_block_f∂f

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

"""
    FerriteBlockMode(σ, G, fe; block=false)

Constructs a `FerriteBlockMode` for synthetic-data generation: given a known
conductivity `σ` (fe.n_σ) and lifted Neumann current patterns `G` (n ×
num_modes), assembles `L(σ)` and solves the forward (state) problem for every
mode, then sets the target boundary data `F` to the resulting boundary trace
`down(U)`. The returned `fbm.L`/`fbm.F` are the assembled stiffness matrix and
synthetic boundary data; `residual!(fbm, fe, σ)` at the same `σ` will be
(numerically) zero, since `F` was generated from that exact solve.
"""
function FerriteBlockMode(σ::AbstractVector, G_in::AbstractMatrix, fe::FerriteFESpace; block::Bool=false)
    @assert length(σ) == fe.n_σ "σ must have fe.n_σ entries"
    @assert size(G_in, 1) == fe.n "G must be lifted to the full space (fe.n rows)"

    num_modes = size(G_in, 2)

    G = copy(G_in)
    w = fe.up(ones(fe.m))

    ba = BlockLAssembler(fe)
    L = allocate_matrix(fe.dh)
    assemble_L!(L, ba, fe, σ)
    Lb = bordered(L, w)

    U = zeros(fe.n, num_modes)
    Λ = zeros(fe.n, num_modes)
    RHS = vcat(G, zeros(1, num_modes))
    solve_bordered!(U, Lb, RHS, block)

    F = zeros(fe.m, num_modes)
    for c in 1:num_modes
        F[:, c] .= fe.down(view(U, :, c))
    end

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

function check_nmodes(fbm::FerriteBlockMode, nmodes::Int)
    @assert 1 <= nmodes <= fbm.num_modes "nmodes must be between 1 and fbm.num_modes ($(fbm.num_modes)), got $nmodes"
    return nmodes
end

# Solves the mean-zero-gauged Neumann state equation for the first `nmodes`
# modes at once (same L(σ), different right-hand sides).
function assemble_state!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false, nmodes::Int=fbm.num_modes)
    check_nmodes(fbm, nmodes)
    assemble_L!(fbm.L, fbm.ba, fe, σ)
    fbm.Lb = bordered(fbm.L, fbm.w)
    RHS = vcat(view(fbm.G, :, 1:nmodes), zeros(1, nmodes))
    solve_bordered!(view(fbm.U, :, 1:nmodes), fbm.Lb, RHS, block)
    return view(fbm.U, :, 1:nmodes)
end

"""
    residual!(fbm, fe, σ; block=false, nmodes=fbm.num_modes)

Solves the forward (state) problem for the first `nmodes` current patterns
(i.e. columns `1:nmodes` of `F`/`G`) with the given conductivity `σ` and
returns the boundary residual `U|∂Ω - F` (m × nmodes). Must be called before
`jacobian!` for the same `σ` and `nmodes`.

`block=true` solves all modes in one shot with Krylov.jl's block-MINRES instead
of looping single-RHS MINRES calls; see `solve_bordered!`.
"""
function residual!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false, nmodes::Int=fbm.num_modes)
    check_nmodes(fbm, nmodes)
    assemble_state!(fbm, fe, σ; block=block, nmodes=nmodes)
    for c in 1:nmodes
        fbm.res[:, c] .= fe.down(view(fbm.U, :, c)) .- view(fbm.F, :, c)
    end
    return view(fbm.res, :, 1:nmodes)
end

"""
    jacobian!(fbm, fe, σ; block=false, nmodes=fbm.num_modes)

Solves the adjoint problem for the first `nmodes` modes (reusing the state
matrix and solution assembled by the last `residual!` call, which must have
been called with the same `nmodes`) and returns the resulting gradient of
`‖U|∂Ω - F‖²` with respect to σ, stacked per mode (n_σ × nmodes).

`block=true` solves all modes in one shot with Krylov.jl's block-MINRES instead
of looping single-RHS MINRES calls; see `solve_bordered!`.
"""
function jacobian!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false, nmodes::Int=fbm.num_modes)
    check_nmodes(fbm, nmodes)
    RHS = zeros(fe.n, nmodes)
    for c in 1:nmodes
        RHS[:, c] .= fe.up(-2 .* view(fbm.res, :, c))
    end
    RHS = vcat(RHS, zeros(1, nmodes))
    solve_bordered!(view(fbm.Λ, :, 1:nmodes), fbm.Lb, RHS, block)
    for c in 1:nmodes
        assemble_gradient!(fbm.grad, fe, view(fbm.Λ, :, c), view(fbm.U, :, c))
        fbm.J[:, c] .= fbm.grad
    end
    return view(fbm.J, :, 1:nmodes)
end

"""
    residual_and_jacobian!(fbm, fe, σ; block=false, nmodes=fbm.num_modes)

Convenience wrapper computing both the residual and the Jacobian for the
current conductivity `σ`, using only the first `nmodes` modes (columns) of
`F`/`G`.
"""
function residual_and_jacobian!(fbm::FerriteBlockMode, fe::FerriteFESpace, σ::AbstractVector; block::Bool=false, nmodes::Int=fbm.num_modes)
    residual!(fbm, fe, σ; block=block, nmodes=nmodes)
    jacobian!(fbm, fe, σ; block=block, nmodes=nmodes)
    return view(fbm.res, :, 1:nmodes), view(fbm.J, :, 1:nmodes)
end

"""
    create_block_f∂f(fbm, fe; nmodes=fbm.num_modes, block=false, norm_const=1.0,
                      gn=false, λ=1e-3, R=nothing, ∇R=nothing, β=0.0)

Builds `(f, ∂f)` objective/gradient closures `σ -> f(σ)` / `σ -> ∂f(σ)` on top
of a `FerriteBlockMode`, in the spirit of `create_f∂f` in `wrapper.jl` but for
the block-MINRES solver. `f`/`∂f` are ordinary `σ -> scalar` / `σ -> vector`
closures, so they plug directly into `create_prox_linesearch`/
`create_proximal_gradient_step` (`wrapper.jl`) unchanged.

`f(σ)` is the *mean* squared boundary residual over the first `nmodes` modes:

    f(σ) = (1 / nmodes) * Σ_{c=1}^{nmodes} ‖res_c(σ)‖² / norm_const

Averaging (rather than summing) over modes keeps `f`/`∂f` on a comparable
scale regardless of how many modes are used in a given call — important if
`nmodes` varies between calls (e.g. a continuation scheme that starts with a
handful of modes and adds more as `σ` converges), since a fixed step size τ /
prox strength ρ / regularization weight β would otherwise need retuning every
time `nmodes` changes. `norm_const` is a separate, orthogonal normalization
(e.g. `fe.m`, for grid-size invariance) and defaults to `1.0`.

If `gn=true`, `∂f` instead returns a Levenberg–Marquardt-regularized
Gauss–Newton step, computed from the SVD of the stacked per-mode gradient
matrix `fbm.J[:, 1:nmodes]` (n_σ × nmodes) and per-mode squared errors — the
same mode-space GN trick as `gauss_newton_svd!`, adapted to `fbm.J`'s
transposed layout. Unlike the plain-gradient path, this system is built from
the *unnormalized* per-mode quantities (no `1/nmodes` averaging): `JᵀJ`
already grows with `nmodes` on its own, so a fixed `λ` becomes relatively
weaker as more modes are included and relatively stronger as fewer are —
i.e. less data automatically means more damping, with no extra bookkeeping.
Rescale `λ` yourself if you want a fixed damping *strength* independent of
`nmodes`.

`R`/`∇R` (e.g. from `create_tikhonov`) add `β*R(σ)` / `β*∇R(σ)` to `f`/`∂f`.

Caches results keyed on `σ` (and `nmodes`, fixed per closure) so repeated
`f`/`∂f` calls at the same point — e.g. from a line search — don't redo the
state/adjoint solves.
"""
function create_block_f∂f(fbm::FerriteBlockMode, fe::FerriteFESpace;
                           nmodes::Int=fbm.num_modes, block::Bool=false,
                           norm_const::Float64=1.0, gn::Bool=false, λ::Float64=1e-3,
                           R=nothing, ∇R=nothing, β::Float64=0.0)
    check_nmodes(fbm, nmodes)
    last_σ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_val = Ref(0.0)
    last_grad = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    grad_valid = Ref(false)

    f = σ -> begin
        σc = max.(σ, 1e-6)
        if last_σ[] !== nothing && σc == last_σ[]
            return last_val[]
        end
        residual!(fbm, fe, σc; block=block, nmodes=nmodes)
        val = sum(c -> sum(abs2, view(fbm.res, :, c)), 1:nmodes) / (nmodes * norm_const)
        if R !== nothing
            val += β * R(σc)
        end
        last_σ[] = σc
        last_val[] = val
        grad_valid[] = false
        return val
    end

    ∂f = σ -> begin
        σc = max.(σ, 1e-6)
        if last_σ[] === nothing || σc != last_σ[]
            f(σc)
        elseif grad_valid[]
            return copy(last_grad[])
        end
        jacobian!(fbm, fe, σc; block=block, nmodes=nmodes)

        if gn
            J_mat = fbm.J[:, 1:nmodes]
            r_vec = [sum(abs2, view(fbm.res, :, c)) for c in 1:nmodes]
            U, Σ, V = svd(J_mat)
            Σ_damped = Σ ./ (Σ .^ 2 .+ λ)
            grad = -U * (Σ_damped .* (V' * r_vec))
        else
            grad = vec(sum(view(fbm.J, :, 1:nmodes); dims=2)) ./ (nmodes * norm_const)
        end
        if ∇R !== nothing
            grad = grad .+ β * ∇R(σc)
        end
        last_grad[] = grad
        grad_valid[] = true
        return copy(grad)
    end

    return f, ∂f
end

