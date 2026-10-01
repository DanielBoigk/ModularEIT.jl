# The interface between the generic EIT layer (forward model, objectives, solvers, optimization)
# and a finite element back end (ModularEITFerrite, ModularEITGridap). A back end defines a
# subtype of AbstractDiscretization and adds methods to the functions declared here; the generic
# layer only calls these functions and works with the data types defined here (FEMatrices,
# ConductivityTensor, the electrode models and ForwardModel).

using LinearAlgebra
using SparseArrays
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

# ---------------------------------------------------------------------------------------
# Back end contract
# ---------------------------------------------------------------------------------------

"""
    ndofs_u(disc)

Number of degrees of freedom of the potential (the free ones on non-conforming meshes).
Back end contract.
"""
function ndofs_u end

"""
    ndofs_σ(disc)

Number of degrees of freedom of the conductivity. Back end contract.
"""
function ndofs_σ end

"""
    interpolate_function(disc, f; field = :σ)

Coefficients of the interpolant of the function `f(x)` in the σ space (`field = :σ`) or the u
space (`field = :u`). Back end contract.
"""
function interpolate_function end

"""
    l2_project(disc, f; field = :σ, kwargs...)

Coefficients of the L² projection of the function `f(x)` onto the σ or u space. Back end
contract.
"""
function l2_project end

"""
    fe_inner(disc, a, b; field = :σ, kind = :L2, mats = nothing)

Inner product of two coefficient vectors as functions: `:L2`, `:H1semi` or `:H1`. Back end
contract.
"""
function fe_inner end

"""
    fe_norm(disc, a; kwargs...)

Norm `√fe_inner(disc, a, a; kwargs...)`. Back end contract.
"""
function fe_norm end

"""
    total_variation(disc, σ; ε = 0)
    total_variation!(g, disc, σ; ε = 0)

Total variation `∫ √(|∇σ|² + ε²)` of a conductivity (for piecewise constants: the facet jumps),
and its coefficient gradient (in place). Back end contract.
"""
function total_variation end
@doc (@doc total_variation) function total_variation! end

"""
    lumped_mass(disc)

Row sums of the mass matrix of the σ space (cell areas for piecewise constants), the diagonal
metric of proximal steps. Back end contract.
"""
function lumped_mass end

"""
    angular_electrodes(disc, L; coverage = 0.5, offset = 0.0, center = nothing, angles = nothing)

`L` electrodes (sets of boundary facets in the back end's representation) around `center`,
centred at the angles `offset + 2π(ℓ-1)/L` or at `angles`, covering the fraction `coverage` of
the boundary. Back end contract.
"""
function angular_electrodes end

"""
    electrode_length(disc, electrode)

Length (area in 3D) of an electrode. Back end contract.
"""
function electrode_length end

"""
    transfer_electrodes(src, electrodes, dst)

The electrodes of `src` as electrodes of the discretization `dst` of the same domain (e.g. a
finer mesh for simulated data). Back end contract.
"""
function transfer_electrodes end

"""
    structured_grid(disc)

The [`StructuredGrid`](@ref) of a discretization on a uniform rectangle grid (bilinear u
space), for the DCT preconditioner. Back end contract (optional).
"""
function structured_grid end

"""
    polar_structure(disc)

The [`PolarStructure`](@ref) of a discretization on a polar disk mesh, for the polar FFT
preconditioner. Back end contract (optional).
"""
function polar_structure end

# the PolarStructure of the reference (disk) mesh of a conformally mapped mesh
function _reference_structure end

# ---------------------------------------------------------------------------------------
# Matrices of a discretization
# ---------------------------------------------------------------------------------------

"""
    FEMatrices(disc)

Assembled matrices of a discretization: `M_u`, `K_u` (mass and stiffness of the u space),
`M_Γ` (boundary mass of the u space on the boundary), `M_σ`, `K_σ` (mass and stiffness of the σ
space; `K_σ = 0` for piecewise constants) and `M_σ_fac` (Cholesky factorisation of `M_σ`, used
for L² projections and L² gradients). The constructor for a discretization is part of the back
end contract.
"""
struct FEMatrices{MT <: SparseMatrixCSC{Float64, Int}, F}
    M_u::MT
    K_u::MT
    M_Γ::MT
    M_σ::MT
    K_σ::MT
    M_σ_fac::F
end

# ---------------------------------------------------------------------------------------
# Conductivity tensor
# ---------------------------------------------------------------------------------------
#
# The weighted stiffness matrix is linear in σ,
#
#     L(σ) = Σₐ σₐ Lₐ,    (Lₐ)ᵢⱼ = ∫ ψₐ ∇φᵢ⋅∇φⱼ dΩ,
#
# so with the fixed sparsity pattern of L its stored values are nzval(L(σ)) = T σ for one sparse
# matrix T of size nnz(L) × n_σ, built once per mesh by the back end. The same T gives every
# σ-derivative of a bilinear form in L: for vectors λ, u
#
#     ∂/∂σₐ (λᵀ L(σ) u) = λᵀ Lₐ u = Σₖ Tₖₐ λ[rowₖ] u[colₖ] = (Tᵀ w)ₐ,   wₖ = λ[rowₖ] u[colₖ],
#
# which is the adjoint-state gradient ∫ ψₐ ∇u⋅∇λ of the discrete functional
# (discretize-then-optimize), exact because L and the gradient use the same quadrature.
# Both products (T σ and Tᵀ w) are sparse matrix-vector products, and w is one
# embarrassingly parallel gather, so assembly and gradients run on any GPU backend.
#
# The L² gradient (optimize-then-discretize with an L² projection onto the σ space) is the Riesz
# representative M_σ⁻¹ Tᵀ w of the same dual vector; see L2Gradient.

"""
    ConductivityTensor(disc; pattern = <u-space pattern>, to_device = identity)

Sparse tensor `T` (`nnz(pattern) × n_σ`) with `nzval(L(σ)) = T σ` for the weighted stiffness
matrix `L(σ) = ∫ σ ∇φᵢ⋅∇φⱼ` in the storage order of `pattern`. `pattern` may be larger than the
u–u block (e.g. the augmented matrix of the complete electrode model), as long as the u dofs come
first; entries outside the u–u block get zero rows. The constructor for a discretization is part
of the back end contract.

Fields: `pattern` (host `SparseMatrixCSC`), `T` and `Tt` (`T` and `Tᵀ`, on the device if
`to_device` is given), `rows`, `cols` (row/column of each stored entry) and a buffer `w`.

See [`assemble_weighted_stiffness!`](@ref) and [`tensor_gradient!`](@ref).
"""
struct ConductivityTensor{MS <: SparseMatrixCSC{Float64, Int}, MT, MTt, VI, VW}
    pattern::MS
    T::MT
    Tt::MTt
    rows::VI
    cols::VI
    w::VW
end

# The tensor from its matrix T on `pattern`, with the row/column of every stored entry.
function ConductivityTensor(pattern::SparseMatrixCSC{Float64, Int}, T::SparseMatrixCSC; to_device = identity)
    rows = copy(pattern.rowval)
    cols = zeros(Int, nnz(pattern))
    for j in 1:size(pattern, 2), p in nzrange(pattern, j)
        cols[p] = j
    end
    if to_device === identity
        return ConductivityTensor(pattern, T, transpose(T), rows, cols, zeros(nnz(pattern)))
    end
    return ConductivityTensor(pattern, to_device(T), to_device(sparse(T')), to_device(rows),
                              to_device(cols), to_device(zeros(nnz(pattern))))
end

# position of the stored entry (i, j) in A.nzval
function _nz_index(A::SparseMatrixCSC, i::Integer, j::Integer)
    r = nzrange(A, j)
    k = searchsortedfirst(view(A.rowval, r), i)
    (k <= length(r) && A.rowval[r[k]] == i) || throw(ArgumentError("($i, $j) is not in the sparsity pattern"))
    return r[k]
end

"""
    assemble_weighted_stiffness!(L, ct::ConductivityTensor, σ; A₀ = nothing)

`nzval(L) ← T σ (+ A₀)`: the weighted stiffness matrix for the conductivity coefficients `σ`
by one sparse matrix-vector product. `L` must have the sparsity pattern `ct.pattern` (e.g.
`copy(ct.pattern)`). `A₀` is an optional constant part of the stored values (e.g. the contact
impedance terms of the complete electrode model).
"""
function assemble_weighted_stiffness!(L::SparseMatrixCSC, ct::ConductivityTensor, σ::AbstractVector;
                                      A₀ = nothing)
    weighted_stiffness_values!(nonzeros(L), ct, σ; A₀)
    return L
end

"""
    assemble_weighted_stiffness(disc, σ)

The weighted stiffness matrix `∫ σ ∇φᵢ⋅∇φⱼ` of a discretization, assembled directly (for
repeated assembly use a [`ConductivityTensor`](@ref)). Back end contract.
"""
function assemble_weighted_stiffness end

"""
    weighted_stiffness_values!(nzval, ct::ConductivityTensor, σ; A₀ = nothing)

Stored values `nzval ← T σ (+ A₀)` of the weighted stiffness matrix, on the device of `ct`.
"""
function weighted_stiffness_values!(nzval::AbstractVector, ct::ConductivityTensor, σ::AbstractVector;
                                    A₀ = nothing)
    if A₀ === nothing
        mul!(nzval, ct.T, σ)
    else
        copyto!(nzval, A₀)
        mul!(nzval, ct.T, σ, true, true)
    end
    return nzval
end

"""
    pair_products!(w, ct::ConductivityTensor, Λ, U)

`wₖ = Σₛ Λ[rowₖ, s] U[colₖ, s]` for every stored entry `k` of the pattern (`Λ`, `U`: vectors or
`n × s` blocks with `n ≥` the size of the pattern's u–u block). KernelAbstractions kernel: runs
multithreaded on the CPU and on every GPU backend.
"""
function pair_products!(w::AbstractVector, ct::ConductivityTensor, Λ::AbstractVecOrMat, U::AbstractVecOrMat)
    Λm, Um = _as_matrix(Λ), _as_matrix(U)
    size(Λm, 2) == size(Um, 2) || throw(DimensionMismatch("Λ and U need the same number of columns"))
    backend = KA.get_backend(w)
    _pair_products_kernel!(backend)(w, ct.rows, ct.cols, Λm, Um, size(Um, 2); ndrange = length(w))
    return w
end

@kernel function _pair_products_kernel!(w, @Const(rows), @Const(cols), @Const(Λ), @Const(U), s)
    k = @index(Global)
    r = rows[k]
    c = cols[k]
    acc = zero(eltype(w))
    @inbounds for j in 1:s
        acc += Λ[r, j] * U[c, j]
    end
    @inbounds w[k] = acc
end

"""
    tensor_gradient!(g, ct::ConductivityTensor, Λ, U; α = 1, β = 0)

`g ← α Σₛ ∂/∂σ (λₛᵀ L(σ) uₛ) + β g`, i.e. `gₐ = α Σₛ λₛᵀ Lₐ uₛ + β gₐ = α Σₛ ∫ ψₐ ∇uₛ⋅∇λₛ + β gₐ`
for the columns `λₛ`, `uₛ` of `Λ`, `U`. This is the conductivity gradient of the adjoint state
method (with `α = -1`) and of the Kohn–Vogelius functional (with `Λ = U`). Two parallel passes:
[`pair_products!`](@ref) and one sparse `Tᵀ w` product.
"""
function tensor_gradient!(g::AbstractVector, ct::ConductivityTensor, Λ::AbstractVecOrMat, U::AbstractVecOrMat;
                          α = true, β = false)
    pair_products!(ct.w, ct, Λ, U)
    mul!(g, ct.Tt, ct.w, α, β)
    return g
end

# ---------------------------------------------------------------------------------------
# Gradient representations (Riesz maps)
# ---------------------------------------------------------------------------------------

"""
    CoefficientGradient()

The gradient as the vector of partial derivatives `∂J/∂σₐ` (a dual vector). This is the exact
gradient of the discrete objective (discretize-then-optimize); it depends on the mesh.
"""
struct CoefficientGradient <: AbstractRieszMap end

"""
    L2Gradient(mats::FEMatrices)
    L2Gradient(M_σ_factorization)

The L² Riesz representative `M_σ⁻¹ ∂J/∂σ`: the L² projection of `∫ ψₐ ∇u⋅∇λ` onto the σ space
(optimize-then-discretize with projection). It approximates the continuous gradient
`-∇u⋅∇λ` independently of the mesh. For piecewise constant σ, `M_σ` is diagonal with the cell
areas.
"""
struct L2Gradient{F} <: AbstractRieszMap
    fac::F
end
L2Gradient(mats::FEMatrices) = L2Gradient(mats.M_σ_fac)

"""
    riesz_map!(g, R::AbstractRieszMap)
    riesz_map(R, g)

Apply the Riesz map `R` to the coefficient gradient `g` (in place / out of place).
"""
riesz_map!(g::AbstractVector, ::CoefficientGradient) = g
riesz_map!(g::AbstractVector, R::L2Gradient) = copyto!(g, R.fac \ g)
riesz_map(R::AbstractRieszMap, g::AbstractVector) = riesz_map!(copy(g), R)
