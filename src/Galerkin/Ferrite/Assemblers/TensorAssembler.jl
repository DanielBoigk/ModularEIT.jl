# Conductivity tensor: the weighted stiffness matrix is linear in σ,
#
#     L(σ) = Σₐ σₐ Lₐ,    (Lₐ)ᵢⱼ = ∫ ψₐ ∇φᵢ⋅∇φⱼ dΩ,
#
# so with the fixed sparsity pattern of L its stored values are nzval(L(σ)) = T σ for one sparse
# matrix T of size nnz(L) × n_σ, built once per mesh. The same T gives every σ-derivative of a
# bilinear form in L: for vectors λ, u
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
first; entries outside the u–u block get zero rows. On non-conforming grids the tensor is
condensed with the conformity constraints (`Cᵀ L(σ) C`).

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

function ConductivityTensor(disc::FerriteDiscretization; pattern = _u_pattern(disc),
                            to_device = identity)
    T = _conductivity_tensor(disc, pattern)
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

# COO assembly of T on the unconstrained pattern of dh_u: for every cell the local tensor
# ∫ ψₐ ∇φᵢ⋅∇φⱼ, scattered to (nz index of (i, j), σ dof a); duplicates are summed by `sparse`.
# Then T = R T_full with the sparse map R from the stored entries (i, j) of the full pattern to
# the entries (p, q) of `pattern`, weighted by the conformity constraints C[i, p] C[j, q]
# (R is a 0/1 selection on conforming grids).
function _conductivity_tensor(disc::FerriteDiscretization, pattern::SparseMatrixCSC)
    full = allocate_matrix(disc.dh_u)
    T_full = _conductivity_tensor_full(disc, full)
    Ct = disc.C_u === nothing ? nothing : sparse(disc.C_u')     # column i of Cᵀ = row i of C
    Ir, Jr, Vr = Int[], Int[], Float64[]
    for j in 1:size(full, 2), k in nzrange(full, j)
        i = full.rowval[k]
        if Ct === nothing
            push!(Ir, _nz_index(pattern, i, j)); push!(Jr, k); push!(Vr, 1.0)
        else
            for a in nzrange(Ct, i), b in nzrange(Ct, j)
                push!(Ir, _nz_index(pattern, Ct.rowval[a], Ct.rowval[b]))
                push!(Jr, k)
                push!(Vr, Ct.nzval[a] * Ct.nzval[b])
            end
        end
    end
    R = sparse(Ir, Jr, Vr, nnz(pattern), nnz(full))
    return R * T_full
end

function _conductivity_tensor_full(disc::FerriteDiscretization, pattern::SparseMatrixCSC)
    cv_u, cv_σ = disc.cv_u, disc.cv_σ
    nu, nσ = getnbasefunctions(cv_u), getnbasefunctions(cv_σ)
    nq = getnquadpoints(cv_u)
    Ae = zeros(nu, nu, nσ)
    σdofs = zeros(Int, nσ)
    nzidx = zeros(Int, nu, nu)
    ncell = getncells(disc.grid)
    Is = Vector{Int}(undef, ncell * nu * nu * nσ)
    Js = similar(Is)
    Vs = Vector{Float64}(undef, length(Is))
    p = 0
    for cell in CellIterator(disc.dh_u)
        reinit!(cv_u, cell)
        reinit!(cv_σ, cell)
        udofs = celldofs(cell)
        celldofs!(σdofs, disc.dh_σ, cellid(cell))
        fill!(Ae, 0)
        for q in 1:nq
            dΩ = getdetJdV(cv_u, q)
            for a in 1:nσ
                ψ = shape_value(cv_σ, q, a) * dΩ
                iszero(ψ) && continue
                for j in 1:nu
                    ∇φⱼ = shape_gradient(cv_u, q, j)
                    for i in 1:nu
                        Ae[i, j, a] += ψ * (shape_gradient(cv_u, q, i) ⋅ ∇φⱼ)
                    end
                end
            end
        end
        for j in 1:nu, i in 1:nu
            nzidx[i, j] = _nz_index(pattern, udofs[i], udofs[j])
        end
        for a in 1:nσ, j in 1:nu, i in 1:nu
            p += 1
            Is[p], Js[p], Vs[p] = nzidx[i, j], σdofs[a], Ae[i, j, a]
        end
    end
    return sparse(Is, Js, Vs, nnz(pattern), ndofs(disc.dh_σ))
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
