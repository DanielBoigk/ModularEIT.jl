using Ferrite
using LinearAlgebra

export FerriteFESpace
export dotL2, dotH1
export normL2sq, normL2, normH1, normH1sq
export normL1, normTV
export metricL2, metricL2sq, metricH1, metricH1sq

export normTV_diff,
       huber_norm_smooth_isotropic,
       huber_norm_smooth_anisotropic


"""
    struct FerriteFESpace{RefElem} <: AbstractHilbertSpace

Finite Element Space for a given reference element type `RefElem`.

# Fields
- `cellvalues::CellValues` : precomputed shape functions and quadrature for elements.
- `dh::DofHandler` : mapping between local and global degrees of freedom.
- `∂Ω` : boundary faces (indices) for Dirichlet conditions.
- `facetvalues::FacetValues` : shape functions for facets used in boundary integrals.
- `ch::ConstraintHandler` : handles Dirichlet (or other) constraints.
- `order::Int` : polynomial order of the FE basis.
- `qr_order::Int` : quadrature order.
- `dim::Int` : spatial dimension.
- `n::Int` : number of global degrees of freedom.
- `num_facet::Int` : number of facets.
- `m::Int` : number of boundary facets.
- `M::AbstractMatrix` : mass matrix.
- `M_fac` : factorization of mass matrix (optional).
- `K::AbstractMatrix` : stiffness matrix.
- `K_fac` : factorization of stiffness matrix (optional).
- `total_volume::Float64` : total volume of the domain.
- `down` : projection from force vector to boundary coefficients.
- `up` : projection from boundary coefficients to force vector.
- `up!` : in-place version of `up`.
"""
struct FerriteFESpace{RefElem} <: AbstractHilbertSpace
    cellvalues::CellValues
    dh::DofHandler
    ∂Ω
    facetvalues::FacetValues
    ch::ConstraintHandler
    order::Int
    qr_order::Int
    dim::Int
    n::Int
    num_facet::Int
    m::Int
    M::AbstractMatrix
    M_fac
    K::AbstractMatrix
    K_fac
    total_volume::Float64
    down # Projection from force vector to coefficients of the basis functions of boundary
    up # Projection from coefficients of the basis functions of boundary to force vector
    up! # Projection from coefficients of the basis functions of boundary to force vector
end

"""
    FerriteFESpace{RefElem}(grid, order, qr_order, ∂Ω)

Constructs a type-stable finite element space.

# Arguments
- `grid` : mesh/grid object
- `order::Int` : polynomial order of the FE basis
- `qr_order::Int` : quadrature order
- `∂Ω` : indices of boundary faces for Dirichlet conditions

Initializes cell/facet values, DOFs, constraints, mass/stiffness matrices,
and projection operators.
"""
function FerriteFESpace{RefElem}(grid, order::Int, qr_order::Int, ∂Ω) where {RefElem}
    dim = Ferrite.getspatialdim(grid)

    # reference element interpolation
    ip = Lagrange{RefElem,order}()

    # quadrature
    qr = QuadratureRule{RefElem}(qr_order)
    qr_face = FacetQuadratureRule{RefElem}(qr_order)

    # cell and facet values
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    # degrees of freedom
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    # constraints
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, ∂Ω, (x) -> 0.0)
    add!(ch, dbc)
    close!(ch)

    n = ndofs(dh)
    num_facet = length(∂Ω)
    M, M_fac = assemble_M(dh, cellvalues)
    K, K_fac = assemble_K(dh, cellvalues)
    total_volume = calc_total_volume(dh, cellvalues)

    m, _, down, up,up!, _ = produce_nonzero_positions(facetvalues, dh, ∂Ω)
    return FerriteFESpace{RefElem}(cellvalues, dh, ∂Ω, facetvalues, ch, order, qr_order, dim, n, num_facet, m, M, M_fac, K, K_fac, total_volume, down, up, up!)
end

"""
    dotH1(fe::FerriteFESpace, a, b)

H¹ inner product of FE coefficient vectors `a` and `b`.

`dotH1(a, b) = aᵀ K b`
"""
function dotH1(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return a' * fe.K * b
end

"""
    dotL2(fe::FerriteFESpace, a, b)

L² inner product of FE coefficient vectors `a` and `b`.

`dotL2(a, b) = aᵀ M b`
"""
function dotL2(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return a' * fe.M * b
end




function normH1sq(fe::FerriteFESpace, a::AbstractVector)
    return dotH1(fe, a, a)
end
function normL2sq(fe::FerriteFESpace, a::AbstractVector)
    return dotL2(fe, a, a)
end

"""
    normL2(fe, a)

L² norm of FE coefficient vector `a`.
"""
function normL2(fe::FerriteFESpace, a::AbstractVector)
    return sqrt(normL2sq(fe, a))
end

"""
    normH1(fe, a)

H¹ norm of FE coefficient vector `a`.
"""
function normH1(fe::FerriteFESpace, a::AbstractVector)
    return sqrt(normH1sq(fe, a))
end
function metricL2sq(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    return normL2sq(fe, a - b)
end

"""
    metricL2(fe, a, b)

L² distance between FE coefficient vectors `a` and `b`.
"""
metricL2(fe, a, b) = normL2(fe, a - b)

"""
    metricH1(fe, a, b)

H¹ distance between FE coefficient vectors `a` and `b`.
"""
metricH1(fe, a, b) = normH1(fe, a - b)

metricH1sq(fe, a, b) = normH1sq(fe, a - b)



function normL1(a::AbstractVector, cellvalues::CellValues, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            uh_q = 0.0
            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                uh_q += ue[i] * ϕᵢ
            end

            total_residual += abs(uh_q) * dΩ
        end
    end
    return total_residual
end

"""
    normL1(fe, a)

L¹ norm (integral of absolute value) of FE function represented by coefficient vector `a`.
Uses quadrature over the finite elements.
"""
function normL1(fe::FerriteFESpace, a::AbstractVector)
    normL1(a, fe.cellvalues, fe.dh)
end

function huber_norm_smooth_isotropic(grad::AbstractVector, δ::Real)
    gnorm = norm(grad)
    return (sqrt(gnorm^2 + δ^2) - δ)
end
function huber_norm_smooth_anisotropic(grad::AbstractVector, δ::Real)
    return (sqrt(g^2 + δ^2) - δ)
end

function huber_smooth_val_grad_hess(g::AbstractVector, δ::Real)
    t = norm(g)
    s = sqrt(t^2 + δ^2)
    φ = s - δ
    φ′ = t / s
    φ″ = δ^2 / s^3
    grad = φ′ * g / t
    H = φ″ * (g * g') / t^2 + φ′ * (I - (g * g') / t^2) / t
    return φ, grad, H
end

function huber_smooth_val_grad_hess_aniso(g::AbstractVector, δ::Real)
    n = length(g)
    φ = zero(eltype(g))
    grad = similar(g)
    H = Diagonal(zeros(eltype(g), n))

    for i in 1:n
        gi = g[i]
        s = sqrt(gi^2 + δ^2)
        φ += s - δ
        grad[i] = gi / s
        H[i,i] = δ^2 / s^3
    end

    return φ, grad, H
end
#= Care later
using StaticArrays

"""
    huber_smooth_val_grad_hess_aniso_svec(g::SVector{N,T}, δ::T) where {N,T<:Real}

Compute Huber-smoothed anisotropic TV, its gradient, and Hessian
for a small fixed-size gradient vector `g` (as SVector).

Returns `(φ, grad, H)`.

- φ: scalar penalty
- grad: SVector of partial derivatives
- H: diagonal SMatrix (Hessian)
"""
function huber_smooth_val_grad_hess_aniso_svec(g::SVector{N,T}, δ::T) where {N,T<:Real}
    φ = zero(T)
    grad = @SVector zeros(T, N)
    H = @SMatrix zeros(T, N, N)

    @inbounds for i in 1:N
        gi = g[i]
        s = sqrt(gi^2 + δ^2)
        φ += s - δ
        grad[i] = gi / s
        H[i,i] = δ^2 / s^3
    end

    return φ, grad, H
end
=#


"""
    normTV(a, cellvalues, dh, ndims, iso=2)

Compute the total variation (TV) seminorm of the field represented by coefficients `a`.

At each quadrature point, this computes
```math
‖∇u(x_q)‖_{iso}
````

and integrates over the domain.

# Arguments

* `a`: Coefficient vector of the FEM solution.
* `cellvalues`: Preallocated `CellValues` for quadrature and shape gradients.
* `dh`: `DofHandler` for iterating over elements.
* `ndims`: Number of spatial dimensions.
* `iso`: Norm degree.

  * `iso = 2` → isotropic TV
  * `iso = 1` → anisotropic TV

# Returns

Scalar value of the total variation seminorm.
"""
function normTV(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64, iso=2)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = zeros(eltype(a), ndims)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                ∇uh_q .+= ue[i] * ∇ϕᵢ
            end
            total_residual += norm(∇uh_q,iso) * dΩ

        end
    end

    return total_residual
end


"""
    normTV_diff(a, cellvalues, dh, ndims; ε=1e-6, huber=huber_norm_smooth_isotropic)

Compute a *Huber-smoothed* total variation regularizer, differentiable w.r.t. `a`.

This replaces the nondifferentiable TV norm with a smooth approximation:
```math
‖∇u‖_ε ≈
  \begin{cases}
    \tfrac{1}{2ε}‖∇u‖^2, & ‖∇u‖ ≤ ε,\\
    ‖∇u‖ - \tfrac{ε}{2}, & ‖∇u‖ > ε.
  \end{cases}
````

# Arguments

* `a`: Coefficient vector of the FEM solution.
* `ε`: Huber smoothing threshold.
* `huber`: Function defining the smooth norm variant
  (`huber_norm_smooth_isotropic` or `huber_norm_smooth_anisotropic`).

# Returns

Scalar smoothed TV value suitable for differentiable regularization.

# Differentiation
This function is fully differentiable.
Gradients or Hessians with respect to `a` can be obtained using **Enzyme.jl**:
"""
function normTV_diff(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64; ε::Float64=1e-6, huber=huber_norm_smooth_isotropic)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = zeros(eltype(a), ndims)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                ∇uh_q .+= ue[i] * ∇ϕᵢ
            end
            total_residual += huber(∇uh_q, ε)* dΩ
        end
    end
    return total_residual
end


"""
    normL1grad(fe, a)

L¹ norm of the gradient of the FE function represented by `a`.
Computes ∫ |∇u_h| dΩ over all elements using quadrature.
"""
function normTV(fe::FerriteFESpace, a::AbstractVector)
    normTV(a, fe.cellvalues, fe.dh, fe.dim)
end
function normTV_diff(fe::FerriteFESpace, a::AbstractVector)
    normTV_diff(a, fe.cellvalues, fe.dh, fe.dim)
end

function calc_total_volume(dh::DofHandler, cellvalues::CellValues)
    total_volume = 0.0
    qpoints = getnquadpoints(cellvalues)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        for q in 1:qpoints
            total_volume += getdetJdV(cellvalues, q)
        end
    end
    return total_volume
end
