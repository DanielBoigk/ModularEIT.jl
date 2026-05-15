
export dotL2, dotH1
export normL2sq, normL2, normH1, normH1sq
export normL1, normTV
export metricL2, metricL2sq, metricH1, metricH1sq

export normTV_diff,
    huber_norm_smooth_isotropic,
    huber_norm_smooth_anisotropic


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
    return dot(a, fe.M * b)
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
    return (sqrt(grad^2 + δ^2) - δ)
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
        H[i, i] = δ^2 / s^3
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




include("TV.jl")