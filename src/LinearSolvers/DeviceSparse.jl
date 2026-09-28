# Backend-agnostic sparse matrices for the GPU code paths.
#
# The solvers only need sparse × dense products on the device. Vendor libraries provide them for
# their own matrix types (e.g. CuSparseMatrixCSR), but GPUArrays.jl has no generic sparse × dense
# kernel. `DeviceSparseMatrixCSR` stores a CSR matrix in arrays of any GPUArrays backend and
# implements `mul!` with a KernelAbstractions.jl kernel, so it runs on CUDA, AMDGPU, oneAPI, Metal
# (Float32 only) and on the CPU reference backend JLArrays.jl.

using LinearAlgebra
using SparseArrays
import KernelAbstractions as KA
# the kernel macros must be used unqualified: @kernel only rewrites a literal `@index` for its CPU path
using KernelAbstractions: @kernel, @index, @Const

"""
    DeviceSparseMatrixCSR(A::SparseMatrixCSC, ArrayType)

Compressed sparse row matrix whose index and value arrays are stored with `ArrayType`
(`CuArray`, `ROCArray`, `oneArray`, `MtlArray`, `JLArray`, or `Array`). Supports
`mul!(Y, A, X, α, β)` for dense vectors/matrices on the same backend through a generic
KernelAbstractions kernel. Usually created by [`device_converter`](@ref).
"""
struct DeviceSparseMatrixCSR{Tv, Ti, VV <: AbstractVector{Tv}, VI <: AbstractVector{Ti}} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rowptr::VI
    colval::VI
    nzval::VV
end

function DeviceSparseMatrixCSR(A::SparseMatrixCSC{Tv, Ti}, ArrayType) where {Tv, Ti}
    At = sparse(transpose(A))                  # CSC of Aᵀ = CSR of A (rows sorted within each row)
    return DeviceSparseMatrixCSR(size(A, 1), size(A, 2), ArrayType(At.colptr), ArrayType(At.rowval),
                                 ArrayType(At.nzval))
end

Base.size(A::DeviceSparseMatrixCSR) = (A.m, A.n)
SparseArrays.nnz(A::DeviceSparseMatrixCSR) = length(A.nzval)
Base.getindex(::DeviceSparseMatrixCSR, ::Int, ::Int) =
    error("scalar indexing of a DeviceSparseMatrixCSR is not supported")
Base.show(io::IO, ::MIME"text/plain", A::DeviceSparseMatrixCSR{Tv}) where {Tv} =
    print(io, "$(A.m)×$(A.n) DeviceSparseMatrixCSR{$Tv} with $(nnz(A)) stored entries on ",
          nameof(typeof(A.nzval)))

# Host copy (for setup work such as factorisations that have no device implementation)
function SparseArrays.SparseMatrixCSC(A::DeviceSparseMatrixCSR)
    At = SparseMatrixCSC(A.n, A.m, Array(A.rowptr), Array(A.colval), Array(A.nzval))  # = (Aᵀ) in CSC
    return sparse(transpose(At))
end

# one work item per entry of Y (row i, column j); linear indexing is supported by every backend
@kernel function _csr_spmm_kernel!(Y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(X), α, β)
    k = @index(Global, Linear)
    m = size(Y, 1)
    i = (k - 1) % m + 1
    j = (k - 1) ÷ m + 1
    acc = zero(eltype(Y))
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        acc += nzval[p] * X[colval[p], j]
    end
    @inbounds Y[i, j] = iszero(β) ? α * acc : α * acc + β * Y[i, j]
end

function _csr_mul!(Y, A::DeviceSparseMatrixCSR, X, α::Number, β::Number)
    size(A, 2) == size(X, 1) && size(A, 1) == size(Y, 1) && size(X, 2) == size(Y, 2) ||
        throw(DimensionMismatch("sizes $(size(A)), $(size(X)) and $(size(Y)) do not match"))
    Ym, Xm = _as_matrix(Y), _as_matrix(X)
    T = eltype(Y)
    backend = KA.get_backend(Ym)
    _csr_spmm_kernel!(backend)(Ym, A.rowptr, A.colval, A.nzval, Xm, T(α), T(β); ndrange = length(Ym))
    return Y
end
# the two signatures below are more specific than LinearAlgebra's generic 5-argument methods
# (avoids method ambiguities); 3-argument mul! falls back to them
LinearAlgebra.mul!(Y::AbstractMatrix, A::DeviceSparseMatrixCSR, X::AbstractMatrix, α::Number, β::Number) =
    _csr_mul!(Y, A, X, α, β)
LinearAlgebra.mul!(Y::AbstractVector, A::DeviceSparseMatrixCSR, X::AbstractVector, α::Number, β::Number) =
    _csr_mul!(Y, A, X, α, β)

Base.:*(A::DeviceSparseMatrixCSR, X::AbstractVector) = mul!(similar(X, size(A, 1)), A, X)
Base.:*(A::DeviceSparseMatrixCSR, X::AbstractMatrix) = mul!(similar(X, size(A, 1), size(X, 2)), A, X)

# new values with the same pattern: for the symmetric matrices of the solvers the CSR value order
# equals the CSC value order of the host copy
_set_values!(A::DeviceSparseMatrixCSR, nzval::Vector) = (copyto!(A.nzval, nzval); A)

"""
    to_device = device_converter(ArrayType)

Conversion function for the `to_device` keyword of the solvers and preconditioners on any
GPUArrays backend: sparse matrices become [`DeviceSparseMatrixCSR`](@ref), dense arrays
`ArrayType(x)`. Examples:

    device_converter(CuArray)      # NVIDIA (CUDA.jl)
    device_converter(ROCArray)     # AMD (AMDGPU.jl)
    device_converter(oneArray)     # Intel (oneAPI.jl)
    device_converter(MtlArray)     # Apple (Metal.jl, Float32 only)
    device_converter(JLArray)      # CPU reference implementation (JLArrays.jl), for testing

On NVIDIA GPUs, `x -> x isa SparseMatrixCSC ? CuSparseMatrixCSR(x) : CuArray(x)` uses cuSPARSE
instead of the generic kernel and enables the cuDSS direct solver.
"""
device_converter(ArrayType) = x -> _to_device(x, ArrayType)
_to_device(x::SparseMatrixCSC, ArrayType) = DeviceSparseMatrixCSR(x, ArrayType)
_to_device(x::AbstractArray, ArrayType) = ArrayType(x)

# ---------------------------------------------------------------------------------------
# Direct solver on devices without a device factorisation: factorise on the host
# ---------------------------------------------------------------------------------------

mutable struct _HostFactorization{F, T}
    fact::F
    hb::Matrix{T}         # host buffers for the right-hand side / solution
    hx::Matrix{T}
end

function _chol_factorize(A::DeviceSparseMatrixCSR{T}) where {T}
    Ah = SparseMatrixCSC(A)
    fact = T == Float64 ? _chol_factorize(Ah) : _chol_factorize(_UpperTriangle(triu(Ah)))
    return _HostFactorization(fact, zeros(T, 0, 0), zeros(T, 0, 0))
end

function _chol_refactor!(F::_HostFactorization, A::DeviceSparseMatrixCSR{T}) where {T}
    Ah = SparseMatrixCSC(A)
    F.fact = F.fact isa LDLFactorizations.LDLFactorization ?
             _chol_refactor!(F.fact, _UpperTriangle(triu(Ah))) : _chol_refactor!(F.fact, Ah)
    return F
end

function _chol_solve!(X, F::_HostFactorization{<:Any, T}, B) where {T}
    if size(F.hb) != size(B)
        F.hb = Matrix{T}(undef, size(B))
        F.hx = Matrix{T}(undef, size(B))
    end
    copyto!(F.hb, B)                                        # device → host
    _chol_solve!(F.hx, F.fact, F.hb)
    copyto!(X, F.hx)                                        # host → device
    return X
end
