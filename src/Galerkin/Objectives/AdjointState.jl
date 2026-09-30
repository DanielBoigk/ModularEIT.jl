# Least-squares data misfit with gradients and Jacobians from the adjoint state method.
#
# Neumann (current-driven) mode, data = measured voltages:
#     A(σ) xₛ = P Iₛ,   eₛ = Π (Q xₛ - Vₛ),   J = ½ Σₛ ‖U eₛ‖²
#     (Π removes the weighted mean with the forward model's measurement weights: voltages are
#     only defined up to a constant)
#     adjoint:  A λₛ = Qᵀ Πᵀ Uᵀ rₛ,   ∂J/∂σₐ = -Σₛ λₛᵀ (∂A/∂σₐ) xₛ
#
# Dirichlet (voltage-driven) mode, data = measured currents:
#     xₛ = Dirichlet solution with x_B = E Vₛ,   eₛ = C⁻¹ Eᵀ (A xₛ)_B - Iₛ
#     For the Schur complement S (discrete DtN map), d(vᵀ S f) = x̃ᵥᵀ (∂A) x_f, where x̃ᵥ is the
#     Dirichlet solution with boundary data v. So the adjoint is another Dirichlet solve:
#     λₛ = Dirichlet solution with x_B = E C⁻¹ Uᵀ rₛ,   ∂J/∂σₐ = +Σₛ λₛᵀ (∂A/∂σₐ) xₛ
#
# Jacobian rows (one per measurement m and pattern s) use the same identity with rₛ replaced by
# unit vectors: Z = A⁺ Qᵀ Πᵀ Uᵀ (Neumann) or Z = Dirichlet solutions for E C⁻¹ Uᵀ, one block
# solve with n_obs right-hand sides, then J[(s, m), a] = ∓ zₘᵀ (∂A/∂σₐ) xₛ via the
# ConductivityTensor. The misfit metric enters only through U, so other metrics plug in by
# new AbstractMisfit types.

"""
    AdjointStateObjective(fm, inputs, data; mode = :neumann, solver = DirectSolver(),
                          misfit = SquaredEuclidean(), gradient = CoefficientGradient())

Least-squares misfit `J(σ) = ½ Σₛ ‖U eₛ(σ)‖²` between predicted and measured data for the
forward model `fm`, with the gradient from one adjoint solve and the Jacobian on request.

- `mode = :neumann`: `inputs` are current patterns (`n_inject × s`), `data` the measured voltages
  (`n_measure × s`). Voltages are compared after removing their mean (the ground is arbitrary),
  weighted by `fm.measure_weights` (boundary lengths for the continuum model).
- `mode = :dirichlet`: `inputs` are voltage patterns (`n_control × s`), `data` the measured
  currents (`n_inject × s`, in the representation of `fm.P`).

`solver` is any [`AbstractLinearSolver`](@ref); `misfit` any [`AbstractMisfit`](@ref);
`gradient` the [`AbstractRieszMap`](@ref) applied to the coefficient gradient. All buffers are
allocated here; evaluations only allocate inside the linear solvers.

Interface: [`objective_value`](@ref), [`value_and_gradient!`](@ref), [`residual!`](@ref),
[`residual_and_jacobian!`](@ref), [`n_residual`](@ref).
"""
mutable struct AdjointStateObjective{FM <: ForwardModel, S <: AbstractLinearSolver, MF <: AbstractMisfit,
                                     RM <: AbstractRieszMap} <: AbstractObjective
    fm::FM
    mode::Symbol
    inputs::Matrix{Float64}
    data::Matrix{Float64}
    solver::S
    state::Any                     # instantiated linear solver (created at the first evaluation)
    misfit::MF
    riesz::RM
    X::Matrix{Float64}             # states                 n × s
    Λ::Matrix{Float64}             # adjoint states         n × s
    B::Matrix{Float64}             # right-hand sides       n × s
    AX::Matrix{Float64}            # A X (Dirichlet mode)   n × s
    Xf::Matrix{Float64}            # free-dof blocks        nf × s
    Rf::Matrix{Float64}
    E::Matrix{Float64}             # error                  n_obs × s
    R::Matrix{Float64}             # whitened residual      k × s (k = n_obs unless projected)
    G::Matrix{Float64}             # adjoint weights        n_obs × s / n_ctrl × s
    jac::Any                       # Jacobian buffers (created on first use)
    version::Int                   # incremented at every new σ (validity of Jacobian operators)
end

function AdjointStateObjective(fm::ForwardModel, inputs::AbstractVecOrMat, data::AbstractVecOrMat;
                               mode::Symbol = :neumann, solver::AbstractLinearSolver = DirectSolver(),
                               misfit::AbstractMisfit = SquaredEuclidean(),
                               gradient::AbstractRieszMap = CoefficientGradient())
    mode in (:neumann, :dirichlet) || throw(ArgumentError("mode must be :neumann or :dirichlet, got :$mode"))
    In, D = Matrix{Float64}(_as_matrix(inputs)), Matrix{Float64}(_as_matrix(data))
    s = size(In, 2)
    size(D, 2) == s || throw(DimensionMismatch("inputs and data need the same number of patterns"))
    n_in, n_obs = mode === :neumann ? (n_inject(fm), n_measure(fm)) : (n_control(fm), n_inject(fm))
    size(In, 1) == n_in || throw(DimensionMismatch("inputs need $n_in rows in $mode mode"))
    size(D, 1) == n_obs || throw(DimensionMismatch("data need $n_obs rows in $mode mode"))
    n, nf = fm.n, length(fm.free_dofs)
    z(r, c) = zeros(r, c)
    return AdjointStateObjective(fm, mode, In, D, solver, nothing, misfit, gradient,
                                 z(n, s), z(n, s), z(n, s), z(n, s), z(nf, s), z(nf, s),
                                 z(n_obs, s), z(_residual_rows(misfit, n_obs), s), z(n_obs, s), nothing, 0)
end

"""
    n_residual(obj)

Length of the residual vector (`n_obs × s`, stacked pattern by pattern).
"""
n_residual(obj::AdjointStateObjective) = length(obj.R)

# assemble A(σ) and (re)factorise / update the solver
function _update!(obj::AdjointStateObjective, σ)
    fm = obj.fm
    obj.version += 1
    system_matrix!(fm, σ)
    if obj.mode === :neumann
        obj.state === nothing ? (obj.state = _init_neumann_solver(obj.solver, fm)) : _update_solver!(obj.state, fm.A)
    else
        obj.state === nothing ? (obj.state = _init_dirichlet_solver(obj.solver, fm)) : _update_solver!(obj.state, fm.A_ff)
    end
    return obj
end

# states, error and whitened residual; returns J
function _forward!(obj::AdjointStateObjective, σ)
    _update!(obj, σ)
    fm = obj.fm
    if obj.mode === :neumann
        mul!(obj.B, fm.P, obj.inputs)
        _solve!(obj.X, obj.state, obj.B)
        mul!(obj.E, fm.Q, obj.X)
        obj.E .-= obj.data
        _project!(obj.E, fm.measure_weights)
    else
        _dirichlet_solve!(obj.X, obj.state, fm, obj.inputs, obj.AX, obj.Xf, obj.Rf)
        _dirichlet_currents!(obj.E, fm, obj.X, obj.AX)
        obj.E .-= obj.data
    end
    _whiten!(obj.R, obj.misfit, obj.E)
    return sum(abs2, obj.R) / 2
end

"""
    objective_value(obj, σ)

Value of the objective at the conductivity coefficients `σ` (forward solves only).
"""
objective_value(obj::AdjointStateObjective, σ::AbstractVector) = _forward!(obj, σ)

"""
    value_and_gradient!(g, obj, σ)

Value of the objective at `σ`; writes the gradient (in the representation of the objective's
`gradient` Riesz map) into `g`.
"""
function value_and_gradient!(g::AbstractVector, obj::AdjointStateObjective, σ::AbstractVector)
    J = _forward!(obj, σ)
    fm = obj.fm
    _whiten_adjoint!(obj.G, obj.misfit, obj.R)          # Uᵀ r
    if obj.mode === :neumann
        _project_adjoint!(obj.G, fm.measure_weights)     # Πᵀ Uᵀ r
        mul!(obj.B, fm.Q', obj.G)                        # Qᵀ Πᵀ Uᵀ r
        _solve!(obj.Λ, obj.state, obj.B)
        tensor_gradient!(g, fm.tensor, obj.Λ, obj.X; α = -1)
    else
        fm.C_fac === nothing || copyto!(obj.G, fm.C_fac \ obj.G)   # C⁻¹ Uᵀ r
        _dirichlet_solve!(obj.Λ, obj.state, fm, obj.G, obj.AX, obj.Xf, obj.Rf)
        tensor_gradient!(g, fm.tensor, obj.Λ, obj.X; α = 1)
    end
    riesz_map!(g, obj.riesz)
    return J
end

"""
    residual!(r, obj, σ)

Whitened residual `r = vec(U e)` at `σ` (so that `J = ½ ‖r‖²`); returns `r`.
"""
function residual!(r::AbstractVector, obj::AdjointStateObjective, σ::AbstractVector)
    _forward!(obj, σ)
    copyto!(r, obj.R)
    return r
end

"""
    residual_and_jacobian!(r, J, obj, σ)

Whitened residual `r` and its Jacobian `J = ∂r/∂σ` (`n_residual(obj) × n_σ`, coefficient
representation) at `σ`. Costs one block solve with `k` right-hand sides plus `k × s` sparse
products with the conductivity tensor, `k` the residual entries per pattern (`n_obs`, or the
rows of a [`ProjectedMisfit`](@ref)).
"""
function residual_and_jacobian!(r::AbstractVector, Jm::AbstractMatrix, obj::AdjointStateObjective,
                                σ::AbstractVector)
    nr, s = size(obj.R)
    size(Jm) == (nr * s, obj.fm.n_σ) || throw(DimensionMismatch("J must be $(nr * s) × $(obj.fm.n_σ)"))
    residual!(r, obj, σ)
    _jacobian_blocks_at_state!(obj) do rows, B, _
        view(Jm, rows, :) .= B
    end
    return r, Jm
end

# Row blocks of the Jacobian at σ: calls f(rows, B, r_rows) with B = J[rows, :] (at most 64
# rows, reused buffer) and the residual rows. The full Jacobian is never stored, so column norms
# or JᵀJ can be accumulated for problems whose Jacobian does not fit into memory.
function _jacobian_blocks!(f, obj::AdjointStateObjective, σ::AbstractVector)
    _forward!(obj, σ)
    _jacobian_blocks_at_state!(f, obj)
    return obj
end

# (the forward solve at σ is done)
function _jacobian_blocks_at_state!(f, obj::AdjointStateObjective)
    fm = obj.fm
    n_obs = size(obj.E, 1)
    nr, s = size(obj.R)                                   # residual rows per pattern
    jb = _jacobian_buffers!(obj)
    Ut = _whitening_adjoint_matrix(obj.misfit, n_obs)
    if obj.mode === :neumann
        _project_adjoint!(Ut, fm.measure_weights)          # Πᵀ Uᵀ
        mul!(jb.B, fm.Q', Ut)
        _solve!(jb.Z, obj.state, jb.B)
        sgn = -1.0
    else
        fm.C_fac === nothing || (Ut = fm.C_fac \ Ut)       # C⁻¹ Uᵀ
        _dirichlet_solve!(jb.Z, obj.state, fm, Ut, jb.AX, jb.Xf, jb.Rf)
        sgn = 1.0
    end
    ct = fm.tensor
    chunk = size(jb.W, 2)
    Bbuf = zeros(chunk, fm.n_σ)
    r = vec(obj.R)
    for k in 1:s, m0 in 1:chunk:nr
        cols = m0:min(m0 + chunk - 1, nr)
        W = view(jb.W, :, 1:length(cols))
        _outer_pair_products!(W, ct, view(jb.Z, :, cols), view(obj.X, :, k))
        Gv = view(jb.Gσ, :, 1:length(cols))
        mul!(Gv, ct.Tt, W)
        B = view(Bbuf, 1:length(cols), :)
        B .= sgn .* Gv'
        rows = (k - 1) * nr .+ cols
        f(rows, B, view(r, rows))
    end
    return obj
end

"""
    jacobian_operator(obj, σ)

The Jacobian `J = ∂r/∂σ` of the residual at `σ` as a matrix-free operator: `J * v` (one
linearized forward solve per pattern) and `J' * w` (one adjoint solve per pattern), with
`mul!` for both, reusing the factorization of the forward solves. Neumann (current-driven)
mode. The operator re-linearizes by itself if the objective has been evaluated at another `σ`
in the meantime. For Krylov methods, Gauss–Newton with `linear_solver = :cg`, and problems whose
Jacobian does not fit into memory.
"""
function jacobian_operator(obj::AdjointStateObjective, σ::AbstractVector)
    obj.mode === :neumann || throw(ArgumentError("matrix-free Jacobians are implemented for mode = :neumann"))
    _forward!(obj, σ)
    fm = obj.fm
    n_obs, s = size(obj.E)
    nr = size(obj.R, 1)
    Lv = copy(fm.tensor.pattern)
    return JacobianOperator(obj, Vector{Float64}(σ), obj.version, nr * s, fm.n_σ, Lv,
                            zeros(fm.n, s), zeros(fm.n, s), zeros(n_obs, s), zeros(nr, s), zeros(fm.n_σ))
end

"""
    JacobianOperator

Matrix-free Jacobian of an [`AdjointStateObjective`](@ref); see [`jacobian_operator`](@ref).
"""
mutable struct JacobianOperator{O}
    obj::O
    σ::Vector{Float64}
    version::Int
    m::Int
    n::Int
    Lv::SparseMatrixCSC{Float64, Int}
    Y::Matrix{Float64}
    Z::Matrix{Float64}
    E::Matrix{Float64}
    R::Matrix{Float64}
    g::Vector{Float64}
end

struct AdjointJacobianOperator{J}
    J::J
end

Base.size(J::JacobianOperator) = (J.m, J.n)
Base.size(J::JacobianOperator, d::Integer) = d == 1 ? J.m : d == 2 ? J.n : 1
Base.eltype(::JacobianOperator) = Float64
Base.adjoint(J::JacobianOperator) = AdjointJacobianOperator(J)
Base.size(A::AdjointJacobianOperator) = reverse(size(A.J))
Base.size(A::AdjointJacobianOperator, d::Integer) = d == 1 ? size(A.J, 2) : d == 2 ? size(A.J, 1) : 1
Base.eltype(::AdjointJacobianOperator) = Float64
Base.adjoint(A::AdjointJacobianOperator) = A.J
Base.:*(J::Union{JacobianOperator, AdjointJacobianOperator}, v::AbstractVector) = mul!(zeros(size(J, 1)), J, v)

# the objective's state (factorization, states X) must be the one at J.σ
function _ensure_state!(J::JacobianOperator)
    J.obj.version == J.version && return J
    _forward!(J.obj, J.σ)
    J.version = J.obj.version
    return J
end

# J v = U Π Q δX,   A δX = -L(v) X
function LinearAlgebra.mul!(y::AbstractVector, J::JacobianOperator, v::AbstractVector)
    _ensure_state!(J)
    obj, fm = J.obj, J.obj.fm
    weighted_stiffness_values!(nonzeros(J.Lv), fm.tensor, v)
    mul!(J.Y, J.Lv, obj.X)
    J.Y .*= -1
    _solve!(J.Z, obj.state, J.Y)
    mul!(J.E, fm.Q, J.Z)
    _project!(J.E, fm.measure_weights)
    _whiten!(J.R, obj.misfit, J.E)
    copyto!(y, J.R)
    return y
end

# Jᵀ w = -Σₛ ∫ ∇λₛ⋅∇xₛ ψ,   A λ = Qᵀ Πᵀ Uᵀ w   (the adjoint gradient with r replaced by w)
function LinearAlgebra.mul!(g::AbstractVector, A::AdjointJacobianOperator{<:JacobianOperator}, w::AbstractVector)
    J = A.J
    _ensure_state!(J)
    obj, fm = J.obj, J.obj.fm
    copyto!(J.R, w)
    _whiten_adjoint!(J.E, obj.misfit, J.R)
    _project_adjoint!(J.E, fm.measure_weights)
    mul!(J.Y, fm.Q', J.E)
    _solve!(J.Z, obj.state, J.Y)
    tensor_gradient!(g, fm.tensor, J.Z, obj.X; α = -1)
    return g
end

function _jacobian_buffers!(obj::AdjointStateObjective)
    obj.jac === nothing || return obj.jac
    fm = obj.fm
    nr = size(obj.R, 1)
    chunk = min(nr, 64)
    nf = length(fm.free_dofs)
    obj.jac = (Z = zeros(fm.n, nr), B = zeros(fm.n, nr), AX = zeros(fm.n, nr),
               Xf = zeros(nf, nr), Rf = zeros(nf, nr), W = zeros(nnz(fm.tensor.pattern), chunk),
               Gσ = zeros(fm.n_σ, chunk))
    return obj.jac
end

# W[k, j] = Z[rowₖ, j] x[colₖ] for every stored entry k: the pair products of every column of Z
# with one state x, so that Tᵀ W holds the σ-derivatives zⱼᵀ (∂A/∂σ) x as columns
function _outer_pair_products!(W, ct::ConductivityTensor, Z, x)
    backend = KA.get_backend(W)
    _outer_pair_products_kernel!(backend)(W, ct.rows, ct.cols, Z, x; ndrange = size(W))
    return W
end

@kernel function _outer_pair_products_kernel!(W, @Const(rows), @Const(cols), @Const(Z), @Const(x))
    k, j = @index(Global, NTuple)
    @inbounds W[k, j] = Z[rows[k], j] * x[cols[k]]
end

_gradient_representation(obj::AdjointStateObjective) = obj.riesz
