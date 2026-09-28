# Least-squares data misfit with gradients and Jacobians from the adjoint state method.
#
# Neumann (current-driven) mode, data = measured voltages:
#     A(σ) xₛ = P Iₛ,   eₛ = Π (Q xₛ - Vₛ),   J = ½ Σₛ ‖U eₛ‖²
#     (Π removes the mean: voltages are only defined up to a constant)
#     adjoint:  A λₛ = Qᵀ Π Uᵀ rₛ,   ∂J/∂σₐ = -Σₛ λₛᵀ (∂A/∂σₐ) xₛ
#
# Dirichlet (voltage-driven) mode, data = measured currents:
#     xₛ = Dirichlet solution with x_B = E Vₛ,   eₛ = C⁻¹ Eᵀ (A xₛ)_B - Iₛ
#     For the Schur complement S (discrete DtN map), d(vᵀ S f) = x̃ᵥᵀ (∂A) x_f, where x̃ᵥ is the
#     Dirichlet solution with boundary data v. So the adjoint is another Dirichlet solve:
#     λₛ = Dirichlet solution with x_B = E C⁻¹ Uᵀ rₛ,   ∂J/∂σₐ = +Σₛ λₛᵀ (∂A/∂σₐ) xₛ
#
# Jacobian rows (one per measurement m and pattern s) use the same identity with rₛ replaced by
# unit vectors: Z = A⁺ Qᵀ Π Uᵀ (Neumann) or Z = Dirichlet solutions for E C⁻¹ Uᵀ, one block
# solve with n_obs right-hand sides, then J[(s, m), a] = ∓ zₘᵀ (∂A/∂σₐ) xₛ via the
# ConductivityTensor. The misfit metric enters only through U, so other metrics plug in by
# new AbstractMisfit types.

"""
    AdjointStateObjective(fm, inputs, data; mode = :neumann, solver = DirectSolver(),
                          misfit = SquaredEuclidean(), gradient = CoefficientGradient())

Least-squares misfit `J(σ) = ½ Σₛ ‖U eₛ(σ)‖²` between predicted and measured data for the
forward model `fm`, with the gradient from one adjoint solve and the Jacobian on request.

- `mode = :neumann`: `inputs` are current patterns (`n_inject × s`), `data` the measured voltages
  (`n_measure × s`). Voltages are compared after removing their mean (the ground is arbitrary).
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
    R::Matrix{Float64}             # whitened residual      n_obs × s
    G::Matrix{Float64}             # adjoint weights        n_obs × s / n_ctrl × s
    jac::Any                       # Jacobian buffers (created on first use)
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
                                 z(n_obs, s), z(n_obs, s), z(n_obs, s), nothing)
end

"""
    n_residual(obj)

Length of the residual vector (`n_obs × s`, stacked pattern by pattern).
"""
n_residual(obj::AdjointStateObjective) = length(obj.R)

# assemble A(σ) and (re)factorise / update the solver
function _update!(obj::AdjointStateObjective, σ)
    fm = obj.fm
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
        _remove_mean!(obj.E)
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
        _remove_mean!(obj.G)                             # Π Uᵀ r
        mul!(obj.B, fm.Q', obj.G)                        # Qᵀ Π Uᵀ r
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
representation) at `σ`. Costs one block solve with `n_obs` right-hand sides plus
`n_obs × s` sparse products with the conductivity tensor.
"""
function residual_and_jacobian!(r::AbstractVector, Jm::AbstractMatrix, obj::AdjointStateObjective,
                                σ::AbstractVector)
    fm = obj.fm
    n_obs, s = size(obj.R)
    size(Jm) == (n_obs * s, fm.n_σ) || throw(DimensionMismatch("J must be $(n_obs * s) × $(fm.n_σ)"))
    residual!(r, obj, σ)
    jb = _jacobian_buffers!(obj)
    Ut = _whitening_adjoint_matrix(obj.misfit, n_obs)
    if obj.mode === :neumann
        _remove_mean!(Ut)                                  # Π Uᵀ
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
    for k in 1:s, m0 in 1:chunk:n_obs
        cols = m0:min(m0 + chunk - 1, n_obs)
        W = view(jb.W, :, 1:length(cols))
        _outer_pair_products!(W, ct, view(jb.Z, :, cols), view(obj.X, :, k))
        Gv = view(jb.Gσ, :, 1:length(cols))
        mul!(Gv, ct.Tt, W)
        view(Jm, (k - 1) * n_obs .+ cols, :) .= sgn .* Gv'
    end
    return r, Jm
end

function _jacobian_buffers!(obj::AdjointStateObjective)
    obj.jac === nothing || return obj.jac
    fm = obj.fm
    n_obs = size(obj.R, 1)
    chunk = min(n_obs, 64)
    nf = length(fm.free_dofs)
    obj.jac = (Z = zeros(fm.n, n_obs), B = zeros(fm.n, n_obs), AX = zeros(fm.n, n_obs),
               Xf = zeros(nf, n_obs), Rf = zeros(nf, n_obs), W = zeros(nnz(fm.tensor.pattern), chunk),
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
