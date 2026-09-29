# Gauss–Newton and Levenberg–Marquardt for J(σ) = ½‖r(σ)‖² + Σₖ αₖ Rₖ(σ).
#
# Model Hessian H = JᵀJ + Σₖ αₖ Hₖ(σ) (Hₖ the Gauss–Newton Hessian of the regularizers), step on
# the free set ℱ:
#     (H + λD)_ℱℱ δ_ℱ = -g_ℱ
# with damping matrix D (identity, Marquardt's diag(H) or a user matrix such as M_σ).
#
# Levenberg–Marquardt (`damping = :lm`) adapts λ with the gain ratio
#     ρ = (J(σ) - J(σ + δ)) / (-gᵀδ - ½ δᵀHδ)
# (Nielsen's update; with bounds δ is the projected step). `damping = :linesearch` keeps λ fixed
# and backtracks along the projection arc instead.
#
# Linear algebra: EIT has few residuals m (electrodes × patterns) and possibly many σ dofs.
# `:dense` forms H densely (n_σ² memory) and uses Cholesky. `:woodbury` never forms JᵀJ: with the
# sparse SPD B = Σ αₖ Hₖ + λD,
#     (B + JᵀJ)⁻¹ b = B⁻¹b - W (I + J W)⁻¹ J B⁻¹b,   W = B⁻¹Jᵀ,
# i.e. one sparse Cholesky of B, m + 1 sparse solves and an m × m dense system. `:auto` uses
# Woodbury when n_σ > m and B is available (λ > 0 with a sparse D).

"""
    GaussNewton(; damping = :lm, λ = nothing, scaling = :identity, linear_solver = :auto)

Gauss–Newton method for least-squares objectives ([`AdjointStateObjective`](@ref), possibly
wrapped in a [`RegularizedObjective`](@ref); regularizers contribute their
[`gauss_newton_hessian`](@ref)).

- `damping = :lm`: Levenberg–Marquardt with adaptive `λ` (initial `λ` relative to the largest
  diagonal entry of `JᵀJ + ΣαH`, default `1e-3`);
- `damping = :linesearch`: fixed damping `λ` (default `1e-8`, relative) and projected Armijo
  backtracking.
- `scaling`: damping matrix `D`: `:identity`, `:marquardt` (`diag(JᵀJ + ΣαH)`, floored) or a
  symmetric positive definite matrix (e.g. the σ mass matrix `FEMatrices(disc).M_σ`).
- `linear_solver`: `:dense` (form the n_σ × n_σ matrix), `:woodbury` (sparse Cholesky of
  `ΣαH + λD` and an m × m system for the m residuals) or `:auto`.
"""
struct GaussNewton{DM} <: AbstractOptimizer
    damping::Symbol
    λ::Float64
    scaling::DM
    linear_solver::Symbol
end

function GaussNewton(; damping::Symbol = :lm, λ = nothing, scaling = :identity, linear_solver::Symbol = :auto)
    damping in (:lm, :linesearch) || throw(ArgumentError("damping must be :lm or :linesearch, got :$damping"))
    linear_solver in (:auto, :dense, :woodbury) ||
        throw(ArgumentError("linear_solver must be :auto, :dense or :woodbury, got :$linear_solver"))
    scaling isa AbstractMatrix || scaling in (:identity, :marquardt) ||
        throw(ArgumentError("scaling must be :identity, :marquardt or a matrix"))
    λ = λ === nothing ? (damping === :lm ? 1e-3 : 1e-8) : Float64(λ)
    λ >= 0 || throw(ArgumentError("λ must be nonnegative"))
    return GaussNewton(damping, λ, scaling, linear_solver)
end

mutable struct _GaussNewtonWorkspace{O}
    data::O                       # least-squares part
    r::Vector{Float64}
    Jm::Matrix{Float64}
    HR::Any                       # Σ αₖ Hₖ (sparse) or nothing
    δ::Vector{Float64}
    σt::Vector{Float64}
    gt::Vector{Float64}
    mask::BitVector
    λ::Float64                    # absolute damping
    ν::Float64
end

_least_squares_part(obj::AbstractObjective) = obj
_least_squares_part(obj::RegularizedObjective) = obj.data

function _workspace(m::GaussNewton, obj, n)
    data = _least_squares_part(obj)
    hasmethod(residual_and_jacobian!, Tuple{Vector{Float64}, Matrix{Float64}, typeof(data), Vector{Float64}}) &&
        hasmethod(n_residual, Tuple{typeof(data)}) ||
        throw(ArgumentError("GaussNewton needs a least-squares objective with residual_and_jacobian! and n_residual " *
                            "(e.g. AdjointStateObjective), got $(nameof(typeof(data)))"))
    nr = n_residual(data)
    return _GaussNewtonWorkspace(data, zeros(nr), zeros(nr, n), nothing, zeros(n), zeros(n), zeros(n),
                                 falses(n), NaN, 2.0)
end

# value, gradient, residual, Jacobian and regularizer Hessians at st.σ
function _linearize!(st, ws::_GaussNewtonWorkspace, obj)
    residual_and_jacobian!(ws.r, ws.Jm, ws.data, st.σ)
    st.value = sum(abs2, ws.r) / 2
    mul!(st.g, ws.Jm', ws.r)
    ws.HR = nothing
    if obj isa RegularizedObjective
        gr = ws.gt
        for (α, reg) in zip(obj.weights, obj.regularizers)
            α == 0 && continue
            st.value += α * value_and_gradient!(gr, reg, st.σ)
            st.g .+= α .* gr
            H = α * gauss_newton_hessian(reg, st.σ)
            ws.HR = ws.HR === nothing ? sparse(H) : ws.HR + H
        end
    end
    st.nevals += 1
    return st
end

function _initialize!(st, ws::_GaussNewtonWorkspace, m::GaussNewton, obj, box)
    _linearize!(st, ws, obj)
    isfinite(st.value) || throw(ArgumentError("the objective is not finite at the initial guess"))
    return st
end

# diag(JᵀJ + H_R)
function _model_diagonal(ws)
    d = vec(sum(abs2, ws.Jm; dims = 1))
    ws.HR === nothing || (d .+= diag(ws.HR))
    return d
end

function _damping_matrix(m::GaussNewton, ws, F)
    if m.scaling === :identity
        return sparse(1.0I, length(F), length(F))
    elseif m.scaling === :marquardt
        d = _model_diagonal(ws)[F]
        floor_ = 1e-10 * max(maximum(d; init = 0.0), floatmin())
        return spdiagm(0 => max.(d, floor_))
    else
        return sparse(m.scaling[F, F])
    end
end

# δ_ℱ = -(JᵀJ + H_R + λD)_ℱℱ⁻¹ g_ℱ; returns false if the system is not positive definite
function _gauss_newton_direction!(ws, m::GaussNewton, g, F, D, λ)
    JF = view(ws.Jm, :, F)
    bF = -g[F]
    B = λ * D
    ws.HR === nothing || (B = B + ws.HR[F, F])
    use_woodbury = m.linear_solver === :woodbury ||
                   (m.linear_solver === :auto && length(F) > size(ws.Jm, 1) && λ > 0)
    fill!(ws.δ, 0)
    if use_woodbury
        Bf = cholesky(Symmetric(sparse(B)); check = false)
        if issuccess(Bf)
            JFt = Matrix(JF')
            W = Bf \ JFt
            u = Bf \ bF
            S = Symmetric(I + JF * W)
            ws.δ[F] .= u .- W * (S \ (JF * u))
            return true
        end
        m.linear_solver === :woodbury && return false
    end
    H = Matrix(B)
    mul!(H, JF', JF, 1.0, 1.0)
    C = cholesky!(Symmetric(H); check = false)
    issuccess(C) || return false
    ws.δ[F] .= C \ bF
    return true
end

# predicted decrease of the quadratic model for the step δ: -gᵀδ - ½ δᵀHδ
function _predicted_decrease(ws, g, δ)
    q = sum(abs2, ws.Jm * δ)
    ws.HR === nothing || (q += dot(δ, ws.HR, δ))
    return -dot(g, δ) - q / 2
end

function _step!(st, ws::_GaussNewtonWorkspace, m::GaussNewton, obj, box)
    gnorm = _projected_gradient_norm(st.σ, st.g, box)
    _binding!(ws.mask, st.σ, st.g, box, _binding_tolerance(st.σ, gnorm))
    F = findall(!, ws.mask)
    isempty(F) && return false
    D = _damping_matrix(m, ws, F)
    if isnan(ws.λ)       # relative → absolute damping
        scale = maximum(_model_diagonal(ws)[F]) / max(maximum(diag(D)), floatmin())
        ws.λ = m.λ * (scale > 0 ? scale : 1.0)
    end
    if m.damping === :linesearch
        _gauss_newton_direction!(ws, m, st.g, F, D, ws.λ) || return false
        ws.δ .*= -1                                   # σ(t) = P(σ - t d) with d = -δ
        t = _projected_backtracking!(st, obj, box, ws.δ, 1.0, ws.σt, ws.gt)
        t > 0 || return false
        _linearize!(st, ws, obj)
        return true
    end
    # Levenberg–Marquardt
    for _ in 1:60
        if _gauss_newton_direction!(ws, m, st.g, F, D, ws.λ)
            ws.σt .= st.σ .+ ws.δ
            _clamp_box!(ws.σt, box)
            ws.δ .= ws.σt .- st.σ                     # the step actually taken
            pred = _predicted_decrease(ws, st.g, ws.δ)
            if pred > 0
                Jt = _try_objective_value(obj, ws.σt)
                st.nevals += 1
                ρ = (st.value - Jt) / pred
                if ρ > 0
                    copyto!(st.σ, ws.σt)
                    _linearize!(st, ws, obj)
                    ws.λ *= max(1 / 3, 1 - (2ρ - 1)^3)
                    ws.ν = 2.0
                    return true
                end
            elseif norm(ws.δ) <= eps() * (1 + norm(st.σ))
                return false                          # stationary on the free set
            end
        end
        ws.λ *= ws.ν
        ws.ν *= 2
    end
    return false
end

function _try_objective_value(obj, σ)
    try
        return objective_value(obj, σ)
    catch e
        e isa Union{PosDefException, SingularException, ZeroPivotException, LAPACKException} || rethrow()
        return Inf
    end
end
