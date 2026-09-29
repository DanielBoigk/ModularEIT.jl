# Kohn–Vogelius functional. For every pattern s solve the current-driven problem with the injected
# currents Iₛ (state x_N) and the voltage-driven problem with the measured voltages Vₛ (state x_D),
# both with the same conductivity, and measure their disagreement in the energy norm:
#
#     J(σ) = ½ Σₛ (x_N - x_D)ᵀ A(σ) (x_N - x_D)   ( = ½ Σₛ ∫ σ |∇(u_N - u_D)|² (+ CEM terms) )
#
# J ≥ 0, and J = 0 exactly when both problems have the same solution, i.e. at a conductivity
# that explains the data. Expanding, x_Nᵀ A x_D = (P I)ᵀ x_D only involves data (the load lives
# on the Dirichlet dofs where x_D is prescribed), and both remaining terms are stationary with
# respect to their state, so
#
#     ∂J/∂σₐ = ½ Σₛ ( x_Dᵀ (∂A/∂σₐ) x_D - x_Nᵀ (∂A/∂σₐ) x_N )  = ½ ∫ ψₐ (|∇u_D|² - |∇u_N|²)
#
# with no adjoint solve. All patterns are independent block solves (two factorisations per σ),
# and the gradient is two ConductivityTensor contractions. This requires the voltage-driven problem
# to prescribe exactly the measured voltages at the sites where current is injected: continuum
# model, point electrodes and the CEM with voltages measured on all electrodes. For the gap model
# the voltage-driven counterpart is the shunt model, a different physical model, so J does not
# vanish at the true conductivity; a warning is issued.

"""
    KohnVogeliusObjective(fm, currents, voltages; solver = DirectSolver(), gradient = CoefficientGradient())

Kohn–Vogelius functional `J(σ) = ½ Σₛ ‖x_N,ₛ - x_D,ₛ‖²_{A(σ)}` for the current patterns
`currents` (`n_inject × s`) and the measured voltages `voltages` (`n_control × s`, at the
injection sites: boundary dofs, points or all CEM electrodes). The gradient needs no adjoint
solve. After an evaluation, [`boundary_error`](@ref) returns the voltage misfit of the
current-driven solution and [`pattern_values`](@ref) the contribution of every pattern.
"""
mutable struct KohnVogeliusObjective{FM <: ForwardModel, S <: AbstractLinearSolver, RM <: AbstractRieszMap} <: AbstractObjective
    fm::FM
    currents::Matrix{Float64}
    voltages::Matrix{Float64}
    solver::S
    state_N::Any
    state_D::Any
    riesz::RM
    XN::Matrix{Float64}      # current-driven states   n × s
    XD::Matrix{Float64}      # voltage-driven states   n × s
    B::Matrix{Float64}
    AX::Matrix{Float64}
    Xf::Matrix{Float64}
    Rf::Matrix{Float64}
    values::Vector{Float64}  # per-pattern values
    berr::Matrix{Float64}    # Q x_N - voltages (mean removed)
end

function KohnVogeliusObjective(fm::ForwardModel, currents::AbstractVecOrMat, voltages::AbstractVecOrMat;
                               solver::AbstractLinearSolver = DirectSolver(),
                               gradient::AbstractRieszMap = CoefficientGradient())
    Ic, V = Matrix{Float64}(_as_matrix(currents)), Matrix{Float64}(_as_matrix(voltages))
    s = size(Ic, 2)
    size(V, 2) == s || throw(DimensionMismatch("currents and voltages need the same number of patterns"))
    size(Ic, 1) == n_inject(fm) || throw(DimensionMismatch("currents need $(n_inject(fm)) rows"))
    size(V, 1) == n_control(fm) || throw(DimensionMismatch("voltages need $(n_control(fm)) rows"))
    fm.kohn_vogelius_consistent ||
        @warn "The Kohn–Vogelius functional is not consistent for $(nameof(typeof(fm.model))) with these " *
              "electrodes: the voltage-driven problem is a different model, so J does not vanish at the " *
              "true conductivity. Use the continuum model, point electrodes or the complete electrode " *
              "model with voltages on all electrodes."
    n, nf = fm.n, length(fm.free_dofs)
    z(r, c) = zeros(r, c)
    return KohnVogeliusObjective(fm, Ic, V, solver, nothing, nothing, gradient, z(n, s), z(n, s), z(n, s),
                                 z(n, s), z(nf, s), z(nf, s), zeros(s), z(n_measure(fm), s))
end

function _forward!(obj::KohnVogeliusObjective, σ)
    fm = obj.fm
    system_matrix!(fm, σ)
    if obj.state_N === nothing
        obj.state_N = _init_neumann_solver(obj.solver, fm)
        obj.state_D = _init_dirichlet_solver(obj.solver, fm)
    else
        _update_solver!(obj.state_N, fm.A)
        _update_solver!(obj.state_D, fm.A_ff)
    end
    mul!(obj.B, fm.P, obj.currents)
    _solve!(obj.XN, obj.state_N, obj.B)
    _dirichlet_solve!(obj.XD, obj.state_D, fm, obj.voltages, obj.AX, obj.Xf, obj.Rf)
    # energy of the difference, pattern by pattern (B, AX as scratch)
    obj.B .= obj.XN .- obj.XD
    mul!(obj.AX, fm.A, obj.B)
    for k in axes(obj.B, 2)
        obj.values[k] = dot(view(obj.B, :, k), view(obj.AX, :, k)) / 2
    end
    if size(obj.berr, 1) == size(obj.voltages, 1)
        mul!(obj.berr, fm.Q, obj.XN)
        obj.berr .-= obj.voltages
        _project!(obj.berr, fm.measure_weights)
    end
    return sum(obj.values)
end

objective_value(obj::KohnVogeliusObjective, σ::AbstractVector) = _forward!(obj, σ)

function value_and_gradient!(g::AbstractVector, obj::KohnVogeliusObjective, σ::AbstractVector)
    J = _forward!(obj, σ)
    ct = obj.fm.tensor
    tensor_gradient!(g, ct, obj.XD, obj.XD; α = 0.5)
    tensor_gradient!(g, ct, obj.XN, obj.XN; α = -0.5, β = 1)
    riesz_map!(g, obj.riesz)
    return J
end

"""
    boundary_error(obj::KohnVogeliusObjective)

Measured minus predicted voltages of the current-driven states at the last evaluation
(`Q x_N - V`, mean removed per pattern).
"""
boundary_error(obj::KohnVogeliusObjective) = obj.berr

"""
    pattern_values(obj::KohnVogeliusObjective)

Contribution `½ ‖x_N,ₛ - x_D,ₛ‖²_A` of every pattern at the last evaluation.
"""
pattern_values(obj::KohnVogeliusObjective) = obj.values

_gradient_representation(obj::KohnVogeliusObjective) = obj.riesz
