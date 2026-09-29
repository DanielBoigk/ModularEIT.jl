# First-order methods: (projected) gradient descent with Barzilai–Borwein steps and (projected)
# L-BFGS. Both work with a Riesz map R (the gradient representation): the descent direction of
# gradient descent is R g, and R is the initial inverse Hessian of L-BFGS. With R = M_σ⁻¹
# (L2Gradient) the iteration is the discretisation of the L² gradient flow and its behaviour
# does not depend on the mesh; with the coefficient gradient (R = I) it does.
#
# Curvature pairs are (s, y) = (Δσ, Δg) with coefficient gradients g, so sᵀy is the dual pairing
# and needs no metric; the scaling γ = sᵀy / yᵀRy of the initial inverse Hessian γR is the
# second Barzilai–Borwein step in the metric of R.

"""
    GradientDescent(; riesz = CoefficientGradient())

Gradient descent `σ ← P(σ - t R g)` with the Riesz map `riesz` (e.g. `L2Gradient(mats)`),
Barzilai–Borwein step sizes `t = sᵀy / yᵀRy` and projected Armijo backtracking.
"""
struct GradientDescent{R <: AbstractRieszMap} <: AbstractOptimizer
    riesz::R
end
GradientDescent(; riesz::AbstractRieszMap = CoefficientGradient()) = GradientDescent(riesz)

"""
    LBFGS(; memory = 10, riesz = CoefficientGradient())

Limited-memory BFGS with `memory` curvature pairs and initial inverse Hessian `γR` for the
Riesz map `riesz`. With bounds, the quasi-Newton direction is restricted to the free variables
and the step is taken along the projection arc (projected L-BFGS); the memory is reset whenever
the direction fails to be a descent direction.
"""
struct LBFGS{R <: AbstractRieszMap} <: AbstractOptimizer
    memory::Int
    riesz::R
end
function LBFGS(; memory::Integer = 10, riesz::AbstractRieszMap = CoefficientGradient())
    memory >= 1 || throw(ArgumentError("memory must be positive"))
    return LBFGS(Int(memory), riesz)
end

mutable struct _FirstOrderWorkspace
    d::Vector{Float64}
    σt::Vector{Float64}
    gt::Vector{Float64}
    σold::Vector{Float64}
    gold::Vector{Float64}
    Ry::Vector{Float64}
    mask::BitVector
    t::Float64                    # step of gradient descent
    S::Vector{Vector{Float64}}    # L-BFGS pairs, oldest first
    Y::Vector{Vector{Float64}}
    ρ::Vector{Float64}
    α::Vector{Float64}
    γ::Float64
end

_workspace(::Union{GradientDescent, LBFGS}, obj, n) =
    _FirstOrderWorkspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), falses(n), NaN,
                         Vector{Float64}[], Vector{Float64}[], Float64[], Float64[], NaN)

function _step!(st, ws::_FirstOrderWorkspace, m::GradientDescent, obj, box)
    gnorm = _projected_gradient_norm(st.σ, st.g, box)
    _binding!(ws.mask, st.σ, st.g, box, _binding_tolerance(st.σ, gnorm))
    _restricted_riesz!(ws.d, m.riesz, st.g, ws.mask)
    t0 = isnan(ws.t) ? _initial_step(st.σ, ws.d) : ws.t
    copyto!(ws.σold, st.σ)
    copyto!(ws.gold, st.g)
    t = _projected_backtracking!(st, obj, box, ws.d, t0, ws.σt, ws.gt)
    t > 0 || return false
    # Barzilai–Borwein step t = sᵀy / yᵀRy for the next iteration (s, y in σt, gt)
    ws.σt .= st.σ .- ws.σold
    ws.gt .= st.g .- ws.gold
    sy = dot(ws.σt, ws.gt)
    copyto!(ws.Ry, ws.gt)
    riesz_map!(ws.Ry, m.riesz)
    yRy = dot(ws.gt, ws.Ry)
    ws.t = sy > 0 && yRy > 0 ? sy / yRy : 2t
    return true
end

function _step!(st, ws::_FirstOrderWorkspace, m::LBFGS, obj, box)
    gnorm = _projected_gradient_norm(st.σ, st.g, box)
    _binding!(ws.mask, st.σ, st.g, box, _binding_tolerance(st.σ, gnorm))
    copyto!(ws.σold, st.σ)
    copyto!(ws.gold, st.g)
    t = 0.0
    if !isempty(ws.S)
        _two_loop!(ws, m, st.g)
        if dot(st.g, ws.d) > 0
            t = _projected_backtracking!(st, obj, box, ws.d, 1.0, ws.σt, ws.gt)
        end
    end
    if t == 0                     # no memory, no descent direction or failed line search: restart
        _reset_memory!(ws)
        _restricted_riesz!(ws.d, m.riesz, st.g, ws.mask)
        t = _projected_backtracking!(st, obj, box, ws.d, _initial_step(st.σ, ws.d), ws.σt, ws.gt)
        t > 0 || return false
    end
    s = st.σ .- ws.σold
    y = st.g .- ws.gold
    sy = dot(s, y)
    riesz_map!(copyto!(ws.Ry, y), m.riesz)
    yRy = dot(y, ws.Ry)
    if sy > eps() * norm(s) * norm(y) && yRy > 0
        if length(ws.S) == m.memory
            popfirst!(ws.S); popfirst!(ws.Y); popfirst!(ws.ρ)
        end
        push!(ws.S, s); push!(ws.Y, y); push!(ws.ρ, 1 / sy)
        ws.γ = sy / yRy
    end
    return true
end

_reset_memory!(ws) = (empty!(ws.S); empty!(ws.Y); empty!(ws.ρ); ws.γ = NaN; ws)

# d = H g_ℱ restricted to the free set ℱ, H the L-BFGS inverse Hessian with H₀ = γR
function _two_loop!(ws, m::LBFGS, g)
    q = ws.d
    q .= g
    q[ws.mask] .= 0
    k = length(ws.S)
    resize!(ws.α, k)
    for i in k:-1:1
        ws.α[i] = ws.ρ[i] * dot(ws.S[i], q)
        q .-= ws.α[i] .* ws.Y[i]
    end
    riesz_map!(q, m.riesz)
    q .*= ws.γ
    for i in 1:k
        β = ws.ρ[i] * dot(ws.Y[i], q)
        q .+= (ws.α[i] - β) .* ws.S[i]
    end
    q[ws.mask] .= 0
    return q
end
