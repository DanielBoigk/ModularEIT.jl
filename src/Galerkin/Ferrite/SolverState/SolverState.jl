using Ferrite
using Enzyme
using LinearMaps

export FerriteSolverState
export FerriteOptState
export add_diff_Regularizer!
export add_ndiff_Regularizer!
export update_L!
export update_σ!
"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector)

Constructs a `FerriteSolverState` with default pseudo-metric and pseudo-norm functions:
- `d(x,y) = norm(x - y)^2`
- `∂d(x,y) = 2*(x-y)`
- `n(x) = norm(x)^2`
- `∂n(x) = 2*x`

Initializes system matrices (`L`, `LD`), singular values, and update vector `δ` to zeros.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector)
    d = (x, y) -> norm(x - y)^2
    ∂d = (x, y) -> 2 * (x - y)
    n = (x) -> norm(x)^2
    ∂n = (x) -> 2 * x
    FerriteSolverState(fe, σ, d, ∂d, n, ∂n)
end

makegradₓ(d) = (x, y) -> Enzyme.gradient(Reverse, Const(d), x, y)[1]
makegrad(n) = (x) -> Enzyme.gradient(Reverse, Const(n), x)

"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, n)

Constructs a `FerriteSolverState` where the pseudo-metric `d` and pseudo-norm `n` are user-defined.
Derivatives `∂d` and `∂n` are automatically created using Enzyme automatic differentiation.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, n)
    ∂d = makegradₓ(d)
    ∂n = makegrad(n)
    FerriteSolverState(fe, σ, d, ∂d, n, ∂n)
end

"""
    FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, ∂d, n, ∂n)

Constructs a `FerriteSolverState` with user-specified pseudo-metric, its derivative,
pseudo-norm, and its derivative. System matrices and boundary definitions are initialized
according to the provided finite element space `fe`.
"""
function FerriteSolverState(fe::FerriteFESpace, σ::AbstractVector, d, ∂d, n, ∂n)
    ∂Ω = fe.∂Ω
    δ = zeros(fe.n)

    L = assemble_L(fe, σ, 0.0)
    L_fak = factorize(L)
    Σ = zeros(fe.m - 1)
    opt = FerriteOptState(nothing, nothing, 0.0, 0.0, 0.1, 0, 1e-5, LinearMap(spdiagm(ones(fe.n))), copy(δ))
    FerriteSolverState(∂Ω, copy(σ), δ, L, L_fak, nothing, nothing, Σ, d, ∂d, n, ∂n, nothing, nothing, nothing, nothing, nothing, nothing, 0, fe.n, opt, false, 1.0)
end


function update_σ!(state::FerriteSolverState, clip::Bool=false, clip_limit::Float64=1.0)
    state.σ .= max.(state.σ .- state.δ, 1e-6)
    if clip
        state.σ .= min.(state.σ, clip_limit)
    end
end

function update_L!(state::FerriteSolverState, fe::FerriteFESpace)
    state.L .= assemble_L!(state.L, fe, state.σ)
end

function add_diff_Regularizer!(state::FerriteSolverState, Reg, R_diff_args, ∇Reg)
    state.R_diff = Reg
    state.R_diff_args = R_diff_args
    state.∇R = ∇Reg
end
function add_diff_Regularizer!(state::FerriteSolverState, Reg, R_diff_args)
    state.R_diff = Reg
    state.R_diff_args = R_diff_args
    state.∇R = Enzyme.gradient(Reg)
end
function add_ndiff_Regularizer!(state::FerriteSolverState, nReg, R_ndiff_args)
    state.R_ndiff = nReg
    state.R_ndiff_args = R_ndiff_args
end


function init_opt!(state::FerriteSolverState, m::Int, n::Int)
    state.opt.J = zeros(n, m)
    state.opt.r = zeros(m)
    state.op.λ = 1e-5
end
