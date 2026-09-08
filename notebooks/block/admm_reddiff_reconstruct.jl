# =============================================================================
# admm_reddiff_reconstruct.jl — ADMM for EIT reconstruction that alternates
# between an exact EIT data-consistency prox and a RED-diff prior prox, using
# the same forward model / diffusion prior as dps_reconstruct.jl /
# diffpir_reconstruct.jl / reddiff_reconstruct.jl (see initialize.jl).
#
# Where reddiff_reconstruct.jl jointly optimizes f(σ(x)) + λ·R_diff(x) with a
# single Adam run, this script splits the two terms with a consensus variable
# `z` and solves
#
#   min_{x,z}  f(σ(x)) + λ·R(z)   s.t.  x = z
#
# via scaled ADMM (x, z, u all live in image/x_pm1 space):
#
#   x ← argmin_x f(σ(x)) + (ρ/2)‖x - z + u‖²     ("EIT prox")
#   z ← argmin_z λ·R(z)  + (ρ/2)‖z - x - u‖²     ("RED-diff prox")
#   u ← u + x - z
#
# EIT prox
# --------
# The x-update is exactly Diff-PIR's data-consistency step (see
# diffpir_reconstruct.jl): map the anchor `z - u` into σ-space (forward-only,
# `sigma_from_image`), run `create_prox_linesearch(f, ∂f)` — the codebase's
# exact-adjoint proximal solver for f(σ)+ρ/2‖σ-σ̃‖² — then map the result back
# to x_pm1 space (`x_pm1_from_sigma`). Penalizing the deviation in σ-space
# rather than x-space is an approximation (same one Diff-PIR already makes),
# but keeps this step exact-gradient / SPSA-free.
#
# RED-diff prox
# -------------
# The z-update has no closed form, so it's approximated with K steps of
# stochastic gradient descent on λ·R(z) + (ρ/2)‖z-y‖², y = x + u:
#
#   for k = 1..K:
#     t ~ Uniform(t_min, T);  ε ~ N(0,I)
#     z_t = √ᾱ(t)·z + √(1-ᾱ(t))·ε
#     ε̂   = ε_θ(z_t, t)                          (single network call, stop-gradient)
#     g   = w(t)·√ᾱ(t)·(ε̂ - ε) + ρ·(z - y)        (RED-diff score residual + prox term)
#     z  ← z - η·g
#   return z
#
# This is `prox_red_diff` below — a direct, single-sample-per-step
# implementation of the RED-diff prox, structurally the z-analogue of R_diff
# in initialize.jl (which instead batches n samples per *outer* iteration for
# a full gradient rather than K prox steps). `t` is drawn continuously from
# [t_min, T] rather than a discrete {1,...,T} grid, matching how ᾱ/sde are
# already defined continuously everywhere else in this codebase.
# =============================================================================

using Dates

include(joinpath(@__DIR__, "initialize.jl"))

# -----------------------------------------------------------------------------
# ADMM parameters
# -----------------------------------------------------------------------------

N_ADMM_ITERS = 50      # outer ADMM iterations (x/z/u sweeps)
RHO          = 1.0f-3    # ADMM penalty / consensus weight, shared by both proxes

# --- EIT prox (x-update): create_prox_linesearch(f, ∂f) called each outer
#     iteration, so keep it cheap — same tradeoff diffpir_reconstruct.jl makes.
EIT_MAXITER = 2

# --- RED-diff prox (z-update) ---
K_INNER    = 16        # inner SGD steps per outer ADMM iteration
ETA_INNER  = 0.01f0    # inner SGD step size
T_MIN_REG  = 0.02f0    # kept away from 0 (degenerate forward marginal), matches R_diff's default
W_REG      = t -> 1.0f0

OUT_DIR = joinpath(@__DIR__, "ADMMRedDiff")
mkpath(OUT_DIR)

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Proxes
# -----------------------------------------------------------------------------

prox_eit = create_prox_linesearch(f, ∂f; maxiter=EIT_MAXITER, verbose=false)

"""
    prox_red_diff(y; rho, T, K, eta, w, t_min)

Approximate proximal operator of the RED-diff regularizer: `K` steps of SGD
on `λ·R(z) + (ρ/2)‖z-y‖²`, starting at `z = y`. See the file header for the
per-step update; `T` is the max diffusion time (this codebase's global `T`
constant from initialize.jl), `w` the RED-diff per-t weighting, and `sde`/`ᾱ`
(also from initialize.jl) the trained noise predictor / VP-SDE schedule.

Returns `(z, mean_err)`, `mean_err` the average `‖ε̂-ε‖²` over the K steps
(for logging only, analogous to `R_diff`'s `err`).
"""
function prox_red_diff(y::AbstractArray; rho::Real, T::Real, K::Int, eta::Real,
                        w=t -> 1.0f0, t_min::Real=0.02f0)
    y32 = Float32.(y)
    z = copy(y32)
    err_sum = 0.0f0
    for _ in 1:K
        t = t_min + (T - t_min) * rand(Float32)
        ε = randn(Float32, size(z))
        αbar_t = ᾱ(t)
        z_t = sqrt(αbar_t) .* z .+ sqrt(1 - αbar_t) .* ε
        ε̂ = sde(z_t, t)
        residual = ε̂ .- ε
        err_sum += sum(abs2, residual) / length(residual)

        g = Float32(w(t)) .* sqrt(αbar_t) .* residual .+ Float32(rho) .* (z .- y32)
        z = z .- Float32(eta) .* g
    end
    return z, err_sum / K
end

# -----------------------------------------------------------------------------
# ADMM loop
# -----------------------------------------------------------------------------

x = zeros(Float32, DIM, DIM)   # EIT-consistent estimate, x_pm1 space
z = zeros(Float32, DIM, DIM)   # RED-diff-consistent estimate, x_pm1 space
u = zeros(Float32, DIM, DIM)   # scaled dual variable, x_pm1 space
loss_data = 0.0
err_reg = 0.0f0

for iter in 1:N_ADMM_ITERS
    global x, z, u, loss_data, err_reg

    # ---- x-update: EIT prox (data consistency, exact adjoint, σ-space) ----
    σ_v = sigma_from_image(z .- u)
    σ_x, loss_data, _ = prox_eit(σ_v, RHO)
    x = x_pm1_from_sigma(σ_x)

    # ---- z-update: RED-diff prox (diffusion prior, x_pm1-space) ----
    z, err_reg = prox_red_diff(x .+ u; rho=RHO, T=T, K=K_INNER, eta=ETA_INNER,
                                w=W_REG, t_min=T_MIN_REG)

    # ---- dual update ----
    u = u .+ x .- z

    if iter % 10 == 0 || iter == N_ADMM_ITERS
        primal_res = norm(x .- z)
        println("admm iter $iter/$N_ADMM_ITERS  data=$(round(loss_data, digits=6))  " *
                "reg=$(round(err_reg, digits=6))  ||x-z||=$(round(primal_res, digits=6))")
    end
end

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

σ_admm = sigma_from_image(x)

admm_img = Gray.(display_image_from_sigma(σ_admm))
true_img = Gray.(display_image_from_conductivity(cond_img))

plt = plot(
    plot(map(clamp01nan, true_img), title="ground truth", axis=false),
    plot(map(clamp01nan, admm_img), title="ADMM RED-diff reconstruction", axis=false),
    layout=(1, 2), size=(700, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "admm_reddiff_reconstruction_$timestamp.png"), map(clamp01nan, admm_img))
savefig(plt, joinpath(OUT_DIR, "admm_reddiff_comparison_$timestamp.png"))
