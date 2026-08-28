# =============================================================================
# diffpir_reconstruct.jl — Diff-PIR (Zhu et al., 2023, "Denoising Diffusion
# Models for Plug-and-Play Image Restoration") for EIT reconstruction, using
# the same forward model / diffusion prior as dps_reconstruct.jl (see
# initialize.jl).
#
# Unlike DPS (one gradient nudge per step), Diff-PIR replaces the correction
# with an actual proximal/MAP solve at each denoising step:
#
#   1. ε̂ = sde(x_t, t)                                    (noise prediction)
#   2. x̂₀ = (x_t - √(1-ᾱ(t))·ε̂) / √ᾱ(t)                   (Tweedie denoise, image space)
#   3. σ̃  = sigma_from_image(x̂₀)                           (map to FEM σ-space, forward only)
#   4. σ̂₀ = argmin_σ f(σ) + (1/2ρ_t)‖σ - σ̃‖²                (proximal data-consistency step)
#   5. x̂₀' = x_pm1_from_sigma(σ̂₀)                          (map the corrected σ back to x_pm1 space, forward only)
#   6. x_{t-1} = √ᾱ(t')x̂₀' + √(1-ᾱ(t'))·(√(1-ζ)ε̂' + √ζ·η)   (DDIM-style renoise, η~N(0,I), t'=next t)
#      where ε̂' = (x_t - √ᾱ(t)x̂₀')/√(1-ᾱ(t)) is the noise direction implied
#      by the corrected estimate, and ζ∈[0,1] interpolates deterministic
#      DDIM (ζ=0) <-> full ancestral/DDPM stochasticity (ζ=1).
#
# Step 4 is exactly `create_prox_linesearch(f, ∂f, ρ)` from initialize.jl —
# this codebase's own proximal solver for f(σ)+ρ/2‖σ-σ̃‖², already used for
# the classical (non-diffusion) reconstruction in 001SolveTest.ipynb. That
# means, unlike DPS, this script needs *no* SPSA/approximate gradient at
# all: the proximal solve stays entirely in σ-space using the exact adjoint
# gradient ∂f, and `sigma_from_image`/`x_pm1_from_sigma` (both gradient-free,
# forward-only maps) just shuttle the estimate between x_pm1 space (where
# the diffusion prior lives) and σ-space (where the physics lives).
#
# ρ_t schedule: Diff-PIR ties the proximal weight to the noise schedule via
# ρ_t = λ·σ_n²/σ̄_t², σ̄_t² = (1-ᾱ_t)/ᾱ_t (so ρ_t grows as t→0, trusting the
# physics more once x_t is nearly clean). `f` here is already a normalized
# mean-squared-residual (see create_block_f∂f's docstring), not literally
# ‖y-Aσ‖²/(2σ_n²), so σ_n² is folded into λ below — treat LAMBDA as a knob to
# tune for your own ‖·‖ scale, not a physically calibrated noise variance.
# =============================================================================

using Dates

include(joinpath(@__DIR__, "initialize.jl"))

# -----------------------------------------------------------------------------
# Diff-PIR parameters
# -----------------------------------------------------------------------------

N_STEPS = 100        # Diff-PIR is a MAP-style sampler, not full ancestral sampling — far fewer steps than DPS
T_MIN   = 1.0f-3
LAMBDA  = 10.0f0     # data-fidelity <-> prior balance; ρ_t = LAMBDA * ᾱ(t) / (1 - ᾱ(t))
ZETA    = 0.3f0      # renoise stochasticity: 0 = deterministic (DDIM), 1 = fully stochastic (DDPM)

OUT_DIR = joinpath(@__DIR__, "DiffPIR")
mkpath(OUT_DIR)

Random.seed!(SEED)


# `maxiter` capped well below the default (20): this prox solve runs once per
# *diffusion* step (N_STEPS of them), each inner iteration itself running a
# Brent line search that calls the full FEM forward/adjoint solve many times
# — at maxiter=20 that's tens of thousands of physics solves for one
# reconstruction. Diff-PIR's inner step doesn't need to fully converge every
# timestep (only the outer trajectory does), so a handful of iterations is
# the standard tradeoff. `verbose=true` here (unlike the shared, quiet
# `prox_obj` in initialize.jl) so progress is visible instead of looking hung.
prox_diffpir = create_prox_linesearch(f, ∂f; maxiter=5, verbose=true)

# -----------------------------------------------------------------------------
# Reverse loop
# -----------------------------------------------------------------------------

ts = Float32.(range(T, T_MIN; length=N_STEPS))

x = randn(Float32, DIM, DIM)         # x_T ~ N(0, I)
x0_hat = zeros(Float32, DIM, DIM)    # holds the last corrected estimate after the loop

for i in 1:N_STEPS
    global x, x0_hat   # reassigned each iteration; see the soft-scope note in dps_reconstruct.jl
    t = ts[i]
    t_next = i < N_STEPS ? ts[i+1] : 0.0f0

    ε̂ = sde(x, t)
    αbar_t = ᾱ(t)
    x0_hat = (x .- sqrt(1 - αbar_t) .* ε̂) ./ sqrt(αbar_t)

    # ---- proximal data-consistency step (exact, in σ-space) ----
    σ_tilde = sigma_from_image(x0_hat)
    ρ_t = LAMBDA * αbar_t / (1 - αbar_t)
    σ_hat, loss, _ = prox_diffpir(σ_tilde, ρ_t)
    x0_hat = x_pm1_from_sigma(σ_hat)

    # ---- DDIM-style renoise ----
    ε_dir = (x .- sqrt(αbar_t) .* x0_hat) ./ sqrt(1 - αbar_t)
    η = randn(Float32, DIM, DIM)
    αbar_next = ᾱ(t_next)
    x = sqrt(αbar_next) .* x0_hat .+ sqrt(1 - αbar_next) .* (sqrt(1 - ZETA) .* ε_dir .+ sqrt(ZETA) .* η)
    # at the final step t_next=0 => αbar_next=1 => x collapses to x0_hat exactly, no branch needed

    if i % 10 == 0 || i == N_STEPS
        println("step $i/$N_STEPS  t=$(round(t, digits=3))  ρ=$(round(ρ_t, digits=3))  loss=$(round(loss, digits=6))")
    end
end

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

σ_diffpir = sigma_from_image(x0_hat)

diffpir_img = Gray.(display_image_from_sigma(σ_diffpir))
true_img = Gray.(display_image_from_conductivity(cond_img))

plt = plot(
    plot(map(clamp01nan, true_img), title="ground truth", axis=false),
    plot(map(clamp01nan, diffpir_img), title="Diff-PIR reconstruction", axis=false),
    layout=(1, 2), size=(700, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "diffpir_reconstruction_$timestamp.png"), map(clamp01nan, diffpir_img))
savefig(plt, joinpath(OUT_DIR, "diffpir_comparison_$timestamp.png"))
