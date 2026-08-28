# =============================================================================
# dps_reconstruct.jl — Diffusion Posterior Sampling (Chung et al., 2022) for
# EIT reconstruction: use a pretrained VP-SDE image prior (CNN or U-Net, see
# initialize.jl) to *regularize* the physics data-fidelity loss `f(σ)`/`∂f(σ)`
# from 001SolveTest.ipynb.
#
# Algorithm, per reverse-diffusion step t_i -> t_{i-1}:
#   1. ε̂ = sde(x_t, t)                                   (noise prediction)
#   2. x̂₀ = (x_t - √(1-ᾱ(t))·ε̂) / √ᾱ(t)                  (Tweedie denoised estimate)
#   3. ordinary reverse-SDE predictor step x_t -> x_{t-1} (Euler–Maruyama, unconditional)
#   4. x_{t-1} ← x_{t-1} - ζ_i · ∇_{x_t} f(σ(x̂₀))          (DPS data-consistency correction)
#
# Step 3 is where the diffusion model does its regularizing — it's the only
# place the network's *output* ε̂ is used, pulling x_t toward whatever the
# network has learned conductivity-like images look like. Step 4 is where the
# EIT physics comes in: the textbook DPS derivation differentiates x̂₀ fully
# w.r.t. x_t, which technically includes ε̂_θ(x_t,t)'s own dependence on x_t
# (i.e. the network's Jacobian) — but that term is dropped here, so step 4
# only ever needs ∇ f(σ(x̂₀)), the EIT gradient, chain-ruled back through the
# fixed 1/√ᾱ(t) Tweedie scaling. The network is only ever called forward,
# never backprop'd. This is a standard DPS simplification (dropping the
# Jacobian term for cost/simplicity is common even in DPS-family follow-up
# papers), and it's the same stop-gradient convention this codebase's own
# R_diff/RED-Diff already uses, just applied to the correction term instead
# of a RED-Diff regularizer.
# =============================================================================

using Dates

include(joinpath(@__DIR__, "initialize.jl"))

# -----------------------------------------------------------------------------
# DPS parameters
# -----------------------------------------------------------------------------

N_STEPS  = 100     # reverse diffusion steps
T_MIN    = 1.0f-3   # stop just short of t=0
ζ        = 1.0f0    # DPS guidance strength (scaled adaptively by residual norm, see below)
SPSA_C   = 0.05f0   # SPSA perturbation size
SPSA_PROBES = 1     # SPSA probes per step (more = lower variance, more `f` calls)

OUT_DIR = joinpath(@__DIR__, "DPS")
mkpath(OUT_DIR)

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Reverse-diffusion loop
# -----------------------------------------------------------------------------

ts = Float32.(range(T, T_MIN; length=N_STEPS))

x = randn(Float32, DIM, DIM)         # x_T ~ N(0, I)
x0_hat = zeros(Float32, DIM, DIM)    # holds the last Tweedie estimate after the loop

for i in 1:N_STEPS
    global x, x0_hat   # `x` and `x0_hat` are reassigned each iteration and must
                        # stay visible after the loop — see Julia's soft-scope
                        # rules for top-level `for` loops (this file is a plain
                        # script, not a function body).
    t = ts[i]
    t_next = i < N_STEPS ? ts[i+1] : 0.0f0
    Δt = t - t_next

    ε̂ = sde(x, t)
    αbar_t = ᾱ(t)
    x0_hat = (x .- sqrt(1 - αbar_t) .* ε̂) ./ sqrt(αbar_t)

    # ---- data-consistency guidance term ----
    loss = data_fidelity(x0_hat)
    grad_x0 = spsa_grad(data_fidelity, x0_hat; c=SPSA_C, n_probes=SPSA_PROBES)
    grad_xt = grad_x0 ./ sqrt(αbar_t)   # stop-gradient through the network (see header note)
    ζ_i = ζ / (sqrt(loss) + 1.0f-8)   # adaptive step size, Chung et al. eq. (17)

    # ---- unconditional reverse-SDE predictor step (Euler–Maruyama) ----
    β_t = β(t)
    score = -ε̂ ./ sqrt(1 - αbar_t)
    drift = -0.5f0 .* β_t .* x .- β_t .* score
    noise = i < N_STEPS ? sqrt(β_t * Δt) .* randn(Float32, DIM, DIM) : zero(x)
    x = x .- Δt .* drift .+ noise

    # ---- DPS correction ----
    x = x .- ζ_i .* grad_xt

    if i % 50 == 0 || i == N_STEPS
        println("step $i/$N_STEPS  t=$(round(t, digits=3))  loss=$(round(loss, digits=6))")
    end
end

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

σ_dps = sigma_from_image(x0_hat)

dps_img = Gray.(display_image_from_sigma(σ_dps))
true_img = Gray.(display_image_from_conductivity(cond_img))

plt = plot(
    plot(map(clamp01nan, true_img), title="ground truth", axis=false),
    plot(map(clamp01nan, dps_img), title="DPS reconstruction", axis=false),
    layout=(1, 2), size=(700, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "dps_reconstruction_$timestamp.png"), map(clamp01nan, dps_img))
savefig(plt, joinpath(OUT_DIR, "dps_comparison_$timestamp.png"))
