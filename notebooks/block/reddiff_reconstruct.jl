# =============================================================================
# reddiff_reconstruct.jl — RED-diff (Mardani et al., 2023, "A Variational
# Perspective on Solving Inverse Problems with Diffusion Models") for EIT
# reconstruction, using the same forward model / diffusion prior as
# dps_reconstruct.jl / diffpir_reconstruct.jl (see initialize.jl).
#
# RED-diff is structurally the odd one out among the three: DPS and Diff-PIR
# both run a reverse-diffusion *trajectory* (x_T -> x_{T-1} -> ... -> x_0).
# RED-diff instead directly optimizes a single image `x` (no trajectory, no
# per-step noise schedule) against:
#
#   min_x  f(σ(x)) + λ · R_diff(x)
#
# where `f(σ(x))` is the usual EIT physics data-fidelity loss and `R_diff` —
# already defined in initialize.jl, ported verbatim from ../CNN/sample.jl /
# ../Unet/sample.jl — is the score-matching-residual regularizer: at each
# outer iteration it draws n random (t, ε) pairs, forward-diffuses `x` to
# each x_t = √ᾱ_t x + √(1-ᾱ_t) ε, and returns the batched
# `w(t)·√ᾱ_t·(ε̂_θ(x_t,t) - ε)` gradient with the network treated as a fixed
# (stop-gradient) denoiser — see R_diff's own docstring in initialize.jl for
# the full derivation. That stop-gradient trick is exactly what makes this
# method (and Diff-PIR's data step) cheap: no backprop through the network.
#
# `x` is optimized directly with Adam (Optimisers.jl, already a dependency
# here — it's what ../CNN/../Unet's own training scripts use), same as the
# original RED-diff paper.
#
# The data term `f(σ(x))` still needs a gradient w.r.t. the *image* `x`, and
# `sigma_from_image` (image -> FEM σ-space) isn't set up for exact
# backprop — same situation as dps_reconstruct.jl, same fix: `spsa_grad`
# (SPSA), reused unchanged from initialize.jl.
# =============================================================================

using Dates

include(joinpath(@__DIR__, "initialize.jl"))

# -----------------------------------------------------------------------------
# RED-diff parameters
# -----------------------------------------------------------------------------

N_ITERS  = 1000       # outer Adam iterations
LR       = 0.02f0     # Adam learning rate over the image `x`
LAMBDA_REG = 1.0f0    # data-fidelity <-> diffusion-regularizer balance (tune this)

N_TIME_SAMPLES = 16    # (t, ε) probes per iteration for R_diff — one batched network call
T_MIN_REG = 0.02f0     # matches R_diff's own default; kept away from 0 (degenerate forward marginal)
W_REG = t -> 1.0f0     # R_diff's per-t weighting; constant matches R_diff's own default

SPSA_C = 0.05f0        # SPSA perturbation size for the data term's image-space gradient
SPSA_PROBES = 1

OUT_DIR = joinpath(@__DIR__, "RedDiff")
mkpath(OUT_DIR)

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Optimization loop
# -----------------------------------------------------------------------------

x = zeros(Float32, DIM, DIM)   # log-conductivity ≡ 0 => conductivity ≡ 1 everywhere,
                                # the same uniform starting guess 001SolveTest.ipynb's
                                # classical reconstruction uses (`σ₁ = ones(fe.n_σ)`)

opt_state = Optimisers.setup(Optimisers.Adam(LR), x)

for iter in 1:N_ITERS
    global x, opt_state
    loss_data = data_fidelity(x)
    grad_data = spsa_grad(data_fidelity, x; c=SPSA_C, n_probes=SPSA_PROBES)

    err_reg, grad_reg = R_diff(x, T, N_TIME_SAMPLES; t_min=T_MIN_REG, w=W_REG)

    grad = grad_data .+ LAMBDA_REG .* grad_reg
    opt_state, x = Optimisers.update!(opt_state, x, grad)

    if iter % 50 == 0 || iter == N_ITERS
        println("iter $iter/$N_ITERS  data=$(round(loss_data, digits=6))  reg=$(round(err_reg, digits=6))")
    end
end

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

σ_reddiff = sigma_from_image(x)

reddiff_img = Gray.(display_image_from_sigma(σ_reddiff))
true_img = Gray.(display_image_from_conductivity(cond_img))

plt = plot(
    plot(map(clamp01nan, true_img), title="ground truth", axis=false),
    plot(map(clamp01nan, reddiff_img), title="RED-diff reconstruction", axis=false),
    layout=(1, 2), size=(700, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "reddiff_reconstruction_$timestamp.png"), map(clamp01nan, reddiff_img))
savefig(plt, joinpath(OUT_DIR, "reddiff_comparison_$timestamp.png"))
