# =============================================================================
# postprocessing.jl — clean up an existing 64x64 grayscale EIT reconstruction
# with the diffusion prior, via the RED-Diff proximal operator from the thesis
# ("Diffusion-Prox for ADMM").
#
# Unlike reddiff_reconstruct.jl this script never touches the EIT forward model.
# It takes a finished reconstruction image `y` (whatever produced it — Tikhonov,
# TV, L-BFGS, an earlier RED-Diff run, ...) and returns
#
#     prox_RED(y) = argmin_x  E_RED(x) + (ρ/2) ‖x - y‖²_P
#
# where
#   • E_RED(x)      is the RED-Diff score-matching error (Monte-Carlo estimate of
#                   E_{t,ε}[ω_t ‖ε̂_θ(x_t,t) - ε‖²]) — exactly `R_diff` from
#                   initialize.jl, which also returns its gradient ∇E_RED with
#                   the network treated as a stop-gradient denoiser.
#   • ‖·‖²_P        is the P-weighted quadratic uᵀ P u with P a *diagonal*
#                   (per-pixel) self-adjoint operator, P(u) = Wp .* u, so
#                   ∇[(ρ/2)‖x-y‖²_P] = ρ P(x - y) exactly as the thesis writes.
#
# The pseudocode's `error_RED` ≡ `R_diff`, and `optimize(f, g!, x₀, opt, ...)` is
# instantiated here with Adam (Optimisers.jl) — R_diff redraws (t, ε) every call
# so the objective/gradient are inherently stochastic and a stochastic optimizer
# is the right fit (this is also what the RED-Diff paper uses).
#
# --- the boundary projection P -----------------------------------------------
# `d(i,j) = min(i-1, j-1, DIM-i, DIM-j)` is the distance in pixels to the nearest
# image edge (0 on the border). Two modes, selected by BOUNDARY_ONLY:
#
#   BOUNDARY_ONLY = true  (default) — no decay. P is a *binary mask*: 1 on the
#       outermost BOUNDARY_WIDTH-pixel ring, 0 everywhere inside. The quadratic
#       (ρ/2)‖x-y‖²_P then only sees the boundary ring, so it contributes to the
#       objective *only if a boundary pixel moves* — the interior is generated
#       completely freely by the diffusion prior.
#
#   BOUNDARY_ONLY = false — smooth fallback: Wp = exp(-d / BOUNDARY_DECAY),
#       1 on the border and decaying exponentially inward.
#
# HARD_RING > 0 additionally forces the outermost `HARD_RING` pixels to exactly
# `y` after every step (a literal back-projection); HARD_RING = 0 leaves the
# boundary to the soft quadratic alone.
# =============================================================================

using Dates

include(joinpath(@__DIR__, "initialize.jl"))

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

# --- input: the reconstruction to post-process -------------------------------
# Set IN_PATH to a grayscale image file (resized to DIM×DIM if needed). There is
# no default — you must write an explicit path here, and the script aborts below
# if it is left empty or points at a file that does not exist.
IN_PATH = "lbfgsb_25.png"   # e.g. joinpath(@__DIR__, "RedDiff", "reddiff_reconstruction_20260831_160704.png")

# --- prox / RED-Diff ---------------------------------------------------------
RHO         = 1.0f0     # ρ — data-consistency weight (how hard the P-weighted pull to `y` is)
ETA         = 1.0f0     # η — scales ∇E_RED relative to the prox gradient (RED-Diff tuning knob)
N_ITERS     = 100      # Adam iterations for the inner optimize()
LR          = 0.01f0   # Adam learning rate over the image `x`
N_TIME_SAMPLES = 16    # (t, ε) probes per iteration for E_RED — one batched network call
T_MIN_REG   = 0.02f0   # matches R_diff's default; kept away from 0 (degenerate forward marginal)
W_REG       = t -> 1.0f0   # ω(t) — per-t weighting inside E_RED; constant matches R_diff's default

# --- boundary projection P -------------------------------------------------
BOUNDARY_ONLY  = false   # true: penalize ONLY the boundary ring (binary P, no decay), interior free.
                        # false: smooth Wp = exp(-d / BOUNDARY_DECAY).
BOUNDARY_WIDTH = 1      # thickness in pixels of the boundary ring P covers (BOUNDARY_ONLY = true)
BOUNDARY_DECAY = 4.0f0  # ℓ, in pixels: Wp = exp(-d/ℓ) — only used when BOUNDARY_ONLY = false
HARD_RING      = 1      # outermost pixels forced exactly to `y` every step (0 disables)

OUT_DIR = joinpath(@__DIR__, "PostProc")
mkpath(OUT_DIR)

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Boundary projection P(u) = Wp .* u  (diagonal, self-adjoint)
# -----------------------------------------------------------------------------

"""
    boundary_projection(dim; boundary_only=BOUNDARY_ONLY, width=BOUNDARY_WIDTH,
                        decay=BOUNDARY_DECAY, hard_ring=HARD_RING)

Returns `(Wp, hard_mask)`, with `d` the distance in pixels to the nearest image
edge (0 on the border):
- `Wp`        : (dim,dim) Float32 per-pixel weights of the diagonal operator
               `P(u) = Wp .* u`. If `boundary_only`, `Wp = (d < width)` — a
               binary mask on the outermost `width`-pixel ring, 0 in the
               interior (the quadratic then only penalizes boundary changes).
               Otherwise `Wp = exp(-d / decay)`.
- `hard_mask` : BitMatrix, true on the outermost `hard_ring` pixels — where the
               prox output is overwritten with `y` exactly.
"""
function boundary_projection(dim::Int; boundary_only::Bool=BOUNDARY_ONLY, width::Int=BOUNDARY_WIDTH,
                             decay::Real=BOUNDARY_DECAY, hard_ring::Int=HARD_RING)
    d = [min(i - 1, j - 1, dim - i, dim - j) for i in 1:dim, j in 1:dim]
    Wp = boundary_only ? Float32.(d .< width) : exp.(-Float32.(d) ./ Float32(decay))
    hard_mask = d .< hard_ring
    return Wp, hard_mask
end

# -----------------------------------------------------------------------------
# prox_RED — the thesis pseudocode, optimize() = Adam
# -----------------------------------------------------------------------------

"""
    prox_RED(y; ρ, Wp, hard_mask, n, η, lr, n_iters, t_min, w)

Proximal operator of the RED-Diff regularizer at `y` (a DIM×DIM image in the
diffusion model's native ≈[-1,1] range):

    argmin_x  E_RED(x; T, w, n) + (ρ/2) ‖x - y‖²_P ,    P(u) = Wp .* u

Returns `(x, obj)` — the minimizer and the final objective value.
"""
function prox_RED(y::AbstractArray; ρ::Real, Wp::AbstractArray, hard_mask,
                  n::Int=N_TIME_SAMPLES, η::Real=ETA, lr::Real=LR, n_iters::Int=N_ITERS,
                  t_min::Real=T_MIN_REG, w=W_REG)
    ρf = Float32(ρ)
    ηf = Float32(η)
    x  = copy(Float32.(y))            # x₀ ← y
    x[hard_mask] .= y[hard_mask]
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)

    obj_best = Inf32
    x_best = copy(x)
    for iter in 1:n_iters
        E, ∇E = R_diff(x, T, n; t_min=t_min, w=w)          # error_RED(x).E , .∇E

        u    = x .- y
        Pu   = Wp .* u                                     # P(x - y)
        grad = ηf .* ∇E .+ ρf .* Pu                        # η·∇E_RED + ρ·P(x-y)

        opt_state, x = Optimisers.update!(opt_state, x, grad)
        x[hard_mask] .= y[hard_mask]                       # straight back-projection on the ring

        obj = E + 0.5f0 * ρf * sum(Wp .* (x .- y) .^ 2)    # E_RED + (ρ/2)‖x-y‖²_P
        if obj < obj_best
            obj_best = obj
            x_best .= x
        end
        if iter % 50 == 0 || iter == n_iters
            println("iter $iter/$n_iters  E_RED=$(round(E, digits=6))  obj=$(round(obj, digits=6))  ‖x-y‖=$(round(norm(x .- y), digits=4))")
        end
    end
    return x_best, obj_best
end

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

isempty(strip(IN_PATH)) &&
    error("postprocessing.jl: IN_PATH is not set — edit the script and give it an explicit path to the reconstruction image to post-process.")
isfile(IN_PATH) ||
    error("postprocessing.jl: IN_PATH does not point at an existing file: $(repr(IN_PATH))")

println("Post-processing: $IN_PATH")

y_raw = Float32.(Gray.(load(IN_PATH)))
if size(y_raw) != (DIM, DIM)
    y_raw = Float32.(imresize(y_raw, (DIM, DIM)))
end
y = normalize_image(y_raw)                     # [0,1] → ≈[-1,1] (diffusion-native)

Wp, hard_mask = boundary_projection(DIM)

x_post, obj = prox_RED(y; ρ=RHO, Wp=Wp, hard_mask=hard_mask)
println("final objective = $obj")

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

in_img   = Gray.(map(clamp01nan, y_raw))
post_img = Gray.(map(clamp01nan, denormalize_image(x_post)))
wp_img   = Gray.(Wp)                           # visualize the projection strength

plt = plot(
    plot(in_img,   title="input reconstruction", axis=false),
    plot(post_img, title="RED-Diff prox", axis=false),
    plot(wp_img,   title="boundary weight Wp", axis=false),
    layout=(1, 3), size=(1000, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "postproc_$timestamp.png"), post_img)
savefig(plt, joinpath(OUT_DIR, "postproc_comparison_$timestamp.png"))
println("saved to $OUT_DIR")
