# =============================================================================
# postprocessing_unconstrained.jl — let RED-Diff run on an existing 64x64
# grayscale EIT reconstruction with NO data-consistency term and NO boundary
# projection at all.
#
# This is postprocessing.jl with the P-weighted prox term deleted entirely:
# no ρ, no Wp, no hard ring, no pull back toward `y` whatsoever. What's left
# is a pure minimization of the diffusion prior's score-matching error
#
#     x* = argmin_x  E_RED(x) = E_{t,ε}[ ω_t ‖ε̂_θ(x_t,t) - ε‖² ]
#
# i.e. `R_diff` from initialize.jl and nothing else. Starting from x₀ = y, the
# image is free to drift anywhere the diffusion prior likes — the input only
# supplies the optimizer's starting point, it never anchors the result. Useful
# as a "what does the prior alone think a plausible image looks like, given
# this starting point" baseline to compare against the constrained prox in
# postprocessing.jl.
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
IN_PATH = "lbfgsb_25.png"

# --- pure RED-Diff prior descent, no guidance --------------------------------
N_ITERS     = 100      # Adam iterations
LR          = 0.01f0   # Adam learning rate over the image `x`
N_TIME_SAMPLES = 16    # (t, ε) probes per iteration for E_RED — one batched network call
T_MIN_REG   = 0.02f0   # matches R_diff's default; kept away from 0 (degenerate forward marginal)
W_REG       = t -> 1.0f0   # ω(t) — per-t weighting inside E_RED; constant matches R_diff's default

OUT_DIR = joinpath(@__DIR__, "PostProc")
mkpath(OUT_DIR)

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# unconstrained_RED — minimize E_RED(x) alone, starting from x₀ = y
# -----------------------------------------------------------------------------

"""
    unconstrained_RED(y; n, lr, n_iters, t_min, w)

Runs Adam on `E_RED(x)` alone (no data-consistency term, no boundary
projection) starting from `x₀ = y`. Returns `(x, E)` — the final iterate and
its final E_RED value. Since there is nothing pulling `x` back toward `y`,
this simply reports the *last* iterate rather than tracking a "best" one
(there is no other objective term to rank iterates against).
"""
function unconstrained_RED(y::AbstractArray; n::Int=N_TIME_SAMPLES, lr::Real=LR,
                           n_iters::Int=N_ITERS, t_min::Real=T_MIN_REG, w=W_REG)
    x = copy(Float32.(y))              # x₀ ← y (starting point only, no anchor)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)

    E = Inf32
    for iter in 1:n_iters
        E, grad = R_diff(x, T, n; t_min=t_min, w=w)   # error_RED(x).E , .∇E
        opt_state, x = Optimisers.update!(opt_state, x, grad)
        if iter % 50 == 0 || iter == n_iters
            println("iter $iter/$n_iters  E_RED=$(round(E, digits=6))  ‖x-y‖=$(round(norm(x .- y), digits=4))")
        end
    end
    return x, E
end

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

isempty(strip(IN_PATH)) &&
    error("postprocessing_unconstrained.jl: IN_PATH is not set — edit the script and give it an explicit path to the reconstruction image to post-process.")
isfile(IN_PATH) ||
    error("postprocessing_unconstrained.jl: IN_PATH does not point at an existing file: $(repr(IN_PATH))")

println("Post-processing (unconstrained): $IN_PATH")

y_raw = Float32.(Gray.(load(IN_PATH)))
if size(y_raw) != (DIM, DIM)
    y_raw = Float32.(imresize(y_raw, (DIM, DIM)))
end
y = normalize_image(y_raw)                     # [0,1] → ≈[-1,1] (diffusion-native)

x_post, E = unconstrained_RED(y)
println("final E_RED = $E")

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

in_img   = Gray.(map(clamp01nan, y_raw))
post_img = Gray.(map(clamp01nan, denormalize_image(x_post)))

plt = plot(
    plot(in_img,   title="input reconstruction", axis=false),
    plot(post_img, title="RED-Diff (unconstrained)", axis=false),
    layout=(1, 2), size=(700, 350),
)
display(plt)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
save(joinpath(OUT_DIR, "postproc_unconstrained_$timestamp.png"), post_img)
savefig(plt, joinpath(OUT_DIR, "postproc_unconstrained_comparison_$timestamp.png"))
println("saved to $OUT_DIR")
