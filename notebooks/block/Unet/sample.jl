using Pkg
Pkg.activate("../../../")
using Lux, Reactant, Enzyme, NNlib
using Optimisers, Random, Statistics, Images, FileIO
using LinearAlgebra, JLD2, ComponentArrays
using Dates

include("../model.jl")   # UNet / unet_tinyimagenet64

# =============================================================================
# Euler-Maruyama sampler for the VP-SDE reverse process: draws new 64x64
# grayscale TinyImageNet-style images from the U-Net trained by
# trainimgnet.jl. Loads the same "ps_latestvn.jld2" / "st_latestvn.jld2"
# checkpoint that script writes, so just run trainimgnet.jl first.
# =============================================================================

emb_dim = 32   # must match the embedding_dims the checkpoint was trained with
dim = 64

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev

model = unet_tinyimagenet64(; embedding_dims=emb_dim)

if isfile("ps_latestvn.jld2") && isfile("st_latestvn.jld2")
    @load "ps_latestvn.jld2" ps_cpu
    @load "st_latestvn.jld2" st_cpu
    ps = ps_cpu |> dev
    st = Lux.testmode(st_cpu) |> dev   # BatchNorm must use running stats, not batch stats, when sampling
    println("Loaded checkpoint from ps_latestvn.jld2 / st_latestvn.jld2.")
else
    @warn "No checkpoint found (ps_latestvn.jld2 / st_latestvn.jld2 not present in this directory) — " *
          "sampling from a randomly initialized model, so the output will be pure noise. " *
          "Run trainimgnet.jl first to get meaningful samples."
    ps, st = Lux.setup(Xoshiro(), model) |> dev
    st = Lux.testmode(st)
end

# --- VP-SDE noise schedule (must match the one trainimgnet.jl was trained with) ---
const βmin = 0.1f0
const βmax = 20.0f0
const T = 1.0f0

β(t) = βmin + (βmax - βmin) * t / T
ᾱ(t) = exp(-βmin * t - (βmax - βmin) / (2 * T) * t^2)

denormalize_image(img) = (0.5f0 .* img) .+ 0.5f0

"""
    euler_maruyama_sample(model, ps, st, dev; num_samples=1, num_steps=1000, rng=Random.default_rng())

Draws `num_samples` images from the trained VP-SDE reverse diffusion process
via the Euler-Maruyama discretization of the reverse-time SDE

    x_{t-Δt} = x_t + [½β(t)x_t - β(t)ε̂(x_t,t)/√(1-ᾱ(t))]·Δt + √(β(t)Δt)·z,   z ~ N(0, I)

starting from the prior `x_1 ~ N(0, I)` (the VP-SDE marginal at t=1 is ≈
standard normal) and integrating down to `t ≈ Δt`. Returns a
`(64, 64, 1, num_samples)` array of denormalized images clamped to `[0, 1]`,
ready to turn into `Gray.(...)` and save.
"""
function euler_maruyama_sample(
    model, ps, st, dev;
    num_samples::Int=1, num_steps::Int=1000, rng::AbstractRNG=Random.default_rng(),
)
    Δt = Float32(T / num_steps)
    xt = randn(rng, Float32, dim, dim, 1, num_samples)   # x_1 ~ N(0, I)

    # Reactant device arrays must go through a `@compile`d function — calling
    # the model directly on them (as one normally could on CPU/plain GPU
    # arrays) fails at run time. The input shapes are the same at every step
    # (only the noise-level *value* changes), so compiling once here and
    # reusing it for the whole trajectory is both correct and, incidentally,
    # far faster than recompiling per step.
    xt_trial = xt |> dev
    t_trial = fill(T, 1, 1, 1, num_samples) |> dev
    model_compiled = @compile model((xt_trial, t_trial), ps, st)

    for step in 1:num_steps
        t = T - (step - 1) * Δt
        β_t = Float32(β(t))
        ᾱ_t = Float32(ᾱ(t))

        t_arr = fill(Float32(t), 1, 1, 1, num_samples)
        ε_pred, st = model_compiled((xt, t_arr) |> dev, ps, st)
        ε_pred = ε_pred |> cdev

        drift = 0.5f0 .* β_t .* xt .- (β_t / sqrt(max(1.0f0 - ᾱ_t, 1.0f-5))) .* ε_pred
        z = randn(rng, Float32, size(xt))
        xt = xt .+ drift .* Δt .+ sqrt(β_t * Δt) .* z
    end

    return clamp.(denormalize_image(xt), 0.0f0, 1.0f0)
end

rng = Xoshiro(1)
num_samples = 8
num_steps = 1000

samples = euler_maruyama_sample(model, ps, st, dev; num_samples, num_steps, rng)

mkpath("samples")
timestamp = now()
for i in 1:num_samples
    img = Gray.(samples[:, :, 1, i])
    save("samples/sample_$(i)_$(timestamp).png", img)
end
println("Saved $num_samples sample(s) to samples/")
