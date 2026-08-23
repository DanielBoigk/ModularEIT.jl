using Pkg
Pkg.activate("../../../")
using Lux, Reactant, Enzyme, NNlib
using Optimisers, Random, Statistics, Images, FileIO
using LinearAlgebra, JLD2, ComponentArrays
using Dates

include("model.jl")   # UNet / unet_tinyimagenet64

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

normalize_image(img) = 2 .* (Float32.(img) .- 0.5)
denormalize_image(img) = (0.5f0 .* img) .+ 0.5f0

function forward_sample(x0, t, ᾱ)
    αbar = ᾱ(t)
    ε = randn(Float32, size(x0))
    xt = sqrt(αbar) .* x0 .+ sqrt(1 - αbar) .* ε
    return xt, ε
end

# model.jl's second input argument is named `noise_variances` after the
# Lux.jl DDIM tutorial it was ported from, but per trainimgnet.jl (both the
# CNN and U-Net variants: `t_trial = rand(...)` fed straight into the model,
# with the U-Net one explicitly commented "the per-sample scalar diffusion
# time is embedded internally") this codebase actually conditions directly
# on raw t, not on a transformed noise variance/rate. Kept as a named
# function in case that ever changes.
noise_variance(t) = t

"""
    sde(x, t)

Evaluate the trained noise predictor ε̂(x_t, t) for a single 64x64 grayscale
image `x` (accepts a (64,64), (64,64,1), or (64,64,1,1) array) at scalar
diffusion time `t ∈ [0, T]`. Returns a (64,64) `Array`.
"""
function sde(x::AbstractArray, t::Real)
    x4 = reshape(Float32.(x), dim, dim, 1, 1) |> dev
    nv = fill(Float32(noise_variance(t)), 1, 1, 1, 1) |> dev
    ε̂, _ = model((x4, nv), ps, st)
    return reshape(Array(ε̂ |> cdev), dim, dim)
end

"""
    R_diff(x, T, n; t_min=0.02f0, w=t -> 1.0f0)

RED-Diff regularizer for diffusion posterior sampling / Diff-PIR: draws `n`
independent `(tᵢ, εᵢ)` pairs with `tᵢ ~ Uniform(t_min, T)` (kept away from 0,
where the VP-SDE forward marginal degenerates) and `εᵢ ~ N(0,I)`,
forward-diffuses `x` to `x_{tᵢ} = √ᾱ(tᵢ) x + √(1-ᾱ(tᵢ)) εᵢ` for each, and
evaluates the noise predictor at every `x_{tᵢ}` in a single batched call
(this is the "parallelized" part — the n samples only differ along the batch
dimension the network already has, so they're one forward pass, not a loop).

RED-Diff treats the network's dependence on `x_t` as fixed (stop-gradient) —
that's what makes it cheap, no backprop through the U-Net is needed — so the
gradient contribution of sample `i` w.r.t. `x` is exactly
`w(tᵢ) √ᾱ(tᵢ) (ε̂(x_{tᵢ},tᵢ) - εᵢ)`, the plain chain-rule factor from
`x_{tᵢ} = √ᾱ(tᵢ) x + ...` times the (stopped) noise-prediction residual.

Returns `(err, grad)`:
- `err`  : the n noise-prediction residuals `ε̂(x_t,t) - ε`, size (64,64,1,n)
- `grad` : the RED-Diff gradient estimate w.r.t. `x`, size (64,64), averaged over the n samples
"""
function R_diff(x::AbstractArray, T::Real, n::Int; t_min::Real=0.02f0, w=t -> 1.0f0)
    x2 = Float32.(reshape(x, dim, dim))
    ts = t_min .+ (T - t_min) .* rand(Float32, n)
    ε = randn(Float32, dim, dim, 1, n)

    αbar = reshape(Float32.(ᾱ.(ts)), 1, 1, 1, n)
    x_batch = reshape(x2, dim, dim, 1, 1) .* sqrt.(αbar) .+ sqrt.(1 .- αbar) .* ε
    nv = reshape(Float32.(noise_variance.(ts)), 1, 1, 1, n)

    ε̂, _ = model((x_batch |> dev, nv |> dev), ps, st)
    err = Array(ε̂ |> cdev) .- ε

    weights = reshape(Float32.(w.(ts)), 1, 1, 1, n) .* sqrt.(αbar)
    grad = dropdims(sum(weights .* err; dims=4); dims=(3, 4)) ./ n

    return err, grad
end

