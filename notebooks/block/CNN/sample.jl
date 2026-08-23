using Pkg
Pkg.activate("../../../")
using Lux, Reactant, Enzyme, NNlib
using Optimisers, Random, Statistics, Images, FileIO
using LinearAlgebra, JLD2, ComponentArrays
using Dates

include("model.jl")   # LocalScoreNet / local_score_net_64

# =============================================================================
# Euler-Maruyama sampler for the VP-SDE reverse process: draws new 64x64
# grayscale TinyImageNet-style images from the local score network trained by
# trainimgnet.jl. Loads the same "ps_latestvn.jld2" / "st_latestvn.jld2"
# checkpoint that script writes, so just run trainimgnet.jl first.
# =============================================================================

emb_dim = 64   # must match the embedding_dims the checkpoint was trained with
dim = 64

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev

model = local_score_net_64(; embedding_dims=emb_dim)

if isfile("ps_latestvn.jld2") && isfile("st_latestvn.jld2")
    @load "ps_latestvn.jld2" ps_cpu
    @load "st_latestvn.jld2" st_cpu
    ps = ps_cpu |> dev
    st = Lux.testmode(st_cpu) |> dev
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

# LocalScoreNet's call signature is ((x, t), ps, st) — unlike the U-Net,
# there's no separate mask argument to pass: model.jl builds an all-ones
# "every pixel valid" mask internally from x's shape (see the `mask = ...`
# line in LocalScoreNet's functor) and threads it through the MaskedConv
# layers itself. That mask machinery exists so this architecture can later
# handle non-rectangular domains (a real, non-constant validity mask), but
# for a plain rectangular 64x64 image there's nothing extra to supply here.
# `t` itself must be a plain Vector (length = batch size), not a (1,1,1,B)
# array like the U-Net's `noise_variances` — sinusoidal_embedding here takes
# an AbstractVector directly.

"""
    sde(x, t)

Evaluate the trained noise predictor ε̂(x_t, t) for a single 64x64 grayscale
image `x` (accepts a (64,64), (64,64,1), or (64,64,1,1) array) at scalar
diffusion time `t ∈ [0, T]`. Returns a (64,64) `Array`.
"""
function sde(x::AbstractArray, t::Real)
    x4 = reshape(Float32.(x), dim, dim, 1, 1) |> dev
    tv = Float32[t] |> dev
    ε̂, _ = model((x4, tv), ps, st)
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
that's what makes it cheap, no backprop through the network is needed — so
the gradient contribution of sample `i` w.r.t. `x` is exactly
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

    ε̂, _ = model((x_batch |> dev, ts |> dev), ps, st)
    err = Array(ε̂ |> cdev) .- ε

    weights = reshape(Float32.(w.(ts)), 1, 1, 1, n) .* sqrt.(αbar)
    grad = dropdims(sum(weights .* err; dims=4); dims=(3, 4)) ./ n

    return err, grad
end

