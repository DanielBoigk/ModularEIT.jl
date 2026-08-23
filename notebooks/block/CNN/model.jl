using Lux, ConcreteStructs, Random, NNlib, Statistics

include("MaskedConvolution.jl")

# =============================================================================
# Local score/noise predictor for the VP-SDE diffusion model.
#
# Unlike the UNet in notebooks/UNet/model.jl, this network never changes
# spatial resolution: no pooling, no upsampling, no bottleneck. Every layer is
# a small (3x3) MaskedConv, so a pixel's receptive field only grows by a fixed
# amount per block and stays local for the whole depth of the network — it
# cannot "unfold" into a globally-mixing architecture the way a U-Net's
# downsample/upsample path (or an attention block) can. That's a deliberate
# constraint, not an oversight: the network is meant to learn a local texture
# pattern, not long-range structure.
#
# Every MaskedResBlock carries an all-ones validity mask through the
# MaskedConv/Aux machinery from MaskedConvolution.jl. There's no missing data
# here, so the mask's job is narrower: because Conv zero-pads at the image
# border, threading a constant mask channel alongside x gives every local
# conv a cheap, explicit signal for "am I near the border" (the mask channel
# itself gets zero-padded, so its receptive field looks different near an
# edge) without adding any long-range mixing.
#
# Time conditioning is proper FiLM: `sinusoidal_embedding(t, ...)` maps the
# scalar diffusion time to a vector, a per-block Dense projects that to a
# per-channel (scale, shift), and the scale/shift is applied as a pointwise
# affine after the first conv of each residual block. That replaces the old
# approach of broadcasting the embedding out to a full-resolution tensor and
# concatenating it as input channels, which wastes channels on a spatially
# constant signal and forces local convolutions to "learn" what is really a
# global multiply-add.
#
# Note on normalization: GroupNorm/BatchNorm/InstanceNorm all reduce over the
# *entire spatial extent* to compute their statistics, which secretly makes
# every output pixel depend on every input pixel — exactly the kind of
# domain-wide message passing this network is meant to avoid, even though it
# doesn't look like one. `PixelNorm` below normalizes each pixel's channel
# vector independently, so no statistic is ever shared across spatial
# locations.
# =============================================================================

# -----------------------------------------------------------------------------
# PixelNorm: normalizes across channels independently at every (h, w, batch)
# location — unlike GroupNorm/BatchNorm/InstanceNorm, it never reduces over
# space, so it introduces no cross-pixel dependency.
# -----------------------------------------------------------------------------
@concrete struct PixelNorm <: AbstractLuxLayer
    channels::Int
    ϵ::Float32
end

PixelNorm(channels::Int; ϵ=1.0f-5) = PixelNorm(channels, Float32(ϵ))

Lux.initialparameters(::AbstractRNG, l::PixelNorm) =
    (scale=ones(Float32, l.channels), bias=zeros(Float32, l.channels))
Lux.initialstates(::AbstractRNG, ::PixelNorm) = NamedTuple()

function (l::PixelNorm)(x::AbstractArray{T,4}, ps, st::NamedTuple) where {T}
    μ = mean(x; dims=3)
    σ2 = mean(abs2, x .- μ; dims=3)
    x̂ = (x .- μ) ./ sqrt.(σ2 .+ T(l.ϵ))
    γ = reshape(ps.scale, 1, 1, :, 1)
    β = reshape(ps.bias, 1, 1, :, 1)
    return γ .* x̂ .+ β, st
end

# -----------------------------------------------------------------------------
# Sinusoidal embedding of the (scalar, per-sample) diffusion time.
# Same construction as the scalar version in helperfuncs.jl, vectorized over
# the batch so it can live inside the model instead of the dataloader.
# -----------------------------------------------------------------------------
function sinusoidal_embedding(t::AbstractVector{T}, embedding_dim::Int, max_positions=10000) where {T}
    half_dim = embedding_dim ÷ 2
    emb_scale = log(T(max_positions)) / (half_dim - 1)
    freqs = exp.(-emb_scale .* T.(0:half_dim-1))   # (half_dim,)
    args = reshape(t, 1, :) .* freqs               # (half_dim, batch)
    return vcat(sin.(args), cos.(args))            # (embedding_dim, batch)
end

# -----------------------------------------------------------------------------
# MaskedResBlock: PixelNorm -> swish -> MaskedConv(3x3) -> FiLM(temb) ->
# PixelNorm -> swish -> MaskedConv(3x3) -> residual add. Channel-count
# agnostic; the skip path is a 1x1 MaskedConv when channels change, otherwise
# a no-op (still lifted into the (x, mask) contract via Aux).
# -----------------------------------------------------------------------------
@concrete struct MaskedResBlock <: AbstractLuxContainerLayer{(:norm1, :conv1, :film, :norm2, :conv2, :skip)}
    norm1
    conv1
    film
    norm2
    conv2
    skip
end

function MaskedResBlock(in_chs::Int, out_chs::Int, embedding_dims::Int)
    return MaskedResBlock(
        Aux(PixelNorm(in_chs)),
        MaskedConv((3, 3), in_chs => out_chs; pad=SamePad()),
        Dense(embedding_dims => 2out_chs),
        Aux(PixelNorm(out_chs)),
        MaskedConv((3, 3), out_chs => out_chs; pad=SamePad()),
        in_chs == out_chs ? Aux(NoOpLayer()) : MaskedConv((1, 1), in_chs => out_chs),
    )
end

function (blk::MaskedResBlock)((x, mask), temb, ps, st::NamedTuple)
    (h, mask1), st_norm1 = blk.norm1((x, mask), ps.norm1, st.norm1)
    h = swish.(h)
    (h, mask1), st_conv1 = blk.conv1((h, mask1), ps.conv1, st.conv1)

    γβ, st_film = blk.film(temb, ps.film, st.film)   # (2 * out_chs, batch)
    out_chs = size(h, 3)
    γ = reshape(view(γβ, 1:out_chs, :), 1, 1, out_chs, :)
    β = reshape(view(γβ, (out_chs+1):(2out_chs), :), 1, 1, out_chs, :)
    h = h .* (1 .+ γ) .+ β

    (h, mask1), st_norm2 = blk.norm2((h, mask1), ps.norm2, st.norm2)
    h = swish.(h)
    (h, mask1), st_conv2 = blk.conv2((h, mask1), ps.conv2, st.conv2)

    (r, mask2), st_skip = blk.skip((x, mask), ps.skip, st.skip)

    return (r .+ h, mask2),
        (; norm1=st_norm1, conv1=st_conv1, film=st_film, norm2=st_norm2, conv2=st_conv2, skip=st_skip)
end

# -----------------------------------------------------------------------------
# Full network: a flat stack of MaskedResBlocks at constant resolution.
# -----------------------------------------------------------------------------
@concrete struct LocalScoreNet <: AbstractLuxContainerLayer{(:blocks, :conv_out)}
    blocks
    conv_out
    embedding_dims::Int
    max_positions::Int
end

"""
    LocalScoreNet(; in_channels=1, channels=(64, 64, 64, 64), embedding_dims=64,
                  max_positions=10000)

Noise predictor `ε̂(x_t, t)` built entirely from local (3x3) MaskedConv
residual blocks at a single, unchanging resolution — no downsampling,
upsampling, or attention, so the receptive field stays bounded regardless of
image size: each block adds a radius of 2 pixels (two 3x3 convs), so
`length(channels)` blocks give a total receptive-field radius of
`2 * length(channels)`. `channels` sets the width of each successive block.
"""
function LocalScoreNet(;
    in_channels::Int=1,
    channels=(64, 64, 64, 64),
    embedding_dims::Int=64,
    max_positions::Int=10000,
)
    blocks = Tuple(
        MaskedResBlock(i == 1 ? in_channels : channels[i-1], channels[i], embedding_dims)
        for i in eachindex(channels)
    )
    conv_out = MaskedConv((1, 1), channels[end] => in_channels; init_weight=zeros32)
    return LocalScoreNet(blocks, conv_out, embedding_dims, max_positions)
end

function (m::LocalScoreNet)((x, t), ps, st::NamedTuple)
    # A constant "all valid" mask channel, built non-mutating (fill! breaks
    # Zygote) and derived from x via broadcast so it keeps x's array type
    # (CuArray, ConcreteRArray, ...) instead of always allocating a CPU Array.
    mask = one(eltype(x)) .+ zero.(view(x, :, :, 1:1, :))
    temb = sinusoidal_embedding(t, m.embedding_dims, m.max_positions)

    y, msk = x, mask
    st_blocks = ()
    for i in eachindex(m.blocks)
        (y, msk), st_new = m.blocks[i]((y, msk), temb, ps.blocks[i], st.blocks[i])
        st_blocks = (st_blocks..., st_new)
    end

    (y, _), st_conv_out = m.conv_out((y, msk), ps.conv_out, st.conv_out)

    return y, (; blocks=st_blocks, conv_out=st_conv_out)
end

"""
    local_score_net_64(; kwargs...)

Default local score network sized for 64x64 grayscale images.
"""
function local_score_net_64(; embedding_dims=64, kwargs...)
    return LocalScoreNet(; in_channels=1, channels=(64, 64, 64, 64), embedding_dims, kwargs...)
end

emb_dim = 64
model = local_score_net_64(; embedding_dims=emb_dim)
