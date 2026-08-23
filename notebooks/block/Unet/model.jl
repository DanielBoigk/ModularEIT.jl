using Lux, ConcreteStructs, Random, NNlib

# =============================================================================
# U-Net noise predictor for the VP-SDE diffusion model.
#
# This is the residual/skip-connection U-Net from the Lux.jl DDIM tutorial,
# ported to grayscale 64x64 TinyImageNet and extended with a self-attention
# block at the bottleneck (the lowest spatial resolution), which is the
# standard place to add attention in diffusion U-Nets (DDPM/ADM): the
# sequence length (H*W) is small there, so full attention is cheap, while it
# lets every pixel attend to every other pixel — something the local 3x3
# convolutions elsewhere in the network cannot do.
# =============================================================================

# -----------------------------------------------------------------------------
# Sinusoidal embedding of the (scalar, per-sample) diffusion time / noise level
# -----------------------------------------------------------------------------
function sinusoidal_embedding(
    x::AbstractArray{T,4}, min_freq, max_freq, embedding_dims::Int
) where {T}
    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    lower, upper = T(log(min_freq)), T(log(max_freq))
    n = embedding_dims ÷ 2
    freqs = exp.(reshape(range(lower, upper; length=n), 1, 1, n, 1))
    x_ = 2 .* x .* freqs
    return cat(sinpi.(x_), cospi.(x_); dims=Val(3))
end

# -----------------------------------------------------------------------------
# Self-attention block: GroupNorm -> multi-head self-attention -> residual add
# -----------------------------------------------------------------------------
@concrete struct AttentionBlock <: AbstractLuxContainerLayer{(:norm, :mha)}
    norm
    mha
end

function AttentionBlock(channels::Int; nheads=4)
    return AttentionBlock(
        GroupNorm(channels, min(32, channels)), MultiHeadAttention(channels; nheads)
    )
end

function (a::AttentionBlock)(x::AbstractArray{T,4}, ps, st) where {T}
    h, w, c, b = size(x)
    y, st_norm = a.norm(x, ps.norm, st.norm)
    y = reshape(y, h * w, c, b)            # (seq, channels, batch)
    y = permutedims(y, (2, 1, 3))          # -> (channels, seq, batch), as MultiHeadAttention expects
    (attn_out, _), st_mha = a.mha(y, ps.mha, st.mha)
    attn_out = permutedims(attn_out, (2, 1, 3))
    attn_out = reshape(attn_out, h, w, c, b)
    return x .+ attn_out, (; norm=st_norm, mha=st_mha)   # residual
end

# -----------------------------------------------------------------------------
# Residual convolution block (channel-count agnostic)
# -----------------------------------------------------------------------------
@concrete struct ResidualBlock <: AbstractLuxWrapperLayer{:layer}
    layer
end

function ResidualBlock(in_channels::Int, out_channels::Int)
    return ResidualBlock(
        Parallel(
            +,
            if in_channels == out_channels
                NoOpLayer()
            else
                Conv((1, 1), in_channels => out_channels; pad=SamePad())
            end,
            Chain(
                BatchNorm(in_channels; affine=false),
                Conv((3, 3), in_channels => out_channels, swish; pad=SamePad()),
                Conv((3, 3), out_channels => out_channels; pad=SamePad()),
            ),
        ),
    )
end

# -----------------------------------------------------------------------------
# Down/up-sampling stages (channel-count agnostic)
# -----------------------------------------------------------------------------
@concrete struct DownsampleBlock <: AbstractLuxContainerLayer{(:residual_blocks, :pool)}
    residual_blocks
    pool
end

function DownsampleBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    residual_blocks = Tuple([
        ResidualBlock(ifelse(i == 1, in_channels, out_channels), out_channels) for
        i in 1:block_depth
    ])
    return DownsampleBlock(residual_blocks, MeanPool((2, 2)))
end

function (d::DownsampleBlock)(x::AbstractArray, ps, st::NamedTuple)
    skips = (x,)
    st_residual_blocks = ()
    for i in eachindex(d.residual_blocks)
        y, st_new = d.residual_blocks[i](
            last(skips), ps.residual_blocks[i], st.residual_blocks[i]
        )
        skips = (skips..., y)
        st_residual_blocks = (st_residual_blocks..., st_new)
    end
    y, st_pool = d.pool(last(skips), ps.pool, st.pool)
    return (y, skips), (; residual_blocks=st_residual_blocks, pool=st_pool)
end

@concrete struct UpsampleBlock <: AbstractLuxContainerLayer{(:residual_blocks, :upsample)}
    residual_blocks
    upsample
end

function UpsampleBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    residual_blocks = Tuple([
        ResidualBlock(
            ifelse(i == 1, in_channels + out_channels, out_channels * 2), out_channels
        ) for i in 1:block_depth
    ])
    return UpsampleBlock(residual_blocks, Upsample(:bilinear; scale=2))
end

function (u::UpsampleBlock)((x, skips), ps, st::NamedTuple)
    x, st_upsample = u.upsample(x, ps.upsample, st.upsample)
    y, st_residual_blocks = x, ()
    for i in eachindex(u.residual_blocks)
        y, st_new = u.residual_blocks[i](
            cat(y, skips[end - i + 1]; dims=Val(3)),
            ps.residual_blocks[i],
            st.residual_blocks[i],
        )
        st_residual_blocks = (st_residual_blocks..., st_new)
    end
    return y, (; residual_blocks=st_residual_blocks, upsample=st_upsample)
end

# -----------------------------------------------------------------------------
# Full U-Net
# -----------------------------------------------------------------------------
@concrete struct UNet <: AbstractLuxContainerLayer{(
    :conv_in, :conv_out, :down_blocks, :bottleneck, :up_blocks, :upsample
)}
    upsample
    conv_in
    conv_out
    down_blocks
    bottleneck
    up_blocks
    min_freq
    max_freq
    embedding_dims
end

"""
    UNet(image_size; channels, block_depth, min_freq, max_freq, embedding_dims,
         in_channels, use_attention, attn_nheads)

Residual U-Net noise predictor `ε̂(x_t, t)` for the VP-SDE diffusion model.

`channels` sets the channel width at each resolution stage; the image is
downsampled by 2x at every stage after the first, so with the default 4
stages a 64x64 input bottlenecks at 8x8. When `use_attention` is true (the
default) a [`AttentionBlock`](@ref) is inserted at that bottleneck, between
the first and remaining residual blocks — global self-attention is only
applied where the spatial resolution is small enough to be cheap.
"""
function UNet(image_size::Dims{2};
        channels=[32, 64, 96, 128], block_depth=2,
        min_freq=1.0f0, max_freq=1000.0f0, embedding_dims=32, in_channels=1,
        use_attention::Bool=true, attn_nheads::Int=4)
    upsample = Upsample(:nearest; size=image_size)
    conv_in  = Conv((1, 1), in_channels => channels[1])
    conv_out = Conv((1, 1), channels[1] => in_channels; init_weight=zeros32)

    channel_input = embedding_dims + channels[1]
    down_blocks = Tuple([
        DownsampleBlock(i == 1 ? channel_input : channels[i - 1], channels[i], block_depth)
        for i in 1:(length(channels) - 1)
    ])

    bottleneck_layers = Any[ResidualBlock(channels[end - 1], channels[end])]
    use_attention && push!(bottleneck_layers, AttentionBlock(channels[end]; nheads=attn_nheads))
    for _ in 2:block_depth
        push!(bottleneck_layers, ResidualBlock(channels[end], channels[end]))
    end
    bottleneck = Chain(bottleneck_layers...)

    rev_channels = reverse(channels)   # don't mutate the caller's `channels`
    up_blocks = Tuple([
        UpsampleBlock(in_chs, out_chs, block_depth) for
        (in_chs, out_chs) in zip(rev_channels[1:(end - 1)], rev_channels[2:end])
    ])

    return UNet(
        upsample,
        conv_in,
        conv_out,
        down_blocks,
        bottleneck,
        up_blocks,
        min_freq,
        max_freq,
        embedding_dims,
    )
end

function (u::UNet)((noisy_images, noise_variances), ps, st::NamedTuple)
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)

    emb, st_upsample = u.upsample(
        sinusoidal_embedding(noise_variances, u.min_freq, u.max_freq, u.embedding_dims),
        ps.upsample,
        st.upsample,
    )
    tmp, st_conv_in = u.conv_in(noisy_images, ps.conv_in, st.conv_in)
    x = cat(tmp, emb; dims=Val(3))

    skips_at_each_stage = ()
    st_down_blocks = ()
    for i in eachindex(u.down_blocks)
        (x, skips), st_new = u.down_blocks[i](x, ps.down_blocks[i], st.down_blocks[i])
        skips_at_each_stage = (skips_at_each_stage..., skips)
        st_down_blocks = (st_down_blocks..., st_new)
    end

    x, st_bottleneck = u.bottleneck(x, ps.bottleneck, st.bottleneck)

    st_up_blocks = ()
    for i in eachindex(u.up_blocks)
        x, st_new = u.up_blocks[i](
            (x, skips_at_each_stage[end - i + 1]), ps.up_blocks[i], st.up_blocks[i]
        )
        st_up_blocks = (st_up_blocks..., st_new)
    end

    x, st_conv_out = u.conv_out(x, ps.conv_out, st.conv_out)

    return (
        x,
        (;
            conv_in=st_conv_in,
            conv_out=st_conv_out,
            down_blocks=st_down_blocks,
            bottleneck=st_bottleneck,
            up_blocks=st_up_blocks,
            upsample=st_upsample,
        ),
    )
end

"""
    unet_tinyimagenet64(; kwargs...)

Default U-Net sized for 64x64 grayscale TinyImageNet: 4 resolution stages
(64 -> 32 -> 16 -> 8), a self-attention bottleneck at 8x8/128 channels, and
a single input/output channel.
"""
function unet_tinyimagenet64(; embedding_dims=32, kwargs...)
    return UNet(
        (64, 64);
        channels=[32, 64, 96, 128], block_depth=2, in_channels=1, embedding_dims, kwargs...
    )
end
