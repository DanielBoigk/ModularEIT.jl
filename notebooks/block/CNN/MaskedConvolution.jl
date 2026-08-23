using Lux, ConcreteStructs, Random, NNlib

# =============================================================================

# The trick: instead of passing `mask` only into the first layer (the way the
# sinusoidal time embedding gets concatenated once at conv_in in
# notebooks/UNet/model.jl), every layer in this network consumes and produces
# an `(x, mask)` tuple. That makes the running value of a plain `Chain`
# `(x, mask)` throughout, so:
#
#   - `MaskedConv` / `MaskedPool` / `MaskedUpsample` are "mask-native": they
#     read the mask (or resample it alongside x) and hand back an updated one.
#   - `Aux(layer)` lifts an ordinary, mask-oblivious layer (BatchNorm, a plain
#     Conv, an attention block, ...) into the same `(x, mask) -> (x, mask)`
#     contract by applying `layer` to `x` and forwarding `mask` untouched.
#
# Because every element of the chain agrees on this contract, they compose
# with a plain `Chain` — no `Parallel` routing required to keep the mask
# alive between layers that don't otherwise care about it.
# =============================================================================

# -----------------------------------------------------------------------------
# Aux: thread an opaque side-channel (here, the mask) through a layer that
# doesn't know about it.
# -----------------------------------------------------------------------------
@concrete struct Aux <: AbstractLuxWrapperLayer{:inner}
    inner
end

function (a::Aux)((x, mask), ps, st::NamedTuple)
    y, st_new = a.inner(x, ps, st)
    return (y, mask), st_new
end

# -----------------------------------------------------------------------------
# MaskedConv: exactly Conv, except the mask is concatenated on as an extra
# input channel before convolving, and the (unchanged, since padding keeps
# spatial size fixed) mask is passed on for the next masked layer to use.
# -----------------------------------------------------------------------------
@concrete struct MaskedConv <: AbstractLuxWrapperLayer{:conv}
    conv
end

function MaskedConv(
    k::NTuple{N,Integer}, (in_chs, out_chs)::Pair{<:Integer,<:Integer}, activation=identity;
    kwargs...
) where {N}
    return MaskedConv(Conv(k, (in_chs + 1) => out_chs, activation; kwargs...))
end

function (m::MaskedConv)((x, mask), ps, st::NamedTuple)
    y, st_new = m.conv(cat(x, mask; dims=Val(3)), ps, st)
    return (y, mask), st_new
end

# -----------------------------------------------------------------------------
# MaskedPool / MaskedUpsample: x gets the usual pooling/upsampling; the mask
# is resampled the same way spatially so it stays aligned with x. The mask
# uses MaxPool (rather than MeanPool) when downsampling, so a window counts as
# valid if any pixel inside it was valid — the mask should only ever grow
# stale, never shrink, purely from resampling.
# -----------------------------------------------------------------------------
@concrete struct MaskedPool <: AbstractLuxContainerLayer{(:pool, :maskpool)}
    pool
    maskpool
end

MaskedPool(window; kwargs...) = MaskedPool(MeanPool(window; kwargs...), MaxPool(window; kwargs...))

function (m::MaskedPool)((x, mask), ps, st::NamedTuple)
    y, st_pool = m.pool(x, ps.pool, st.pool)
    mask_new, st_maskpool = m.maskpool(mask, ps.maskpool, st.maskpool)
    return (y, mask_new), (; pool=st_pool, maskpool=st_maskpool)
end

@concrete struct MaskedUpsample <: AbstractLuxWrapperLayer{:up}
    up
end

MaskedUpsample(; kwargs...) = MaskedUpsample(Upsample(:nearest; kwargs...))

function (m::MaskedUpsample)((x, mask), ps, st::NamedTuple)
    y, st_new = m.up(x, ps, st)
    mask_new, _ = m.up(mask, ps, st)
    return (y, mask_new), st_new
end

# -----------------------------------------------------------------------------
# DropMask: at the output of the network, discard the mask and hand back a
# plain array again — the ordinary Lux contract everything downstream (loss
# functions, Training.single_train_step!, ...) expects.
# -----------------------------------------------------------------------------
struct DropMask <: AbstractLuxLayer end
(::DropMask)((x, _mask), ps, st::NamedTuple) = x, st
