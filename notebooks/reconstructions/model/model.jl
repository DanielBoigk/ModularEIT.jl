function sinusoidal_embedding(t::Float32, embedding_dim::Int, max_positions::Int=10000)
    half_dim = embedding_dim ÷ 2
    emb_scale = log(Float32(max_positions)) / (half_dim - 1)
    emb = exp.(-emb_scale .* Float32.(0:half_dim-1))
    emb = t .* emb  # shape: (half_dim,)
    emb = vcat(sin.(emb), cos.(emb))  # shape: (embedding_dim,)
    return Float32.(emb)
end


spatial_conv_layer = Conv((17, 17), 2 => 256; pad = 8)
struct ConvFirstTwo{C}<: Lux.AbstractLuxWrapperLayer{:conv}
    conv::C
    length::Int
end
# 2. Explicitly define how state is generated
function Lux.initialstates(rng::AbstractRNG, m::ConvFirstTwo)
    return (conv = Lux.initialstates(rng, m.conv),)
end

# 3. Explicitly define how parameters are generated
function Lux.initialparameters(rng::AbstractRNG, m::ConvFirstTwo)
    return (conv = Lux.initialparameters(rng, m.conv),)
end
function (m::ConvFirstTwo)(x, ps, st) 
    # Use standard slicing that maps cleanly to XLA operations
    x1 = x[:, :, 1:2, :]
    x2 = x[:, :, 3:end, :] # 'end' is perfectly fine and statically resolved by XLA
    y1, st_conv = m.conv(x1, ps.conv, st.conv)
    return cat(y1, x2; dims=3), (conv = st_conv,)
end


first_conv = ConvFirstTwo(spatial_conv_layer, 34)

emb_dim = 32 
model = Chain(
    #Conv((17, 17), 2 => 256; pad = 8),
    first_conv,
    #last_conv,
    SkipConnection(
        Chain(
            Conv((1, 1), 256+emb_dim => 512, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                ),
                +
            ),
            Conv((1, 1), 512 => 256, gelu),
            Conv((1, 1), 256 => 256+emb_dim, gelu),
        ),
        +
    ),
    Conv((1, 1), 256+emb_dim => 128, gelu),
    Conv((1, 1), 128 => 64, gelu),
    Conv((1, 1), 64 => 1)
)
