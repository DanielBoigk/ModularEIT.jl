using Lux, Reactant, Enzyme, NNlib
using Optimisers, Random, Statistics, Images
using LinearAlgebra, Images, JLD2, ComponentArrays
using Dates, Plots #, UnicodePlots

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = cdev
rng = Xoshiro()

@load "model/ps_latestvn.jld2" ps_cpu 
@load "model/st_latestvn.jld2" st_cpu
ps = ps_cpu |> dev
st = st_cpu |> dev

include("model.jl")
include("helperfuncs.jl")

function wrap_model(st, ps, model, emb_dim)
    return (x,t) -> begin
        x_dim, y_dim = size(x)
        out = ones(Float32, x_dim, y_dim, 2+emb_dim)
        emb = reshape(sinusoidal_embedding(Float32(t), emb_dim), (1,1,emb_dim))
        out[:, :, 1] = x
        out[:, :, 3:end] .= emb
        return reshape(model(out |> dev, ps,st)[1],(x_dim,y_dim))
    end
end

sde = wrap_model(st,ps,model, emb_dim)