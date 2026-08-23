
using Lux, Reactant, Enzyme, MLUtils, NNlib
using Optimisers, Random, Statistics, Images
using LinearAlgebra, Images, JLD2, ComponentArrays
using Dates, Plots, UnicodePlots


include("model.jl")     # defines LocalScoreNet (see local_score_net_64)

batch_size = 128
dim = 64

# Width of the internal sinusoidal embedding of the diffusion time. Purely an
# architectural hyperparameter of LocalScoreNet (embedded and FiLM-conditioned
# internally) — it no longer needs to match anything about the input tensor's
# channel count, since the image stays a single grayscale channel throughout.
emb_dim = 64

load_model = true
test_model = true

model = local_score_net_64(; embedding_dims=emb_dim)

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev
rng = Xoshiro()
opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.NAdam(2.0f-4))

if load_model && isfile("ps_latestvn.jld2") && isfile("st_latestvn.jld2")
    @load "ps_latestvn.jld2" ps_cpu
    @load "st_latestvn.jld2" st_cpu
    ps = ps_cpu |> dev
    st = st_cpu |> dev
else
    ps, st = Lux.setup(rng, model) |> dev
end

train_state = Training.TrainState(model, ps, st, opt)

train!(data, train_state) = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    data,
    train_state;
    return_gradients=Val(false),
)

if test_model
    # Model input is a (noisy_image, t) tuple, not a single concatenated
    # tensor: the image keeps its 1 (grayscale) channel, and the per-sample
    # scalar diffusion time is embedded internally by LocalScoreNet.
    x_trial = randn(Float32, dim, dim, 1, batch_size) |> dev
    t_trial = rand(Float32, batch_size) |> dev
    y_trial = randn(Float32, dim, dim, 1, batch_size) |> dev
    data_trial = ((x_trial, t_trial), y_trial)

    model_compiled = @compile model((x_trial, t_trial), ps, st)
    y_pred, st = model_compiled((x_trial, t_trial), ps, st)
    println("Model successfully compiled!")

end

# Hyperparameters for the Variance Preserving (VP) SDE
const βmin = 0.1
const βmax = 20.0
T = 1

function normalize_image(img)
    return 2 .* (Float32.(img) .- 0.5)
end

function denormalize_image(img)
    return (0.5 .* img) .+ 0.5
end

function forward_sample(x0, t, ᾱ)
    αbar = ᾱ(t)
    ε = randn(Float32, size(x0))
    xt = sqrt(αbar) .* x0 .+ sqrt(1 - αbar) .* ε
    return xt, ε
end

β(t) = βmin + (βmax - βmin) * t / T
ᾱ(t) = exp(-βmin * t - (βmax - βmin) / (2 * T) * t^2)

forward(x, t) = forward_sample(x, t, ᾱ)




