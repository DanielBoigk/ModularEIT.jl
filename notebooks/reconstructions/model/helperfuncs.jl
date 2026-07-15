const βmin = 0.1
const βmax = 20.0
T = 1

function normalize_image(img)
    out= 2 .* (Float32.(img) .- 0.5)
    return out
end

function denormalize_image(img)
    out = (0.5 .* img) .+ 0.5
    return out
end

function forward_sample(x0, t, ᾱ)
    αbar = ᾱ(t)
    ε = randn(size(x0))
    xt = sqrt(αbar) .* x0 .+ sqrt(1 - αbar) .* ε
    return xt, ε
end


β(t) = βmin + (βmax-βmin)*t/T
ᾱ(t) = exp(-βmin*t - (βmax-βmin)/(2*T) * t^2)

forward(x,t) = forward_sample(x,t,ᾱ)