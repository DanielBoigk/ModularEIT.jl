function ADMM_step!(opt::GalerkinOptState, prox_obj, prox_reg)
    x = opt.σ
    y = opt.y
    u = opt.u
    x = prox_obj(y - u)
    y = prox_reg(x + u)
    u .+= x
    u .-= y
    return x
end
function ADMM!(opt::GalerkinOptState, prox_obj, prox_reg, steps::Integer)
    for i in 1:steps
        ADMM_step!(opt, prox_obj, prox_reg)
    end
end
