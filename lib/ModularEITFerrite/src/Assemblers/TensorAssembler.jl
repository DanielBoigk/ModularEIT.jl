# Conductivity tensor of a Ferrite discretization (the type and its kernels are in ModularEIT).

function ConductivityTensor(disc::FerriteDiscretization; pattern = _u_pattern(disc), to_device = identity)
    return ConductivityTensor(pattern, _conductivity_tensor(disc, pattern); to_device)
end

# COO assembly of T on the unconstrained pattern of dh_u: for every cell the local tensor
# ∫ ψₐ ∇φᵢ⋅∇φⱼ, scattered to (nz index of (i, j), σ dof a); duplicates are summed by `sparse`.
# Then T = R T_full with the sparse map R from the stored entries (i, j) of the full pattern to
# the entries (p, q) of `pattern`, weighted by the conformity constraints C[i, p] C[j, q]
# (R is a 0/1 selection on conforming grids).
function _conductivity_tensor(disc::FerriteDiscretization, pattern::SparseMatrixCSC)
    full = allocate_matrix(disc.dh_u)
    T_full = _conductivity_tensor_full(disc, full)
    Ct = disc.C_u === nothing ? nothing : sparse(disc.C_u')     # column i of Cᵀ = row i of C
    Ir, Jr, Vr = Int[], Int[], Float64[]
    for j in 1:size(full, 2), k in nzrange(full, j)
        i = full.rowval[k]
        if Ct === nothing
            push!(Ir, _nz_index(pattern, i, j)); push!(Jr, k); push!(Vr, 1.0)
        else
            for a in nzrange(Ct, i), b in nzrange(Ct, j)
                push!(Ir, _nz_index(pattern, Ct.rowval[a], Ct.rowval[b]))
                push!(Jr, k)
                push!(Vr, Ct.nzval[a] * Ct.nzval[b])
            end
        end
    end
    R = sparse(Ir, Jr, Vr, nnz(pattern), nnz(full))
    return R * T_full
end

function _conductivity_tensor_full(disc::FerriteDiscretization, pattern::SparseMatrixCSC)
    cv_u, cv_σ = disc.cv_u, disc.cv_σ
    nu, nσ = getnbasefunctions(cv_u), getnbasefunctions(cv_σ)
    nq = getnquadpoints(cv_u)
    Ae = zeros(nu, nu, nσ)
    σdofs = zeros(Int, nσ)
    nzidx = zeros(Int, nu, nu)
    ncell = getncells(disc.grid)
    Is = Vector{Int}(undef, ncell * nu * nu * nσ)
    Js = similar(Is)
    Vs = Vector{Float64}(undef, length(Is))
    p = 0
    for cell in CellIterator(disc.dh_u)
        reinit!(cv_u, cell)
        reinit!(cv_σ, cell)
        udofs = celldofs(cell)
        celldofs!(σdofs, disc.dh_σ, cellid(cell))
        fill!(Ae, 0)
        for q in 1:nq
            dΩ = getdetJdV(cv_u, q)
            for a in 1:nσ
                ψ = shape_value(cv_σ, q, a) * dΩ
                iszero(ψ) && continue
                for j in 1:nu
                    ∇φⱼ = shape_gradient(cv_u, q, j)
                    for i in 1:nu
                        Ae[i, j, a] += ψ * (shape_gradient(cv_u, q, i) ⋅ ∇φⱼ)
                    end
                end
            end
        end
        for j in 1:nu, i in 1:nu
            nzidx[i, j] = _nz_index(pattern, udofs[i], udofs[j])
        end
        for a in 1:nσ, j in 1:nu, i in 1:nu
            p += 1
            Is[p], Js[p], Vs[p] = nzidx[i, j], σdofs[a], Ae[i, j, a]
        end
    end
    return sparse(Is, Js, Vs, nnz(pattern), ndofs(disc.dh_σ))
end
