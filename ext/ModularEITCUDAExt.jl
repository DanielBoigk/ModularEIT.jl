module ModularEITCUDAExt

using ModularEIT
using CUDA
using LinearAlgebra

# cuBLAS GEMM for XᵀY with tall-skinny Float64 blocks and 2 ≤ s ≤ 8 columns picks a kernel that
# does not split the long reduction dimension and is 10–30× slower than one GEMV per column
# (measured on an RTX 3080, cuBLAS 13). For larger blocks GEMM is faster again.
const GEMV_MAX_COLS = 8

function ModularEIT._gram!(G::DenseCuMatrix{Float64}, X::DenseCuMatrix{Float64}, Y::DenseCuMatrix{Float64})
    if 1 < size(Y, 2) <= GEMV_MAX_COLS
        for j in axes(Y, 2)
            mul!(view(G, :, j), X', view(Y, :, j))
        end
        return G
    end
    return mul!(G, X', Y)
end

end
