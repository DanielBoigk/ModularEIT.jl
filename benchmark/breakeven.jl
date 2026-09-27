# CPU vs GPU break-even point of the projected block CG solver.
#
#   julia --project=benchmark -t auto benchmark/breakeven.jl
#
# Sweeps small grids and block sizes s (number of right-hand sides) and reports the
# wall time of the CPU (Float64) and GPU (Float64 / Float32) solves, with and without AMG.
# Writes benchmark/breakeven_results.md.

include("projected_block_cg.jl")

function breakeven(; sizes = (8, 16, 24, 32, 48, 64, 96, 128), blocks = (1, 4, 16, 64), pres = (:none, :amg))
    lines = ["| grid | dofs | s | preconditioner | CPU Float64 [ms] | GPU Float64 [ms] | GPU Float32 [ms] | speed-up GPU64 | speed-up GPU32 |",
             "|:--|--:|--:|:--|--:|--:|--:|--:|--:|"]
    for n in sizes
        A, bd = assemble_neumann(n)
        N = size(A, 1)
        w = boundary_grounding(N, bd)
        for s in blocks
            B = boundary_rhs(N, bd, s)
            for pre in pres
                t = Dict{Tuple{Symbol, DataType}, Float64}()
                for (device, T) in ((:cpu, Float64), (:gpu, Float64), (:gpu, Float32))
                    Ah = SparseMatrixCSC{T, Int}(A)
                    conv = device == :gpu ? todev : identity
                    M = pre == :amg ? AMGPreconditioner(Ah, s; to_device = conv) : nothing
                    t[(device, T)] = run_case(A, B, w, M; device, T, rtol = T == Float64 ? 1e-8 : 1f-5)[1]
                end
                c, g64, g32 = t[(:cpu, Float64)], t[(:gpu, Float64)], t[(:gpu, Float32)]
                line = @sprintf("| %d² | %d | %d | %s | %.2f | %.2f | %.2f | %.2f | %.2f |",
                                n, N, s, pre, 1e3c, 1e3g64, 1e3g32, c / g64, c / g32)
                println(line); flush(stdout)
                push!(lines, line)
            end
        end
    end
    header = """
    # CPU / GPU break-even point

    Projected block CG on the EIT Neumann stiffness matrix (Q1, cell-wise random σ), s random
    mean-zero boundary currents, grounding Σ_boundary uᵢ = 0. Wall time of the solve only
    (minimum of 3 runs); AMG setup excluded. Speed-up = CPU time / GPU time (> 1: GPU faster).
    CPU: $(Sys.cpu_info()[1].model) ($(Threads.nthreads()) threads), GPU: $(CUDA.name(CUDA.device())).

    """
    open(joinpath(@__DIR__, "breakeven_results.md"), "w") do io
        print(io, header, join(lines, "\n"), "\n")
    end
end

breakeven()
