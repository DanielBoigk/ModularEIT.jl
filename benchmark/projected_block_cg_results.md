# Projected block CG benchmark

EIT Neumann stiffness matrix (Q1, cell-wise random σ ∈ [1, 10]), 32 right-hand sides with
mean-zero random boundary currents, grounding Σ_boundary uᵢ = 0, rtol = 1e-8 (Float64) /
1e-5 (Float32). CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 Julia threads,
4 BLAS threads). GPU: NVIDIA GeForce RTX 3080.
AMG setup (hierarchy construction) runs on the CPU in both cases. Unpreconditioned runs are skipped
above 512², CPU Jacobi above 512². The last row was re-measured separately: the original run
overlapped with a package precompilation that kept the CPU busy (0.786 s).

| grid | dofs | device | precision | preconditioner | setup [s] | solve [s] | iterations | time/iteration [ms] | rel. residual |
|:--|--:|:--|:--|:--|--:|--:|--:|--:|--:|
| 128² | 16641 | cpu | Float64 | none | 0.000 | 1.775 | 380 | 4.67 | 4.1e-09 |
| 128² | 16641 | cpu | Float64 | jacobi | 0.043 | 1.472 | 286 | 5.15 | 5.2e-09 |
| 128² | 16641 | cpu | Float64 | amg | 1.093 | 0.207 | 14 | 14.80 | 5.2e-09 |
| 128² | 16641 | gpu | Float64 | none | 0.000 | 0.520 | 366 | 1.42 | 4.0e-09 |
| 128² | 16641 | gpu | Float64 | jacobi | 0.019 | 0.411 | 286 | 1.44 | 5.3e-09 |
| 128² | 16641 | gpu | Float64 | amg | 0.235 | 0.033 | 14 | 2.38 | 5.0e-09 |
| 128² | 16641 | gpu | Float32 | none | 0.000 | 0.111 | 303 | 0.37 | 8.3e-06 |
| 128² | 16641 | gpu | Float32 | jacobi | 0.055 | 0.096 | 265 | 0.36 | 1.1e-05 |
| 128² | 16641 | gpu | Float32 | amg | 1.423 | 0.006 | 9 | 0.70 | 1.3e-05 |
| 256² | 66049 | cpu | Float64 | none | 0.000 | 13.971 | 731 | 19.11 | 4.1e-09 |
| 256² | 66049 | cpu | Float64 | jacobi | 0.000 | 11.420 | 563 | 20.28 | 5.3e-09 |
| 256² | 66049 | cpu | Float64 | amg | 0.106 | 0.881 | 14 | 62.92 | 5.4e-09 |
| 256² | 66049 | gpu | Float64 | none | 0.000 | 2.733 | 748 | 3.65 | 4.8e-09 |
| 256² | 66049 | gpu | Float64 | jacobi | 0.000 | 2.113 | 566 | 3.73 | 5.2e-09 |
| 256² | 66049 | gpu | Float64 | amg | 0.095 | 0.093 | 14 | 6.68 | 5.0e-09 |
| 256² | 66049 | gpu | Float32 | none | 0.000 | 0.403 | 654 | 0.62 | 1.3e-05 |
| 256² | 66049 | gpu | Float32 | jacobi | 0.001 | 0.343 | 548 | 0.63 | 2.7e-05 |
| 256² | 66049 | gpu | Float32 | amg | 0.084 | 0.016 | 11 | 1.43 | 2.2e-05 |
| 512² | 263169 | cpu | Float64 | none | 0.000 | 123.193 | 1402 | 87.87 | 5.2e-09 |
| 512² | 263169 | cpu | Float64 | jacobi | 0.004 | 101.537 | 1169 | 86.86 | 4.1e-09 |
| 512² | 263169 | cpu | Float64 | amg | 0.294 | 4.578 | 18 | 254.32 | 4.3e-09 |
| 512² | 263169 | gpu | Float64 | none | 0.000 | 18.005 | 1363 | 13.21 | 5.3e-09 |
| 512² | 263169 | gpu | Float64 | jacobi | 0.002 | 15.788 | 1168 | 13.52 | 4.0e-09 |
| 512² | 263169 | gpu | Float64 | amg | 0.335 | 0.419 | 17 | 24.67 | 5.4e-09 |
| 512² | 263169 | gpu | Float32 | none | 0.000 | 3.676 | 2279 | 1.61 | 3.4e-05 |
| 512² | 263169 | gpu | Float32 | jacobi | 0.002 | 2.223 | 1282 | 1.73 | 4.4e-05 |
| 512² | 263169 | gpu | Float32 | amg | 0.247 | 0.054 | 12 | 4.51 | 4.6e-05 |
| 1024² | 1050625 | cpu | Float64 | amg | 0.910 | 20.477 | 19 | 1077.76 | 5.0e-09 |
| 1024² | 1050625 | gpu | Float64 | jacobi | 0.023 | 124.245 | 2347 | 52.94 | 4.4e-09 |
| 1024² | 1050625 | gpu | Float64 | amg | 0.915 | 1.807 | 19 | 95.11 | 4.4e-09 |
| 1024² | 1050625 | gpu | Float32 | jacobi | 0.008 | 39.766 | 6130 | 6.49 | 7.7e-05 |
| 1024² | 1050625 | gpu | Float32 | amg | 0.797 | 0.253 | 55 | 4.60 | 7.0e-05 |
