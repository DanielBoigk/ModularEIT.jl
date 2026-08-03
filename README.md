# ModularEIT.jl (Under Development)

This is a Julia library for electrical impedance tomography similar in spirit to [PyEIT](https://github.com/eitcom/pyEIT) or [EIDORS](https://eidors3d.sourceforge.net/).
This library is build on top of the FEM library [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/)although  it is planned that one can plug in other Galerkin methods as solvers. Currently this project is still under construction although  multiple things work already. Planned features:

- [x] Gauss-Newton to resolve Gradients
- [x] Classical Regularizers:
  - [x] Tikhonov
  - [x] TV
    - [x] Huber smoothed
    - [ ] Chambolle-Pock
- [ ] Learned regularizers
  - [ ] CNN trained as VP-SDE for quadrilateral grid. 
    - [ ] via Diffusion Posteroir Sampling
	- [ ] via Variational Stochastic Sampling 
- [ ]  Documentation
- [ ] Various Optimizers:
	- [x] Line Search after Gradient
	- [x] L-BFGS
	- [x] Wrapper for Prox step
	- [x] ADMM
	- [x] Wrapper for Adam, RMSProp, nAdam, ...
- [ ] Galerkin methods:
	- [x] Ferrite.jl
		- [ ] Compilation of whole FEM routine with Reactant.jl
		- [ ] Adaptive Meshing
	- [ ] GalerkinToolkit/Gridap
	- [ ] ApproxFun
	- [ ] FFTA
- [x] Helper functions for common domain shapes:
  - [x] Rectangle
  - [x] Circle

This is supposed to be very modular, so just plugin other (pseudo) metrics/loss maps, regularizers, optimizers, ... and run with it.

## How to use

### Installation:
Make sure `gmsh` is installed on your system and added to `PATH`. (If you intend on using Gmsh)
inside Julia run:
`]activate`
then 
`] add "github.com/DanielBoigk/ModularEIT.jl"`
this instantiates the environment. Inside the enviroment `using ModularEIT` adds the library.

### General workflow:
Import necessary libraries:
```
using Ferrite
using ModularEIT
```
Generate or import some grid: 
```
grid = generate_grid(Quadrilateral, (127, 127))
```
Define where the boundary is: 
```
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
```
*Note:  I have no separate electron model defined yet. I assume that the boundary is the electrodes for now. Everything in the force vector outside the boundary nodes will be set to zero by default.*
with that one can build the finite element space:
```
fe = FerriteFESpace{RefQuadrilateral}(grid, 2, 3, ∂Ω)
```
Define some function that simulates the conductivity:
```
conductivity(x) = ...
```
Convert into coefficient vector of `FESpace` 
``` 
cond_vec = project_function_to_fem(fe, conductivity)
```

Choose some basis for the boundary patterns and generate simulation data:
```
G_full = real_fourier_basis(8)
rhs_vec = Vector{Any}(undef, 255)
Threads.@threads for i in 2:256
    M = make_boundary(G_full[:, i], 64)
    itp = interpolate_array_2D(M)
    rhs_vec[i-1] = assemble_rhs_func(fe, itp)
end
    # Assemble true stiffness matrix
_ , K_fac = assemble_L(fe, cond_vec, factorize = true)
mode_vec = Vector{Any}(undef, 255)
for i in 1:255
    mode_vec[i] = create_mode_from_g(fe, rhs_vec[i], K_fac)
end    
```
Now we have synthetic boundary pairs that allow us to define an `EITProblem`
We define some starting guess:
```
σ_vec = project_function_to_fem(fe, x -> 0.5)
```
This struct stores all the Information about the solution:    
```
sol = FerriteSolverState(fe, σ_vec)
```
We define the `EITProblem`:
```
prblm = FerriteProblem(fe, mode_vec, sol)
```


To add a differentiable regularizer we define a function and add it via `add_diff_Regularizer`.
Julias Autodiff functionality does the differentiation. Setting the gradient manually is also possible.
```
Tikhonov(x) = normH1sq(prblm.fe, x)
add_diff_Regularizer!(prblm.state, Tikhonov)
```

alternatively one can add a nondifferentiable regularizers via:
```    
TV(x) = normTV(prblm.fe, x)
add_nondiff_Regularizer!(prblm.state, TV)
```
    
    prblm.state.opt.β_diff = 1e-2

For easy use with other optimizers one can wrap the problem to get the objective and the gradient.
``` 
f, ∂f = create_f∂f(prblm, 100; regularize=false, gn=false)
```
Here one can also specify which solver should be used:
```
f, ∂f = create_f∂f(prblm, 10; regularize=false, gn=false, mode="neumann", obj=objective_neumann_init!, grad=gradient_neumann_init!)
```
This can then be solved with some optimization algorithm, like: 
```
solution = lbfgs_b(f, ∂f, σ_vec; m=10, tol=1e-6, maxiter=30)
```

Then one can use some method to visualize the finite element function that is returned:
Like using a `Feritte.PointHandler` to project unto an image and 
```    
using Images
eval_points = reshape(equidistant_grid(128), :)
ph = PointEvalHandler(grid, eval_points)
img_solution = Gray.(max.(0.0, min.(1.0, reshape(evaluate_at_points(ph, prblm.fe.dh, solution), (128, 128)))))
```

## Example Reconstruction:
Original Image:
![Original Image](/notebooks/reconstruction/1096.jpg)
Reconstruction with Tikhonov regularisation:
![Reconstruction with Tikhonov regularisation](/notebooks/reconstruction/Tikhonov/000Tikhonov64.png)
Reconstruction with TV regularization:
![Reconstruction with Tikhonov regularisation](/notebooks/reconstruction/TotalVariation/1.000e-01_0.000e+00_0.000e+00_0.000e+00.png)
