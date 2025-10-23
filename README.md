# Julia EIT (Under Development)

Dear Visitor: Over the next weeks I'm gonna build a Electrical Impedance Tomography (EIT) Library similar in spirit to [PyEIT](https://github.com/eitcom/pyEIT) or [EIDORS](https://eidors3d.sourceforge.net/).

I'm doing this because of there being several downsides with the existing libraries, like:
- Lack of Integration with Machine Learning Frameworks
- Lack of Documentation
- Performance deficits and parallelization of GPU
- Lack of integration with FEM frameworks


This Library wil be build on [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/), [Enzyme.jl](https://enzyme.mit.edu/julia/stable/), [Lux.jl](https://lux.csail.mit.edu/stable/) and [Reactant.jl](https://enzymead.github.io/Reactant.jl/dev/introduction/). However lateron I will also consider integration with other Galerkin methods like based on spectral methods (like [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) or  [ApproxFun.jl](https://juliaapproximation.github.io/ApproxFun.jl/stable/)).
This is supposed to be very modular, so just plugin other (pseudo) metrics/loss maps, regularizer, optimizers, ...
