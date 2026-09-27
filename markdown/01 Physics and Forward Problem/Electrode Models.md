---
tags: [forward-problem, electrodes]
---

The mathematical boundary conditions have to model how current actually enters the body through $L$ metal electrodes $e_1,\dots,e_L\subset\partial\Omega$. The standard models, in increasing realism:

1. **Continuum model.** Current density and voltage are known at every boundary point. This idealisation underlies the [[Dirichlet-to-Neumann Map]] and the [[Calderón Problem]]. It is convenient for theory and for simulations in which every boundary node of the mesh acts as an electrode. Measuring on all boundary nodes is equivalent to choosing a basis of boundary functions, such as [[Current Patterns|Fourier modes]].
2. **Gap model.** Current density $I_\ell/|e_\ell|$ is constant on each electrode and zero in the gaps. It ignores that the electrode, being a good conductor, forces the voltage (not the current) to be constant, and it overestimates resistivity.
3. **Shunt model.** The electrode is a perfect conductor: $u = U_\ell$ on $e_\ell$ and $\int_{e_\ell}\gamma\partial_\nu u = I_\ell$. It ignores the thin, highly resistive layer at the electrode–skin contact.
4. **[[Complete Electrode Model]] (CEM).** The shunt model plus a contact impedance $z_\ell$. It predicts measured voltages to within measurement precision and is the standard for real data.

Cheng et al. (1989) showed experimentally that only the CEM matches measured data to instrument precision.

## References

1. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
2. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
