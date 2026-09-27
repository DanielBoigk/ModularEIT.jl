---
tags: [optimization, splitting]
aliases: [Alternating direction method of multipliers]
---

The **alternating direction method of multipliers (ADMM)** minimises a sum of two terms that are easy to handle separately:

$$
\min_x\ F(x)+G(x)
\quad\longrightarrow\quad
\min_{x,z}\ F(x)+G(z)\quad\text{s.t.}\quad x = z .
$$

With penalty parameter $\rho>0$ and the *scaled* dual variable $u$, the iteration is

$$
\begin{aligned}
x^{k+1} &= \operatorname{prox}_{F/\rho}\big(z^k-u^k\big) = \arg\min_x F(x)+\tfrac\rho2\|x-z^k+u^k\|^2,\\
z^{k+1} &= \operatorname{prox}_{G/\rho}\big(x^{k+1}+u^k\big) = \arg\min_z G(z)+\tfrac\rho2\|x^{k+1}-z+u^k\|^2,\\
u^{k+1} &= u^k+x^{k+1}-z^{k+1},
\end{aligned}
$$

initialised for example with $z^0 = x^0$ and $u^0 = 0$ (see [[Proximal Operator]]). The same $\rho$ appears in both proximal steps; it is the penalty of the augmented Lagrangian $F(x)+G(z)+\tfrac\rho2\|x-z+u\|^2$.

**Convergence.** For closed, proper, convex $F,G$, the residuals $x^k-z^k\to0$, the objective converges to the optimum, and $\rho u^k$ converges to a dual solution, for *any* $\rho>0$. The choice of $\rho$ affects speed. Residual balancing adapts it: increase $\rho$ if the primal residual $\|x-z\|$ dominates, and decrease it if the dual residual $\rho\|z^{k+1}-z^k\|$ dominates. For nonconvex $F$, such as the EIT misfit, convergence is only guaranteed under extra assumptions, but ADMM often works well in practice.

**Stopping.** Stop when both the primal residual $\|x^k-z^k\|$ and the dual residual $\rho\|z^k-z^{k-1}\|$ are small.

**In EIT.** $F$ is the data misfit, whose prox is computed iteratively, and $G = \beta\mathcal R$ is a regulariser with a cheap prox ([[Total Variation]], [[Tikhonov Regularization]], box constraints) or a learned denoiser ([[Plug-and-Play Priors]], [[Diffusion Proximal Operator]]). See [[Nested ADMM Reconstruction]].

## References

1. S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Found. Trends Mach. Learn. 3(1), 1–122. [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
2. Y. Wang (2022). *Anisotropic TV Regularization in Electrical Impedance Tomography: An Experimental Study*. Engineering 14(3), 138–146. [doi:10.4236/eng.2022.143013](https://doi.org/10.4236/eng.2022.143013)
3. C. Park, S. Shoushtari, W. Gan, U. S. Kamilov (2023). *Convergence of Nonconvex PnP-ADMM with MMSE Denoisers*. IEEE CAMSAP 2023, 511–515. [doi:10.1109/CAMSAP58249.2023.10403463](https://doi.org/10.1109/CAMSAP58249.2023.10403463)
