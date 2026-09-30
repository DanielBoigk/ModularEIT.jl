---
tags: [outlook, plasma, pde-optimization, inverse-problems, proposal]
aliases: [Birkeland-Eyde process, Plasma NOx synthesis, Moonshot proposal]
---

**Proposal in one paragraph.** Fertiliser nitrogen is made by Haber–Bosch: ammonia from hydrogen (today mostly from natural gas) and nitrogen, followed by the Ostwald process to nitric acid. A century ago the **Birkeland–Eyde process** made nitric oxide directly from air in an electric arc, and lost only because it needed more energy. With cheap renewable electricity, plasma-based NOx synthesis becomes economically competitive once its energy cost falls to about **1.0–1.5 MJ per mol N**. The best reactors today reach about **1.8–2.5**. The remaining factor of about two is largely a *design* problem: gas bypassing the arc, back-reactions, heat losses. Yet reactor geometries are still designed by hand and a few experimental parameter sweeps. Designing them systematically is PDE-constrained shape and topology optimisation, calibrated by inverse problems and accelerated by machine learning: exactly the toolbox of an applied-analysis group.

## Where the numbers stand

Energy cost per mol of fixed nitrogen (Rouwenhorst et al. 2021, with the 2023 correction; Tsonev et al. 2023):

| | MJ/mol N |
|---|---|
| Birkeland–Eyde (1900s, magnetically spread arc) | 2.4–3.1 |
| best modern reactors at atmospheric or elevated pressure (arcs, microwave) | ≈ 1.8–2.5 |
| **break-even with electrolysis-based Haber–Bosch + Ostwald** | **1.0–1.5** (depending on reactor cost) |
| full competitiveness with today's fossil Haber–Bosch + Ostwald | ≈ 0.7 |
| Haber–Bosch + Ostwald today | 0.5–0.6 |
| limit: thermal plasma, ideal quench | 0.72 |
| limit: non-equilibrium, vibrationally driven | ≈ 0.2 |

Figure 5 of Rouwenhorst et al. compiles the reported attempts across reactor types. Two findings of that paper point at design:

- in a gliding arc plasmatron only about **15 % of the gas passes through the arc**; the rest bypasses it;
- the NO formed must be **quenched fast**, or it decomposes again downstream.

The authors name "smart reactor design" as one of the routes to break-even. A plasma process also scales *down* better than Haber–Bosch, and it can be switched on and off with the electricity supply. That makes decentralised production of nitrate fertiliser from air, water and local renewable power possible, a natural start-up case.

## The mathematics

**Model.** An atmospheric arc is a hot, electrically conducting gas channel in a flow:

$$
\begin{aligned}
\nabla\cdot\big(\sigma(T)\,\nabla\phi\big) &= 0 &&\text{current continuity}\\
\rho c_p\big(\partial_t T + u\cdot\nabla T\big) &= \nabla\cdot(k\nabla T) + \sigma(T)\,|\nabla\phi|^2 - q_{\text{rad}} &&\text{energy, Joule heating}\\
\rho\big(\partial_t u + u\cdot\nabla u\big) &= -\nabla p + \nabla\cdot\tau + j\times B &&\text{flow, Lorentz force}\\
\partial_t c_s + u\cdot\nabla c_s &= \nabla\cdot(D_s\nabla c_s) + R_s(c, T, T_v) &&\text{species, chemistry}
\end{aligned}
$$

The first equation is the **conductivity equation of EIT**, now with a conductivity that rises by orders of magnitude with temperature, so the arc is a sharp, moving internal layer. The chemistry is led by the Zeldovich mechanism, $\mathrm{O} + \mathrm N_2 \to \mathrm{NO} + \mathrm N$ (strongly endothermic, rate-limiting) and $\mathrm N + \mathrm O_2\to\mathrm{NO}+\mathrm O$. In thermal plasmas this gives the equilibrium limit above. Vibrationally excited nitrogen ($T_v > T$, two-temperature or state-to-state kinetics) lowers the effective barrier, which is where the 0.2 MJ/mol limit comes from.

**Design problem.** Minimise the energy cost over the geometry $\Omega$ (electrodes, gas inlets and swirl, quench zone) and the operating parameters $q$ (flow, pressure, power waveform, magnetic field):

$$
\min_{\Omega,\,q}\ \mathcal E(\Omega,q) = \frac{\overline{\int_\Omega \sigma(T)\,|\nabla\phi|^2\,\mathrm dx}}{\overline{\int_{\Gamma_{\text{out}}}(c_{\text{NO}}+c_{\text{NO}_2})\,u\cdot n\,\mathrm ds}}
\quad\text{subject to the model above},
$$

with time averages $\overline{\,\cdot\,}$.

**What is needed.**

- **PDE-constrained optimisation.** The adjoint method gives the gradient with respect to thousands of design variables for the cost of one additional solve (see [[Adjoint State Method]]). Shape derivatives move boundaries (Sokołowski and Zolésio 1992). Topology optimisation with a porosity field in the flow (Borrvall and Petersson 2003; Bendsøe and Sigmund 2004) lets channels, baffles and quench zones emerge instead of being prescribed. The hard parts are unsteady, restriking arcs, whose time-averaged sensitivities call for shadowing-type adjoints (Wang et al. 2014) or surrogates; stiff kinetics with hundreds of reactions; and scales from electrode sheaths to the reactor.
- **Numerics.** Adaptive finite elements with a posteriori error control for the thin arc layer (see [[A Posteriori Error Estimation and Adaptive Meshing]]), and fast solvers for the coupled system.
- **Inverse problems.** The model must be calibrated before its optimum means anything. That means identifying rate and transport coefficients and sheath models from current–voltage traces, optical emission and outlet composition, with uncertainty quantification (see [[Bayesian Inversion]]). The diagnostics are themselves tomography: the conductivity distribution of the arc from electrode measurements is an EIT problem for the same equation, and emission tomography gives temperature fields.
- **Machine learning.** Neural-operator surrogates of the coupled simulation (Li et al. 2021) make design loops fast. Learned reduced kinetics replace detailed mechanisms. Bayesian optimisation and active learning choose the few expensive experiments. Language models accelerate the rest: literature, code and the modelling itself. That changes what a small group can attempt.

## Why this group, why now

The tools are the ones developed for tomography: adjoint gradients, Jacobians, parametrisations of the unknown, adaptive meshes, regularisation and learned priors. In this code base they already exist for the conductivity equation (see [[Design Principles for EIT Solvers]]). The step is from *identifying* a conductivity to *designing* a device that is governed by one.

**Possible first milestones.**

1. Reproduce a published reactor in simulation, e.g. the modelled gliding arc plasmatron of Vervloessem et al. (2020), as validation.
2. Adjoint shape optimisation of flow and quench geometry with a fixed arc model: raise the fraction of gas treated and cut back-reactions.
3. Coupled optimisation including electrodes and magnetic field, calibrated against experiments with a plasma-physics partner.
4. Target: below 1.5 MJ/mol N at atmospheric pressure, i.e. competitive with green ammonia plus Ostwald.

**Risks.** Whether design alone closes the gap is unproven: thermal arcs cannot beat 0.72 MJ/mol, so the largest gains need non-equilibrium operation. Model fidelity for arcs is limited. And the experimental side needs partners.

## References

1. K. H. R. Rouwenhorst, F. Jardali, A. Bogaerts, L. Lefferts (2021). *From the Birkeland–Eyde process towards energy-efficient plasma-based NOX synthesis: a techno-economic analysis*. Energy Environ. Sci. 14, 2520–2534. [doi:10.1039/D0EE03763J](https://doi.org/10.1039/D0EE03763J). Correction: Energy Environ. Sci. 16 (2023), 6170–6173. [doi:10.1039/D3EE90066E](https://doi.org/10.1039/D3EE90066E)
2. I. Tsonev, C. O'Modhrain, A. Bogaerts, Y. Gorbanev (2023). *Nitrogen Fixation by an Arc Plasma at Elevated Pressure to Increase the Energy Efficiency and Production Rate of NOx*. ACS Sustainable Chem. Eng. 11, 1888–1897. [doi:10.1021/acssuschemeng.2c06357](https://doi.org/10.1021/acssuschemeng.2c06357)
3. E. Vervloessem, M. Aghaei, F. Jardali, S. Hafezkhiabani, A. Bogaerts (2020). *Plasma-Based N2 Fixation into NOx: Insights from Modeling toward Optimum Yields and Energy Costs in a Gliding Arc Plasmatron*. ACS Sustainable Chem. Eng. 8, 9711–9720. [doi:10.1021/acssuschemeng.0c01815](https://doi.org/10.1021/acssuschemeng.0c01815)
4. A. Fridman (2008). *Plasma Chemistry*. Cambridge University Press. [doi:10.1017/CBO9780511546075](https://doi.org/10.1017/CBO9780511546075)
5. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
6. J. Sokołowski, J.-P. Zolésio (1992). *Introduction to Shape Optimization*. Springer. [doi:10.1007/978-3-642-58106-9](https://doi.org/10.1007/978-3-642-58106-9)
7. T. Borrvall, J. Petersson (2003). *Topology optimization of fluids in Stokes flow*. Int. J. Numer. Methods Fluids 41, 77–107. [doi:10.1002/fld.426](https://doi.org/10.1002/fld.426)
8. M. P. Bendsøe, O. Sigmund (2004). *Topology Optimization*. Springer. [doi:10.1007/978-3-662-05086-6](https://doi.org/10.1007/978-3-662-05086-6)
9. Q. Wang, R. Hu, P. Blonigan (2014). *Least Squares Shadowing sensitivity analysis of chaotic limit cycle oscillations*. J. Comput. Phys. 267, 210–224. [doi:10.1016/j.jcp.2014.03.002](https://doi.org/10.1016/j.jcp.2014.03.002)
10. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
