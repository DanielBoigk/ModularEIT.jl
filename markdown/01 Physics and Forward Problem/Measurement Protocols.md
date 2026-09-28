---
tags: [forward-problem, electrodes, measurement]
aliases: [Stimulation and measurement patterns, Drive patterns, Four-electrode measurement]
---

A **measurement protocol** specifies which electrodes carry current (the *drive* or *injection* electrodes) and between which electrodes voltages are recorded (the *measurement* electrodes). The two sets need not be the same, and the choice decides which [[Electrode Models|electrode model]] can describe the data.

## Measuring on current-carrying electrodes

Integrating the boundary condition $u + z_\ell\sigma\partial_\nu u = U_\ell$ of the [[Complete Electrode Model]] over $e_\ell$ gives the exact identity

$$
U_\ell = \frac1{|e_\ell|}\int_{e_\ell}u\,\mathrm ds + \frac{z_\ell}{|e_\ell|}\,I_\ell .
$$

The voltage on an electrode is the mean potential under it plus the voltage drop over the contact layer. The contact impedance $z_\ell$ depends on skin, gel and pressure, so it is poorly known and changes over time.

- **Current-carrying electrode** ($I_\ell\neq 0$): the contact term $z_\ell I_\ell/|e_\ell|$ enters the measurement directly and is often larger than the signal from the interior. Only the CEM models it. The [[Gap Model]] predicts the mean potential without it, and the [[Point Electrode Model]] predicts an infinite voltage.
- **Electrode without current** ($I_\ell = 0$): the contact term vanishes, and the voltage is essentially the mean potential under the electrode. Gap, point and complete models agree closely here.

This is the **four-electrode (tetrapolar) principle**: drive current through one pair of electrodes and measure the voltage with a high-impedance voltmeter on a *different* pair, so no current flows through the measuring contacts. Two-electrode measurements, with voltage taken on the driving pair, measure the contact impedances along with the object.

## Common protocols

- **Adjacent (neighbouring) protocol.** Current between electrodes $\ell,\ell+1$. Voltages between all other adjacent pairs, skipping the pairs that involve a driven electrode. With $L$ electrodes this gives $L(L-3)$ measurements, of which half are independent by reciprocity ($104$ for $L=16$). Only electrodes without current are measured, so a gap or point model suffices. Its sensitivity in the centre is poor.
- **Opposite and skip-$k$ protocols.** Current between electrodes a fixed distance apart. Larger separations reach deeper and distinguish interior changes better than the adjacent protocol.
- **Trigonometric (optimal) patterns.** All electrodes carry current at once, $I_\ell = \cos(k\theta_\ell)$ or $\sin(k\theta_\ell)$ (see [[Current Patterns]]). Voltages are then necessarily measured on current-carrying electrodes, so the data must be modelled with the CEM.
- **Continuum data.** In simulations, every boundary node can inject and measure. This is the idealised [[Neumann-to-Dirichlet Map]] without contact effects.

## Which model for which protocol

| protocol | voltages measured on | adequate models |
|:--|:--|:--|
| adjacent, skip-$k$ | electrodes without current | CEM, gap, point |
| all-electrode (trigonometric) patterns | current-carrying electrodes | CEM |
| idealised boundary data | every boundary point | continuum |

With separate drive and measurement electrodes, the forward operator maps $L_{\mathrm{drive}}$ currents to $L_{\mathrm{meas}}$ voltages. In the discrete model this means different injection and measurement matrices (see [[Discrete Electrode Models]]).

## References

1. B. H. Brown (2003). *Electrical impedance tomography (EIT): a review*. J. Med. Eng. Technol. 27(3), 97–108. [doi:10.1080/0309190021000059687](https://doi.org/10.1080/0309190021000059687)
2. K. Boone, D. Barber, B. Brown (1997). *Imaging with electricity: Report of the European Concerted Action on Impedance Tomography*. J. Med. Eng. Technol. 21(6), 201–232. [doi:10.3109/03091909709070013](https://doi.org/10.3109/03091909709070013)
3. A. Adler, P. O. Gaggero, Y. Maimaitijiang (2011). *Adjacent stimulation and measurement patterns considered harmful*. Physiol. Meas. 32(7), 731–744. [doi:10.1088/0967-3334/32/7/S01](https://doi.org/10.1088/0967-3334/32/7/S01)
4. K.-S. Cheng, D. Isaacson, J. C. Newell, D. G. Gisser (1989). *Electrode models for electric current computed tomography*. IEEE Trans. Biomed. Eng. 36(9), 918–924. [doi:10.1109/10.35300](https://doi.org/10.1109/10.35300)
5. D. Isaacson (1986). *Distinguishability of Conductivities by Electric Current Computed Tomography*. IEEE Trans. Med. Imaging 5(2), 91–95. [doi:10.1109/TMI.1986.4307752](https://doi.org/10.1109/TMI.1986.4307752)
